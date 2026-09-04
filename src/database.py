"""
database.py
===========

SQLite persistence for the L.O.C.K.I. system
(Live Optical Custody & Key Identifier).

Every key checkout / removal event detected by the pipeline is recorded in
the ``checkouts`` table of ``data/logs/key_events.db`` (path configurable via
the ``database.path`` config key):

    id INTEGER PRIMARY KEY AUTOINCREMENT
    timestamp     TEXT    ISO-8601 UTC, e.g. '2026-09-04T12:30:00.123456+00:00'
    driver_id     TEXT    matched driver, or the fallback id ('UNIDENTIFIED')
    assigned_route TEXT   schedule route for that driver ('' when unknown)
    taken_route   TEXT    route the key actually belongs to ('' when unknown)
    status        TEXT    'SUCCESS' | 'WRONG_ROUTE_ALERT' | 'UNAUTHORIZED_REMOVAL'
    clip_path     TEXT    evidence clip written by CameraStream.save_clip()

Status semantics
----------------
* ``SUCCESS``             - the driver took a key of their assigned route color.
* ``WRONG_ROUTE_ALERT``   - the driver took a key of a *different* route color.
* ``UNAUTHORIZED_REMOVAL``- a key vanished from a slot with no matching
  scheduled driver identified at the station (face match failed and no
  unassigned person could be attributed), or the slot's key was taken while
  the station saw nobody.

The module is thread-safe (SQLite connection guarded by an RLock) so the
detection pipeline and a UI/logger thread can write concurrently.

Typical usage::

    from src.database import DatabaseManager, EventStatus

    db = DatabaseManager("config/config.json")
    event_id = db.log_event(
        driver_id="Driver_101",
        assigned_route="Route_RED",
        taken_route="Route_RED",
        status=EventStatus.SUCCESS,
        clip_path="evidence/2026-09-04/incident.mp4",
    )
    recent = db.recent_events(limit=20)
    db.close()
"""

from __future__ import annotations

import enum
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.color_detector import ConfigError, extract_color, load_config

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DatabaseError",
    "DatabaseManager",
    "EventRecord",
    "EventStatus",
]

_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkouts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    driver_id      TEXT NOT NULL,
    assigned_route TEXT NOT NULL DEFAULT '',
    taken_route    TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL,
    clip_path      TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_checkouts_timestamp ON checkouts (timestamp);
CREATE INDEX IF NOT EXISTS idx_checkouts_driver    ON checkouts (driver_id);
CREATE INDEX IF NOT EXISTS idx_checkouts_status    ON checkouts (status);
"""


class DatabaseError(RuntimeError):
    """Raised when the database cannot be opened, migrated, or written."""


class EventStatus(str, enum.Enum):
    """Discrete custody outcomes recorded in the ``checkouts`` table."""

    SUCCESS = "SUCCESS"
    WRONG_ROUTE_ALERT = "WRONG_ROUTE_ALERT"
    UNAUTHORIZED_REMOVAL = "UNAUTHORIZED_REMOVAL"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @classmethod
    def classify(
        cls,
        *,
        assigned_route: Optional[str],
        taken_route: Optional[str],
        driver_known: bool,
    ) -> "EventStatus":
        """Derive the event status from an identification + color decision.

        Route text is compared via canonical colors (``extract_color``), so
        ``"red"`` matches ``"Route_RED"`` regardless of formatting.

        Args:
            assigned_route: The driver's scheduled route (``None``/``''`` if
                the driver is not on today's schedule).
            taken_route: The route of the key actually taken.
            driver_known: True if the face recognizer matched a scheduled
                driver; False for the fallback / unidentified path.
        """
        taken_color = extract_color(taken_route)
        if taken_color is None:
            # No idea which key was taken: treat as unauthorized until the
            # evidence clip says otherwise.
            return cls.UNAUTHORIZED_REMOVAL
        if not driver_known:
            return cls.UNAUTHORIZED_REMOVAL
        assigned_color = extract_color(assigned_route)
        if assigned_color is None:
            # Identified driver, but not on today's schedule: the removal is
            # unauthorized (not merely a wrong-route mistake).
            return cls.UNAUTHORIZED_REMOVAL
        if assigned_color == taken_color:
            return cls.SUCCESS
        return cls.WRONG_ROUTE_ALERT


@dataclass(frozen=True)
class EventRecord:
    """One row of the ``checkouts`` table."""

    id: int
    timestamp: str
    driver_id: str
    assigned_route: str
    taken_route: str
    status: str
    clip_path: str

    @classmethod
    def from_row(cls, row: Tuple[Any, ...]) -> "EventRecord":
        """Build a record from a raw SQLite row tuple."""
        (
            event_id,
            timestamp,
            driver_id,
            assigned_route,
            taken_route,
            status,
            clip_path,
        ) = row
        return cls(
            id=int(event_id),
            timestamp=str(timestamp),
            driver_id=str(driver_id),
            assigned_route=str(assigned_route),
            taken_route=str(taken_route),
            status=str(status),
            clip_path=str(clip_path),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the record as a plain dict (JSON-friendly)."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "driver_id": self.driver_id,
            "assigned_route": self.assigned_route,
            "taken_route": self.taken_route,
            "status": self.status,
            "clip_path": self.clip_path,
        }


class DatabaseManager:
    """Thread-safe SQLite manager for key custody events.

    Args:
        config_path: Path to the L.O.C.K.I. JSON configuration file. The
            ``database.path`` key selects the SQLite file.
        config: Pre-parsed config dict (alternative to ``config_path``).
        db_path: Explicit DB path override; wins over the config value.
            Pass ``":memory:"`` for an ephemeral in-memory database (used by
            the unit tests).

    Raises:
        ConfigError: If the configuration is invalid.
        DatabaseError: If the database file cannot be opened or migrated.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
        db_path: Optional[str] = None,
    ) -> None:
        if config is None:
            config = load_config(Path(config_path) if config_path else "config/config.json")
        raw = config.get("database", {}) or {}
        if not isinstance(raw, dict):
            raise ConfigError("Configuration section 'database' must be an object.")

        configured_path = str(raw.get("path", "data/logs/key_events.db"))
        self.db_path = db_path if db_path is not None else configured_path

        # WAL needs a real filesystem; ':memory:' stays in default mode.
        self._is_memory = self.db_path == ":memory:"
        if not self._is_memory:
            parent = Path(self.db_path).parent
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise DatabaseError(
                    f"Cannot create database directory {parent}: {exc}"
                ) from exc

        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connect_and_migrate()

    # ------------------------------------------------------------------ #
    # Connection / schema
    # ------------------------------------------------------------------ #

    def _connect_and_migrate(self) -> None:
        """Open the connection and ensure the schema exists.

        Raises:
            DatabaseError: If the file cannot be opened or the schema cannot
                be created.
        """
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA foreign_keys = ON")
            if not self._is_memory:
                # WAL lets readers work while a writer commits; the rollback
                # journal avoids a corrupted file on power loss mid-write.
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = FULL")
            with self._lock:
                self._conn = conn
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Cannot open/migrate database '{self.db_path}': {exc}"
            ) from exc
        LOGGER.info("Database ready: %s (schema v%s)", self.db_path, _SCHEMA_VERSION)

    def _ensure_conn(self) -> sqlite3.Connection:
        """Return the live connection or raise."""
        if self._conn is None:
            raise DatabaseError("Database connection is closed.")
        return self._conn

    def close(self) -> None:
        """Close the connection. Idempotent."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Error closing database: %s", exc)
                self._conn = None
        LOGGER.info("Database closed: %s", self.db_path)

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def log_event(
        self,
        driver_id: str,
        assigned_route: str = "",
        taken_route: str = "",
        status: EventStatus | str = EventStatus.SUCCESS,
        clip_path: str = "",
        timestamp: Optional[str] = None,
    ) -> int:
        """Insert one checkout event and return its row id.

        Args:
            driver_id: Matched driver id or the fallback id.
            assigned_route: Schedule route for the driver ('' if unknown).
            taken_route: Route of the key actually taken ('' if unknown).
            status: An :class:`EventStatus` or its exact string value.
            clip_path: Path of the exported evidence clip ('' if none).
            timestamp: Optional ISO-8601 override; defaults to now (UTC).

        Raises:
            DatabaseError: If the insert fails or ``status`` is invalid.
        """
        status_value = status.value if isinstance(status, EventStatus) else str(status)
        if status_value not in {s.value for s in EventStatus}:
            raise DatabaseError(
                f"Invalid status {status_value!r}; expected one of "
                f"{[s.value for s in EventStatus]}."
            )
        ts = timestamp or _utc_now_iso()
        sql = (
            "INSERT INTO checkouts "
            "(timestamp, driver_id, assigned_route, taken_route, status, clip_path) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        try:
            with self._lock:
                conn = self._ensure_conn()
                cursor = conn.execute(
                    sql,
                    (
                        ts,
                        str(driver_id),
                        str(assigned_route or ""),
                        str(taken_route or ""),
                        status_value,
                        str(clip_path or ""),
                    ),
                )
                conn.commit()
                return int(cursor.lastrowid or 0)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to insert event: {exc}") from exc

    def log_bulk_events(self, events: List[Dict[str, Any]]) -> int:
        """Insert many events in a single transaction; returns count written.

        Raises:
            DatabaseError: If any row fails validation or the insert fails.
        """
        rows: List[Tuple[Any, ...]] = []
        valid = {s.value for s in EventStatus}
        now = _utc_now_iso()
        for event in events:
            status_value = event.get("status", "")
            status_value = (
                status_value.value if isinstance(status_value, EventStatus) else str(status_value)
            )
            if status_value not in valid:
                raise DatabaseError(f"Invalid status {status_value!r} in bulk batch.")
            rows.append(
                (
                    str(event.get("timestamp") or now),
                    str(event.get("driver_id", "")),
                    str(event.get("assigned_route", "")),
                    str(event.get("taken_route", "")),
                    status_value,
                    str(event.get("clip_path", "")),
                )
            )
        sql = (
            "INSERT INTO checkouts "
            "(timestamp, driver_id, assigned_route, taken_route, status, clip_path) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        try:
            with self._lock:
                conn = self._ensure_conn()
                conn.executemany(sql, rows)
                conn.commit()
            return len(rows)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Bulk insert failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def recent_events(self, limit: int = 50) -> List[EventRecord]:
        """Return the newest ``limit`` events (id descending)."""
        try:
            with self._lock:
                conn = self._ensure_conn()
                rows = conn.execute(
                    "SELECT id, timestamp, driver_id, assigned_route, taken_route, "
                    "status, clip_path FROM checkouts ORDER BY id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc
        return [EventRecord.from_row(r) for r in rows]

    def events_by_driver(
        self,
        driver_id: str,
        limit: int = 50,
    ) -> List[EventRecord]:
        """Return a driver's newest events, newest first."""
        try:
            with self._lock:
                conn = self._ensure_conn()
                rows = conn.execute(
                    "SELECT id, timestamp, driver_id, assigned_route, taken_route, "
                    "status, clip_path FROM checkouts WHERE driver_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (str(driver_id), int(limit)),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc
        return [EventRecord.from_row(r) for r in rows]

    def events_by_status(
        self,
        status: EventStatus | str,
        limit: int = 50,
    ) -> List[EventRecord]:
        """Return the newest events with the given status."""
        status_value = status.value if isinstance(status, EventStatus) else str(status)
        try:
            with self._lock:
                conn = self._ensure_conn()
                rows = conn.execute(
                    "SELECT id, timestamp, driver_id, assigned_route, taken_route, "
                    "status, clip_path FROM checkouts WHERE status = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (status_value, int(limit)),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc
        return [EventRecord.from_row(r) for r in rows]

    def event_count(self) -> int:
        """Total number of checkout rows (test/monitoring convenience)."""
        try:
            with self._lock:
                conn = self._ensure_conn()
                row = conn.execute("SELECT COUNT(*) FROM checkouts").fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Count failed: {exc}") from exc
        return int(row[0]) if row else 0

    def export_json(self, path: str | Path, limit: int = 1000) -> int:
        """Dump the newest ``limit`` events to a JSON file; returns count.

        Raises:
            DatabaseError: If the query or file write fails.
        """
        records = self.recent_events(limit=limit)
        payload = [record.as_dict() for record in records]
        out_path = Path(path)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise DatabaseError(f"Cannot write export {out_path}: {exc}") from exc
        LOGGER.info("Exported %d events to %s", len(payload), out_path)
        return len(payload)


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with explicit offset."""
    return datetime.now(timezone.utc).isoformat()
