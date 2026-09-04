"""
schedule_manager.py
===================

Daily driver schedule loading for the L.O.C.K.I. system
(Live Optical Custody & Key Identifier).

Reads a CSV file (default ``config/schedule.csv``) mapping each driver to
their assigned route color::

    driver_id,assigned_route
    Driver_101,Route_RED
    Driver_102,Route_BLUE

The route color is normalized (case/spacing tolerant) and translated into
the canonical color names used by :mod:`src.color_detector`, so the rest of
the pipeline can answer "is the key this driver just took the color they
were scheduled for?".

Typical usage::

    from src.schedule_manager import ScheduleManager

    schedule = ScheduleManager("config/config.json")
    schedule.get_route("Driver_101")       # -> "red"
    schedule.get_driver_for_color("red")   # -> ["Driver_101", "Driver_105"]
    schedule.is_scheduled("Driver_101")    # -> True
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from src.color_detector import ConfigError, extract_color, load_config

LOGGER = logging.getLogger(__name__)

__all__ = ["ScheduleError", "ScheduleManager", "ScheduleEntry"]

#: Canonical fob colors shared with the color detector (for error messages).
_CANONICAL_COLORS = ("red", "blue", "green", "orange")


class ScheduleError(RuntimeError):
    """Raised when the schedule CSV is missing, malformed, or inconsistent."""


class ScheduleEntry(NamedTuple):
    """One driver-to-route assignment from the schedule."""

    driver_id: str
    route: str  # raw value from the CSV, e.g. "Route_RED"
    color: str  # canonical color, e.g. "red"


def _normalize_route(raw: str) -> Optional[str]:
    """Translate a free-text route name into a canonical color.

    Thin wrapper over the shared :func:`src.color_detector.extract_color`
    so schedule, detector, and database all parse route text identically.
    """
    return extract_color(raw)


class ScheduleManager:
    """Loads and queries the daily driver -> route assignment schedule.

    Args:
        config_path: Path to the L.O.C.K.I. JSON configuration file. The
            ``schedule.csv_path`` key selects the CSV file.
        config: Pre-parsed config dict (alternative to ``config_path``).
        csv_path: Explicit CSV path override; wins over the config value.

    Raises:
        ConfigError: If the configuration is invalid.
        ScheduleError: If the CSV is missing (and required), unreadable, or
            malformed.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
        csv_path: Optional[str] = None,
    ) -> None:
        if config is None:
            config = load_config(Path(config_path) if config_path else "config/config.json")
        raw = config.get("schedule", {}) or {}
        if not isinstance(raw, dict):
            raise ConfigError("Configuration section 'schedule' must be an object.")

        self._required = bool(raw.get("required", True))
        configured_path = str(raw.get("csv_path", "config/schedule.csv"))
        self.csv_path = csv_path if csv_path is not None else configured_path

        self._by_driver: Dict[str, ScheduleEntry] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Parse the schedule CSV into the internal driver map.

        Raises:
            ScheduleError: If the file is required but missing, unreadable,
                or contains malformed rows.
        """
        path = Path(self.csv_path)
        if not path.is_file():
            if self._required:
                raise ScheduleError(
                    f"Schedule CSV not found: {path}. Create it or set "
                    "schedule.required=false in the config."
                )
            LOGGER.warning("Schedule CSV missing (non-fatal): %s", path)
            return

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ScheduleError(f"Schedule CSV is empty: {path}")

                # Tolerate column-order/casing variations.
                columns = {name.strip().lower(): name for name in reader.fieldnames}
                driver_col = columns.get("driver_id")
                route_col = columns.get("assigned_route")
                if driver_col is None or route_col is None:
                    raise ScheduleError(
                        f"Schedule CSV must have 'driver_id' and 'assigned_route' "
                        f"columns; found {reader.fieldnames} in {path}."
                    )

                for row_number, row in enumerate(reader, start=2):  # header = line 1
                    driver_id = (row.get(driver_col) or "").strip()
                    route_raw = (row.get(route_col) or "").strip()
                    if not driver_id:
                        LOGGER.warning("%s:%d: empty driver_id, skipping", path, row_number)
                        continue
                    color = _normalize_route(route_raw)
                    if color is None:
                        raise ScheduleError(
                            f"{path}:{row_number}: route '{route_raw}' for "
                            f"'{driver_id}' does not contain a recognized color "
                            f"({', '.join(_CANONICAL_COLORS)})."
                        )
                    if driver_id in self._by_driver:
                        LOGGER.warning(
                            "%s:%d: duplicate driver '%s'; keeping first "
                            "assignment",
                            path,
                            row_number,
                            driver_id,
                        )
                        continue
                    self._by_driver[driver_id] = ScheduleEntry(
                        driver_id=driver_id,
                        route=route_raw,
                        color=color,
                    )
        except UnicodeDecodeError as exc:
            raise ScheduleError(f"Schedule CSV is not valid UTF-8: {path} ({exc})") from exc
        except csv.Error as exc:
            raise ScheduleError(f"Malformed schedule CSV {path}: {exc}") from exc
        except OSError as exc:
            raise ScheduleError(f"Cannot read schedule CSV {path}: {exc}") from exc

        if not self._by_driver and self._required:
            raise ScheduleError(f"Schedule CSV contains no driver rows: {path}")

        LOGGER.info(
            "Schedule loaded: %d driver(s) from %s", len(self._by_driver), path
        )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_route(self, driver_id: str) -> Optional[str]:
        """Canonical color for a driver, or ``None`` if not scheduled."""
        entry = self._by_driver.get(str(driver_id).strip())
        return entry.color if entry else None

    def get_route_raw(self, driver_id: str) -> Optional[str]:
        """Raw route text from the CSV (e.g. ``"Route_RED"``), or ``None``."""
        entry = self._by_driver.get(str(driver_id).strip())
        return entry.route if entry else None

    def get_driver_for_color(self, color: str) -> List[str]:
        """All driver ids assigned to a canonical color (config order)."""
        color_l = str(color).strip().lower()
        return [
            entry.driver_id
            for entry in self._by_driver.values()
            if entry.color == color_l
        ]

    def is_scheduled(self, driver_id: str) -> bool:
        """True if the driver appears on today's schedule."""
        return str(driver_id).strip() in self._by_driver

    def all_entries(self) -> List[ScheduleEntry]:
        """Every assignment, in schedule order."""
        return list(self._by_driver.values())

    def __len__(self) -> int:
        return len(self._by_driver)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ScheduleManager({len(self._by_driver)} drivers, {self.csv_path})"
