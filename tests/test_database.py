"""Unit tests for :mod:`src.database` (in-memory SQLite, no file I/O)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import (  # noqa: E402
    DatabaseError,
    DatabaseManager,
    EventRecord,
    EventStatus,
)


@pytest.fixture()
def db() -> DatabaseManager:
    """A fresh in-memory database per test."""
    manager = DatabaseManager(db_path=":memory:")
    yield manager
    manager.close()


class TestSchemaAndLifecycle:
    def test_log_and_read_back(self, db: DatabaseManager) -> None:
        event_id = db.log_event(
            driver_id="Driver_101",
            assigned_route="red",
            taken_route="red",
            status=EventStatus.SUCCESS,
            clip_path="evidence/a.mp4",
        )
        assert event_id == 1
        events = db.recent_events()
        assert len(events) == 1
        record = events[0]
        assert record.driver_id == "Driver_101"
        assert record.status == "SUCCESS"
        assert record.clip_path == "evidence/a.mp4"
        assert record.timestamp  # ISO string present

    def test_autoincrement_ids(self, db: DatabaseManager) -> None:
        first = db.log_event(driver_id="A", status=EventStatus.SUCCESS)
        second = db.log_event(driver_id="B", status=EventStatus.SUCCESS)
        assert second == first + 1

    def test_close_is_idempotent(self, db: DatabaseManager) -> None:
        db.close()
        db.close()  # must not raise

    def test_write_after_close_raises(self, db: DatabaseManager) -> None:
        db.close()
        with pytest.raises(DatabaseError, match="closed"):
            db.log_event(driver_id="A", status=EventStatus.SUCCESS)

    def test_context_manager_closes(self) -> None:
        with DatabaseManager(db_path=":memory:") as manager:
            manager.log_event(driver_id="A", status=EventStatus.SUCCESS)
        assert manager._conn is None

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "logs" / "events.db"
        with DatabaseManager(db_path=str(target)) as manager:
            manager.log_event(driver_id="A", status=EventStatus.SUCCESS)
        assert target.exists()


class TestStatuses:
    def test_invalid_status_raises(self, db: DatabaseManager) -> None:
        with pytest.raises(DatabaseError, match="Invalid status"):
            db.log_event(driver_id="A", status="BANANA")

    def test_string_status_accepted(self, db: DatabaseManager) -> None:
        db.log_event(driver_id="A", status="WRONG_ROUTE_ALERT")
        assert db.recent_events()[0].status == "WRONG_ROUTE_ALERT"


class TestClassify:
    def test_success(self) -> None:
        assert (
            EventStatus.classify(
                assigned_route="red", taken_route="Route_RED", driver_known=True
            )
            is EventStatus.SUCCESS
        )

    def test_wrong_route(self) -> None:
        assert (
            EventStatus.classify(
                assigned_route="red", taken_route="blue", driver_known=True
            )
            is EventStatus.WRONG_ROUTE_ALERT
        )

    def test_unknown_driver_unauthorized(self) -> None:
        assert (
            EventStatus.classify(
                assigned_route=None, taken_route="blue", driver_known=False
            )
            is EventStatus.UNAUTHORIZED_REMOVAL
        )

    def test_known_driver_off_schedule_unauthorized(self) -> None:
        assert (
            EventStatus.classify(
                assigned_route=None, taken_route="red", driver_known=True
            )
            is EventStatus.UNAUTHORIZED_REMOVAL
        )

    def test_unknown_taken_route_unauthorized(self) -> None:
        assert (
            EventStatus.classify(
                assigned_route="red", taken_route="", driver_known=True
            )
            is EventStatus.UNAUTHORIZED_REMOVAL
        )


class TestQueries:
    @pytest.fixture()
    def populated(self, db: DatabaseManager) -> DatabaseManager:
        db.log_bulk_events(
            [
                {"driver_id": "Driver_101", "assigned_route": "red", "taken_route": "red", "status": "SUCCESS"},
                {"driver_id": "Driver_102", "assigned_route": "blue", "taken_route": "red", "status": "WRONG_ROUTE_ALERT"},
                {"driver_id": "UNIDENTIFIED", "taken_route": "green", "status": "UNAUTHORIZED_REMOVAL"},
                {"driver_id": "Driver_101", "assigned_route": "red", "taken_route": "orange", "status": "WRONG_ROUTE_ALERT"},
            ]
        )
        return db

    def test_recent_order(self, populated: DatabaseManager) -> None:
        events = populated.recent_events()
        assert [e.id for e in events] == [4, 3, 2, 1]

    def test_by_driver(self, populated: DatabaseManager) -> None:
        events = populated.events_by_driver("Driver_101")
        assert len(events) == 2
        assert all(e.driver_id == "Driver_101" for e in events)

    def test_by_status(self, populated: DatabaseManager) -> None:
        alerts = populated.events_by_status(EventStatus.WRONG_ROUTE_ALERT)
        assert len(alerts) == 2
        unauthorized = populated.events_by_status("UNAUTHORIZED_REMOVAL")
        assert len(unauthorized) == 1
        assert unauthorized[0].driver_id == "UNIDENTIFIED"

    def test_limit(self, populated: DatabaseManager) -> None:
        assert len(populated.recent_events(limit=2)) == 2

    def test_count(self, populated: DatabaseManager) -> None:
        assert populated.event_count() == 4

    def test_bulk_invalid_status_raises_and_writes_nothing(
        self, db: DatabaseManager
    ) -> None:
        with pytest.raises(DatabaseError, match="Invalid status"):
            db.log_bulk_events(
                [
                    {"driver_id": "A", "status": "SUCCESS"},
                    {"driver_id": "B", "status": "NOPE"},
                ]
            )
        assert db.event_count() == 0

    def test_record_as_dict(self, populated: DatabaseManager) -> None:
        record = populated.recent_events()[0]
        payload = record.as_dict()
        assert set(payload) == {
            "id", "timestamp", "driver_id", "assigned_route",
            "taken_route", "status", "clip_path",
        }
        json.dumps(payload)  # must be JSON-serializable

    def test_export_json(self, populated: DatabaseManager, tmp_path: Path) -> None:
        out = tmp_path / "export" / "events.json"
        count = populated.export_json(out, limit=10)
        assert count == 4
        data = json.loads(out.read_text())
        assert len(data) == 4


class TestConfigParsing:
    def test_default_path(self) -> None:
        from src.database import DatabaseManager as DM

        manager = DM(config={"database": {"path": ":memory:"}})
        assert manager.db_path == ":memory:"
        manager.close()

    def test_db_path_override_wins(self, tmp_path: Path) -> None:
        target = tmp_path / "override.db"
        manager = DatabaseManager(
            config={"database": {"path": ":memory:"}}, db_path=str(target)
        )
        assert manager.db_path == str(target)
        manager.close()
