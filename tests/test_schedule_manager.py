"""Unit tests for :mod:`src.schedule_manager`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schedule_manager import (  # noqa: E402
    ScheduleError,
    ScheduleManager,
    _normalize_route,
)


@pytest.fixture()
def config(tmp_path: Path, request) -> dict:
    """Config pointing at a temp CSV; rows supplied via @pytest.mark.rows."""
    rows = getattr(request, "param", None) or [
        "driver_id,assigned_route",
        "Driver_101,Route_RED",
        "Driver_102,Route_BLUE",
        "Driver_103,Route_GREEN",
        "Driver_104,Route_ORANGE",
    ]
    csv_path = tmp_path / "schedule.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"schedule": {"csv_path": str(csv_path), "required": True}}


class TestLoading:
    def test_loads_sample_assignments(self, config: dict) -> None:
        schedule = ScheduleManager(config=config)
        assert len(schedule) == 4
        assert schedule.get_route("Driver_101") == "red"

    def test_project_schedule_csv_loads(self) -> None:
        schedule = ScheduleManager()  # real config/schedule.csv
        assert len(schedule) >= 1
        assert schedule.get_route("Driver_101") == "red"
        assert schedule.get_route_raw("Driver_101") == "Route_RED"

    def test_missing_required_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ScheduleError, match="not found"):
            ScheduleManager(
                config={"schedule": {"csv_path": str(tmp_path / "nope.csv")}}
            )

    def test_missing_optional_csv_is_empty(self, tmp_path: Path) -> None:
        schedule = ScheduleManager(
            config={"schedule": {"csv_path": str(tmp_path / "nope.csv"), "required": False}}
        )
        assert len(schedule) == 0
        assert schedule.get_route("Driver_101") is None

    def test_header_only_no_rows_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "schedule.csv"
        csv_path.write_text("driver_id,assigned_route\n", encoding="utf-8")
        with pytest.raises(ScheduleError, match="no driver rows"):
            ScheduleManager(
                config={"schedule": {"csv_path": str(csv_path)}}
            )


@pytest.mark.parametrize(
    "config",
    [
        [  # type: ignore[reportUnknownArgumentType]
            "driver_id,assigned_route",
            "Driver_201,route blue",       # lowercase, no underscore
            "Driver_202,ROUTE-GREEN",      # caps
            "Driver_203,BLUE ROUTE 7",     # color buried in text
        ]
    ],
    indirect=True,
)
class TestNormalization:
    def test_case_and_spacing_variants(self, config: dict) -> None:
        schedule = ScheduleManager(config=config)
        assert schedule.get_route("Driver_201") == "blue"
        assert schedule.get_route("Driver_202") == "green"
        assert schedule.get_route("Driver_203") == "blue"

    def test_queries(self, config: dict) -> None:
        schedule = ScheduleManager(config=config)
        assert schedule.is_scheduled("Driver_201") is True
        assert schedule.is_scheduled("Driver_999") is False
        assert schedule.get_driver_for_color("blue") == ["Driver_201", "Driver_203"]
        assert schedule.get_driver_for_color("red") == []
        entries = schedule.all_entries()
        assert entries[0].driver_id == "Driver_201"
        assert entries[0].route == "route blue"


class TestMalformedInput:
    def test_unrecognized_route_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "schedule.csv"
        csv_path.write_text(
            "driver_id,assigned_route\nDriver_301,PURPLE\n", encoding="utf-8"
        )
        with pytest.raises(ScheduleError, match="PURPLE"):
            ScheduleManager(config={"schedule": {"csv_path": str(csv_path)}})

    def test_missing_columns_raise(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "schedule.csv"
        csv_path.write_text("name,route\nAlice,red\n", encoding="utf-8")
        with pytest.raises(ScheduleError, match="assigned_route"):
            ScheduleManager(config={"schedule": {"csv_path": str(csv_path)}})

    def test_empty_driver_skipped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "schedule.csv"
        csv_path.write_text(
            "driver_id,assigned_route\n,red\nDriver_302,blue\n", encoding="utf-8"
        )
        schedule = ScheduleManager(config={"schedule": {"csv_path": str(csv_path)}})
        assert len(schedule) == 1

    def test_duplicate_driver_keeps_first(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "schedule.csv"
        csv_path.write_text(
            "driver_id,assigned_route\n"
            "Driver_303,red\nDriver_303,blue\n",
            encoding="utf-8",
        )
        schedule = ScheduleManager(config={"schedule": {"csv_path": str(csv_path)}})
        assert schedule.get_route("Driver_303") == "red"


class TestNormalizeRoute:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Route_RED", "red"),
            ("route blue", "blue"),
            ("GREEN", "green"),
            ("orange", "orange"),
            ("  Route_Orange  ", "orange"),
            ("", None),
            ("purple", None),
        ],
    )
    def test_variants(self, raw: str, expected: str | None) -> None:
        assert _normalize_route(raw) == expected
