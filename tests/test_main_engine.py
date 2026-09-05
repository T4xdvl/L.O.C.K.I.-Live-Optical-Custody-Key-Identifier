"""Unit tests for :mod:`main` (engine loop + checkout tracking)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from main import CheckoutTracker, LockiEngine  # noqa: E402
from src.color_detector import load_config  # noqa: E402
from src.database import EventStatus  # noqa: E402


@pytest.fixture()
def engine(tmp_path: Path) -> LockiEngine:
    """Engine over the real config with heavy/inference stages stubbed.

    Each test gets its own SQLite file so event history never leaks
    between tests.
    """
    config = load_config("config/config.json")
    config["hand_tracking"]["enabled"] = False  # grasp paths driven directly
    config["face_recognition"]["enabled"] = False  # identity simulated
    config["ui"]["enable_pygame"] = False  # headless test host
    config["ui"]["audio"] = False
    config["database"]["path"] = str(tmp_path / "events.db")
    eng = LockiEngine(config=config)
    yield eng
    eng.stop()


class TestCheckoutTracker:
    def test_requires_confirmation_window(self) -> None:
        tracker = CheckoutTracker(confirm_seconds=0.4, removal_timeout_seconds=3.0)
        t = 100.0
        assert tracker.register_presence("S1", True, t) is False
        assert tracker.register_presence("S1", True, t + 0.2) is False  # too soon
        assert tracker.register_presence("S1", True, t + 0.5) is True  # confirmed

    def test_flyby_cancels(self) -> None:
        tracker = CheckoutTracker(confirm_seconds=0.4, removal_timeout_seconds=3.0)
        t = 100.0
        tracker.register_presence("S1", True, t)
        tracker.register_presence("S1", False, t + 0.1)  # hand left
        assert tracker.register_presence("S1", True, t + 0.3) is False  # restart
        assert tracker.register_presence("S1", True, t + 0.8) is True  # 0.5s held

    def test_one_shot_latch(self) -> None:
        tracker = CheckoutTracker(confirm_seconds=0.1, removal_timeout_seconds=3.0)
        t = 100.0
        assert tracker.register_presence("S1", True, t) is False
        assert tracker.register_presence("S1", True, t + 0.2) is True
        # Still latched: no duplicate events while the latch is live.
        assert tracker.register_presence("S1", True, t + 1.0) is False
        assert tracker.is_latched("S1", t + 1.0) is True
        # After the timeout the slot re-arms.
        assert tracker.is_latched("S1", t + 3.5) is False

    def test_slots_are_independent(self) -> None:
        tracker = CheckoutTracker(confirm_seconds=0.1, removal_timeout_seconds=1.0)
        t = 0.0
        assert tracker.register_presence("A", True, t) is False  # arms at t
        assert tracker.register_presence("A", True, t + 0.2) is True
        assert tracker.register_presence("B", True, t) is False
        assert tracker.register_presence("B", True, t + 0.2) is True
        assert tracker.is_latched("A", t + 0.3) and tracker.is_latched("B", t + 0.3)


class TestEngineGrasp:
    def test_success_flow(self, engine: LockiEngine, tmp_path: Path) -> None:
        engine.simulate_identity("Driver_101")  # Route_RED per schedule
        frame = np.full((1080, 1920, 3), (70, 70, 70), dtype=np.uint8)
        x1, y1, x2, y2 = engine.detector.get_slot("SLOT-01").roi
        frame[y1:y2, x1:x2] = (0, 0, 255)  # red key in SLOT-01

        with patch.object(engine, "_save_evidence", return_value=None):
            event_id = engine.handle_grasp_event("SLOT-01", frame, driver="Driver_101")
        assert event_id is not None
        record = engine.db.recent_events()[0]
        assert record.status == "SUCCESS"
        assert record.driver_id == "Driver_101"
        assert record.taken_route == "red"

    def test_wrong_route_flow(self, engine: LockiEngine) -> None:
        engine.simulate_identity("Driver_102")  # Route_BLUE
        frame = np.full((1080, 1920, 3), (70, 70, 70), dtype=np.uint8)
        x1, y1, x2, y2 = engine.detector.get_slot("SLOT-01").roi
        frame[y1:y2, x1:x2] = (0, 0, 255)  # takes a RED key instead

        with patch.object(engine, "_save_evidence", return_value=None):
            engine.handle_grasp_event("SLOT-01", frame, driver="Driver_102")
        record = engine.db.recent_events()[0]
        assert record.status == "WRONG_ROUTE_ALERT"

    def test_unidentified_flow(self, engine: LockiEngine) -> None:
        frame = np.full((1080, 1920, 3), (70, 70, 70), dtype=np.uint8)
        x1, y1, x2, y2 = engine.detector.get_slot("SLOT-01").roi
        frame[y1:y2, x1:x2] = (0, 0, 255)
        with patch.object(engine, "_save_evidence", return_value=None):
            engine.handle_grasp_event("SLOT-01", frame, driver=None)
        record = engine.db.recent_events()[0]
        assert record.status == "UNAUTHORIZED_REMOVAL"
        assert record.driver_id == "UNIDENTIFIED"

    def test_unknown_slot_returns_none(self, engine: LockiEngine) -> None:
        frame = np.full((1080, 1920, 3), 70, dtype=np.uint8)
        assert engine.handle_grasp_event("NOPE", frame) is None

    def test_vanished_key_uses_slot_assignment(self, engine: LockiEngine) -> None:
        # Key already gone: evidence falls back to the slot's route color.
        engine.simulate_identity("Driver_101")
        frame = np.full((1080, 1920, 3), (70, 70, 70), dtype=np.uint8)
        with patch.object(engine, "_save_evidence", return_value=None):
            engine.handle_grasp_event("SLOT-01", frame, driver="Driver_101")
        record = engine.db.recent_events()[0]
        assert record.taken_route == "red"  # SLOT-01 is Route_A -> red
        assert record.status == "SUCCESS"

    def test_process_frame_smoke(self, engine: LockiEngine) -> None:
        engine.simulate_identity("Driver_101")
        frame = np.full((1080, 1920, 3), (70, 70, 70), dtype=np.uint8)
        x1, y1, x2, y2 = engine.detector.get_slot("SLOT-01").roi
        frame[y1:y2, x1:x2] = (0, 0, 255)
        out = engine.process_frame(frame)
        assert out.shape == frame.shape
        assert out is not frame

    def test_demo_scenarios_log_all_outcomes(self, engine: LockiEngine, tmp_path: Path) -> None:
        """Run the real scripted demo with a fake clock (no waits)."""
        ticks = iter(float(i) * 0.1 for i in range(100000))  # 10 Hz fake clock

        # Media file writes go to a tmp dir; evidence export is stubbed.
        with patch.object(engine, "_save_evidence", return_value=None):
            frames = main.run_demo(
                engine,
                fps=10,
                clock=lambda: next(ticks),
                max_elapsed=13.0,
                stop_on_exit=False,  # keep the DB open for assertions
            )
        assert frames > 100
        # The demo's three events are the newest in this test's own DB.
        statuses = sorted(r.status for r in engine.db.recent_events(limit=3))
        assert statuses == ["SUCCESS", "UNAUTHORIZED_REMOVAL", "WRONG_ROUTE_ALERT"]
