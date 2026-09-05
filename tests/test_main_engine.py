"""Unit tests for :mod:`main` (engine loop + checkout tracking)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from main import CheckoutTracker, LockiEngine  # noqa: E402
from src.camera import rois_outside_frame, validate_slot_rois  # noqa: E402
from src.color_detector import ConfigError, load_config, parse_slots  # noqa: E402
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
        frame = np.full((1920, 1080, 3), (70, 70, 70), dtype=np.uint8)
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
        frame = np.full((1920, 1080, 3), (70, 70, 70), dtype=np.uint8)
        x1, y1, x2, y2 = engine.detector.get_slot("SLOT-01").roi
        frame[y1:y2, x1:x2] = (0, 0, 255)  # takes a RED key instead

        with patch.object(engine, "_save_evidence", return_value=None):
            engine.handle_grasp_event("SLOT-01", frame, driver="Driver_102")
        record = engine.db.recent_events()[0]
        assert record.status == "WRONG_ROUTE_ALERT"

    def test_unidentified_flow(self, engine: LockiEngine) -> None:
        frame = np.full((1920, 1080, 3), (70, 70, 70), dtype=np.uint8)
        x1, y1, x2, y2 = engine.detector.get_slot("SLOT-01").roi
        frame[y1:y2, x1:x2] = (0, 0, 255)
        with patch.object(engine, "_save_evidence", return_value=None):
            engine.handle_grasp_event("SLOT-01", frame, driver=None)
        record = engine.db.recent_events()[0]
        assert record.status == "UNAUTHORIZED_REMOVAL"
        assert record.driver_id == "UNIDENTIFIED"

    def test_unknown_slot_returns_none(self, engine: LockiEngine) -> None:
        frame = np.full((1920, 1080, 3), 70, dtype=np.uint8)
        assert engine.handle_grasp_event("NOPE", frame) is None

    def test_vanished_key_uses_slot_assignment(self, engine: LockiEngine) -> None:
        # Key already gone: evidence falls back to the slot's route color.
        engine.simulate_identity("Driver_101")
        frame = np.full((1920, 1080, 3), (70, 70, 70), dtype=np.uint8)
        with patch.object(engine, "_save_evidence", return_value=None):
            engine.handle_grasp_event("SLOT-01", frame, driver="Driver_101")
        record = engine.db.recent_events()[0]
        assert record.taken_route == "red"  # SLOT-01 is Route_A -> red
        assert record.status == "SUCCESS"

    def test_process_frame_smoke(self, engine: LockiEngine) -> None:
        engine.simulate_identity("Driver_101")
        frame = np.full((1920, 1080, 3), (70, 70, 70), dtype=np.uint8)
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


class TestCheckWindowAspect:
    """``main.check_window_aspect``: white-bar (letterbox) detection guard."""

    # Portrait 1080x1920 frame as delivered by the camera (h, w, c).
    frame = np.full((1920, 1080, 3), 70, dtype=np.uint8)

    def test_matching_aspect_returns_one(self) -> None:
        with patch.object(main, "_HAS_DISPLAY", True), patch.object(
            main.cv2, "getWindowImageRect", return_value=(0, 0, 540, 960)
        ):
            assert main.check_window_aspect("monitor", self.frame) == pytest.approx(1.0)

    def test_mismatched_aspect_reported(self) -> None:
        # Landscape window (960x540) showing a portrait frame: ~0.316x.
        with patch.object(main, "_HAS_DISPLAY", True), patch.object(
            main.cv2, "getWindowImageRect", return_value=(0, 0, 960, 540)
        ):
            ratio = main.check_window_aspect("monitor", self.frame)
        assert ratio == pytest.approx(0.31640625)
        assert not (1.0 - main.ASPECT_TOLERANCE <= ratio <= 1.0 + main.ASPECT_TOLERANCE)

    def test_window_closed_returns_none(self) -> None:
        with patch.object(main, "_HAS_DISPLAY", True), patch.object(
            main.cv2, "getWindowImageRect", side_effect=cv2.error("no window")
        ):
            assert main.check_window_aspect("monitor", self.frame) is None

    def test_headless_returns_none_without_probing(self) -> None:
        with patch.object(main, "_HAS_DISPLAY", False), patch.object(
            main.cv2, "getWindowImageRect", return_value=(0, 0, 540, 960)
        ) as probe:
            assert main.check_window_aspect("monitor", self.frame) is None
        probe.assert_not_called()

    @pytest.mark.parametrize("rect", [None, (0, 0, 0, 960), (0, 0, 540, 0)])
    def test_degenerate_rect_returns_none(self, rect) -> None:
        with patch.object(main, "_HAS_DISPLAY", True), patch.object(
            main.cv2, "getWindowImageRect", return_value=rect
        ):
            assert main.check_window_aspect("monitor", self.frame) is None


class TestSlotRoiValidation:
    """Slot ROIs must fit inside the camera frame at startup."""

    def test_rois_all_inside(self) -> None:
        cfg = {"slots": [{"id": "A", "roi": [10, 10, 100, 100]}]}
        assert rois_outside_frame(parse_slots(cfg), 200, 200) == []

    def test_roi_overflowing_frame(self) -> None:
        cfg = {"slots": [{"id": "A", "roi": [150, 10, 300, 100]}]}
        assert rois_outside_frame(parse_slots(cfg), 200, 200) == ["A"]

    def test_roi_negative_origin(self) -> None:
        cfg = {"slots": [{"id": "B", "roi": [-5, 0, 50, 50]}]}
        assert rois_outside_frame(parse_slots(cfg), 200, 200) == ["B"]

    def test_zero_size_frame_reports_all(self) -> None:
        cfg = {"slots": [{"id": "A", "roi": [0, 0, 10, 10]}, {"id": "B", "roi": [0, 0, 10, 10]}]}
        assert rois_outside_frame(parse_slots(cfg), 0, 0) == ["A", "B"]

    def test_validate_raises_config_error(self) -> None:
        config = load_config("config/config.json")
        config["slots"][0]["roi"] = [0, 0, 5000, 5000]
        with pytest.raises(ConfigError, match="SLOT-01"):
            validate_slot_rois(
                config,
                config["camera"]["frame_width"],
                config["camera"]["frame_height"],
            )

    def test_engine_fails_fast_on_bad_roi(self, tmp_path: Path) -> None:
        config = load_config("config/config.json")
        config["slots"][0]["roi"] = [0, 0, 5000, 5000]
        config["database"]["path"] = str(tmp_path / "events.db")
        with pytest.raises(ConfigError, match="SLOT-01"):
            LockiEngine(config=config)

    def test_actual_smaller_than_rois_warns(
        self, engine: LockiEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch.object(
            engine.camera, "actual_frame_size", return_value=(800, 1200)
        ), caplog.at_level(logging.WARNING):
            engine._warn_rois_vs_actual_device()
        assert "SLOT-06" in caplog.text  # bottom-right ROI, outside 800 wide

    def test_actual_smaller_than_rois_sets_overlay_warning(
        self, engine: LockiEngine
    ) -> None:
        with patch.object(engine.camera, "actual_frame_size", return_value=(800, 1200)):
            engine._warn_rois_vs_actual_device()
        assert "ROI/CAMERA MISMATCH" in engine.alarm._system_warning
        assert "SLOT-06" in engine.alarm._system_warning

    def test_actual_fits_rois_is_silent(
        self, engine: LockiEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch.object(
            engine.camera, "actual_frame_size", return_value=(1080, 1920)
        ), caplog.at_level(logging.WARNING):
            engine._warn_rois_vs_actual_device()
        assert "outside" not in caplog.text.lower()

    def test_actual_fits_rois_clears_overlay_warning(
        self, engine: LockiEngine
    ) -> None:
        engine.alarm.set_system_warning("ROI/CAMERA MISMATCH: stale")
        with patch.object(engine.camera, "actual_frame_size", return_value=(1080, 1920)):
            engine._warn_rois_vs_actual_device()
        assert engine.alarm._system_warning == ""


class TestDebugRoisOverlay:
    """``--debug-rois``: slot ROI alignment boxes on the displayed frame."""

    def test_debug_flag_default_off(self, engine: LockiEngine) -> None:
        assert engine.debug_rois is False

    def test_overlay_off_is_identity(self, engine: LockiEngine) -> None:
        frame = np.full((1920, 1080, 3), (70, 70, 70), dtype=np.uint8)
        out = engine._apply_debug_overlays(frame)
        assert out is frame  # untouched, no copy churn per frame

    def test_overlay_on_draws_roi_boxes(self, engine: LockiEngine) -> None:
        engine.debug_rois = True
        frame = np.full((1920, 1080, 3), (70, 70, 70), dtype=np.uint8)
        out = engine._apply_debug_overlays(frame)
        assert out.shape == frame.shape
        # Slot-01 border (left edge, mid height) drawn green: G dominates.
        px = out[400, 40]
        assert int(px[1]) > 120 and int(px[1]) > px[0] and int(px[1]) > px[2]

    def test_parse_args_flag(self) -> None:
        assert main.parse_args(["--demo"]).debug_rois is False
        assert main.parse_args(["--demo", "--debug-rois"]).debug_rois is True
