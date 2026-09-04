"""Unit tests for :mod:`src.hand_tracker`.

MediaPipe is an optional heavy dependency: config/geometry tests always run,
while the end-to-end inference test is skipped automatically when mediapipe
is absent or the model asset cannot be fetched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hand_tracker import (  # noqa: E402
    HandTracker,
    HandTrackerError,
    HandTrackingConfig,
)

try:  # optional dependency
    import mediapipe  # noqa: F401

    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


@pytest.fixture()
def config_dict() -> dict:
    """Minimal valid config including slots + hand_tracking sections."""
    return {
        "slots": [
            {"id": "SLOT-01", "route": "ROUTE-A", "roi": [10, 10, 60, 60]},
            {"id": "SLOT-02", "route": "ROUTE-B", "roi": [70, 10, 120, 60]},
        ],
        "route_colors": {"ROUTE-A": "red", "ROUTE-B": "blue"},
        "hsv_ranges": {
            "red": [{"h_min": 0, "s_min": 80, "v_min": 60, "h_max": 10, "s_max": 255, "v_max": 255}],
            "blue": [{"h_min": 90, "s_min": 80, "v_min": 40, "h_max": 130, "s_max": 255, "v_max": 255}],
        },
        "hand_tracking": {
            "enabled": True,
            "num_hands": 1,
            "center_trigger": True,
            "edge_trigger": False,
            # Never download the real model during unit tests.
            "allow_model_download": False,
            "model_path": "models/definitely-missing.task",
        },
    }


class TestHandTrackingConfig:
    def test_defaults(self, config_dict: dict) -> None:
        cfg = HandTrackingConfig.from_config(config_dict)
        assert cfg.enabled is True
        assert cfg.num_hands == 1
        assert cfg.running_mode == "video"
        assert cfg.center_trigger is True and cfg.edge_trigger is False
        assert cfg.min_hand_detection_confidence == pytest.approx(0.5)  # Tasks default

    def test_missing_section_uses_defaults(self, config_dict: dict) -> None:
        del config_dict["hand_tracking"]
        cfg = HandTrackingConfig.from_config(config_dict)
        assert cfg.num_hands == 2
        assert cfg.running_mode == "video"
        assert cfg.allow_model_download is True

    def test_invalid_mode_raises(self, config_dict: dict) -> None:
        config_dict["hand_tracking"]["running_mode"] = "backwards"
        with pytest.raises(Exception, match="running_mode"):
            HandTrackingConfig.from_config(config_dict)

    def test_both_triggers_off_raises(self, config_dict: dict) -> None:
        config_dict["hand_tracking"].update(center_trigger=False, edge_trigger=False)
        with pytest.raises(Exception, match="center_trigger"):
            HandTrackingConfig.from_config(config_dict)

    def test_bad_num_hands_raises(self, config_dict: dict) -> None:
        config_dict["hand_tracking"]["num_hands"] = 0
        with pytest.raises(Exception, match="num_hands"):
            HandTrackingConfig.from_config(config_dict)

    def test_bad_confidence_raises(self, config_dict: dict) -> None:
        config_dict["hand_tracking"]["min_hand_detection_confidence"] = 1.5
        with pytest.raises(Exception, match="confidence"):
            HandTrackingConfig.from_config(config_dict)

    def test_bad_padding_raises(self, config_dict: dict) -> None:
        config_dict["hand_tracking"]["bbox_padding"] = 0.9
        with pytest.raises(Exception, match="bbox_padding"):
            HandTrackingConfig.from_config(config_dict)


class TestGeometry:
    def test_rects_overlap_true(self) -> None:
        assert HandTracker._rects_overlap((0, 0, 10, 10), (5, 5, 20, 20)) is True

    def test_rects_overlap_edge_touch(self) -> None:
        # Inclusive edges: touching borders count as overlapping.
        assert HandTracker._rects_overlap((0, 0, 10, 10), (10, 0, 20, 10)) is True

    def test_rects_overlap_false(self) -> None:
        assert HandTracker._rects_overlap((0, 0, 10, 10), (11, 11, 20, 20)) is False

    def test_point_in_rect(self) -> None:
        assert HandTracker._point_in_rect((10, 10), (10, 10, 60, 60)) is True
        assert HandTracker._point_in_rect((9, 10), (10, 10, 60, 60)) is False
        assert HandTracker._point_in_rect((60, 60), (10, 10, 60, 60)) is True

    def test_hand_bbox_clamps_and_pads(self) -> None:
        class LM:  # minimal stand-in for a NormalizedLandmark
            def __init__(self, x: float, y: float) -> None:
                self.x, self.y = x, y

        landmarks = [LM(0.0, 0.0), LM(0.5, 0.5)]
        assert HandTracker._hand_bbox(landmarks, 100, 100) == (0, 0, 50, 50)
        padded = HandTracker._hand_bbox(landmarks, 100, 100, padding=1.0)
        assert padded[0] <= 0 and padded[1] <= 0  # pads and clamps at 0


class TestLifecycle:
    def test_enabled_false_is_inert(self, config_dict: dict) -> None:
        config_dict["hand_tracking"]["enabled"] = False
        tracker = HandTracker(config=config_dict)
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        assert tracker.get_active_slots(frame) == []  # no landmarker, no crash
        tracker.close()
        assert tracker._closed is True

    def test_missing_model_without_download_raises(self, config_dict: dict) -> None:
        with pytest.raises(HandTrackerError, match="allow_model_download"):
            HandTracker(config=config_dict)

    def test_close_is_idempotent(self, config_dict: dict) -> None:
        config_dict["hand_tracking"]["enabled"] = False
        tracker = HandTracker(config=config_dict)
        tracker.close()
        tracker.close()  # must not raise

    def test_frame_validation(self, config_dict: dict) -> None:
        config_dict["hand_tracking"]["enabled"] = False
        tracker = HandTracker(config=config_dict)
        with pytest.raises(HandTrackerError, match="non-empty BGR"):
            tracker.get_active_slots(None)  # type: ignore[arg-type]
        with pytest.raises(HandTrackerError, match="non-empty BGR"):
            tracker.get_active_slots(np.zeros((0, 0, 3), dtype=np.uint8))
        tracker.close()


@pytest.mark.skipif(not HAS_MEDIAPIPE, reason="mediapipe not installed")
class TestRealInference:
    """End-to-end check against the real model (downloads ~7 MB once)."""

    def test_no_hands_returns_empty(self, config_dict: dict) -> None:
        config_dict["hand_tracking"].update(
            model_path="models/hand_landmarker.task", allow_model_download=True
        )
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        with HandTracker(config=config_dict) as tracker:
            assert tracker.get_active_slots(frame) == []
            # Slot filtering paths still behave with a live landmarker.
            assert tracker.get_active_slots(frame, slot_ids=["SLOT-01"]) == []
            assert tracker.get_active_slots(frame, slot_ids=["NOPE"]) == []

    def test_annotate_runs(self, config_dict: dict) -> None:
        config_dict["hand_tracking"].update(
            model_path="models/hand_landmarker.task", allow_model_download=True
        )
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        with HandTracker(config=config_dict) as tracker:
            out = tracker.annotate(frame, activations=[])
        assert out.shape == frame.shape
        assert out is not frame


@pytest.mark.skipif(HAS_MEDIAPIPE, reason="mediapipe installed")
class TestMissingMediapipe:
    def test_helpful_error_when_missing(self, config_dict: dict) -> None:
        with pytest.raises(HandTrackerError, match="pip install"):
            HandTracker(config=config_dict)
