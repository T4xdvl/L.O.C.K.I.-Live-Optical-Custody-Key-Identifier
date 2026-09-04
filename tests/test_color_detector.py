"""Unit tests for :mod:`src.color_detector`.

Synthetic BGR frames are drawn with numpy so the tests exercise the real
OpenCV masking pipeline without needing a camera.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.color_detector import (  # noqa: E402
    ColorDetector,
    ConfigError,
    DetectionSettings,
    FrameError,
    Slot,
    SlotStatus,
)


BGR = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "orange": (0, 165, 255),
}

# Slot ROIs, shared across test classes.
ROI_A = (10, 10, 60, 60)
ROI_B = (70, 10, 120, 60)
ROI_C = (130, 10, 180, 60)


@pytest.fixture()
def config_dict() -> dict:
    """A minimal valid config, as it would be loaded from config.json."""
    return {
        "slots": [
            {"id": "SLOT-01", "route": "ROUTE-A", "roi": [10, 10, 60, 60]},
            {"id": "SLOT-02", "route": "ROUTE-B", "roi": [70, 10, 120, 60]},
            {"id": "SLOT-03", "route": "UNASSIGNED", "roi": [130, 10, 180, 60]},
        ],
        "route_colors": {"ROUTE-A": "red", "ROUTE-B": "blue"},
        "hsv_ranges": {
            "red": [
                {"h_min": 0, "s_min": 80, "v_min": 60, "h_max": 10, "s_max": 255, "v_max": 255},
                {"h_min": 160, "s_min": 80, "v_min": 60, "h_max": 179, "s_max": 255, "v_max": 255},
            ],
            "blue": [
                {"h_min": 90, "s_min": 80, "v_min": 40, "h_max": 130, "s_max": 255, "v_max": 255},
            ],
            "green": [
                {"h_min": 35, "s_min": 60, "v_min": 40, "h_max": 85, "s_max": 255, "v_max": 255},
            ],
            "orange": [
                {"h_min": 10, "s_min": 90, "v_min": 60, "h_max": 25, "s_max": 255, "v_max": 255},
            ],
        },
    }


@pytest.fixture()
def config_file(tmp_path: Path, config_dict: dict) -> Path:
    """The same config written to a temp JSON file."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict), encoding="utf-8")
    return path


@pytest.fixture()
def detector(config_file: Path) -> ColorDetector:
    return ColorDetector(config_file)


def make_frame(color: str | None, roi: tuple[int, int, int, int], size=(200, 200)) -> np.ndarray:
    """Draw a solid-colored ROI on a dark gray background."""
    frame = np.full((size[1], size[0], 3), (40, 40, 40), dtype=np.uint8)
    if color is not None:
        x1, y1, x2, y2 = roi
        frame[y1:y2, x1:x2] = BGR[color]
    return frame


# --------------------------------------------------------------------------- #
# Config loading / validation
# --------------------------------------------------------------------------- #

class TestConfigLoading:
    def test_loads_valid_config(self, detector: ColorDetector) -> None:
        assert len(detector.slots) == 3
        assert detector.route_colors["ROUTE-A"] == "red"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            ColorDetector(tmp_path / "nope.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError):
            ColorDetector(path)

    def test_unsupported_color_raises(self, tmp_path: Path, config_dict: dict) -> None:
        config_dict["route_colors"]["ROUTE-A"] = "purple"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_dict), encoding="utf-8")
        with pytest.raises(ConfigError, match="unsupported color"):
            ColorDetector(path)

    def test_degenerate_roi_raises(self, tmp_path: Path, config_dict: dict) -> None:
        config_dict["slots"][0]["roi"] = [50, 50, 50, 50]
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_dict), encoding="utf-8")
        with pytest.raises(ConfigError, match="degenerate ROI"):
            ColorDetector(path)

    def test_bad_hue_bounds_raise(self, tmp_path: Path, config_dict: dict) -> None:
        config_dict["hsv_ranges"]["blue"][0]["h_max"] = 200
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_dict), encoding="utf-8")
        with pytest.raises(ConfigError, match="hue bounds"):
            ColorDetector(path)

    def test_duplicate_slot_id_raises(self, tmp_path: Path, config_dict: dict) -> None:
        config_dict["slots"][1]["id"] = "SLOT-01"
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_dict), encoding="utf-8")
        with pytest.raises(ConfigError, match="Duplicate slot"):
            ColorDetector(path)


# --------------------------------------------------------------------------- #
# Classification states
# --------------------------------------------------------------------------- #

class TestClassification:
    def test_present(self, detector: ColorDetector) -> None:
        frame = make_frame("red", ROI_A)
        report = detector.analyze_frame(frame)[0]
        assert report.status is SlotStatus.PRESENT
        assert report.confidence > 0.5

    def test_absent(self, detector: ColorDetector) -> None:
        frame = make_frame(None, ROI_A)  # dark gray board
        report = detector.analyze_frame(frame)[0]
        assert report.status is SlotStatus.ABSENT

    def test_misplaced(self, detector: ColorDetector) -> None:
        # Blue key sitting in ROUTE-A's (red) slot.
        frame = make_frame("blue", ROI_A)
        report = detector.analyze_frame(frame)[0]
        assert report.status is SlotStatus.MISPLACED
        assert report.coverage["blue"] > report.coverage["red"]

    def test_unassigned_slot_empty(self, detector: ColorDetector) -> None:
        frame = make_frame(None, ROI_C)
        report = detector.analyze_frame(frame)[2]
        assert report.status is SlotStatus.EMPTY

    def test_unassigned_slot_unknown_when_occupied(self, detector: ColorDetector) -> None:
        frame = make_frame("green", ROI_C)
        report = detector.analyze_frame(frame)[2]
        assert report.status is SlotStatus.UNKNOWN

    def test_unknown_when_ambiguous(self, config_file: Path) -> None:
        detector = ColorDetector(
            config_file, settings=DetectionSettings(dominance_margin=5.0)
        )
        # Split SLOT-01's ROI into a red half and a blue half: both colors
        # clear min_coverage, but neither dominates by the 5.0x margin.
        frame = np.full((200, 200, 3), (40, 40, 40), dtype=np.uint8)
        frame[10:60, 10:35] = BGR["red"]
        frame[10:60, 35:60] = BGR["blue"]
        report = detector.analyze_frame(frame)[0]
        assert report.status is SlotStatus.UNKNOWN

    def test_partial_coverage_counts_as_present(self, detector: ColorDetector) -> None:
        # Fob occupying ~10% of the ROI: above the 2% default threshold.
        frame = make_frame("red", (10, 10, 25, 25))
        report = detector.analyze_frame(frame)[0]
        assert report.status is SlotStatus.PRESENT


# --------------------------------------------------------------------------- #
# Frame handling / robustness
# --------------------------------------------------------------------------- #

class TestFrameHandling:
    def test_none_frame_raises(self, detector: ColorDetector) -> None:
        with pytest.raises(FrameError):
            detector.analyze_frame(None)

    def test_empty_frame_raises(self, detector: ColorDetector) -> None:
        with pytest.raises(FrameError):
            detector.analyze_frame(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_non_array_raises(self, detector: ColorDetector) -> None:
        with pytest.raises(FrameError):
            detector.analyze_frame([1, 2, 3])  # type: ignore[arg-type]

    def test_roi_outside_frame_is_skipped(
        self, detector: ColorDetector, caplog: pytest.LogCaptureFixture
    ) -> None:
        tiny = np.full((40, 40, 3), 40, dtype=np.uint8)
        reports = detector.analyze_frame(tiny)
        assert len(reports) < len(detector.slots)  # out-of-frame slots skipped

    def test_slot_subset_selection(self, detector: ColorDetector) -> None:
        frame = make_frame("blue", ROI_B)
        reports = detector.analyze_frame(frame, slot_ids=["SLOT-02"])
        assert [r.slot_id for r in reports] == ["SLOT-02"]

    def test_unknown_slot_ids_are_ignored(self, detector: ColorDetector) -> None:
        frame = make_frame("blue", ROI_B)
        reports = detector.analyze_frame(frame, slot_ids=["NOPE"])
        assert reports == []

    def test_red_wraparound_detected(self, detector: ColorDetector) -> None:
        # Hue 175 (pinkish red) must still match red's second band.
        frame = np.full((200, 200, 3), (40, 40, 40), dtype=np.uint8)
        frame[10:60, 10:60] = (80, 80, 255)  # light red/pink
        report = detector.analyze_frame(frame)[0]
        assert report.status is SlotStatus.PRESENT

    def test_get_slot_helper(self, detector: ColorDetector) -> None:
        assert detector.get_slot("SLOT-01") is not None
        assert detector.get_slot("SLOT-99") is None

    def test_slot_dataclass(self) -> None:
        slot = Slot("X", None, (0, 0, 10, 10))
        assert slot.x2 == 10 and slot.route is None
