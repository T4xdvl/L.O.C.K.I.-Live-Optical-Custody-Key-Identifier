"""Integrity checks for the shipped ``config/config.json``.

Runs on every CI pass (any ``pytest`` invocation): the slot ROIs, the
configured camera frame, and the route/color wiring must stay coherent as
the board geometry evolves (e.g. switching between landscape 1920x1080 and
portrait 1080x1920 layouts). A broken config here fails loudly instead of
silently detecting nothing on the live board.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera import rois_outside_frame, validate_slot_rois  # noqa: E402
from src.color_detector import ConfigError, load_config, parse_slots  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.json"

#: The canonical board wiring: slot -> route -> key color. The demo,
#: schedule CSV, and checkout evaluation all depend on this mapping.
EXPECTED_SLOTS: tuple[tuple[str, str, str], ...] = (
    ("SLOT-01", "ROUTE-A", "red"),
    ("SLOT-02", "ROUTE-B", "blue"),
    ("SLOT-03", "ROUTE-C", "green"),
    ("SLOT-04", "ROUTE-D", "orange"),
    ("SLOT-05", "ROUTE-E", "red"),
    ("SLOT-06", "ROUTE-F", "blue"),
)

#: The portrait frame the virtual board (demo) and slot ROIs are drawn for.
EXPECTED_FRAME = (1080, 1920)  # (width, height)


@pytest.fixture(scope="module")
def config() -> dict:
    """The real shipped configuration, parsed exactly as the engine does."""
    return load_config(CONFIG_PATH)


class TestCameraFrame:
    def test_portrait_1080x1920(self, config: dict) -> None:
        cam = config["camera"]
        assert (cam["frame_width"], cam["frame_height"]) == EXPECTED_FRAME

    def test_portrait_orientation(self, config: dict) -> None:
        cam = config["camera"]
        assert cam["frame_height"] > cam["frame_width"]


class TestSlotLayout:
    """Slot ROIs must line up with the configured camera frame."""

    def test_expected_slot_wiring(self, config: dict) -> None:
        slots = parse_slots(config)
        route_colors = config.get("route_colors", {})
        actual = [
            (s.slot_id, s.route or "", route_colors.get(s.route or "", ""))
            for s in slots
        ]
        assert actual == list(EXPECTED_SLOTS)

    def test_six_slots_topleft_to_bottomright(self, config: dict) -> None:
        slots = parse_slots(config)
        assert len(slots) == 6
        # Document intent: SLOT-01 is the first physical position (top-left).
        assert slots[0].slot_id == "SLOT-01"
        assert slots[0].roi == (40, 240, 360, 560)

    def test_all_rois_fit_camera_frame(self, config: dict) -> None:
        width, height = EXPECTED_FRAME
        validate_slot_rois(config, width, height)  # raises ConfigError on failure
        assert rois_outside_frame(parse_slots(config), width, height) == []

    def test_rois_inside_frame_bounds(self, config: dict) -> None:
        width, height = EXPECTED_FRAME
        for slot in parse_slots(config):
            x1, y1, x2, y2 = slot.roi
            assert 0 <= x1 < x2 <= width
            assert 0 <= y1 < y2 <= height

    def test_rois_do_not_overlap(self, config: dict) -> None:
        """Physical slots must not share pixels (crowded board layouts)."""
        slots = parse_slots(config)
        for i, a in enumerate(slots):
            for b in slots[i + 1 :]:
                ax1, ay1, ax2, ay2 = a.roi
                bx1, by1, bx2, by2 = b.roi
                overlaps = ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2
                assert not overlaps, f"{a.slot_id} overlaps {b.slot_id}"

    def test_rois_clear_of_status_banner(self, config: dict) -> None:
        """ROIs should start below the top overlay banner (~34px tall)."""
        for slot in parse_slots(config):
            _, y1, _, _ = slot.roi
            assert y1 >= 40, f"{slot.slot_id} sits under the status banner"


class TestRoutesAndColors:
    def test_every_route_has_a_color(self, config: dict) -> None:
        known = parse_slots(config)
        route_colors = config.get("route_colors", {})
        for slot in known:
            if slot.route:
                assert slot.route in route_colors, f"{slot.slot_id}: missing {slot.route}"

    def test_colors_are_supported(self, config: dict) -> None:
        supported = {"red", "blue", "green", "orange"}
        for color in config.get("route_colors", {}).values():
            assert color in supported, f"unsupported route color {color!r}"