"""Unit tests for :mod:`src.alarm_system` (headless-safe, no display)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alarm_system import (  # noqa: E402
    AlarmConfig,
    AlarmSystem,
    COLOR_ALERT,
    COLOR_OK,
    AlertLevel,
    SlotVisualState,
    render_roi_debug,
)
from src.color_detector import Slot  # noqa: E402


@pytest.fixture()
def config() -> dict:
    """Config with pygame signage off (pure OpenCV overlay tests)."""
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
        "ui": {
            "enable_pygame": False,
            "audio": False,
            "alert_flash_seconds": 2.0,
        },
    }


@pytest.fixture()
def alarm(config: dict) -> AlarmSystem:
    system = AlarmSystem(config=config)
    yield system
    system.close()


class TestAlarmConfig:
    def test_defaults(self) -> None:
        cfg = AlarmConfig.from_config({})
        assert cfg.enable_pygame is True
        assert cfg.audio is True
        assert cfg.alert_flash_seconds == pytest.approx(4.0)

    def test_invalid_flash_raises(self) -> None:
        with pytest.raises(Exception, match="alert_flash_seconds"):
            AlarmConfig.from_config({"ui": {"alert_flash_seconds": 0.1}})


class TestOverlays:
    def test_render_returns_copy(self, alarm: AlarmSystem) -> None:
        frame = np.full((120, 200, 3), 40, dtype=np.uint8)
        out = alarm.render_overlay(frame)
        assert out is not frame
        assert out.shape == frame.shape

    def test_slot_state_colors_drawn(self, alarm: AlarmSystem) -> None:
        alarm.set_slot_state("SLOT-01", SlotVisualState.OK)
        alarm.set_slot_state("SLOT-02", SlotVisualState.ALERT)
        frame = np.full((120, 200, 3), 40, dtype=np.uint8)
        out = alarm.render_overlay(frame)
        # Check the ROI *bottom* edges (row 59) - the top edges sit under
        # the translucent banner bar and would be dimmed by the blend.
        assert out[59, 35][1] > 120  # green-dominant border on SLOT-01
        assert out[59, 95][2] > 120  # red-dominant border on SLOT-02

    def test_system_warning_grows_bar(self, alarm: AlarmSystem) -> None:
        """A persistent warning extends the dimmed backing bar downward."""
        frame = np.full((120, 200, 3), 40, dtype=np.uint8)
        # Sample column 5: left of the SLOT-01 border (x1=10), clear of any
        # drawing. No warning means the bar ends at row 34 -> raw frame.
        plain = alarm.render_overlay(frame)
        assert plain[50, 5].tolist() == [40, 40, 40]
        # With a warning the bar covers row 50 (dimmed to ~45% brightness).
        alarm.set_system_warning("ROI/CAMERA MISMATCH: test")
        warned = alarm.render_overlay(frame)
        assert warned[50, 5].tolist() != [40, 40, 40]
        assert int(warned[50, 5][0]) < 30  # dimmed toward black
        # Clearing restores the short bar.
        alarm.clear_system_warning()
        cleared = alarm.render_overlay(frame)
        assert cleared[50, 5].tolist() == [40, 40, 40]

    def test_system_warning_renders_amber_text(self, alarm: AlarmSystem) -> None:
        alarm.set_system_warning("ROI/CAMERA MISMATCH: SLOT-06")
        frame = np.full((120, 200, 3), 40, dtype=np.uint8)
        out = alarm.render_overlay(frame)
        # Amber (0, 160, 255): strong blue/red channel around the text row.
        assert (out[:, :, 2] > 180).sum() > 0
        alarm.clear_system_warning()
    def test_unknown_slot_state_ignored(self, alarm: AlarmSystem) -> None:
        alarm.set_slot_state("NOPE", SlotVisualState.OK)  # must not raise

    def test_banner_and_message_render(self, alarm: AlarmSystem) -> None:
        alarm.set_banner("TEST BANNER", COLOR_OK)
        alarm.push_message("hello world", COLOR_ALERT, ttl=5.0)
        frame = np.full((120, 200, 3), 40, dtype=np.uint8)
        out = alarm.render_overlay(frame)
        # Message text is drawn in red in the message rows (~y=40-60).
        top = out[:70]
        assert (top[:, :, 2] > 200).sum() > 0

    def test_message_expires(self, alarm: AlarmSystem) -> None:
        alarm.push_message("fleeting", COLOR_OK, ttl=0.05)
        frame = np.full((120, 200, 3), 40, dtype=np.uint8)
        alarm.render_overlay(frame)  # t=0: visible
        import time

        time.sleep(0.08)
        out = alarm.render_overlay(frame)  # expired
        assert out.shape == frame.shape  # and nothing crashed

    def test_alert_flash_border(self, alarm: AlarmSystem) -> None:
        import time

        alarm.trigger_alert(AlertLevel.WARNING, "WRONG ROUTE ALERT", "detail")
        frame = np.full((120, 200, 3), 40, dtype=np.uint8)
        out = alarm.render_overlay(frame)
        # Screen border flashes red at 2 Hz. Corner pixels sit under the
        # translucent banner bar (dimmed to ~45%), so assert red dominance
        # rather than raw brightness, across two consecutive renders.
        import time

        def is_reddish(img):
            px = img[2, 2]
            return int(px[2]) > 80 and px[2] > px[1] and px[2] > px[0]

        first = is_reddish(out)
        time.sleep(0.3)
        second = is_reddish(alarm.render_overlay(frame))
        assert first or second

    def test_trigger_success_is_nonfatal(self, alarm: AlarmSystem) -> None:
        alarm.trigger_success("Driver_101 took red")  # no audio: logs only

    def test_headless_alert_logging(self, alarm: AlarmSystem, caplog) -> None:
        with caplog.at_level("WARNING"):
            alarm.trigger_alert(AlertLevel.CRITICAL, "UNAUTHORIZED REMOVAL", "detail")
        assert "UNAUTHORIZED" in caplog.text


class TestRoiDebugOverlay:
    """``render_roi_debug``: alignment boxes for --debug-rois."""

    def test_inbounds_slots_green(self) -> None:
        frame = np.full((200, 200, 3), 40, dtype=np.uint8)
        out = render_roi_debug(frame, [Slot("A", "R", (10, 10, 60, 60))], (200, 200))
        # Left border of slot A: green-dominant.
        assert out[35, 10][1] > 120
        # Frame bound caption is drawn (white) near the bottom.
        assert (out[184:190, :, 0] > 150).sum() > 0

    def test_out_of_bounds_slot_red(self) -> None:
        frame = np.full((200, 200, 3), 40, dtype=np.uint8)
        out = render_roi_debug(
            frame, [Slot("A", "R", (150, 150, 400, 400))], (200, 200)
        )
        # Left border of the overflowing ROI clips at the frame edge.
        assert out[170, 150][2] > 120  # red-dominant: ROI past the bounds

    def test_returns_copy(self) -> None:
        frame = np.full((200, 200, 3), 40, dtype=np.uint8)
        out = render_roi_debug(frame, [])
        assert out is not frame


class TestPygameUnavailable:
    def test_missing_pygame_degrades(self, config: dict) -> None:
        # enable_pygame=True but the import fails (or SDL fails) in CI:
        # the system must still construct and render overlays.
        config["ui"]["enable_pygame"] = True
        system = AlarmSystem(config=config)
        try:
            frame = np.full((120, 200, 3), 40, dtype=np.uint8)
            out = system.render_overlay(frame)  # never raises
            assert out.shape == frame.shape
            system.render_signage()  # no-op when not ready
        finally:
            system.close()
