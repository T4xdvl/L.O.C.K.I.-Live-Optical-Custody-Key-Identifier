"""Unit tests for :mod:`src.camera`.

No physical camera is required: device-dependent paths are exercised through
configuration validation, and the evidence ring buffer is filled directly
(white-box) to test ``save_clip`` end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera import CameraConfig, CameraStream, CameraStreamError  # noqa: E402
from src.color_detector import ConfigError  # noqa: E402


def make_frame(color=(50, 50, 50), size=(48, 64)) -> np.ndarray:
    """A small solid-color BGR frame (height, width)."""
    h, w = size
    return np.full((h, w, 3), color, dtype=np.uint8)


@pytest.fixture()
def camera_config() -> dict:
    """Minimal valid camera/evidence config sections."""
    return {
        "camera": {
            "device_index": 0,
            "frame_width": 64,
            "frame_height": 48,
            "fps": 10,
            "buffer_seconds": 2,
            "reader_thread_hz": 30,
        },
        "evidence": {"output_dir": "evidence-test", "downscale_factor": 1.0},
    }


class TestCameraConfig:
    def test_from_config_defaults(self, camera_config: dict) -> None:
        cfg = CameraConfig.from_config(camera_config)
        assert cfg.fps == 10
        assert cfg.buffer_seconds == 2
        assert cfg.buffer_maxlen == 20  # 10 fps * 2 s
        assert cfg.evidence_downscale == 1.0

    def test_buffer_maxlen_floor(self, camera_config: dict) -> None:
        camera_config["camera"]["fps"] = 1
        camera_config["camera"]["buffer_seconds"] = 0.1
        assert CameraConfig.from_config(camera_config).buffer_maxlen == 1

    def test_missing_camera_section_raises(self) -> None:
        with pytest.raises(ConfigError, match="'camera' is required"):
            CameraConfig.from_config({})

    def test_bad_fps_raises(self, camera_config: dict) -> None:
        camera_config["camera"]["fps"] = 0
        with pytest.raises(ConfigError, match="fps"):
            CameraConfig.from_config(camera_config)

    def test_bad_buffer_seconds_raises(self, camera_config: dict) -> None:
        camera_config["camera"]["buffer_seconds"] = -1
        with pytest.raises(ConfigError, match="buffer_seconds"):
            CameraConfig.from_config(camera_config)

    def test_bad_downscale_raises(self, camera_config: dict) -> None:
        camera_config["evidence"]["downscale_factor"] = 1.5
        with pytest.raises(ConfigError, match="downscale_factor"):
            CameraConfig.from_config(camera_config)

    def test_loads_from_project_config(self) -> None:
        from src.color_detector import load_config

        cfg = CameraConfig.from_config(load_config("config/config.json"))
        assert cfg.buffer_seconds == 5
        assert cfg.buffer_maxlen == 150  # 30 fps * 5 s
        assert cfg.evidence_dir == Path("evidence")


class TestCameraStreamBuffer:
    def test_read_before_start_returns_none(self, camera_config: dict) -> None:
        cam = CameraStream(config=camera_config)
        assert cam.read() is None
        assert not cam.is_running

    def test_save_clip_without_frames_raises(self, camera_config: dict) -> None:
        cam = CameraStream(config=camera_config)
        with pytest.raises(CameraStreamError, match="No frames"):
            cam.save_clip("evidence-test/empty.mp4")

    def test_save_clip_writes_mp4(self, camera_config: dict, tmp_path: Path) -> None:
        cam = CameraStream(config=camera_config)
        interval = 1.0 / camera_config["camera"]["fps"]
        monotonic = 1000.0
        for i in range(20):
            frame = make_frame((i * 10 % 255, 50, 50))
            from src.camera import _BufferedFrame

            cam._ring.append(_BufferedFrame(timestamp=monotonic, frame=frame))
            monotonic += interval

        out = cam.save_clip(tmp_path / "incident")  # no suffix -> .mp4 appended
        assert out.name == "incident.mp4"
        assert out.exists() and out.stat().st_size > 0

        probe = cv2.VideoCapture(str(out))
        assert probe.isOpened()
        count = 0
        while True:
            ok, _ = probe.read()
            if not ok:
                break
            count += 1
        probe.release()
        assert count == 20

    def test_save_clip_partial_window(self, camera_config: dict, tmp_path: Path) -> None:
        from src.camera import _BufferedFrame

        cam = CameraStream(config=camera_config)
        interval = 1.0 / camera_config["camera"]["fps"]
        for i in range(20):
            cam._ring.append(
                _BufferedFrame(timestamp=1000.0 + i * interval, frame=make_frame())
            )
        out = cam.save_clip(tmp_path / "short.mp4", seconds=1.0)
        probe = cv2.VideoCapture(str(out))
        count = 0
        while probe.read()[0]:
            count += 1
        probe.release()
        assert count == 10  # 1 s at 10 fps

    def test_save_clip_invalid_seconds(self, camera_config: dict, tmp_path: Path) -> None:
        from src.camera import _BufferedFrame

        cam = CameraStream(config=camera_config)
        cam._ring.append(_BufferedFrame(timestamp=1.0, frame=make_frame()))
        with pytest.raises(CameraStreamError, match="positive"):
            cam.save_clip(tmp_path / "x.mp4", seconds=0)

    def test_stats_shape(self, camera_config: dict) -> None:
        cam = CameraStream(config=camera_config, device_override=2)
        stats = cam.stats
        assert stats["device_index"] == 2
        assert stats["frames_captured"] == 0
        assert stats["buffered_frames"] == 0

    def test_stop_is_idempotent(self, camera_config: dict) -> None:
        cam = CameraStream(config=camera_config)
        cam.stop()
        cam.stop()  # must not raise
        assert not cam.is_running
