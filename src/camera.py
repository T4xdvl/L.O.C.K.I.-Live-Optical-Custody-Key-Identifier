"""
camera.py
=========

Threaded video ingestion for the L.O.C.K.I. system
(Live Optical Custody & Key Identifier).

Reading frames from ``cv2.VideoCapture`` synchronously bottlenecks the whole
detection pipeline: every HSV analysis, MediaPipe inference, and JPEG encode
blocks the next ``read()`` call, so the driver's internal buffer silently
fills with stale frames and "live" processing lags seconds behind reality.
This module decouples capture from processing:

* A dedicated daemon thread drains the camera as fast as the driver allows
  (default 60 Hz), keeping only the freshest frame for consumers.
* A rolling circular buffer (:class:`collections.deque`, thread-safe under
  the GIL for append/popleft) retains the last ``buffer_seconds`` of raw
  frames so an incident (key removed, key misplaced) can be exported as
  evidence **after** it happened.

Typical usage::

    from src.camera import CameraStream

    with CameraStream("config/config.json") as cam:
        while True:
            frame = cam.read()          # always the newest frame, or None
            if frame is None:
                continue                # camera hiccup, not a stale frame
            ...                         # run detection / hand tracking
        cam.save_clip("evidence/incident.mp4")   # last 5 s, on demand
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.color_detector import ConfigError, load_config

LOGGER = logging.getLogger(__name__)

__all__ = ["CameraConfig", "CameraStream", "CameraStreamError"]


class CameraStreamError(RuntimeError):
    """Raised when the camera cannot be opened or stream threads fail."""


@dataclass(frozen=True)
class CameraConfig:
    """Frozen snapshot of the ``camera`` and ``evidence`` config sections.

    The buffer length is derived from ``buffer_seconds`` and ``fps`` so the
    evidence window tracks the configured frame rate automatically.
    """

    device_index: int
    frame_width: int
    frame_height: int
    fps: int
    autofocus_disabled: bool
    reconnect_delay_sec: float
    max_reconnect_attempts: int
    buffer_seconds: float
    reader_thread_hz: float
    evidence_dir: Path
    evidence_downscale: float

    @property
    def buffer_maxlen(self) -> int:
        """Frames retained in the evidence buffer (fps x seconds, min 1)."""
        return max(1, int(round(self.fps * self.buffer_seconds)))

    @classmethod
    def from_config(cls, config: dict) -> "CameraConfig":
        """Build a validated :class:`CameraConfig` from a parsed config dict.

        Raises:
            ConfigError: If the camera section is missing or invalid.
        """
        raw = config.get("camera")
        if not isinstance(raw, dict):
            raise ConfigError("Configuration section 'camera' is required.")

        def _get(key: str, default, cast=float):
            """Fetch a config key with a default and a numeric cast."""
            try:
                return cast(raw.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"camera.{key}: invalid value {raw.get(key)!r}") from exc

        device_index = _get("device_index", 0, int)
        frame_width = _get("frame_width", 1280, int)
        frame_height = _get("frame_height", 720, int)
        fps = _get("fps", 30, int)
        if fps <= 0:
            raise ConfigError("camera.fps must be a positive integer.")
        autofocus_disabled = bool(raw.get("autofocus_disabled", True))
        reconnect_delay_sec = _get("reconnect_delay_sec", 2.0)
        max_reconnect_attempts = _get("max_reconnect_attempts", 5, int)
        buffer_seconds = _get("buffer_seconds", 5.0)
        if buffer_seconds <= 0:
            raise ConfigError("camera.buffer_seconds must be positive.")
        reader_thread_hz = _get("reader_thread_hz", 60.0)
        if reader_thread_hz <= 0:
            raise ConfigError("camera.reader_thread_hz must be positive.")

        evidence = config.get("evidence", {}) or {}
        if not isinstance(evidence, dict):
            raise ConfigError("Configuration section 'evidence' must be an object.")
        evidence_dir = Path(str(evidence.get("output_dir", "evidence")))
        try:
            evidence_downscale = float(evidence.get("downscale_factor", 0.5))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"evidence.downscale_factor: invalid value {evidence.get('downscale_factor')!r}"
            ) from exc
        if not (0.0 < evidence_downscale <= 1.0):
            raise ConfigError("evidence.downscale_factor must be in (0, 1].")

        return cls(
            device_index=device_index,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            autofocus_disabled=autofocus_disabled,
            reconnect_delay_sec=reconnect_delay_sec,
            max_reconnect_attempts=max_reconnect_attempts,
            buffer_seconds=buffer_seconds,
            reader_thread_hz=reader_thread_hz,
            evidence_dir=evidence_dir,
            evidence_downscale=evidence_downscale,
        )


@dataclass(frozen=True)
class _BufferedFrame:
    """A timestamped frame held in the evidence buffer."""

    timestamp: float  # time.monotonic() at capture
    frame: np.ndarray


class CameraStream:
    """Multi-threaded camera reader with a rolling evidence buffer.

    Architecture::

        [camera] --driver buffer--> [reader thread] --copy--> [latest frame]
                                                      \\-----> [evidence deque]

    Args:
        config_path: Path to the L.O.C.K.I. JSON configuration file.
        config: Optionally pass a pre-parsed config dict instead of a path.
            Exactly one of ``config_path``/``config`` should be given; when
            both are ``None`` the default ``config/config.json`` is used.
        device_override: Override the configured camera index (e.g. for CLI
            ``--camera`` flags). ``None`` uses the config value.

    Raises:
        ConfigError: If the configuration is invalid.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        config: Optional[dict] = None,
        device_override: Optional[int | str] = None,
    ) -> None:
        if config is None:
            config = load_config(Path(config_path) if config_path else "config/config.json")
        self._cfg = CameraConfig.from_config(config)

        # A string device is a media file path (video/image sequence) rather
        # than a physical camera index; used by demos and replay tooling.
        self._device_index: int | str = (
            self._cfg.device_index if device_override is None else device_override
        )
        self._is_file_source = isinstance(self._device_index, str)
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Freshest frame for consumers; never a stale driver-buffer frame.
        self._latest: Optional[np.ndarray] = None
        self._latest_seq: int = 0  # incremented per captured frame
        self._last_seq_seen: int = 0  # last seq handed out by read()

        # Rolling evidence window: deque(maxlen=N) auto-drops the oldest
        # frame on append, so this is a true circular buffer.
        self._ring: Deque[_BufferedFrame] = deque(maxlen=self._cfg.buffer_maxlen)

        self._frames_captured: int = 0
        self._frames_dropped: int = 0
        self._last_capture_time: float = 0.0
        self._reconnects: int = 0
        self._started = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> "CameraStream":
        """Open the camera and launch the reader thread.

        Returns:
            ``self`` so callers can chain (``cam = CameraStream(...).start()``).

        Raises:
            CameraStreamError: If the device cannot be opened after retries.
        """
        if self._started:
            return self
        self._open_device()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"locki-camera-{self._device_index}",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        LOGGER.info(
            "CameraStream started: device=%d %dx%d@%dfps, buffer=%d frames "
            "(%.1fs), thread=%dHz",
            self._device_index,
            self._cfg.frame_width,
            self._cfg.frame_height,
            self._cfg.fps,
            self._cfg.buffer_maxlen,
            self._cfg.buffer_seconds,
            self._cfg.reader_thread_hz,
        )
        return self

    def stop(self) -> None:
        """Signal the reader thread to stop, then release the device.

        Idempotent: safe to call multiple times or on a never-started stream.
        """
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self._cfg.reconnect_delay_sec + 5.0)
            if self._thread.is_alive():
                LOGGER.warning("Camera reader thread did not stop cleanly.")
            self._thread = None
        self._release_device()
        self._started = False
        LOGGER.info(
            "CameraStream stopped: captured=%d dropped=%d reconnects=%d",
            self._frames_captured,
            self._frames_dropped,
            self._reconnects,
        )

    # Context-manager sugar for `with CameraStream(...) as cam:` blocks.
    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Device handling
    # ------------------------------------------------------------------ #

    def _open_device(self) -> None:
        """Open the capture device, retrying per the configured backoff.

        Media-file sources open deterministically, so they skip the retry
        loop entirely.

        Raises:
            CameraStreamError: If the device never opens.
        """
        attempts = 1 if self._is_file_source else self._cfg.max_reconnect_attempts
        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            capture = self._try_open()
            if capture is not None:
                self._capture = capture
                self._configure_device()
                return
            last_error = f"attempt {attempt}/{self._cfg.max_reconnect_attempts}"
            LOGGER.warning(
                "Could not open camera %s (%s); retrying in %.1fs",
                self._device_index,
                last_error,
                self._cfg.reconnect_delay_sec,
            )
            time.sleep(self._cfg.reconnect_delay_sec)
        raise CameraStreamError(
            f"Failed to open camera {self._device_index} after "
            f"{self._cfg.max_reconnect_attempts} attempts ({last_error})."
        )

    def _try_open(self) -> Optional[cv2.VideoCapture]:
        """Single open attempt; V4L2-first on Linux, default backend elsewhere."""
        capture: Optional[cv2.VideoCapture] = None
        try:
            if self._is_file_source:
                capture = cv2.VideoCapture(str(self._device_index))
            else:
                if sys.platform.startswith("linux"):
                    capture = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)
                if capture is None or not capture.isOpened():
                    capture = cv2.VideoCapture(self._device_index)
        except cv2.error as exc:  # pragma: no cover - backend specific
            LOGGER.warning("Camera backend error: %s", exc)
            return None
        return capture if capture.isOpened() else None

    def _configure_device(self) -> None:
        """Apply resolution/FPS/autofocus settings to the open device."""
        assert self._capture is not None
        if self._is_file_source:
            # A file plays back at whatever FPS it was encoded with.
            actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            LOGGER.info(
                "Opened media file %s at %dx%d", self._device_index, actual_w, actual_h
            )
            return
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
        self._capture.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        if self._cfg.autofocus_disabled:
            # Fixed focus avoids focus-hunt wobble in the evidence footage.
            self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (self._cfg.frame_width, self._cfg.frame_height):
            LOGGER.warning(
                "Camera delivered %dx%d instead of the requested %dx%d; ROIs "
                "in config.json must match the ACTUAL resolution.",
                actual_w,
                actual_h,
                self._cfg.frame_width,
                self._cfg.frame_height,
            )

    def _release_device(self) -> None:
        """Release the capture device, swallowing backend quirks."""
        if self._capture is not None:
            try:
                self._capture.release()
            except cv2.error as exc:  # pragma: no cover - backend specific
                LOGGER.warning("Error releasing camera: %s", exc)
            self._capture = None

    def _attempt_reconnect(self) -> bool:
        """Try to re-open a camera that stopped delivering frames.

        Returns:
            True if the device was re-opened, False if retries are exhausted
            (the reader thread then exits; ``is_running()`` reflects this).
        """
        self._release_device()
        self._reconnects += 1
        LOGGER.warning(
            "Reconnecting camera %d (reconnect #%d)...",
            self._device_index,
            self._reconnects,
        )
        time.sleep(self._cfg.reconnect_delay_sec)
        try:
            self._open_device()
        except CameraStreamError as exc:
            LOGGER.error("Reconnect failed: %s", exc)
            return False
        LOGGER.info("Camera %d reconnected.", self._device_index)
        return True

    # ------------------------------------------------------------------ #
    # Reader thread
    # ------------------------------------------------------------------ #

    def _reader_loop(self) -> None:
        """Continuously drain the camera into the latest frame + evidence ring.

        Runs until :meth:`stop` sets the stop event, or a permanent camera
        failure exhausts reconnection attempts. All state handoff happens
        under ``self._lock``.
        """
        interval = 1.0 / self._cfg.reader_thread_hz
        consecutive_failures = 0
        while not self._stop_event.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                if self._is_file_source:
                    LOGGER.info("Media file fully read; stream complete.")
                    break
                consecutive_failures += 1
                with self._lock:
                    self._frames_dropped += 1
                if consecutive_failures >= 10:
                    if not self._attempt_reconnect():
                        break
                    consecutive_failures = 0
                else:
                    time.sleep(0.05)
                continue

            consecutive_failures = 0
            now = time.monotonic()
            with self._lock:
                self._latest = frame
                self._latest_seq += 1
                self._last_capture_time = now
                self._frames_captured += 1
                self._ring.append(_BufferedFrame(timestamp=now, frame=frame))
            # Pace the drain; the driver buffers anything faster than this.
            time.sleep(interval)
        LOGGER.debug("Reader thread exiting.")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def read(self) -> Optional[np.ndarray]:
        """Return the newest captured frame (shared numpy buffer).

        The returned array is the live buffer the reader thread overwrites
        on the next capture, so consumers that hold a frame across multiple
        reader cycles should clone it via :meth:`read_copy`. Frames are
        never replayed: once consumed, subsequent calls yield only *newer*
        frames, which is what keeps processing latency bounded.

        Returns:
            The latest BGR frame, or ``None`` if no new frame is available.
        """
        with self._lock:
            if self._latest is None or self._last_seq_seen == self._latest_seq:
                return None
            self._last_seq_seen = self._latest_seq
            return self._latest

    def read_copy(self) -> Optional[np.ndarray]:
        """Return a private clone of the newest frame (or ``None``).

        Convenience wrapper for consumers that stash frames for async
        processing.
        """
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def submit_frame(self, frame: np.ndarray) -> None:
        """Inject an externally produced frame (demo/replay mode).

        Feeds the same latest-frame + evidence-ring plumbing as the reader
        thread, so ``read()``/``save_clip()`` work identically for synthetic
        or recorded sources without starting the capture thread.

        Raises:
            CameraStreamError: If ``frame`` is not a non-empty numpy image.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise CameraStreamError("submit_frame expects a non-empty numpy image.")
        now = time.monotonic()
        with self._lock:
            self._latest = frame
            self._latest_seq += 1
            self._last_capture_time = now
            self._frames_captured += 1
            self._ring.append(_BufferedFrame(timestamp=now, frame=frame))

    def save_clip(self, filename: str | Path, seconds: Optional[float] = None) -> Path:
        """Write the buffered window to an MP4 evidence file.

        The clip is encoded from the evidence ring buffer, so it contains
        the past (``buffer_seconds`` by default) — useful when an incident
        is detected *after* it started. Each buffered frame is cloned before
        encoding, so a concurrent capture cannot corrupt the output.

        Args:
            filename: Destination path. Parent directories are created.
                A ``.mp4`` suffix is appended if missing.
            seconds: Length of footage to export. ``None`` exports the full
                buffer; values beyond what's buffered export everything.

        Returns:
            The resolved path of the written clip.

        Raises:
            CameraStreamError: If the codec/writer cannot be initialized or
                no frames are available to write.
        """
        with self._lock:
            frames: List[np.ndarray] = [entry.frame.copy() for entry in self._ring]
            frame_interval = (
                (self._ring[-1].timestamp - self._ring[0].timestamp)
                / max(1, len(self._ring) - 1)
                if len(self._ring) > 1
                else (1.0 / self._cfg.fps)
            )

        if seconds is not None:
            if seconds <= 0:
                raise CameraStreamError("seconds must be positive.")
            keep = max(1, int(round(seconds / max(frame_interval, 1e-6))))
            frames = frames[-keep:]

        if not frames:
            raise CameraStreamError(
                "No frames in the evidence buffer; has the camera produced "
                "any frames yet?"
            )

        out_path = Path(filename)
        if out_path.suffix.lower() != ".mp4":
            out_path = out_path.with_suffix(".mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = frames[0].shape[:2]
        if self._cfg.evidence_downscale != 1.0:
            width = max(2, int(width * self._cfg.evidence_downscale))
            height = max(2, int(height * self._cfg.evidence_downscale))

        out_fps = 1.0 / max(frame_interval, 1e-6)
        # Four-character codec id for the MP4 container ('mp4v' = MPEG-4 Part 2).
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer: Optional[cv2.VideoWriter] = None
        try:
            writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))
            if not writer.isOpened():
                raise CameraStreamError(
                    f"Could not open MP4 writer for {out_path} "
                    "(codec 'mp4v' unavailable in this OpenCV build?)."
                )
            for frame in frames:
                if self._cfg.evidence_downscale != 1.0:
                    frame = cv2.resize(
                        frame, (width, height), interpolation=cv2.INTER_AREA
                    )
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()

        duration = len(frames) * frame_interval
        LOGGER.info(
            "Evidence clip saved: %s (%d frames, %.2fs, %dx%d)",
            out_path,
            len(frames),
            duration,
            width,
            height,
        )
        return out_path

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> CameraConfig:
        """The frozen camera/evidence configuration in use."""
        return self._cfg

    @property
    def is_running(self) -> bool:
        """True while the reader thread is alive and the device is open."""
        return (
            self._started
            and self._thread is not None
            and self._thread.is_alive()
            and self._capture is not None
        )

    @property
    def stats(self) -> dict:
        """Counters for health monitoring (captured/dropped/reconnects)."""
        with self._lock:
            return {
                "device_index": self._device_index,
                "frames_captured": self._frames_captured,
                "frames_dropped": self._frames_dropped,
                "reconnects": self._reconnects,
                "buffered_frames": len(self._ring),
                "buffer_seconds": (
                    self._ring[-1].timestamp - self._ring[0].timestamp
                    if len(self._ring) > 1
                    else 0.0
                ),
                "latest_seq": self._latest_seq,
                "last_capture_age_sec": (
                    time.monotonic() - self._last_capture_time
                    if self._last_capture_time > 0
                    else float("inf")
                ),
            }

    def frames_in_buffer(self) -> int:
        """Number of frames currently retained for evidence."""
        with self._lock:
            return len(self._ring)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        state = "running" if self.is_running else "stopped"
        return (
            f"CameraStream(device={self._device_index}, {state}, "
            f"buffer={self.frames_in_buffer()}/{self._cfg.buffer_maxlen})"
        )
