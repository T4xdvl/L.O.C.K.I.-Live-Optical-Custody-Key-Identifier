#!/usr/bin/env python3
"""
main.py
=======

L.O.C.K.I. core engine (Live Optical Custody & Key Identifier).

Unifies every subsystem into one real-time custody loop::

    CameraStream ──frames──► ColorDetector ──slot states──► AlarmSystem overlay
         │                        ▲                              ▲
         │                        │                              │
         ├──► FaceRecognizer ──driver──► ScheduleManager ──route──┤
         │                        ▲                              │
         └──► HandTracker ──grasp─┘                    DatabaseManager
                                                       (checkouts + clips)

Per-frame pipeline
------------------
1. Grab the freshest frame from :class:`src.camera.CameraStream`.
2. Run :class:`src.color_detector.ColorDetector` over every slot ROI and
   paint the slot states (green/gray/red/amber).
3. Identify the driver at the station with :class:`src.face_recognizer.FaceRecognizer`
   (cooldown-throttled) and fetch their route from
   :class:`src.schedule_manager.ScheduleManager`.
4. When :class:`src.hand_tracker.HandTracker` sees a hand grasp a slot
   (confirmed for a short window to avoid fly-bys), evaluate the checkout
   *while the key is still on the board*: the key's color decides
   ``SUCCESS`` vs ``WRONG_ROUTE_ALERT`` vs ``UNAUTHORIZED_REMOVAL``,
   a 3-second evidence MP4 is exported from the camera's rolling buffer,
   and the event is written to the ``checkouts`` table.
5. Render the OpenCV overlay and, when available, the pygame wall signage
   (screen border + chime/alarm audio).

Run it::

    python main.py                      # live camera from config.json
    python main.py --camera 1           # second camera
    python main.py --no-pygame          # OpenCV overlay only (headless-ish)
    python main.py --demo               # synthetic board, no camera needed

Press ``q`` (or ``ESC``) in the video window to exit.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from src.alarm_system import AlertLevel, AlarmSystem, SlotVisualState
from src.camera import CameraStream, CameraStreamError
from src.color_detector import (
    ColorDetector,
    ConfigError,
    SlotReport,
    SlotStatus,
    load_config,
)
from src.database import DatabaseManager, EventStatus
from src.face_recognizer import FaceRecognizer, FaceRecognizerError
from src.hand_tracker import HandTracker, HandTrackerError
from src.schedule_manager import ScheduleError, ScheduleManager

LOGGER = logging.getLogger("locki.main")

__all__ = ["CheckoutTracker", "LockiEngine", "check_window_aspect", "main"]

#: Length of the evidence clip exported when a checkout fires.
EVIDENCE_CLIP_SECONDS = 3.0
#: How long a hand must stay in a slot before a grasp counts (anti fly-by).
GRASP_CONFIRM_SECONDS = 0.4
#: Consecutive hand-tracker failures before the engine disables that stage.
MAX_TRACKER_FAILURES = 3
#: Seconds between window/stream aspect-divergence checks (throttled).
ASPECT_CHECK_INTERVAL_SEC = 5.0
#: Allowed relative divergence before the letterbox warning fires (15%).
ASPECT_TOLERANCE = 0.15


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 (used for clip file naming)."""
    return datetime.now(timezone.utc).isoformat()


def _display_available() -> bool:
    """Best-effort check for an interactive display on this host.

    Calling HighGUI (``namedWindow``/``imshow``) without a display can hard-
    abort the process on some OpenCV builds, so the engine skips all window
    calls when none is present (overlays are still rendered and recorded).
    """
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


#: Cached once: display presence does not change mid-run.
_HAS_DISPLAY = _display_available()


def check_window_aspect(window_name: str, frame: np.ndarray) -> Optional[float]:
    """Compare the OpenCV window's image rect to the frame's aspect ratio.

    HighGUI letterboxes a frame whose aspect ratio does not match the
    window's image area — visible as white bars on the sides. This probe
    quantifies the mismatch so the engine can warn the operator.

    Returns:
        ``window_aspect / frame_aspect`` (1.0 = perfect match), or ``None``
        when it cannot be measured (headless host, window not yet created,
        window closed by the OS, or degenerate geometry).
    """
    if not _HAS_DISPLAY or not window_name:
        return None
    try:
        rect = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return None
    if not rect or rect[2] <= 0 or rect[3] <= 0:
        return None
    frame_h, frame_w = frame.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return None
    return (frame_w / frame_h) / (rect[2] / rect[3])


class CheckoutTracker:
    """Rising-edge hand detector with a confirmation window and one-shot latch.

    A grasp event fires only when a hand has been continuously present in a
    slot for :attr:`confirm_seconds`. After firing, the slot is latched for
    :attr:`removal_timeout_seconds` so one physical grab cannot log a burst
    of duplicate events while the key leaves the board.
    """

    def __init__(
        self,
        confirm_seconds: float = GRASP_CONFIRM_SECONDS,
        removal_timeout_seconds: float = 3.0,
    ) -> None:
        self.confirm_seconds = confirm_seconds
        self.removal_timeout_seconds = removal_timeout_seconds
        self._armed_since: Dict[str, float] = {}
        self._fired_until: Dict[str, float] = {}

    def register_presence(self, slot_id: str, active: bool, now: float) -> bool:
        """Feed one observation; returns True exactly once per confirmed grasp."""
        if active:
            since = self._armed_since.setdefault(slot_id, now)
            if now - since >= self.confirm_seconds and not self.is_latched(slot_id, now):
                self._fired_until[slot_id] = now + self.removal_timeout_seconds
                self._armed_since.pop(slot_id, None)
                return True
            return False
        self._armed_since.pop(slot_id, None)  # hand left: cancel confirmation
        return False

    def is_latched(self, slot_id: str, now: float) -> bool:
        """True while a fired grasp for this slot is still being processed."""
        return now <= self._fired_until.get(slot_id, 0.0)

    def reset(self) -> None:
        """Clear all per-slot tracking state."""
        self._armed_since.clear()
        self._fired_until.clear()


class LockiEngine:
    """Orchestrates camera, recognition, detection, alarms, and persistence.

    Args:
        config_path: Path to the L.O.C.K.I. JSON configuration file.
        config: Pre-parsed config dict (alternative to ``config_path``).
        camera_source: Override the camera: an int device index or a str
            path to a media file (replay mode). ``None`` uses the config.

    Raises:
        ConfigError: If the configuration is invalid.
        ScheduleError: If the required schedule CSV is missing/malformed.
        FaceRecognizerError: If recognition is enabled but unavailable.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
        camera_source: Optional[int | str] = None,
    ) -> None:
        if config is None:
            config = load_config(Path(config_path) if config_path else "config/config.json")
        self.config = config

        self.detector = ColorDetector(config=config)
        self.schedule = ScheduleManager(config=config)
        self.db = DatabaseManager(config=config)
        try:
            self.recognizer = FaceRecognizer(config=config)
        except FaceRecognizerError as exc:
            # Missing insightface: keep running; every identity event will
            # use the fallback driver id (conservative custody stance).
            LOGGER.warning("Face recognition unavailable (%s).", exc)
            degraded = copy.deepcopy(config)
            degraded.setdefault("face_recognition", {})["enabled"] = False
            self.recognizer = FaceRecognizer(config=degraded)

        hand_cfg = config.get("hand_tracking", {}) or {}
        self.hand_tracker: Optional[HandTracker] = None
        if hand_cfg.get("enabled", True):
            try:
                self.hand_tracker = HandTracker(config=config)
            except HandTrackerError as exc:
                # Missing mediapipe / model / display: grasp triggers are
                # disabled but the rest of the pipeline keeps running.
                LOGGER.warning("Hand tracking unavailable (%s); grasp events disabled.", exc)
        self._tracker_failures = 0

        self.camera = CameraStream(config=config, device_override=camera_source)
        self.alarm = AlarmSystem(config=config)

        engine_cfg = config.get("engine", {}) or {}
        self.identify_cooldown = max(0.5, float(engine_cfg.get("identify_cooldown_sec", 2.0)))
        self.removal_timeout = max(0.5, float(engine_cfg.get("removal_timeout_sec", 3.0)))
        self.tracker = CheckoutTracker(
            confirm_seconds=GRASP_CONFIRM_SECONDS,
            removal_timeout_seconds=self.removal_timeout,
        )

        ui_cfg = config.get("ui", {}) or {}
        self.window_name = str(ui_cfg.get("window_name", "L.O.C.K.I. Key Custody Monitor"))

        self.current_driver: Optional[str] = None
        self._identity_until = 0.0
        self._aspect_check_due = 0.0  # throttled window/stream aspect probe
        self._running = False

        fallback = self.recognizer.config.fallback_driver_id
        LOGGER.info(
            "Engine ready: %d slot(s), %d scheduled driver(s), hand_tracking=%s",
            len(self.detector.slots),
            len(self.schedule),
            self.hand_tracker is not None,
        )
        self._fallback_driver_id = fallback

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self, with_camera: bool = True) -> "LockiEngine":
        """Start the pipeline and show the video window.

        Args:
            with_camera: Open the physical/media camera. Demo mode passes
                ``False`` and feeds synthetic frames via
                :meth:`CameraStream.submit_frame` instead.
        """
        if with_camera:
            self.camera.start()
        self._window_ok = False
        if _HAS_DISPLAY:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                # Size the window to the stream's aspect ratio; otherwise
                # HighGUI letterboxes the frame with white bars.
                cam_cfg = self.config.get("camera", {}) or {}
                frame_w = max(1, int(cam_cfg.get("frame_width", 1080)))
                frame_h = max(1, int(cam_cfg.get("frame_height", 1920)))
                display_h = 960
                display_w = max(2, int(round(frame_w * display_h / frame_h)))
                cv2.resizeWindow(self.window_name, display_w, display_h)
                self._window_ok = True
            except cv2.error as exc:
                LOGGER.warning("Video window unavailable (%s); running headless.", exc)
        else:
            LOGGER.info("No display detected; video window disabled.")
        self.alarm.set_banner("SYSTEM READY - monitoring key board")
        self._running = True
        return self

    def stop(self) -> None:
        """Tear down every subsystem. Idempotent."""
        self._running = False
        self.camera.stop()
        if self.hand_tracker is not None:
            self.hand_tracker.close()
            self.hand_tracker = None
        self.recognizer.close()
        self.alarm.close()
        if getattr(self, "_window_ok", False):
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            self._window_ok = False
        self.db.close()
        LOGGER.info("Engine stopped.")

    def __enter__(self) -> "LockiEngine":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        """True while the main loop should keep iterating."""
        return self._running

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    def _identify_driver(self, frame: np.ndarray, now: float) -> Optional[str]:
        """Identify the driver at the station (cooldown-throttled).

        Returns the current driver id, or ``None`` when nobody is matched.
        Unmatched faces clear any previous identity so custody events can't
        be attributed to a driver who already walked away.
        """
        if now < self._identity_until:
            return self.current_driver
        self._identity_until = now + self.identify_cooldown

        match = self.recognizer.identify(frame)
        if match.matched and self.schedule.is_scheduled(match.driver_id):
            self.current_driver = match.driver_id
            route = self.schedule.get_route(match.driver_id)
            self.alarm.set_banner(
                f"Driver: {match.driver_id}   Route: {route}",
                (0, 200, 0),
            )
            return match.driver_id

        if match.matched:
            # Known face, but not on today's schedule: flag, don't trust.
            self.current_driver = None
            self.alarm.set_banner(
                f"UNLISTED DRIVER: {match.driver_id}", (0, 160, 255)
            )
            self.alarm.push_message(
                f"Driver {match.driver_id} is not on today's schedule",
                (0, 160, 255),
            )
            return None

        self.current_driver = None
        self.alarm.set_banner("No driver identified", (160, 160, 160))
        return None

    def simulate_identity(self, driver_id: Optional[str]) -> None:
        """Force the current identity (demo mode / tests only)."""
        self.current_driver = driver_id
        self._identity_until = time.monotonic() + 3600.0
        if driver_id:
            route = self.schedule.get_route(driver_id)
            self.alarm.set_banner(f"Driver: {driver_id}   Route: {route}", (0, 200, 0))
        else:
            self.alarm.set_banner("No driver identified", (160, 160, 160))

    # ------------------------------------------------------------------ #
    # Per-frame processing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _slot_visual_states(reports: List[SlotReport]) -> Dict[str, str]:
        """Map detector reports onto overlay slot colors."""
        states: Dict[str, str] = {}
        for report in reports:
            if report.status == SlotStatus.PRESENT:
                states[report.slot_id] = SlotVisualState.OK
            elif report.status == SlotStatus.MISPLACED:
                states[report.slot_id] = SlotVisualState.ALERT
            else:  # ABSENT / UNKNOWN / EMPTY
                states[report.slot_id] = SlotVisualState.IDLE
        return states

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run one full pipeline iteration; returns the annotated frame.

        Safe to call repeatedly; every stage is failure-isolated so a single
        bad frame or hiccuping tracker never takes the engine down.
        """
        now = time.monotonic()

        # 1. Key presence per slot + overlay colors.
        reports = self.detector.analyze_frame(frame)
        self.alarm.set_slot_states(self._slot_visual_states(reports))

        # 2. Driver identity (throttled internally).
        driver = self._identify_driver(frame, now)

        # 3. Hand-in-slot -> grasp events (rising edge, confirmed, latched).
        grasp_slots: List[str] = []
        if self.hand_tracker is not None:
            try:
                activations = self.hand_tracker.get_active_slots(frame)
                self._tracker_failures = 0
            except HandTrackerError as exc:
                self._tracker_failures += 1
                LOGGER.warning("Hand tracker failure #%d: %s", self._tracker_failures, exc)
                if self._tracker_failures >= MAX_TRACKER_FAILURES:
                    LOGGER.error("Disabling hand tracking for this session.")
                    self.hand_tracker.close()
                    self.hand_tracker = None
                activations = []
            for activation in activations:
                if self.tracker.register_presence(activation.slot_id, True, now):
                    grasp_slots.append(activation.slot_id)
            active_ids = {a.slot_id for a in activations}
            for slot_id in active_ids:
                if not self.tracker.is_latched(slot_id, now):
                    self.alarm.set_banner(
                        f"Pick in progress at {slot_id}...", (255, 200, 0)
                    )

        # 4. Evaluate each confirmed grasp.
        for slot_id in grasp_slots:
            self.handle_grasp_event(slot_id, frame, driver=driver)

        # 5. Render.
        return self.alarm.render_overlay(frame)

    # ------------------------------------------------------------------ #
    # Checkout evaluation
    # ------------------------------------------------------------------ #

    def handle_grasp_event(
        self,
        slot_id: str,
        frame: np.ndarray,
        driver: Optional[str] = None,
    ) -> Optional[int]:
        """Evaluate a key grasp at ``slot_id`` and persist the outcome.

        Called automatically by :meth:`process_frame` when the hand tracker
        confirms a grasp, and by the demo to script scenarios. The key's
        color is read *before* it can leave the board; when the key is
        already gone (fast grab), the slot's assigned route color is used
        as the best available evidence.

        Returns:
            The new ``checkouts`` row id, or ``None`` if the slot is unknown.
        """
        now = time.monotonic()
        driver = driver if driver is not None else self.current_driver
        slot = self.detector.get_slot(slot_id)
        if slot is None:
            LOGGER.warning("Grasp event for unknown slot %s ignored.", slot_id)
            return None
        expected_color = (
            self.detector.route_colors.get(slot.route) if slot.route else None
        )

        # What color is the key being taken? Read the board right now.
        reports = {r.slot_id: r for r in self.detector.analyze_frame(frame)}
        report = reports.get(slot_id)
        taken_color: Optional[str] = None
        if report is not None and report.coverage:
            strongest = max(report.coverage, key=lambda c: report.coverage[c])
            if report.coverage[strongest] >= self.detector.settings.min_coverage_ratio:
                taken_color = strongest
        if taken_color is None:
            # Key already vanished: fall back to the slot's assigned color.
            taken_color = expected_color
            LOGGER.info(
                "Key color unreadable at %s; using slot assignment '%s'.",
                slot_id,
                expected_color,
            )

        clip_path = self._save_evidence(frame)
        assigned_route = self.schedule.get_route(driver) if driver else None

        status = EventStatus.classify(
            assigned_route=assigned_route,
            taken_route=taken_color,
            driver_known=driver is not None,
        )
        try:
            event_id = self.db.log_event(
                driver_id=driver or self._fallback_driver_id,
                assigned_route=assigned_route or "",
                taken_route=taken_color or "",
                status=status,
                clip_path=str(clip_path) if clip_path else "",
                timestamp=utc_now_iso(),
            )
        except Exception as exc:  # DB down must not kill the loop
            LOGGER.error("Failed to persist checkout event: %s", exc)
            self.alarm.push_message("DATABASE WRITE FAILED", (0, 0, 255))
            return None

        detail = f"{driver or 'Unidentified person'} took the {taken_color} key at {slot_id}"
        if status is EventStatus.SUCCESS:
            LOGGER.info("EVENT #%d SUCCESS: %s", event_id, detail)
            self.alarm.trigger_success(detail)
        elif status is EventStatus.WRONG_ROUTE_ALERT:
            LOGGER.warning(
                "EVENT #%d WRONG_ROUTE_ALERT: %s (assigned %s)",
                event_id,
                detail,
                assigned_route,
            )
            self.alarm.trigger_alert(
                AlertLevel.WARNING,
                "WRONG ROUTE ALERT",
                f"{driver} (assigned {assigned_route}) took the {taken_color} key at {slot_id}",
            )
        else:
            LOGGER.error("EVENT #%d UNAUTHORIZED_REMOVAL: %s", event_id, detail)
            self.alarm.trigger_alert(
                AlertLevel.CRITICAL,
                "UNAUTHORIZED REMOVAL",
                f"A {taken_color} key was taken at {slot_id} by an unrecognized person",
            )
        return event_id

    def _save_evidence(self, frame: np.ndarray) -> Optional[Path]:
        """Export a short pre-incident MP4 from the camera buffer."""
        evidence_dir = Path(
            str((self.config.get("evidence") or {}).get("output_dir", "evidence"))
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")[:-3]
        target = evidence_dir / f"{stamp}.mp4"
        try:
            return self.camera.save_clip(target, seconds=EVIDENCE_CLIP_SECONDS)
        except CameraStreamError as exc:
            LOGGER.error("Evidence clip export failed: %s", exc)
            self.alarm.push_message("EVIDENCE SAVE FAILED", (0, 0, 255))
            return None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self, max_frames: Optional[int] = None) -> int:
        """Run the custody loop until interrupted or the source ends.

        Args:
            max_frames: Stop after processing this many frames (tests/demo).

        Returns:
            Number of frames processed.
        """
        self.start()
        processed = 0
        last_heartbeat = time.monotonic()
        LOGGER.info("Main loop running - press q/ESC in the video window to exit.")
        try:
            while self._running:
                frame = self.camera.read()
                if frame is None:
                    self.alarm.render_signage()
                    if not self.camera.is_running:
                        LOGGER.info("Video source ended; stopping.")
                        break
                    time.sleep(0.005)  # camera hiccup: brief backoff
                    continue

                display = self.process_frame(frame)
                if self._window_ok:
                    now_monotonic = time.monotonic()
                    if now_monotonic >= self._aspect_check_due:
                        self._aspect_check_due = now_monotonic + ASPECT_CHECK_INTERVAL_SEC
                        divergence = check_window_aspect(self.window_name, display)
                        if divergence is not None and not (
                            1.0 - ASPECT_TOLERANCE
                            <= divergence
                            <= 1.0 + ASPECT_TOLERANCE
                        ):
                            LOGGER.warning(
                                "Window/stream aspect mismatch (%.2fx): OpenCV is "
                                "letterboxing the stream (white bars). Resize the "
                                "window or fix camera frame_width/frame_height.",
                                divergence,
                            )
                    try:
                        cv2.imshow(self.window_name, display)
                    except cv2.error:
                        pass  # window closed by the OS mid-run
                    try:
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error:
                        key = 0
                    if key in (ord("q"), 27):
                        LOGGER.info("Quit key pressed.")
                        break
                self.alarm.render_signage()

                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break
                if time.monotonic() - last_heartbeat >= 30.0:
                    last_heartbeat = time.monotonic()
                    LOGGER.info("Heartbeat: %s", self.camera.stats)
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user (Ctrl+C).")
        finally:
            self.stop()
        return processed


# --------------------------------------------------------------------------- #
# Demo mode: synthetic board exercising all three custody outcomes
# --------------------------------------------------------------------------- #

BOARD_SIZE = (1920, 1080)  # (height, width) portrait, matching config ROIs
BOARD_BG = (70, 70, 70)
KEY_BGRS = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "orange": (0, 165, 255),
}


def _draw_key(frame: np.ndarray, roi, color: str, inset: int = 20) -> None:
    """Paint a key fob (colored square) inside a slot ROI."""
    x1, y1, x2, y2 = roi
    frame[y1 + inset : y2 - inset, x1 + inset : x2 - inset] = KEY_BGRS[color]


def run_demo(
    engine: LockiEngine,
    fps: int = 30,
    clock: Optional[Callable[[], float]] = None,
    max_elapsed: float = 14.0,
    stop_on_exit: bool = True,
) -> int:
    """Scripted end-to-end demo without any camera hardware.

    Timeline (all three custody outcomes):

    * t=2  : Driver_101 (Route_RED) takes the red key at SLOT-01  -> SUCCESS
    * t=6  : Driver_102 (Route_BLUE) takes a red key at SLOT-05   -> WRONG_ROUTE_ALERT
    * t=11 : nobody identified; green key vanishes at SLOT-03     -> UNAUTHORIZED_REMOVAL

    Args:
        engine: Engine to drive; stopped in ``finally``.
        fps: Simulated frame rate (paces the loop).
        clock: Optional replacement for ``time.monotonic`` (tests).
        max_elapsed: Stop after this many simulated seconds.
        stop_on_exit: Stop the engine when the demo ends (tests pass
            ``False`` to inspect state afterwards).

    Returns:
        Frames processed.
    """
    monotonic = clock or time.monotonic
    slots = engine.detector.slots
    roi_by_id = {s.slot_id: s.roi for s in slots}
    color_by_id = {
        s.slot_id: engine.detector.route_colors[s.route]
        for s in slots
        if s.route and s.route in engine.detector.route_colors
    }

    key_present = {slot_id: True for slot_id in color_by_id}
    script = [
        (2.0, "SLOT-01", "Driver_101"),
        (6.0, "SLOT-05", "Driver_102"),
        (11.0, "SLOT-03", None),
    ]
    fired = [False] * len(script)

    engine.start(with_camera=False)  # demo feeds synthetic frames instead
    start = monotonic()
    frame_interval = 1.0 / fps
    processed = 0
    LOGGER.info(
        "DEMO: scripted board; outcomes at t=%s seconds",
        [t for t, _, _ in script],
    )
    try:
        while True:
            elapsed = monotonic() - start

            frame = np.full(
                (BOARD_SIZE[0], BOARD_SIZE[1], 3), BOARD_BG, dtype=np.uint8
            )
            for slot in slots:
                x1, y1, x2, y2 = slot.roi
                frame[y1:y2, x1:x2] = (95, 95, 95)
            for slot_id, color in color_by_id.items():
                if key_present[slot_id]:
                    _draw_key(frame, roi_by_id[slot_id], color)

            # Scripted removals + custody evaluation.
            for index, (at, slot_id, driver_id) in enumerate(script):
                if not fired[index] and elapsed >= at:
                    engine.simulate_identity(driver_id)
                    key_present[slot_id] = False  # key leaves the board now
                    engine.handle_grasp_event(slot_id, frame, driver=driver_id)
                    fired[index] = True

            engine.camera.submit_frame(frame)  # feed evidence buffer too
            display = engine.process_frame(frame)
            if getattr(engine, "_window_ok", False):
                try:
                    cv2.imshow(engine.window_name, display)
                except cv2.error:
                    pass
                try:
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    key = 0
                if key in (ord("q"), 27):
                    break
            engine.alarm.render_signage()

            processed += 1
            time.sleep(frame_interval)
            if elapsed > max_elapsed:
                break
    except KeyboardInterrupt:
        LOGGER.info("Demo interrupted.")
    finally:
        if stop_on_exit:
            engine.stop()
    return processed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse main.py command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="L.O.C.K.I. real-time key custody engine.",
    )
    parser.add_argument("--config", default="config/config.json", help="Config JSON path")
    parser.add_argument(
        "--camera",
        default=None,
        help="Camera index (int) or path to a video file to replay",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the scripted synthetic-board demo (no camera required)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--no-pygame", action="store_true", help="Disable signage window")
    parser.add_argument("--no-audio", action="store_true", help="Disable chimes/alarms")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point; returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    if args.no_pygame:
        config.setdefault("ui", {})["enable_pygame"] = False
    if args.no_audio:
        config.setdefault("ui", {})["audio"] = False

    camera_source = None
    if args.camera is not None:
        camera_source = args.camera if not args.camera.isdigit() else int(args.camera)

    try:
        engine = LockiEngine(config=config, camera_source=camera_source)
    except (ConfigError, ScheduleError, FaceRecognizerError) as exc:
        LOGGER.error("Engine startup failed: %s", exc)
        return 3

    try:
        if args.demo:
            run_demo(engine)
            return 0
        engine.run(max_frames=args.max_frames)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:  # pragma: no cover - last-resort crash guard
        LOGGER.exception("Fatal error in main loop")
        engine.stop()
        return 1


if __name__ == "__main__":
    sys.exit(main())
