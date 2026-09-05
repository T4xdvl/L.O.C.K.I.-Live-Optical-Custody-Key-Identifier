"""
alarm_system.py
===============

Visual and audible alert rendering for the L.O.C.K.I. system
(Live Optical Custody & Key Identifier).

Two output layers, both driven from a single :class:`AlarmSystem` facade:

**OpenCV overlay layer** (always available, headless-safe)
    Draws directly onto the video frame the operator watches:
    per-slot state rectangles (green = key present & correct, red = wrong
    route, gray = empty), a status banner, and transient event messages.
    :meth:`AlarmSystem.render_overlay` returns a *new* frame; the input is
    never mutated.

**Pygame signage layer** (optional, needs a display + SDL)
    A dedicated window for a wall-mounted status monitor: full-screen
    borders (green = valid pick in progress, flashing red = wrong-route
    alert, amber = unauthorized removal, blue idle), large verdict text,
    and audio cues via ``pygame.mixer`` — a pleasant chime for valid
    checkouts, a harsh repeated alarm for wrong-route/unauthorized events.

Every call is failure-tolerant: a missing audio device or headless X server
degrades to logs, never crashes the custody pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from src.color_detector import ConfigError, Slot, load_config, parse_slots

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AlarmSystem",
    "AlarmConfig",
    "AlertLevel",
    "SlotVisualState",
    "render_roi_debug",
]

#: BGR colors used by the OpenCV overlay layer.
COLOR_IDLE = (160, 160, 160)       # gray   - slot inactive
COLOR_OK = (0, 200, 0)             # green  - key present, route correct
COLOR_ALERT = (0, 0, 255)          # red    - wrong route
COLOR_UNAUTHORIZED = (0, 160, 255) # amber  - unauthorized removal
COLOR_ACCENT = (255, 200, 0)       # blue-ish banner accent
COLOR_TEXT = (255, 255, 255)       # white

#: States a slot can be rendered in (mapped to colors/thickness in one place).
class SlotVisualState:
    """Per-slot rendering state names (strings to stay serializable)."""

    IDLE = "IDLE"
    OK = "OK"
    ALERT = "ALERT"
    UNAUTHORIZED = "UNAUTHORIZED"


_SLOT_COLORS: Dict[str, Tuple[int, int, int]] = {
    SlotVisualState.IDLE: COLOR_IDLE,
    SlotVisualState.OK: COLOR_OK,
    SlotVisualState.ALERT: COLOR_ALERT,
    SlotVisualState.UNAUTHORIZED: COLOR_UNAUTHORIZED,
}


class AlertLevel:
    """Severity ladder for the signage layer."""

    NONE = "NONE"                # idle signage
    INFO = "INFO"                # valid pick / status text
    WARNING = "WARNING"          # wrong route
    CRITICAL = "CRITICAL"        # unauthorized removal


class AlarmConfig(NamedTuple):
    """Settings from the config's ``ui`` section."""

    enable_pygame: bool
    audio: bool
    window_name: str
    alert_flash_seconds: float
    audio_backend: str

    @classmethod
    def from_config(cls, config: dict) -> "AlarmConfig":
        """Build from the ``ui`` config section with defaults.

        Raises:
            ConfigError: If section values are malformed.
        """
        raw = config.get("ui", {}) or {}
        if not isinstance(raw, dict):
            raise ConfigError("Configuration section 'ui' must be an object.")

        def _get(key: str, default, cast=None):
            """Fetch a config key with default + optional cast + validation."""
            value = raw.get(key, default)
            try:
                if cast is not None:
                    value = cast(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"ui.{key}: invalid value {raw.get(key)!r}") from exc
            return value

        alert_flash = _get("alert_flash_seconds", 4.0, float)
        if alert_flash < 0.5:
            raise ConfigError("ui.alert_flash_seconds must be >= 0.5.")
        return cls(
            enable_pygame=bool(_get("enable_pygame", True)),
            audio=bool(_get("audio", True)),
            window_name=str(_get("window_name", "L.O.C.K.I. Key Custody Monitor")),
            alert_flash_seconds=alert_flash,
            audio_backend=str(_get("audio_backend", "auto")),
        )


@dataclass
class _TimedMessage:
    """A transient overlay message with an expiry timestamp."""

    text: str
    color: Tuple[int, int, int]
    expires_at: float


class AlarmSystem:
    """Unified alert renderer: OpenCV overlays + optional pygame signage.

    Args:
        config_path: Path to the L.O.C.K.I. JSON configuration file.
        config: Pre-parsed config dict (alternative to ``config_path``).

    Raises:
        ConfigError: If the configuration is invalid.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = load_config(Path(config_path) if config_path else "config/config.json")
        self._cfg = AlarmConfig.from_config(config)
        self._slots: List[Slot] = parse_slots(config)
        self._slot_states: Dict[str, str] = {
            slot.slot_id: SlotVisualState.IDLE for slot in self._slots
        }
        self._messages: List[_TimedMessage] = []
        self._system_warning = ""  # persistent amber line under the banner
        self._banner_text = "SYSTEM READY"
        self._banner_color = COLOR_ACCENT
        self._alert_until = 0.0  # monotonic deadline of the active alert
        self._alert_level = AlertLevel.NONE
        self._last_flash_toggle = 0.0
        self._flash_on = False
        self._pending_alert: Optional[Tuple[str, str, Tuple[int, int, int]]] = None
        self._pending_success: Optional[str] = None
        self._lock = threading.RLock()

        self._pygame_ready = False
        self._pygame: Any = None  # module object when signage is available
        if self._cfg.enable_pygame:
            self._init_pygame()

    # ------------------------------------------------------------------ #
    # Pygame signage layer
    # ------------------------------------------------------------------ #

    def _init_pygame(self) -> None:
        """Initialize pygame video+audio, degrading to logs on failure."""
        try:
            import pygame  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            LOGGER.warning("pygame not installed (%s); signage window disabled.", exc)
            return
        self._pygame = pygame
        try:
            pygame.init()
            if self._cfg.audio:
                # auto → SDL default driver; explicit backends (e.g. 'pulse',
                # 'alsa', 'dummy') are forced via SDL_AUDIODRIVER when given.
                if self._cfg.audio_backend.lower() not in ("", "auto"):
                    import os  # noqa: PLC0415

                    os.environ.setdefault(
                        "SDL_AUDIODRIVER", self._cfg.audio_backend.lower()
                    )
                pygame.mixer.init()
                self._make_sounds()
            info = pygame.display.Info()
            self._screen = pygame.display.set_mode((info.current_w, info.current_h))
            pygame.display.set_caption(self._cfg.window_name)
            self._font_large = pygame.font.SysFont(None, 96, bold=True)
            self._font_small = pygame.font.SysFont(None, 40)
            self._pygame_ready = True
            LOGGER.info("Pygame signage ready (%dx%d).", info.current_w, info.current_h)
        except Exception as exc:
            LOGGER.warning(
                "Pygame signage unavailable (%s); continuing with OpenCV "
                "overlays only.",
                exc,
            )
            self._pygame_ready = False

    def _make_sounds(self) -> None:
        """Synthesize the chime/alarm WAVs in memory (no asset files needed).

        A two-note rising chime signals a valid checkout; a dissonant
        repeating buzz signals wrong-route / unauthorized removal.
        """
        import io  # noqa: PLC0415
        import struct  # noqa: PLC0415
        import wave  # noqa: PLC0415

        sample_rate = 22050

        def synth_tone(freq_hz: float, duration_s: float, volume: float) -> List[int]:
            """One sine wave, gently squared off for a less piercing buzz."""
            frames: List[int] = []
            for i in range(int(sample_rate * duration_s)):
                value = np.sin(2.0 * np.pi * freq_hz * i / sample_rate)
                value = 0.6 * value + 0.4 * np.tanh(3.0 * value)
                frames.append(int(volume * 32767 * value))
            return frames

        def to_wav(frames: List[int]) -> Any:
            """Encode PCM frames into an in-memory WAV and wrap as a Sound."""
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(
                    struct.pack("<%dh" % len(frames), *frames)
                )
            buffer.seek(0)
            return self._pygame.mixer.Sound(file=buffer)

        # Valid checkout: rising major third (E5 -> A5), short and pleasant.
        chime = synth_tone(659.25, 0.12, 0.5) + synth_tone(880.0, 0.22, 0.5)
        fade = int(0.05 * sample_rate)
        for i in range(fade):  # de-click both ends
            gain = i / fade
            chime[i] = int(chime[i] * gain)
            chime[-1 - i] = int(chime[-1 - i] * gain)
        self._sound_chime = to_wav(chime)

        # Alert: minor-second buzz (466 + 494 Hz), long and grating.
        buzz = synth_tone(466.16, 0.45, 0.6)
        buzz2 = synth_tone(493.88, 0.45, 0.6)
        alarm = [int(a * 0.5 + b * 0.5) for a, b in zip(buzz, buzz2)] * 2
        self._sound_alarm = to_wav(alarm)
        LOGGER.info("Audio cues synthesized (chime + alarm).")

    # ------------------------------------------------------------------ #
    # Event ingestion
    # ------------------------------------------------------------------ #

    def set_slot_state(self, slot_id: str, state: str) -> None:
        """Update one slot's rectangle color (``SlotVisualState`` constants)."""
        with self._lock:
            if slot_id in self._slot_states:
                self._slot_states[slot_id] = state

    def set_slot_states(self, states: Dict[str, str]) -> None:
        """Bulk-update slot states (e.g. from one detector pass)."""
        with self._lock:
            for slot_id, state in states.items():
                if slot_id in self._slot_states:
                    self._slot_states[slot_id] = state

    def set_banner(self, text: str, color: Tuple[int, int, int] = COLOR_ACCENT) -> None:
        """Set the persistent status banner line."""
        with self._lock:
            self._banner_text = text
            self._banner_color = color

    def push_message(
        self, text: str, color: Tuple[int, int, int] = COLOR_TEXT, ttl: float = 3.0
    ) -> None:
        """Show a transient message on the overlay for ``ttl`` seconds."""
        with self._lock:
            self._messages.append(
                _TimedMessage(text=text, color=color, expires_at=time.monotonic() + ttl)
            )

    def set_system_warning(self, text: str) -> None:
        """Show a persistent amber warning line under the banner.

        Unlike :meth:`push_message` messages, this line has no expiry and
        stays visible until :meth:`clear_system_warning` is called - used
        for conditions like "camera size smaller than slot ROIs" that last
        for the whole session.
        """
        with self._lock:
            self._system_warning = text

    def clear_system_warning(self) -> None:
        """Remove the persistent warning line (idempotent)."""
        with self._lock:
            self._system_warning = ""

    # ------------------------------------------------------------------ #
    # Alert triggering
    # ------------------------------------------------------------------ #

    def trigger_alert(self, level: str, title: str, detail: str = "") -> None:
        """Raise a timed alert on both layers and play the matching sound.

        Args:
            level: ``AlertLevel.WARNING`` (wrong route) or
                ``AlertLevel.CRITICAL`` (unauthorized removal); INFO is
                handled by :meth:`push_message` instead.
            title: Short verdict, e.g. ``"WRONG ROUTE ALERT"``.
            detail: One-line explanation shown under the title.
        """
        now = time.monotonic()
        with self._lock:
            self._alert_level = level
            self._alert_until = now + self._cfg.alert_flash_seconds
            color = COLOR_ALERT if level == AlertLevel.WARNING else COLOR_UNAUTHORIZED
            self.push_message(f"{title}: {detail}".strip(": "), color, ttl=self._cfg.alert_flash_seconds)
            self._pending_alert = (title, detail, color)

        if self._pygame_ready and self._cfg.audio:
            try:
                self._sound_alarm.play()
            except Exception as exc:  # audio device vanished mid-run
                LOGGER.warning("Alarm sound failed: %s", exc)
        else:
            LOGGER.warning("ALERT [%s] %s %s", level, title, detail)

    def trigger_success(self, detail: str = "") -> None:
        """Play the checkout chime and flash the signage green."""
        if self._pygame_ready and self._cfg.audio:
            try:
                self._sound_chime.play()
            except Exception as exc:
                LOGGER.warning("Chime failed: %s", exc)
        self.push_message(f"CHECKOUT OK: {detail}".strip(": "), COLOR_OK)
        with self._lock:
            self._pending_success = detail

    # ------------------------------------------------------------------ #
    # Per-frame rendering
    # ------------------------------------------------------------------ #

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of ``frame`` with slot boxes, banner, and messages.

        Safe on headless hosts: pure numpy/cv2 drawing, no display needed.
        """
        now = time.monotonic()
        with self._lock:
            self._messages = [m for m in self._messages if m.expires_at > now]
            # Flash active while an alert is live (2 Hz square wave).
            if now < self._alert_until:
                if now - self._last_flash_toggle >= 0.25:
                    self._flash_on = not self._flash_on
                    self._last_flash_toggle = now
                flash_active = self._flash_on
            else:
                self._alert_level = AlertLevel.NONE
                flash_active = False
            slot_states = dict(self._slot_states)
            banner_text, banner_color = self._banner_text, self._banner_color
            messages = list(self._messages)
            system_warning = self._system_warning
            if now < self._alert_until:
                flash_border = (
                    COLOR_ALERT
                    if self._alert_level == AlertLevel.WARNING
                    else COLOR_UNAUTHORIZED
                )
            else:
                flash_border = None

        canvas = frame.copy()
        height, width = canvas.shape[:2]

        # 1. Slot rectangles with labels.
        for slot in self._slots:
            state = slot_states.get(slot.slot_id, SlotVisualState.IDLE)
            color = _SLOT_COLORS.get(state, COLOR_IDLE)
            x1, y1, x2, y2 = slot.roi
            thickness = 3 if state in (SlotVisualState.ALERT, SlotVisualState.UNAUTHORIZED) else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                canvas,
                f"{slot.slot_id} [{state}]",
                (x1 + 2, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # 2. Flashing screen border during an active alert.
        if flash_border is not None and flash_active:
            for thickness in range(0, 10):
                cv2.rectangle(
                    canvas,
                    (thickness, thickness),
                    (width - 1 - thickness, height - 1 - thickness),
                    flash_border,
                    1,
                )

        # 3. Banner + message stack + persistent warning (translucent bar).
        bar_height = 34 + 26 * (len(messages) + (1 if system_warning else 0))
        overlay_layer = canvas.copy()
        cv2.rectangle(overlay_layer, (0, 0), (width, bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay_layer, 0.55, canvas, 0.45, 0, canvas)
        cv2.putText(
            canvas,
            banner_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            banner_color,
            2,
            cv2.LINE_AA,
        )
        for index, message in enumerate(messages):
            cv2.putText(
                canvas,
                message.text,
                (12, 52 + 26 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                message.color,
                1,
                cv2.LINE_AA,
            )
        if system_warning:
            cv2.putText(
                canvas,
                system_warning,
                (12, 52 + 26 * len(messages)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_UNAUTHORIZED,
                1,
                cv2.LINE_AA,
            )
        return canvas

    # ------------------------------------------------------------------ #
    # Pygame signage frame (call once per main-loop iteration)
    # ------------------------------------------------------------------ #

    def render_signage(self) -> None:
        """Refresh the pygame status window (no-op when unavailable)."""
        if not self._pygame_ready:
            return
        now = time.monotonic()
        with self._lock:
            alert_live = now < self._alert_until
            level = self._alert_level
            flash_on = self._flash_on
            banner = self._banner_text
            pending_alert = self._pending_alert
            pending_success = self._pending_success
            self._pending_alert = None
            self._pending_success = None

        assert self._pygame is not None
        pygame = self._pygame
        screen_w, screen_h = self._screen.get_size()
        try:
            pygame.event.pump()  # keep the OS window manager happy

            if alert_live and level == AlertLevel.WARNING and not flash_on:
                border_color, fill_color, verdict = (200, 0, 0), (60, 0, 0), "WRONG ROUTE"
            elif alert_live and level == AlertLevel.WARNING:
                border_color, fill_color, verdict = (255, 60, 60), (60, 0, 0), "WRONG ROUTE"
            elif alert_live:
                border_color, fill_color, verdict = (
                    (255, 170, 40) if flash_on else (200, 120, 20)
                ), (55, 35, 0), "UNAUTHORIZED"
            else:
                border_color, fill_color, verdict = (40, 90, 200), (5, 8, 20), banner

            self._screen.fill(fill_color)
            border = 14
            pygame.draw.rect(
                self._screen, border_color, (0, 0, screen_w, screen_h), border
            )

            detail_text = ""
            if pending_alert is not None:
                verdict = pending_alert[0]
                detail_text = pending_alert[1]
            elif pending_success is not None:
                verdict = "CHECKOUT OK"
                detail_text = pending_success

            lines = [verdict, detail_text] if detail_text else [verdict]
            y = screen_h // 2 - 40 * (len(lines) - 1)
            for index, line in enumerate(lines):
                font = self._font_large if index == 0 else self._font_small
                surface = font.render(line, True, (255, 255, 255))
                rect = surface.get_rect(center=(screen_w // 2, y + 80 * index))
                self._screen.blit(surface, rect)
            pygame.display.flip()
        except Exception as exc:  # display vanished (HDMI unplug etc.)
            LOGGER.warning("Signage render failed, disabling pygame layer: %s", exc)
            self._pygame_ready = False

    def close(self) -> None:
        """Shut down pygame cleanly. Idempotent."""
        if getattr(self, "_pygame_ready", False):
            try:
                self._pygame.quit()
            except Exception:  # pragma: no cover - teardown robustness
                pass
            self._pygame_ready = False

    def __enter__(self) -> "AlarmSystem":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"AlarmSystem(slots={len(self._slots)}, "
            f"signage={self._pygame_ready}, audio={self._cfg.audio})"
        )


def render_roi_debug(
    frame: np.ndarray,
    slots: List[Slot],
    frame_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with slot ROIs drawn for alignment checks.

    Debug aid (``main.py --debug-rois``): every configured slot box is
    outlined and labeled with its id/route/key color so an operator can
    verify the ROIs line up with the physical board at a glance.

    * Boxes whose ROI fits ``frame_size`` are green; boxes outside are red.
    * ``frame_size`` is the camera's *actual* delivered size; ``None`` uses
      the frame's own shape (the operating assumption when unknown).
    * A white border and ``WxH`` caption mark the full frame bounds, so
      ROIs that fall outside the delivered picture are obvious.

    Safe on headless hosts: pure numpy/cv2 drawing.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    bounds_w, bounds_h = frame_size if frame_size is not None else (width, height)

    # Full-frame bounds so out-of-picture ROIs are visually obvious.
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), COLOR_TEXT, 1)
    cv2.putText(
        canvas,
        f"frame {bounds_w}x{bounds_h}",
        (12, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    for slot in slots:
        x1, y1, x2, y2 = slot.roi
        fits = x1 >= 0 and y1 >= 0 and x2 <= bounds_w and y2 <= bounds_h
        color = COLOR_OK if fits else COLOR_ALERT
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{slot.slot_id} ({slot.route or 'UNASSIGNED'})"
        cv2.putText(
            canvas,
            label,
            (x1 + 2, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
