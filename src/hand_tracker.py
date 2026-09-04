"""
hand_tracker.py
===============

MediaPipe hand tracking for the L.O.C.K.I. system
(Live Optical Custody & Key Identifier).

Wraps the MediaPipe **Tasks** ``HandLandmarker`` and reports which slot ROIs
(from ``config/config.json``) a user is currently reaching into. Two
intersection strategies are supported and can be combined:

* **Center trigger** - the index fingertip landmark (the finger actually
  pointing at the board) sits inside the slot's rectangle. Precise, and the
  natural signal for "this person is grabbing THAT key".
* **Edge trigger** - the padded hand bounding box overlaps the slot
  rectangle. More forgiving; useful for wide slots or loose grips.

Landmark indices (MediaPipe's 21-point hand model):

* ``4``  - thumb tip
* ``8``  - **index fingertip** (the primary trigger point)
* ``12`` - middle fingertip
* ``16`` - ring fingertip
* ``20`` - pinky fingertip

The ``HandLandmarker`` requires a trained model asset (``hand_landmarker.task``).
If ``model_path`` does not exist on disk, the tracker fetches it from
``model_url`` once and caches it (configurable via ``model_url``;
``allow_model_download: false`` disables this).

Typical usage::

    from src.hand_tracker import HandTracker

    tracker = HandTracker("config/config.json")
    for frame in camera_frames:                     # BGR numpy frames
        hits = tracker.get_active_slots(frame)
        for hit in hits:
            print(hit.slot_id, hit.triggered_by)    # e.g. "SLOT-03", "index_tip"
    tracker.close()

    # IMPORTANT: when running_mode is "video", feed frames from a single
    # source in capture order (CameraStream guarantees this) - MediaPipe
    # uses monotonically increasing timestamps to track between frames.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.color_detector import ConfigError, Slot, load_config, parse_slots

LOGGER = logging.getLogger(__name__)

__all__ = [
    "HandTracker",
    "HandTrackerError",
    "HandTrackingConfig",
    "INDEX_FINGERTIP",
    "SlotActivation",
]

#: MediaPipe hand landmark index for the index fingertip.
INDEX_FINGERTIP: int = 8
#: MediaPipe hand landmark indices for all five fingertips.
FINGERTIP_LANDMARKS: Tuple[int, ...] = (4, 8, 12, 16, 20)

#: Default model asset URL (MediaPipe's official float16 hand landmarker).
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

_VALID_MODES = ("image", "video", "live_stream")


class HandTrackerError(RuntimeError):
    """Raised when hand tracking cannot start or process a frame."""


class HandTrackingConfig(NamedTuple):
    """Immutable hand-tracking settings (the config's ``hand_tracking`` section).

    ``center_trigger`` uses the index fingertip; ``edge_trigger`` uses the
    padded hand bounding box. Both may be enabled together.
    """

    enabled: bool
    running_mode: str
    num_hands: int
    min_hand_detection_confidence: float
    min_hand_presence_confidence: float
    min_tracking_confidence: float
    model_path: str
    model_url: str
    allow_model_download: bool
    bbox_padding: float
    center_trigger: bool
    edge_trigger: bool

    @classmethod
    def from_config(cls, config: dict) -> "HandTrackingConfig":
        """Build from the ``hand_tracking`` config section, with defaults.

        Raises:
            ConfigError: If section values are malformed.
        """
        raw = config.get("hand_tracking", {}) or {}
        if not isinstance(raw, dict):
            raise ConfigError("Configuration section 'hand_tracking' must be an object.")

        def _get(key: str, default, cast=None):
            """Fetch a config key with default + optional cast + validation."""
            value = raw.get(key, default)
            try:
                if cast is not None:
                    value = cast(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"hand_tracking.{key}: invalid value {raw.get(key)!r}"
                ) from exc
            return value

        enabled = bool(_get("enabled", True))
        running_mode = str(_get("running_mode", "video"))
        if running_mode not in _VALID_MODES:
            raise ConfigError(
                f"hand_tracking.running_mode must be one of {_VALID_MODES}."
            )
        num_hands = _get("num_hands", 2, int)
        if num_hands < 1:
            raise ConfigError("hand_tracking.num_hands must be >= 1.")
        min_det = _get("min_hand_detection_confidence", 0.5, float)
        min_presence = _get("min_hand_presence_confidence", 0.5, float)
        min_trk = _get("min_tracking_confidence", 0.5, float)
        for name, value in (
            ("min_hand_detection_confidence", min_det),
            ("min_hand_presence_confidence", min_presence),
            ("min_tracking_confidence", min_trk),
        ):
            if not (0.0 <= value <= 1.0):
                raise ConfigError(f"hand_tracking.{name} must be within [0, 1].")

        model_path = str(_get("model_path", "models/hand_landmarker.task"))
        model_url = str(_get("model_url", DEFAULT_MODEL_URL))
        allow_download = bool(_get("allow_model_download", True))
        bbox_padding = _get("bbox_padding", 0.0, float)
        if not (0.0 <= bbox_padding <= 0.5):
            raise ConfigError("hand_tracking.bbox_padding must be within [0, 0.5].")
        center_trigger = bool(_get("center_trigger", True))
        edge_trigger = bool(_get("edge_trigger", False))
        if not (center_trigger or edge_trigger):
            raise ConfigError(
                "hand_tracking: at least one of center_trigger / edge_trigger "
                "must be enabled."
            )
        return cls(
            enabled=enabled,
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_trk,
            model_path=model_path,
            model_url=model_url,
            allow_model_download=allow_download,
            bbox_padding=bbox_padding,
            center_trigger=center_trigger,
            edge_trigger=edge_trigger,
        )


@dataclass(frozen=True)
class SlotActivation:
    """A slot currently intersected by a hand.

    Attributes:
        slot_id: Config slot identifier (e.g. ``"SLOT-03"``).
        triggered_by: ``"index_tip"`` or ``"hand_bbox"``.
        confidence: MediaPipe's handedness score for the triggering hand.
        handedness: ``"Left"`` / ``"Right"`` as seen by the camera.
    """

    slot_id: str
    triggered_by: str
    confidence: float
    handedness: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"{self.slot_id} <- {self.triggered_by} "
            f"({self.handedness}, {self.confidence:.0%})"
        )


def _import_mediapipe():
    """Import and return the ``mediapipe`` module.

    Raises:
        HandTrackerError: With operator guidance when the package is absent
            or broken (e.g. libGL missing from a GUI OpenCV install).
    """
    try:
        import mediapipe  # noqa: PLC0415 - deferred: optional heavy dependency
    except ImportError as exc:
        raise HandTrackerError(
            "mediapipe is not installed or failed to import. Install with: "
            "pip install 'mediapipe>=0.10'. If the import fails with a "
            "libGL error, run: pip install opencv-python-headless "
            "--force-reinstall"
        ) from exc
    return mediapipe


def ensure_model(config: HandTrackingConfig) -> Path:
    """Return a usable hand-landmarker model path, downloading if allowed.

    Raises:
        HandTrackerError: If the model is missing and downloads are disabled
            or the download fails.
    """
    path = Path(config.model_path)
    if path.is_file() and path.stat().st_size > 0:
        return path
    if not config.allow_model_download:
        raise HandTrackerError(
            f"Hand landmarker model not found at '{path}' and model download "
            "is disabled (hand_tracking.allow_model_download=false). Place "
            "the .task file there or enable the download."
        )
    LOGGER.info("Downloading hand landmarker model to %s ...", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(config.model_url, timeout=30) as response:
            tmp_path.write_bytes(response.read())
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HandTrackerError(
            f"Failed to download hand landmarker model from "
            f"{config.model_url}: {exc}"
        ) from exc
    LOGGER.info("Model ready: %s (%d bytes)", path, path.stat().st_size)
    return path


class HandTracker:
    """Detects hand-vs-slot-ROI intersections using MediaPipe HandLandmarker.

    Args:
        config_path: Path to the L.O.C.K.I. JSON configuration file.
        config: Pre-parsed config dict (alternative to ``config_path``).

    Raises:
        ConfigError: If the configuration is invalid.
        HandTrackerError: If MediaPipe is unavailable, the model asset is
            missing, or the landmarker fails to initialize.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = load_config(Path(config_path) if config_path else "config/config.json")
        self._slots: List[Slot] = parse_slots(config)
        self._cfg = HandTrackingConfig.from_config(config)
        self._mp = _import_mediapipe()
        self._landmarker = None
        self._closed = False
        self._result = None

        if not self._cfg.enabled:
            LOGGER.info("HandTracker: hand_tracking.enabled=false; idle.")
            return

        model_path = ensure_model(self._cfg)
        (
            base_options_cls,
            options_cls,
            running_mode_enum,
            landmarker_cls,
        ) = self._import_tasks_classes()

        options = options_cls(
            base_options=base_options_cls(model_asset_path=str(model_path)),
            running_mode=running_mode_enum[self._cfg.running_mode.upper()],
            num_hands=self._cfg.num_hands,
            min_hand_detection_confidence=self._cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=self._cfg.min_hand_presence_confidence,
            min_tracking_confidence=self._cfg.min_tracking_confidence,
            result_callback=(
                self._on_live_result if self._cfg.running_mode == "live_stream" else None
            ),
        )

        try:
            self._landmarker = landmarker_cls.create_from_options(options)
        except Exception as exc:
            raise HandTrackerError(
                f"Failed to initialize MediaPipe HandLandmarker: {exc}"
            ) from exc

        LOGGER.info(
            "HandTracker ready: %d slot(s), mode=%s, num_hands=%d, triggers=%s",
            len(self._slots),
            self._cfg.running_mode,
            self._cfg.num_hands,
            "+".join(
                name
                for name, on in (
                    ("center", self._cfg.center_trigger),
                    ("edge", self._cfg.edge_trigger),
                )
                if on
            ),
        )

    @staticmethod
    def _import_tasks_classes() -> Tuple[Any, Any, Any, Any]:
        """Return ``(BaseOptions, HandLandmarkerOptions, RunningMode, HandLandmarker)``.

        mediapipe 1.0 re-exports ``BaseOptions`` on ``mediapipe.tasks.python``
        and deletes the ``core`` submodule; 0.10.x keeps it under
        ``mediapipe.tasks.python.core``. Both layouts are supported.

        Raises:
            HandTrackerError: If neither layout is importable.
        """
        try:  # mediapipe >= 1.0
            from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415
            return (
                BaseOptions,
                vision.HandLandmarkerOptions,
                vision.RunningMode,
                vision.HandLandmarker,
            )
        except ImportError:
            pass
        try:  # mediapipe 0.10.x
            from mediapipe.tasks.python import vision  # noqa: PLC0415
            from mediapipe.tasks.python.core import BaseOptions  # noqa: PLC0415
            return (
                BaseOptions,
                vision.HandLandmarkerOptions,
                vision.RunningMode,
                vision.HandLandmarker,
            )
        except ImportError as exc:
            raise HandTrackerError(
                "mediapipe is installed but the Tasks API is unavailable "
                f"({exc}). Install mediapipe>=0.10."
            ) from exc

    # ------------------------------------------------------------------ #
    # Live-stream plumbing (result delivered on a MediaPipe worker thread)
    # ------------------------------------------------------------------ #

    def _on_live_result(self, result, output_image, timestamp_ms) -> None:  # noqa: ANN001
        """Callback for ``live_stream`` mode; stores the latest result."""
        del output_image, timestamp_ms
        self._result = result

    def _detect(self, frame: np.ndarray, timestamp_ms: Optional[int]):
        """Run inference and return a landmarker result (or ``None``).

        Raises:
            HandTrackerError: If the tracker is closed or inference fails.
        """
        if self._closed:
            raise HandTrackerError("HandTracker has been closed.")
        if self._landmarker is None:
            return None
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame
        )
        try:
            if self._cfg.running_mode == "image":
                return self._landmarker.detect(mp_image)
            if self._cfg.running_mode == "live_stream":
                ts = timestamp_ms if timestamp_ms is not None else int(time.monotonic() * 1000)
                self._landmarker.detect_async(mp_image, ts)
                return self._result
            # "video" mode: timestamps must be strictly increasing.
            self._last_ts = getattr(self, "_last_ts", -1) + 1
            return self._landmarker.detect_for_video(mp_image, self._last_ts)
        except Exception as exc:
            raise HandTrackerError(f"MediaPipe inference failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hand_bbox(
        hand_landmarks,  # mediapipe List[NormalizedLandmark]
        frame_w: int,
        frame_h: int,
        padding: float = 0.0,
    ) -> Tuple[int, int, int, int]:
        """Pixel-space hand bounding box ``(x1, y1, x2, y2)``, clamped to frame.

        Args:
            hand_landmarks: List of landmarks with normalized x/y in [0, 1].
            frame_w / frame_h: Frame dimensions the coords map to.
            padding: Fraction of the box's larger side added on every side.
        """
        xs = [lm.x * frame_w for lm in hand_landmarks]
        ys = [lm.y * frame_h for lm in hand_landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if padding > 0.0:
            pad = padding * max(x2 - x1, y2 - y1)
            x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        return (
            max(0, int(x1)),
            max(0, int(y1)),
            min(frame_w - 1, int(x2)),
            min(frame_h - 1, int(y2)),
        )

    @staticmethod
    def _rects_overlap(
        a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
    ) -> bool:
        """True if axis-aligned rects ``a`` and ``b`` intersect (inclusive edges)."""
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    @staticmethod
    def _point_in_rect(point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
        """True if ``point`` lies inside ``rect`` (inclusive edges)."""
        return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #

    def get_active_slots(
        self,
        frame: np.ndarray,
        slot_ids: Optional[Sequence[str]] = None,
        timestamp_ms: Optional[int] = None,
    ) -> List[SlotActivation]:
        """Return slots currently touched by any detected hand.

        Args:
            frame: BGR numpy frame (e.g. from :class:`src.camera.CameraStream`).
            slot_ids: Optional subset of slot ids to test.
            timestamp_ms: Frame timestamp for video/live_stream modes; when
                omitted, a synthetic monotonically increasing clock is used.

        Returns:
            Activations sorted by slot id. One entry per (slot, hand) pair;
            ``"index_tip"`` wins over ``"hand_bbox"`` when both fire for the
            same slot and hand.

        Raises:
            HandTrackerError: If the tracker is closed or inference fails.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise HandTrackerError("get_active_slots expects a non-empty BGR frame.")

        targets = self._slots
        if slot_ids is not None:
            wanted = set(slot_ids)
            targets = [s for s in self._slots if s.slot_id in wanted]
        if not targets or self._landmarker is None:
            return []

        height, width = frame.shape[:2]
        result = self._detect(frame, timestamp_ms)
        hands = list(getattr(result, "hand_landmarks", None) or [])
        if not hands:
            return []
        handedness_list = list(
            getattr(result, "handedness", None) or []
        )

        activations: Dict[Tuple[str, int], SlotActivation] = {}
        for hand_index, hand_landmarks in enumerate(hands):
            score, label = 0.0, "Unknown"
            if hand_index < len(handedness_list) and handedness_list[hand_index]:
                category = handedness_list[hand_index][0]
                score = float(category.score)
                label = str(category.category_name)

            bbox = self._hand_bbox(
                hand_landmarks, width, height, self._cfg.bbox_padding
            )
            tip = hand_landmarks[INDEX_FINGERTIP]
            tip_px = (int(tip.x * width), int(tip.y * height))

            for slot in targets:
                triggered_by: Optional[str] = None
                if self._cfg.center_trigger and self._point_in_rect(tip_px, slot.roi):
                    triggered_by = "index_tip"
                if (
                    self._cfg.edge_trigger
                    and triggered_by is None
                    and self._rects_overlap(bbox, slot.roi)
                ):
                    triggered_by = "hand_bbox"
                if triggered_by is not None:
                    activations[(slot.slot_id, hand_index)] = SlotActivation(
                        slot_id=slot.slot_id,
                        triggered_by=triggered_by,
                        confidence=score,
                        handedness=label,
                    )
        return [activations[key] for key in sorted(activations)]

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def annotate(
        self,
        frame: np.ndarray,
        activations: Optional[Sequence[SlotActivation]] = None,
        draw_all_slots: bool = True,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with slot rectangles and overlays.

        Args:
            frame: BGR frame to copy and draw on (the input is not mutated).
            activations: Output of :meth:`get_active_slots` for this frame;
                ``None`` runs detection internally.
            draw_all_slots: Also draw inactive slots as thin gray outlines.

        Returns:
            The annotated frame.
        """
        canvas = frame.copy()
        activations = list(activations) if activations is not None else self.get_active_slots(frame)
        active_ids = {a.slot_id for a in activations}

        if draw_all_slots:
            for slot in self._slots:
                is_active = slot.slot_id in active_ids
                color = (0, 200, 0) if is_active else (160, 160, 160)
                thickness = 2 if is_active else 1
                x1, y1, x2, y2 = slot.roi
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    canvas,
                    slot.slot_id,
                    (x1 + 2, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        for row, activation in enumerate(activations):
            cv2.putText(
                canvas,
                f"{activation.handedness}:{activation.triggered_by}",
                (10, 24 + 22 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Release the MediaPipe landmarker. Idempotent."""
        landmarker = getattr(self, "_landmarker", None)
        if not self._closed and landmarker is not None:
            try:
                landmarker.close()
            except Exception as exc:  # pragma: no cover - teardown robustness
                LOGGER.warning("Error closing HandLandmarker: %s", exc)
        self._landmarker = None
        self._closed = True

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        state = "closed" if self._closed else "ready"
        return f"HandTracker({state}, {len(self._slots)} slots)"
