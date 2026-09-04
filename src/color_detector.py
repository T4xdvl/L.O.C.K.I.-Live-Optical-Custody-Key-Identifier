"""
color_detector.py
=================

Color-based key-fob presence analysis for the L.O.C.K.I. system
(Live Optical Custody & Key Identifier).

The public entry point is :class:`ColorDetector`. Given a BGR frame from a
wall-board camera and the slot / route / HSV configuration file, it reports
for every slot whether the expected route's key fob is:

* ``PRESENT``    - the expected color dominates the slot's ROI.
* ``ABSENT``     - too few pixels of any expected color are visible.
* ``MISPLACED``  - a *different* route's key color dominates the slot.
* ``UNKNOWN``    - ambiguous result (multiple colors pass the dominance
  test), usually caused by bad lighting or bad HSV thresholds.
* ``EMPTY``      - a convenience alias emitted when the slot is configured
  with no route ("UNASSIGNED"): the slot should be free of any key color.

Typical usage::

    from src.color_detector import ColorDetector

    detector = ColorDetector("config/config.json")
    for frame in camera_frames:                # BGR frames from cv2.VideoCapture
        for report in detector.analyze_frame(frame):
            print(report.slot_id, report.status, report.confidence)

All OpenCV interaction is local to this module, so the detector can be unit
tested with synthetic images.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Hue values in OpenCV's 8-bit HSV space span 0..179.
MAX_HUE: int = 179
MAX_SAT: int = 255
MAX_VAL: int = 255

#: Colors a key fob may be painted with. Kept as a frozen tuple so callers
#: can iterate safely without mutating module state.
SUPPORTED_COLORS: Tuple[str, ...] = ("red", "blue", "green", "orange")

import re as _re

#: Pre-compiled matchers translating free-text route names onto canonical colors.
_COLOR_WORD_PATTERNS: Tuple[Tuple[str, "_re.Pattern[str]"], ...] = tuple(
    (color, _re.compile(color, _re.IGNORECASE)) for color in SUPPORTED_COLORS
)


def extract_color(text: Optional[str]) -> Optional[str]:
    """Translate free-text route names (e.g. ``"Route_RED"``) to canonical colors.

    Shared by the schedule loader and the database status classifier so
    every module agrees on what route text means.

    Returns:
        The canonical color name, or ``None`` when no color word matches.
    """
    value = (text or "").strip()
    if not value:
        return None
    for color, pattern in _COLOR_WORD_PATTERNS:
        if pattern.search(value):
            return color
    return None


class DetectorError(RuntimeError):
    """Raised when the detector cannot operate on the given configuration."""


class ConfigError(DetectorError):
    """Raised when the configuration file is missing, malformed, or invalid."""


class FrameError(DetectorError):
    """Raised when a frame cannot be analyzed (wrong type, empty, etc.)."""


def load_config(config_path: str | Path) -> dict:
    """Read and minimally validate a L.O.C.K.I. JSON configuration file.

    Shared by every L.O.C.K.I. module so all components parse the config
    identically and fail identically.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        The parsed configuration as a dictionary.

    Raises:
        ConfigError: If the file is missing, unreadable, or not valid JSON.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError("Top-level configuration must be a JSON object.")

    for section in ("slots", "route_colors", "hsv_ranges"):
        if not isinstance(config.get(section), (dict, list)):
            raise ConfigError(f"Configuration section '{section}' is required.")

    if not config["slots"]:
        raise ConfigError("No slots defined in configuration.")
    return config


def parse_slots(config: dict) -> List[Slot]:
    """Convert a config dict's ``slots`` section into validated :class:`Slot` objects.

    Shared helper: camera, hand-tracking, and detection code all need the
    same ROI geometry, so it is parsed exactly once this way.

    Raises:
        ConfigError: If any slot entry is missing keys, malformed, or invalid.
    """
    slots: List[Slot] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(config["slots"]):
        try:
            slot_id = str(raw["id"])
            if slot_id in seen_ids:
                raise ConfigError(f"Duplicate slot id '{slot_id}'.")
            seen_ids.add(slot_id)

            roi_raw = raw["roi"]
            if not isinstance(roi_raw, (list, tuple)) or len(roi_raw) != 4:
                raise ConfigError(
                    f"Slot '{slot_id}': 'roi' must be [x1, y1, x2, y2]."
                )
            x1, y1, x2, y2 = (int(v) for v in roi_raw)
            if x2 <= x1 or y2 <= y1:
                raise ConfigError(
                    f"Slot '{slot_id}': degenerate ROI {roi_raw} "
                    "(need x2 > x1 and y2 > y1)."
                )

            route_raw = raw.get("route")
            route = None if route_raw in (None, "", "UNASSIGNED") else str(route_raw)

            slots.append(
                Slot(
                    slot_id=slot_id,
                    route=route,
                    roi=(x1, y1, x2, y2),
                    description=str(raw.get("description", "")),
                )
            )
        except KeyError as exc:
            raise ConfigError(f"Slot #{idx}: missing required key {exc}.") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Slot #{idx}: invalid values ({exc}).") from exc
    return slots


class SlotStatus(str, enum.Enum):
    """Discrete custody states a slot can be reported in."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    MISPLACED = "MISPLACED"
    UNKNOWN = "UNKNOWN"
    EMPTY = "EMPTY"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class HSVRange:
    """Inclusive HSV bounds (OpenCV 8-bit scale: H 0-179, S/V 0-255)."""

    h_min: int
    s_min: int
    v_min: int
    h_max: int
    s_max: int
    v_max: int

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` uint8 arrays suitable for ``cv2.inRange``."""
        return (
            np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8),
            np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8),
        )


@dataclass(frozen=True)
class Slot:
    """A physical key position on the board with its route assignment."""

    slot_id: str
    route: Optional[str]  # ``None`` / "UNASSIGNED" means the slot must stay empty
    roi: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    description: str = ""

    @property
    def x1(self) -> int:
        return self.roi[0]

    @property
    def y1(self) -> int:
        return self.roi[1]

    @property
    def x2(self) -> int:
        return self.roi[2]

    @property
    def y2(self) -> int:
        return self.roi[3]


@dataclass(frozen=True)
class DetectionSettings:
    """Tuning knobs for the classification heuristics."""

    #: Fraction of ROI pixels a color must cover to count as "visible".
    min_coverage_ratio: float = 0.02
    #: Winning color must cover at least this many times the runner-up.
    dominance_margin: float = 1.6
    #: Median-blur kernel size applied before masking (odd, >= 1).
    blur_kernel_size: int = 5


@dataclass(frozen=True)
class SlotReport:
    """Result of analyzing a single slot in a single frame."""

    slot_id: str
    route: Optional[str]
    expected_color: Optional[str]
    status: SlotStatus
    confidence: float  # 0.0 - 1.0 coverage of the deciding color
    coverage: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - trivial
        expected = self.expected_color or "none"
        return (
            f"{self.slot_id}: {self.status.value} "
            f"(expected {expected}, confidence {self.confidence:.2%})"
        )


class ColorDetector:
    """Analyzes camera frames and reports per-slot key custody states.

    Args:
        config_path: Path to the JSON configuration file.
        settings: Optional overrides for the detection heuristics. When
            omitted, values are read from the config's ``detection`` section.

    Raises:
        ConfigError: If the configuration is missing or invalid.
    """

    def __init__(
        self,
        config_path: str | Path = "config/config.json",
        settings: Optional[DetectionSettings] = None,
        config: Optional[dict] = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._config = config if config is not None else load_config(self._config_path)

        self._slots: List[Slot] = parse_slots(self._config)
        self._route_colors: Dict[str, str] = self._parse_route_colors(self._config)
        self._hsv_ranges: Dict[str, List[HSVRange]] = self._parse_hsv_ranges(self._config)
        self._settings: DetectionSettings = settings or self._parse_settings(self._config)

        LOGGER.info(
            "ColorDetector ready: %d slot(s), colors=%s, config=%s",
            len(self._slots),
            sorted(self._hsv_ranges),
            self._config_path,
        )

    @staticmethod
    def _parse_route_colors(config: dict) -> Dict[str, str]:
        """Extract the route -> color mapping, validating color names."""
        raw = config["route_colors"]
        if not isinstance(raw, dict):
            raise ConfigError("'route_colors' must be an object.")
        mapping: Dict[str, str] = {}
        for route, color in raw.items():
            color_l = str(color).lower()
            if color_l not in SUPPORTED_COLORS:
                raise ConfigError(
                    f"Route '{route}' uses unsupported color '{color}'. "
                    f"Supported: {', '.join(SUPPORTED_COLORS)}."
                )
            mapping[str(route)] = color_l
        return mapping

    @staticmethod
    def _parse_hsv_ranges(config: dict) -> Dict[str, List[HSVRange]]:
        """Extract per-color HSV ranges.

        Red needs *two* ranges (it wraps around hue 0), so every color maps
        to a list of ranges; a single object is accepted and wrapped.
        """
        raw = config["hsv_ranges"]
        if not isinstance(raw, dict):
            raise ConfigError("'hsv_ranges' must be an object.")

        parsed: Dict[str, List[HSVRange]] = {}
        for color, value in raw.items():
            color_l = str(color).lower()
            if color_l not in SUPPORTED_COLORS:
                raise ConfigError(
                    f"hsv_ranges entry '{color}' is not a supported color "
                    f"({', '.join(SUPPORTED_COLORS)})."
                )
            entries = value if isinstance(value, list) else [value]
            ranges: List[HSVRange] = []
            for entry in entries:
                try:
                    rng = HSVRange(
                        h_min=int(entry["h_min"]),
                        s_min=int(entry["s_min"]),
                        v_min=int(entry["v_min"]),
                        h_max=int(entry["h_max"]),
                        s_max=int(entry["s_max"]),
                        v_max=int(entry["v_max"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"hsv_ranges['{color}']: invalid range entry ({exc})."
                    ) from exc
                if not (0 <= rng.h_min <= rng.h_max <= MAX_HUE):
                    raise ConfigError(
                        f"hsv_ranges['{color}']: hue bounds invalid "
                        f"(0 <= h_min <= h_max <= {MAX_HUE})."
                    )
                if not (0 <= rng.s_min <= rng.s_max <= MAX_SAT):
                    raise ConfigError(
                        f"hsv_ranges['{color}']: saturation bounds invalid."
                    )
                if not (0 <= rng.v_min <= rng.v_max <= MAX_VAL):
                    raise ConfigError(f"hsv_ranges['{color}']: value bounds invalid.")
                ranges.append(rng)
            if not ranges:
                raise ConfigError(f"hsv_ranges['{color}']: at least one range required.")
            parsed[color_l] = ranges
        return parsed

    @staticmethod
    def _parse_settings(config: dict) -> DetectionSettings:
        """Read the optional 'detection' tuning section, falling back to defaults."""
        raw = config.get("detection", {}) or {}
        try:
            return DetectionSettings(
                min_coverage_ratio=float(raw.get("min_coverage_ratio", 0.02)),
                dominance_margin=float(raw.get("dominance_margin", 1.6)),
                blur_kernel_size=int(raw.get("blur_kernel_size", 5)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid 'detection' settings: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #

    @property
    def slots(self) -> List[Slot]:
        """Configured slots (read-only copy)."""
        return list(self._slots)

    @property
    def route_colors(self) -> Dict[str, str]:
        """Route -> color mapping (read-only copy)."""
        return dict(self._route_colors)

    @property
    def settings(self) -> DetectionSettings:
        """Active detection heuristics."""
        return self._settings

    def get_slot(self, slot_id: str) -> Optional[Slot]:
        """Return the slot with the given id, or ``None``."""
        for slot in self._slots:
            if slot.slot_id == slot_id:
                return slot
        return None

    # ------------------------------------------------------------------ #
    # Detection pipeline
    # ------------------------------------------------------------------ #

    def _crop_roi(self, frame: np.ndarray, slot: Slot) -> np.ndarray:
        """Crop the slot's ROI out of ``frame``, clamping to frame bounds.

        Slots whose ROI *center* falls outside the frame are rejected: the
        slot is simply not visible, and analyzing a clamped sliver of it
        would produce misleading coverage numbers.

        Raises:
            FrameError: If the slot's ROI center lies outside the frame.
        """
        height, width = frame.shape[:2]
        center_x = (slot.x1 + slot.x2) / 2.0
        center_y = (slot.y1 + slot.y2) / 2.0
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise FrameError(
                f"Slot '{slot.slot_id}' ROI {slot.roi} lies outside the "
                f"{width}x{height} frame."
            )
        x1 = max(0, min(slot.x1, width - 1))
        y1 = max(0, min(slot.y1, height - 1))
        x2 = max(x1 + 1, min(slot.x2, width))
        y2 = max(y1 + 1, min(slot.y2, height))
        return frame[y1:y2, x1:x2]

    def _color_coverage(self, hsv_roi: np.ndarray, color: str) -> float:
        """Return the fraction of pixels in ``hsv_roi`` matching ``color``.

        Multiple ranges (e.g. red's two bands) are OR-combined. The mask is
        median-blurred first to suppress sensor noise and specular speckle.
        """
        ranges = self._hsv_ranges.get(color)
        if not ranges:
            return 0.0

        kernel = self._settings.blur_kernel_size
        blurred = hsv_roi
        if kernel > 1:
            if kernel % 2 == 0:  # cv2.medianBlur requires an odd kernel.
                kernel += 1
            blurred = cv2.medianBlur(hsv_roi, kernel)

        total_mask: Optional[np.ndarray] = None
        for rng in ranges:
            lower, upper = rng.to_arrays()
            mask = cv2.inRange(blurred, lower, upper)
            total_mask = mask if total_mask is None else cv2.bitwise_or(total_mask, mask)

        assert total_mask is not None  # narrowed by the `if not ranges` guard
        return float(np.count_nonzero(total_mask)) / float(total_mask.size)

    def analyze_slot(self, frame: np.ndarray, slot: Slot) -> SlotReport:
        """Analyze a single slot in a BGR frame.

        Raises:
            FrameError: If ``frame`` is not a valid non-empty BGR ndarray.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise FrameError("analyze_slot expects a non-empty numpy image array.")
        if frame.size == 0:
            raise FrameError("analyze_slot received an empty frame.")

        expected_color = (
            self._route_colors.get(slot.route) if slot.route is not None else None
        )
        coverage: Dict[str, float] = {}

        # Unassigned slots must stay empty: any strong color is an anomaly.
        if expected_color is None:
            hsv_roi = self._crop_to_hsv(frame, slot)
            for color in self._hsv_ranges:
                coverage[color] = self._color_coverage(hsv_roi, color)
            strongest = max(coverage, key=lambda c: coverage[c], default=None)
            strongest_cov = coverage[strongest] if strongest else 0.0
            if strongest_cov >= self._settings.min_coverage_ratio:
                status, confidence = SlotStatus.UNKNOWN, strongest_cov
            else:
                status, confidence = SlotStatus.EMPTY, 1.0 - strongest_cov
            return SlotReport(slot.slot_id, slot.route, None, status, confidence, coverage)

        if expected_color not in self._hsv_ranges:
            LOGGER.warning("No HSV range configured for color '%s'.", expected_color)
            return SlotReport(
                slot.slot_id, slot.route, expected_color, SlotStatus.UNKNOWN, 0.0, coverage
            )

        hsv_roi = self._crop_to_hsv(frame, slot)

        for color in self._hsv_ranges:
            coverage[color] = self._color_coverage(hsv_roi, color)

        expected_cov = coverage.get(expected_color, 0.0)
        others = {c: v for c, v in coverage.items() if c != expected_color}
        best_other_color = max(others, key=lambda c: others[c], default=None)
        best_other_cov = others.get(best_other_color, 0.0) if best_other_color else 0.0

        min_cov = self._settings.min_coverage_ratio
        margin = self._settings.dominance_margin

        expected_present = expected_cov >= min_cov
        other_present = best_other_cov >= min_cov

        if expected_present and not other_present:
            status, confidence = SlotStatus.PRESENT, expected_cov
        elif expected_present and other_present and expected_cov >= best_other_cov * margin:
            status, confidence = SlotStatus.PRESENT, expected_cov
        elif not expected_present and not other_present:
            status, confidence = SlotStatus.ABSENT, 0.0
        elif not expected_present and other_present:
            assert best_other_color is not None
            status, confidence = SlotStatus.MISPLACED, best_other_cov
        else:
            # Both visible and neither dominates -> ambiguous.
            status, confidence = SlotStatus.UNKNOWN, max(expected_cov, best_other_cov)

        return SlotReport(
            slot.slot_id, slot.route, expected_color, status, confidence, coverage
        )

    def _crop_to_hsv(self, frame: np.ndarray, slot: Slot) -> np.ndarray:
        """Crop the slot ROI and convert it to HSV (one shared code path)."""
        crop = self._crop_roi(frame, slot)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    def analyze_frame(
        self,
        frame: np.ndarray,
        slot_ids: Optional[Sequence[str]] = None,
    ) -> List[SlotReport]:
        """Analyze one frame and return a report for each requested slot.

        Args:
            frame: BGR image (numpy array) from the wall-board camera.
            slot_ids: Optional subset of slot ids to analyze. ``None`` means all.

        Returns:
            A list of :class:`SlotReport` in configuration order.

        Raises:
            FrameError: If the frame itself is unusable. A failure limited to a
                single slot (e.g. ROI outside the frame) is logged and skipped
                so one bad slot cannot stall the whole pipeline.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise FrameError("analyze_frame expects a non-empty BGR numpy array.")

        targets = self._slots
        if slot_ids is not None:
            wanted = set(slot_ids)
            targets = [s for s in self._slots if s.slot_id in wanted]
            missing = wanted - {s.slot_id for s in targets}
            if missing:
                LOGGER.warning("Unknown slot id(s) ignored: %s", sorted(missing))

        reports: List[SlotReport] = []
        for slot in targets:
            try:
                reports.append(self.analyze_slot(frame, slot))
            except FrameError as exc:
                LOGGER.error("Skipping slot %s: %s", slot.slot_id, exc)
        return reports


__all__ = [
    "ColorDetector",
    "ConfigError",
    "DetectionSettings",
    "DetectorError",
    "FrameError",
    "HSVRange",
    "MAX_HUE",
    "MAX_SAT",
    "MAX_VAL",
    "Slot",
    "SlotReport",
    "SlotStatus",
    "SUPPORTED_COLORS",
    "load_config",
    "parse_slots",
]
