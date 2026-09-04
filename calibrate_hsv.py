#!/usr/bin/env python3
"""
calibrate_hsv.py
================

Interactive HSV calibration utility for the L.O.C.K.I. system
(Live Optical Custody & Key Identifier).

Room lighting in a shipping center changes through the day (dock doors open,
overhead banks switch on). This tool lets an operator re-tune the HSV
min/max thresholds used by :mod:`src.color_detector` and write them straight
back into ``config/config.json``.

Workflow
--------
1. Point the camera at the key board and run::

       python calibrate_hsv.py                       # default config/camera
       python calibrate_hsv.py --camera 1 --width 1280 --height 720

   (On macOS you may need to grant Terminal camera permission first.)

2. Place a key fob of one color in view. The live preview shows, on the
   right, the pixels currently inside the threshold in white.
3. Drag the **H/S/V min** and **H/S/V max** trackbars until the mask cleanly
   covers only the fob's colored plastic.
4. Press ``s`` to save the values for the active color into the config file,
   or ``p`` to print them to the console. ``h`` cycles Red -> Blue -> Green
   -> Orange; ``r`` resets the trackbars to the config values; ``q``/``ESC``
   quits without saving.

Every save writes a timestamped backup next to the config file, e.g.
``config.json.bak-2026-09-04T10-15-30``.

Requires an OpenCV build with HighGUI support (``opencv-python`` provides
one). Exit code is non-zero when no camera can be opened, making the script
safe to call from provisioning scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("calibrate_hsv")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"

# Order the operator cycles through with the 'h' key.
CALIBRATION_ORDER: Tuple[str, ...] = ("red", "blue", "green", "orange")

# Trackbar names as shown in the control window, mapped to config keys.
TRACKBARS: Tuple[Tuple[str, str], ...] = (
    ("H Min", "h_min"),
    ("S Min", "s_min"),
    ("V Min", "v_min"),
    ("H Max", "h_max"),
    ("S Max", "s_max"),
    ("V Max", "v_max"),
)

# Fall back to a fully-open window if a color has no usable config entry.
FALLBACK_RANGE: Dict[str, int] = {
    "h_min": 0, "s_min": 0, "v_min": 0, "h_max": 179, "s_max": 255, "v_max": 255,
}

WINDOW_CONTROLS = "L.O.C.K.I. HSV Calibration - Controls"
WINDOW_PREVIEW = "L.O.C.K.I. HSV Calibration - Preview"

KEY_SAVE = ord("s")
KEY_PRINT = ord("p")
KEY_NEXT_COLOR = ord("h")
KEY_RESET = ord("r")
KEY_QUIT = ord("q")
KEY_ESC = 27

#: Consecutive unreadable frames tolerated before giving up.
MAX_CAMERA_FAILURES = 30


class CalibrationError(RuntimeError):
    """Raised when calibration cannot start or persist its results."""


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> dict:
    """Load the JSON config, raising :class:`CalibrationError` on failure."""
    if not path.is_file():
        raise CalibrationError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise CalibrationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CalibrationError("Top-level configuration must be a JSON object.")
    return config


def backup_config(path: Path) -> Optional[Path]:
    """Copy ``path`` to a timestamped ``.bak-<timestamp>`` sibling.

    Best-effort: a failed backup is logged but never blocks a save, because
    the freshly tuned calibration values are more valuable than the old file.
    """
    if not path.is_file():
        return None
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    backup_path = path.with_name(f"{path.name}.bak-{stamp}")
    try:
        shutil.copy2(path, backup_path)
        LOGGER.info("Backed up config to %s", backup_path)
        return backup_path
    except OSError as exc:
        LOGGER.warning("Could not back up %s: %s", path, exc)
        return None


def save_hsv_ranges(
    config: dict,
    color: str,
    ranges: List[Dict[str, int]],
    config_path: Path,
) -> Path:
    """Write ``ranges`` for ``color`` into ``config`` and persist it atomically.

    The JSON is written to a temp sibling and renamed, so a crash mid-write
    can never leave a truncated ``config.json`` behind.

    Raises:
        CalibrationError: If the file cannot be written.
    """
    config.setdefault("hsv_ranges", {})[color] = ranges
    backup_config(config_path)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4)
            handle.write("\n")
        tmp_path.replace(config_path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise CalibrationError(f"Failed to save {config_path}: {exc}") from exc
    LOGGER.info("Saved %s HSV range(s) to %s", color, config_path)
    return config_path


def get_color_ranges(config: dict, color: str) -> List[Dict[str, int]]:
    """Return the stored range dict(s) for ``color`` with sane fallbacks.

    Accepts either a single range object or a list (red stores two). The
    first range is what the trackbars are seeded with; saving always writes
    a single range, replacing any multi-band entry.
    """
    raw = config.get("hsv_ranges", {}).get(color, [])
    if isinstance(raw, dict):
        raw = [raw]
    ranges: List[Dict[str, int]] = []
    for entry in raw:
        try:
            ranges.append(
                {
                    "h_min": int(entry["h_min"]),
                    "s_min": int(entry["s_min"]),
                    "v_min": int(entry["v_min"]),
                    "h_max": int(entry["h_max"]),
                    "s_max": int(entry["s_max"]),
                    "v_max": int(entry["v_max"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping malformed range for '%s': %s", color, exc)
    return ranges or [dict(FALLBACK_RANGE)]


# --------------------------------------------------------------------------- #
# Interactive calibrator
# --------------------------------------------------------------------------- #

class HSVCameraCalibrator:
    """OpenCV trackbar UI driving live HSV threshold tuning.

    Args:
        config: Parsed configuration dictionary (mutated on save).
        config_path: File the updated ranges are written back to.
        camera_index: OpenCV capture device index.
        frame_w / frame_h: Requested capture resolution.
        active_color: Color calibrated when the loop starts.
    """

    def __init__(
        self,
        config: dict,
        config_path: Path,
        camera_index: int = 0,
        frame_w: int = 1280,
        frame_h: int = 720,
        active_color: str = CALIBRATION_ORDER[0],
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.camera_index = camera_index
        self.frame_w = frame_w
        self.frame_h = frame_h
        if active_color not in CALIBRATION_ORDER:
            raise CalibrationError(
                f"Unknown color '{active_color}'. Choose one of: "
                f"{', '.join(CALIBRATION_ORDER)}."
            )
        self.active_color = active_color
        self._status = f"Calibrating {active_color.upper()}"
        self._capture: Optional[cv2.VideoCapture] = None
        self._windows_created = False

    # -- camera --------------------------------------------------------- #

    def open_camera(self) -> None:
        """Open the capture device.

        V4L2 is requested explicitly on Linux for predictable device naming;
        other platforms use OpenCV's default backend.

        Raises:
            CalibrationError: If no camera is available.
        """
        try:
            if sys.platform.startswith("linux"):
                self._capture = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if self._capture is None or not self._capture.isOpened():
                self._capture = cv2.VideoCapture(self.camera_index)
        except cv2.error as exc:  # pragma: no cover - backend specific
            LOGGER.warning("Camera backend error: %s", exc)
            self._capture = None

        if self._capture is None or not self._capture.isOpened():
            raise CalibrationError(
                f"Could not open camera index {self.camera_index}. Check the "
                "device, OS camera permissions, or try --camera N."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        LOGGER.info("Camera %d opened at %dx%d", self.camera_index, actual_w, actual_h)

    def close_camera(self) -> None:
        """Release the capture device and destroy all OpenCV windows."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        cv2.destroyAllWindows()

    def _read_frame(self) -> Optional[np.ndarray]:
        """Grab one BGR frame, or ``None`` if the camera hiccupped."""
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            return None
        return frame

    # -- trackbars ------------------------------------------------------- #

    def _create_windows(self) -> None:
        """Create the UI windows and seed the trackbars with saved values.

        Raises:
            CalibrationError: If the GUI backend is unavailable (e.g. running
                on a headless server with ``opencv-python-headless``).
        """
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            # Calling HighGUI without a display can hard-abort the process
            # on some OpenCV builds - fail cleanly instead.
            raise CalibrationError(
                "No display detected (headless host). Run the calibration "
                "tool on a machine with a GUI session."
            )
        try:
            cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
            cv2.namedWindow(WINDOW_PREVIEW, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise CalibrationError(
                "OpenCV GUI is unavailable on this system. Install a build "
                "with HighGUI support (e.g. 'pip install opencv-python') and "
                "run on a machine with a display."
            ) from exc
        self._windows_created = True
        for label, _ in TRACKBARS:
            cv2.createTrackbar(label, WINDOW_CONTROLS, 0, 255, lambda *_: None)
        # Hue maxes out at 179 in OpenCV's 8-bit HSV space.
        cv2.setTrackbarMax("H Min", WINDOW_CONTROLS, 179)
        cv2.setTrackbarMax("H Max", WINDOW_CONTROLS, 179)
        self.apply_ranges(get_color_ranges(self.config, self.active_color)[0])

    def apply_ranges(self, rng: Dict[str, int]) -> None:
        """Push a range dict into the six trackbars."""
        for label, key in TRACKBARS:
            cv2.setTrackbarPos(label, WINDOW_CONTROLS, int(rng[key]))

    def current_range(self) -> Dict[str, int]:
        """Read the six trackbar positions into a range dict.

        Min/max are swapped per channel if the operator dragged them past
        each other, so a careless drag can't produce an inverted mask.
        """
        values: Dict[str, int] = {}
        for label, key in TRACKBARS:
            values[key] = cv2.getTrackbarPos(label, WINDOW_CONTROLS)
        for channel in "hsv":
            lo, hi = f"{channel}_min", f"{channel}_max"
            if values[lo] > values[hi]:
                values[lo], values[hi] = values[hi], values[lo]
        return values

    def build_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Binarize ``hsv_frame`` using the current trackbar values."""
        values = self.current_range()
        lower = np.array(
            [values["h_min"], values["s_min"], values["v_min"]], dtype=np.uint8
        )
        upper = np.array(
            [values["h_max"], values["s_max"], values["v_max"]], dtype=np.uint8
        )
        return cv2.inRange(hsv_frame, lower, upper)

    # -- preview ---------------------------------------------------------- #

    def compose_preview(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compose the [live frame | mask] preview with a status banner."""
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview = np.hstack((frame, mask_bgr))

        height, width = preview.shape[:2]
        banner = np.zeros((30, width, 3), dtype=np.uint8)
        cv2.putText(
            banner,
            f"{self._status}  |  {self.range_summary()}  |  "
            "s=save  p=print  h=next color  r=reset  q=quit",
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return np.vstack((preview, banner))

    def range_summary(self) -> str:
        """Compact '[H 0-10 S 80-255 V 60-255]' style summary string."""
        v = self.current_range()
        return (
            f"[H {v['h_min']}-{v['h_max']} "
            f"S {v['s_min']}-{v['s_max']} "
            f"V {v['v_min']}-{v['v_max']}]"
        )

    # -- actions ------------------------------------------------------------ #

    def cycle_color(self) -> None:
        """Switch to the next color in :data:`CALIBRATION_ORDER`."""
        idx = CALIBRATION_ORDER.index(self.active_color)
        self.active_color = CALIBRATION_ORDER[(idx + 1) % len(CALIBRATION_ORDER)]
        self.apply_ranges(get_color_ranges(self.config, self.active_color)[0])
        self._status = f"Calibrating {self.active_color.upper()}"

    def reset_active(self) -> None:
        """Reload the trackbars from the last saved values for this color."""
        self.apply_ranges(get_color_ranges(self.config, self.active_color)[0])
        self._status = f"Reset to saved {self.active_color.upper()} values"

    def save_active(self) -> None:
        """Persist the current trackbar values for the active color."""
        try:
            save_hsv_ranges(
                self.config,
                self.active_color,
                [self.current_range()],
                self.config_path,
            )
        except CalibrationError as exc:
            LOGGER.error("Save failed: %s", exc)
            self._status = "SAVE FAILED - see console"
            return
        self._status = f"Saved {self.active_color.upper()}"

    # -- main loop ---------------------------------------------------------- #

    def run(self) -> None:
        """Run the interactive loop until the user quits.

        Raises:
            CalibrationError: If the GUI cannot be created or the camera
                stops delivering frames.
        """
        self._create_windows()
        if not self._windows_created:  # pragma: no cover - defensive
            raise CalibrationError("Failed to create OpenCV windows.")

        consecutive_failures = 0
        while True:
            frame = self._read_frame()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CAMERA_FAILURES:
                    raise CalibrationError(
                        f"Camera stopped delivering frames after "
                        f"{MAX_CAMERA_FAILURES} consecutive attempts."
                    )
                if consecutive_failures == 1:
                    LOGGER.warning("No frame from camera; retrying...")
                time.sleep(0.1)
                if (cv2.waitKey(1) & 0xFF) in (KEY_QUIT, KEY_ESC):
                    break
                continue
            consecutive_failures = 0

            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = self.build_mask(hsv)
            except cv2.error as exc:
                raise CalibrationError(f"OpenCV processing error: {exc}") from exc

            cv2.setWindowTitle(
                WINDOW_PREVIEW,
                f"{WINDOW_PREVIEW} | {self.active_color.upper()}",
            )
            cv2.imshow(WINDOW_PREVIEW, self.compose_preview(frame, mask))

            key = cv2.waitKey(1) & 0xFF
            if key in (KEY_QUIT, KEY_ESC):
                LOGGER.info("Calibration closed.")
                break
            if key == KEY_SAVE:
                self.save_active()
            elif key == KEY_PRINT:
                print(f"{self.active_color}: {self.current_range()}")
            elif key == KEY_NEXT_COLOR:
                self.cycle_color()
            elif key == KEY_RESET:
                self.reset_active()


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the calibration utility."""
    parser = argparse.ArgumentParser(
        prog="calibrate_hsv.py",
        description=(
            "Interactively fine-tune L.O.C.K.I. HSV color thresholds with "
            "OpenCV trackbars and save them to the config file."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Path to config JSON (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Requested frame width (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Requested frame height (default: 720)",
    )
    parser.add_argument(
        "--color", choices=CALIBRATION_ORDER, default=None,
        help="Color to calibrate first (default: red)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns a process exit code.

    Exit codes: 0 success, 2 bad config, 3 no camera, 4 runtime failure,
    130 interrupted.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
    except CalibrationError as exc:
        LOGGER.error("%s", exc)
        return 2

    try:
        calibrator = HSVCameraCalibrator(
            config=config,
            config_path=args.config,
            camera_index=args.camera,
            frame_w=args.width,
            frame_h=args.height,
            active_color=args.color or CALIBRATION_ORDER[0],
        )
        calibrator.open_camera()
    except CalibrationError as exc:
        LOGGER.error("%s", exc)
        return 3

    try:
        calibrator.run()
    except CalibrationError as exc:
        LOGGER.error("%s", exc)
        return 4
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 130
    finally:
        calibrator.close_camera()
    return 0


if __name__ == "__main__":
    sys.exit(main())
