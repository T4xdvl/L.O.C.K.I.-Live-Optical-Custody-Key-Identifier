"""
face_recognizer.py
==================

Driver face identification for the L.O.C.K.I. system
(Live Optical Custody & Key Identifier).

Uses `InsightFace <https://github.com/deepinsight/insightface>`_ (buffalo
model packs: SCRFD detection + ArcFace embeddings) to detect faces at the
key station and match them against a gallery of enrolled drivers. When no
library model reaches the configured cosine threshold — or InsightFace is
not installed / its models are missing — the recognizer degrades gracefully
and returns the configured **fallback driver id** (default
``UNIDENTIFIED``), which the pipeline records as an
``UNAUTHORIZED_REMOVAL`` instead of crashing or guessing.

Enrollment
----------
Drop one or more JPEG/PNG photos of each driver into ``data/faces/<driver_id>/``
(the ``face_recognition.gallery_dir`` config key). The first frame of each
image is embedded on startup; every embedding for a driver is kept, so
multiple photos per driver improve matching. For deployment, re-embed the
gallery with :meth:`FaceRecognizer.rebuild_gallery` after adding photos.

Typical usage::

    from src.face_recognizer import FaceRecognizer

    recognizer = FaceRecognizer("config/config.json")
    match = recognizer.identify(frame)      # whole frame
    print(match.driver_id, match.confidence)   # "Driver_101", 0.62

    # Or restrict detection to the station area of the frame:
    match = recognizer.identify(frame, region=(1200, 100, 1900, 900))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from src.color_detector import ConfigError, load_config

LOGGER = logging.getLogger(__name__)

__all__ = [
    "FaceRecognizer",
    "FaceRecognizerError",
    "FaceMatch",
    "FaceRecognitionConfig",
]

#: Cosine distance below this value counts as a match (buffalo_l typical
#: operating point: <= 0.4 strict, <= 0.5 lenient).
DEFAULT_RECOGNITION_THRESHOLD = 0.45


class FaceRecognizerError(RuntimeError):
    """Raised when face recognition cannot start or process a frame."""


class FaceRecognitionConfig(NamedTuple):
    """Immutable settings from the config's ``face_recognition`` section."""

    enabled: bool
    model_pack: str
    providers: Tuple[str, ...]
    det_size: Tuple[int, int]
    recognition_threshold: float
    min_face_size: int
    gallery_dir: Path
    fallback_driver_id: str

    @classmethod
    def from_config(cls, config: dict) -> "FaceRecognitionConfig":
        """Build from the ``face_recognition`` section, with defaults.

        Raises:
            ConfigError: If section values are malformed.
        """
        raw = config.get("face_recognition", {}) or {}
        if not isinstance(raw, dict):
            raise ConfigError("Configuration section 'face_recognition' must be an object.")

        def _get(key: str, default, cast=None):
            """Fetch a config key with default + optional cast + validation."""
            value = raw.get(key, default)
            try:
                if cast is not None:
                    value = cast(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"face_recognition.{key}: invalid value {raw.get(key)!r}"
                ) from exc
            return value

        enabled = bool(_get("enabled", True))
        model_pack = str(_get("model_pack", "buffalo_s"))
        providers_raw = _get("providers", ["CPUExecutionProvider"])
        if isinstance(providers_raw, str):
            providers_raw = [providers_raw]
        providers = tuple(str(p) for p in providers_raw)
        det_size_raw = _get("det_size", [640, 640])
        try:
            det_size = (int(det_size_raw[0]), int(det_size_raw[1]))
        except (TypeError, KeyError, IndexError, ValueError) as exc:
            raise ConfigError(f"face_recognition.det_size: invalid value {det_size_raw!r}") from exc
        threshold = _get("recognition_threshold", DEFAULT_RECOGNITION_THRESHOLD, float)
        # Cosine distance threshold: 0 = identical, 1 = unrelated. Cap the
        # lenient end so operators can't configure random-face matches.
        if not (0.0 < threshold < 1.0):
            raise ConfigError("face_recognition.recognition_threshold must be in (0, 1).")
        min_face = _get("min_face_size", 60, int)
        if min_face < 16:
            raise ConfigError("face_recognition.min_face_size must be >= 16 px.")
        gallery_dir = Path(str(_get("gallery_dir", "data/faces")))
        fallback_id = str(_get("fallback_driver_id", "UNIDENTIFIED"))
        if not fallback_id:
            raise ConfigError("face_recognition.fallback_driver_id must not be empty.")
        return cls(
            enabled=enabled,
            model_pack=model_pack,
            providers=providers,
            det_size=det_size,
            recognition_threshold=threshold,
            min_face_size=min_face,
            gallery_dir=gallery_dir,
            fallback_driver_id=fallback_id,
        )


@dataclass(frozen=True)
class FaceMatch:
    """Result of one identification attempt.

    Attributes:
        driver_id: Matched driver, or the fallback id when nothing matched.
        confidence: ``1 - cosine_distance`` for the best library face (0-1).
            Exactly ``0.0`` when identification fell back.
        matched: True when a gallery driver passed the threshold.
        bbox: Detected face box ``(x1, y1, x2, y2)`` in frame coordinates,
            ``None`` when no face was found at all.
    """

    driver_id: str
    confidence: float
    matched: bool
    bbox: Optional[Tuple[int, int, int, int]] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        label = self.driver_id if self.matched else f"{self.driver_id} (fallback)"
        return f"{label} @ {self.confidence:.0%}"


class _Gallery:
    """Driver embeddings with per-driver cosine-distance best-match lookup."""

    def __init__(self) -> None:
        self._embeddings: Dict[str, List[np.ndarray]] = {}

    def add(self, driver_id: str, embedding: np.ndarray) -> None:
        """Store one embedding for ``driver_id`` (copied, L2-normalized)."""
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm == 0 or not np.isfinite(norm):
            raise FaceRecognizerError("Refusing to add a degenerate embedding.")
        vec = vec / norm
        self._embeddings.setdefault(driver_id, []).append(vec)

    def drivers(self) -> List[str]:
        """Enrolled driver ids."""
        return sorted(self._embeddings)

    def __len__(self) -> int:
        return sum(len(vecs) for vecs in self._embeddings.values())

    def best_match(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Return ``(driver_id, cosine_distance)`` of the closest embedding.

        Distance is the minimum over every enrolled embedding of every
        driver. Returns ``(None, inf)`` when the gallery is empty.
        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None, float("inf")
        vec = vec / norm

        best_driver: Optional[str] = None
        best_distance = float("inf")
        for driver_id, vectors in self._embeddings.items():
            stack = np.stack(vectors)  # (n, d), already normalized
            # Normalized dot product == cosine similarity; distance = 1 - sim.
            distances = 1.0 - stack @ vec
            driver_best = float(np.min(distances))
            if driver_best < best_distance:
                best_distance = driver_best
                best_driver = driver_id
        return best_driver, best_distance


class FaceRecognizer:
    """Detects faces and matches them against the enrolled driver gallery.

    Args:
        config_path: Path to the L.O.C.K.I. JSON configuration file.
        config: Pre-parsed config dict (alternative to ``config_path``).
        gallery_dir: Explicit gallery override; wins over the config value.

    Raises:
        ConfigError: If the configuration is invalid.
        FaceRecognizerError: If InsightFace is unavailable AND the config
            does not explicitly disable recognition.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
        gallery_dir: Optional[str] = None,
    ) -> None:
        if config is None:
            config = load_config(Path(config_path) if config_path else "config/config.json")
        self._cfg = FaceRecognitionConfig.from_config(config)
        if gallery_dir is not None:
            self._cfg = self._cfg._replace(gallery_dir=Path(gallery_dir))

        self._gallery = _Gallery()
        self._app: Optional[Any] = None

        if not self._cfg.enabled:
            LOGGER.info("FaceRecognizer: face_recognition.enabled=false; all matches fall back.")
            return

        self._app = self._build_analyzer()
        self._enroll_gallery()

    # ------------------------------------------------------------------ #
    # Engine construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _import_face_analyzer():
        """Return the InsightFace ``FaceAnalysis`` class.

        Raises:
            FaceRecognizerError: With remediation guidance when unavailable.
        """
        try:
            from insightface.app import FaceAnalysis  # noqa: PLC0415
        except ImportError as exc:
            raise FaceRecognizerError(
                "InsightFace is not installed. Install with: "
                "pip install 'insightface>=0.7' (see requirements-face.txt)."
            ) from exc
        return FaceAnalysis

    def _build_analyzer(self) -> Any:
        """Instantiate the InsightFace analyzer with configured providers.

        Raises:
            FaceRecognizerError: If initialization or the first model
                download fails.
        """
        FaceAnalysis = self._import_face_analyzer()
        try:
            app = FaceAnalysis(
                name=self._cfg.model_pack,
                providers=list(self._cfg.providers),
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=-1, det_size=self._cfg.det_size)
        except Exception as exc:
            raise FaceRecognizerError(
                f"Failed to initialize InsightFace '{self._cfg.model_pack}': {exc}. "
                "If this is the first run, models download from the InsightFace "
                "GitHub releases; check network access and disk space."
            ) from exc
        LOGGER.info(
            "InsightFace ready: pack=%s det_size=%s providers=%s",
            self._cfg.model_pack,
            self._cfg.det_size,
            list(self._cfg.providers),
        )
        return app

    # ------------------------------------------------------------------ #
    # Enrollment
    # ------------------------------------------------------------------ #

    def _enroll_gallery(self) -> None:
        """Embed every image in the gallery directory (non-fatal on errors).

        A completely unreadable gallery logs an error but does not raise:
        the recognizer still functions and returns fallback matches, which
        is the safer operational behavior for a custody system.
        """
        gallery = self._cfg.gallery_dir
        if not gallery.is_dir():
            LOGGER.warning(
                "Face gallery dir %s does not exist; every identification "
                "will fall back to '%s'. Enroll drivers by adding "
                "data/faces/<driver_id>/*.jpg",
                gallery,
                self._cfg.fallback_driver_id,
            )
            return

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        enrolled_drivers = 0
        skipped = 0
        for driver_dir in sorted(p for p in gallery.iterdir() if p.is_dir()):
            driver_id = driver_dir.name
            added_for_driver = 0
            for image_path in sorted(driver_dir.iterdir()):
                if image_path.suffix.lower() not in image_extensions:
                    continue
                try:
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise FaceRecognizerError(f"unreadable image")
                    embedding = self._embed_first_face(image)
                    if embedding is None:
                        raise FaceRecognizerError("no face found")
                    self._gallery.add(driver_id, embedding)
                    added_for_driver += 1
                except FaceRecognizerError as exc:
                    skipped += 1
                    LOGGER.warning("Skipping %s: %s", image_path.name, exc)
            if added_for_driver:
                enrolled_drivers += 1
                LOGGER.info("Enrolled '%s' (%d photo(s))", driver_id, added_for_driver)

        if enrolled_drivers == 0:
            LOGGER.warning(
                "Face gallery at %s produced no usable embeddings; every "
                "identification will fall back to '%s'.",
                gallery,
                self._cfg.fallback_driver_id,
            )
        else:
            LOGGER.info(
                "Gallery ready: %d driver(s), %d embedding(s), %d file(s) skipped",
                enrolled_drivers,
                len(self._gallery),
                skipped,
            )

    def rebuild_gallery(self) -> int:
        """Re-embed the gallery directory; returns total embedding count.

        Call after adding/changing enrollment photos without restarting the
        service. Existing embeddings are replaced.

        Raises:
            FaceRecognizerError: If the analyzer is unavailable (disabled
            recognition) — otherwise returns silently with 0 on an empty
            gallery.
        """
        self._gallery = _Gallery()
        if self._app is not None:
            self._enroll_gallery()
        return len(self._gallery)

    # ------------------------------------------------------------------ #
    # Embedding / matching
    # ------------------------------------------------------------------ #

    def _embed_first_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Embed the largest face in ``image``, or ``None`` if no face passes
        the ``min_face_size`` filter."""
        faces = self._detect_faces(image)
        if not faces:
            return None
        return self._face_embedding(faces[0], image)

    def _detect_faces(self, image: np.ndarray) -> List[Any]:
        """Run InsightFace detection on a BGR image, largest face first."""
        if self._app is None:
            return []
        try:
            faces = self._app.get(image)
        except Exception as exc:
            raise FaceRecognizerError(f"InsightFace detection failed: {exc}") from exc
        faces = [f for f in faces if f.kps is not None or f.bbox is not None]
        faces.sort(
            key=lambda f: float(f.bbox[2] - f.bbox[0]) * float(f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces

    @staticmethod
    def _face_embedding(face: Any, image: np.ndarray) -> Optional[np.ndarray]:
        """Return the ArcFace embedding for a detected face (512-d, or None)."""
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            return None
        return np.asarray(embedding, dtype=np.float32)

    @staticmethod
    def _clip_region_to_frame(
        frame: np.ndarray, region: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[slice, slice]:
        """Convert an ``(x1, y1, x2, y2)`` region into numpy slices."""
        height, width = frame.shape[:2]
        if region is None:
            return slice(0, height), slice(0, width)
        x1, y1, x2, y2 = region
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(x1 + 1, min(int(x2), width))
        y2 = max(y1 + 1, min(int(y2), height))
        return slice(y1, y2), slice(x1, x2)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> FaceRecognitionConfig:
        """Active recognition settings."""
        return self._cfg

    @property
    def enrolled_drivers(self) -> List[str]:
        """Driver ids currently in the embedding gallery."""
        return self._gallery.drivers()

    def identify(
        self,
        frame: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> FaceMatch:
        """Identify the most prominent face in ``frame``.

        Args:
            frame: BGR numpy image (full camera frame).
            region: Optional ``(x1, y1, x2, y2)`` crop (the key-station
                area) to search; coordinates are frame-absolute and the
                returned bbox stays in frame coordinates.

        Returns:
            A :class:`FaceMatch`. When recognition is disabled, InsightFace
            is unavailable, no face is found, or no embedding passes the
            threshold, the match carries the fallback driver id with
            ``matched=False`` (never raises for "no face").
        """
        fallback = FaceMatch(
            driver_id=self._cfg.fallback_driver_id,
            confidence=0.0,
            matched=False,
            bbox=None,
        )
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            LOGGER.warning("identify() received an empty frame; returning fallback.")
            return fallback
        if self._app is None:
            return fallback

        y_slice, x_slice = self._clip_region_to_frame(frame, region)
        crop = frame[y_slice, x_slice]
        offset_x = x_slice.start
        offset_y = y_slice.start

        try:
            faces = self._detect_faces(crop)
        except FaceRecognizerError as exc:
            LOGGER.error("Face detection failed; returning fallback: %s", exc)
            return fallback
        if not faces:
            return fallback

        face = faces[0]  # largest face = closest to the station
        x1, y1, x2, y2 = (int(v) for v in face.bbox[:4])
        width, height = x2 - x1, y2 - y1
        if width < self._cfg.min_face_size or height < self._cfg.min_face_size:
            LOGGER.debug(
                "Face too small (%dx%d < %d px); fallback.",
                width,
                height,
                self._cfg.min_face_size,
            )
            return FaceMatch(
                driver_id=self._cfg.fallback_driver_id,
                confidence=0.0,
                matched=False,
                bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
            )

        embedding = self._face_embedding(face, crop)
        if embedding is None:
            return FaceMatch(
                driver_id=self._cfg.fallback_driver_id,
                confidence=0.0,
                matched=False,
                bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
            )

        best_driver, distance = self._gallery.best_match(embedding)
        if best_driver is None:
            return FaceMatch(
                driver_id=self._cfg.fallback_driver_id,
                confidence=0.0,
                matched=False,
                bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
            )

        confidence = max(0.0, min(1.0, 1.0 - distance))
        if distance <= self._cfg.recognition_threshold:
            return FaceMatch(
                driver_id=best_driver,
                confidence=confidence,
                matched=True,
                bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
            )
        return FaceMatch(
            driver_id=self._cfg.fallback_driver_id,
            confidence=confidence,
            matched=False,
            bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
        )

    def close(self) -> None:
        """Release analyzer resources. InsightFace holds no explicit close,
        so this drops the reference; safe to call repeatedly."""
        self._app = None

    def __enter__(self) -> "FaceRecognizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        state = "disabled" if self._app is None else "ready"
        return f"FaceRecognizer({state}, gallery={len(self._gallery)})"
