"""Unit tests for :mod:`src.face_recognizer`.

InsightFace is an optional heavy dependency: config, gallery, and fallback
tests always run, while the real-model test is skipped automatically when
insightface is absent or its model pack cannot be fetched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.face_recognizer import (  # noqa: E402
    FaceMatch,
    FaceRecognizer,
    FaceRecognizerError,
    FaceRecognitionConfig,
    _Gallery,
)

try:
    from insightface.app import FaceAnalysis  # noqa: F401

    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False


def make_config(tmp_path: Path, **overrides) -> dict:
    """Base config dict with a temp gallery dir."""
    cfg = {
        "face_recognition": {
            "enabled": False,  # keep unit tests off the heavy path by default
            "gallery_dir": str(tmp_path / "faces"),
        }
    }
    cfg["face_recognition"].update(overrides)
    return cfg


class TestConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = FaceRecognitionConfig.from_config({})
        assert cfg.enabled is True
        assert cfg.model_pack == "buffalo_s"
        assert cfg.recognition_threshold == pytest.approx(0.45)
        assert cfg.fallback_driver_id == "UNIDENTIFIED"
        assert cfg.gallery_dir == Path("data/faces")

    def test_overrides(self, tmp_path: Path) -> None:
        cfg = FaceRecognitionConfig.from_config(
            make_config(tmp_path, recognition_threshold=0.3, model_pack="buffalo_l")
        )
        assert cfg.recognition_threshold == pytest.approx(0.3)
        assert cfg.model_pack == "buffalo_l"

    @pytest.mark.parametrize(
        "key, value, match",
        [
            ("recognition_threshold", 1.5, "recognition_threshold"),
            ("min_face_size", 4, "min_face_size"),
            ("fallback_driver_id", "", "fallback_driver_id"),
        ],
    )
    def test_invalid_values_raise(
        self, tmp_path: Path, key: str, value, match: str
    ) -> None:
        with pytest.raises(Exception, match=match):
            FaceRecognitionConfig.from_config(make_config(tmp_path, **{key: value}))


class TestGallery:
    def test_add_and_best_match(self) -> None:
        gallery = _Gallery()
        base = np.ones(512, dtype=np.float32)
        gallery.add("Driver_101", base)
        gallery.add("Driver_102", -base)  # orthogonal-ish opposite vector

        driver, distance = gallery.best_match(base)
        assert driver == "Driver_101"
        assert distance == pytest.approx(0.0, abs=1e-5)

        driver, distance = gallery.best_match(-base)
        assert driver == "Driver_102"
        assert distance == pytest.approx(0.0, abs=1e-5)

    def test_distance_ranges(self) -> None:
        gallery = _Gallery()
        vec_a = np.zeros(512, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b = np.zeros(512, dtype=np.float32)
        vec_b[1] = 1.0
        gallery.add("A", vec_a)
        _, distance = gallery.best_match(vec_b)
        assert distance == pytest.approx(1.0, abs=1e-6)  # orthogonal = unrelated

    def test_multiple_photos_per_driver(self) -> None:
        gallery = _Gallery()
        base = np.ones(64, dtype=np.float32)
        gallery.add("Driver_101", base)
        gallery.add("Driver_101", base * 0.5)  # same direction, other magnitude
        driver, distance = gallery.best_match(base)
        assert driver == "Driver_101"
        assert distance == pytest.approx(0.0, abs=1e-5)
        assert len(gallery) == 2

    def test_degenerate_embedding_rejected(self) -> None:
        gallery = _Gallery()
        with pytest.raises(FaceRecognizerError, match="degenerate"):
            gallery.add("X", np.zeros(512, dtype=np.float32))

    def test_empty_gallery_returns_none(self) -> None:
        driver, distance = _Gallery().best_match(np.ones(8, dtype=np.float32))
        assert driver is None
        assert distance == float("inf")

    def test_zero_query_vector(self) -> None:
        gallery = _Gallery()
        gallery.add("A", np.ones(8, dtype=np.float32))
        driver, distance = gallery.best_match(np.zeros(8, dtype=np.float32))
        assert driver is None and distance == float("inf")


class TestDisabledMode:
    """enabled=false must short-circuit to fallback without InsightFace."""

    def test_identify_returns_fallback(self, tmp_path: Path) -> None:
        recognizer = FaceRecognizer(config=make_config(tmp_path))
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        match = recognizer.identify(frame)
        assert match.matched is False
        assert match.driver_id == "UNIDENTIFIED"
        assert match.confidence == 0.0

    def test_empty_frame_falls_back(self, tmp_path: Path) -> None:
        recognizer = FaceRecognizer(config=make_config(tmp_path))
        match = recognizer.identify(np.zeros((0, 0, 3), dtype=np.uint8))
        assert match.matched is False

    def test_none_frame_falls_back(self, tmp_path: Path) -> None:
        recognizer = FaceRecognizer(config=make_config(tmp_path))
        match = recognizer.identify(None)  # type: ignore[arg-type]
        assert match.driver_id == "UNIDENTIFIED"

    def test_custom_fallback_id(self, tmp_path: Path) -> None:
        recognizer = FaceRecognizer(
            config=make_config(tmp_path, fallback_driver_id="UNKNOWN_DRIVER")
        )
        match = recognizer.identify(np.full((60, 60, 3), 80, dtype=np.uint8))
        assert match.driver_id == "UNKNOWN_DRIVER"

    def test_rebuild_gallery_disabled_is_zero(self, tmp_path: Path) -> None:
        recognizer = FaceRecognizer(config=make_config(tmp_path))
        assert recognizer.rebuild_gallery() == 0


class TestRegionClamping:
    def test_region_slices_stay_in_bounds(self, tmp_path: Path) -> None:
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        # Region beyond frame edges must clamp, not crash (disabled mode:
        # falls back immediately after clamping).
        recognizer = FaceRecognizer(config=make_config(tmp_path))
        match = recognizer.identify(frame, region=(300, 200, 900, 900))
        assert match.driver_id == "UNIDENTIFIED"


@pytest.mark.skipif(HAS_INSIGHTFACE, reason="insightface installed")
class TestMissingInsightface:
    def test_enabled_without_library_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FaceRecognizerError, match="pip install"):
            FaceRecognizer(config=make_config(tmp_path, enabled=True))


class FakeFace:
    """Duck-typed stand-in for an insightface Face object."""

    def __init__(self, bbox, embedding):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.kps = np.zeros((5, 2), dtype=np.float32)  # detection marker


class TestIdentifyWithFakeAnalyzer:
    """Patch the analyzer to exercise the full matching path."""

    @pytest.fixture()
    def analyzer(self):
        """A fake FaceAnalysis-like object with configurable faces."""

        class FakeAnalyzer:
            def __init__(self) -> None:
                self.faces: list = []

            def get(self, image):
                return self.faces

        return FakeAnalyzer()

    def _make_recognizer(self, tmp_path: Path, analyzer) -> FaceRecognizer:
        with patch.object(FaceRecognizer, "_build_analyzer", return_value=analyzer):
            recognizer = FaceRecognizer(config=make_config(tmp_path, enabled=True))
        return recognizer

    def test_match_success(self, tmp_path: Path, analyzer) -> None:
        emb = np.ones(512, dtype=np.float32)
        analyzer.faces = [FakeFace((100, 100, 200, 200), emb)]
        recognizer = self._make_recognizer(tmp_path, analyzer)
        # Enroll an identical embedding for a known driver.
        recognizer._gallery.add("Driver_101", emb)

        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        match = recognizer.identify(frame)
        assert match.matched is True
        assert match.driver_id == "Driver_101"
        assert match.confidence == pytest.approx(1.0, abs=1e-5)
        assert match.bbox == (100, 100, 200, 200)

    def test_match_below_threshold_falls_back(self, tmp_path: Path, analyzer) -> None:
        analyzer.faces = [FakeFace((100, 100, 200, 200), np.ones(512, dtype=np.float32))]
        recognizer = self._make_recognizer(
            tmp_path, analyzer
        )
        orthogonal = np.zeros(512, dtype=np.float32)
        orthogonal[0] = 1.0
        recognizer._gallery.add("Driver_101", orthogonal)

        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        match = recognizer.identify(frame)
        assert match.matched is False
        assert match.driver_id == "UNIDENTIFIED"

    def test_no_faces_falls_back(self, tmp_path: Path, analyzer) -> None:
        analyzer.faces = []
        recognizer = self._make_recognizer(tmp_path, analyzer)
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        match = recognizer.identify(frame)
        assert match.matched is False and match.bbox is None

    def test_small_face_falls_back(self, tmp_path: Path, analyzer) -> None:
        analyzer.faces = [FakeFace((10, 10, 30, 30), np.ones(512, dtype=np.float32))]
        recognizer = self._make_recognizer(
            tmp_path, analyzer
        )
        recognizer._gallery.add("Driver_101", np.ones(512, dtype=np.float32))
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        match = recognizer.identify(frame)
        assert match.matched is False  # 20 px < min_face_size (60 default)
        assert match.bbox == (10, 10, 30, 30)

    def test_region_offset_applied_to_bbox(self, tmp_path: Path, analyzer) -> None:
        analyzer.faces = [FakeFace((50, 50, 150, 150), np.ones(512, dtype=np.float32))]
        recognizer = self._make_recognizer(tmp_path, analyzer)
        recognizer._gallery.add("Driver_101", np.ones(512, dtype=np.float32))
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        match = recognizer.identify(frame, region=(20, 20, 400, 400))
        assert match.bbox == (70, 70, 170, 170)  # +20 offset on both axes

    def test_detection_error_falls_back(self, tmp_path: Path, analyzer) -> None:
        def boom(image):
            raise RuntimeError("GPU went away")

        analyzer.get = boom
        recognizer = self._make_recognizer(tmp_path, analyzer)
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        match = recognizer.identify(frame)
        assert match.matched is False  # logged + fallback, never raises


@pytest.mark.skipif(not HAS_INSIGHTFACE, reason="insightface not installed")
class TestRealModel:
    """End-to-end test against the real InsightFace model pack (~100 MB)."""

    def test_synthetic_frame_returns_fallback(self, tmp_path: Path) -> None:
        recognizer = FaceRecognizer(
            config=make_config(
                tmp_path, enabled=True, det_size=[320, 320], min_face_size=16
            )
        )
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        match = recognizer.identify(frame)
        assert match.matched is False  # no face in a blank frame
        assert match.driver_id == "UNIDENTIFIED"
        recognizer.close()
