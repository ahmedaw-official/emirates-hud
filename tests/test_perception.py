"""Tests for the perception engine package."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.perception import (
    DEFAULT_MODEL_WEIGHTS,
    DETECTION_CLASSES,
    MockPerceptionEngine,
    PerceptionEngine,
    YOLODatasetManager,
    YOLOPerceptionEngine,
    create_perception_engine,
    train_yolo,
)
from src.schemas import BoundingBox, DetectionResult


# --------------------------------------------------------------------------- #
# Helpers - fake YOLO results
# --------------------------------------------------------------------------- #
class FakeBoxes:
    """Mimics the ``ultralytics.results.Boxes`` interface for testing."""

    def __init__(
        self,
        xyxyn: list[list[float]],
        xyxy: list[list[int]],
        conf: list[float],
        cls: list[int],
    ) -> None:
        if xyxyn:
            self.xyxyn = torch.tensor(xyxyn, dtype=torch.float32)
            self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
            self.conf = torch.tensor(conf, dtype=torch.float32)
            self.cls = torch.tensor(cls, dtype=torch.int64)
        else:
            self.xyxyn = torch.empty((0, 4), dtype=torch.float32)
            self.xyxy = torch.empty((0, 4), dtype=torch.float32)
            self.conf = torch.empty((0,), dtype=torch.float32)
            self.cls = torch.empty((0,), dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.conf)


class FakeResult:
    """Mimics a single ``ultralytics.results.Results`` object."""

    def __init__(self, boxes: FakeBoxes | None, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class FakeYOLO:
    """Stand-in for ``ultralytics.YOLO`` used in tests."""

    #: Class-level results that ``__call__`` returns.
    results: ClassVar[list[FakeResult]] = []
    names: ClassVar[dict[int, str]] = {}

    def __call__(self, frame: np.ndarray, verbose: bool = False, device=None) -> list[FakeResult]:
        return self.results

    def export(self, **kwargs: object) -> str:
        return "fake_export.onnx"


def _make_frame(h: int = 480, w: int = 640, c: int = 3) -> np.ndarray:
    """Create a dummy BGR OpenCV frame."""
    return np.zeros((h, w, c), dtype=np.uint8)


def _make_fake_boxes() -> FakeBoxes:
    """Build fake YOLO boxes for two detections (class 0 and class 3)."""
    return FakeBoxes(
        xyxyn=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        xyxy=[[64, 96, 192, 192], [320, 288, 448, 384]],
        conf=[0.85, 0.55],
        cls=[0, 3],
    )


# --------------------------------------------------------------------------- #
# DetectionResult schema extension
# --------------------------------------------------------------------------- #
class TestDetectionResultSchema:
    def test_low_confidence_defaults_false(self):
        r = DetectionResult(frame_id=1, timestamp_ms=1000, processing_time_ms=5.0)
        assert r.low_confidence is False

    def test_low_confidence_can_be_true(self):
        r = DetectionResult(
            frame_id=1, timestamp_ms=1000, processing_time_ms=0.0, low_confidence=True
        )
        assert r.low_confidence is True

    def test_filter_by_class_preserves_low_confidence(self):
        r = DetectionResult(
            boxes=[
                BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1, confidence=0.5, class_id=0, class_name="scooter"),
                BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1, confidence=0.5, class_id=1, class_name="person"),
            ],
            frame_id=1,
            timestamp_ms=1000,
            processing_time_ms=3.0,
            low_confidence=True,
        )
        filtered = r.filter_by_class("scooter")
        assert filtered.low_confidence is True


# --------------------------------------------------------------------------- #
# MockPerceptionEngine
# --------------------------------------------------------------------------- #
class TestMockPerceptionEngine:
    def test_detect_frame_returns_empty_low_confidence(self):
        engine = MockPerceptionEngine()
        result = engine.detect_frame(_make_frame(), frame_id=42)
        assert isinstance(result, DetectionResult)
        assert result.boxes == []
        assert result.low_confidence is True
        assert result.frame_id == 42

    def test_is_mock_is_true(self):
        assert MockPerceptionEngine().is_mock is True

    def test_frame_corruption_safety_none(self):
        engine = MockPerceptionEngine()
        result = engine.detect_frame(None, frame_id=1)  # type: ignore[arg-type]
        assert result.boxes == []
        assert result.low_confidence is True

    def test_frame_corruption_safety_empty(self):
        engine = MockPerceptionEngine()
        result = engine.detect_frame(np.array([]), frame_id=1)
        assert result.boxes == []
        assert result.low_confidence is True

    def test_frame_corruption_safety_invalid_shape(self):
        engine = MockPerceptionEngine()
        result = engine.detect_frame(np.zeros((0, 10, 3), dtype=np.uint8), frame_id=1)
        assert result.boxes == []
        assert result.low_confidence is True

    def test_records_otel_span(self, spans):
        engine = MockPerceptionEngine()
        engine.detect_frame(_make_frame(), frame_id=1)
        finished = spans.get_finished_spans()
        assert any(s.name == "yolo.detect_frame" for s in finished)


# --------------------------------------------------------------------------- #
# YOLOPerceptionEngine
# --------------------------------------------------------------------------- #
class TestYOLOPerceptionEngine:
    def _make_engine(self, fake_results: list[FakeResult]) -> YOLOPerceptionEngine:
        """Create a YOLOPerceptionEngine with a mocked model."""
        FakeYOLO.results = fake_results
        FakeYOLO.names = {0: "red_track", 1: "grey_sidewalk", 2: "zebra_crosswalk",
                          3: "sign_no_scooter", 4: "sign_speed_limit"}
        with patch("ultralytics.YOLO", return_value=FakeYOLO()):
            return YOLOPerceptionEngine(custom_weights="best.pt")

    def test_detect_frame_returns_valid_boxes(self):
        names = {0: "red_track", 1: "grey_sidewalk", 2: "zebra_crosswalk",
                 3: "sign_no_scooter", 4: "sign_speed_limit"}
        results = [FakeResult(_make_fake_boxes(), names)]
        engine = self._make_engine(results)

        result = engine.detect_frame(_make_frame(), frame_id=10)

        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 2
        assert result.boxes[0].class_name == "red_track"
        assert result.boxes[0].confidence == pytest.approx(0.85)
        assert result.boxes[1].class_name == "sign_no_scooter"
        assert result.frame_id == 10
        assert result.processing_time_ms >= 0.0

    def test_detect_frame_invalid_frame_returns_empty(self):
        engine = self._make_engine([])
        result = engine.detect_frame(None, frame_id=1)  # type: ignore[arg-type]
        assert result.boxes == []
        assert result.low_confidence is True

    def test_detect_frame_empty_frame(self):
        engine = self._make_engine([])
        result = engine.detect_frame(np.zeros((0, 10, 3), dtype=np.uint8), frame_id=1)
        assert result.boxes == []
        assert result.low_confidence is True

    def test_detect_frame_invalid_ndim(self):
        engine = self._make_engine([])
        result = engine.detect_frame(np.array([[[1]]], dtype=np.uint8), frame_id=1)
        assert result.boxes == []
        assert result.low_confidence is True

    def test_detect_frame_confidence_threshold_filters(self):
        names = {0: "red_track"}
        results = [FakeResult(FakeBoxes(
            xyxyn=[[0.1, 0.1, 0.5, 0.5]],
            xyxy=[[64, 64, 320, 320]],
            conf=[0.3],
            cls=[0],
        ), names)]
        engine = self._make_engine(results)
        # Default threshold is 0.40, so 0.3 should be filtered out.
        result = engine.detect_frame(_make_frame(), frame_id=1)
        assert result.boxes == []

    def test_detect_frame_custom_threshold_keeps_low_conf(self):
        names = {0: "red_track"}
        results = [FakeResult(FakeBoxes(
            xyxyn=[[0.1, 0.1, 0.5, 0.5]],
            xyxy=[[64, 64, 320, 320]],
            conf=[0.3],
            cls=[0],
        ), names)]
        with patch("ultralytics.YOLO", return_value=FakeYOLO()):
            FakeYOLO.results = results
            FakeYOLO.names = {0: "red_track"}
            engine = YOLOPerceptionEngine(custom_weights="best.pt", confidence_threshold=0.2)
        result = engine.detect_frame(_make_frame(), frame_id=1)
        assert len(result.boxes) == 1
        assert result.boxes[0].class_name == "red_track"

    def test_detect_frame_no_detections(self):
        names = {0: "red_track"}
        results = [FakeResult(FakeBoxes([], [], [], []), names)]
        engine = self._make_engine(results)
        result = engine.detect_frame(_make_frame(), frame_id=1)
        assert result.boxes == []
        assert result.low_confidence is False  # valid frame, no detections, not mock

    def test_detect_frame_records_otel_attributes(self, spans):
        names = {0: "red_track", 1: "grey_sidewalk", 2: "zebra_crosswalk",
                 3: "sign_no_scooter", 4: "sign_speed_limit"}
        results = [FakeResult(_make_fake_boxes(), names)]
        engine = self._make_engine(results)

        engine.detect_frame(_make_frame(), frame_id=7)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "yolo.detect_frame")
        assert span.attributes.get("detection.count") == 2
        assert span.attributes.get("frame.dimensions") == "640x480x3"
        classes = span.attributes.get("detection.classes")
        assert set(classes) == {"red_track", "sign_no_scooter"}
        latency = span.attributes.get("inference.latency_ms")
        assert latency is not None and latency >= 0

    def test_detect_frame_grayscale_frame(self):
        names = {0: "red_track"}
        results = [FakeResult(_make_fake_boxes(), names)]
        engine = self._make_engine(results)
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        result = engine.detect_frame(gray_frame, frame_id=1)
        assert len(result.boxes) == 2

    def test_fallback_to_default_weights_on_corrupt_custom(self):
        """When custom weights fail to load, the engine falls back to default."""
        with patch("ultralytics.YOLO", side_effect=[
            RuntimeError("corrupt"),
            FakeYOLO(),
        ]):
            FakeYOLO.results = []
            FakeYOLO.names = {}
            engine = YOLOPerceptionEngine(custom_weights="best.pt")

        assert engine.uses_default_weights is True
        assert isinstance(engine.model, FakeYOLO)

    def test_fallback_logs_warning(self, caplog):
        """A warning is logged when falling back from custom to default weights."""
        with patch("ultralytics.YOLO", side_effect=[
            RuntimeError("corrupt"),
            FakeYOLO(),
        ]):
            FakeYOLO.results = []
            FakeYOLO.names = {}
            engine = YOLOPerceptionEngine(custom_weights="best.pt")

        assert engine.uses_default_weights is True
        assert any("falling back to default" in rec.message for rec in caplog.records)

    def test_is_mock_false(self):
        with patch("ultralytics.YOLO", return_value=FakeYOLO()):
            FakeYOLO.results = []
            FakeYOLO.names = {}
            engine = YOLOPerceptionEngine()
        assert engine.is_mock is False

    def test_create_engine_with_mock_model(self):
        """YOLOPerceptionEngine can be constructed with a pre-loaded fake model."""
        with patch("ultralytics.YOLO", return_value=FakeYOLO()):
            FakeYOLO.results = []
            FakeYOLO.names = {}
            engine = YOLOPerceptionEngine()
            assert engine.model is not None


# --------------------------------------------------------------------------- #
# create_perception_engine factory
# --------------------------------------------------------------------------- #
class TestCreatePerceptionEngine:
    def test_returns_yolo_engine_when_available(self):
        with patch("ultralytics.YOLO", return_value=FakeYOLO()):
            FakeYOLO.results = []
            FakeYOLO.names = {}
            engine = create_perception_engine()
        assert isinstance(engine, YOLOPerceptionEngine)
        assert engine.is_mock is False

    def test_falls_back_to_mock_on_import_error(self):
        with patch("ultralytics.YOLO", side_effect=ImportError("no torch")):
            engine = create_perception_engine()
        assert isinstance(engine, MockPerceptionEngine)
        assert engine.is_mock is True

    def test_falls_back_to_mock_on_runtime_error(self):
        with patch("ultralytics.YOLO", side_effect=RuntimeError("corrupt")):
            engine = create_perception_engine()
        assert isinstance(engine, MockPerceptionEngine)
        assert engine.is_mock is True

    def test_custom_weights_passed_through(self):
        with patch("ultralytics.YOLO", return_value=FakeYOLO()) as mock_yolo:
            FakeYOLO.results = []
            FakeYOLO.names = {}
            create_perception_engine(custom_weights="./my_weights.pt")
            # Verify the custom weights path was used.
            mock_yolo.assert_called_with("./my_weights.pt")

    def test_mock_engine_works_after_fallback(self):
        with patch("ultralytics.YOLO", side_effect=ImportError("no ultralytics")):
            engine = create_perception_engine()
        result = engine.detect_frame(_make_frame(), frame_id=1)
        assert result.boxes == []
        assert result.low_confidence is True


# --------------------------------------------------------------------------- #
# DETECTION_CLASSES constant
# --------------------------------------------------------------------------- #
class TestDetectionClasses:
    def test_contains_expected_classes(self):
        expected = {"red_track", "grey_sidewalk", "zebra_crosswalk",
                    "sign_no_scooter", "sign_speed_limit"}
        assert set(DETECTION_CLASSES) == expected

    def test_default_model_weights(self):
        assert DEFAULT_MODEL_WEIGHTS == "yolov11n.pt"

    def test_perception_engine_is_abstract(self):
        with pytest.raises(TypeError):
            PerceptionEngine()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# YOLODatasetManager (local)
# --------------------------------------------------------------------------- #
class TestYOLODatasetManagerLocal:
    @pytest.fixture
    def local_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal local dataset in a temp directory."""
        ds_dir = tmp_path / "my_dataset"
        images_train = ds_dir / "images" / "train"
        images_val = ds_dir / "images" / "val"
        labels_train = ds_dir / "labels" / "train"
        labels_val = ds_dir / "labels" / "val"
        for d in [images_train, images_val, labels_train, labels_val]:
            d.mkdir(parents=True)

        # Fake images
        (images_train / "img1.jpg").write_bytes(b"fake")
        (images_train / "img2.png").write_bytes(b"fake")
        (images_val / "val1.jpg").write_bytes(b"fake")

        # Fake data.yaml
        (ds_dir / "data.yaml").write_text(
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: red_track\n"
            "  1: grey_sidewalk\n"
        )
        return ds_dir

    def test_init_from_local_dir(self, local_dataset: Path):
        manager = YOLODatasetManager(local_dataset)
        assert manager.dataset_dir == str(local_dataset)

    def test_get_data_yaml(self, local_dataset: Path):
        manager = YOLODatasetManager(local_dataset)
        yaml_path = manager.get_data_yaml()
        assert yaml_path.endswith("data.yaml")
        assert Path(yaml_path).exists()

    def test_get_info(self, local_dataset: Path):
        manager = YOLODatasetManager(local_dataset)
        info = manager.get_info()
        assert info.num_train_images == 2
        assert info.num_val_images == 1
        assert info.classes == ["red_track", "grey_sidewalk"]

    def test_get_info_empty_dataset(self, tmp_path: Path):
        ds_dir = tmp_path / "empty"
        ds_dir.mkdir()
        (ds_dir / "data.yaml").write_text("names: []\n")
        manager = YOLODatasetManager(ds_dir)
        info = manager.get_info()
        assert info.num_train_images == 0
        assert info.num_val_images == 0
        assert info.classes == []

    def test_invalid_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            YOLODatasetManager(tmp_path / "nonexistent")

    def test_get_data_yaml_missing(self, tmp_path: Path):
        ds_dir = tmp_path / "no_yaml"
        ds_dir.mkdir()
        manager = YOLODatasetManager(ds_dir)
        with pytest.raises(FileNotFoundError, match=r"data\.yaml"):
            manager.get_data_yaml()


# --------------------------------------------------------------------------- #
# train_yolo error handling
# --------------------------------------------------------------------------- #
class TestTrainYolo:
    def test_raises_without_ultralytics(self):
        """train_yolo raises a helpful error when ultralytics is unavailable."""
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ultralytics":
                raise ImportError("No module named 'ultralytics'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="Ultralytics is required"):
                train_yolo(data="dummy.yaml", epochs=1)

    @pytest.mark.parametrize("fmt", ["onnx", "engine", "coreml", "torchscript"])
    def test_export_model_raises_without_ultralytics(self, fmt):
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ultralytics":
                raise ImportError("No module named 'ultralytics'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            from src.perception.dataset import export_model

            with pytest.raises(ImportError, match="Ultralytics is required"):
                export_model("best.pt", format=fmt)
