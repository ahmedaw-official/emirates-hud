"""Fast perception engine backed by Ultralytics YOLOv11.

The module provides a robust, high-throughput wrapper around the YOLOv11
inference pipeline that processes BGR OpenCV frames and emits validated
:class:`~src.schemas.DetectionResult` models.

Architecture & fallback chain
-----------------------------

    YOLOPerceptionEngine  - real engine, custom weights → default weights
    └─ __init__ tries *best.pt* first; on failure logs an OTEL warning and
       falls back to *yolov11n.pt* (COCO pretrained).

    create_perception_engine - factory that wraps the above and, if even the
    default model cannot load (PyTorch / Ultralytics missing or unsupported
    hardware), transparently returns a :class:`MockPerceptionEngine`.

Classes detected
----------------

The e-scooter commute domain uses five canonical classes:

    ``red_track``         - red tracking tape / lane delineator
    ``grey_sidewalk``     - grey sidewalk surface
    ``zebra_crosswalk``   - zebra crossing pattern
    ``sign_no_scooter``   - "no scooter" regulatory sign
    ``sign_speed_limit``  - speed-limit sign

These are the classes our custom-trained ``best.pt`` model is expected to
report.  When the default COCO model is used as a fallback, whatever classes
the model produces are passed through verbatim.

OpenTelemetry tracing
---------------------

Every ``detect_frame`` call is wrapped in a span named
``yolo.detect_frame`` with the following attributes:

    ``inference.latency_ms``   - wall-clock inference time in ms
    ``detection.count``        - boxes above the confidence threshold
    ``detection.classes``      - sorted list of unique class names
    ``frame.dimensions``       - "WxHxC" string
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.schemas import BoundingBox, DetectionResult
from src.telemetry import trace_span
from src.telemetry.logging import get_logger

__all__ = [
    "DEFAULT_MODEL_WEIGHTS",
    "DETECTION_CLASSES",
    "MockPerceptionEngine",
    "PerceptionEngine",
    "YOLOPerceptionEngine",
    "create_perception_engine",
]

logger = get_logger(__name__)

#: Canonical detection classes for the e-scooter commute HUD.
DETECTION_CLASSES: tuple[str, ...] = (
    "red_track",
    "grey_sidewalk",
    "zebra_crosswalk",
    "sign_no_scooter",
    "sign_speed_limit",
)

#: Standard pretrained YOLOv11n weights (COCO) used as the last-resort model.
DEFAULT_MODEL_WEIGHTS: str = "yolov11n.pt"

#: Minimum image dimension for a frame to be considered valid.
_MIN_FRAME_DIM: int = 1

#: Low-confidence sentinel value (never returned by a real model, only by mocks).
_MOCK_CONFIDENCE: float = 0.01


class PerceptionEngine(ABC):
    """Abstract base class for all perception engines.

    Subclasses implement :meth:`detect_frame` and expose an :attr:`is_mock`
    flag so downstream consumers can gauge the trustworthiness of results.
    """

    @abstractmethod
    def detect_frame(self, frame: np.ndarray, frame_id: int) -> DetectionResult:
        """Run detection on a single BGR OpenCV frame.

        Parameters
        ----------
        frame:
            BGR image as a ``numpy.ndarray`` (H x W x C).  Grayscale
            (H x W) is accepted; it is treated as single-channel.
        frame_id:
            Monotonic frame identifier passed through to the result.

        Returns
        -------
        DetectionResult
            Validated detection output.  May be empty (zero boxes) when
            the frame is invalid or no objects are above the confidence
            threshold.
        """
        ...

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """``True`` for the zero-dependency mock engine, ``False`` otherwise."""
        ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_valid_frame(frame: Any) -> bool:
    """Return ``True`` when *frame* looks like a usable image array."""
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0:
        return False
    if frame.ndim not in (2, 3):
        return False
    if _MIN_FRAME_DIM >= frame.shape[0] or _MIN_FRAME_DIM >= frame.shape[1]:
        return False
    return True


def _empty_result(frame_id: int, *, low_confidence: bool = True) -> DetectionResult:
    """Build an empty :class:`DetectionResult` with the current timestamp."""
    return DetectionResult(
        boxes=[],
        frame_id=frame_id,
        timestamp_ms=int(time.time() * 1000),
        processing_time_ms=0.0,
        low_confidence=low_confidence,
    )


def _frame_dimensions(frame: np.ndarray) -> str:
    """Return a ``"WxHxC"`` string for the given frame."""
    if frame.ndim == 3:
        h, w, c = frame.shape
    else:
        h, w = frame.shape
        c = 1
    return f"{w}x{h}x{c}"


# --------------------------------------------------------------------------- #
# Real engine
# --------------------------------------------------------------------------- #
class YOLOPerceptionEngine(PerceptionEngine):
    """Perception engine backed by Ultralytics YOLOv11.

    Parameters
    ----------
    custom_weights:
        Path/identifier to custom-trained weights (e.g. ``"./best.pt"``).
        When ``None`` or empty, the default ``yolov11n.pt`` is loaded
        directly.
    confidence_threshold:
        Minimum confidence to keep a detection (default 0.40, matching
        :class:`~src.config.Settings`).
    device:
        Optional device string passed through to ``YOLO.predict`` (e.g.
        ``"cpu"``, ``"cuda:0"``).  ``None`` lets Ultralytics auto-detect.
    """

    def __init__(
        self,
        custom_weights: str | None = None,
        confidence_threshold: float = 0.40,
        device: str | None = None,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model, self._is_default_model = self._load_model(custom_weights)

    # ------------------------------------------------------------------ model loading
    def _load_model(self, custom_weights: str | None) -> tuple[Any, bool]:
        """Load the YOLO model with a fallback to default weights.

        1. If *custom_weights* is provided, try loading it.  On failure,
           log an OTEL warning and fall back to ``yolov11n.pt``.
        2. If no custom weights are specified, load ``yolov11n.pt``.

        Returns
        -------
        tuple
            ``(model, used_default_weights)`` where *used_default_weights*
            is ``True`` when the default model was loaded (either directly
            or via a fallback from corrupt custom weights).

        Raises
        ------
        ImportError
            If the ``ultralytics`` package is not installed.
        RuntimeError
            If neither the custom nor the default model can be loaded.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise ImportError(
                "Ultralytics is not installed. Install with `pip install ultralytics` "
                "or use MockPerceptionEngine via `create_perception_engine()`."
            ) from exc

        default = DEFAULT_MODEL_WEIGHTS

        if custom_weights and custom_weights != default:
            try:
                logger.warning(
                    "loading custom weights",
                    extra={"weights": custom_weights},
                )
                return YOLO(custom_weights), False
            except Exception as exc:
                # Primary fallback: try default weights.
                logger.warning(
                    "custom weights failed, falling back to default",
                    extra={
                        "weights": custom_weights,
                        "fallback": default,
                        "error": str(exc)[:256],
                    },
                )
                try:
                    return YOLO(default), True
                except Exception as fallback_exc:  # pragma: no cover
                    raise RuntimeError(
                        f"Failed to load custom weights ({custom_weights}) and default "
                        f"weights ({default}): {fallback_exc}"
                    ) from fallback_exc

        # No custom weights - load the default directly.
        try:
            return YOLO(default), True
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load default model ({default}): {exc}") from exc

    # ------------------------------------------------------------------ inference
    def detect_frame(self, frame: np.ndarray, frame_id: int) -> DetectionResult:
        """Run YOLOv11 detection on a BGR OpenCV frame.

        The method is wrapped in an OTel span ``yolo.detect_frame`` and
        records ``inference.latency_ms``, ``detection.count``,
        ``detection.classes`` and ``frame.dimensions`` attributes.

        Frame-corruption safety: if *frame* is ``None``, empty, or has an
        invalid shape, an empty :class:`DetectionResult` with
        ``low_confidence=True`` is returned instead of raising.
        """
        # --- Frame corruption safety (before entering the span) ---
        if not _is_valid_frame(frame):
            logger.warning(
                "invalid frame received, returning empty result",
                extra={"frame_id": frame_id, "low_confidence": True},
            )
            return _empty_result(frame_id, low_confidence=True)

        dimensions = _frame_dimensions(frame)

        with trace_span("yolo.detect_frame", attributes={"frame.dimensions": dimensions}):
            start = time.perf_counter()
            results = self._model(frame, verbose=False, device=self._device)
            latency_ms = (time.perf_counter() - start) * 1000.0

            boxes = self._parse_results(results)

            # Record OTel span attributes.
            span = _get_current_span()
            if span is not None:
                span.set_attribute("inference.latency_ms", round(latency_ms, 3))
                span.set_attribute("detection.count", len(boxes))
                span.set_attribute(
                    "detection.classes",
                    sorted({b.class_name for b in boxes}),
                )

            return DetectionResult(
                boxes=boxes,
                frame_id=frame_id,
                timestamp_ms=int(time.time() * 1000),
                processing_time_ms=round(latency_ms, 3),
                low_confidence=self._is_default_model,
            )

    def _parse_results(self, results: Any) -> list[BoundingBox]:
        """Convert raw YOLOv11 ``Results`` into validated ``BoundingBox`` objects.

        Normalised coordinates (required by :class:`BoundingBox`) are
        obtained from ``boxes.xyxyn`` (0..1), pixel coordinates from
        ``boxes.xyxy``.  Detections below the confidence threshold are
        discarded.
        """
        parsed: list[BoundingBox] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Normalised coords: (n, 4) tensor  [x1, y1, x2, y2] in [0, 1]
            xyxyn = result.boxes.xyxyn
            # Pixel coords: (n, 4) tensor  [x1, y1, x2, y2]
            xyxy = result.boxes.xyxy
            confs = result.boxes.conf
            cls_ids = result.boxes.cls
            names = result.names  # dict[int, str]

            for i in range(len(cls_ids)):
                confidence = float(confs[i].item())
                if confidence < self._confidence_threshold:
                    continue

                cls_id = int(cls_ids[i].item())
                class_name = str(names.get(cls_id, f"class_{cls_id}"))

                nx = float(xyxyn[i][0].item())
                ny = float(xyxyn[i][1].item())
                nx2 = float(xyxyn[i][2].item())
                ny2 = float(xyxyn[i][3].item())

                x1_px = int(xyxy[i][0].item())
                y1_px = int(xyxy[i][1].item())
                x2_px = int(xyxy[i][2].item())
                y2_px = int(xyxy[i][3].item())

                parsed.append(
                    BoundingBox(
                        xmin=nx,
                        ymin=ny,
                        xmax=nx2,
                        ymax=ny2,
                        x_min_px=x1_px,
                        y_min_px=y1_px,
                        x_max_px=x2_px,
                        y_max_px=y2_px,
                        confidence=confidence,
                        class_id=cls_id,
                        class_name=class_name,
                    )
                )

        return parsed

    @property
    def is_mock(self) -> bool:
        """Always ``False`` for the real engine."""
        return False

    @property
    def model(self) -> Any:
        """Expose the underlying YOLO model (for advanced use / testing)."""
        return self._model

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence threshold for keeping detections."""
        return self._confidence_threshold

    @property
    def uses_default_weights(self) -> bool:
        """``True`` when the engine is running on the fallback default model."""
        return self._is_default_model


# --------------------------------------------------------------------------- #
# Mock engine
# --------------------------------------------------------------------------- #
class MockPerceptionEngine(PerceptionEngine):
    """Zero-dependency perception engine for degraded environments.

    When PyTorch / Ultralytics is unavailable, or the hardware is unsupported,
    this mock returns a minimal, non-blocking :class:`DetectionResult` with
    ``low_confidence=True`` so downstream rendering pipelines never crash.

    The mock never produces real bounding boxes; it returns an empty result
    with a single sentinel attribute signalling that the caller should treat
    the output with caution.
    """

    MOCK_CONFIDENCE: float = _MOCK_CONFIDENCE

    def __init__(self, confidence_threshold: float = 0.40) -> None:
        self._confidence_threshold = confidence_threshold

    def detect_frame(self, frame: np.ndarray, frame_id: int) -> DetectionResult:
        """Return an empty, low-confidence :class:`DetectionResult`.

        If the frame is corrupt, the result is still returned (never raised)
        but with ``low_confidence=True`` and zero processing time.
        """
        if not _is_valid_frame(frame):
            logger.warning(
                "mock engine received invalid frame",
                extra={"frame_id": frame_id},
            )

        with trace_span("yolo.detect_frame", attributes={"engine": "mock"}):
            return DetectionResult(
                boxes=[],
                frame_id=frame_id,
                timestamp_ms=int(time.time() * 1000),
                processing_time_ms=0.0,
                low_confidence=True,
            )

    @property
    def is_mock(self) -> bool:
        """Always ``True`` for the mock engine."""
        return True

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_perception_engine(
    custom_weights: str | None = None,
    confidence_threshold: float = 0.40,
    device: str | None = None,
) -> PerceptionEngine:
    """Create a perception engine with the full fallback chain.

    1. Attempt :class:`YOLOPerceptionEngine` with *custom_weights*.
    2. If that raises (missing Ultralytics / corrupt model / unsupported
       hardware), log an OTEL warning and return :class:`MockPerceptionEngine`.

    Parameters
    ----------
    custom_weights:
        Path to custom weights (e.g. ``"./best.pt"``).  ``None`` loads the
        default ``yolov11n.pt`` inside the engine.
    confidence_threshold:
        Minimum detection confidence.
    device:
        Optional device override for ``YOLO.predict``.

    Returns
    -------
    PerceptionEngine
        A real :class:`YOLOPerceptionEngine` when available, otherwise a
        :class:`MockPerceptionEngine`.
    """
    try:
        return YOLOPerceptionEngine(
            custom_weights=custom_weights,
            confidence_threshold=confidence_threshold,
            device=device,
        )
    except (ImportError, RuntimeError) as exc:
        logger.warning(
            "falling back to MockPerceptionEngine",
            extra={"reason": str(exc)[:256], "engine": "mock"},
        )
        return MockPerceptionEngine(confidence_threshold=confidence_threshold)


# --------------------------------------------------------------------------- #
# Internal: span accessor
# --------------------------------------------------------------------------- #
def _get_current_span() -> Any | None:
    """Return the currently-active OTel span, or ``None``.

    This avoids a hard import-time dependency on OpenTelemetry when the
    engine runs in environments without a provider installed.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            return span
    except Exception:  # pragma: no cover - OTel absent
        pass
    return None
