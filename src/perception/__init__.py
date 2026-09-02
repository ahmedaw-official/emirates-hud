"""Perception engine package: YOLOv11-based surface segmentation and target detection.

Public API:

    :class:`PerceptionEngine`      - Abstract base for all perception engines.
    :class:`YOLOPerceptionEngine`  - Real-time engine backed by Ultralytics YOLOv11.
    :class:`MockPerceptionEngine`  - Zero-dependency fallback for degraded environments.
    :func:`create_perception_engine` - Factory with automatic fallback chain.

The engine ingests BGR OpenCV frames (``np.ndarray``) and returns validated
:class:`~src.schemas.DetectionResult` Pydantic models.  Detection classes
mirror the e-scooter commute domain::

    red_track, grey_sidewalk, zebra_crosswalk,
    sign_no_scooter, sign_speed_limit
"""

from __future__ import annotations

from .dataset import DatasetInfo, YOLODatasetManager, export_model, train_yolo
from .yolo_engine import (
    DEFAULT_MODEL_WEIGHTS,
    DETECTION_CLASSES,
    MockPerceptionEngine,
    PerceptionEngine,
    YOLOPerceptionEngine,
    create_perception_engine,
)

__all__ = [
    "DEFAULT_MODEL_WEIGHTS",
    "DETECTION_CLASSES",
    "DatasetInfo",
    "MockPerceptionEngine",
    "PerceptionEngine",
    "YOLODatasetManager",
    "YOLOPerceptionEngine",
    "create_perception_engine",
    "export_model",
    "train_yolo",
]

__version__ = "0.1.0"
