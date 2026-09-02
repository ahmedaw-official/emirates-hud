"""Perception-layer data models.

These describe the raw material coming out of the YOLOv11 detector:
bounding boxes and the per-frame detection result. All coordinate systems
are validated so downstream compliance logic can trust the numbers it is
handed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = ["BoundingBox", "DetectionResult"]


class BoundingBox(BaseModel):
    """A single detection window.

    The ``xmin``/``ymin``/``xmax``/``ymax`` fields are the *normalised*
    coordinates in the closed interval ``[0.0, 1.0]``. Absolute pixel
    coordinates are exposed through the ``*_px`` fields which are optional
    because the image dimensions are not always known at detection time.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "xmin": 0.12,
                    "ymin": 0.08,
                    "xmax": 0.45,
                    "ymax": 0.91,
                    "confidence": 0.87,
                    "class_id": 0,
                    "class_name": "scooter",
                }
            ]
        },
    )

    # Canonical, normalised box coordinates.
    xmin: float = Field(ge=0.0, le=1.0, description="Left edge, normalised [0,1].")
    ymin: float = Field(ge=0.0, le=1.0, description="Top edge, normalised [0,1].")
    xmax: float = Field(ge=0.0, le=1.0, description="Right edge, normalised [0,1].")
    ymax: float = Field(ge=0.0, le=1.0, description="Bottom edge, normalised [0,1].")

    # Absolute pixel coordinates (only valid when image dimensions are known).
    x_min_px: int | None = Field(default=None, ge=0, description="Left edge in pixels.")
    y_min_px: int | None = Field(default=None, ge=0, description="Top edge in pixels.")
    x_max_px: int | None = Field(default=None, ge=0, description="Right edge in pixels.")
    y_max_px: int | None = Field(default=None, ge=0, description="Bottom edge in pixels.")

    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence.")
    class_id: int = Field(ge=0, description="YOLO class index.")
    class_name: str = Field(min_length=1, description="Human readable class label.")

    @field_validator("class_name", mode="after")
    @classmethod
    def _class_name_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("class_name must not be empty")
        return v

    # ------------------------------------------------------------------ validators
    @model_validator(mode="after")
    def _box_geometry(self) -> BoundingBox:
        if self.xmax < self.xmin:
            raise ValueError("xmax must be >= xmin (normalised coordinates)")
        if self.ymax < self.ymin:
            raise ValueError("ymax must be >= ymin (normalised coordinates)")
        if (
            self.x_min_px is not None
            and self.x_max_px is not None
            and self.x_max_px < self.x_min_px
        ):
            raise ValueError("x_max_px must be >= x_min_px")
        if (
            self.y_min_px is not None
            and self.y_max_px is not None
            and self.y_max_px < self.y_min_px
        ):
            raise ValueError("y_max_px must be >= y_min_px")
        return self

    # ----------------------------------------------------------------- derived
    @property
    def width(self) -> float:
        """Normalised box width ``xmax - xmin``."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Normalised box height ``ymax - ymin``."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Normalised box area (0..1)."""
        return self.width * self.height


class DetectionResult(BaseModel):
    """The complete output of a single perception pass over one frame."""

    model_config = ConfigDict(extra="forbid")

    boxes: list[BoundingBox] = Field(default_factory=list, description="Detected objects.")
    frame_id: int = Field(ge=0, description="Monotonic frame identifier.")
    timestamp_ms: int = Field(ge=0, description="Epoch time in milliseconds.")
    processing_time_ms: float = Field(ge=0.0, description="Per-frame wall-clock processing time (ms).")
    low_confidence: bool = Field(
        default=False,
        description="True when results may be unreliable (empty frame, mock engine, or fallback).",
    )

    @property
    def detection_count(self) -> int:
        """Number of bounding boxes in this result."""
        return len(self.boxes)

    @property
    def max_confidence(self) -> float:
        """Highest confidence among all detections (0.0 if empty)."""
        if not self.boxes:
            return 0.0
        return max(b.confidence for b in self.boxes)

    @property
    def detections_by_class(self) -> dict[str, int]:
        """Count of detections grouped by ``class_name``."""
        counts: dict[str, int] = {}
        for box in self.boxes:
            counts[box.class_name] = counts.get(box.class_name, 0) + 1
        return counts

    # ------------------------------------------------------------------ filters
    def filter_by_class(self, class_names: str | list[str]) -> DetectionResult:
        """Return a new ``DetectionResult`` keeping only the requested classes.

        ``class_names`` may be a single class name or a list of names. The
        returned result preserves the original ``frame_id``, ``timestamp_ms``
        and ``processing_time_ms`` metadata, only the ``boxes`` list is
        filtered. Class matching is case-sensitive and exact.

        Example::

            result = DetectionResult(...)
            scooters = result.filter_by_class("scooter")
            people_and_scooters = result.filter_by_class(["scooter", "person"])
        """
        if isinstance(class_names, str):
            wanted = {class_names}
        else:
            wanted = set(class_names)
        filtered = [b for b in self.boxes if b.class_name in wanted]
        return DetectionResult(
            boxes=filtered,
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp_ms,
            processing_time_ms=self.processing_time_ms,
            low_confidence=self.low_confidence,
        )

    def filter_by_confidence(self, min_confidence: float = 0.0) -> DetectionResult:
        """Return a new ``DetectionResult`` dropping detections below a threshold.

        ``min_confidence`` is the inclusive lower bound for ``confidence``.
        Metadata fields are preserved as with :meth:`filter_by_class`.
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0.0, 1.0]")
        filtered = [b for b in self.boxes if b.confidence >= min_confidence]
        return DetectionResult(
            boxes=filtered,
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp_ms,
            processing_time_ms=self.processing_time_ms,
            low_confidence=self.low_confidence,
        )


__all__ = ["BoundingBox", "DetectionResult"]
