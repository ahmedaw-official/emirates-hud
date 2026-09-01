"""Schema package: type-safe, validated Pydantic v2 models.

Public API - every model and enum used by the rest of the engine is
re-exported here so callers can import from a single location::

    from src.schemas import BoundingBox, FrameState, RiskLevel
"""

from __future__ import annotations

from .compliance import VLMAuditResponse
from .enums import (
    ComplianceState,
    RiskLevel,
    ViolationType,
    ZoneClassification,
)
from .frame import FrameState
from .geospatial import OSMRoadPolicy, SpatialCoordinate, ViolationDetail
from .perception import BoundingBox, DetectionResult

__all__ = [
    "BoundingBox",
    "ComplianceState",
    "DetectionResult",
    "FrameState",
    "OSMRoadPolicy",
    "RiskLevel",
    "SpatialCoordinate",
    "VLMAuditResponse",
    "ViolationDetail",
    "ViolationType",
    "ZoneClassification",
]

# Eagerly build nested models so that any forward-reference / Annotated
# quirks surface at import time rather than at first use.
FrameState.model_rebuild()
VLMAuditResponse.model_rebuild()
OSMRoadPolicy.model_rebuild()
