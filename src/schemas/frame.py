"""Aggregate per-frame compliance state.

:class:`FrameState` is the *single source of truth* handed to the HUD
rendering layer: it fuses the perception output (bounding boxes), the
geospatial road policy, and the computed compliance verdict together with
an alert message and an exposed fine ceiling in AED.

Keeping it as a dedicated model - rather than a loose dict passed around
between modules - means the renderer can trust that every required field
is present and valid before it hits the glass.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .compliance import VLMAuditResponse
from .enums import ComplianceState
from .geospatial import OSMRoadPolicy
from .perception import BoundingBox

__all__ = ["FrameState"]


class FrameState(BaseModel):
    """The complete compliance snapshot for a single rendered frame."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(ge=0, description="Unique, monotonic frame identifier.")
    timestamp: datetime = Field(description="UTC moment this frame state was finalised.")
    visual_detections: list[BoundingBox] = Field(
        default_factory=list,
        description="Perception output: the objects detected in the frame.",
    )
    geospatial_policy: OSMRoadPolicy = Field(
        description="Road policy derived from the latest OSM/Overpass query.",
    )
    compliance_audit: VLMAuditResponse | None = Field(
        default=None,
        description="Optional VLM audit verdict that was applied to produce this state.",
    )
    active_compliance_state: ComplianceState = Field(
        description="Aggregate verdict: SAFE, WARNING or VIOLATION.",
    )
    hud_alert_message: str = Field(description="Text to display on the HUD.")
    fine_risk_aed: int = Field(
        default=0,
        ge=0,
        description="Ceiling of the fine exposure (AED) for the current frame.",
    )

    @property
    def is_violation(self) -> bool:
        """Convenience accessor for the most safety-critical frames."""
        return self.active_compliance_state == ComplianceState.VIOLATION
