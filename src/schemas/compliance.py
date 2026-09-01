"""VLM (Vision-Language Model) scene-understanding response model.

The VLM ingests the current camera frame and, optionally, the geospatial
road policy and produces a structured audit verdict: which kind of zone the
scooter is in, whether a dismount is mandated, a risk grade, any explicit
violations it has spotted, and the ready-to-render HUD warning text.

This module is intentionally decoupled from the perception and geospatial
modules - it only re-uses :class:`~src.schemas.geospatial.ViolationDetail`
so that a single source of truth for infractions is shared.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import RiskLevel, ZoneClassification
from .geospatial import ViolationDetail

__all__ = ["VLMAuditResponse"]


class VLMAuditResponse(BaseModel):
    """Structured output of a VLM compliance audit over one frame."""

    model_config = ConfigDict(extra="forbid")

    zone_classification: ZoneClassification = Field(
        description="Where the scooter currently is, per the VLM.",
    )
    dismount_required: bool = Field(description="Whether the rider must dismount.")
    risk_level: RiskLevel = Field(description="Qualitative risk grade for this frame.")
    violations: list[ViolationDetail] = Field(
        default_factory=list,
        description="Explicit violations the VLM flagged, if any.",
    )
    hud_warning_text: str = Field(description="Human-facing alert text shown on the HUD.")

    @property
    def has_violations(self) -> bool:
        """``True`` when the VLM reported one or more violations."""
        return len(self.violations) > 0

    @property
    def max_fine_aed(self) -> int:
        """Largest single fine attached to any reported violation (0 if none)."""
        if not self.violations:
            return 0
        return max(v.fine_amount_aed for v in self.violations)
