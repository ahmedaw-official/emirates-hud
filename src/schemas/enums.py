"""Enumerated types shared across the schema package.

Pydantic v2 serialises ``str``-enum members as their value (e.g. ``"LOW"``)
both in model instances and in ``model_dump()`` / ``model_dump_json()``,
while still giving full static type safety. Keeping the enums in a single
module avoids circular imports between the domain model files.
"""

from __future__ import annotations

from enum import Enum


class ZoneClassification(str, Enum):
    """Where the scooter is, from the VLM's scene understanding."""

    DESIGNATED_LANE = "DESIGNATED_LANE"
    PEDESTRIAN_SIDEWALK = "PEDESTRIAN_SIDEWALK"
    CROSSWALK = "CROSSWALK"


class RiskLevel(str, Enum):
    """Risk appetite used to size the HUD alert & fine exposure."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ComplianceState(str, Enum):
    """Aggregate compliance verdict for a single frame."""

    SAFE = "SAFE"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"


class ViolationType(str, Enum):
    """Canonical catalogue of micro-mobility infractions.

    ``ViolationDetail.violation_type`` is declared as ``str`` so that ad-hoc
    local infractions can still be represented; the members below are the
    official RTA catalogue and may be passed straight through::

        ViolationDetail(violation_type=ViolationType.NO_DISMOUNT, ...)
    """

    SIDEWALK_DISOBEDENCE = "SIDEWALK_DISOBEDENCE"
    NO_DISMOUNT = "NO_DISMOUNT"
    PROHIBITED_ROAD_USAGE = "PROHIBITED_ROAD_USAGE"
    EXCEEDING_MAXSPEED = "EXCEEDING_MAXSPEED"
    WRONG_LANE_DIRECTION = "WRONG_LANE_DIRECTION"
    UNATTENDED_PARKING = "UNATTENDED_PARKING"
    PEDESTRIAN_CONFLICT = "PEDESTRIAN_CONFLICT"
    INSUFFICIENT_LIGHTING = "INSUFFICIENT_LIGHTING"


__all__ = [
    "ComplianceState",
    "RiskLevel",
    "ViolationType",
    "ZoneClassification",
]
