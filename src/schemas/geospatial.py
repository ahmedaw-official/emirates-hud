"""Geospatial and legal-domain data models.

* :class:`SpatialCoordinate` - a single, validated GPS fix.
* :class:`OSMRoadPolicy`  - the result of an OpenStreetMap Overpass query
  describing the road the scooter is currently on.
* :class:`ViolationDetail` - a discrete compliance infraction with a legal
  reference and the associated fine.

These models are intentionally self-contained so they can be reused by the
perception, routing and compliance engines alike.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import ViolationType

__all__ = ["OSMRoadPolicy", "SpatialCoordinate", "ViolationDetail", "ViolationType"]

# A (latitude, longitude) pair in signed degrees.
LatLonPair = Annotated[
    tuple[float, float],
    Field(description="``(latitude, longitude)`` in signed decimal degrees."),
]


class SpatialCoordinate(BaseModel):
    """A single, validated WGS-84 GPS fix."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 25.2048,
                    "longitude": 55.2708,
                    "altitude": 12.4,
                    "accuracy": 2.5,
                    "timestamp": "2026-09-01T13:18:57+04:00",
                }
            ]
        },
    )

    latitude: float = Field(description="Latitude in signed degrees, -90..90.")
    longitude: float = Field(description="Longitude in signed degrees, -180..180.")
    altitude: float = Field(description="Altitude above mean sea level (metres).")
    accuracy: float = Field(ge=0.0, description="Horizontal GPS accuracy (metres).")
    timestamp: datetime = Field(description="UTC time the fix was captured.")

    @field_validator("latitude")
    @classmethod
    def _latitude_range(cls, v: float) -> float:
        if not -90.0 <= v <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        return v

    @field_validator("longitude")
    @classmethod
    def _longitude_range(cls, v: float) -> float:
        if not -180.0 <= v <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")
        return v


class OSMRoadPolicy(BaseModel):
    """Road-policy verdict sourced from OpenStreetMap via Overpass.

    ``query_lat_lon`` captures the exact coordinate the Overpass query was
    issued from so that downstream consumers can audit the decision, while
    ``is_fallback`` flags results produced when the live query failed and a
    cached/derived policy had to be used instead.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "way_id": 123456789,
                    "highway_type": "path",
                    "maxspeed_kmh": 15,
                    "is_prohibited_road": False,
                    "query_lat_lon": (25.2048, 55.2708),
                    "is_fallback": False,
                }
            ]
        },
    )

    way_id: int = Field(ge=0, description="OpenStreetMap way identifier.")
    highway_type: str = Field(min_length=1, description="OSM `highway=*` tag value.")
    maxspeed_kmh: int | None = Field(default=None, ge=0, description="Posted max speed (km/h).")
    is_prohibited_road: bool = Field(description="True if scooters are barred on this road.")
    query_lat_lon: LatLonPair = Field(description="Coordinates the policy query was issued from.")
    is_fallback: bool = Field(default=False, description="True if the policy came from a fallback source.")

    @field_validator("query_lat_lon")
    @classmethod
    def _validate_query_coords(cls, v: tuple[float, float]) -> tuple[float, float]:
        lat, lon = v
        if not -90.0 <= lat <= 90.0:
            raise ValueError("query_lat_lon latitude must be in [-90, 90]")
        if not -180.0 <= lon <= 180.0:
            raise ValueError("query_lat_lon longitude must be in [-180, 180]")
        return v


class ViolationDetail(BaseModel):
    """A single infraction record tied to a legal reference."""

    model_config = ConfigDict(extra="forbid")

    violation_type: str = Field(
        description="Machine-readable infraction class. Prefer a ``ViolationType`` member.",
    )
    legal_reference: str = Field(
        min_length=1,
        description='e.g. "Dubai RTA Resolution No. 13 (2022) Article 4".',
    )
    fine_amount_aed: int = Field(ge=0, description="Fine in UAE Dirhams (AED).")

    @field_validator("violation_type")
    @classmethod
    def _normalize_violation_type(cls, v: str) -> str:
        # Accept either a bare string or a ViolationType member (str subclass).
        if isinstance(v, ViolationType):
            return v.value
        if isinstance(v, str):
            return v.strip()
        return str(v)

    @field_validator("legal_reference", mode="after")
    @classmethod
    def _legal_reference_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("legal_reference must not be empty")
        return v


__all__ = ["OSMRoadPolicy", "SpatialCoordinate", "ViolationDetail", "ViolationType"]
