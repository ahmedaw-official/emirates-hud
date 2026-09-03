"""Geospatial policy engine package.

Provides OpenStreetMap-based road policy lookups with spatial caching,
fallback resilience, and full OpenTelemetry observability.

Public API:

    :class:`OSMPolicyEngine`    - Queries OSM Overpass API for road policy.
    :class:`SpatialCache`       - In-memory spatial cache keyed by GPS coordinate.
    :func:`haversine_m`         - Haversine distance between two coordinates.
"""

from __future__ import annotations

from .osm_client import OSMPolicyEngine
from .spatial_cache import SpatialCache, haversine_m

__all__ = [
    "OSMPolicyEngine",
    "SpatialCache",
    "haversine_m",
]

__version__ = "0.1.0"
