"""In-memory spatial cache for OSM road-policy lookups.

The cache keys cached :class:`~src.schemas.OSMRoadPolicy` objects by GPS
coordinate.  When a new coordinate falls within a configurable radius
(default 3 meters) of a previously cached entry, the cached policy is
returned immediately without hitting the Overpass network API.

Distance is computed using the great-circle (Haversine) formula, which is
accurate enough for the sub-10-meter distances involved in edge-case
scooter tracking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from src.schemas import OSMRoadPolicy
from src.telemetry.logging import get_logger

__all__ = ["SpatialCache", "haversine_m"]

logger = get_logger(__name__)

#: Mean Earth radius in meters (WGS-84).
_EARTH_RADIUS_M: float = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points.

    Parameters
    ----------
    lat1, lon1:
        First point in signed decimal degrees.
    lat2, lon2:
        Second point in signed decimal degrees.

    Returns
    -------
    float
        Distance in meters.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass
class _CacheEntry:
    """A single spatial-cache entry."""

    lat: float
    lon: float
    policy: OSMRoadPolicy
    timestamp: float


class SpatialCache:
    """In-memory cache of :class:`OSMRoadPolicy` objects keyed by coordinate.

    Parameters
    ----------
    radius_meters:
        Maximum distance (in meters) within which a cached entry is
        considered a match for a new query coordinate.  Default 3.0 m.
    max_entries:
        Maximum number of entries to retain.  When exceeded, the oldest
        entry is evicted (FIFO).  Default 1000.
    """

    def __init__(self, radius_meters: float = 3.0, max_entries: int = 1000) -> None:
        if radius_meters < 0:
            raise ValueError("radius_meters must be non-negative")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._radius = radius_meters
        self._max_entries = max_entries
        self._entries: list[_CacheEntry] = []

    # ------------------------------------------------------------------ public
    def get(self, lat: float, lon: float) -> OSMRoadPolicy | None:
        """Return the cached policy if a nearby entry exists, else ``None``.

        "Nearby" means any cached entry whose Haversine distance to
        ``(lat, lon)`` is less than or equal to :attr:`radius_meters`.
        """
        for entry in self._entries:
            dist = haversine_m(lat, lon, entry.lat, entry.lon)
            if dist <= self._radius:
                logger.debug(
                    "spatial cache hit",
                    extra={
                        "lat": lat,
                        "lon": lon,
                        "cached_lat": entry.lat,
                        "cached_lon": entry.lon,
                        "distance_m": round(dist, 4),
                    },
                )
                return entry.policy
        return None

    def put(self, lat: float, lon: float, policy: OSMRoadPolicy) -> None:
        """Store *policy* for coordinate ``(lat, lon)``.

        If the cache is full, the oldest entry is evicted first (FIFO).
        """
        # Evict oldest if at capacity.
        if len(self._entries) >= self._max_entries:
            evicted = self._entries.pop(0)
            logger.debug(
                "cache eviction (FIFO)",
                extra={"evicted_lat": evicted.lat, "evicted_lon": evicted.lon},
            )

        self._entries.append(
            _CacheEntry(lat=lat, lon=lon, policy=policy, timestamp=time.time())
        )

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._entries.clear()

    # ------------------------------------------------------------------ accessors
    @property
    def radius_meters(self) -> float:
        """The distance radius (meters) within which a cache hit is returned."""
        return self._radius

    @property
    def size(self) -> int:
        """Number of entries currently cached."""
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, coord: tuple[float, float]) -> bool:
        """Return ``True`` if a nearby entry exists for ``(lat, lon)``."""
        lat, lon = coord
        return self.get(lat, lon) is not None
