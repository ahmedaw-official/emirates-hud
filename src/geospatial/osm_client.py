"""Fast OSM Overpass API client with spatial caching and resilience.

The :class:`OSMPolicyEngine` queries the OpenStreetMap Overpass API for
road-policy information (highway type, posted speed limit) around a GPS
coordinate, applies UAE RTA/ITC regulatory rules, caches results in a
spatial cache (3-meter radius by default), and degrades gracefully when
the network is unavailable.

RTA / ITC enforcement rules
---------------------------

    * If ``maxspeed`` > 60 km/h  -> ``is_prohibited_road = True``
    * If ``highway`` is one of ``footway``, ``pedagian``, ``steps``
      -> ``is_prohibited_road = True``

Fallback policy
---------------

    * **Network error / rate-limit / timeout**: returns a safe default
      with ``is_fallback=True`` and ``maxspeed_kmh=30``.
    * **Missing GPS (lat=0 or lon=0)**: returns a neutral policy with
      ``is_fallback=True`` and no maxspeed constraint.

OpenTelemetry
-------------

Every ``get_road_policy`` call is wrapped in a span named
``geospatial.osm_query`` recording:

    ``osm.cache_hit``           - True if served from spatial cache.
    ``osm.http_status``         - HTTP response status code (0 on error).
    ``osm.latency_ms``          - Network request duration in ms.
    ``osm.fallback_triggered``  - True when a fallback policy was used.
    ``osm.ways_found``          - Count of matching OSM ways returned.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.schemas import OSMRoadPolicy
from src.telemetry import trace_span
from src.telemetry.logging import get_logger

from .spatial_cache import SpatialCache

__all__ = [
    "CACHE_RADIUS_M",
    "DEFAULT_OVERPASS_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "FALLBACK_MAXSPEED_KMH",
    "OSM_QUERY_RADIUS_M",
    "PROHIBITED_HIGHWAY_TYPES",
    "SPEED_LIMIT_THRESHOLD_KMH",
    "OSMPolicyEngine",
]

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"
DEFAULT_TIMEOUT_SECONDS: float = 1.5
OSM_QUERY_RADIUS_M: int = 15
CACHE_RADIUS_M: float = 3.0
SPEED_LIMIT_THRESHOLD_KMH: int = 60
FALLBACK_MAXSPEED_KMH: int = 30
MAXSPEED_PATTERN: re.Pattern[str] = re.compile(r"(\d+(?:\.\d+)?)")
PROHIBITED_HIGHWAY_TYPES: frozenset[str] = frozenset(
    {"footway", "pedestrian", "steps"}
)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class OSMPolicyEngine:
    """Query OSM Overpass API for road policy and enforce RTA/ITC rules.

    Parameters
    ----------
    overpass_url:
        Overpass API endpoint URL.  Defaults to the public
        ``https://overpass-api.de/api/interpreter``.
    timeout:
        Maximum network timeout in seconds (default 1.5).
    cache_radius_meters:
        Radius within which a cached entry is considered a match
        (default 3.0 m).
    fallback_maxspeed_kmh:
        Speed used for the network-error fallback policy (default 30).
    cache:
        Optional pre-existing :class:`SpatialCache` to share across engines.
    """

    def __init__(
        self,
        overpass_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        cache_radius_meters: float = CACHE_RADIUS_M,
        fallback_maxspeed_kmh: int = FALLBACK_MAXSPEED_KMH,
        cache: SpatialCache | None = None,
    ) -> None:
        self._url = overpass_url or DEFAULT_OVERPASS_URL
        self._timeout = timeout
        self._fallback_maxspeed = fallback_maxspeed_kmh
        self._cache = cache or SpatialCache(radius_meters=cache_radius_meters)

    # ------------------------------------------------------------------ public API
    def get_road_policy(self, lat: float, lon: float) -> OSMRoadPolicy:
        """Return the :class:`OSMRoadPolicy` for ``(lat, lon)``.

        Lookup order:

        1. **Spatial cache** - if a nearby entry exists, return it.
        2. **Missing GPS** - if ``lat`` or ``lon`` is ``0.0``, return a
           neutral fallback.
        3. **Overpass API** - query OSM and parse the response.
        4. **Network fallback** - on timeout, HTTP error, or parse
           failure, return a safe default.

        All paths are wrapped in an OTel span ``geospatial.osm_query``.
        """
        return self._get_policy_with_tracing(lat, lon)

    # ------------------------------------------------------------------ internals
    def _get_policy_with_tracing(self, lat: float, lon: float) -> OSMRoadPolicy:
        """Core lookup logic wrapped in an OTel span."""
        with trace_span("geospatial.osm_query") as span:
            # --- 1. Spatial cache check ---
            cached = self._cache.get(lat, lon)
            cache_hit = cached is not None

            if span is not None:
                span.set_attribute("osm.cache_hit", cache_hit)

            if cached is not None:
                return cached

            # --- 2. Missing GPS signal ---
            if lat == 0.0 or lon == 0.0:
                policy = self._neutral_policy(lat, lon)
                self._cache.put(lat, lon, policy)
                if span is not None:
                    span.set_attribute("osm.fallback_triggered", True)
                    span.set_attribute("osm.http_status", 0)
                    span.set_attribute("osm.latency_ms", 0.0)
                    span.set_attribute("osm.ways_found", 0)
                logger.warning(
                    "missing GPS signal, returning neutral policy",
                    extra={"lat": lat, "lon": lon},
                )
                return policy

            # --- 3. Overpass API query ---
            elements, http_status, latency_ms, fallback_triggered = (
                self._query_overpass(lat, lon)
            )

            if span is not None:
                span.set_attribute("osm.http_status", http_status)
                span.set_attribute("osm.latency_ms", round(latency_ms, 3))
                span.set_attribute("osm.ways_found", len(elements))
                span.set_attribute("osm.fallback_triggered", fallback_triggered)

            if fallback_triggered:
                policy = self._fallback_policy(lat, lon)
            else:
                policy = self._parse_overpass_result(elements, lat, lon)

            self._cache.put(lat, lon, policy)
            return policy

    def _query_overpass(
        self, lat: float, lon: float
    ) -> tuple[list[dict[str, Any]], int, float, bool]:
        """Execute the Overpass QL query and return parsed elements.

        Returns
        -------
        tuple
            ``(elements, http_status, latency_ms, fallback_triggered)``
        """
        query = self._build_query(lat, lon)
        start = time.perf_counter()

        try:
            req = urllib.request.Request(
                self._url,
                data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                http_status = resp.status
                body = json.loads(resp.read().decode("utf-8"))
            latency_ms = (time.perf_counter() - start) * 1000.0
            elements = body.get("elements", [])
            return elements, http_status, latency_ms, False

        except urllib.error.HTTPError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning(
                "overpass API HTTP error, using fallback",
                extra={
                    "status": exc.code,
                    "latency_ms": round(latency_ms, 3),
                    "lat": lat,
                    "lon": lon,
                },
            )
            return [], exc.code, latency_ms, True

        except (urllib.error.URLError, TimeoutError) as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning(
                "overpass API network error, using fallback",
                extra={
                    "error": type(exc).__name__,
                    "latency_ms": round(latency_ms, 3),
                    "lat": lat,
                    "lon": lon,
                },
            )
            return [], 0, latency_ms, True

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning(
                "overpass API unexpected error, using fallback",
                extra={
                    "error": str(exc)[:256],
                    "latency_ms": round(latency_ms, 3),
                    "lat": lat,
                    "lon": lon,
                },
            )
            return [], 0, latency_ms, True

    def _build_query(self, lat: float, lon: float) -> str:
        """Build an Overpass QL query for ways within the radius."""
        return (
            f"[out:json][timeout:2];"
            f"(way(around:{OSM_QUERY_RADIUS_M},{lat},{lon})[highway];"
            f"way(around:{OSM_QUERY_RADIUS_M},{lat},{lon})[maxspeed];"
            f");out tags;"
        )

    def _parse_overpass_result(
        self, elements: list[dict[str, Any]], lat: float, lon: float
    ) -> OSMRoadPolicy:
        """Convert Overpass API response elements into an :class:`OSMRoadPolicy`.

        If no elements are found, a fallback policy is returned.  When
        multiple elements are present, the first way with a ``highway`` tag
        is preferred; otherwise the first element with a ``maxspeed`` tag
        is used.
        """
        if not elements:
            logger.info("overpass returned no ways, using fallback")
            return self._fallback_policy(lat, lon)

        # Prefer elements with a highway tag; fall back to any element.
        primary = next(
            (e for e in elements if "highway" in _tags_of(e)),
            elements[0],
        )
        tags = _tags_of(primary)

        way_id = int(primary.get("id", 0))
        highway_type = str(tags.get("highway", "unknown"))
        maxspeed_raw = tags.get("maxspeed")
        maxspeed_kmh = _parse_maxspeed(maxspeed_raw) if maxspeed_raw else None

        # Apply RTA/ITC regulatory rules.
        is_prohibited = self._is_prohibited(highway_type, maxspeed_kmh)

        return OSMRoadPolicy(
            way_id=way_id,
            highway_type=highway_type,
            maxspeed_kmh=maxspeed_kmh,
            is_prohibited_road=is_prohibited,
            query_lat_lon=(lat, lon),
            is_fallback=False,
        )

    @staticmethod
    def _is_prohibited(
        highway_type: str | None, maxspeed_kmh: int | None
    ) -> bool:
        """Apply UAE RTA/ITC regulatory rules."""
        if highway_type and highway_type.lower() in PROHIBITED_HIGHWAY_TYPES:
            return True
        if maxspeed_kmh is not None and maxspeed_kmh > SPEED_LIMIT_THRESHOLD_KMH:
            return True
        return False

    def _fallback_policy(self, lat: float, lon: float) -> OSMRoadPolicy:
        """Return a safe-default policy used when the API fails."""
        return OSMRoadPolicy(
            way_id=0,
            highway_type="fallback",
            maxspeed_kmh=self._fallback_maxspeed,
            is_prohibited_road=False,
            query_lat_lon=(lat, lon),
            is_fallback=True,
        )

    def _neutral_policy(self, lat: float, lon: float) -> OSMRoadPolicy:
        """Return a neutral policy for missing-GPS scenarios."""
        return OSMRoadPolicy(
            way_id=0,
            highway_type="unknown",
            maxspeed_kmh=None,
            is_prohibited_road=False,
            query_lat_lon=(lat, lon),
            is_fallback=True,
        )

    # ------------------------------------------------------------------ accessors
    @property
    def cache(self) -> SpatialCache:
        """The underlying spatial cache."""
        return self._cache

    @property
    def overpass_url(self) -> str:
        """The Overpass API endpoint URL."""
        return self._url

    @property
    def timeout(self) -> float:
        """Network timeout in seconds."""
        return self._timeout


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tags_of(element: dict[str, Any]) -> dict[str, str]:
    """Safely extract the ``tags`` dict from an Overpass element."""
    tags = element.get("tags", {})
    if not isinstance(tags, dict):
        return {}
    return dict(tags)


def _parse_maxspeed(raw: str | None) -> int | None:
    """Parse a maxspeed string from OSM tags into an integer km/h.

    Handles common formats: ``"30"``, ``"30 km/h"``, ``"30;40"``
    (first value wins).  Returns ``None`` for ``"none"`` or unparseable
    values.
    """
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in ("none", "null", ""):
        return None
    # For compound values like "30;40" or "30 mph;40", take the first number.
    match = MAXSPEED_PATTERN.search(raw)
    if match:
        return int(float(match.group(1)))
    return None
