"""Tests for the geospatial policy engine package."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest

from src.geospatial import OSMPolicyEngine, SpatialCache, haversine_m
from src.geospatial.osm_client import (
    CACHE_RADIUS_M,
    DEFAULT_TIMEOUT_SECONDS,
    FALLBACK_MAXSPEED_KMH,
    OSM_QUERY_RADIUS_M,
    PROHIBITED_HIGHWAY_TYPES,
)
from src.schemas import OSMRoadPolicy


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class MockHTTPResponse:
    """Minimal stand-in for an ``http.client.HTTPResponse``."""

    def __init__(self, data: dict, status: int = 200) -> None:
        self._data = data
        self.status = status
        self._body = json.dumps(data).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> MockHTTPResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class MockHTTPError(Exception):
    """Stand-in for ``urllib.error.HTTPError`` with a ``.code`` attribute."""

    def __init__(self, code: int, reason: str = "") -> None:
        self.code = code
        self.reason = reason
        super().__init__(reason)


def _make_way_element(way_id: int, tags: dict[str, str]) -> dict:
    """Build an Overpass-style element dict."""
    return {"type": "way", "id": way_id, "tags": tags}


def _make_overpass_response(elements: list[dict]) -> dict:
    """Wrap elements in the Overpass JSON envelope."""
    return {"version": 0.6, "elements": elements}


# --------------------------------------------------------------------------- #
# haversine_m
# --------------------------------------------------------------------------- #
class TestHaversine:
    def test_zero_distance_same_point(self):
        assert haversine_m(25.0, 55.0, 25.0, 55.0) == pytest.approx(0.0)

    def test_known_distance(self):
        # Roughly 1 degree of latitude at the equator ~ 111.32 km.
        d = haversine_m(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111_320, rel=0.01)

    def test_symmetric(self):
        d1 = haversine_m(25.1, 55.2, 25.3, 55.8)
        d2 = haversine_m(25.3, 55.8, 25.1, 55.2)
        assert d1 == pytest.approx(d2)

    def test_small_distance_3m(self):
        # ~3 meter difference in latitude at a given lon.
        d = haversine_m(25.2708, 55.2708, 25.2708 + 0.00003, 55.2708)
        assert d < 10  # well within the 3-meter cache radius
        assert d > 0


# --------------------------------------------------------------------------- #
# SpatialCache
# --------------------------------------------------------------------------- #
class TestSpatialCache:
    def test_get_returns_cached_policy_within_radius(self):
        cache = SpatialCache(radius_meters=10.0)
        policy = OSMRoadPolicy(
            way_id=42, highway_type="residential", maxspeed_kmh=30,
            is_prohibited_road=False, query_lat_lon=(25.2708, 55.2708),
        )
        cache.put(25.2708, 55.2708, policy)
        # ~3 meter offset - within 10m radius
        result = cache.get(25.2708 + 0.00003, 55.2708)
        assert result is not None
        assert result.way_id == 42

    def test_get_returns_none_outside_radius(self):
        cache = SpatialCache(radius_meters=3.0)
        policy = OSMRoadPolicy(
            way_id=42, highway_type="residential", maxspeed_kmh=30,
            is_prohibited_road=False, query_lat_lon=(25.2708, 55.2708),
        )
        cache.put(25.2708, 55.2708, policy)
        # ~111 km offset - way outside 3m radius
        result = cache.get(26.2708, 55.2708)
        assert result is None

    def test_empty_cache_returns_none(self):
        cache = SpatialCache()
        assert cache.get(25.0, 55.0) is None

    def test_clear(self):
        cache = SpatialCache(radius_meters=10.0)
        policy = OSMRoadPolicy(
            way_id=1, highway_type="x", maxspeed_kmh=30,
            is_prohibited_road=False, query_lat_lon=(25.0, 55.0),
        )
        cache.put(25.0, 55.0, policy)
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0
        assert cache.get(25.0, 55.0) is None

    def test_max_entries_eviction(self):
        cache = SpatialCache(radius_meters=10.0, max_entries=2)
        p1 = OSMRoadPolicy(way_id=1, highway_type="a", maxspeed_kmh=10,
                           is_prohibited_road=False, query_lat_lon=(25.0, 55.0))
        p2 = OSMRoadPolicy(way_id=2, highway_type="b", maxspeed_kmh=20,
                           is_prohibited_road=False, query_lat_lon=(26.0, 56.0))
        p3 = OSMRoadPolicy(way_id=3, highway_type="c", maxspeed_kmh=30,
                           is_prohibited_road=False, query_lat_lon=(27.0, 57.0))
        cache.put(25.0, 55.0, p1)
        cache.put(26.0, 56.0, p2)
        cache.put(27.0, 57.0, p3)
        assert cache.size == 2
        # p1 should have been evicted (FIFO)
        assert cache.get(25.0, 55.0) is None

    def test_len_and_contains(self):
        cache = SpatialCache(radius_meters=10.0)
        assert len(cache) == 0
        policy = OSMRoadPolicy(
            way_id=1, highway_type="x", maxspeed_kmh=30,
            is_prohibited_road=False, query_lat_lon=(25.0, 55.0),
        )
        cache.put(25.0, 55.0, policy)
        assert len(cache) == 1
        assert (25.0, 55.0) in cache
        assert (26.0, 56.0) not in cache

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SpatialCache(radius_meters=-1.0)

    def test_invalid_max_entries_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            SpatialCache(max_entries=0)

    def test_default_radius_is_3m(self):
        cache = SpatialCache()
        assert cache.radius_meters == CACHE_RADIUS_M


# --------------------------------------------------------------------------- #
# OSMPolicyEngine - cache behavior
# --------------------------------------------------------------------------- #
class TestEngineCache:
    def test_cache_hit_returns_cached_without_network(self):
        """A second call within the cache radius should not hit the network."""
        engine = OSMPolicyEngine(cache_radius_meters=10.0)
        policy = OSMRoadPolicy(
            way_id=42, highway_type="residential", maxspeed_kmh=30,
            is_prohibited_road=False, query_lat_lon=(25.2708, 55.2708),
        )
        engine.cache.put(25.2708, 55.2708, policy)

        # Mock urlopen to ensure it's NOT called.
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = engine.get_road_policy(25.2708 + 0.00001, 55.2708)
            mock_urlopen.assert_not_called()

        assert result.way_id == 42
        assert result.is_fallback is False

    def test_cache_miss_hits_api(self):
        """A cache miss should trigger the Overpass API query."""
        engine = OSMPolicyEngine(cache_radius_meters=3.0)
        elements = [_make_way_element(99, {"highway": "residential", "maxspeed": "30"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = engine.get_road_policy(25.2708, 55.2708)
            mock_urlopen.assert_called_once()

        assert result.way_id == 99
        assert result.highway_type == "residential"
        assert result.maxspeed_kmh == 30
        assert result.is_fallback is False

    def test_cache_populated_after_query(self):
        """After an API query, the result should be cached."""
        engine = OSMPolicyEngine(cache_radius_meters=5.0)
        elements = [_make_way_element(42, {"highway": "primary"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            engine.get_road_policy(25.2708, 55.2708)

        # Second call within cache radius should be a cache hit.
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = engine.get_road_policy(25.2708 + 0.00001, 55.2708)
            mock_urlopen.assert_not_called()

        assert result.way_id == 42


# --------------------------------------------------------------------------- #
# OSMPolicyEngine - API query & OTel
# --------------------------------------------------------------------------- #
class TestEngineQuery:
    def test_successful_api_query_parses_way(self, spans):
        """A successful Overpass response is parsed into an OSMRoadPolicy."""
        engine = OSMPolicyEngine()
        elements = [_make_way_element(12345, {"highway": "residential", "maxspeed": "50"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert isinstance(result, OSMRoadPolicy)
        assert result.way_id == 12345
        assert result.highway_type == "residential"
        assert result.maxspeed_kmh == 50
        assert result.is_fallback is False
        assert result.query_lat_lon == (25.2048, 55.2708)

    def test_query_contains_highway_tag(self):
        """The Overpass QL query should search for ways with highway tags."""
        engine = OSMPolicyEngine()
        query = engine._build_query(25.2048, 55.2708)
        assert "[highway]" in query
        assert "[maxspeed]" in query
        assert str(OSM_QUERY_RADIUS_M) in query
        assert "25.2048" in query
        assert "55.2708" in query

    def test_otel_attributes_on_success(self, spans):
        """OTel span attributes are recorded on a successful query."""
        engine = OSMPolicyEngine(cache_radius_meters=3.0)
        elements = [_make_way_element(123, {"highway": "residential", "maxspeed": "30"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            engine.get_road_policy(25.2048, 55.2708)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "geospatial.osm_query")
        assert span.attributes.get("osm.cache_hit") is False
        assert span.attributes.get("osm.http_status") == 200
        assert span.attributes.get("osm.fallback_triggered") is False
        assert span.attributes.get("osm.ways_found") == 1
        assert span.attributes.get("osm.latency_ms") is not None

    def test_otel_attributes_on_cache_hit(self, spans):
        """OTel span records cache_hit=True when served from cache."""
        engine = OSMPolicyEngine(cache_radius_meters=10.0)
        policy = OSMRoadPolicy(
            way_id=1, highway_type="x", maxspeed_kmh=30,
            is_prohibited_road=False, query_lat_lon=(25.2708, 55.2708),
        )
        engine.cache.put(25.2708, 55.2708, policy)

        engine.get_road_policy(25.2708 + 0.00001, 55.2708)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "geospatial.osm_query")
        assert span.attributes.get("osm.cache_hit") is True

    def test_otel_attributes_on_fallback(self, spans):
        """OTel span records fallback_triggered=True on network error."""
        engine = OSMPolicyEngine()

        def raise_timeout(*args, **kwargs):
            raise TimeoutError("timed out")

        with patch("urllib.request.urlopen", side_effect=raise_timeout):
            result = engine.get_road_policy(25.2048, 55.2708)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "geospatial.osm_query")
        assert span.attributes.get("osm.fallback_triggered") is True
        assert span.attributes.get("osm.http_status") == 0
        assert span.attributes.get("osm.ways_found") == 0
        assert result.is_fallback is True


# --------------------------------------------------------------------------- #
# OSMPolicyEngine - fallback mechanisms
# --------------------------------------------------------------------------- #
class TestEngineFallbacks:
    def test_missing_gps_returns_neutral_policy(self):
        """lat=0.0 or lon=0.0 returns a neutral fallback without hitting API."""
        engine = OSMPolicyEngine()
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = engine.get_road_policy(0.0, 55.0)
            mock_urlopen.assert_not_called()

        assert result.is_fallback is True
        assert result.maxspeed_kmh is None
        assert result.is_prohibited_road is False
        assert result.query_lat_lon == (0.0, 55.0)

    def test_missing_gps_lon_zero(self):
        engine = OSMPolicyEngine()
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = engine.get_road_policy(25.0, 0.0)
            mock_urlopen.assert_not_called()

        assert result.is_fallback is True
        assert result.maxspeed_kmh is None

    def test_network_timeout_returns_fallback(self):
        """A socket timeout triggers the network fallback policy."""
        engine = OSMPolicyEngine(timeout=1.5)
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_fallback is True
        assert result.maxspeed_kmh == FALLBACK_MAXSPEED_KMH
        assert result.highway_type == "fallback"
        assert result.is_prohibited_road is False

    def test_http_429_returns_fallback(self):
        """HTTP 429 (rate-limited) triggers the fallback policy."""
        engine = OSMPolicyEngine()
        err = urllib.error.HTTPError(
            url="http://test", code=429, msg="Too Many Requests", hdrs={}, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=err):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_fallback is True
        assert result.maxspeed_kmh == FALLBACK_MAXSPEED_KMH

    def test_http_504_returns_fallback(self):
        """HTTP 504 (gateway timeout) triggers the fallback policy."""
        engine = OSMPolicyEngine()
        err = urllib.error.HTTPError(
            url="http://test", code=504, msg="Gateway Timeout", hdrs={}, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=err):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_fallback is True
        assert result.maxspeed_kmh == FALLBACK_MAXSPEED_KMH

    def test_url_error_returns_fallback(self):
        """A URLError (DNS failure, connection refused) triggers fallback."""
        engine = OSMPolicyEngine()
        err = urllib.error.URLError("dns failure")

        with patch("urllib.request.urlopen", side_effect=err):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_fallback is True
        assert result.maxspeed_kmh == FALLBACK_MAXSPEED_KMH

    def test_fallback_result_cached(self):
        """Fallback results are cached so subsequent calls don't retry."""
        engine = OSMPolicyEngine(cache_radius_meters=5.0)
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result1 = engine.get_road_policy(25.2048, 55.2708)

        # Second call should return from cache without network.
        with patch("urllib.request.urlopen") as mock_urlopen:
            result2 = engine.get_road_policy(25.2048 + 0.00001, 55.2708)
            mock_urlopen.assert_not_called()

        assert result1.is_fallback is True
        assert result2.is_fallback is True
        assert result2.maxspeed_kmh == FALLBACK_MAXSPEED_KMH

    def test_default_timeout_is_1_5s(self):
        engine = OSMPolicyEngine()
        assert engine.timeout == DEFAULT_TIMEOUT_SECONDS

    def test_default_overpass_url(self):
        engine = OSMPolicyEngine()
        assert engine.overpass_url == "https://overpass-api.de/api/interpreter"


# --------------------------------------------------------------------------- #
# OSMPolicyEngine - RTA/ITC rule enforcement
# --------------------------------------------------------------------------- #
class TestRTAIRules:
    @pytest.mark.parametrize("highway", ["footway", "pedestrian", "steps"])
    def test_prohibited_highway_types(self, highway):
        """Highway types in the prohibited list trigger is_prohibited_road."""
        engine = OSMPolicyEngine()
        elements = [_make_way_element(1, {"highway": highway, "maxspeed": "30"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_prohibited_road is True

    def test_highway_not_in_prohibited_list(self):
        """Non-prohibited highway types are allowed."""
        engine = OSMPolicyEngine()
        elements = [_make_way_element(1, {"highway": "residential", "maxspeed": "30"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_prohibited_road is False

    def test_maxspeed_above_threshold_prohibits(self):
        """maxspeed > 60 km/h triggers prohibition."""
        engine = OSMPolicyEngine()
        elements = [_make_way_element(1, {"highway": "primary", "maxspeed": "80"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_prohibited_road is True
        assert result.maxspeed_kmh == 80

    def test_maxspeed_at_threshold_not_prohibited(self):
        """maxspeed == 60 km/h is NOT prohibited (strictly greater than)."""
        engine = OSMPolicyEngine()
        elements = [_make_way_element(1, {"highway": "primary", "maxspeed": "60"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_prohibited_road is False
        assert result.maxspeed_kmh == 60

    def test_maxspeed_below_threshold_not_prohibited(self):
        engine = OSMPolicyEngine()
        elements = [_make_way_element(1, {"highway": "residential", "maxspeed": "50"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_prohibited_road is False

    def test_no_maxspeed_not_prohibited(self):
        """A way with highway but no maxspeed should not be auto-prohibited."""
        engine = OSMPolicyEngine()
        elements = [_make_way_element(1, {"highway": "residential"})]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.maxspeed_kmh is None
        assert result.is_prohibited_road is False

    def test_prohibited_highway_types_constant(self):
        assert PROHIBITED_HIGHWAY_TYPES == frozenset({"footway", "pedestrian", "steps"})


# --------------------------------------------------------------------------- #
# OSMPolicyEngine - maxspeed parsing
# --------------------------------------------------------------------------- #
class TestMaxspeedParsing:
    def test_simple_integer(self):
        from src.geospatial.osm_client import _parse_maxspeed
        assert _parse_maxspeed("30") == 30

    def test_with_units(self):
        from src.geospatial.osm_client import _parse_maxspeed
        assert _parse_maxspeed("30 km/h") == 30

    def test_compound_value(self):
        from src.geospatial.osm_client import _parse_maxspeed
        assert _parse_maxspeed("30;40") == 30

    def test_none_value(self):
        from src.geospatial.osm_client import _parse_maxspeed
        assert _parse_maxspeed(None) is None

    def test_empty_string(self):
        from src.geospatial.osm_client import _parse_maxspeed
        assert _parse_maxspeed("") is None

    def test_none_keyword(self):
        from src.geospatial.osm_client import _parse_maxspeed
        assert _parse_maxspeed("none") is None


# --------------------------------------------------------------------------- #
# OSMPolicyEngine - edge cases
# --------------------------------------------------------------------------- #
class TestEngineEdgeCases:
    def test_no_ways_found_returns_fallback(self):
        """Empty Overpass response returns a fallback policy."""
        engine = OSMPolicyEngine()
        mock_resp = MockHTTPResponse(_make_overpass_response([]))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.is_fallback is True
        assert result.maxspeed_kmh == FALLBACK_MAXSPEED_KMH

    def test_missing_highway_falls_back_to_any_way(self):
        """If no way has a highway tag, the first element is used."""
        engine = OSMPolicyEngine()
        elements = [
            _make_way_element(5, {"maxspeed": "25"}),  # no highway tag
            _make_way_element(7, {"highway": "residential", "maxspeed": "30"}),
        ]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        # Prefer the element with a highway tag.
        assert result.way_id == 7
        assert result.highway_type == "residential"

    def test_multiple_ways_picks_first_highway(self):
        """When multiple ways match, the first with highway is preferred."""
        engine = OSMPolicyEngine()
        elements = [
            _make_way_element(10, {"highway": "footway", "maxspeed": "20"}),
            _make_way_element(20, {"highway": "residential", "maxspeed": "30"}),
        ]
        mock_resp = MockHTTPResponse(_make_overpass_response(elements))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.way_id == 10  # first with highway tag

    def test_custom_overpass_url(self):
        engine = OSMPolicyEngine(overpass_url="http://localhost:8080/api/interpreter")
        assert engine.overpass_url == "http://localhost:8080/api/interpreter"

    def test_custom_fallback_maxspeed(self):
        engine = OSMPolicyEngine(fallback_maxspeed_kmh=25)
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
            result = engine.get_road_policy(25.2048, 55.2708)

        assert result.maxspeed_kmh == 25

    def test_http_status_recorded_on_error(self, spans):
        """HTTP error status code is recorded in the OTel span."""
        engine = OSMPolicyEngine()
        err = urllib.error.HTTPError(
            url="http://test", code=504, msg="Gateway Timeout", hdrs={}, fp=None
        )

        with patch("urllib.request.urlopen", side_effect=err):
            engine.get_road_policy(25.2048, 55.2708)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "geospatial.osm_query")
        assert span.attributes.get("osm.http_status") == 504
        assert span.attributes.get("osm.fallback_triggered") is True
