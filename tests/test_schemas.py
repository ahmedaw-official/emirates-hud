"""Unit tests for the validated Pydantic schema models."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from src.schemas import (
    BoundingBox,
    ComplianceState,
    DetectionResult,
    FrameState,
    OSMRoadPolicy,
    RiskLevel,
    SpatialCoordinate,
    ViolationDetail,
    ViolationType,
    VLMAuditResponse,
    ZoneClassification,
)
from src.schemas import (
    BoundingBox as _B,  # sanity alias
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now() -> dt.datetime:
    return dt.datetime(2026, 9, 1, 13, 18, 57, tzinfo=dt.timezone.utc)


def _bbox(**over) -> BoundingBox:
    defaults = {
        "xmin": 0.1, "ymin": 0.1, "xmax": 0.9, "ymax": 0.9, "confidence": 0.87,
        "class_id": 0, "class_name": "scooter",
    }
    defaults.update(over)
    return BoundingBox(**defaults)


# --------------------------------------------------------------------------- #
# BoundingBox
# --------------------------------------------------------------------------- #
class TestBoundingBox:
    def test_basic_construction_and_computed_fields(self):
        b = _bbox()
        assert b.width == pytest.approx(0.8)
        assert b.height == pytest.approx(0.8)
        assert b.area == pytest.approx(0.64)

    def test_rejects_out_of_range_coords(self):
        with pytest.raises(ValidationError):
            _bbox(xmin=-0.1)
        with pytest.raises(ValidationError):
            _bbox(ymax=1.5)

    def test_rejects_inverted_box(self):
        with pytest.raises(ValidationError, match="xmax"):
            _bbox(xmin=0.8, xmax=0.2)
        with pytest.raises(ValidationError, match="ymax"):
            _bbox(ymin=0.9, ymax=0.1)

    def test_confidence_bounds(self):
        _bbox(confidence=0.0)
        _bbox(confidence=1.0)
        with pytest.raises(ValidationError):
            _bbox(confidence=-0.01)
        with pytest.raises(ValidationError):
            _bbox(confidence=1.01)

    def test_pixel_coordinates_optional(self):
        b = _bbox(x_min_px=12, y_min_px=24, x_max_px=48, y_max_px=96)
        assert b.x_min_px == 12 and b.x_max_px == 48

    def test_pixel_coords_must_be_ordered(self):
        with pytest.raises(ValidationError, match="x_max_px"):
            _bbox(x_min_px=100, x_max_px=10)

    def test_class_name_required(self):
        with pytest.raises(ValidationError):
            BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1, confidence=0.5, class_id=0, class_name="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="extra"):
            BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1, confidence=0.5, class_id=0,
                        class_name="x", rogue=1)

    def test_frozen(self):
        b = _bbox()
        with pytest.raises(Exception):
            b.class_id = 9  # pydantic raises on frozen model

    def test_json_round_trip(self):
        b = _bbox()
        dumped = b.model_dump_json()
        restored = BoundingBox.model_validate_json(dumped)
        assert restored == b


# --------------------------------------------------------------------------- #
# DetectionResult
# --------------------------------------------------------------------------- #
class TestDetectionResult:
    def test_construction(self):
        r = DetectionResult(boxes=[_bbox(), _bbox(class_name="person")], frame_id=7,
                            timestamp_ms=1_700_000_000_000, processing_time_ms=12.5)
        assert r.detection_count == 2
        assert r.max_confidence == pytest.approx(0.87)
        assert r.detections_by_class == {"scooter": 1, "person": 1}

    def test_empty_results(self):
        r = DetectionResult(frame_id=0, timestamp_ms=0, processing_time_ms=1.0)
        assert r.detection_count == 0
        assert r.max_confidence == 0.0
        assert r.detections_by_class == {}

    def test_negative_frame_id_rejected(self):
        with pytest.raises(ValidationError):
            DetectionResult(frame_id=-1, timestamp_ms=0, processing_time_ms=1.0)

    def test_negative_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            DetectionResult(frame_id=0, timestamp_ms=-5, processing_time_ms=1.0)

    def test_negative_processing_time_rejected(self):
        with pytest.raises(ValidationError):
            DetectionResult(frame_id=0, timestamp_ms=0, processing_time_ms=-0.1)

    # -- filter_by_class --------------------------------------------------------
    def _multi_class_result(self) -> DetectionResult:
        return DetectionResult(
            boxes=[
                _bbox(class_name="scooter", confidence=0.9),
                _bbox(class_name="person", confidence=0.8),
                _bbox(class_name="scooter", confidence=0.7),
                _bbox(class_name="bike", confidence=0.6),
            ],
            frame_id=5,
            timestamp_ms=1_700_000_000_000,
            processing_time_ms=12.5,
        )

    def test_filter_by_class_single_string(self):
        r = self._multi_class_result()
        scooters = r.filter_by_class("scooter")
        assert scooters.detection_count == 2
        assert all(b.class_name == "scooter" for b in scooters.boxes)

    def test_filter_by_class_list(self):
        r = self._multi_class_result()
        filtered = r.filter_by_class(["person", "bike"])
        assert filtered.detection_count == 2
        names = {b.class_name for b in filtered.boxes}
        assert names == {"person", "bike"}

    def test_filter_by_class_preserves_metadata(self):
        r = self._multi_class_result()
        filtered = r.filter_by_class("scooter")
        assert filtered.frame_id == r.frame_id
        assert filtered.timestamp_ms == r.timestamp_ms
        assert filtered.processing_time_ms == r.processing_time_ms

    def test_filter_by_class_no_match(self):
        r = self._multi_class_result()
        filtered = r.filter_by_class("unknown")
        assert filtered.detection_count == 0
        assert filtered.boxes == []

    def test_filter_by_class_returns_independent_list(self):
        r = self._multi_class_result()
        filtered = r.filter_by_class("scooter")
        # The filtered result has its own boxes list (not a view of the original).
        assert filtered.boxes is not r.boxes
        assert len(filtered.boxes) != len(r.boxes)

    def test_filter_by_class_empty_result(self):
        r = DetectionResult(frame_id=0, timestamp_ms=0, processing_time_ms=1.0)
        filtered = r.filter_by_class("scooter")
        assert filtered.detection_count == 0
        assert filtered.frame_id == 0

    # -- filter_by_confidence --------------------------------------------------
    def test_filter_by_confidence_threshold(self):
        r = self._multi_class_result()
        confident = r.filter_by_confidence(0.8)
        assert confident.detection_count == 2
        assert all(b.confidence >= 0.8 for b in confident.boxes)

    def test_filter_by_confidence_preserves_metadata(self):
        r = self._multi_class_result()
        filtered = r.filter_by_confidence(0.75)
        assert filtered.frame_id == r.frame_id
        assert filtered.timestamp_ms == r.timestamp_ms
        assert filtered.processing_time_ms == r.processing_time_ms

    def test_filter_by_confidence_invalid_range(self):
        r = self._multi_class_result()
        with pytest.raises(ValueError, match="min_confidence"):
            r.filter_by_confidence(1.5)
        with pytest.raises(ValueError, match="min_confidence"):
            r.filter_by_confidence(-0.1)

    def test_filter_by_confidence_zero_keeps_all(self):
        r = self._multi_class_result()
        filtered = r.filter_by_confidence(0.0)
        assert filtered.detection_count == r.detection_count


# --------------------------------------------------------------------------- #
# SpatialCoordinate
# --------------------------------------------------------------------------- #
class TestSpatialCoordinate:
    def test_valid(self):
        c = SpatialCoordinate(latitude=25.2, longitude=55.27, altitude=12.4,
                              accuracy=2.5, timestamp=_now())
        assert c.latitude == 25.2
        assert c.accuracy == 2.5

    def test_latitude_range(self):
        with pytest.raises(ValidationError, match="latitude"):
            SpatialCoordinate(latitude=91, longitude=0, altitude=0, accuracy=1, timestamp=_now())

    def test_longitude_range(self):
        with pytest.raises(ValidationError, match="longitude"):
            SpatialCoordinate(latitude=0, longitude=-181, altitude=0, accuracy=1, timestamp=_now())

    def test_negative_accuracy_rejected(self):
        with pytest.raises(ValidationError):
            SpatialCoordinate(latitude=0, longitude=0, altitude=0, accuracy=-1, timestamp=_now())

    def test_default_timestamp(self):
        # ``timestamp`` has no default, so omitting it must fail.
        with pytest.raises(ValidationError):
            SpatialCoordinate(latitude=0, longitude=0, altitude=0, accuracy=1)


# --------------------------------------------------------------------------- #
# OSMRoadPolicy
# --------------------------------------------------------------------------- #
class TestOSMRoadPolicy:
    def _policy(self, **over):
        base = {
            "way_id": 42, "highway_type": "path", "is_prohibited_road": False,
            "query_lat_lon": (25.2, 55.27),
        }
        base.update(over)
        return OSMRoadPolicy(**base)

    def test_defaults(self):
        p = self._policy()
        assert p.maxspeed_kmh is None
        assert p.is_fallback is False

    def test_maxspeed_optional(self):
        p = self._policy(maxspeed_kmh=30)
        assert p.maxspeed_kmh == 30

    def test_query_coords_validation(self):
        with pytest.raises(ValidationError, match="latitude"):
            self._policy(query_lat_lon=(95.0, 55.0))
        with pytest.raises(ValidationError, match="longitude"):
            self._policy(query_lat_lon=(25.0, 190.0))

    def test_negative_way_id_rejected(self):
        with pytest.raises(ValidationError):
            self._policy(way_id=-1)


# --------------------------------------------------------------------------- #
# ViolationDetail
# --------------------------------------------------------------------------- #
class TestViolationDetail:
    def test_from_enum_member(self):
        v = ViolationDetail(
            violation_type=ViolationType.NO_DISMOUNT,
            legal_reference="Dubai RTA Resolution No. 13 (2022) Article 4",
            fine_amount_aed=300,
        )
        assert v.violation_type == "NO_DISMOUNT"
        assert isinstance(v.violation_type, str)

    def test_from_raw_string(self):
        v = ViolationDetail(violation_type="CUSTOM_INFRACTION",
                            legal_reference="Local Bylaw §7", fine_amount_aed=150)
        assert v.violation_type == "CUSTOM_INFRACTION"

    def test_negative_fine_rejected(self):
        with pytest.raises(ValidationError):
            ViolationDetail(violation_type="X", legal_reference="Y", fine_amount_aed=-10)

    def test_empty_legal_reference_rejected(self):
        with pytest.raises(ValidationError):
            ViolationDetail(violation_type="X", legal_reference="  ", fine_amount_aed=0)


# --------------------------------------------------------------------------- #
# VLMAuditResponse
# --------------------------------------------------------------------------- #
class TestVLMAuditResponse:
    def test_enum_coercion_from_strings(self):
        a = VLMAuditResponse(
            zone_classification="PEDESTRIAN_SIDEWALK",
            dismount_required=True,
            risk_level="CRITICAL",
            hud_warning_text="Dismount immediately",
        )
        assert a.zone_classification is ZoneClassification.PEDESTRIAN_SIDEWALK
        assert a.risk_level is RiskLevel.CRITICAL
        assert a.has_violations is False
        assert a.max_fine_aed == 0

    def test_with_violations(self):
        v = ViolationDetail(violation_type=ViolationType.NO_DISMOUNT,
                            legal_reference="RTA 13(2022) Art4", fine_amount_aed=300)
        a = VLMAuditResponse(
            zone_classification=ZoneClassification.CROSSWALK,
            dismount_required=False,
            risk_level=RiskLevel.HIGH,
            violations=[v],
            hud_warning_text="Caution",
        )
        assert a.has_violations is True
        assert a.max_fine_aed == 300

    def test_invalid_zone_rejected(self):
        with pytest.raises(ValidationError):
            VLMAuditResponse(zone_classification="MAGIC_ROAD", dismount_required=False,
                             risk_level="LOW", hud_warning_text="ok")


# --------------------------------------------------------------------------- #
# FrameState
# --------------------------------------------------------------------------- #
class TestFrameState:
    def _make(self, **over) -> FrameState:
        base = {
            "frame_id": 1,
            "timestamp": _now(),
            "geospatial_policy": OSMRoadPolicy(
                way_id=42, highway_type="path", is_prohibited_road=False,
                query_lat_lon=(25.2, 55.27),
            ),
            "active_compliance_state": ComplianceState.SAFE,
            "hud_alert_message": "All clear",
        }
        base.update(over)
        return FrameState(**base)

    def test_minimal_valid(self):
        fs = self._make()
        assert fs.is_violation is False
        assert fs.fine_risk_aed == 0
        assert fs.visual_detections == []

    def test_with_detections_and_violation(self):
        v = ViolationDetail(violation_type=ViolationType.SIDEWALK_DISOBEDENCE,
                            legal_reference="RTA", fine_amount_aed=500)
        audit = VLMAuditResponse(
            zone_classification=ZoneClassification.PEDESTRIAN_SIDEWALK,
            dismount_required=True,
            risk_level=RiskLevel.CRITICAL,
            violations=[v],
            hud_warning_text="VIOLATION",
        )
        fs = self._make(
            visual_detections=[_bbox()],
            compliance_audit=audit,
            active_compliance_state=ComplianceState.VIOLATION,
            hud_alert_message="VIOLATION",
            fine_risk_aed=500,
        )
        assert fs.is_violation is True
        assert fs.fine_risk_aed == 500

    def test_enum_coercion(self):
        fs = self._make(active_compliance_state="VIOLATION")
        assert fs.active_compliance_state is ComplianceState.VIOLATION

    def test_negative_fine_risk_rejected(self):
        with pytest.raises(ValidationError):
            self._make(fine_risk_aed=-50)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            self._make(does_not_exist=1)

    def test_json_round_trip(self):
        fs = self._make(visual_detections=[_bbox()])
        dumped = fs.model_dump_json()
        restored = FrameState.model_validate_json(dumped)
        assert restored == fs
        # enums survive as their string values in JSON
        import json as _json
        assert _json.loads(dumped)["active_compliance_state"] == "SAFE"


# --------------------------------------------------------------------------- #
# Cross-model sanity
# --------------------------------------------------------------------------- #
def test_bounding_box_is_same_object_as_public_alias():
    assert BoundingBox is _B


def test_exception_exports_are_not_schemas_but_importable():
    # Guard against accidentally re-exporting exception types from schemas.
    assert "HUDBaseException" not in dir()
    from src.schemas import __all__ as schemas_all

    assert all(name not in {"HUDBaseException", "PerceptionEngineError"} for name in schemas_all)
