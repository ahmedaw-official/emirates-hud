"""Tests for the state engine: trigger router + compliance state machine.

Covers:
* TriggerRouter: surface transitions, critical signs, heartbeat, no-trigger
* ComplianceStateMachine: logic matrix (VIOLATION/WARNING/SAFE),
  hysteresis buffer (3-frame hold), VLM escalation, fallback when VLM is None,
  fallback when OSM is in fallback mode
* OTel span attributes: state.active_status, state.fine_risk_aed,
  router.vlm_triggered, router.trigger_reason, state.hysteresis_active
"""

from __future__ import annotations

from src.schemas import (
    BoundingBox,
    ComplianceState,
    DetectionResult,
    OSMRoadPolicy,
    RiskLevel,
    ViolationDetail,
    ViolationType,
    VLMAuditResponse,
    ZoneClassification,
)
from src.state_engine import ComplianceStateMachine, TriggerRouter
from src.state_engine.compliance_state_machine import (
    HYSTERESIS_THRESHOLD,
    SAFE_FINE_AED,
    VIOLATION_FINE_AED,
    WARNING_FINE_AED,
)


# --------------------------------------------------------------------------- #
# Test data builders
# --------------------------------------------------------------------------- #
def _box(class_name: str, confidence: float = 0.9, **kwargs: float) -> BoundingBox:
    """Build a BoundingBox with sensible defaults for the given class."""
    return BoundingBox(
        xmin=kwargs.get("xmin", 0.1),
        ymin=kwargs.get("ymin", 0.1),
        xmax=kwargs.get("xmax", 0.5),
        ymax=kwargs.get("ymax", 0.5),
        confidence=confidence,
        class_id=kwargs.get("class_id", 0),
        class_name=class_name,
    )


def _detections(boxes: list[BoundingBox] | None = None) -> DetectionResult:
    return DetectionResult(
        boxes=boxes or [],
        frame_id=42,
        timestamp_ms=1_700_000_000_000,
        processing_time_ms=45.3,
    )


def _osm(
    highway_type: str = "residential",
    maxspeed_kmh: int | None = 30,
    is_prohibited: bool = False,
    way_id: int = 12345,
    is_fallback: bool = False,
    lat: float = 25.2048,
    lon: float = 55.2708,
) -> OSMRoadPolicy:
    return OSMRoadPolicy(
        way_id=way_id,
        highway_type=highway_type,
        maxspeed_kmh=maxspeed_kmh,
        is_prohibited_road=is_prohibited,
        query_lat_lon=(lat, lon),
        is_fallback=is_fallback,
    )


def _vlm_audit(
    risk_level: RiskLevel = RiskLevel.LOW,
    violations: list[ViolationDetail] | None = None,
    zone: ZoneClassification = ZoneClassification.DESIGNATED_LANE,
    hud_text: str = "OK",
) -> VLMAuditResponse:
    return VLMAuditResponse(
        zone_classification=zone,
        dismount_required=False,
        risk_level=risk_level,
        violations=violations or [],
        hud_warning_text=hud_text,
    )


# --------------------------------------------------------------------------- #
# Test 1: TriggerRouter
# --------------------------------------------------------------------------- #
class TestTriggerRouter:
    def test_default_heartbeat_threshold(self):
        router = TriggerRouter()
        assert router.heartbeat_threshold == 30

    def test_custom_heartbeat_threshold(self):
        router = TriggerRouter(heartbeat_threshold=15)
        assert router.heartbeat_threshold == 15

    def test_surface_transition_triggers_vlm(self):
        router = TriggerRouter()
        # Prime with previous surface
        router.should_trigger_vlm(_detections([_box("red_track")]))

        # New frame: surface changed to grey_sidewalk
        result, reason = router.should_trigger_vlm(
            _detections([_box("grey_sidewalk")]),
        )
        assert result is True
        assert "surface_transition" in reason

    def test_same_surface_no_transition(self):
        router = TriggerRouter()
        router.should_trigger_vlm(_detections([_box("red_track")]))

        result, reason = router.should_trigger_vlm(
            _detections([_box("red_track")]),
        )
        # No critical sign, under heartbeat threshold -> no trigger
        assert result is False
        assert reason == "no_trigger"

    def test_critical_sign_triggers_vlm(self):
        router = TriggerRouter()
        # No surface transition, no heartbeat
        result, reason = router.should_trigger_vlm(
            _detections([_box("zebra_crosswalk", confidence=0.8)]),
        )
        assert result is True
        assert "critical_sign:zebra_crosswalk" in reason

    def test_critical_sign_below_confidence_threshold(self):
        """Confidence <= 0.50 does NOT trigger."""
        router = TriggerRouter()
        result, reason = router.should_trigger_vlm(
            _detections([_box("zebra_crosswalk", confidence=0.50)]),
        )
        assert result is False
        assert reason == "no_trigger"

    def test_sign_no_scooter_triggers_vlm(self):
        router = TriggerRouter()
        result, reason = router.should_trigger_vlm(
            _detections([_box("sign_no_scooter", confidence=0.85)]),
        )
        assert result is True
        assert "critical_sign:sign_no_scooter" in reason

    def test_heartbeat_triggers_vlm(self):
        """30 frames since last VLM triggers a heartbeat VLM call."""
        router = TriggerRouter()
        # Simulate 30 frames of no-trigger (brings counter from 0 to 30)
        for i in range(30):
            # Each frame: red_track, same surface, no critical signs
            result, reason = router.should_trigger_vlm(
                _detections([_box("red_track")]),
            )
            assert result is False

        # 31st call: should trigger by heartbeat (frames_since_last_vlm == 30)
        result, reason = router.should_trigger_vlm(
            _detections([_box("red_track")]),
        )
        assert result is True
        assert reason == "heartbeat"

    def test_heartbeat_exactly_at_threshold(self):
        """Exactly 30 frames elapsed triggers heartbeat."""
        router = TriggerRouter()
        result, reason = router.should_trigger_vlm(
            _detections([_box("red_track")]),
            frames_since_last_vlm=30,
        )
        assert result is True
        assert reason == "heartbeat"

    def test_no_trigger_when_below_all_thresholds(self):
        router = TriggerRouter()
        # First call, same surface as primed, no critical signs
        router.should_trigger_vlm(_detections([_box("red_track")]))
        result, reason = router.should_trigger_vlm(
            _detections([_box("red_track")]),
            frames_since_last_vlm=5,
        )
        assert result is False
        assert reason == "no_trigger"

    def test_trigger_resets_frame_counter(self):
        router = TriggerRouter()
        # Trigger via heartbeat
        router.should_trigger_vlm(
            _detections([_box("red_track")]),
            frames_since_last_vlm=30,
        )
        assert router.frames_since_last_vlm == 0

    def test_previous_surface_property(self):
        router = TriggerRouter()
        router.should_trigger_vlm(_detections([_box("red_track")]))
        assert router.previous_surface == "red_track"

    def test_reset(self):
        router = TriggerRouter()
        router.should_trigger_vlm(_detections([_box("red_track")]))
        assert router.previous_surface is not None
        assert router.frames_since_last_vlm is not None  # may be 0 or 1
        router.reset()
        assert router.previous_surface is None
        assert router.frames_since_last_vlm == 0

    def test_explicit_frames_override(self):
        """Passing frames_since_last_vlm as kwarg overrides internal state."""
        router = TriggerRouter()
        result, reason = router.should_trigger_vlm(
            _detections([_box("red_track")]),
            frames_since_last_vlm=45,
        )
        assert result is True
        assert reason == "heartbeat"

    def test_explicit_previous_surface_override(self):
        """Passing previous_surface as kwarg overrides internal state."""
        router = TriggerRouter()
        result, reason = router.should_trigger_vlm(
            _detections([_box("grey_sidewalk")]),
            previous_surface="red_track",
        )
        assert result is True
        assert "surface_transition" in reason

    def test_no_boxes_no_trigger(self):
        router = TriggerRouter()
        result, reason = router.should_trigger_vlm(_detections([]))
        assert result is False
        assert reason == "no_trigger"

    def test_critical_sign_takes_priority_over_no_transition(self):
        """Even without surface change, critical sign triggers."""
        router = TriggerRouter()
        router.should_trigger_vlm(_detections([_box("red_track", confidence=0.1)]))
        result, reason = router.should_trigger_vlm(
            _detections([_box("red_track", confidence=0.1), _box("sign_no_scooter", confidence=0.9)]),
        )
        assert result is True
        assert "critical_sign" in reason


# --------------------------------------------------------------------------- #
# Test 2: ComplianceStateMachine - logic matrix
# --------------------------------------------------------------------------- #
class TestLogicMatrix:
    def test_grey_sidewalk_violation(self):
        machine = ComplianceStateMachine()
        detections = _detections([_box("grey_sidewalk"), _box("scooter")])
        result = machine.evaluate_frame_state(detections, _osm())

        assert result.active_compliance_state == ComplianceState.VIOLATION
        assert result.fine_risk_aed == VIOLATION_FINE_AED
        assert result.hud_alert_message == "CRITICAL: PROHIBITED ZONE - AED 300 FINE"

    def test_prohibited_road_violation(self):
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        policy = _osm(is_prohibited=True)
        result = machine.evaluate_frame_state(detections, policy)

        assert result.active_compliance_state == ComplianceState.VIOLATION
        assert result.fine_risk_aed == 300

    def test_prohibited_road_not_triggered_when_osm_fallback(self):
        """When OSM is in fallback, is_prohibited_road is ignored."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        policy = _osm(is_prohibited=True, is_fallback=True)
        result = machine.evaluate_frame_state(detections, policy)

        # grey_sidewalk not detected, OSM is fallback so is_prohibited_road ignored
        # -> falls through to red_track -> SAFE
        assert result.active_compliance_state == ComplianceState.SAFE

    def test_zebra_crosswalk_warning(self):
        machine = ComplianceStateMachine()
        detections = _detections([_box("zebra_crosswalk"), _box("scooter")])
        result = machine.evaluate_frame_state(detections, _osm())

        assert result.active_compliance_state == ComplianceState.WARNING
        assert result.fine_risk_aed == WARNING_FINE_AED
        assert result.hud_alert_message == "DISMOUNT REQUIRED AT CROSSWALK"

    def test_red_track_safe(self):
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        result = machine.evaluate_frame_state(detections, _osm())

        assert result.active_compliance_state == ComplianceState.SAFE
        assert result.fine_risk_aed == SAFE_FINE_AED
        assert result.hud_alert_message == "DESIGNATED LANE - SAFE"

    def test_no_surface_detected_defaults_safe(self):
        machine = ComplianceStateMachine()
        detections = _detections([_box("scooter")])
        result = machine.evaluate_frame_state(detections, _osm())

        assert result.active_compliance_state == ComplianceState.SAFE
        assert result.fine_risk_aed == 0

    def test_grey_sidewalk_takes_priority_over_crosswalk(self):
        """grey_sidewalk is checked first (Rule 1) before zebra_crosswalk (Rule 2)."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("grey_sidewalk"), _box("zebra_crosswalk")])
        result = machine.evaluate_frame_state(detections, _osm())

        assert result.active_compliance_state == ComplianceState.VIOLATION

    def test_vlm_escalation_critical(self):
        """VLM CRITICAL risk escalates SAFE state to VIOLATION."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        vlm = _vlm_audit(risk_level=RiskLevel.CRITICAL, violations=[
            ViolationDetail(
                violation_type=ViolationType.PROHIBITED_ROAD_USAGE,
                legal_reference="test",
                fine_amount_aed=400,
            )
        ])
        result = machine.evaluate_frame_state(detections, _osm(), vlm)

        assert result.active_compliance_state == ComplianceState.VIOLATION
        assert result.fine_risk_aed == 400  # VLM fine overrides

    def test_vlm_escalation_high(self):
        """VLM HIGH risk escalates SAFE state to WARNING."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        vlm = _vlm_audit(risk_level=RiskLevel.HIGH)
        result = machine.evaluate_frame_state(detections, _osm(), vlm)

        assert result.active_compliance_state == ComplianceState.WARNING

    def test_vlm_no_escalation_when_already_violation(self):
        """VLM LOW risk doesn't downgrade an already VIOLATION state."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("grey_sidewalk"), _box("scooter")])
        vlm = _vlm_audit(risk_level=RiskLevel.LOW)
        result = machine.evaluate_frame_state(detections, _osm(), vlm)

        assert result.active_compliance_state == ComplianceState.VIOLATION

    def test_vlm_none_state_from_yolo_only(self):
        """VLM=None: state determined exclusively by YOLO + OSM."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        result = machine.evaluate_frame_state(detections, _osm(), vlm_response=None)

        assert result.active_compliance_state == ComplianceState.SAFE
        assert result.compliance_audit is None

    def test_frame_state_carries_vlm_audit(self):
        """The VLM response is stored in FrameState.compliance_audit."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track")])
        vlm = _vlm_audit(risk_level=RiskLevel.LOW)
        result = machine.evaluate_frame_state(detections, _osm(), vlm)

        assert result.compliance_audit is vlm

    def test_frame_state_carries_detections(self):
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        result = machine.evaluate_frame_state(detections, _osm())

        assert len(result.visual_detections) == 2
        assert result.frame_id == 42


# --------------------------------------------------------------------------- #
# Test 3: Hysteresis
# --------------------------------------------------------------------------- #
class TestHysteresis:
    def test_hysteresis_threshold_default(self):
        assert HYSTERESIS_THRESHOLD == 3

    def test_violation_to_safe_requires_3_frames(self):
        """Transition from VIOLATION to SAFE requires 3 consecutive SAFE frames."""
        machine = ComplianceStateMachine()
        violation_detections = _detections([_box("grey_sidewalk")])
        safe_detections = _detections([_box("red_track")])

        # Frame 1: VIOLATION
        r1 = machine.evaluate_frame_state(violation_detections, _osm())
        assert r1.active_compliance_state == ComplianceState.VIOLATION

        # Frame 2: try to go SAFE - hysteresis holds
        r2 = machine.evaluate_frame_state(safe_detections, _osm())
        assert r2.active_compliance_state == ComplianceState.VIOLATION
        assert r2.hud_alert_message == "CRITICAL: PROHIBITED ZONE - AED 300 FINE"

        # Frame 3: still trying SAFE - still held
        r3 = machine.evaluate_frame_state(safe_detections, _osm())
        assert r3.active_compliance_state == ComplianceState.VIOLATION

        # Frame 4: 3rd attempt - now transitions
        r4 = machine.evaluate_frame_state(safe_detections, _osm())
        assert r4.active_compliance_state == ComplianceState.SAFE

    def test_violation_reverts_on_consecutive_violation(self):
        """If VIOLATION condition reappears, counter resets."""
        machine = ComplianceStateMachine()
        violation_detections = _detections([_box("grey_sidewalk")])
        safe_detections = _detections([_box("red_track")])

        machine.evaluate_frame_state(violation_detections, _osm())
        # Attempt 1: safe
        machine.evaluate_frame_state(safe_detections, _osm())
        # Back to violation
        machine.evaluate_frame_state(violation_detections, _osm())
        # Counter should reset - safe attempt 1
        r = machine.evaluate_frame_state(safe_detections, _osm())
        assert r.active_compliance_state == ComplianceState.VIOLATION

    def test_safe_to_violation_no_hysteresis(self):
        """Transition from SAFE to VIOLATION is immediate (no hysteresis)."""
        machine = ComplianceStateMachine()
        safe_detections = _detections([_box("red_track")])
        violation_detections = _detections([_box("grey_sidewalk")])

        machine.evaluate_frame_state(safe_detections, _osm())
        r = machine.evaluate_frame_state(violation_detections, _osm())
        assert r.active_compliance_state == ComplianceState.VIOLATION

    def test_hysteresis_active_flag(self):
        """The is_hysteresis_active property reflects counting state."""
        machine = ComplianceStateMachine()
        violation_detections = _detections([_box("grey_sidewalk")])
        safe_detections = _detections([_box("red_track")])

        machine.evaluate_frame_state(violation_detections, _osm())
        assert machine.is_hysteresis_active is False

        # Attempt to transition - counter starts
        machine.evaluate_frame_state(safe_detections, _osm())
        assert machine.is_hysteresis_active is True

    def test_hysteresis_reset_on_violation(self):
        """Re-entering VIOLATION resets hysteresis counter."""
        machine = ComplianceStateMachine()
        machine.force_state(ComplianceState.VIOLATION)

        safe_dets = _detections([_box("red_track")])
        viol_dets = _detections([_box("grey_sidewalk")])

        # Attempt safe
        machine.evaluate_frame_state(safe_dets, _osm())
        assert machine.hysteresis_counter == 1

        # Back to violation
        machine.evaluate_frame_state(viol_dets, _osm())
        assert machine.hysteresis_counter == 0

    def test_warning_to_safe_no_hysteresis(self):
        """Transition from WARNING to SAFE is immediate."""
        machine = ComplianceStateMachine()
        crosswalk_dets = _detections([_box("zebra_crosswalk")])
        safe_dets = _detections([_box("red_track")])

        machine.evaluate_frame_state(crosswalk_dets, _osm())
        r = machine.evaluate_frame_state(safe_dets, _osm())
        assert r.active_compliance_state == ComplianceState.SAFE

    def test_force_state(self):
        machine = ComplianceStateMachine()
        machine.force_state(ComplianceState.VIOLATION)
        assert machine.current_state == ComplianceState.VIOLATION
        assert machine.hysteresis_counter == 0

    def test_reset(self):
        machine = ComplianceStateMachine()
        machine.force_state(ComplianceState.VIOLATION)
        machine.reset()
        assert machine.current_state == ComplianceState.SAFE
        assert machine.hysteresis_counter == 0


# --------------------------------------------------------------------------- #
# Test 4: OTel span attributes
# --------------------------------------------------------------------------- #
class TestStateEngineOTel:
    def test_span_attributes_on_safe(self, spans):
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track")])
        machine.evaluate_frame_state(detections, _osm())

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "state_engine.evaluate_frame_state")

        assert span.attributes.get("state.active_status") == "SAFE"
        assert span.attributes.get("state.fine_risk_aed") == 0
        assert span.attributes.get("router.vlm_triggered") is False
        assert span.attributes.get("router.trigger_reason") == "no_trigger"
        assert span.attributes.get("state.hysteresis_active") is False

    def test_span_attributes_on_violation(self, spans):
        machine = ComplianceStateMachine()
        detections = _detections([_box("grey_sidewalk")])
        machine.evaluate_frame_state(detections, _osm())

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "state_engine.evaluate_frame_state")

        assert span.attributes.get("state.active_status") == "VIOLATION"
        assert span.attributes.get("state.fine_risk_aed") == 300
        assert span.attributes.get("router.vlm_triggered") is False

    def test_span_attributes_on_warning(self, spans):
        machine = ComplianceStateMachine()
        detections = _detections([_box("zebra_crosswalk")])
        machine.evaluate_frame_state(detections, _osm())

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "state_engine.evaluate_frame_state")

        assert span.attributes.get("state.active_status") == "WARNING"
        assert span.attributes.get("state.fine_risk_aed") == 200

    def test_span_attributes_vlm_triggered(self, spans):
        """When VLM is triggered (heartbeat), router.vlm_triggered is True."""
        machine = ComplianceStateMachine()
        # Force heartbeat trigger by passing frames_since_last_vlm=30
        machine._frames_since_last_vlm = 30
        detections = _detections([_box("red_track")])
        machine.evaluate_frame_state(detections, _osm())

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "state_engine.evaluate_frame_state")

        assert span.attributes.get("router.vlm_triggered") is True
        assert span.attributes.get("router.trigger_reason") == "heartbeat"

    def test_span_attributes_hysteresis_active(self, spans):
        """span.state.hysteresis_active is True when VIOLATION is held."""
        machine = ComplianceStateMachine()
        # Enter VIOLATION
        machine.evaluate_frame_state(_detections([_box("grey_sidewalk")]), _osm())

        # Attempt to transition - hysteresis holds
        machine.evaluate_frame_state(
            _detections([_box("red_track")]), _osm()
        )

        # Get the LAST span (second evaluate_frame_state call)
        finished = spans.get_finished_spans()
        hysteresis_span = finished[-1]
        assert hysteresis_span.attributes.get("state.hysteresis_active") is True
        assert hysteresis_span.attributes.get("state.active_status") == "VIOLATION"

    def test_span_attributes_trigger_reason_surface_transition(self, spans):
        """span.router.trigger_reason reflects surface transition."""
        machine = ComplianceStateMachine()
        # Prime with red_track
        machine.evaluate_frame_state(
            _detections([_box("red_track")]), _osm()
        )
        # Surface changes to grey_sidewalk
        machine.evaluate_frame_state(
            _detections([_box("grey_sidewalk")]), _osm()
        )

        finished = spans.get_finished_spans()
        span = finished[-1]

        assert span.attributes.get("router.vlm_triggered") is True
        assert "surface_transition" in span.attributes.get("router.trigger_reason")


# --------------------------------------------------------------------------- #
# Test 6: Fallback handling
# --------------------------------------------------------------------------- #
class TestFallbackHandling:
    def test_vlm_none_uses_yolo_and_osm(self):
        """VLM=None: state from YOLO + OSM matrix only."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("grey_sidewalk")])
        result = machine.evaluate_frame_state(detections, _osm(), vlm_response=None)

        assert result.active_compliance_state == ComplianceState.VIOLATION
        assert result.compliance_audit is None

    def test_osm_fallback_ignores_prohibited_road(self):
        """OSM fallback mode: is_prohibited_road is ignored."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        policy = _osm(is_prohibited=True, is_fallback=True)
        result = machine.evaluate_frame_state(detections, policy)

        # OSM in fallback -> is_prohibited_road ignored
        # red_track detected -> SAFE
        assert result.active_compliance_state == ComplianceState.SAFE

    def test_osm_fallback_still_detects_grey_sidewalk(self):
        """OSM in fallback: YOLO grey_sidewalk still triggers VIOLATION."""
        machine = ComplianceStateMachine()
        detections = _detections([_box("grey_sidewalk"), _box("scooter")])
        policy = _osm(is_fallback=True)
        result = machine.evaluate_frame_state(detections, policy)

        assert result.active_compliance_state == ComplianceState.VIOLATION

    def test_rule_based_fallback_audit_integration(self):
        """When VLM is None and OSM is fallback, rule-based logic applies."""
        from src.vlm.qwen_engine import rule_based_fallback_audit

        machine = ComplianceStateMachine()
        detections = _detections([_box("grey_sidewalk"), _box("scooter")])
        policy = _osm(is_fallback=True)

        # First call to evaluate_frame_state uses the logic matrix directly
        machine.evaluate_frame_state(detections, policy, vlm_response=None)

        # Also verify rule_based_fallback_audit works standalone
        fallback = rule_based_fallback_audit(detections, policy)
        assert fallback.risk_level == RiskLevel.HIGH
        assert fallback.zone_classification == ZoneClassification.PEDESTRIAN_SIDEWALK


# --------------------------------------------------------------------------- #
# Test 7: FrameState construction
# --------------------------------------------------------------------------- #
class TestFrameStateOutput:
    def test_frame_id_propagated(self):
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track")])
        result = machine.evaluate_frame_state(detections, _osm())
        assert result.frame_id == 42

    def test_visual_detections_propagated(self):
        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track"), _box("scooter")])
        result = machine.evaluate_frame_state(detections, _osm())
        assert len(result.visual_detections) == 2

    def test_geospatial_policy_propagated(self):
        machine = ComplianceStateMachine()
        policy = _osm(highway_type="motorway", maxspeed_kmh=80)
        result = machine.evaluate_frame_state(_detections([_box("red_track")]), policy)
        assert result.geospatial_policy.highway_type == "motorway"
        assert result.geospatial_policy.maxspeed_kmh == 80

    def test_compliance_audit_none_when_vlm_none(self):
        machine = ComplianceStateMachine()
        result = machine.evaluate_frame_state(
            _detections([_box("red_track")]), _osm(), vlm_response=None
        )
        assert result.compliance_audit is None

    def test_timestamp_is_set(self):
        machine = ComplianceStateMachine()
        result = machine.evaluate_frame_state(
            _detections([_box("red_track")]), _osm()
        )
        assert result.timestamp is not None

    def test_is_violation_property(self):
        """FrameState.is_violation convenience accessor."""
        from src.schemas.frame import FrameState

        machine = ComplianceStateMachine()
        result = machine.evaluate_frame_state(
            _detections([_box("grey_sidewalk")]), _osm()
        )
        assert isinstance(result, FrameState)
        assert result.is_violation is True

    def test_is_violation_false_for_safe(self):
        machine = ComplianceStateMachine()
        result = machine.evaluate_frame_state(
            _detections([_box("red_track")]), _osm()
        )
        assert result.is_violation is False
