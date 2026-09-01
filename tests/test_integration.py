"""End-to-end integration test stitching schemas, telemetry and exceptions."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from src.config import get_settings
from src.exceptions import PerceptionEngineError
from src.schemas import (
    BoundingBox,
    ComplianceState,
    FrameState,
    OSMRoadPolicy,
    RiskLevel,
    SpatialCoordinate,
    ViolationDetail,
    ViolationType,
    VLMAuditResponse,
    ZoneClassification,
)
from src.telemetry import get_logger, trace_span


def test_end_to_end_pipeline(spans, log_stream):
    settings = get_settings()
    log = get_logger("integration")

    @trace_span(name="detect-and-audit", attributes={"domain": "compliance", "threshold": settings.yolo_confidence_threshold})
    def run(frame_id: int) -> FrameState:
        # --- perception -----------------------------------------------------
        box = BoundingBox(
            xmin=0.2, ymin=0.2, xmax=0.5, ymax=0.6,
            confidence=0.9, class_id=0, class_name="scooter",
        )
        # --- geospatial -----------------------------------------------------
        gps = SpatialCoordinate(
            latitude=25.2048, longitude=55.2708, altitude=12.4,
            accuracy=2.5, timestamp=dt.datetime.now(dt.timezone.utc),
        )
        policy = OSMRoadPolicy(
            way_id=98765, highway_type="path", maxspeed_kmh=15,
            is_prohibited_road=False, query_lat_lon=(gps.latitude, gps.longitude),
        )
        # --- vlm / compliance ----------------------------------------------
        if box.confidence >= settings.yolo_confidence_threshold:
            audit = VLMAuditResponse(
                zone_classification=ZoneClassification.PEDESTRIAN_SIDEWALK,
                dismount_required=True,
                risk_level=RiskLevel.CRITICAL,
                violations=[
                    ViolationDetail(
                        violation_type=ViolationType.NO_DISMOUNT,
                        legal_reference="Dubai RTA Resolution No. 13 (2022) Article 4",
                        fine_amount_aed=300,
                    )
                ],
                hud_warning_text="Dismount required - pedestrian zone",
            )
            state = ComplianceState.VIOLATION
        else:
            audit, state = None, ComplianceState.SAFE

        log.info(
            "frame evaluated",
            extra={"frame_id": frame_id, "state": state.value, "boxes": 1},
        )
        return FrameState(
            frame_id=frame_id,
            timestamp=dt.datetime.now(dt.timezone.utc),
            visual_detections=[box],
            geospatial_policy=policy,
            compliance_audit=audit,
            active_compliance_state=state,
            hud_alert_message=audit.hud_warning_text if audit else "All clear",
            fine_risk_aed=audit.max_fine_aed,
        )

    frame = run(101)
    assert frame.is_violation is True
    assert frame.active_compliance_state is ComplianceState.VIOLATION
    assert frame.fine_risk_aed == 300
    assert frame.compliance_audit.hud_warning_text == "Dismount required - pedestrian zone"
    # JSON round-trip preserves enums as their string values.
    dumped = frame.model_dump_json(by_alias=True)
    restored = FrameState.model_validate_json(dumped)
    assert restored.active_compliance_state is ComplianceState.VIOLATION

    span = spans.get_finished_spans()[0]
    assert span.attributes["domain"] == "compliance"
    assert span.attributes["execution.duration_ms"] is not None

    line = log_stream.getvalue().strip().splitlines()[-1]
    record = json.loads(line)
    ctx = span.get_span_context()
    assert record["trace_id"] == format(ctx.trace_id, "032x")
    assert record["span_id"] == format(ctx.span_id, "016x")
    assert record["frame_id"] == 101
    assert record["state"] == "VIOLATION"


def test_pipeline_records_error_on_failure(spans):
    @trace_span(name="perceive-fail")
    def perceive() -> None:
        raise PerceptionEngineError("GPU memory exhausted", details={"gpu": 0})

    with pytest.raises(PerceptionEngineError):
        perceive()

    span = spans.get_finished_spans()[0]
    assert span.status.status_code.value == 2  # StatusCode.ERROR == 2
    assert span.attributes["error.type"] == "PerceptionEngineError"
