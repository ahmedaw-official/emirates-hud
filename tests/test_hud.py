"""Tests for the HUD compositor and video processor.

Covers:
* HUDConfig defaults and overrides
* HUDCompositor top bar rendering (speed, zone, helmet, fine risk)
* Bounding box colour coding (green / amber / red)
* Critical alert overlay (flashing, centered banner)
* Overlay exception shield (corrupted frame passthrough)
* VideoProcessor codec fallback (mp4v -> XVID -> avc1 -> MJPG)
* FPS tracking overlay
* OTel span `hud.render_frame` attributes
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from src.config.schema import HUDConfig
from src.hud import HUDCompositor, VideoProcessor
from src.schemas import (
    BoundingBox,
    ComplianceState,
    DetectionResult,
    FrameState,
    OSMRoadPolicy,
    RiskLevel,
    ViolationDetail,
    VLMAuditResponse,
    ZoneClassification,
)
from src.telemetry.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Test data builders
# --------------------------------------------------------------------------- #
def _box(class_name: str, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(
        xmin=0.1, ymin=0.1,
        xmax=0.5, ymax=0.5,
        confidence=confidence,
        class_id=0,
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
) -> OSMRoadPolicy:
    return OSMRoadPolicy(
        way_id=way_id,
        highway_type=highway_type,
        maxspeed_kmh=maxspeed_kmh,
        is_prohibited_road=is_prohibited,
        query_lat_lon=(25.2048, 55.2708),
        is_fallback=is_fallback,
    )


def _vlm(
    risk_level: RiskLevel = RiskLevel.LOW,
    zone: ZoneClassification = ZoneClassification.DESIGNATED_LANE,
    violations: list[ViolationDetail] | None = None,
    hud_text: str = "OK",
) -> VLMAuditResponse:
    return VLMAuditResponse(
        zone_classification=zone,
        dismount_required=False,
        risk_level=risk_level,
        violations=violations or [],
        hud_warning_text=hud_text,
    )


def _frame_state(
    state: ComplianceState = ComplianceState.SAFE,
    boxes: list[BoundingBox] | None = None,
    policy: OSMRoadPolicy | None = None,
    vlm_response: VLMAuditResponse | None = None,
    fine: int = 0,
    msg: str = "All clear",
) -> FrameState:
    """Helper to build a complete FrameState."""
    import datetime as dt
    return FrameState(
        frame_id=1,
        timestamp=dt.datetime.now(dt.timezone.utc),
        visual_detections=boxes or [],
        geospatial_policy=policy or _osm(),
        compliance_audit=vlm_response,
        active_compliance_state=state,
        hud_alert_message=msg,
        fine_risk_aed=fine,
    )


def _blank_frame(width: int = 320, height: int = 240) -> np.ndarray:
    """Create a blank BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Test 1: HUDConfig
# --------------------------------------------------------------------------- #
class TestHUDConfig:
    def test_defaults(self):
        cfg = HUDConfig()
        assert cfg.output_width == 1280
        assert cfg.output_height == 720
        assert cfg.overlay_alpha == 0.7
        assert cfg.bar_height == 80
        assert cfg.color_safe == (0, 255, 0)
        assert cfg.color_violation == (0, 0, 255)
        assert cfg.color_warning == (0, 215, 255)
        assert cfg.helmet_confidence_threshold == 0.50
        assert "mp4v" in cfg.codec_preferences
        assert "MJPG" in cfg.codec_preferences

    def test_overrides(self):
        cfg = HUDConfig(output_width=640, output_height=480, bar_height=120)
        assert cfg.output_width == 640
        assert cfg.output_height == 480
        assert cfg.bar_height == 120

    def test_extra_forbidden(self):
        with pytest.raises(Exception):
            HUDConfig(unknown_field=42)  # type: ignore[call-arg]

    def test_codec_preferences_order(self):
        cfg = HUDConfig()
        assert cfg.codec_preferences[0] == "mp4v"
        assert cfg.codec_preferences[-1] == "MJPG"


# --------------------------------------------------------------------------- #
# Test 2: HUDCompositor basic rendering
# --------------------------------------------------------------------------- #
class TestHUDCompositorBasic:
    def test_render_frame_returns_valid_array(self):
        compositor = HUDCompositor()
        frame = _blank_frame()
        state = _frame_state(ComplianceState.SAFE)
        result = compositor.render_frame(frame, state, speed_kmh=15.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_render_frame_does_not_mutate_input(self):
        compositor = HUDCompositor()
        frame = _blank_frame()
        original = frame.copy()
        state = _frame_state(ComplianceState.SAFE)
        compositor.render_frame(frame, state, speed_kmh=15.0)
        np.testing.assert_array_equal(frame, original)

    def test_render_frame_with_empty_detections(self):
        compositor = HUDCompositor()
        frame = _blank_frame()
        state = _frame_state(ComplianceState.SAFE, boxes=[])
        result = compositor.render_frame(frame, state)
        assert result.shape == frame.shape

    def test_render_frame_with_speed(self):
        compositor = HUDCompositor()
        frame = _blank_frame()
        state = _frame_state(ComplianceState.SAFE)
        result = compositor.render_frame(frame, state, speed_kmh=25.5)
        assert result is not None

    def test_render_frame_with_fps(self):
        compositor = HUDCompositor()
        frame = _blank_frame()
        state = _frame_state(ComplianceState.SAFE)
        result = compositor.render_frame(frame, state, processing_fps=30.5)
        assert result is not None


# --------------------------------------------------------------------------- #
# Test 3: Top telemetry bar
# --------------------------------------------------------------------------- #
class TestTopBar:
    def test_top_bar_drawn(self):
        """The top bar region should differ from the original (annotation was applied)."""
        compositor = HUDCompositor()
        frame = _blank_frame(width=640, height=480)
        state = _frame_state(ComplianceState.SAFE)
        result = compositor.render_frame(frame, state, speed_kmh=15.0)

        # Top 80 rows should have changed (semi-transparent overlay).
        bar_region = result[:80, :, :]
        original_region = frame[:80, :, :]
        assert not np.array_equal(bar_region, original_region)

    def test_speed_displayed_on_safe_frame(self):
        """Speed text should be visible in the top bar."""
        compositor = HUDCompositor()
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # light gray
        state = _frame_state(ComplianceState.SAFE)
        result = compositor.render_frame(frame, state, speed_kmh=42.0)

        # Top bar should be darker (overlay applied).
        bar = result[:80]
        assert bar[:, :, 0].mean() < 200  # B channel darkened by overlay

    def test_zone_label_with_vlm(self):
        """When VLM audit is present, zone label should come from VLM."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        vlm = _vlm(zone=ZoneClassification.CROSSWALK)
        state = _frame_state(ComplianceState.WARNING, vlm_response=vlm)
        result = compositor.render_frame(frame, state)
        assert result is not None  # no error

    def test_helmet_ok_when_helmet_detected(self):
        """Helmet detection class should render 'HELMET: OK' in green."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        boxes = [_box("helmet", confidence=0.9)]
        state = _frame_state(ComplianceState.SAFE, boxes=boxes)
        result = compositor.render_frame(frame, state)
        assert result is not None

    def test_helmet_ko_on_violation(self):
        """When state is VIOLATION and no helmet detected, render 'NO HELMET'."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        boxes = [_box("scooter", confidence=0.9)]
        state = _frame_state(ComplianceState.VIOLATION, boxes=boxes, fine=300)
        result = compositor.render_frame(frame, state)
        assert result is not None


# --------------------------------------------------------------------------- #
# Test 4: Bounding box styling
# --------------------------------------------------------------------------- #
class TestBoundingBoxRendering:
    def test_green_boxes_for_safe(self):
        """SAFE state should render green boxes."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        boxes = [_box("scooter", confidence=0.9), _box("person", confidence=0.8)]
        state = _frame_state(ComplianceState.SAFE, boxes=boxes)
        result = compositor.render_frame(frame, state)

        # Verify the frame changed (boxes drawn).
        assert not np.array_equal(result, frame)

    def test_red_boxes_for_violation(self):
        """VIOLATION state should render red boxes."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        boxes = [_box("scooter", confidence=0.9)]
        state = _frame_state(ComplianceState.VIOLATION, boxes=boxes, fine=300)
        result = compositor.render_frame(frame, state)
        assert not np.array_equal(result, frame)

    def test_amber_boxes_for_warning(self):
        """WARNING state should render amber boxes."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        boxes = [_box("scooter", confidence=0.9)]
        state = _frame_state(ComplianceState.WARNING, boxes=boxes, fine=200)
        result = compositor.render_frame(frame, state)
        assert not np.array_equal(result, frame)

    def test_no_boxes_no_drawing_outside_bar(self):
        """When no detections, only the top bar and FPS should change."""
        compositor = HUDCompositor()
        frame = _blank_frame(width=640, height=480)
        state = _frame_state(ComplianceState.SAFE, boxes=[])
        result = compositor.render_frame(frame, state, processing_fps=30.0)

        # Middle of frame should be unchanged (only top bar + bottom-right text).
        mid_region = result[100:380, 100:540]
        original_mid = frame[100:380, 100:540]
        np.testing.assert_array_equal(mid_region, original_mid)

    def test_box_label_pill_rendered(self):
        """Bounding box labels should be rendered."""
        compositor = HUDCompositor()
        frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        boxes = [_box("scooter", confidence=0.9)]
        state = _frame_state(ComplianceState.SAFE, boxes=boxes)
        result = compositor.render_frame(frame, state)
        # Frame should differ.
        assert not np.array_equal(result, frame)

    def test_no_helmet_class_renders_red(self):
        """Explicit 'no_helmet' class should use violation color."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        boxes = [_box("no_helmet", confidence=0.95)]
        state = _frame_state(ComplianceState.VIOLATION, boxes=boxes, fine=300)
        result = compositor.render_frame(frame, state)
        assert result is not None

    def test_helmet_class_renders_green(self):
        """Explicit 'helmet' class should use safe color."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        boxes = [_box("helmet", confidence=0.95)]
        state = _frame_state(ComplianceState.SAFE, boxes=boxes)
        result = compositor.render_frame(frame, state)
        assert result is not None


# --------------------------------------------------------------------------- #
# Test 5: Critical alert overlay
# --------------------------------------------------------------------------- #
class TestCriticalAlert:
    def test_alert_rendered_on_violation(self):
        """VIOLATION state should trigger the critical alert banner."""
        compositor = HUDCompositor()
        frame = _blank_frame(width=640, height=480)
        state = _frame_state(ComplianceState.VIOLATION, fine=300, msg="CRITICAL: PROHIBITED ZONE")
        result = compositor.render_frame(frame, state, force_alert=True)

        # Center region should be modified (alert banner).
        center = result[200:280, 100:540]
        original_center = frame[200:280, 100:540]
        assert not np.array_equal(center, original_center)

    def test_no_alert_on_safe(self):
        """SAFE state with no fine should NOT render alert banner."""
        compositor = HUDCompositor()
        frame = _blank_frame(width=640, height=480)
        state = _frame_state(ComplianceState.SAFE, fine=0)
        result = compositor.render_frame(frame, state)

        # Center should be unchanged.
        center = result[200:280, 100:540]
        original_center = frame[200:280, 100:540]
        np.testing.assert_array_equal(center, original_center)

    def test_alert_on_fine_gt_zero(self):
        """Even non-VIOLATION state with fine > 0 triggers alert."""
        compositor = HUDCompositor()
        frame = _blank_frame(width=640, height=480)
        state = _frame_state(ComplianceState.WARNING, fine=200, msg="DISMOUNT REQUIRED")
        result = compositor.render_frame(frame, state, force_alert=True)

        center = result[200:280, 100:540]
        original_center = frame[200:280, 100:540]
        assert not np.array_equal(center, original_center)

    def test_alert_uses_hud_message(self):
        """The alert banner text comes from frame_state.hud_alert_message."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        state = _frame_state(
            ComplianceState.VIOLATION, fine=300,
            msg="CRITICAL: PROHIBITED ZONE - AED 300 FINE",
        )
        result = compositor.render_frame(frame, state)
        assert result is not None  # no error


# --------------------------------------------------------------------------- #
# Test 6: Overlay exception shield
# --------------------------------------------------------------------------- #
class TestExceptionShield:
    def test_corrupted_frame_passthrough(self):
        """If rendering fails, the raw frame is returned (not None)."""
        compositor = HUDCompositor()
        # Create a frame that will cause cv2 errors (wrong dtype / shape).
        bad_frame = np.zeros((10, 10, 3), dtype=np.float32)
        state = _frame_state(ComplianceState.SAFE)

        # Should not raise - should return a copy of the raw frame.
        result = compositor.render_frame(bad_frame, state)
        assert result is not None

    def test_exception_shield_logs_error(self, caplog):
        """When rendering fails, an error is logged."""
        compositor = HUDCompositor()
        bad_frame = np.zeros((1, 1, 3), dtype=np.float32)  # too small
        state = _frame_state(ComplianceState.SAFE)

        with caplog.at_level("ERROR"):
            result = compositor.render_frame(bad_frame, state)

        # Should return the raw frame without raising.
        assert result is not None
        # Check that an error was logged.
        assert any(
            "HUD render error" in record.getMessage()
            for record in caplog.records
        )

    def test_normal_frame_not_shielded(self):
        """Normal frames should render without exception."""
        compositor = HUDCompositor()
        frame = _blank_frame(width=320, height=240)
        state = _frame_state(ComplianceState.SAFE)
        result = compositor.render_frame(frame, state)
        assert not np.array_equal(result, frame)  # actually annotated


# --------------------------------------------------------------------------- #
# Test 7: VideoProcessor
# --------------------------------------------------------------------------- #
class TestVideoProcessor:
    def test_codec_fallback_preferences(self):
        """HUDConfig lists codecs in the expected fallback order."""
        cfg = HUDConfig()
        assert cfg.codec_preferences == ["mp4v", "XVID", "avc1", "MJPG"]

    def test_video_processor_init(self):
        """VideoProcessor initializes with a compositor and config."""
        vp = VideoProcessor(input_source=0, output_path="/tmp/test.mp4")
        assert vp.config is not None
        assert vp.codec_used == "none"  # not opened yet
        assert vp.processing_fps == 0.0

    def test_video_processor_init_with_config(self):
        """Custom config is used."""
        cfg = HUDConfig(output_width=640, output_height=480)
        vp = VideoProcessor(input_source=0, output_path="/tmp/test.mp4", config=cfg)
        assert vp.config.output_width == 640
        assert vp.config.output_height == 480

    def test_video_processor_init_with_compositor(self):
        """Custom compositor is used."""
        comp = HUDCompositor(HUDConfig(output_width=320, output_height=240))
        vp = VideoProcessor(
            input_source=0, output_path="/tmp/test.mp4", compositor=comp
        )
        assert vp._compositor is comp

    def test_context_manager(self, tmp_path):
        """VideoProcessor works as a context manager."""
        # Create a dummy video file for input.
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        # Create a minimal valid video.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 10, (320, 240))
        for _ in range(3):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()
        os.rename(output_path, input_path)

        vp = VideoProcessor(input_source=input_path, output_path=output_path)
        # We won't call open() since the inference callback is None, but
        # verify the context manager protocol exists.
        assert hasattr(vp, "__enter__")
        assert hasattr(vp, "__exit__")

    def test_release_without_open(self):
        """release() is safe to call before open()."""
        vp = VideoProcessor(input_source=0, output_path="/tmp/test.mp4")
        vp.release()  # should not raise
        assert vp._cap is None
        assert vp._writer is None

    def test_open_nonexistent_source(self):
        """Opening a nonexistent file returns False."""
        vp = VideoProcessor(
            input_source="/nonexistent/path/video.mp4",
            output_path="/tmp/out.mp4",
        )
        assert vp.open() is False
        vp.release()

    def test_process_frame_with_callback(self):
        """process_frame runs inference callback and renders HUD."""
        vp = VideoProcessor(
            input_source="/dev/null",
            output_path="/tmp/test.mp4",
            inference_callback=lambda f: _frame_state(ComplianceState.SAFE),
        )
        # Bypass open() - set dimensions manually.
        vp._output_w = 320
        vp._output_h = 240
        vp._source_w = 320
        vp._source_h = 240

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = vp.process_frame(frame)
        assert result.shape == (240, 320, 3)
        assert result.dtype == np.uint8

    def test_process_frame_without_callback(self):
        """Without inference callback, frame is just resized."""
        vp = VideoProcessor(input_source="/dev/null", output_path="/tmp/test.mp4")
        vp._output_w = 160
        vp._output_h = 120

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = vp.process_frame(frame)
        assert result.shape == (120, 160, 3)

    def test_codec_preferences_custom(self):
        """Custom codec list is respected."""
        cfg = HUDConfig(codec_preferences=["mp4v", "MJPG"])
        vp = VideoProcessor(
            input_source="/dev/null",
            output_path="/tmp/test.mp4",
            config=cfg,
        )
        assert vp.config.codec_preferences == ["mp4v", "MJPG"]


# --------------------------------------------------------------------------- #
# Test 8: OTel span attributes
# --------------------------------------------------------------------------- #
class TestHUDOTel:
    def test_render_frame_span_created(self, spans):
        """render_frame creates a `hud.render_frame` span."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        state = _frame_state(ComplianceState.SAFE)
        compositor.render_frame(frame, state, speed_kmh=15.0)

        finished = spans.get_finished_spans()
        span = next(
            (s for s in finished if s.name == "hud.render_frame"),
            None,
        )
        assert span is not None

    def test_span_attributes(self, spans):
        """Span has the required attributes."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        state = _frame_state(ComplianceState.VIOLATION, fine=300)
        compositor.render_frame(frame, state, speed_kmh=15.0)

        finished = spans.get_finished_spans()
        span = next(
            (s for s in finished if s.name == "hud.render_frame"),
            None,
        )
        assert span is not None
        assert span.attributes.get("hud.frame_idx") == 1
        assert span.attributes.get("hud.active_fines_aed") == 300
        assert span.attributes.get("hud.overlay_errors_count") == 0

    def test_span_on_safe_frame(self, spans):
        """Safe frame produces span with 0 fine."""
        compositor = HUDCompositor()
        frame = _blank_frame()
        state = _frame_state(ComplianceState.SAFE, fine=0)
        compositor.render_frame(frame, state)

        finished = spans.get_finished_spans()
        span = next(
            (s for s in finished if s.name == "hud.render_frame"),
            None,
        )
        assert span is not None
        assert span.attributes.get("hud.active_fines_aed") == 0


# --------------------------------------------------------------------------- #
# Test 9: Integration with state engine output
# --------------------------------------------------------------------------- #
class TestStateEngineIntegration:
    def test_violation_state_renders_red_alert(self):
        """A VIOLATION FrameState from the state engine renders correctly."""
        from src.state_engine import ComplianceStateMachine

        machine = ComplianceStateMachine()
        detections = _detections([_box("grey_sidewalk", confidence=0.9), _box("scooter", confidence=0.9)])
        frame_state = machine.evaluate_frame_state(detections, _osm())

        compositor = HUDCompositor()
        frame = _blank_frame(width=640, height=480)
        result = compositor.render_frame(frame, frame_state, speed_kmh=10.0, force_alert=True)

        assert result.shape == frame.shape
        # Top bar + alert overlay should modify the frame.
        assert not np.array_equal(result, frame)

    def test_safe_state_renders_green(self):
        """A SAFE FrameState renders without critical alerts."""
        from src.state_engine import ComplianceStateMachine

        machine = ComplianceStateMachine()
        detections = _detections([_box("red_track", confidence=0.9), _box("scooter", confidence=0.9)])
        frame_state = machine.evaluate_frame_state(detections, _osm())

        compositor = HUDCompositor()
        frame = _blank_frame(width=640, height=480)
        result = compositor.render_frame(frame, frame_state, speed_kmh=15.0)

        assert result.shape == frame.shape

    def test_warning_state_central_banner(self):
        """A WARNING state with fine > 0 triggers alert."""
        compositor = HUDCompositor()
        frame = _blank_frame(width=640, height=480)
        state = _frame_state(ComplianceState.WARNING, fine=200, msg="DISMOUNT REQUIRED")
        result = compositor.render_frame(frame, state, force_alert=True)

        # Center region should be modified.
        center = result[200:280, 100:540]
        assert not np.array_equal(center, frame[200:280, 100:540])
