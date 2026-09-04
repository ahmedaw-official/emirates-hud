"""Tests for the Multimodal Regulatory Engine (src/vlm/).

Covers:
* System prompts encoding RTA Resolution No. 13 (2022)
* Prompt formatting (detection tags, OSM context)
* Qwen2VLEngine construction and model loading
* audit_frame happy path with mock VLM model/processor
* Malformed-JSON fallback (regex recovery + safe default)
* VLM/CUDA failure -> rule-based deterministic audit
* OTel span ``vlm.audit_frame`` attributes
* rule_based_fallback_audit across all RTA/ITC scenarios
* JSON recovery helpers (_strip_markdown_fence, _extract_json_object)
* Fine-tuning config (LoRA r=16, alpha=32, target modules)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.schemas import (
    BoundingBox,
    DetectionResult,
    OSMRoadPolicy,
    RiskLevel,
    ViolationType,
    VLMAuditResponse,
    ZoneClassification,
)
from src.vlm import DEFAULT_FALLBACK_WARNING, SAFE_DEFAULT_AUDIT, Qwen2VLEngine
from src.vlm.finetune import (
    LORA_ALPHA,
    LORA_RANK,
    TARGET_MODULES,
    build_lora_config,
    load_jsonl_dataset,
)
from src.vlm.prompts import (
    JSON_SCHEMA_DESCRIPTION,
    RTA_SYSTEM_PROMPT,
    format_audit_prompt,
    format_detection_tags,
    format_osm_context,
)
from src.vlm.qwen_engine import (
    _attempt_json_recovery,
    _extract_json_object,
    _strip_markdown_fence,
    rule_based_fallback_audit,
)

# --------------------------------------------------------------------------- #
# Shared valid JSON outputs from a (simulated) VLM
# --------------------------------------------------------------------------- #
VALID_COMPLIANT_JSON = json.dumps({
    "zone_classification": "DESIGNATED_LANE",
    "dismount_required": False,
    "risk_level": "LOW",
    "violations": [],
    "hud_warning_text": "Safe to proceed.",
})

VALID_VIOLATION_JSON = json.dumps({
    "zone_classification": "PEDESTRIAN_SIDEWALK",
    "dismount_required": True,
    "risk_level": "HIGH",
    "violations": [
        {
            "violation_type": "SIDEWALK_DISOBEDENCE",
            "legal_reference": "Dubai RTA Resolution No. 13 (2022) Article 4",
            "fine_amount_aed": 250,
        }
    ],
    "hud_warning_text": "Riding on sidewalk prohibited - AED 250.",
})

VALID_CRITICAL_JSON = json.dumps({
    "zone_classification": "DESIGNATED_LANE",
    "dismount_required": False,
    "risk_level": "CRITICAL",
    "violations": [
        {
            "violation_type": "PROHIBITED_ROAD_USAGE",
            "legal_reference": "Dubai RTA Resolution No. 13 (2022) Article 5",
            "fine_amount_aed": 300,
        }
    ],
    "hud_warning_text": "Prohibited road - AED 300.",
})


# --------------------------------------------------------------------------- #
# Test helpers - mock VLM model and processor
# --------------------------------------------------------------------------- #
class _MockTensor:
    """Minimal tensor stand-in for prompt / completion token counting."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def to(self, device: str | None = None, **kwargs: object) -> _MockTensor:
        return self

    def dim(self) -> int:
        return len(self._shape)

    def __getitem__(self, idx: int) -> str:
        return f"mock_tokens_{idx}"


class _MockInputs(dict):
    """Dict subclass that also exposes keys as attributes (BatchEncoding-like)."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class MockProcessor:
    """Mock HF processor: callable returns dict-like inputs, decode returns text."""

    def __init__(self, decode_output: str, prompt_token_count: int = 12) -> None:
        self._decode_output = decode_output
        self._prompt_token_count = prompt_token_count
        self.call_count = 0
        self.decode_count = 0

    def __call__(self, **kwargs: object) -> _MockInputs:
        self.call_count += 1
        inputs = _MockInputs()
        inputs["input_ids"] = _MockTensor(shape=(1, self._prompt_token_count))
        inputs["attention_mask"] = _MockTensor(shape=(1, self._prompt_token_count))
        return inputs

    def decode(self, ids: object, skip_special_tokens: bool = True) -> str:
        self.decode_count += 1
        return self._decode_output


class MockModel:
    """Mock HF model: generate() returns a tensor-like result or raises."""

    def __init__(
        self,
        decode_output: str = VALID_COMPLIANT_JSON,
        completion_token_count: int = 45,
        generate_error: Exception | None = None,
    ) -> None:
        self._completion_token_count = completion_token_count
        self._generate_error = generate_error
        self.generate_called = False
        self.generate_kwargs: dict | None = None

    def generate(self, **kwargs: object) -> _MockTensor:
        self.generate_called = True
        self.generate_kwargs = kwargs
        if self._generate_error is not None:
            raise self._generate_error
        return _MockTensor(shape=(1, self._completion_token_count))


def _make_scooter_box(confidence: float = 0.95) -> BoundingBox:
    return BoundingBox(
        xmin=0.12, ymin=0.08, xmax=0.45, ymax=0.91,
        confidence=confidence, class_id=0, class_name="scooter",
    )


def _make_detection_result(boxes: list[BoundingBox] | None = None) -> DetectionResult:
    return DetectionResult(
        boxes=boxes or [_make_scooter_box()],
        frame_id=1,
        timestamp_ms=1_700_000_000_000,
        processing_time_ms=45.3,
    )


def _make_osm_policy(
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


def _make_mock_engine(
    decode_output: str = VALID_COMPLIANT_JSON,
    model_error: Exception | None = None,
    **engine_kwargs: object,
) -> Qwen2VLEngine:
    """Create a Qwen2VLEngine with mock model/processor injected."""
    model = MockModel(decode_output=decode_output, generate_error=model_error)
    processor = MockProcessor(decode_output=decode_output)
    return Qwen2VLEngine(model=model, processor=processor, **engine_kwargs)


# --------------------------------------------------------------------------- #
# Test 1: prompts.py
# --------------------------------------------------------------------------- #
class TestPrompts:
    def test_system_prompt_contains_rta_resolution(self):
        assert "Resolution No. 13" in RTA_SYSTEM_PROMPT
        assert "2022" in RTA_SYSTEM_PROMPT

    def test_system_prompt_contains_article_references(self):
        for article in ["Article 4", "Article 5", "Article 6"]:
            assert article in RTA_SYSTEM_PROMPT

    def test_system_prompt_contains_fine_amounts(self):
        assert "AED 200" in RTA_SYSTEM_PROMPT
        assert "AED 300" in RTA_SYSTEM_PROMPT

    def test_system_prompt_mentions_abu_dhabi_itc(self):
        assert "Abu Dhabi ITC" in RTA_SYSTEM_PROMPT

    def test_system_prompt_enforces_json_output(self):
        assert "JSON" in RTA_SYSTEM_PROMPT
        assert "zone_classification" in RTA_SYSTEM_PROMPT

    def test_system_prompt_lists_valid_enum_values(self):
        assert "DESIGNATED_LANE" in RTA_SYSTEM_PROMPT
        assert "PEDESTRIAN_SIDEWALK" in RTA_SYSTEM_PROMPT
        assert "CROSSWALK" in RTA_SYSTEM_PROMPT
        assert "LOW" in RTA_SYSTEM_PROMPT
        assert "MEDIUM" in RTA_SYSTEM_PROMPT
        assert "HIGH" in RTA_SYSTEM_PROMPT
        assert "CRITICAL" in RTA_SYSTEM_PROMPT

    def test_json_schema_description(self):
        assert "zone_classification" in JSON_SCHEMA_DESCRIPTION
        assert "dismount_required" in JSON_SCHEMA_DESCRIPTION
        assert "risk_level" in JSON_SCHEMA_DESCRIPTION
        assert "violations" in JSON_SCHEMA_DESCRIPTION
        assert "hud_warning_text" in JSON_SCHEMA_DESCRIPTION

    def test_format_detection_tags_empty(self):
        empty = DetectionResult(boxes=[], frame_id=0, timestamp_ms=0, processing_time_ms=0.0)
        assert "[DETECTIONS] None detected." in format_detection_tags(empty)

    def test_format_detection_tags_with_boxes(self):
        dr = _make_detection_result([_make_scooter_box(0.95)])
        result = format_detection_tags(dr)
        assert "scooter" in result
        assert "0.95" in result
        assert "[DETECTION" in result

    def test_format_detection_tags_multiple_classes(self):
        boxes = [
            BoundingBox(xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.3,
                        confidence=0.9, class_id=0, class_name="scooter"),
            BoundingBox(xmin=0.3, ymin=0.1, xmax=0.4, ymax=0.5,
                        confidence=0.8, class_id=1, class_name="person"),
        ]
        dr = _make_detection_result(boxes)
        result = format_detection_tags(dr)
        assert "scooter" in result
        assert "person" in result

    def test_format_osm_context(self):
        policy = _make_osm_policy(highway_type="residential", maxspeed_kmh=30)
        result = format_osm_context(policy)
        assert "residential" in result
        assert "30 km/h" in result
        assert "[OSM_ROAD_POLICY]" in result
        assert "way_id" in result

    def test_format_osm_context_null_maxspeed(self):
        policy = _make_osm_policy(maxspeed_kmh=None)
        result = format_osm_context(policy)
        assert "unknown" in result

    def test_format_audit_prompt_combines_all(self):
        detections = _make_detection_result([_make_scooter_box()])
        policy = _make_osm_policy()
        prompt = format_audit_prompt(detections, policy)
        assert "Resolution No. 13" in prompt
        assert "[DETECTIONS]" in prompt
        assert "[OSM_ROAD_POLICY]" in prompt
        assert "zone_classification" in prompt


# --------------------------------------------------------------------------- #
# Test 2: JSON recovery helpers
# --------------------------------------------------------------------------- #
class TestJSONRecovery:
    def test_strip_markdown_fence_removed(self):
        text = '```json\n{"json": "val"}\n```'
        result = _strip_markdown_fence(text)
        assert "```" not in result

    def test_extract_json_valid(self):
        text = 'Here is the result: {"zone_classification": "DESIGNATED_LANE"}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["zone_classification"] == "DESIGNATED_LANE"

    def test_extract_json_from_markdown(self):
        text = '```json\n{"zone_classification": "LOW", "risk_level": "LOW"}\n```'
        result = _extract_json_object(text)
        assert result is not None
        assert result["zone_classification"] == "LOW"

    def test_extract_json_nested(self):
        text = '{"violations": [{"type": "x", "fine": 100}]}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["violations"][0]["type"] == "x"

    def test_extract_json_no_object(self):
        assert _extract_json_object("no json here") is None

    def test_extract_json_invalid(self):
        assert _extract_json_object('{bad json}') is None

    def test_extract_json_with_trailing_text(self):
        text = '{"a": 1} some trailing text'
        result = _extract_json_object(text)
        assert result is not None
        assert result["a"] == 1

    def test_extract_json_with_leading_text(self):
        text = 'prefix text {"a": 1}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["a"] == 1

    def test_attempt_recovery_valid_json(self):
        result = _attempt_json_recovery(VALID_COMPLIANT_JSON)
        assert result is not None
        assert result.risk_level == RiskLevel.LOW

    def test_attempt_recovery_embedded_json(self):
        text = "Analysis result: " + VALID_VIOLATION_JSON + " Done."
        result = _attempt_json_recovery(text)
        assert result is not None
        assert result.risk_level == RiskLevel.HIGH

    def test_attempt_recovery_invalid_returns_none(self):
        result = _attempt_json_recovery("totally not json")
        assert result is None

    def test_attempt_recovery_missing_required_field(self):
        """JSON missing a required field fails Pydantic validation."""
        bad_json = '{"zone_classification": "LOW", "risk_level": "LOW"}'  # no hud_warning_text
        result = _attempt_json_recovery(bad_json)
        assert result is None

    def test_attempt_recovery_extra_field_rejected(self):
        """VLMAuditResponse has extra='forbid' so extra fields are rejected."""
        bad_json = '{"extra_field": "bad", "zone_classification": "DESIGNATED_LANE", "dismount_required": false, "risk_level": "LOW", "violations": [], "hud_warning_text": "ok"}'
        result = _attempt_json_recovery(bad_json)
        assert result is None


# --------------------------------------------------------------------------- #
# Test 3: rule_based_fallback_audit
# --------------------------------------------------------------------------- #
class TestRuleBasedFallback:
    def test_clean_road_no_violations(self):
        policy = _make_osm_policy(highway_type="residential", maxspeed_kmh=30)
        detections = _make_detection_result([_make_scooter_box()])
        result = rule_based_fallback_audit(detections, policy)

        assert result.risk_level == RiskLevel.LOW
        assert len(result.violations) == 0
        assert result.zone_classification == ZoneClassification.DESIGNATED_LANE
        assert result.dismount_required is False
        assert "fallback" in result.hud_warning_text.lower()

    def test_prohibited_road_maxspeed_over_60(self):
        policy = _make_osm_policy(highway_type="primary", maxspeed_kmh=80)
        detections = _make_detection_result([_make_scooter_box()])
        result = rule_based_fallback_audit(detections, policy)

        assert result.risk_level == RiskLevel.CRITICAL
        assert any(
            v.violation_type == ViolationType.PROHIBITED_ROAD_USAGE.value
            for v in result.violations
        )
        max_fine = max(v.fine_amount_aed for v in result.violations)
        assert max_fine == 300

    def test_prohibited_road_flag(self):
        policy = _make_osm_policy(highway_type="residential", maxspeed_kmh=40, is_prohibited=True)
        detections = _make_detection_result([_make_scooter_box()])
        result = rule_based_fallback_audit(detections, policy)

        assert result.risk_level == RiskLevel.CRITICAL
        assert any(
            v.violation_type == ViolationType.PROHIBITED_ROAD_USAGE.value
            for v in result.violations
        )

    def test_maxspeed_at_threshold_not_prohibited(self):
        """maxspeed == 60 should NOT trigger prohibition (strictly greater than)."""
        policy = _make_osm_policy(highway_type="primary", maxspeed_kmh=60, is_prohibited=False)
        detections = _make_detection_result([_make_scooter_box()])
        result = rule_based_fallback_audit(detections, policy)

        assert not any(
            v.violation_type == ViolationType.PROHIBITED_ROAD_USAGE.value
            for v in result.violations
        )

    @pytest.mark.parametrize("hw_type", ["footway", "pedestrian", "steps"])
    def test_pedestrian_highway_types(self, hw_type: str):
        policy = _make_osm_policy(
            highway_type=hw_type, maxspeed_kmh=30, is_prohibited=True
        )
        detections = _make_detection_result([_make_scooter_box()])
        result = rule_based_fallback_audit(detections, policy)

        assert result.zone_classification == ZoneClassification.PEDESTRIAN_SIDEWALK
        assert result.risk_level == RiskLevel.HIGH
        assert result.dismount_required is True
        assert any(
            v.violation_type == ViolationType.SIDEWALK_DISOBEDENCE.value
            for v in result.violations
        )

    def test_crosswalk_with_scooter_triggers_dismount(self):
        policy = _make_osm_policy(highway_type="residential", maxspeed_kmh=30)
        scooter = _make_scooter_box()
        crosswalk = BoundingBox(
            xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.3,
            confidence=0.9, class_id=1, class_name="zebra_crosswalk",
        )
        detections = _make_detection_result([scooter, crosswalk])
        result = rule_based_fallback_audit(detections, policy)

        assert result.zone_classification == ZoneClassification.CROSSWALK
        assert result.dismount_required is True
        assert result.risk_level == RiskLevel.MEDIUM
        assert any(
            v.violation_type == ViolationType.NO_DISMOUNT.value
            for v in result.violations
        )

    def test_crosswalk_without_scooter(self):
        """No scooter detection means no NO_DISMOUNT violation."""
        policy = _make_osm_policy()
        crosswalk = BoundingBox(
            xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.3,
            confidence=0.9, class_id=1, class_name="zebra_crosswalk",
        )
        detections = _make_detection_result([crosswalk])
        result = rule_based_fallback_audit(detections, policy)

        assert result.zone_classification == ZoneClassification.CROSSWALK
        # No scooter means no NO_DISMOUNT violation
        assert not any(
            v.violation_type == ViolationType.NO_DISMOUNT.value
            for v in result.violations
        )

    def test_person_scooter_conflict(self):
        policy = _make_osm_policy()
        scooter = _make_scooter_box()
        person = BoundingBox(
            xmin=0.5, ymin=0.1, xmax=0.6, ymax=0.5,
            confidence=0.9, class_id=1, class_name="person",
        )
        detections = _make_detection_result([scooter, person])
        result = rule_based_fallback_audit(detections, policy)

        assert any(
            v.violation_type == ViolationType.PEDESTRIAN_CONFLICT.value
            for v in result.violations
        )
        assert result.risk_level == RiskLevel.MEDIUM

    def test_no_detections(self):
        policy = _make_osm_policy()
        detections = DetectionResult(
            boxes=[], frame_id=0, timestamp_ms=0, processing_time_ms=0.0
        )
        result = rule_based_fallback_audit(detections, policy)

        assert result.risk_level == RiskLevel.LOW
        assert len(result.violations) == 0
        assert result.zone_classification == ZoneClassification.DESIGNATED_LANE

    def test_multiple_violations(self):
        policy = _make_osm_policy(
            highway_type="footway", maxspeed_kmh=80, is_prohibited=True
        )
        scooter = _make_scooter_box()
        person = BoundingBox(
            xmin=0.5, ymin=0.1, xmax=0.6, ymax=0.5,
            confidence=0.9, class_id=1, class_name="person",
        )
        detections = _make_detection_result([scooter, person])
        result = rule_based_fallback_audit(detections, policy)

        assert result.risk_level == RiskLevel.CRITICAL
        assert len(result.violations) >= 2

    def test_hud_warning_text_with_violations(self):
        policy = _make_osm_policy(highway_type="footway", is_prohibited=True)
        detections = _make_detection_result([_make_scooter_box()])
        result = rule_based_fallback_audit(detections, policy)

        assert "VIOLATION" in result.hud_warning_text
        assert "AED" in result.hud_warning_text

    def test_hud_warning_text_clean(self):
        policy = _make_osm_policy()
        detections = _make_detection_result([_make_scooter_box()])
        result = rule_based_fallback_audit(detections, policy)

        assert "OK" in result.hud_warning_text


# --------------------------------------------------------------------------- #
# Test 4: Safe default & constants
# --------------------------------------------------------------------------- #
class TestSafeDefault:
    def test_safe_default_audit_fields(self):
        assert SAFE_DEFAULT_AUDIT.risk_level == RiskLevel.MEDIUM
        assert SAFE_DEFAULT_AUDIT.hud_warning_text == DEFAULT_FALLBACK_WARNING
        assert len(SAFE_DEFAULT_AUDIT.violations) == 0
        assert SAFE_DEFAULT_AUDIT.dismount_required is False
        assert SAFE_DEFAULT_AUDIT.zone_classification == ZoneClassification.DESIGNATED_LANE

    def test_fallback_warning_message(self):
        assert DEFAULT_FALLBACK_WARNING == "VLM Schema Parsing Error - Rule Fallback Active"


# --------------------------------------------------------------------------- #
# Test 5: Image conversion
# --------------------------------------------------------------------------- #
class TestImageConversion:
    def test_pil_image_passes_through(self):
        img = Image.new("RGB", (64, 64), color="blue")
        assert Qwen2VLEngine._to_pil_image(img) is img

    def test_numpy_array_converted(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = Qwen2VLEngine._to_pil_image(arr)
        assert isinstance(result, Image.Image)
        assert result.size == (64, 64)

    def test_invalid_image_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported image type"):
            Qwen2VLEngine._to_pil_image("not an image")


# --------------------------------------------------------------------------- #
# Test 6: Engine construction
# --------------------------------------------------------------------------- #
class TestEngineConstruction:
    def test_default_model_id(self):
        engine = Qwen2VLEngine()
        assert engine.model_id == "Qwen/Qwen2-VL-2B-Instruct"

    def test_custom_model_id(self):
        engine = Qwen2VLEngine(model_id="custom/model")
        assert engine.model_id == "custom/model"

    def test_default_adapter_path_is_none(self):
        engine = Qwen2VLEngine()
        assert engine.adapter_path is None

    def test_custom_adapter_path(self):
        engine = Qwen2VLEngine(adapter_path="/path/to/adapter")
        assert engine.adapter_path == "/path/to/adapter"

    def test_is_model_loaded_false_by_default(self):
        engine = Qwen2VLEngine()
        assert engine.is_model_loaded is False

    def test_is_model_loaded_true_with_mock(self):
        engine = Qwen2VLEngine(
            model=MagicMock(), processor=MagicMock()
        )
        assert engine.is_model_loaded is True

    def test_fallback_enabled_default(self):
        engine = Qwen2VLEngine()
        assert engine.fallback_enabled is True

    def test_fallback_disabled(self):
        engine = Qwen2VLEngine(fallback_enabled=False)
        assert engine.fallback_enabled is False

    def test_max_tokens_default(self):
        engine = Qwen2VLEngine()
        assert engine._max_tokens == 512

    def test_temperature_default(self):
        engine = Qwen2VLEngine()
        assert engine._temperature == 0.2


# --------------------------------------------------------------------------- #
# Test 7: audit_frame happy path
# --------------------------------------------------------------------------- #
class TestAuditFrameSuccess:
    def test_successful_audit_compliant(self):
        engine = _make_mock_engine(VALID_COMPLIANT_JSON)
        image = Image.new("RGB", (224, 224), color="blue")
        detections = _make_detection_result([_make_scooter_box()])
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        assert isinstance(result, VLMAuditResponse)
        assert result.risk_level == RiskLevel.LOW
        assert result.zone_classification == ZoneClassification.DESIGNATED_LANE
        assert result.dismount_required is False

    def test_successful_audit_violation(self):
        engine = _make_mock_engine(VALID_VIOLATION_JSON)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result([_make_scooter_box()])
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        assert result.risk_level == RiskLevel.HIGH
        assert result.zone_classification == ZoneClassification.PEDESTRIAN_SIDEWALK
        assert result.dismount_required is True
        assert len(result.violations) == 1

    def test_successful_audit_critical(self):
        engine = _make_mock_engine(VALID_CRITICAL_JSON)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result([_make_scooter_box()])
        policy = _make_osm_policy(highway_type="primary", maxspeed_kmh=80)

        result = engine.audit_frame(image, detections, policy)

        assert result.risk_level == RiskLevel.CRITICAL
        assert len(result.violations) == 1

    def test_model_generate_called(self):
        engine = _make_mock_engine(VALID_COMPLIANT_JSON)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        engine.audit_frame(image, detections, policy)

        # Verify the mock model's generate was called
        assert engine._model.generate_called is True

    def test_processor_called_correctly(self):
        engine = _make_mock_engine(VALID_COMPLIANT_JSON)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        engine.audit_frame(image, detections, policy)

        assert engine._processor.call_count == 1  # processor() called
        # decode is called once
        assert engine._processor.decode_count == 1

    def test_numpy_image_accepted(self):
        engine = _make_mock_engine(VALID_COMPLIANT_JSON)
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        detections = _make_detection_result()
        policy = _make_osm_policy()

        result = engine.audit_frame(arr, detections, policy)
        assert isinstance(result, VLMAuditResponse)


# --------------------------------------------------------------------------- #
# Test 8: audit_frame - malformed JSON fallback
# --------------------------------------------------------------------------- #
class TestAuditFrameJsonFallback:
    def test_malformed_json_recovers_via_regex(self):
        """VLM output with extra text but valid JSON embedded is recovered."""
        messy = 'Here is the analysis: ' + VALID_COMPLIANT_JSON + ' - Done!'
        engine = _make_mock_engine(messy)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        assert result.risk_level == RiskLevel.LOW
        assert result.zone_classification == ZoneClassification.DESIGNATED_LANE

    def test_malformed_json_markdown_fences(self):
        """VLM output wrapped in markdown code fences is recovered."""
        messy = '```json\n' + VALID_VIOLATION_JSON + '\n```'
        engine = _make_mock_engine(messy)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        assert result.risk_level == RiskLevel.HIGH
        assert "fallback" in result.hud_warning_text.lower() or len(result.violations) > 0

    def test_completely_invalid_output_safe_default(self):
        """VLM output with no parseable JSON returns safe default."""
        engine = _make_mock_engine("This is not JSON at all.")
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        assert result.risk_level == RiskLevel.MEDIUM
        assert result.hud_warning_text == DEFAULT_FALLBACK_WARNING
        assert result is SAFE_DEFAULT_AUDIT

    def test_unparseable_json_safe_default(self):
        """VLM output with broken JSON object returns safe default."""
        engine = _make_mock_engine('{"zone_classification": "BROKEN"')
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        assert result.risk_level == RiskLevel.MEDIUM
        assert result.hud_warning_text == DEFAULT_FALLBACK_WARNING

    def test_json_with_extra_fields_rejected_safe_default(self):
        """JSON with unexpected extra fields (extra='forbid') fails and falls back."""
        bad = json.dumps({
            "zone_classification": "DESIGNATED_LANE",
            "dismount_required": False,
            "risk_level": "LOW",
            "violations": [],
            "hud_warning_text": "ok",
            "unexpected_field": "bad",
        })
        engine = _make_mock_engine(bad)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        assert result.risk_level == RiskLevel.MEDIUM
        assert result.hud_warning_text == DEFAULT_FALLBACK_WARNING


# --------------------------------------------------------------------------- #
# Test 9: audit_frame - VLM/CUDA failure fallback
# --------------------------------------------------------------------------- #
class TestAuditFrameModelFailure:
    def test_cuda_oom_triggers_rule_fallback(self):
        """GPU OOM triggers the rule-based deterministic fallback."""
        engine = _make_mock_engine(
            VALID_COMPLIANT_JSON,
            model_error=RuntimeError("CUDA out of memory"),
        )
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result([_make_scooter_box()])
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        # Rule-based fallback for clean road returns LOW risk
        assert result.risk_level == RiskLevel.LOW
        assert "fallback" in result.hud_warning_text.lower()

    def test_model_load_error_triggers_rule_fallback(self):
        """Generic model execution error triggers rule-based fallback."""
        engine = _make_mock_engine(
            VALID_COMPLIANT_JSON,
            model_error=ValueError("model not found"),
        )
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result([_make_scooter_box()])
        policy = _make_osm_policy()

        result = engine.audit_frame(image, detections, policy)

        assert isinstance(result, VLMAuditResponse)
        assert "fallback" in result.hud_warning_text.lower()

    def test_rule_fallback_with_prohibited_road(self):
        """Fallback audit correctly identifies prohibited roads."""
        policy = _make_osm_policy(
            highway_type="primary", maxspeed_kmh=80, is_prohibited=False
        )
        detections = _make_detection_result([_make_scooter_box()])
        result = rule_based_fallback_audit(detections, policy)

        assert result.risk_level == RiskLevel.CRITICAL
        assert any(
            v.violation_type == ViolationType.PROHIBITED_ROAD_USAGE.value
            for v in result.violations
        )

    def test_fallback_disabled_reraises(self):
        """When fallback is disabled, exceptions propagate to the caller."""
        engine = _make_mock_engine(
            VALID_COMPLIANT_JSON,
            model_error=RuntimeError("OOM"),
            fallback_enabled=False,
        )
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        with pytest.raises(RuntimeError, match="OOM"):
            engine.audit_frame(image, detections, policy)


# --------------------------------------------------------------------------- #
# Test 10: OTel span attributes
# --------------------------------------------------------------------------- #
class TestOTelAttributes:
    def test_attributes_on_success(self, spans):
        engine = _make_mock_engine(VALID_COMPLIANT_JSON)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        engine.audit_frame(image, detections, policy)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "vlm.audit_frame")

        assert "Qwen/Qwen2-VL-2B-Instruct" in span.attributes.get("vlm.model_id")
        assert span.attributes.get("vlm.fallback_triggered") is False
        assert span.attributes.get("vlm.risk_level") == RiskLevel.LOW.value
        assert span.attributes.get("vlm.prompt_tokens") == 12
        assert span.attributes.get("vlm.completion_tokens") == 45
        assert span.attributes.get("vlm.latency_ms") is not None
        assert isinstance(span.attributes.get("vlm.latency_ms"), (int, float))

    def test_attributes_on_fallback(self, spans):
        """OTel records fallback_triggered=True on malformed JSON."""
        engine = _make_mock_engine("completely invalid output")
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        engine.audit_frame(image, detections, policy)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "vlm.audit_frame")

        assert span.attributes.get("vlm.fallback_triggered") is True
        assert span.attributes.get("vlm.risk_level") == RiskLevel.MEDIUM.value

    def test_attributes_on_model_failure(self, spans):
        """OTel records fallback_triggered=True on model execution error."""
        engine = _make_mock_engine(
            VALID_COMPLIANT_JSON,
            model_error=RuntimeError("CUDA OOM"),
        )
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        engine.audit_frame(image, detections, policy)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "vlm.audit_frame")

        assert span.attributes.get("vlm.fallback_triggered") is True

    def test_model_id_with_adapter(self, spans):
        """model_id attribute includes adapter path when present."""
        engine = _make_mock_engine(
            VALID_COMPLIANT_JSON,
            adapter_path="/checkpoints/rta_lora",
        )
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        engine.audit_frame(image, detections, policy)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "vlm.audit_frame")
        assert "/checkpoints/rta_lora" in span.attributes.get("vlm.model_id")

    def test_latency_ms_recorded(self, spans):
        engine = _make_mock_engine(VALID_COMPLIANT_JSON)
        image = Image.new("RGB", (224, 224))
        detections = _make_detection_result()
        policy = _make_osm_policy()

        engine.audit_frame(image, detections, policy)

        finished = spans.get_finished_spans()
        span = next(s for s in finished if s.name == "vlm.audit_frame")
        assert span.attributes.get("vlm.latency_ms") is not None
        assert span.attributes.get("vlm.latency_ms") > 0


# --------------------------------------------------------------------------- #
# Test 11: Finetune config
# --------------------------------------------------------------------------- #
class TestFinetuneConfig:
    def test_lora_rank_is_16(self):
        assert LORA_RANK == 16

    def test_lora_alpha_is_32(self):
        assert LORA_ALPHA == 32

    def test_target_modules_include_projection(self):
        for mod in ["q_proj", "v_proj", "k_proj", "o_proj"]:
            assert mod in TARGET_MODULES

    def test_build_lora_config(self):
        config = build_lora_config()
        assert config["r"] == 16
        assert config["lora_alpha"] == 32
        assert "q_proj" in config["target_modules"]
        assert "v_proj" in config["target_modules"]
        assert "k_proj" in config["target_modules"]
        assert "o_proj" in config["target_modules"]

    def test_build_lora_config_returns_copy(self):
        c1 = build_lora_config()
        c2 = build_lora_config()
        assert c1 is not c2
        assert c1["target_modules"] is not c2["target_modules"]

    def test_load_jsonl_dataset_valid(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            '{"image": "a.jpg", "text": "test1"}\n'
            '{"image": "b.jpg", "text": "test2"}\n'
        )
        records = load_jsonl_dataset(str(data_file))
        assert len(records) == 2
        assert records[0]["text"] == "test1"
        assert records[1]["text"] == "test2"

    def test_load_jsonl_dataset_skips_invalid(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            '{"image": "a.jpg", "text": "test1"}\n'
            'not valid json\n'
            '{"image": ""}\n'  # missing text
            '\n'  # empty line
            '{"image": "b.jpg", "text": "test2"}\n'
        )
        records = load_jsonl_dataset(str(data_file))
        assert len(records) == 2
        assert records[0]["text"] == "test1"
        assert records[1]["text"] == "test2"

    def test_finetune_module_importable(self):
        """The finetune module can be imported without unsloth installed."""
        import src.vlm.finetune as ft
        assert hasattr(ft, "finetune_qwen2_vl")
        assert hasattr(ft, "main")
        assert hasattr(ft, "build_lora_config")
