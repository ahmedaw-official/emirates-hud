"""VLM inference engine wrapper.

This module wraps Hugging Face ``Qwen2-VL-2B-Instruct`` behind a small
:class:`Qwen2VLEngine` that:

* lazily imports heavy ML dependencies (transformers / peft / bitsandbytes)
  so the module can be imported and unit-tested without a GPU,
* accepts an injectable ``model`` / ``processor`` for testing,
* enforces UAE RTA / Abu Dhabi ITC micro-mobility regulations,
* degrades gracefully to three fallback tiers:
  1. regex JSON recovery from malformed VLM output,
  2. safe-default response on persistent parse failure,
  3. deterministic rule-based audit on CUDA OOM / model execution errors,
* records OpenTelemetry metrics on span ``vlm.audit_frame``.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np
from PIL import Image
from pydantic import ValidationError

from src.schemas import (
    DetectionResult,
    OSMRoadPolicy,
    RiskLevel,
    ViolationDetail,
    ViolationType,
    VLMAuditResponse,
    ZoneClassification,
)
from src.telemetry import trace_span
from src.telemetry.logging import get_logger
from src.vlm.prompts import format_audit_prompt

__all__ = [
    "DEFAULT_FALLBACK_WARNING",
    "SAFE_DEFAULT_AUDIT",
    "Qwen2VLEngine",
    "rule_based_fallback_audit",
]

logger = get_logger(__name__)

#: Warning text shown when VLM JSON parsing fails after all recovery attempts.
DEFAULT_FALLBACK_WARNING: str = "VLM Schema Parsing Error - Rule Fallback Active"

#: A safe default response used when the VLM produces un-parseable output
#: and regex recovery also fails.
SAFE_DEFAULT_AUDIT: VLMAuditResponse = VLMAuditResponse(
    zone_classification=ZoneClassification.DESIGNATED_LANE,
    dismount_required=False,
    risk_level=RiskLevel.MEDIUM,
    violations=[],
    hud_warning_text=DEFAULT_FALLBACK_WARNING,
)


# --------------------------------------------------------------------------- #
# JSON recovery helpers
# --------------------------------------------------------------------------- #
def _strip_markdown_fence(text: str) -> str:
    """Remove markdown code-fence wrappers (```json ... ```) from text."""
    # Remove opening fences like ` ```json ` or ` ``` `
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from arbitrary text.

    Uses a brace-matching scan so nested objects are handled correctly.
    Falls back to a regex for flat (non-nested) objects if the scan fails.
    """
    text = _strip_markdown_fence(text)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    # Regex fallback for non-nested objects
    flat_pattern = r"\{[^{}]*\}"
    for match in re.finditer(flat_pattern, text):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None


def _attempt_json_recovery(raw_output: str) -> VLMAuditResponse | None:
    """Try to recover a valid VLMAuditResponse from malformed VLM output.

    Returns the parsed response on success, or ``None`` if recovery fails.
    """
    extracted = _extract_json_object(raw_output)
    if extracted is None:
        # Even JSON recovery failed - signal the caller to use safe default.
        logger.error("VLM JSON recovery: no JSON object found in raw output.")
        return None

    try:
        return VLMAuditResponse(**extracted)
    except (ValidationError, TypeError, ValueError) as exc:
        logger.warning("VLM JSON recovery: Pydantic validation failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Deterministic rule-based fallback
# --------------------------------------------------------------------------- #
def rule_based_fallback_audit(
    detection_context: DetectionResult,
    osm_policy: OSMRoadPolicy,
) -> VLMAuditResponse:
    """Construct a compliance audit deterministically from YOLO tags and OSM.

    This is the last-resort fallback invoked when the VLM cannot run
    (CUDA OOM, model load failure, etc.). It applies RTA regulations purely
    from the detection context and road policy.
    """
    # Detect key object classes from the YOLO results.
    class_names = {b.class_name.lower() for b in detection_context.boxes}
    has_scooter = "scooter" in class_names
    has_person = "person" in class_names
    has_crosswalk = any(
        "crosswalk" in c or "zebra" in c for c in class_names
    )
    has_sidewalk = any("sidewalk" in c for c in class_names)
    has_grey_sidewalk = "grey_sidewalk" in class_names

    violations: list[ViolationDetail] = []
    dismount_required = False
    risk_level = RiskLevel.LOW

    rta_art4 = "Dubai RTA Resolution No. 13 (2022) Article 4"
    rta_speed = "Dubai RTA Resolution No. 13 (2022) Article 5"
    rta_crosswalk = "Dubai RTA Resolution No. 13 (2022) Article 6"
    itc_bylaw = "Abu Dhabi ITC Bylaw Section 3.2"

    # --- Prohibited road (maxspeed > 60 km/h) ---
    if (
        osm_policy.maxspeed_kmh is not None
        and osm_policy.maxspeed_kmh > 60
    ):
        violations.append(
            ViolationDetail(
                violation_type=ViolationType.PROHIBITED_ROAD_USAGE,
                legal_reference=rta_speed,
                fine_amount_aed=300,
            )
        )
        risk_level = RiskLevel.CRITICAL

    # --- Prohibited road (is_prohibited_road flag from OSM) ---
    # Skip when the highway is already a pedestrian type (handled below with
    # a higher-fidelity SIDEWALK_DISOBEDENCE classification).
    pedestrian_highways = {"footway", "pedestrian", "steps"}
    if (
        osm_policy.is_prohibited_road
        and osm_policy.highway_type.lower() not in pedestrian_highways
        and not any(
            v.violation_type == ViolationType.PROHIBITED_ROAD_USAGE.value
            for v in violations
        )
    ):
        violations.append(
            ViolationDetail(
                violation_type=ViolationType.PROHIBITED_ROAD_USAGE,
                legal_reference=rta_art4,
                fine_amount_aed=300,
            )
        )
        risk_level = RiskLevel.CRITICAL

    # --- Grey sidewalk detected via YOLO (implies sidewalk riding) ---
    if has_grey_sidewalk:
        violations.append(
            ViolationDetail(
                violation_type=ViolationType.SIDEWALK_DISOBEDENCE,
                legal_reference=rta_art4,
                fine_amount_aed=250,
            )
        )
        if risk_level != RiskLevel.CRITICAL:
            risk_level = RiskLevel.HIGH
        dismount_required = True

    # --- Pedestrian / prohibited highway types (from OSM) ---
    if osm_policy.highway_type.lower() in {"footway", "pedestrian", "steps"}:
        violations.append(
            ViolationDetail(
                violation_type=ViolationType.SIDEWALK_DISOBEDENCE,
                legal_reference=rta_art4,
                fine_amount_aed=250,
            )
        )
        if risk_level != RiskLevel.CRITICAL:
            risk_level = RiskLevel.HIGH
        dismount_required = True

    # --- Crosswalk: must dismount ---
    if has_crosswalk and has_scooter:
        dismount_required = True
        violations.append(
            ViolationDetail(
                violation_type=ViolationType.NO_DISMOUNT,
                legal_reference=rta_crosswalk,
                fine_amount_aed=200,
            )
        )
        if risk_level == RiskLevel.LOW:
            risk_level = RiskLevel.MEDIUM

    # --- Pedestrian conflict ---
    if has_person and has_scooter and not dismount_required:
        violations.append(
            ViolationDetail(
                violation_type=ViolationType.PEDESTRIAN_CONFLICT,
                legal_reference=itc_bylaw,
                fine_amount_aed=200,
            )
        )
        if risk_level == RiskLevel.LOW:
            risk_level = RiskLevel.MEDIUM

    # --- Zone classification ---
    if has_crosswalk:
        zone = ZoneClassification.CROSSWALK
    elif osm_policy.highway_type.lower() in {"footway", "pedestrian", "steps"}:
        zone = ZoneClassification.PEDESTRIAN_SIDEWALK
    elif has_sidewalk:
        zone = ZoneClassification.PEDESTRIAN_SIDEWALK
    else:
        zone = ZoneClassification.DESIGNATED_LANE

    # --- HUD warning text ---
    if violations:
        max_fine = max(v.fine_amount_aed for v in violations)
        violation_names = [v.violation_type for v in violations]
        hud_text = (
            "VIOLATION: "
            + ", ".join(violation_names)
            + f" | Max fine: AED {max_fine} | Rule-based audit (fallback mode)"
        )
    else:
        hud_text = "OK - Designated lane, no violations detected (rule-based fallback)."

    return VLMAuditResponse(
        zone_classification=zone,
        dismount_required=dismount_required,
        risk_level=risk_level,
        violations=violations,
        hud_warning_text=hud_text,
    )


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #
def _violation_exists(violation_type: str, violations: list[ViolationDetail]) -> bool:
    """Check whether a violation of the given type is already in the list."""
    return any(v.violation_type == violation_type for v in violations)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class Qwen2VLEngine:
    """Scene-understanding engine wrapping Qwen2-VL for compliance auditing.

    Heavy ML dependencies (``transformers``, ``peft``, ``bitsandbytes``) are
    imported lazily so the module can be used in test environments without
    GPU access.  For unit testing, inject ``model`` and ``processor`` directly.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        adapter_path: str | None = None,
        device: str = "auto",
        model: Any | None = None,
        processor: Any | None = None,
        fallback_enabled: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self._model_id: str = model_id
        self._adapter_path: str | None = adapter_path
        self._device: str = device
        self._model: Any | None = model
        self._processor: Any | None = processor
        self._fallback_enabled: bool = fallback_enabled
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def adapter_path(self) -> str | None:
        return self._adapter_path

    @property
    def is_model_loaded(self) -> bool:
        """``True`` when a model and processor are attached (loaded or injected)."""
        return self._model is not None and self._processor is not None

    @property
    def fallback_enabled(self) -> bool:
        return self._fallback_enabled

    def _ensure_model(self) -> None:
        """Lazily load the Qwen2-VL model and processor if not already present."""
        if self._model is not None and self._processor is not None:
            return

        # Lazy imports so the module loads without heavy ML deps.
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        logger.info("Loading VLM model: %s", self._model_id)

        use_4bit = self._device != "cpu"
        quant_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
            if use_4bit
            else None
        )

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_id,
            quantization_config=quant_config,
            torch_dtype="float16" if use_4bit else "float32",
            device_map=self._device,
            attn_implementation="sdpa",
        )

        if self._adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
            self._model.eval()
            logger.info("Loaded LoRA adapter from %s", self._adapter_path)

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        logger.info("VLM model and processor loaded successfully.")

    # ------------------------------------------------------------------ #
    # Image helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_pil_image(image: Any) -> Image.Image:
        """Convert a numpy array or PIL Image to a PIL Image."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            "Expected PIL.Image.Image or np.ndarray."
        )

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_messages(
        self,
        image: Image.Image,
        detection_context: DetectionResult,
        osm_policy: OSMRoadPolicy,
    ) -> list[dict[str, Any]]:
        """Build the message list for the Qwen2-VL chat template."""
        prompt = format_audit_prompt(detection_context, osm_policy)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # ------------------------------------------------------------------ #
    # JSON output parsing
    # ------------------------------------------------------------------ #
    def _parse_vlm_output(self, raw_output: str) -> tuple[VLMAuditResponse, bool]:
        """Parse VLM output, attempting JSON and regex recovery.

        Returns ``(response, fallback_triggered)`` where ``fallback_triggered``
        is ``True`` when recovery was needed (or safe default used).
        """
        # Fast path: valid JSON on the first try.
        try:
            data = json.loads(raw_output.strip())
            response = VLMAuditResponse(**data)
            return response, False
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            pass

        # Recovery: try regex / brace-matching extraction.
        recovered = _attempt_json_recovery(raw_output)
        if recovered is not None:
            logger.warning(
                "VLM output did not parse as direct JSON; recovered via regex."
            )
            return recovered, True

        # Recovery failed - return safe default.
        logger.error(
            "VLM JSON recovery failed. Returning safe default response. "
            "Raw output: %s",
            raw_output[:500],
        )
        return SAFE_DEFAULT_AUDIT, True

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def audit_frame(
        self,
        image: Any,
        detection_context: DetectionResult,
        osm_policy: OSMRoadPolicy,
    ) -> VLMAuditResponse:
        """Run a compliance audit on a single frame.

        Parameters
        ----------
        image:
            A :class:`PIL.Image.Image` or ``np.ndarray`` frame.
        detection_context:
            YOLO detection results with bounding boxes.
        osm_policy:
            The road policy from the OSM geospatial engine.

        Returns
        -------
        VLMAuditResponse
            Structured compliance verdict.  When the VLM cannot run or
            produces un-parseable output, a fallback response is returned.
        """
        with trace_span("vlm.audit_frame") as span:
            start = time.perf_counter()

            if span is not None:
                span.set_attribute("vlm.model_id", self._model_id + (
                    f" + {self._adapter_path}" if self._adapter_path else ""
                ))

            fallback_triggered = False

            try:
                self._ensure_model()
                pil_image = self._to_pil_image(image)
                messages = self._build_messages(
                    pil_image, detection_context, osm_policy
                )

                # --- Processor call ---
                inputs = self._processor(
                    text=[m["content"][1]["text"] for m in messages],
                    images=[pil_image],
                    padding=True,
                    return_tensors="pt",
                )

                # Record token usage if available
                prompt_tokens: int | None = None
                if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
                    # input_ids can be (batch, seq) or list of lists
                    input_ids = inputs.input_ids
                    if hasattr(input_ids, "shape"):
                        prompt_tokens = input_ids.shape[1]
                    elif isinstance(input_ids, list) and len(input_ids) > 0:
                        prompt_tokens = len(input_ids[0])

                # Move inputs to device
                device = self._device if self._device != "auto" else "cuda"
                try:
                    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
                except Exception:
                    pass  # Some tensors might not need/want device transfer

                # --- VLM generation ---
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": self._max_tokens,
                    "temperature": self._temperature,
                    "do_sample": True,
                }
                generated_ids = self._model.generate(**inputs, **generation_kwargs)

                # --- Decode ---
                raw_output = self._processor.decode(
                    generated_ids[0], skip_special_tokens=True
                )

                # Record completion token count if available
                completion_tokens: int | None = None
                if hasattr(generated_ids, "shape") and generated_ids.dim() >= 2:
                    completion_tokens = generated_ids.shape[1]

                # --- Parse output ---
                response, parse_fallback = self._parse_vlm_output(raw_output)
                fallback_triggered = parse_fallback

                if span is not None and prompt_tokens is not None:
                    span.set_attribute("vlm.prompt_tokens", prompt_tokens)
                if span is not None and completion_tokens is not None:
                    span.set_attribute("vlm.completion_tokens", completion_tokens)

            except Exception as exc:
                logger.warning(
                    "VLM execution failed (%s). Falling back to rule-based audit.",
                    exc,
                )
                if self._fallback_enabled:
                    response = rule_based_fallback_audit(
                        detection_context, osm_policy
                    )
                    fallback_triggered = True
                else:
                    # If fallback is disabled, re-raise the error.
                    raise

                if span is not None:
                    from src.telemetry.tracing import _record_exception

                    _record_exception(span, exc)

            # --- OTel attribute recording ---
            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

            if span is not None:
                span.set_attribute("vlm.latency_ms", latency_ms)
                span.set_attribute("vlm.risk_level", response.risk_level.value)
                span.set_attribute("vlm.fallback_triggered", fallback_triggered)

            logger.info(
                "audit_frame complete: risk=%s, fallback=%s, latency=%.1fms",
                response.risk_level.value,
                fallback_triggered,
                latency_ms,
            )

            return response
