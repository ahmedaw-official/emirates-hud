"""Regulatory system prompts and audit-prompt formatters.

Every prompt that drives the Qwen2-VL scene-understanding call is defined here
so the legal context, JSON schema contract, and detection/OSM wiring are kept
in one auditable location.

Legal references encoded below:
    - Dubai RTA Executive Council Resolution No. 13 of 2022
      Article 4 (sidewalk prohibition), Article 5 (prohibited roads),
      Article 6 (crosswalk dismount), Schedule of Fines.
    - Abu Dhabi ITC bylaws (general micro-mobility provisions).
"""

from __future__ import annotations

from src.schemas import DetectionResult, OSMRoadPolicy

__all__ = [
    "JSON_SCHEMA_DESCRIPTION",
    "RTA_SYSTEM_PROMPT",
    "format_audit_prompt",
    "format_detection_tags",
    "format_osm_context",
]


# --------------------------------------------------------------------------- #
# Static prompt components
# --------------------------------------------------------------------------- #
RTA_SYSTEM_PROMPT: str = """You are a multimodal regulatory compliance auditor
for UAE micro-mobility (electric scooters and similar devices). You evaluate
camera-frame scenes against the following legal framework and emit STRICT JSON:

## Legal Framework

### Dubai RTA - Executive Council Resolution No. 13 of 2022

**Article 4 - Designated Use**
Micro-mobility devices are prohibited on non-designated sidewalks and
pedestrian-only zones. Violation: AED 200-300 fine.

**Article 5 - Prohibited Roads**
Riding on roads with a posted speed limit exceeding 60 km/h is prohibited.
Violation: AED 300 fine.

**Article 6 - Crosswalk Dismount**
The rider must dismount at marked zebra crosswalks when crossing a
street. Failure to do so: AED 200 fine.

### Abu Dhabi ITC Bylaws
Similar provisions apply regarding designated lanes, pedestrian areas,
and crosswalk dismounting requirements.

## Output Format

Respond with ONLY a valid JSON object matching this schema. No prose,
no markdown fences, no explanatory text:

{
  "zone_classification": "DESIGNATED_LANE",
  "dismount_required": false,
  "risk_level": "LOW",
  "violations": [
    {
      "violation_type": "SIDEWALK_DISOBEDENCE",
      "legal_reference": "Dubai RTA Resolution No. 13 (2022) Article 4",
      "fine_amount_aed": 250
    }
  ],
  "hud_warning_text": "warning message for the heads-up display"
}

zone_classification MUST be one of: DESIGNATED_LANE, PEDESTRIAN_SIDEWALK, CROSSWALK
risk_level MUST be one of: LOW, MEDIUM, HIGH, CRITICAL
violations is a list of objects with: violation_type (string), legal_reference (string), fine_amount_aed (integer)
hud_warning_text MUST be a non-empty string
"""

#: Human-readable description of the JSON schema the VLM must produce.
JSON_SCHEMA_DESCRIPTION: str = """zone_classification: DESIGNATED_LANE | PEDESTRIAN_SIDEWALK | CROSSWALK
dismount_required: bool
risk_level: LOW | MEDIUM | HIGH | CRITICAL
violations: list of {violation_type: str, legal_reference: str, fine_amount_aed: int}
hud_warning_text: str"""


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
def format_detection_tags(detection_context: DetectionResult) -> str:
    """Serialise YOLO detections into human-readable VLM prompt tags.

    Each bounding box becomes a line like::

        [DETECTION class=scooter confidence=0.93 bbox=(0.12, 0.08, 0.45, 0.91)]
    """
    if not detection_context.boxes:
        return "[DETECTIONS] None detected."

    lines: list[str] = ["[DETECTIONS]"]
    for box in detection_context.boxes:
        lines.append(
            f"  [DETECTION class={box.class_name} confidence={box.confidence:.2f} "
            f"bbox=({box.xmin:.3f}, {box.ymin:.3f}, {box.xmax:.3f}, {box.ymax:.3f})]"
        )
    return "\n".join(lines)


def format_osm_context(osm_policy: OSMRoadPolicy) -> str:
    """Serialise the OSM road policy into VLM prompt context."""
    maxspeed_str = f"{osm_policy.maxspeed_kmh} km/h" if osm_policy.maxspeed_kmh is not None else "unknown"
    return (
        "[OSM_ROAD_POLICY]\n"
        f"  highway_type: {osm_policy.highway_type}\n"
        f"  maxspeed: {maxspeed_str}\n"
        f"  is_prohibited_road: {osm_policy.is_prohibited_road}\n"
        f"  is_fallback: {osm_policy.is_fallback}\n"
        f"  way_id: {osm_policy.way_id}"
    )


def format_audit_prompt(
    detection_context: DetectionResult,
    osm_policy: OSMRoadPolicy,
) -> str:
    """Build the full user message prompt for the VLM.

    Combines the YOLO detection tags and OSM road policy into a structured
    prompt that instructs the model to output the compliance JSON schema.
    """
    detection_block = format_detection_tags(detection_context)
    osm_block = format_osm_context(osm_policy)
    return (
        f"{RTA_SYSTEM_PROMPT}\n\n"
        f"[USER_MESSAGE]\n"
        f"Evaluate this scene for micro-mobility compliance.\n\n"
        f"{detection_block}\n\n"
        f"{osm_block}\n\n"
        f"[INSTRUCTIONS]\n"
        f"- Analyse the detected objects relative to the road context.\n"
        f"- Apply RTA Resolution No. 13 (2022) rules.\n"
        f"- Output ONLY the JSON object described above.\n"
        f"- If the rider is on a prohibited road (maxspeed > 60 km/h) or a "
        f"pedestrian-only way (footway/pedestrian/steps), flag it.\n"
        f"- If a crosswalk is present and the rider has not dismounted, flag it.\n"
        f"- Set risk_level to CRITICAL for prohibited-road usage, HIGH for "
        f"sidewalk riding, MEDIUM for minor infractions, LOW for full compliance.\n"
        f"{JSON_SCHEMA_DESCRIPTION}\n"
    )
