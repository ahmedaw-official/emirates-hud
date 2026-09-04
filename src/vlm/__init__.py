"""Multimodal Regulatory Engine package.

Provides the VLM inference engine (``Qwen2VLEngine``), regulatory system
prompts, and Unsloth QLoRA fine-tuning utilities for UAE micro-mobility
compliance auditing.

Public API::

    from src.vlm import Qwen2VLEngine, rule_based_fallback_audit

    engine = Qwen2VLEngine()
    audit = engine.audit_frame(image, detections, osm_policy)
"""

from __future__ import annotations

from src.vlm.qwen_engine import (
    DEFAULT_FALLBACK_WARNING,
    SAFE_DEFAULT_AUDIT,
    Qwen2VLEngine,
    rule_based_fallback_audit,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_FALLBACK_WARNING",
    "SAFE_DEFAULT_AUDIT",
    "Qwen2VLEngine",
    "__version__",
    "rule_based_fallback_audit",
]
