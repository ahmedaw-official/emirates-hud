"""Asynchronous Trigger Router for VLM invocation decisions.

The :class:`TriggerRouter` decides whether a frame warrants a full Qwen2-VL
VLM audit (expensive GPU compute) or can be fast-passed through the rule-based
fallback path.  Three heuristics are evaluated in priority order:

1. **Surface transition** - the detected primary surface changed between
   consecutive frames (e.g. ``red_track`` -> ``grey_sidewalk``), which may
   indicate an illegal manoeuvre crossing a legality boundary.
2. **Critical sign detected** - high-confidence (> 0.50) detection of
   ``sign_no_scooter`` or ``zebra_crosswalk``.
3. **Heartbeat timer** - at least 30 frames (~1 second at 30 FPS) since the
   last VLM evaluation, ensuring periodic freshness even in static scenes.

The router is stateful: it remembers the previous surface and a frame
counter, both updated on each call.
"""

from __future__ import annotations

from src.schemas import DetectionResult
from src.telemetry.logging import get_logger

__all__ = ["TriggerRouter"]

logger = get_logger(__name__)


class TriggerRouter:
    """Determines whether a frame should trigger the expensive VLM pass."""

    #: Frames to skip between VLM evaluations when no urgent signal fires.
    HEARTBEAT_THRESHOLD: int = 30

    #: Minimum confidence for a detection to count as a "critical sign".
    CRITICAL_SIGN_CONFIDENCE: float = 0.50

    #: YOLO class names that always warrant VLM scrutiny when detected confidently.
    CRITICAL_SIGNS: frozenset[str] = frozenset({"sign_no_scooter", "zebra_crosswalk"})

    #: YOLO class names that represent the road surface under the scooter.
    SURFACE_CLASSES: frozenset[str] = frozenset({"red_track", "grey_sidewalk"})

    def __init__(
        self,
        heartbeat_threshold: int = HEARTBEAT_THRESHOLD,
        critical_sign_confidence: float = CRITICAL_SIGN_CONFIDENCE,
    ) -> None:
        self._heartbeat_threshold = heartbeat_threshold
        self._critical_sign_confidence = critical_sign_confidence
        self._previous_surface: str | None = None
        self._frames_since_last_vlm: int = 0

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #
    @property
    def previous_surface(self) -> str | None:
        return self._previous_surface

    @property
    def frames_since_last_vlm(self) -> int:
        return self._frames_since_last_vlm

    @property
    def heartbeat_threshold(self) -> int:
        return self._heartbeat_threshold

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _extract_surface(self, detection_result: DetectionResult) -> str | None:
        """Return the highest-confidence surface class found, or ``None``."""
        best_surface: str | None = None
        best_confidence: float = -1.0
        for box in detection_result.boxes:
            if box.class_name in self.SURFACE_CLASSES:
                if box.confidence > best_confidence:
                    best_surface = box.class_name
                    best_confidence = box.confidence
        return best_surface

    def _has_critical_sign(self, detection_result: DetectionResult) -> str | None:
        """Return the class name of the first high-confidence critical sign."""
        for box in detection_result.boxes:
            if (
                box.class_name in self.CRITICAL_SIGNS
                and box.confidence > self._critical_sign_confidence
            ):
                return box.class_name
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def should_trigger_vlm(
        self,
        detection_result: DetectionResult,
        frames_since_last_vlm: int | None = None,
        previous_surface: str | None = None,
    ) -> tuple[bool, str]:
        """Decide whether the frame warrants a VLM audit.

        Parameters
        ----------
        detection_result:
            Current frame's YOLO detections.
        frames_since_last_vlm:
            Optional override for the internal frame counter.  When provided
            it updates the router's internal state.
        previous_surface:
            Optional override for the previous frame's surface class.

        Returns
        -------
        tuple[bool, str]
            ``(should_trigger, trigger_reason)`` where ``trigger_reason``
            is a short human-readable string such as ``"surface_transition"``,
            ``"critical_sign:zebra_crosswalk"``, ``"heartbeat"``, or
            ``"no_trigger"``.
        """
        if frames_since_last_vlm is not None:
            self._frames_since_last_vlm = frames_since_last_vlm

        current_surface = self._extract_surface(detection_result)

        if previous_surface is not None:
            self._previous_surface = previous_surface

        # --- a) Surface transition ---
        if (
            current_surface is not None
            and self._previous_surface is not None
            and current_surface != self._previous_surface
        ):
            self._previous_surface = current_surface
            self._frames_since_last_vlm = 0
            reason = f"surface_transition:{self._previous_surface}->{current_surface}"
            logger.debug("VLM trigger: %s", reason)
            return True, reason

        # --- b) Critical sign detected ---
        critical_sign = self._has_critical_sign(detection_result)
        if critical_sign is not None:
            self._previous_surface = current_surface
            self._frames_since_last_vlm = 0
            reason = f"critical_sign:{critical_sign}"
            logger.debug("VLM trigger: %s", reason)
            return True, reason

        # --- c) Heartbeat timer ---
        if self._frames_since_last_vlm >= self._heartbeat_threshold:
            self._previous_surface = current_surface
            self._frames_since_last_vlm = 0
            logger.debug("VLM trigger: heartbeat (threshold=%d)", self._heartbeat_threshold)
            return True, "heartbeat"

        # --- No trigger ---
        self._previous_surface = current_surface
        self._frames_since_last_vlm += 1
        return False, "no_trigger"

    def reset(self) -> None:
        """Reset internal state (previous surface and frame counter)."""
        self._previous_surface = None
        self._frames_since_last_vlm = 0
