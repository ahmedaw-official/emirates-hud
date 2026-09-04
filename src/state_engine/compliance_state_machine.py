"""Compliance state machine - fuses perception, geospatial, and VLM signals.

The :class:`ComplianceStateMachine` is the final arbiter of per-frame
compliance.  It consumes:

* :class:`~src.schemas.DetectionResult` - YOLO bounding boxes (surface type,
  critical signs, pedestrian/scooter presence).
* :class:`~src.schemas.OSMRoadPolicy` - OSM Overpass road policy.
* :class:`~src.schemas.VLMAuditResponse` - optional Qwen2-VL audit.

and produces a :class:`~src.schemas.FrameState` carrying the aggregate
``ComplianceState`` (``SAFE`` / ``WARNING`` / ``VIOLATION``), the ceiling
fine in AED, and a HUD-ready alert message.

A hysteresis buffer requires **3 consecutive** non-VIOLATION frames before
the state machine will clear a ``VIOLATION`` status, preventing jarring HUD
flicker when the scene briefly dips into a compliant zone.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.schemas import (
    ComplianceState,
    DetectionResult,
    OSMRoadPolicy,
    RiskLevel,
    VLMAuditResponse,
)
from src.schemas.frame import FrameState
from src.telemetry import trace_span
from src.telemetry.logging import get_logger

from .trigger_router import TriggerRouter

__all__ = ["ComplianceStateMachine"]

logger = get_logger(__name__)

#: Number of consecutive non-VIOLATION frames required before clearing a
#: VIOLATION state (hysteresis buffer).
HYSTERESIS_THRESHOLD: int = 3

#: Fine amounts by compliance state (AED).
VIOLATION_FINE_AED: int = 300
WARNING_FINE_AED: int = 200
SAFE_FINE_AED: int = 0


class ComplianceStateMachine:
    """Stateful frame-level compliance evaluator with hysteresis smoothing.

    Parameters
    ----------
    router:
        Optional :class:`TriggerRouter` instance.  If not provided a default
        one is created.
    hysteresis_threshold:
        Number of consecutive non-VIOLATION frames required before the
        state machine clears a VIOLATION status (default 3).
    """

    def __init__(
        self,
        router: TriggerRouter | None = None,
        hysteresis_threshold: int = HYSTERESIS_THRESHOLD,
    ) -> None:
        self._router: TriggerRouter = router if router is not None else TriggerRouter()
        self._hysteresis_threshold: int = hysteresis_threshold
        self._current_state: ComplianceState = ComplianceState.SAFE
        self._hysteresis_counter: int = 0
        self._previous_surface: str | None = None
        self._frames_since_last_vlm: int = 0

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #
    @property
    def current_state(self) -> ComplianceState:
        return self._current_state

    @property
    def hysteresis_counter(self) -> int:
        return self._hysteresis_counter

    @property
    def is_hysteresis_active(self) -> bool:
        """True when the state machine is holding a VIOLATION due to hysteresis."""
        return self._current_state == ComplianceState.VIOLATION and self._hysteresis_counter > 0

    @property
    def router(self) -> TriggerRouter:
        return self._router

    # ------------------------------------------------------------------ #
    # Surface detection helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _has_surface(detection_result: DetectionResult, surface_class: str) -> bool:
        """Return True if *surface_class* appears among the YOLO detections."""
        return any(b.class_name == surface_class for b in detection_result.boxes)

    @staticmethod
    def _extract_surface(detection_result: DetectionResult) -> str | None:
        """Extract the primary surface class from detections."""
        surface_classes = {"red_track", "grey_sidewalk"}
        for box in detection_result.boxes:
            if box.class_name in surface_classes:
                return box.class_name
        return None

    # ------------------------------------------------------------------ #
    # Core logic matrix
    # ------------------------------------------------------------------ #
    def _evaluate_logic_matrix(
        self,
        detection_result: DetectionResult,
        osm_policy: OSMRoadPolicy,
        vlm_response: VLMAuditResponse | None,
    ) -> tuple[ComplianceState, int, str]:
        """Apply the deterministic YOLO + OSM + VLM logic matrix.

        Returns ``(state, fine_aed, hud_message)``.
        """
        # --- Rule 1: grey_sidewalk OR prohibited road -> VIOLATION ---
        if self._has_surface(detection_result, "grey_sidewalk"):
            logger.debug("State matrix: grey_sidewalk detected -> VIOLATION")
            fine = VIOLATION_FINE_AED
            message = "CRITICAL: PROHIBITED ZONE - AED 300 FINE"
            state = ComplianceState.VIOLATION
        elif not osm_policy.is_fallback and osm_policy.is_prohibited_road:
            logger.debug("State matrix: OSM prohibited road -> VIOLATION")
            fine = VIOLATION_FINE_AED
            message = "CRITICAL: PROHIBITED ZONE - AED 300 FINE"
            state = ComplianceState.VIOLATION

        # --- Rule 2: zebra_crosswalk -> WARNING ---
        elif any(b.class_name == "zebra_crosswalk" for b in detection_result.boxes):
            logger.debug("State matrix: zebra_crosswalk -> WARNING")
            fine = WARNING_FINE_AED
            message = "DISMOUNT REQUIRED AT CROSSWALK"
            state = ComplianceState.WARNING

        # --- Rule 3: red_track -> SAFE ---
        elif self._has_surface(detection_result, "red_track"):
            logger.debug("State matrix: red_track -> SAFE")
            fine = SAFE_FINE_AED
            message = "DESIGNATED LANE - SAFE"
            state = ComplianceState.SAFE

        # --- Default: no recognisable surface ---
        else:
            logger.debug("State matrix: no surface detected -> SAFE (default)")
            fine = SAFE_FINE_AED
            message = "DESIGNATED LANE - SAFE"
            state = ComplianceState.SAFE

        # --- VLM escalation ---
        if vlm_response is not None:
            vlm_risk = vlm_response.risk_level
            vlm_fine = vlm_response.max_fine_aed

            # Escalate state based on VLM risk level
            if vlm_risk == RiskLevel.CRITICAL:
                state = ComplianceState.VIOLATION
            elif vlm_risk == RiskLevel.HIGH and state == ComplianceState.SAFE:
                state = ComplianceState.WARNING

            # Use the higher fine between matrix and VLM
            fine = max(fine, vlm_fine)

            # Re-derive message from final state
            fine, message = self._state_to_fine_and_message(state, fine)

            logger.debug(
                "VLM escalation: risk=%s, fine=%d, final_state=%s",
                vlm_risk.value if hasattr(vlm_risk, "value") else vlm_risk,
                fine,
                state.value,
            )

        return state, fine, message

    @staticmethod
    def _state_to_fine_and_message(state: ComplianceState, fine_override: int | None = None) -> tuple[int, str]:
        """Map a :class:`ComplianceState` to its canonical fine and message."""
        if state == ComplianceState.VIOLATION:
            return (fine_override or VIOLATION_FINE_AED), "CRITICAL: PROHIBITED ZONE - AED 300 FINE"
        if state == ComplianceState.WARNING:
            return (fine_override or WARNING_FINE_AED), "DISMOUNT REQUIRED AT CROSSWALK"
        return (fine_override or SAFE_FINE_AED), "DESIGNATED LANE - SAFE"

    # ------------------------------------------------------------------ #
    # Hysteresis
    # ------------------------------------------------------------------ #
    def _apply_hysteresis(self, new_state: ComplianceState) -> tuple[ComplianceState, bool]:
        """Apply hysteresis buffer before clearing a VIOLATION state.

        Returns ``(final_state, hysteresis_active)``.
        """
        if (
            self._current_state == ComplianceState.VIOLATION
            and new_state != ComplianceState.VIOLATION
        ):
            self._hysteresis_counter += 1

            if self._hysteresis_counter < self._hysteresis_threshold:
                # Hold VIOLATION until threshold reached
                logger.debug(
                    "Hysteresis: holding VIOLATION (count=%d/%d)",
                    self._hysteresis_counter,
                    self._hysteresis_threshold,
                )
                return ComplianceState.VIOLATION, True

            # Threshold reached - allow transition
            logger.debug(
                "Hysteresis: threshold reached, transitioning to %s",
                new_state.value,
            )
            self._current_state = new_state
            self._hysteresis_counter = 0
            return new_state, False

        # Transition into VIOLATION, or any non-VIOLATION-to-VIOLATION change
        if new_state == self._current_state:
            self._hysteresis_counter = 0
            return new_state, False

        if new_state != ComplianceState.VIOLATION:
            # Non-VIOLATION state changing to another non-VIOLATION state
            self._hysteresis_counter = 0

        self._current_state = new_state
        return new_state, False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate_frame_state(
        self,
        detection_result: DetectionResult,
        osm_policy: OSMRoadPolicy,
        vlm_response: VLMAuditResponse | None = None,
    ) -> FrameState:
        """Evaluate the compliance state for a single frame.

        Parameters
        ----------
        detection_result:
            YOLO detections for the current frame.
        osm_policy:
            OSM road policy for the current location.
        vlm_response:
            Optional VLM audit verdict.  When ``None`` the state is
            determined exclusively from YOLO + OSM.

        Returns
        -------
        FrameState
            The complete per-frame compliance snapshot with OTel attributes
            recorded on span ``state_engine.evaluate_frame_state``.
        """
        with trace_span("state_engine.evaluate_frame_state") as span:
            # --- Trigger router decision ---
            should_trigger, trigger_reason = self._router.should_trigger_vlm(
                detection_result,
                frames_since_last_vlm=self._frames_since_last_vlm,
                previous_surface=self._previous_surface,
            )

            # Update router state
            self._frames_since_last_vlm = 0 if should_trigger else self._frames_since_last_vlm + 1
            current_surface = self._extract_surface(detection_result)
            self._previous_surface = current_surface

            # --- Logic matrix evaluation ---
            new_state, new_fine, new_message = self._evaluate_logic_matrix(
                detection_result, osm_policy, vlm_response
            )

            # --- Hysteresis ---
            final_state, hysteresis_active = self._apply_hysteresis(new_state)

            # If hysteresis held the state, recompute fine/message for the held state
            if final_state != new_state:
                new_fine, new_message = self._state_to_fine_and_message(final_state)

            # --- OTel recording ---
            if span is not None:
                span.set_attribute("state.active_status", final_state.value)
                span.set_attribute("state.fine_risk_aed", new_fine)
                span.set_attribute("router.vlm_triggered", should_trigger)
                span.set_attribute("router.trigger_reason", trigger_reason)
                span.set_attribute("state.hysteresis_active", hysteresis_active)

            logger.info(
                "evaluate_frame_state: state=%s, fine=%d AED, vlm_triggered=%s, "
                "hysteresis=%s, reason=%s",
                final_state.value,
                new_fine,
                should_trigger,
                hysteresis_active,
                trigger_reason,
            )

            return FrameState(
                frame_id=detection_result.frame_id,
                timestamp=datetime.now(timezone.utc),
                visual_detections=list(detection_result.boxes),
                geospatial_policy=osm_policy,
                compliance_audit=vlm_response,
                active_compliance_state=final_state,
                hud_alert_message=new_message,
                fine_risk_aed=new_fine,
            )

    # ------------------------------------------------------------------ #
    # Manual state manipulation (useful for testing / recovery)
    # ------------------------------------------------------------------ #
    def force_state(self, state: ComplianceState) -> None:
        """Force the state machine into a specific state (resets hysteresis)."""
        self._current_state = state
        self._hysteresis_counter = 0

    def reset(self) -> None:
        """Reset all internal state to defaults."""
        self._current_state = ComplianceState.SAFE
        self._hysteresis_counter = 0
        self._previous_surface = None
        self._frames_since_last_vlm = 0
        self._router.reset()
