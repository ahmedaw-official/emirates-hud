"""HUD compositor: renders compliance overlays onto OpenCV BGR frames.

The :class:`HUDCompositor` is the visual front-end of the compliance engine.
It consumes a :class:`~src.schemas.frame.FrameState` (or its raw constituents)
together with a live video frame and produces an annotated frame suitable
for display or recording.

Design goals
------------
* **No frame interruption** - any drawing error is caught, logged and the
  raw frame is returned untouched (the "overlay exception shield").
* **High contrast** - semi-transparent backgrounds behind every text label
  guarantee legibility on bright desert footage.
* **Deterministic colours** - green / amber / red map 1:1 to compliance
  state, matching UAE road-marking conventions.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.config.schema import HUDConfig
from src.schemas import (
    BoundingBox,
    ComplianceState,
    FrameState,
)
from src.telemetry import get_logger, trace_span

__all__ = [
    "BoundingBoxStyle",
    "HUDCompositor",
]

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Helper data structures
# --------------------------------------------------------------------------- #
class BoundingBoxStyle(BaseModel):
    """Colour and label template for a single object class."""

    model_config = ConfigDict(extra="forbid")

    color: tuple[int, int, int] = Field(
        description="OpenCV BGR colour tuple for the bounding box.",
    )
    label: str = Field(description="Human-readable label shown above the box.")


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def _scale_box(
    box: BoundingBox,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Convert a normalised BoundingBox to absolute pixel coordinates."""
    x1 = int(box.xmin * frame_width)
    y1 = int(box.ymin * frame_height)
    x2 = int(box.xmax * frame_width)
    y2 = int(box.ymax * frame_height)
    return x1, y1, x2, y2


def _render_text_with_pill(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    scale: float = 0.6,
    thickness: int = 1,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.7,
    padding: int = 6,
    font=cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """Render ``text`` on ``frame`` with a semi-transparent background pill.

    The pill is sized to the rendered text bounds so there is no wasted
    background when the label is short.  All operations are best-effort;
    a drawing failure raises so the caller's exception shield can handle it.
    """
    if not text:
        return

    x, y = position
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    baseline_val = max(baseline, 1)

    # Pill rectangle (slightly larger than text for breathing room).
    pill_x1 = x
    pill_y1 = y - text_h - padding
    pill_x2 = x + text_w + padding
    pill_y2 = y + baseline_val + padding

    # Clamp to frame bounds.
    h, w = frame.shape[:2]
    pill_x1 = max(0, pill_x1)
    pill_y1 = max(0, pill_y1)
    pill_x2 = min(w, pill_x2)
    pill_y2 = min(h, pill_y2)

    # Semi-transparent overlay.
    overlay = frame.copy()
    cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Text on top.
    cv2.putText(frame, text, (x + padding // 2, y + text_h // 2),
                font, scale, color, thickness, cv2.LINE_AA)


def _render_filled_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float = 0.5,
) -> None:
    """Render a semi-transparent filled rectangle (for overlay panels)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# --------------------------------------------------------------------------- #
# HUDCompositor
# --------------------------------------------------------------------------- #
class HUDCompositor:
    """Renders compliance overlays onto OpenCV BGR frames.

    Parameters
    ----------
    config:
        :class:`~src.config.schema.HUDConfig` instance.  Defaults to a
        standard 720p configuration when ``None``.
    """

    def __init__(self, config: HUDConfig | None = None) -> None:
        self.config = config or HUDConfig()
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------ public
    def render_frame(
        self,
        frame: np.ndarray,
        frame_state: FrameState,
        *,
        speed_kmh: float = 0.0,
        processing_fps: float = 0.0,
        force_alert: bool = False,
    ) -> np.ndarray:
        """Render all HUD overlays onto ``frame`` and return the annotated copy.

        Parameters
        ----------
        frame:
            Raw OpenCV BGR frame (numpy ``uint8`` array).
        frame_state:
            The :class:`~src.schemas.frame.FrameState` from the state engine.
        speed_kmh:
            Current scooter speed in km/h (from GPS or wheel odometer).
        processing_fps:
            Real-time rendering FPS to overlay in the bottom-right corner.
        force_alert:
            When ``True``, the critical alert banner is always rendered
            (bypassing the time-based flashing toggle).  Useful in testing
            or when a persistent alert is desired.

        Returns
        -------
        np.ndarray
            Annotated BGR frame.  If any drawing operation fails, the
            original *unannotated* frame is returned so the video stream
            never breaks.
        """
        # Work on a copy so the caller's frame is never mutated in-place.
        canvas = frame.copy()

        try:
            with trace_span("hud.render_frame") as span:
                # --- top telemetry bar ---
                self._render_top_bar(canvas, frame_state, speed_kmh)

                # --- bounding boxes ---
                self._render_bounding_boxes(canvas, frame_state)

                # --- critical alert overlay ---
                self._render_critical_alert(canvas, frame_state, force_alert)

                # --- bottom-right FPS ---
                self._render_fps(canvas, processing_fps)

                # --- span attributes ---
                span.set_attribute("hud.frame_idx", frame_state.frame_id)
                span.set_attribute(
                    "hud.active_fines_aed", frame_state.fine_risk_aed
                )
                span.set_attribute("hud.codec_used", "none")  # set by VideoProcessor
                span.set_attribute("hud.overlay_errors_count", 0)

        except Exception as exc:
            self._logger.error(
                "HUD render error on frame %d: %s - returning raw frame",
                frame_state.frame_id,
                exc,
                exc_info=True,
            )
            # Re-raise is NOT done - the raw frame is returned (overlay exception shield).
            # OTel: record the error on the current (or no) span.
            try:
                span.set_attribute("hud.overlay_errors_count", 1)
                span.record_exception(exc)
            except Exception:
                pass
            return frame.copy()

        return canvas

    # ------------------------------------------------------------------ private
    def _render_top_bar(
        self,
        frame: np.ndarray,
        frame_state: FrameState,
        speed_kmh: float,
    ) -> None:
        """Render the semi-transparent top telemetry overlay."""
        cfg = self.config
        h, w = frame.shape[:2]
        bar_h = min(cfg.bar_height, h // 6)

        # --- semi-transparent background ---
        _render_filled_rect(
            frame, 0, 0, w, bar_h,
            cfg.color_overlay_bg, alpha=cfg.overlay_alpha,
        )

        # --- speed & limit ---
        limit = frame_state.geospatial_policy.maxspeed_kmh
        limit_str = f"{limit}" if limit is not None else "—"
        speed_text = f"{speed_kmh:.0f} / {limit_str} KM/H"
        speed_color = (
            cfg.color_violation
            if frame_state.active_compliance_state == ComplianceState.VIOLATION
            else cfg.color_text
        )
        _render_text_with_pill(
            frame, speed_text, (10, bar_h // 2 + 10),
            scale=cfg.font_scale, thickness=cfg.font_thickness,
            color=speed_color, bg_color=cfg.color_overlay_bg,
            alpha=cfg.overlay_alpha,
        )

        # --- zone label ---
        zone_text = self._zone_label(frame_state)
        _render_text_with_pill(
            frame, zone_text, (180, bar_h // 2 + 10),
            scale=cfg.font_scale, thickness=cfg.font_thickness,
            color=cfg.color_text, bg_color=cfg.color_overlay_bg,
            alpha=cfg.overlay_alpha,
        )

        # --- helmet status ---
        helmet_text, helmet_color = self._helmet_status(frame_state)
        _render_text_with_pill(
            frame, helmet_text, (w // 2 - 60, bar_h // 2 + 10),
            scale=cfg.font_scale, thickness=cfg.font_thickness,
            color=helmet_color, bg_color=cfg.color_overlay_bg,
            alpha=cfg.overlay_alpha,
        )

        # --- fine risk ---
        fine_text = self._fine_risk_text(frame_state)
        fine_color = (
            cfg.color_violation
            if frame_state.fine_risk_aed > 0
            else cfg.color_safe
        )
        _render_text_with_pill(
            frame, fine_text, (w - 280, bar_h // 2 + 10),
            scale=cfg.font_scale, thickness=cfg.font_thickness,
            color=fine_color, bg_color=cfg.color_overlay_bg,
            alpha=cfg.overlay_alpha,
        )

    def _render_bounding_boxes(
        self,
        frame: np.ndarray,
        frame_state: FrameState,
    ) -> None:
        """Draw each detection box with a state-appropriate colour and label."""
        cfg = self.config
        h, w = frame.shape[:2]

        # Determine box colour from the compliance state.
        state_color = self._state_to_color(frame_state.active_compliance_state)

        for box in frame_state.visual_detections:
            x1, y1, x2, y2 = _scale_box(box, w, h)
            color = self._class_specific_color(box.class_name, state_color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.font_thickness + 1, cv2.LINE_AA)

            # Label pill above the box.
            label = f"{box.class_name} {box.confidence:.2f}"
            label_color = cfg.color_text
            bg_color = tuple(int(c * 0.5) for c in color)  # darker shade of box colour
            _render_text_with_pill(
                frame, label, (x1, y1 - 4),
                scale=cfg.font_scale * 0.8,
                thickness=1,
                color=label_color,
                bg_color=bg_color,
                alpha=cfg.overlay_alpha,
                padding=4,
            )

    def _render_critical_alert(
        self,
        frame: np.ndarray,
        frame_state: FrameState,
        force_alert: bool = False,
    ) -> None:
        """Render a centered critical alert banner when warranted."""
        should_alert = (
            frame_state.active_compliance_state == ComplianceState.VIOLATION
            or frame_state.fine_risk_aed > 0
        )
        if not should_alert:
            return

        cfg = self.config
        h, w = frame.shape[:2]

        # Flashing logic: toggle every `flashing_alert_interval_ms`.
        # Skip the toggle when ``force_alert`` is True (testing / persistent mode).
        if not force_alert:
            now_ms = int(time.time() * 1000)
            is_visible = (
                (now_ms % (cfg.flashing_alert_interval_ms * 2))
                < cfg.flashing_alert_interval_ms
            )
            if not is_visible:
                return

        alert_text = frame_state.hud_alert_message
        if not alert_text:
            alert_text = "CRITICAL ALERT"

        # Center the text.
        (text_w, text_h), _baseline = cv2.getTextSize(
            alert_text, cv2.FONT_HERSHEY_SIMPLEX, cfg.title_font_scale, cfg.title_font_thickness
        )
        cx = (w - text_w) // 2
        cy = (h - text_h) // 2

        # Red semi-transparent banner behind the text.
        banner_x1 = max(0, cx - 20)
        banner_y1 = max(0, cy - 20)
        banner_x2 = min(w, cx + text_w + 20)
        banner_y2 = min(h, cy + text_h + 20)
        _render_filled_rect(
            frame, banner_x1, banner_y1, banner_x2, banner_y2,
            cfg.color_violation, alpha=0.6,
        )

        # White bold text with black outline for max contrast.
        cv2.putText(
            frame, alert_text,
            (cx, cy + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.title_font_scale,
            cfg.color_overlay_bg,  # black outline
            cfg.title_font_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, alert_text,
            (cx, cy + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.title_font_scale,
            cfg.color_text,  # white text
            cfg.title_font_thickness,
            cv2.LINE_AA,
        )

    def _render_fps(
        self,
        frame: np.ndarray,
        fps: float,
    ) -> None:
        """Overlay the real-time FPS in the bottom-right corner."""
        if fps <= 0:
            return
        cfg = self.config
        h, w = frame.shape[:2]
        fps_text = f"FPS: {fps:.1f}"
        _render_text_with_pill(
            frame, fps_text, (w - 140, h - 10),
            scale=cfg.font_scale * 0.7,
            thickness=1,
            color=cfg.color_text,
            bg_color=cfg.color_overlay_bg,
            alpha=cfg.overlay_alpha,
            padding=4,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _state_to_color(
        state: ComplianceState,
    ) -> tuple[int, int, int]:
        """Map a :class:`ComplianceState` to a BGR colour."""
        if state == ComplianceState.VIOLATION:
            return (0, 0, 255)   # red
        if state == ComplianceState.WARNING:
            return (0, 215, 255)  # amber
        return (0, 255, 0)        # green (SAFE)

    def _class_specific_color(
        self,
        class_name: str,
        fallback: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Override colour for specific classes (e.g. no-helmet → red)."""
        cfg = self.config
        name = class_name.lower()
        if "no_helmet" in name or "no-helmet" in name:
            return cfg.color_violation
        if "helmet" in name:
            return cfg.color_safe
        return fallback

    def _zone_label(self, frame_state: FrameState) -> str:
        """Build the zone description text from OSM + VLM data."""
        policy = frame_state.geospatial_policy
        vlm = frame_state.compliance_audit

        if vlm is not None:
            zone = vlm.zone_classification.value.replace("_", " ").title()
            return f"VLM Zone: {zone}"

        # Fall back to OSM highway type.
        ht = policy.highway_type
        if policy.is_fallback:
            ht += " (fallback)"
        return f"OSM: {ht}"

    def _helmet_status(self, frame_state: FrameState) -> tuple[str, tuple[int, int, int]]:
        """Determine helmet status text and colour from detections."""
        cfg = self.config
        boxes = frame_state.visual_detections
        detected_helmet = any(
            "helmet" in b.class_name.lower()
            and b.confidence >= cfg.helmet_confidence_threshold
            for b in boxes
        )
        # Detect explicit no-helmet class.
        no_helmet = any(
            "no_helmet" in b.class_name.lower()
            or "no-helmet" in b.class_name.lower()
            for b in boxes
        )

        if no_helmet:
            return "HELMET: NO HELMET", cfg.color_violation
        if detected_helmet:
            return "HELMET: OK", cfg.color_safe
        # No helmet-related class detected - use state-based inference.
        if frame_state.active_compliance_state == ComplianceState.VIOLATION:
            return "HELMET: NO HELMET", cfg.color_violation
        return "HELMET: OK", cfg.color_safe

    def _fine_risk_text(self, frame_state: FrameState) -> str:
        """Build the fine-risk text string."""
        fine = frame_state.fine_risk_aed
        if fine > 0:
            # Find the most severe violation for the message.
            vlm = frame_state.compliance_audit
            if vlm is not None and vlm.violations:
                top = max(vlm.violations, key=lambda v: v.fine_amount_aed)
                return f"FINE RISK: {fine} AED - {top.violation_type.replace('_', ' ')}"
            return f"FINE RISK: {fine} AED"
        return f"FINE RISK: {fine} AED"
