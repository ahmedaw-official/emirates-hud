"""Pydantic schema models for the HUD rendering configuration.

``HUDConfig`` is the single source of truth for the :class:`~src.hud.compositor.HUDCompositor`
and :class:`~src.hud.video_processor.VideoProcessor`.  It centralises all
tunable colours, geometry, font settings, codec preferences and alert
thresholds so that the rendering layer never needs to reach into the
runtime :class:`~src.config.Settings`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["HUDConfig"]


def _bgr(red: int, green: int, blue: int) -> tuple[int, int, int]:
    """Return an OpenCV BGR tuple (red/green/blue in 0-255 range)."""
    return (blue, green, red)


# Convenience presets that mirror UAE road-marking semantics.
COLOR_SAFE: tuple[int, int, int] = _bgr(0, 255, 0)       # green
COLOR_WARNING: tuple[int, int, int] = _bgr(255, 215, 0)  # amber
COLOR_VIOLATION: tuple[int, int, int] = _bgr(255, 0, 0)  # red
COLOR_TEXT: tuple[int, int, int] = _bgr(255, 255, 255)   # white
COLOR_OVERLAY_BG: tuple[int, int, int] = _bgr(0, 0, 0)   # black


class HUDConfig(BaseModel):
    """Configuration for the HUD compositor and video processor.

    Every field has a sensible default so a bare ``HUDConfig()`` is ready
    to use, but any value can be overridden (typically from environment
    variables parsed through :class:`~src.config.Settings`).
    """

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------ geometry
    output_width: int = Field(
        default=1280,
        ge=1,
        description="Target output frame width in pixels.",
    )
    output_height: int = Field(
        default=720,
        ge=1,
        description="Target output frame height in pixels.",
    )

    # ------------------------------------------------------------------ overlay
    overlay_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Transparency of semi-transparent overlay panels (0 = invisible, 1 = opaque).",
    )
    bar_height: int = Field(
        default=80,
        ge=1,
        description="Height of the top telemetry bar in pixels.",
    )

    # ------------------------------------------------------------------ fonts
    font_scale: float = Field(
        default=0.6,
        gt=0.0,
        description="OpenCV font scale for body text.",
    )
    font_thickness: int = Field(
        default=1,
        ge=1,
        description="OpenCV font thickness for body text.",
    )
    title_font_scale: float = Field(
        default=0.8,
        gt=0.0,
        description="OpenCV font scale for titled / header text.",
    )
    title_font_thickness: int = Field(
        default=2,
        ge=1,
        description="OpenCV font thickness for titled / header text.",
    )

    # ------------------------------------------------------------------ colours
    color_safe: tuple[int, int, int] = Field(
        default=COLOR_SAFE,
        description="BGR colour for compliant (SAFE) elements.",
    )
    color_warning: tuple[int, int, int] = Field(
        default=COLOR_WARNING,
        description="BGR colour for warning / non-compliant elements.",
    )
    color_violation: tuple[int, int, int] = Field(
        default=COLOR_VIOLATION,
        description="BGR colour for critical violation elements.",
    )
    color_text: tuple[int, int, int] = Field(
        default=COLOR_TEXT,
        description="BGR colour for body text.",
    )
    color_overlay_bg: tuple[int, int, int] = Field(
        default=COLOR_OVERLAY_BG,
        description="BGR colour for overlay background panels.",
    )

    # ------------------------------------------------------------------ detection
    helmet_confidence_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for a helmet detection to be considered 'OK'.",
    )
    critical_sign_confidence_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for a sign/crosswalk detection to be considered critical.",
    )

    # ------------------------------------------------------------------ video codecs
    codec_preferences: list[str] = Field(
        default=["mp4v", "XVID", "avc1", "MJPG"],
        min_length=1,
        description="Ordered list of FourCC codecs to try when opening a VideoWriter.",
    )

    # ------------------------------------------------------------------ misc
    flashing_alert_interval_ms: int = Field(
        default=500,
        ge=1,
        description="Interval (ms) for toggling the critical alert banner visibility.",
    )


# Re-export the presets so callers can reference them without importing
# the private module-level constants.
__all__ = ["HUDConfig"]
