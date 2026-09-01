"""UAE Micro-Mobility Compliance & Safety HUD Engine.

Top-level package for the ``src`` layout. The package exposes the
configuration object, schema models, the observability stack and the
custom exception hierarchy used across the engine.

Typical usage::

    from src.config import get_settings
    from src.schemas import FrameState
    from src.telemetry import trace_span, get_logger

    settings = get_settings()
    logger = get_logger(__name__)

    @trace_span(name="process-frame")
    def process_frame(frame_id: int) -> FrameState:
        ...
"""

from __future__ import annotations

__all__ = ["__title__", "__version__"]

__title__ = "emirates-hud"
__version__ = "0.1.0"
