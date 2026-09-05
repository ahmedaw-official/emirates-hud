"""HUD rendering package for the UAE Micro-Mobility Compliance & Safety HUD Engine.

Public API::

    from src.hud import HUDCompositor, VideoProcessor, HUDConfig

The package contains two main components:

* :class:`HUDCompositor`  - Draws the telemetry bar, bounding boxes, critical
  alerts and fine-risk text onto raw OpenCV BGR frames.
* :class:`VideoProcessor` - Opens a video source, feeds frames through the
  inference pipeline, passes the rendered frames to the compositor and writes
  the annotated output to a video file with codec recovery.
"""

from __future__ import annotations

from .compositor import HUDCompositor
from .video_processor import VideoProcessor

__all__ = ["HUDCompositor", "VideoProcessor"]

__version__ = "0.1.0"
