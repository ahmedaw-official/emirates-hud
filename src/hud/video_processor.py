"""Video I/O processor for the HUD engine.

The :class:`VideoProcessor` is the pipeline that glues perception, geospatial,
VLM and state-engine together with the :class:`~src.hud.compositor.HUDCompositor`
and OpenCV's :class:`cv2.VideoCapture` / :class:`cv2.VideoWriter`.

Responsibilities
----------------
* Open an input video source (file path or camera index).
* Optionally resize each frame to the configured output resolution.
* Feed each frame through an externally-supplied inference callback that
  returns a :class:`~src.schemas.frame.FrameState`.
* Hand the frame + state to :class:`HUDCompositor.render_frame`.
* Write annotated frames to an output ``.mp4`` (or other container) with
  automatic codec fallback (``mp4v`` -> ``XVID`` -> ``avc1`` -> ``MJPG``).
* Track and display real-time processing FPS.

If the input source ends or an unrecoverable error occurs, the processor
stops gracefully rather than crashing the stream.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.config.schema import HUDConfig
from src.hud.compositor import HUDCompositor
from src.schemas import FrameState
from src.telemetry import get_logger, trace_span

__all__ = ["VideoProcessor"]

logger = get_logger(__name__)

# FourCC helper: cv2 uses different signatures depending on version.
# ``cv2.VideoWriter_fourcc`` is the standard path on all modern versions.
_FOURCC = cv2.VideoWriter_fourcc


class VideoProcessor:
    """Open a video source, render HUD overlays and write annotated output.

    Parameters
    ----------
    input_source:
        Integer camera index (e.g. ``0``) or path to a video file.
    output_path:
        Path where the annotated video will be written.
    config:
        :class:`~src.config.schema.HUDConfig` instance.  Defaults to a
        standard 720p configuration.
    inference_callback:
        Callable that accepts a raw BGR frame and returns a
        :class:`~src.schemas.frame.FrameState` (or ``None`` to skip HUD
        annotation for that frame).
    compositor:
        Optional pre-configured :class:`HUDCompositor`; one is created from
        ``config`` if not supplied.
    """

    def __init__(
        self,
        input_source: int | str,
        output_path: str,
        config: HUDConfig | None = None,
        inference_callback=None,
        compositor: HUDCompositor | None = None,
    ) -> None:
        self.input_source = input_source
        self.output_path = output_path
        self.config = config or HUDConfig()
        self._inference_callback = inference_callback
        self._compositor = compositor or HUDCompositor(self.config)
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._codec_used: str = "none"

        # Output / source dimensions (set during open()).
        self._output_w = self.config.output_width
        self._output_h = self.config.output_height
        self._source_w = self.config.output_width
        self._source_h = self.config.output_height

        # FPS tracking.
        self._frame_times: list[float] = []
        self._processing_fps: float = 0.0

    # ------------------------------------------------------------------ properties
    @property
    def codec_used(self) -> str:
        """Return the codec FourCC that was successfully opened."""
        return self._codec_used

    @property
    def processing_fps(self) -> float:
        """Smoothed FPS of the processing loop (0 until enough samples)."""
        return self._processing_fps

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> VideoProcessor:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def open(self) -> bool:
        """Open the input video source and prepare the output writer.

        Returns ``True`` if both input and output were initialised.
        """
        self._cap = cv2.VideoCapture(self.input_source)
        if not self._cap.isOpened():
            logger.error("Failed to open video source: %s", self.input_source)
            return False

        # Read first frame to determine source dimensions (fallback to config).
        ret, first_frame = self._cap.read()
        if not ret or first_frame is None:
            logger.error("Failed to read first frame from video source.")
            self._cap.release()
            self._cap = None
            return False

        h, w = first_frame.shape[:2]
        self._output_w = self.config.output_width
        self._output_h = self.config.output_height
        self._source_w = w
        self._source_h = h

        # Re-create the capture to start from frame 0.
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Open the writer with codec recovery.
        if not self._open_writer():
            logger.error("Failed to open VideoWriter with any codec.")
            return False

        # Store the first frame for the processing loop.
        self._first_frame = first_frame

        logger.info(
            "VideoProcessor opened: %s -> %s (codec=%s, %dx%d)",
            self.input_source, self.output_path,
            self._codec_used, self._output_w, self._output_h,
        )
        return True

    def _open_writer(self) -> bool:
        """Try each codec in the preference list until one works."""
        for codec in self.config.codec_preferences:
            fourcc = _FOURCC(*list(codec))
            writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                30.0,  # default FPS; real FPS tracked separately
                (self._output_w, self._output_h),
                True,
            )
            if writer.isOpened():
                self._writer = writer
                self._codec_used = codec
                logger.info("VideoWriter opened with codec '%s'", codec)
                return True
            # Clean up failed attempt.
            try:
                writer.release()
            except Exception:
                pass
            logger.warning("Codec '%s' failed, trying next.", codec)

        return False

    def release(self) -> None:
        """Release the capture and writer resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None

    # ------------------------------------------------------------------ processing
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: resize, run inference, render HUD.

        Returns the annotated frame.  If inference or rendering fails,
        the resized raw frame is returned (overlay exception shield).
        """
        # Resize to output resolution.
        resized = cv2.resize(
            frame, (self._output_w, self._output_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Run inference if a callback is configured.
        frame_state: FrameState | None = None
        if self._inference_callback is not None:
            try:
                frame_state = self._inference_callback(resized)
            except Exception as exc:
                logger.error("Inference callback error: %s", exc, exc_info=True)
                frame_state = None

        # Render HUD.
        if frame_state is not None:
            try:
                annotated = self._compositor.render_frame(
                    resized,
                    frame_state,
                    speed_kmh=getattr(frame_state, "speed_kmh", 0.0),
                    processing_fps=self._processing_fps,
                )
            except Exception as exc:
                logger.error(
                    "HUD render error, using raw frame: %s", exc, exc_info=True
                )
                annotated = resized
        else:
            annotated = resized

        return annotated

    def run(
        self,
        max_frames: int | None = None,
    ) -> int:
        """Run the full video processing loop.

        Parameters
        ----------
        max_frames:
            Optional limit on the number of frames to process (for testing).

        Returns
        -------
        int
            Number of frames successfully written.
        """
        if self._cap is None or self._writer is None:
            if not self.open():
                return 0

        assert self._cap is not None
        assert self._writer is not None

        frame_idx = 0
        start_time = time.perf_counter()

        try:
            with trace_span("hud.video_processor.run") as span:
                span.set_attribute("hud.codec_used", self._codec_used)

                while True:
                    if max_frames is not None and frame_idx >= max_frames:
                        break

                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        logger.info("End of video stream or read error at frame %d", frame_idx)
                        break

                    processed = self.process_frame(frame)

                    # Ensure the processed frame matches writer dimensions.
                    if processed.shape[0] != self._output_h or processed.shape[1] != self._output_w:
                        processed = cv2.resize(
                            processed, (self._output_w, self._output_h),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    self._writer.write(processed)

                    # Update FPS tracking.
                    self._frame_times.append(time.perf_counter())
                    if len(self._frame_times) > 30:
                        self._frame_times.pop(0)
                    if len(self._frame_times) >= 2:
                        elapsed = self._frame_times[-1] - self._frame_times[0]
                        if elapsed > 0:
                            self._processing_fps = (len(self._frame_times) - 1) / elapsed
                    frame_idx += 1

                elapsed = time.perf_counter() - start_time
                span.set_attribute(
                    "hud.frames_processed", frame_idx
                )
                span.set_attribute(
                    "hud.total_duration_s", round(elapsed, 3)
                )
                span.set_attribute(
                    "hud.avg_fps", round(frame_idx / elapsed, 2) if elapsed > 0 else 0
                )
                logger.info(
                    "Video processing complete: %d frames in %.2fs (avg %.1f FPS)",
                    frame_idx, elapsed,
                    frame_idx / elapsed if elapsed > 0 else 0,
                )

        except Exception as exc:
            logger.error(
                "Fatal error in processing loop (wrote %d frames): %s",
                frame_idx, exc, exc_info=True,
            )
            raise

        return frame_idx
