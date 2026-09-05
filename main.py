#!/usr/bin/env python
"""Master pipeline controller for the UAE Micro-Mobility Compliance & Safety HUD Engine.

This module orchestrates the entire inference pipeline end-to-end:

    raw video  --YOLOv11-->  detections
                    |
               OSM Overpass  -->  road policy
                    |
              Trigger Router -->  VLM decision
                    |
            Qwen2-VL (if triggered) -->  audit verdict
                    |
        Compliance State Machine -->  FrameState
                    |
             HUD Compositor -->  annotated frame
                    |
               VideoWriter -->  output video

The pipeline runs on CPU or GPU.  When ``--mock-vlm`` is set, the VLM is
skipped entirely and the rule-based fallback audit is used instead, making
the pipeline runnable on zero-GPU Colab instances.

Usage::

    python main.py \
        --input-video commute_raw.mp4 \
        --output-video commute_hud_demo.mp4 \
        --gps-log gps_log.json \
        --audit-log audit_trail.json
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import get_settings
from src.config.schema import HUDConfig
from src.geospatial import OSMPolicyEngine
from src.hud import HUDCompositor
from src.perception import create_perception_engine
from src.schemas import (
    ComplianceState,
    DetectionResult,
    FrameState,
    OSMRoadPolicy,
)
from src.state_engine import ComplianceStateMachine, TriggerRouter
from src.telemetry import get_logger, init_telemetry, trace_span
from src.vlm import Qwen2VLEngine, rule_based_fallback_audit

__all__ = ["CompliancePipeline", "main"]

logger = get_logger(__name__)

#: Codecs tried in order when opening a VideoWriter.
CODEC_PREFERENCES: list[str] = ["mp4v", "XVID", "avc1", "MJPG"]

#: Default GPS coordinates (Dubai Marina waterfront area) used when no GPS log is supplied.
DEFAULT_LAT: float = 25.0755
DEFAULT_LON: float = 55.1384


class CompliancePipeline:
    """End-to-end compliance pipeline: perception -> OSM -> VLM -> state -> HUD.

    Parameters
    ----------
    input_video:
        Path to the raw `.mp4` commute video.
    output_video:
        Path where the annotated HUD video will be written.
    gps_log:
        Optional path to a JSON or CSV GPS track log.  When ``None``,
        a default Dubai Marina coordinate is used for every frame.
    audit_log:
        Path to the JSON audit trail file written at the end of the run.
    mock_vlm:
        When ``True``, skip VLM model loading entirely and always use
        :func:`rule_based_fallback_audit` for VLM-triggered frames.
    """

    def __init__(
        self,
        input_video: str,
        output_video: str,
        gps_log: str | None = None,
        audit_log: str | None = None,
        mock_vlm: bool = False,
    ) -> None:
        self.input_video = str(input_video)
        self.output_video = str(output_video)
        self.audit_log_path = str(audit_log) if audit_log else None
        self.mock_vlm = mock_vlm

        # --- Telemetry bootstrap ---
        # console=False avoids BatchSpanProcessor schedule_delay_msec param
        # mismatch in some OTel SDK versions; spans are still recorded via
        # the trace context manager for pipeline.run / hud.render_frame.
        init_telemetry(console=False)

        # --- Component initialisation ---
        settings = get_settings()
        self.hud_config = HUDConfig()
        self.hud = HUDCompositor(self.hud_config)

        self.perception = create_perception_engine(
            confidence_threshold=settings.yolo_confidence_threshold,
        )

        self.osm_engine = OSMPolicyEngine(
            overpass_url=str(settings.overpass_endpoint_url),
            timeout=settings.overpass_timeout_seconds,
        )

        self.router = TriggerRouter()
        self.state_machine = ComplianceStateMachine(router=self.router)

        # VLM engine (lazy-loaded inside Qwen2VLEngine._ensure_model).
        self.vlm: Qwen2VLEngine | None = None
        self.vlm_available: bool = False
        if not self.mock_vlm:
            try:
                self.vlm = Qwen2VLEngine(
                    model_id=settings.vlm_model_id,
                    max_tokens=settings.vlm_max_tokens,
                    temperature=settings.vlm_temperature,
                )
                self.vlm_available = True
                logger.info("VLM engine initialised (model=%s)", settings.vlm_model_id)
            except Exception as exc:
                logger.warning(
                    "VLM init failed, falling back to rule-based audit: %s", exc
                )
                self.vlm = None
                self.vlm_available = False
        else:
            logger.info("VLM in mock mode (rule-based fallback only)")

        # --- GPS log ---
        self.gps_data: list[dict] = []
        if gps_log:
            self.gps_data = self._load_gps_log(gps_log)
            logger.info("Loaded %d GPS points from %s", len(self.gps_data), gps_log)

        # --- Audit log entries ---
        self._audit_entries: list[dict] = []

        # --- Counters ---
        self._total_frames: int = 0
        self._total_fines: int = 0
        self._violations_count: int = 0
        self._vlm_triggers: int = 0
        self._processing_errors: int = 0

        # --- Shutdown flag ---
        self._shutdown_requested: bool = False

        # Register signal handlers for graceful shutdown.
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------ #
    # Signal handling
    # ------------------------------------------------------------------ #
    def _signal_handler(self, signum: int, frame: object) -> None:
        """Intercept SIGINT/SIGTERM and initiate graceful shutdown."""
        logger.warning("Signal %d received - initiating graceful shutdown", signum)
        self._shutdown_requested = True

    # ------------------------------------------------------------------ #
    # GPS helpers
    # ------------------------------------------------------------------ #
    def _load_gps_log(self, path: str) -> list[dict]:
        """Load a GPS track log from a JSON or CSV file.

        Supported JSON format::

            [{"timestamp": 0.0, "lat": 25.2, "lon": 55.1, "speed_kmh": 15.0}, ...]

        Supported CSV header::

            timestamp,lat,lon,speed_kmh
        """
        p = Path(path)
        if not p.exists():
            logger.warning("GPS log not found at %s - using default coordinates", path)
            return []

        if p.suffix.lower() == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            # Normalise keys: accept "latitude"/"longitude" as aliases.
            for entry in data:
                if "latitude" in entry and "lat" not in entry:
                    entry["lat"] = entry.pop("latitude")
                if "longitude" in entry and "lon" not in entry:
                    entry["lon"] = entry.pop("longitude")
            return data

        if p.suffix.lower() == ".csv":
            entries: list[dict] = []
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entries.append({
                        "timestamp": float(row.get("timestamp", 0)),
                        "lat": float(row.get("lat", row.get("latitude", DEFAULT_LAT))),
                        "lon": float(row.get("lon", row.get("longitude", DEFAULT_LON))),
                        "speed_kmh": float(row.get("speed_kmh", 0)),
                    })
            return entries

        logger.warning("Unsupported GPS log format: %s", p.suffix)
        return []

    def _gps_for_frame(
        self, frame_idx: int, video_fps: float, frame_timestamp: float
    ) -> tuple[float, float, float]:
        """Return ``(lat, lon, speed_kmh)`` for the given frame.

        Uses the closest GPS entry by timestamp.  Falls back to default
        Dubai Marina coordinates when no GPS log is available.
        """
        if not self.gps_data:
            return DEFAULT_LAT, DEFAULT_LON, 0.0

        # Find the GPS entry with the closest timestamp.
        best_entry = None
        best_diff = float("inf")
        for entry in self.gps_data:
            diff = abs(entry.get("timestamp", 0) - frame_timestamp)
            if diff < best_diff:
                best_diff = diff
                best_entry = entry

        if best_entry is None:
            return DEFAULT_LAT, DEFAULT_LON, 0.0

        return (
            float(best_entry.get("lat", DEFAULT_LAT)),
            float(best_entry.get("lon", DEFAULT_LON)),
            float(best_entry.get("speed_kmh", 0.0)),
        )

    # ------------------------------------------------------------------ #
    # Codec fallback
    # ------------------------------------------------------------------ #
    def _open_writer(
        self, output_path: str, width: int, height: int, fps: float = 30.0
    ) -> tuple[cv2.VideoWriter, str]:
        """Open a :class:`cv2.VideoWriter` trying each codec in preference order.

        Returns ``(writer, codec_used)``.
        """
        for codec in CODEC_PREFERENCES:
            fourcc = cv2.VideoWriter_fourcc(*list(codec))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
            if writer.isOpened():
                logger.info("VideoWriter opened with codec '%s'", codec)
                return writer, codec
            try:
                writer.release()
            except Exception:
                pass
            logger.warning("Codec '%s' failed, trying next.", codec)

        raise RuntimeError(
            f"Failed to open VideoWriter with any codec. "
            f"Tried: {CODEC_PREFERENCES}"
        )

    # ------------------------------------------------------------------ #
    # Single-frame processing
    # ------------------------------------------------------------------ #
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        gps_lat: float,
        gps_lon: float,
        speed_kmh: float,
    ) -> FrameState | None:
        """Run the full pipeline on a single frame and return its :class:`FrameState`.

        The annotated frame is returned by writing it to the caller via
        :meth:`render_frame`.  If any component fails, the error is logged
        and ``None`` is returned so the caller can write the raw frame.
        """
        try:
            # Step 1: YOLO detection.
            detections: DetectionResult = self.perception.detect_frame(frame, frame_id)

            # Step 2: OSM geospatial policy.
            try:
                osm_policy: OSMRoadPolicy = self.osm_engine.get_road_policy(gps_lat, gps_lon)
            except Exception as exc:
                logger.warning("OSM query failed for frame %d: %s", frame_id, exc)
                self._processing_errors += 1
                osm_policy = OSMRoadPolicy(
                    way_id=0,
                    highway_type="unknown",
                    maxspeed_kmh=30,
                    is_prohibited_road=False,
                    query_lat_lon=(gps_lat, gps_lon),
                    is_fallback=True,
                )

            # Step 3: Trigger router - decide if VLM should run.
            should_trigger, trigger_reason = self.router.should_trigger_vlm(
                detections,
            )

            # Step 4: VLM audit (if triggered and available).
            vlm_response = None
            if should_trigger:
                self._vlm_triggers += 1
                if self.vlm_available and self.vlm is not None:
                    try:
                        vlm_response = self.vlm.audit_frame(frame, detections, osm_policy)
                    except Exception as oom_exc:
                        # Check for CUDA OOM specifically.
                        exc_msg = str(oom_exc).lower()
                        if "out of memory" in exc_msg or "cuda" in exc_msg:
                            logger.warning(
                                "CUDA OOM on frame %d: %s - "
                                "downgrading VLM to skip-mode for remaining frames.",
                                frame_id, oom_exc,
                            )
                            # Record OTel warning event on the current span.
                            try:
                                from opentelemetry import trace as otel_trace_module
                                current_span = otel_trace_module.get_current_span()
                                if current_span and current_span.is_recording():
                                    from src.telemetry.tracing import _record_exception
                                    _record_exception(current_span, oom_exc)
                                    current_span.add_event(
                                        "vlm.oom_fallback",
                                        attributes={
                                            "vlm.fallback_reason": "cuda_oom",
                                            "vlm.frame_id": frame_id,
                                        },
                                    )
                            except Exception:  # pragma: no cover
                                pass
                            self.vlm_available = False
                            self.vlm = None
                            vlm_response = None
                        else:
                            # Non-OOM VLM error - use rule-based fallback.
                            logger.warning(
                                "VLM error on frame %d: %s - using rule-based fallback.",
                                frame_id, oom_exc,
                            )
                            try:
                                vlm_response = rule_based_fallback_audit(detections, osm_policy)
                            except Exception:
                                vlm_response = None
                else:
                    # VLM not available (mock mode or OOM fallback).
                    # Use rule-based fallback for audit data.
                    try:
                        vlm_response = rule_based_fallback_audit(detections, osm_policy)
                    except Exception:
                        vlm_response = None

            # Step 5: State machine evaluation.
            frame_state = self.state_machine.evaluate_frame_state(
                detections, osm_policy, vlm_response
            )

            # Step 6: Update counters.
            if frame_state.active_compliance_state == ComplianceState.VIOLATION:
                self._violations_count += 1
            if frame_state.fine_risk_aed > 0:
                self._total_fines += frame_state.fine_risk_aed

            # Step 7: Record audit log entry.
            self._audit_entries.append({
                "frame_id": frame_state.frame_id,
                "timestamp": frame_state.timestamp.isoformat(),
                "state": frame_state.active_compliance_state.value,
                "fine_risk_aed": frame_state.fine_risk_aed,
                "vlm_triggered": should_trigger,
                "trigger_reason": trigger_reason,
                "zone_classification": (
                    vlm_response.zone_classification.value
                    if vlm_response
                    else "N/A"
                ),
                "risk_level": (
                    vlm_response.risk_level.value if vlm_response else "N/A"
                ),
                "hud_alert_message": frame_state.hud_alert_message,
            })

            return frame_state

        except Exception as exc:
            logger.error(
                "Pipeline error on frame %d: %s - passing raw frame",
                frame_id, exc, exc_info=True
            )
            self._processing_errors += 1
            return None

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        """Open input video, process all frames, write output, flush audit log.

        Returns
        -------
        int
            Number of frames successfully written to the output video.
        """
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            logger.error("Failed to open input video: %s", self.input_video)
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        logger.info(
            "Input video: %s (FPS=%.1f, estimated frames=%d)",
            self.input_video, fps, total_frames_est,
        )

        # Read first frame to get dimensions.
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            logger.error("Cannot read first frame from video.")
            cap.release()
            return 0

        h, w = first_frame.shape[:2]
        out_w = self.hud_config.output_width
        out_h = self.hud_config.output_height

        # Open output writer with codec fallback.
        try:
            writer, codec_used = self._open_writer(
                self.output_video, out_w, out_h, fps=fps
            )
        except RuntimeError as exc:
            logger.error("Output writer initialization failed: %s", exc)
            cap.release()
            return 0

        start_time = time.perf_counter()
        frames_written: int = 0
        processing_fps: float = 0.0
        frame_timestamps: list[float] = []

        try:
            with trace_span("pipeline.run") as span:
                span.set_attribute("pipeline.input_video", self.input_video)
                span.set_attribute("pipeline.output_video", self.output_video)
                span.set_attribute("pipeline.codec_used", codec_used)
                span.set_attribute("pipeline.mock_vlm", self.mock_vlm)
                span.set_attribute("pipeline.input_fps", fps)
                span.set_attribute("pipeline.input_resolution", f"{w}x{h}")
                span.set_attribute("pipeline.output_resolution", f"{out_w}x{out_h}")

                frame_idx: int = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                while True:
                    if self._shutdown_requested:
                        logger.info("Shutdown requested, flushing remaining buffers...")
                        break

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.info("End of video stream or read error at frame %d", frame_idx)
                        break

                    frame_timestamp = frame_idx / fps
                    lat, lon, speed = self._gps_for_frame(frame_idx, fps, frame_timestamp)

                    # Process the frame.
                    frame_state = self._process_frame(
                        frame, frame_idx, lat, lon, speed
                    )

                    # Render HUD onto the frame.
                    if frame_state is not None:
                        annotated = self.hud.render_frame(
                            frame, frame_state,
                            speed_kmh=speed,
                            processing_fps=processing_fps,
                        )
                    else:
                        # Exception shield: write raw resized frame.
                        annotated = cv2.resize(
                            frame, (out_w, out_h),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    # Ensure output dimensions match writer.
                    if annotated.shape[0] != out_h or annotated.shape[1] != out_w:
                        annotated = cv2.resize(
                            annotated, (out_w, out_h),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    writer.write(annotated)
                    frames_written += 1
                    self._total_frames = frame_idx + 1

                    # Track FPS.
                    now = time.perf_counter()
                    frame_timestamps.append(now)
                    if len(frame_timestamps) > 30:
                        frame_timestamps.pop(0)
                    if len(frame_timestamps) >= 2:
                        elapsed = frame_timestamps[-1] - frame_timestamps[0]
                        if elapsed > 0:
                            processing_fps = (len(frame_timestamps) - 1) / elapsed

                    frame_idx += 1

                    if total_frames_est > 0 and frame_idx % 50 == 0:
                        progress = (frame_idx / total_frames_est) * 100
                        logger.info(
                            "Progress: %d/%d frames (%.1f%%) - "
                            "state=%s, fine=%d AED, vlm_trig=%s, fps=%.1f",
                            frame_idx, total_frames_est, progress,
                            frame_state.active_compliance_state.value if frame_state else "N/A",
                            frame_state.fine_risk_aed if frame_state else 0,
                            self.router.frames_since_last_vlm,
                            processing_fps,
                        )

                # Flush writer buffers.
                try:
                    writer.release()
                except Exception:
                    pass

                elapsed = time.perf_counter() - start_time
                avg_fps = frames_written / elapsed if elapsed > 0 else 0.0

                # Record pipeline summary metrics on the span.
                span.set_attribute("pipeline.total_frames_processed", frames_written)
                span.set_attribute("pipeline.average_fps", round(avg_fps, 2))
                span.set_attribute("pipeline.vlm_triggers_count", self._vlm_triggers)
                span.set_attribute("pipeline.total_fines_accrued_aed", self._total_fines)
                span.set_attribute("pipeline.violations_detected_count", self._violations_count)
                span.set_attribute("pipeline.processing_errors", self._processing_errors)

                logger.info(
                    "Pipeline complete: %d frames in %.2fs (avg %.1f FPS)",
                    frames_written, elapsed, avg_fps,
                )
                logger.info(
                    "Summary: VLM triggers=%d, total fines=%d AED, "
                    "violations=%d, errors=%d",
                    self._vlm_triggers, self._total_fines,
                    self._violations_count, self._processing_errors,
                )

        except Exception as exc:
            logger.error("Fatal pipeline error: %s", exc, exc_info=True)
        finally:
            cap.release()

        # Write audit log.
        self._write_audit_log()

        return frames_written

    def _write_audit_log(self) -> None:
        """Write the accumulated audit entries to the audit log file."""
        if not self.audit_log_path:
            return
        try:
            with open(self.audit_log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": {
                            "total_frames_processed": self._total_frames,
                            "vlm_triggers_count": self._vlm_triggers,
                            "total_fines_accrued_aed": self._total_fines,
                            "violations_detected_count": self._violations_count,
                            "processing_errors": self._processing_errors,
                        },
                        "entries": self._audit_entries,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info("Audit log written to %s", self.audit_log_path)
        except Exception as exc:
            logger.error("Failed to write audit log: %s", exc)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UAE Micro-Mobility Compliance & Safety HUD Engine - Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python main.py --input-video commute_raw.mp4 \\\n"
            "    --output-video commute_hud_demo.mp4 \\\n"
            "    --gps-log gps_log.json --audit-log audit_trail.json\n"
        ),
    )
    parser.add_argument(
        "--input-video",
        type=str,
        required=True,
        help="Path to raw input commute video (.mp4).",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        required=True,
        help="Path for processed HUD video export (.mp4).",
    )
    parser.add_argument(
        "--gps-log",
        type=str,
        default=None,
        help="Path to JSON/CSV simulated GPS track log.",
    )
    parser.add_argument(
        "--mock-vlm",
        action="store_true",
        default=False,
        help="Run VLM in mock mode (skip model loading, use rule-based fallback).",
    )
    parser.add_argument(
        "--audit-log",
        type=str,
        default=None,
        help="Path to export audit trail JSON file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the master pipeline."""
    args = parse_args(argv)
    pipeline = CompliancePipeline(
        input_video=args.input_video,
        output_video=args.output_video,
        gps_log=args.gps_log,
        audit_log=args.audit_log,
        mock_vlm=args.mock_vlm,
    )
    frames_written = pipeline.run()
    logger.info("Pipeline finished. Output: %s (%d frames)", args.output_video, frames_written)
    return 0 if frames_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
