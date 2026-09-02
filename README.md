# Emirates HUD

**UAE Micro-Mobility Compliance & Safety HUD Engine**

A real-time compliance and safety heads-up-display engine for micro-mobility
devices (e-scooters, e-bikes) operating in the United Arab Emirates. It fuses
computer-vision object detection, OpenStreetMap road-policy queries, and
vision-language-model scene understanding to produce a per-frame verdict that is
surfaced to the rider's HUD.

## Features

- **Perception pipeline** – Validated Pydantic models for YOLOv11 bounding-box
  detection results with normalised and pixel coordinate support.
- **Geospatial compliance** – OpenStreetMap / Overpass road-policy lookups with
  coordinate validation, prohibited-road detection, and fallback handling.
- **VLM scene understanding** – Structured audit responses from a
  vision-language model, including zone classification, dismount requirements,
  risk grading, and violation reporting with RTA fine attribution.
- **Frame state aggregation** – A single `FrameState` model that fuses
  perception, geospatial, and VLM data into a HUD-ready snapshot with alert
  messages and fine-risk exposure in AED.
- **Observability** – Full OpenTelemetry instrumentation (tracing + structured
  JSON logging with trace-context correlation).
- **Dynamic configuration** – `pydantic-settings` powered config with `.env`
  support and a cached, dynamically reloadable settings singleton.

## Requirements

- Python ≥ 3.10
- Dependencies are managed via `pyproject.toml` (setuptools backend).

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from src.config import get_settings
from src.schemas import FrameState
from src.telemetry import init_telemetry, get_logger, trace_span

# Bootstrap observability
init_telemetry()

# Runtime settings (loaded from .env, environment, or defaults)
settings = get_settings()
log = get_logger("my-app")

@trace_span(name="process-frame")
def process_frame(frame_id: int) -> FrameState:
    # ... build detections, geospatial policy, VLM audit ...
    return FrameState(
        frame_id=frame_id,
        timestamp=dt.datetime.now(dt.timezone.utc),
        geospatial_policy=policy,
        active_compliance_state=ComplianceState.SAFE,
        hud_alert_message="All clear",
    )
```

## Configuration

All runtime knobs are defined in `src/config.py` and can be overridden via
environment variables or a `.env` file. Copy the example to get started:

```bash
cp .env.example .env
```

Key settings include YOLO model path and thresholds, Overpass API endpoint,
VLM model ID, OpenTelemetry exporter configuration, and log level.

## Testing

```bash
pytest -q
```

The test suite includes unit tests for schemas, configuration, exceptions,
telemetry, and an end-to-end integration test.

## License

MIT – see [LICENSE](LICENSE) for details.
