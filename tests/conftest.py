"""Shared pytest fixtures for the HUD engine test-suite.

* A session-scoped fixture installs a *deterministic* OpenTelemetry
  ``TracerProvider`` backed by an in-memory exporter so that every test
  can inspect the spans produced by ``@trace_span`` without depending on
  network access or a real collector.
* A function-scoped fixture captures structured JSON log lines emitted
  through :mod:`src.telemetry.logging` and restores the root logger on
  teardown so tests do not leak handlers.
"""

from __future__ import annotations

import io
import logging

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.telemetry import init_telemetry, shutdown_telemetry
from src.telemetry.logging import configure_logging

# Ensure a clean, deterministic OTel baseline before the session starts.
shutdown_telemetry()


@pytest.fixture(scope="session")
def otel_exporter() -> InMemorySpanExporter:
    """Session-wide in-memory span recorder attached to the global provider."""
    exporter = InMemorySpanExporter()
    # force=True lets us (re)install even though shutdown_telemetry ran above.
    init_telemetry(
        console=False,
        span_exporter=exporter,
        service_name="uae-scooter-hud-test",
        force=True,
    )
    yield exporter
    shutdown_telemetry()


@pytest.fixture
def spans(otel_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    """Clear the recorder at the start of each test and return it for assertions."""
    otel_exporter.clear()
    return otel_exporter


@pytest.fixture
def log_stream():
    """Capture structured JSON logs on the root logger for one test.

    Yields the backing :class:`io.StringIO` so the test can ``json.loads`` the
    emitted line(s). The root logger is restored on teardown.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_propagate = root.propagate

    stream = io.StringIO()
    configure_logging(
        level="DEBUG",
        service_name="uae-scooter-hud-test",
        stream=stream,
        reconfigure=True,
    )

    try:
        yield stream
    finally:
        root = logging.getLogger()
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        root.propagate = saved_propagate


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Make settings tests deterministic by clearing overridden env vars."""
    for key in [
        "YOLO_MODEL_PATH",
        "YOLO_CONFIDENCE_THRESHOLD",
        "OVERPASS_ENDPOINT_URL",
        "VLM_MODEL_ID",
        "OTEL_SERVICE_NAME",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORT_CONSOLE",
        "OTEL_EXPORT_OTLP",
        "ENVIRONMENT",
        "LOG_LEVEL",
        "OTEL_RESOURCE_ATTRIBUTES",
    ]:
        monkeypatch.delenv(key, raising=False)
