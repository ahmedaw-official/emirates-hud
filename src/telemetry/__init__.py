"""OpenTelemetry observability stack for the HUD engine.

Public API::

    from src.telemetry import (
        init_telemetry,
        trace_span,
        get_logger,
        configure_logging,
    )

The package is split into three concerns:

* :mod:`src.telemetry.provider` - ``TracerProvider`` bootstrap + exporters.
* :mod:`src.telemetry.tracing`   - the ``@trace_span`` decorator/context manager.
* :mod:`src.telemetry.logging`   - structured JSON logging with trace context.
"""

from __future__ import annotations

from .logging import (
    JsonOTELFormatter,
    configure_logging,
    get_json_logger,
    get_logger,
    is_logging_configured,
)
from .provider import (
    get_tracer,
    get_tracer_provider,
    init_telemetry,
    is_telemetry_initialized,
    shutdown_telemetry,
)
from .tracing import annotate_span, summarise_value, trace_span

__all__ = [
    "JsonOTELFormatter",
    "annotate_span",
    "configure_logging",
    "get_json_logger",
    "get_logger",
    "get_tracer",
    "get_tracer_provider",
    "init_telemetry",
    "is_logging_configured",
    "is_telemetry_initialized",
    "shutdown_telemetry",
    "summarise_value",
    "trace_span",
]

__version__ = "0.1.0"
