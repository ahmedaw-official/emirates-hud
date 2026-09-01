"""OpenTelemetry provider bootstrap.

Initialises the *global* :class:`~opentelemetry.sdk.trace.TracerProvider`,
wires up span processors (console and/or OTLP/HTTP) and exposes a typed
:func:`get_tracer` accessor. Telemetry is opt-in: if
:func:`init_telemetry` is never called, :func:`get_tracer` falls back to a
harmless no-op tracer so the rest of the engine keeps working offline.

The provider is intentionally *idempotent*: the first call wins and
subsequent calls are ignored unless ``force=True`` is passed (used by the
test-suite to install a capture exporter).
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, TraceIdRatioBased

__all__ = [
    "get_tracer",
    "get_tracer_provider",
    "init_telemetry",
    "is_telemetry_initialized",
    "shutdown_telemetry",
]

# Sentinel package version advertised to the OTel API as the tracer's
# instrumentation library version.
_TRACER_LIB_NAME = "src.telemetry"
_TRACER_LIB_VERSION = "0.1.0"

_provider: TracerProvider | None = None
_initialized: bool = False


def is_telemetry_initialized() -> bool:
    """Return ``True`` once :func:`init_telemetry` has successfully run."""
    return _initialized


def get_tracer_provider() -> TracerProvider:
    """Return the active :class:`TracerProvider` (initialising it if needed)."""
    global _provider, _initialized
    if _provider is None:
        init_telemetry()
    return _provider  # type: ignore[return-value]


def get_tracer(instrumenting_module_name: str = _TRACER_LIB_NAME) -> trace.Tracer:
    """Return a tracer bound to the global provider.

    Safe to call before :func:`init_telemetry`: OpenTelemetry silently
    falls back to a no-op tracer in that case.
    """
    return trace.get_tracer(instrumenting_module_name, _TRACER_LIB_VERSION)


def init_telemetry(
    settings: Any = None,
    *,
    service_name: str | None = None,
    console: bool | None = None,
    otlp_endpoint: str | None = None,
    sampler: Any = None,
    span_exporter: SpanExporter | None = None,
    force: bool = False,
) -> TracerProvider:
    """Configure the global OpenTelemetry tracer provider.

    Parameters
    ----------
    settings:
        Optional :class:`src.config.Settings`; defaults to ``get_settings()``.
    service_name:
        Override for the ``service.name`` resource attribute.
    console:
        When ``True`` add a :class:`ConsoleSpanExporter`; ``False`` disables
        it. ``None`` defers to ``settings.otel_export_console``.
    otlp_endpoint:
        Override the OTLP/HTTP collector endpoint. ``None`` defers to
        ``settings.otel_endpoint``.
    sampler:
        Custom :class:`~opentelemetry.sdk.trace.sampling.Sampler`. ``None``
        derives one from ``settings.otel_sampling_ratio``.
    span_exporter:
        Optional extra :class:`SpanExporter` (e.g. an in-memory exporter in
        tests) attached via :class:`SimpleSpanProcessor` for synchronous,
        deterministic export.
    force:
        Re-install even if a provider is already registered.
    """
    global _provider, _initialized

    if _initialized and not force:
        return _provider  # type: ignore[return-value]

    # Lazy import avoids importing config at module import time (config
    # itself is lightweight, but deferring keeps the telemetry module usable
    # in isolation and avoids import cycles).
    if settings is None:
        from src.config import get_settings  # local import to avoid cycles

        settings = get_settings()

    name = service_name or settings.otel_service_name

    resource = Resource.create(
        {
            "service.name": name,
            "environment": settings.environment.value,
            **settings.resource_attributes,
        }
    )

    if sampler is None:
        ratio = settings.otel_sampling_ratio
        sampler = ALWAYS_ON if ratio >= 1.0 else TraceIdRatioBased(ratio)

    provider = TracerProvider(resource=resource, sampler=sampler)

    # Console (stdout) span exporter - handy during development.
    _console = settings.otel_export_console if console is None else console
    if _console:
        provider.add_span_processor(
            BatchSpanProcessor(
                ConsoleSpanExporter(),
                schedule_delay_msec=settings.otel_bsp_schedule_delay_ms,
            )
        )

    # OTLP/HTTP exporter - ships spans to an external collector.
    _otlp_endpoint = settings.otel_endpoint if otlp_endpoint is None else otlp_endpoint
    if settings.otel_export_otlp and _otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=str(_otlp_endpoint)),
                schedule_delay_msec=settings.otel_bsp_schedule_delay_ms,
            )
        )

    # Arbitrary extra exporter (primarily for tests).
    if span_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    _register_provider(provider, force=force)
    _provider = provider
    _initialized = True
    return provider


def _register_provider(provider: TracerProvider, *, force: bool) -> None:
    """Install ``provider`` as the global tracer provider.

    ``trace.set_tracer_provider`` raises if a provider was already set; for
    test scenarios requiring a fresh install we fall back to the internal
    setter used by the SDK itself.
    """
    try:
        trace.set_tracer_provider(provider)
    except Exception:  # pragma: no cover - branch depends on provider state
        if not force:
            raise
        # Best-effort reset for the global tracer provider.
        setattr(trace, "_TRACER_PROVIDER", provider)
        trace.set_tracer_provider(provider)


def shutdown_telemetry() -> None:
    """Flush and tear down the global provider."""
    global _provider, _initialized
    if _provider is not None:
        _provider.force_flush()
        _provider.shutdown()
        _provider = None
    _initialized = False
    # Reset the global so a later init_telemetry() call is allowed again.
    setattr(trace, "_TRACER_PROVIDER", None)
