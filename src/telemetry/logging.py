"""Structured JSON logging wired into the OpenTelemetry context.

Every log record is emitted as a single JSON object on stdout. When the
calling thread is inside an active span, ``trace_id`` and ``span_id``
(plus sampling flags) are injected automatically so that logs and traces
can be correlated in a backend such as Grafana Loki + Tempo.

Usage::

    from src.telemetry import configure_logging, get_logger

    configure_logging()               # called once at startup
    log = get_logger("compliance")
    log.info("frame evaluated", extra={"frame_id": 42, "state": "SAFE"})
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from typing import Any

from opentelemetry import trace

__all__ = [
    "JsonOTELFormatter",
    "configure_logging",
    "get_json_logger",
    "get_logger",
    "is_logging_configured",
]

# Keys that exist on every stdlib LogRecord and therefore must not be
# re-emitted verbatim as user "extra" attributes.
_LOG_RECORD_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class JsonOTELFormatter(logging.Formatter):
    """Render :class:`~logging.LogRecord` as a JSON line with OTel context."""

    def __init__(
        self,
        service_name: str = "uae-scooter-hud-engine",
        include_trace: bool = True,
        timestamp_key: str = "timestamp",
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.include_trace = include_trace
        self.timestamp_key = timestamp_key

    # ------------------------------------------------------------------ format
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            self.timestamp_key: self._format_time(record),
            "level": record.levelname,
            "logger": record.name,
            "service.name": self.service_name,
            "message": self._get_message(record),
        }

        if self.include_trace:
            ctx = trace.get_current_span().get_span_context()
            if ctx.is_valid:
                payload["trace_id"] = format(ctx.trace_id, "032x")
                payload["span_id"] = format(ctx.span_id, "016x")
                payload["trace_flags"] = "01" if ctx.trace_flags else "00"

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Attach any caller-supplied ``extra={...}`` keys at top level.
        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_RESERVED:
                continue
            if key in payload:
                continue
            if callable(value):
                continue
            payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _format_time(record: logging.LogRecord) -> str:
        return _dt.datetime.fromtimestamp(record.created, tz=_dt.timezone.utc).isoformat()

    @staticmethod
    def _get_message(record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.args:
            # getMessage already interpolates; nothing else to do.
            return msg
        return msg


_configured = False
_configured_service_name: str = "uae-scooter-hud-engine"


def is_logging_configured() -> bool:
    """Whether :func:`configure_logging` has already run."""
    return _configured


def configure_logging(
    level: str | int = "INFO",
    service_name: str = "uae-scooter-hud-engine",
    *,
    stream: Any = sys.stdout,
    logger: logging.Logger | None = None,
    reconfigure: bool = True,
) -> logging.Logger:
    """Configure a JSON logger on the root (or given) logger.

    Parameters
    ----------
    level:
        Logging level name ("INFO") or numeric.
    service_name:
        Emitted as ``service.name`` on every record.
    stream:
        Where to write (defaults to stdout).
    logger:
        Target logger; defaults to the root logger.
    reconfigure:
        Replace any existing handlers on the target logger.
    """
    global _configured, _configured_service_name

    log_level = level if isinstance(level, int) else _resolve_level(level)

    formatter = JsonOTELFormatter(service_name=service_name, include_trace=True)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    target = logger or logging.getLogger()
    if reconfigure:
        # Drop handlers we may have added previously, then replace the rest.
        target.handlers = [h for h in target.handlers if not isinstance(h, logging.StreamHandler)]
    target.addHandler(handler)
    target.setLevel(log_level)
    # Avoid duplicate emission through an ancestor logger.
    if logger is None:
        target.propagate = False

    _configured = True
    _configured_service_name = service_name
    return target


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger, auto-configuring JSON logging on first use."""
    if not _configured:
        from src.config import get_settings  # local import avoids a cycle

        settings = get_settings()
        configure_logging(level=settings.log_level, service_name=settings.otel_service_name)
    return logging.getLogger(name)


def get_json_logger(
    name: str | None = None,
    level: str | int = "INFO",
    service_name: str | None = None,
) -> logging.Logger:
    """Convenience wrapper returning a logger backed by the JSON formatter."""
    if not _configured:
        svc = service_name or "uae-scooter-hud-engine"
        configure_logging(level=level, service_name=svc)
    return logging.getLogger(name)


def _resolve_level(level: str) -> int:
    name = level.upper().strip()
    resolved = getattr(logging, name, None)
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown log level: {level!r}")
    return resolved
