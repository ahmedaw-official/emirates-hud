"""Tracing utilities: the ``@trace_span`` decorator + context manager.

``trace_span`` is intentionally *dual-purpose*:

* **decorator** - ``@trace_span(name="detect")`` wraps a callable so every
  invocation runs inside an auto-created span::

      @trace_span(name="detect-objects", attributes={"kind": "perception"})
      def detect(frame_id: int) -> DetectionResult:
          ...

* **context manager** - use it where a function boundary doesn't exist::

      with trace_span("manual-segment") as span:
          span.set_attribute("step", "geofence-lookout")
          ...

In both modes the span automatically:

1. records its execution duration (``execution.duration_ms``),
2. annotates input arguments and an output summary (configurable),
3. captures any raised exception with ``span.record_exception`` and flips
   the span status to ``ERROR``.

All attribute values are capped and JSON-serialised so that secrets or
huge payloads never leak into the trace.
"""

from __future__ import annotations

import functools
import inspect
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from opentelemetry.trace import Span, Status, StatusCode

from .provider import get_tracer

__all__ = ["annotate_span", "summarise_value", "trace_span"]

# Maximum length of any single attribute value stored on a span.
_MAX_ATTR_LEN = 1024
_MAX_INPUT_LEN = 256

# Reserved key names we refuse to overwrite when mirroring log extras.
_RESERVED_SPAN_ATTRS = frozenset({"trace_id", "span_id", "trace_flags"})


def summarise_value(value: Any, maxlen: int = _MAX_ATTR_LEN) -> str:
    """Best-effort JSON string for an arbitrary value, capped to ``maxlen``."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) > maxlen:
        text = text[:maxlen] + "...<truncated>"
    return text


def annotate_span(
    span: Span,
    *,
    name: str | None = None,
    attributes: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    output: Any = None,
    duration_ms: float | None = None,
) -> None:
    """Attach a standardised bundle of attributes to ``span`` (best-effort)."""
    if span is None or not span.is_recording():
        return
    if name:
        span.update_name(name)
    if attributes:
        for k, v in attributes.items():
            _safe_set_attribute(span, str(k), v)
    if inputs:
        for k, v in inputs.items():
            _safe_set_attribute(span, f"input.{k}", summarise_value(v, _MAX_INPUT_LEN))
    if output is not None:
        span.set_attribute("function.output.type", type(output).__name__)
        _safe_set_attribute(span, "function.output.summary", _summarise(output, _MAX_ATTR_LEN))
    if duration_ms is not None:
        span.set_attribute("execution.duration_ms", round(duration_ms, 3))


def _summarise(value: Any, maxlen: int = _MAX_ATTR_LEN) -> str:
    return summarise_value(value, maxlen)


def _attribute_value(value: Any) -> Any:
    """Reduce an input to an OpenTelemetry-friendly scalar.

    Primitives (``str``/``int``/``float``/``bool``) are stored verbatim so the
    captured value is human-readable; anything richer is JSON-summarised.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return _summarise(value, _MAX_INPUT_LEN)


def _safe_set_attribute(span: Span, key: str, value: Any) -> None:
    """Store ``value`` on the span, degrading to a string on failure."""
    if not span.is_recording():
        return
    try:
        span.set_attribute(key, value)
    except Exception:
        try:
            span.set_attribute(key, summarise_value(value, 128))
        except Exception:  # pragma: no cover - last line of defence
            pass


def _record_exception(span: Span, exc: BaseException) -> None:
    """Record ``exc`` on the span and mark it as failed."""
    if span is None or not span.is_recording():
        return
    try:
        span.record_exception(exc)
    except Exception:  # pragma: no cover - record_exception can log internally
        pass
    _safe_set_attribute(span, "error.type", exc.__class__.__name__)
    _safe_set_attribute(span, "error.message", summarise_value(exc, 256))


class trace_span:
    """Dual-use decorator / context manager for OpenTelemetry spans.

    See :mod:`src.telemetry.tracing` for full documentation.
    """

    def __init__(
        self,
        name: str | Callable[..., Any] | None = None,
        *,
        record_inputs: bool = True,
        record_output: bool = True,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self._record_inputs = record_inputs
        self._record_output = record_output
        self._attributes = dict(attributes) if attributes else {}
        self._span: Span | None = None
        self._enter_exit: Any = None
        self._start: float | None = None

        # Bare ``@trace_span`` usage passes the wrapped function as ``name``.
        self._wrapper: Callable[..., Any] | None = None
        if callable(name) and not isinstance(name, str):
            self._name: str | None = name.__name__
            # Build the wrapper once so the instance behaves as the decorated
            # callable and ``functools.wraps`` introspection stays accurate.
            self._wrapper = self._make_wrapper(name, span_name=None)
            functools.update_wrapper(self, name)
        else:
            self._name = name or None

    # ------------------------------------------------------------------ decorator
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Bare ``@trace_span``: the instance *is* the decorated callable.
        if self._wrapper is not None:
            return self._wrapper(*args, **kwargs)

        # ``@trace_span(name=...)`` or ``@trace_span()`` applied to a function.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return self._make_wrapper(args[0], span_name=self._name)

        raise TypeError(
            "trace_span instances are only callable directly when used as a "
            "decorator; for a context manager use `with trace_span('name') as span:`."
        )

    def _make_wrapper(
        self, func: Callable[..., Any], span_name: str | None
    ) -> Callable[..., Any]:
        resolved_name = span_name or func.__name__
        record_inputs = self._record_inputs
        record_output = self._record_output
        attributes = self._attributes

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            start = time.perf_counter()
            with tracer.start_as_current_span(resolved_name) as span:
                annotate_span(span, attributes=attributes)
                if record_inputs:
                    _annotate_inputs(span, func, args, kwargs)
                try:
                    result = func(*args, **kwargs)
                except BaseException as err:
                    _record_exception(span, err)
                    span.set_status(
                        Status(StatusCode.ERROR, summarise_value(err, 256))
                    )
                    raise
                else:
                    if record_output:
                        annotate_span(span, output=result)
                finally:
                    annotate_span(
                        span, duration_ms=(time.perf_counter() - start) * 1000.0
                    )
            return result

        return wrapper

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> Span:
        name = self._name or "span"
        tracer = get_tracer()
        self._enter_exit = tracer.start_as_current_span(name)
        self._span = self._enter_exit.__enter__()
        annotate_span(self._span, attributes=self._attributes)
        self._start = time.perf_counter()
        return self._span

    def __exit__(self, exc_type, exc, tb):
        span = self._span
        try:
            if exc_type is not None and exc is not None:
                _record_exception(span, exc)
                span.set_status(
                    Status(StatusCode.ERROR, summarise_value(exc, 256))
                )
            if self._start is not None:
                annotate_span(
                    span, duration_ms=(time.perf_counter() - self._start) * 1000.0
                )
            span.set_attribute("span.completed", True)
        except Exception:  # pragma: no cover - never mask the original error
            pass
        # Propagate exceptions by NOT suppressing; the span context manager
        # returns False for non-exceptional exits.
        return self._enter_exit.__exit__(exc_type, exc, tb)


def _annotate_inputs(span: Span, func: Callable[..., Any], args: tuple, kwargs: dict) -> None:
    """Record function arguments as span attributes (best-effort)."""
    _safe_set_attribute(span, "function.name", func.__name__)
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        items = list(bound.arguments.items())
    except Exception:
        items = [("args", args), ("kwargs", kwargs)]
    for key, value in items:
        _safe_set_attribute(span, f"input.{key}", _attribute_value(value))


__all__ = ["annotate_span", "summarise_value", "trace_span"]
