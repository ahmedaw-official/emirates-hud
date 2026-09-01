"""Tests for the OpenTelemetry tracing & structured-logging stack."""

from __future__ import annotations

import json
import time

import pytest
from opentelemetry import trace

from src.telemetry import get_logger, trace_span


# --------------------------------------------------------------------------- #
# Decorator behaviour
# --------------------------------------------------------------------------- #
def test_decorator_records_span(spans):
    @trace_span(name="double")
    def double(x: int, y: int) -> int:
        return x + y

    assert double(2, 3) == 5

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]
    assert span.name == "double"
    # Duration must be a positive float (ms).
    dur = span.attributes.get("execution.duration_ms")
    assert isinstance(dur, (int, float)) and dur >= 0
    assert span.status.status_code == trace.StatusCode.UNSET


def test_decorator_records_inputs(spans):
    @trace_span(name="echo")
    def echo(msg: str, n: int) -> str:
        return msg * n

    echo("ab", 2)

    span = spans.get_finished_spans()[0]
    assert span.attributes.get("function.name") == "echo"
    assert span.attributes.get("input.msg") == "ab"
    assert span.attributes.get("input.n") == 2


def test_decorator_records_output(spans):
    @trace_span(name="compute")
    def compute() -> dict:
        return {"result": 42, "items": [1, 2, 3]}

    compute()

    span = spans.get_finished_spans()[0]
    assert span.attributes.get("function.output.type") == "dict"
    summary = span.attributes.get("function.output.summary")
    assert summary is not None and "42" in summary


def test_decorator_records_custom_attributes(spans):
    @trace_span(name="tagged", attributes={"domain": "perception", "kind": "detection"})
    def f(): ...

    f()
    span = spans.get_finished_spans()[0]
    assert span.attributes.get("domain") == "perception"
    assert span.attributes.get("kind") == "detection"


def test_decorator_captures_exceptions(spans):
    class BoomError(Exception):
        pass

    @trace_span(name="doomed")
    def doomed(x: int) -> int:
        raise BoomError("kaboom")

    with pytest.raises(BoomError, match="kaboom"):
        doomed(1)

    span = spans.get_finished_spans()[0]
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.attributes.get("error.type") == "BoomError"
    # The exception should also be recorded as an event on the span.
    events = [e.name for e in span.events]
    assert "exception" in events


def test_decorator_output_not_recorded_on_failure(spans):
    @trace_span(name="bad")
    def bad():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        bad()

    span = spans.get_finished_spans()[0]
    assert span.attributes.get("function.output.summary") is None


def test_bare_decorator_uses_qualname(spans):
    @trace_span
    def helper(value: int) -> int:
        return value

    helper(7)

    span = spans.get_finished_spans()[0]
    assert span.name == "helper"


def test_decorator_exception_propagates(spans):
    @trace_span(name="raise")
    def raise_() -> None:
        1 / 0

    with pytest.raises(ZeroDivisionError):
        raise_()
    span = spans.get_finished_spans()[0]
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.attributes.get("error.type") == "ZeroDivisionError"


# --------------------------------------------------------------------------- #
# Context-manager behaviour
# --------------------------------------------------------------------------- #
def test_context_manager_records_span(spans):
    with trace_span("manual-segment") as span:
        span.set_attribute("step", "alpha")

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]
    assert span.name == "manual-segment"
    assert span.attributes.get("step") == "alpha"
    assert span.attributes.get("span.completed") is True
    assert span.status.status_code == trace.StatusCode.UNSET


def test_context_manager_captures_exception(spans):
    with pytest.raises(RuntimeError, match="boom"):
        with trace_span("ctx-fail") as span:
            span.set_attribute("phase", "init")
            raise RuntimeError("boom")

    span = spans.get_finished_spans()[0]
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.attributes.get("error.type") == "RuntimeError"
    assert span.attributes.get("phase") == "init"


def test_nested_spans_share_trace_id(spans):
    @trace_span(name="outer")
    def outer() -> None:
        time.sleep(0.001)
        inner()

    @trace_span(name="inner")
    def inner() -> None: ...

    outer()

    finished = spans.get_finished_spans()
    assert len(finished) == 2
    trace_ids = {s.get_span_context().trace_id for s in finished}
    assert len(trace_ids) == 1
    names = {s.name for s in finished}
    assert names == {"outer", "inner"}


# --------------------------------------------------------------------------- #
# Structured logging + OTel context propagation
# --------------------------------------------------------------------------- #
def test_log_line_inherits_trace_context(spans, log_stream):
    log = get_logger("compliance")

    @trace_span(name="evaluate")
    def evaluate(frame_id: int) -> str:
        log.info("evaluating frame", extra={"frame_id": frame_id, "risk": "LOW"})
        return "SAFE"

    evaluate(42)

    line = log_stream.getvalue().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["service.name"] == "uae-scooter-hud-test"
    assert record["level"] == "INFO"
    assert record["message"] == "evaluating frame"
    assert record["frame_id"] == 42
    assert record["risk"] == "LOW"
    assert record["trace_id"] and record["span_id"]

    # The trace id on the log line must match the span produced by the decorator.
    span = spans.get_finished_spans()[0]
    ctx = span.get_span_context()
    assert record["trace_id"] == format(ctx.trace_id, "032x")
    assert record["span_id"] == format(ctx.span_id, "016x")
