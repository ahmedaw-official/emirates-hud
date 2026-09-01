"""Tests for the custom exception hierarchy (src/exceptions.py)."""

from __future__ import annotations

import pytest

from src.exceptions import (
    GeospatialAPIError,
    HUDBaseException,
    PerceptionEngineError,
    VLMInferenceError,
)


class TestHierarchy:
    @pytest.mark.parametrize(
        "exc_type,expected_code,expected_status",
        [
            (PerceptionEngineError, "perception_error", 503),
            (GeospatialAPIError, "geospatial_api_error", 502),
            (VLMInferenceError, "vlm_inference_error", 503),
        ],
    )
    def test_subclasses_share_base_and_codes(self, exc_type, expected_code, expected_status):
        err = exc_type("boom")
        assert isinstance(err, HUDBaseException)
        assert err.code == expected_code
        assert err.status_code == expected_status
        assert err.message == "boom"
        assert err.details == {}

    def test_base_is_not_leaf_of_others(self):
        # Each leaf must be a distinct subclass.
        assert not issubclass(PerceptionEngineError, GeospatialAPIError)
        assert not issubclass(VLMInferenceError, PerceptionEngineError)


class TestConstruction:
    def test_details_can_carry_structured_context(self):
        err = PerceptionEngineError(
            "yolo blew up",
            code="custom_code",
            details={"model": "yolov11n.pt", "frame_id": 42},
            status_code=500,
        )
        assert err.code == "custom_code"
        assert err.status_code == 500
        assert err.details == {"model": "yolov11n.pt", "frame_id": 42}

    def test_cause_chaining(self):
        root = RuntimeError("the root cause")
        err = VLMInferenceError("vlm failed", cause=root)
        assert err.__cause__ is root

    def test_str_representation(self):
        err = GeospatialAPIError("timeout", details={"url": "https://overpass.api"})
        text = str(err)
        assert "GeospatialAPIError" in text
        assert "timeout" in text
        assert "[geospatial_api_error]" in text  # includes the code

    def test_to_dict_is_json_serialisable(self):
        import json

        err = VLMInferenceError("failed", code="vlm_x", details={"k": "v"}, status_code=503)
        rendered = json.dumps(err.to_dict())
        decoded = json.loads(rendered)
        assert decoded["error"] == "VLMInferenceError"
        assert decoded["code"] == "vlm_x"
        assert decoded["details"] == {"k": "v"}


class TestUsage:
    def test_catch_by_base(self):
        with pytest.raises(HUDBaseException):
            raise PerceptionEngineError("perceive")

    def test_raises_are_exceptions(self):
        for exc in (HUDBaseException(), PerceptionEngineError(), GeospatialAPIError(), VLMInferenceError()):
            assert isinstance(exc, Exception)
