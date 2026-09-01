"""Unit tests for :mod:`src.config`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import (
    Environment,
    Settings,
    get_settings,
    parse_resource_attributes,
    reload_settings,
)


class TestDefaults:
    def test_defaults_match_spec(self):
        s = get_settings()
        assert s.yolo_model_path == "yolov11n.pt"
        assert s.yolo_confidence_threshold == 0.40
        assert s.overpass_endpoint_url is not None
        assert str(s.overpass_endpoint_url) == "https://overpass-api.de/api/interpreter"
        assert s.vlm_model_id == "Qwen/Qwen2-VL-2B-Instruct"
        assert s.otel_service_name == "uae-scooter-hud-engine"
        assert s.osm_fallback_enabled is True
        assert s.vlm_fallback_enabled is True
        assert s.environment is Environment.DEVELOPMENT

    def test_cache_returns_same_instance(self):
        assert get_settings() is get_settings()

    def test_override_changes_value_and_breaks_cache(self):
        a = get_settings()
        b = get_settings(yolo_confidence_threshold=0.6)
        assert a is not b
        assert a.yolo_confidence_threshold == 0.4
        assert b.yolo_confidence_threshold == 0.6


class TestOverridesAndReload:
    def test_keyword_override_is_highest_priority(self, monkeypatch):
        monkeypatch.setenv("YOLO_CONFIDENCE_THRESHOLD", "0.5")
        # kwarg wins over env
        s = get_settings(yolo_confidence_threshold=0.9)
        assert s.yolo_confidence_threshold == 0.9
        monkeypatch.delenv("YOLO_CONFIDENCE_THRESHOLD", raising=False)

    def test_reload_picks_up_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = reload_settings()
        assert s.log_level == "DEBUG"
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        # second reload restores default once env is gone
        s2 = reload_settings()
        assert s2.log_level == "INFO"

    def test_env_file_overrides_defaults(self):
        # .env.example ships non-default values for several keys.
        s = Settings(_env_file=".env.example")
        assert s.otel_resource_attributes_parsed == {"env": "production", "region": "uae-dubai"}
        assert s.resource_attributes["service.name"] == "uae-scooter-hud-engine"
        assert s.resource_attributes["environment"] == "development"
        # merged attributes keep user-supplied + defaults
        ra = s.resource_attributes
        assert ra["env"] == "production"
        assert ra["region"] == "uae-dubai"

    def test_resource_attributes_merge_order(self):
        s = get_settings(otel_resource_attributes="env=staging,team=vision")
        ra = s.resource_attributes
        assert ra["env"] == "staging"
        assert ra["team"] == "vision"
        # service.name & environment auto-injected
        assert ra["service.name"] == "uae-scooter-hud-engine"
        assert ra["environment"] == "development"


class TestValidation:
    @pytest.mark.parametrize("bad", ["not-a-url", "ht!tp://", ""])
    def test_invalid_overpass_url_rejected(self, bad, monkeypatch):
        if bad == "":
            monkeypatch.delenv("OVERPASS_ENDPOINT_URL", raising=False)
        with pytest.raises(ValidationError):
            get_settings(overpass_endpoint_url=bad)

    def test_confidence_threshold_out_of_range(self):
        with pytest.raises(ValidationError):
            get_settings(yolo_confidence_threshold=1.5)
        with pytest.raises(ValidationError):
            get_settings(yolo_confidence_threshold=-0.1)

    def test_sampling_ratio_out_of_range(self):
        with pytest.raises(ValidationError):
            get_settings(otel_sampling_ratio=1.5)

    def test_invalid_environment_raises(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "mars")
        with pytest.raises(ValidationError):
            reload_settings()
        monkeypatch.delenv("ENVIRONMENT", raising=False)

    def test_invalid_log_level_raises(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            reload_settings()
        monkeypatch.delenv("LOG_LEVEL", raising=False)

    def test_empty_model_path_rejected(self):
        with pytest.raises(ValidationError):
            get_settings(yolo_model_path="   ")


class TestResourceAttributeParser:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("", {}),
            ("env=prod", {"env": "prod"}),
            ("env=prod,region=dxb", {"env": "prod", "region": "dxb"}),
            ("env=prod;region=dxb", {"env": "prod", "region": "dxb"}),
            ({"a": "b"}, {"a": "b"}),
            ([{"x": "y"}, "z=w"], {"x": "y", "z": "w"}),
        ],
    )
    def test_parse(self, raw, expected):
        assert parse_resource_attributes(raw) == expected

    def test_parse_invalid_segment(self):
        with pytest.raises(ValueError, match="k=v"):
            parse_resource_attributes("no_equals_here")

    def test_parse_unsupported_type(self):
        with pytest.raises(ValueError, match="unsupported"):
            parse_resource_attributes(12345)  # type: ignore[arg-type]
