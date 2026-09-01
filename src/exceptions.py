"""Custom exception hierarchy for the HUD engine.

Every domain-specific error derives from :class:`HUDBaseException` so that
application code can catch the whole family with a single ``except`` while
still being able to discriminate between a perception failure, a
geospatial API outage and a VLM inference error.

Each exception carries:

* ``message``      - human readable description.
* ``code``         - stable, machine-readable error code (snake_case).
* ``details``      - free-form structured context (never serialised to the
                     exception string, so secrets can safely live here).
* ``status_code``  - optional HTTP-ish status code for API surfaces that map
                     exceptions directly to responses.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GeospatialAPIError",
    "HUDBaseException",
    "PerceptionEngineError",
    "VLMInferenceError",
]


class HUDBaseException(Exception):
    """Base class for every error raised by the HUD engine."""

    #: Stable machine-readable identifier for the error family.
    code: str = "hud_error"

    #: Suggested HTTP status code (0 means "not an HTTP error").
    status_code: int = 0

    def __init__(
        self,
        message: str = "An unexpected error occurred in the HUD engine.",
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details: dict[str, Any] = dict(details) if details else {}
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return f"{self.__class__.__name__} [{self.code}]: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (safe to log)."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


class PerceptionEngineError(HUDBaseException):
    """Raised when the YOLO / vision pipeline cannot produce a result."""

    code = "perception_error"
    status_code = 503

    def __init__(self, message: str = "Perception engine failure.", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class GeospatialAPIError(HUDBaseException):
    """Raised when the OpenStreetMap / Overpass request fails or times out."""

    code = "geospatial_api_error"
    status_code = 502

    def __init__(self, message: str = "Geospatial API request failed.", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class VLMInferenceError(HUDBaseException):
    """Raised when the Vision-Language Model cannot return an audit verdict."""

    code = "vlm_inference_error"
    status_code = 503

    def __init__(self, message: str = "VLM inference failed.", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
