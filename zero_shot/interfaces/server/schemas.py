"""Pydantic models for the HTTP API request/response contract.

Kept in the server interface because they describe the wire format, not the
classifier's domain. ``questions`` stays ``dict[str, Any]`` here so the core
validator (which returns a 400 with a human message) is the single source of
truth for question validation, shared by the text and multipart endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    """JSON body for the text endpoints (TypeSafe-compatible)."""

    state: Any = None
    questions: dict[str, Any] | None = None
    question: dict[str, Any] | None = None  # legacy alias for `questions`
    model: str | None = None
    temperature: float | None = None
    kv_cache: bool | None = None
    calibrate: bool | None = None
    calibration_context: str | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    timing_ms: float


class ClassifyResponse(BaseModel):
    model: str
    answers: dict[str, dict[str, Any]]
    results: list[dict[str, Any]]
    usage: Usage


class HealthResponse(BaseModel):
    ok: bool
    model_id: str
    device: str
    gpu: str
    quantize: str
    kv_cache: bool
    temperature: float
    calibrate: bool
    model_loaded: bool
    # None until the model is loaded, so the UI can hide/disable image mode.
    multimodal: bool | None
    # Image pixel cap configured in `max_image_pixels` (None = no cap, model
    # native). Lets the UI show whether downsampling is active.
    max_image_pixels: int | None = None


class ErrorResponse(BaseModel):
    error: str
