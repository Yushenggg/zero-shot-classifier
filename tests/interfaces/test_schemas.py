"""Regression tests for the HTTP request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zero_shot.interfaces.server.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
    Usage,
)


def test_classify_request_defaults_and_legacy_alias():
    request = ClassifyRequest(question={"q": {"type": "noul"}}, model="acme/model")
    assert request.questions is None
    assert request.question == {"q": {"type": "noul"}}
    assert request.model == "acme/model"
    assert request.temperature is None
    assert request.state is None


def test_classify_response_roundtrip():
    response = ClassifyResponse(
        model="acme/model",
        answers={"q": {"type": "noul", "noul": 0.4}},
        results=[{"type": "noul", "noul": 0.4}],
        usage=Usage(input_tokens=3, output_tokens=1, timing_ms=12.5),
    )
    dumped = response.model_dump()
    assert dumped["model"] == "acme/model"
    assert dumped["usage"] == {"input_tokens": 3, "output_tokens": 1, "timing_ms": 12.5}


def test_classify_response_requires_usage():
    with pytest.raises(ValidationError):
        ClassifyResponse(model="m", answers={}, results=[])


def test_health_response_allows_unknown_multimodal():
    health = HealthResponse(
        ok=True,
        model_id="m",
        device="cpu",
        gpu="rtx_5060_ti",
        quantize="fp32",
        kv_cache=True,
        temperature=1.0,
        calibrate=True,
        model_loaded=False,
        multimodal=None,
    )
    assert health.multimodal is None


def test_error_response():
    assert ErrorResponse(error="boom").model_dump() == {"error": "boom"}
