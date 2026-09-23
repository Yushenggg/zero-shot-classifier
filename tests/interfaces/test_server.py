"""Regression tests for the FastAPI server (no model is ever loaded)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from zero_shot.interfaces.server.app import app


@pytest.fixture
def client():
    # Intentionally no `with`: skips the lifespan so no model is loaded.
    return TestClient(app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["model_loaded"] is False
    assert body["multimodal"] is None
    assert {"model_id", "device", "gpu", "quantize", "kv_cache"} <= set(body)


def test_classify_text_success(client, fake_scorer):
    fake_scorer.logprobs = {"yes": 0.0, "no": -1.0}
    response = client.post(
        "/v1/classify/text",
        json={
            "state": {"input": "hello"},
            "questions": {"urgent": {"type": "noul", "instructions": "?"}},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "answers", "results", "usage"}
    assert body["answers"]["urgent"]["type"] == "noul"
    assert body["usage"]["input_tokens"] == 1
    assert body["usage"]["output_tokens"] == 2


def test_systemone_legacy_question_alias(client, fake_scorer):
    fake_scorer.logprobs = {"yes": 0.0, "no": -1.0}
    response = client.post(
        "/v1/systemone",
        json={"question": {"q": {"type": "noul", "instructions": "?"}}},
    )
    assert response.status_code == 200
    assert "q" in response.json()["answers"]


def test_missing_questions_is_422(client):
    response = client.post("/v1/classify/text", json={})
    assert response.status_code == 422
    assert response.json() == {"error": "`questions` is required"}


def test_invalid_question_is_400(client):
    response = client.post(
        "/v1/classify/text",
        json={"questions": {"q": {"type": "choice", "criteria": {}}}},
    )
    assert response.status_code == 400
    assert "criteria" in response.json()["error"]


def test_classify_image_success(client, fake_scorer, png_bytes):
    fake_scorer.logprobs = {"0": 0.0, "1": -1.0}
    response = client.post(
        "/v1/classify/image",
        files={"file": ("photo.png", png_bytes, "image/png")},
        data={"questions": json.dumps({"mood": {"type": "score", "criteria": ["lo", "hi"]}})},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answers"]["mood"]["type"] == "score"


def test_classify_image_empty_file_is_422(client):
    response = client.post(
        "/v1/classify/image",
        files={"file": ("empty.png", b"", "image/png")},
        data={"questions": "{}"},
    )
    assert response.status_code == 422


def test_classify_image_bad_questions_json_is_422(client, png_bytes):
    response = client.post(
        "/v1/classify/image",
        files={"file": ("photo.png", png_bytes, "image/png")},
        data={"questions": "{not json"},
    )
    assert response.status_code == 422
    assert "not valid JSON" in response.json()["error"]


def test_openapi_documents_response_and_error_models(client):
    schema = client.get("/openapi.json").json()
    post = schema["paths"]["/v1/classify/text"]["post"]
    assert "200" in post["responses"]
    assert "400" in post["responses"]
    assert "ClassifyResponse" in schema["components"]["schemas"]
    assert "ErrorResponse" in schema["components"]["schemas"]


def test_request_body_too_large_is_413(client, monkeypatch):
    from zero_shot.interfaces.server import app as app_module

    monkeypatch.setattr(app_module, "MAX_REQUEST_BYTES", 8)
    response = client.post(
        "/v1/classify/text",
        json={"questions": {"q": {"type": "noul", "instructions": "?"}}},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["error"]


def test_image_over_upload_limit_is_413(client, png_bytes, monkeypatch):
    from zero_shot.interfaces.server import app as app_module

    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 4)
    response = client.post(
        "/v1/classify/image",
        files={"file": ("photo.png", png_bytes, "image/png")},
        data={"questions": "{}"},
    )
    assert response.status_code == 413


def test_configure_swaps_config_and_resets_resolved_device():
    from zero_shot.core.config import Config
    from zero_shot.interfaces.server import app as app_module

    original_config = app_module.CONFIG
    original_device = app_module._RESOLVED_DEVICE
    try:
        app_module._RESOLVED_DEVICE = "cuda"
        returned = app_module.configure(Config(model="acme/model", device="cpu"))
        assert returned is app_module.CONFIG
        assert app_module.CONFIG.model == "acme/model"
        assert app_module.CONFIG.device == "cpu"
        # A stale resolved device must not leak into /api/health.
        assert app_module._RESOLVED_DEVICE is None
    finally:
        app_module.CONFIG = original_config
        app_module._RESOLVED_DEVICE = original_device


def test_cpu_low_config_path_points_at_bundled_preset():
    from zero_shot.interfaces.server.main import cpu_low_config_path

    path = cpu_low_config_path()
    assert path.name == "config.cpu.toml"
    # Run from the source tree, the bundled preset is found next to the project.
    assert path.exists()

