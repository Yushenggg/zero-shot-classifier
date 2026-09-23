"""Regression tests for device/precision resolution and the scorer cache."""

from __future__ import annotations

import pytest

from zero_shot.core.scorer import (
    get_scorer,
    is_loaded,
    loaded_scorer,
    resolve_device,
    resolve_model_dir,
    resolve_precision,
)


@pytest.mark.parametrize(
    ("device", "quantize", "expected"),
    [
        ("cpu", "auto", ("float32", False)),
        ("cpu", "fp32", ("float32", False)),
        ("cpu", "bf16", ("bfloat16", False)),
        ("cpu", "int8", ("float32", True)),
        ("cpu", "none", ("bfloat16", False)),  # legacy alias for bf16
        ("cuda", "auto", ("bfloat16", False)),
        ("cuda", "int8", ("bfloat16", False)),  # CUDA is never quantized
    ],
)
def test_resolve_precision(monkeypatch, device, quantize, expected):
    monkeypatch.setattr("torch.cuda.is_available", lambda: device == "cuda")
    assert resolve_precision(device, quantize) == expected


def test_resolve_precision_none_quantize_is_auto(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_precision("cpu", None) == ("float32", False)


def test_resolve_precision_auto_detects_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_precision("auto", "auto") == ("bfloat16", False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_precision("auto", "auto") == ("float32", False)


def test_resolve_precision_rejects_unknown_mode():
    with pytest.raises(ValueError):
        resolve_precision("cpu", "nope")


def test_resolve_device_cpu_and_auto(monkeypatch):
    assert resolve_device("cpu", None) == "cpu"
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("auto", None) == "cuda"
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto", None) == "cpu"


def test_resolve_device_rejects_unknown_and_missing_cuda(monkeypatch):
    with pytest.raises(ValueError):
        resolve_device("tpu", None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("gpu", None)
    with pytest.raises(RuntimeError):
        resolve_device("cuda", None)


def test_resolve_device_validates_gpu_name(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda index=0: "NVIDIA GeForce RTX 5060 Ti")
    assert resolve_device("gpu", "rtx_5060_ti") == "cuda"
    with pytest.raises(ValueError):
        resolve_device("gpu", "not-a-gpu")


def test_resolve_device_rejects_gpu_name_mismatch(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda index=0: "NVIDIA RTX A4000")
    with pytest.raises(RuntimeError):
        resolve_device("gpu", "rtx_5060_ti")


def test_resolve_model_dir_without_save_to():
    assert resolve_model_dir("acme/model", None) == ("acme/model", False)


def test_resolve_model_dir_uses_existing_weights(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    source, from_disk = resolve_model_dir("acme/model", str(tmp_path))
    assert source == str(tmp_path)
    assert from_disk is True


class _DummyScorer:
    instances = 0

    def __init__(self, model_id, save_to=None, device="cpu", quantize="auto"):
        type(self).instances += 1
        self.model_id = model_id
        self.device = device


def test_get_scorer_caches_per_key(monkeypatch, clean_scorer_registry):
    import zero_shot.core.scorer as scorer_module

    _DummyScorer.instances = 0
    monkeypatch.setattr(scorer_module, "Scorer", _DummyScorer)
    monkeypatch.setattr(scorer_module, "resolve_device", lambda device, gpu: "cpu")
    monkeypatch.setattr(
        scorer_module, "resolve_precision", lambda device, quantize: ("float32", False)
    )

    first = get_scorer("acme/model", device="cpu")
    second = get_scorer("acme/model", device="cpu")
    assert first is second
    assert _DummyScorer.instances == 1

    assert is_loaded("acme/model") is True
    assert is_loaded("other/model") is False
    assert loaded_scorer("acme/model") is first
    assert loaded_scorer("other/model") is None


def test_get_scorer_distinguishes_models(monkeypatch, clean_scorer_registry):
    import zero_shot.core.scorer as scorer_module

    _DummyScorer.instances = 0
    monkeypatch.setattr(scorer_module, "Scorer", _DummyScorer)
    monkeypatch.setattr(scorer_module, "resolve_device", lambda device, gpu: "cpu")
    monkeypatch.setattr(
        scorer_module, "resolve_precision", lambda device, quantize: ("float32", False)
    )

    get_scorer("a/model", device="cpu")
    get_scorer("b/model", device="cpu")
    assert _DummyScorer.instances == 2
