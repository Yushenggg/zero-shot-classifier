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


def test_resolve_device_gpu_accepts_any_label(monkeypatch):
    """The gpu argument is a free-form label — no whitelist, no name check.

    The scorer used to validate gpu against a hardcoded SUPPORTED_GPUS
    table and reject anything not in it. That blocked onboarding any new
    GPU (each new model required a code change) and silently accepted the
    bypass ``device = "auto"`` anyway. torch itself errors at model load
    time if CUDA is incompatible, so the validation was redundant.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("gpu", "anything-at-all") == "cuda"
    assert resolve_device("gpu", "") == "cuda"
    assert resolve_device("gpu", None) == "cuda"


def test_resolve_model_dir_without_save_to():
    assert resolve_model_dir("acme/model", None) == ("acme/model", False)


def test_resolve_model_dir_uses_existing_weights(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    source, from_disk = resolve_model_dir("acme/model", str(tmp_path))
    assert source == str(tmp_path)
    assert from_disk is True


class _DummyScorer:
    instances = 0

    def __init__(
        self,
        model_id,
        save_to=None,
        device="cpu",
        quantize="auto",
        max_image_pixels=None,
    ):
        type(self).instances += 1
        self.model_id = model_id
        self.device = device
        self.max_image_pixels = max_image_pixels


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


def test_get_scorer_distinguishes_image_pixel_cap(monkeypatch, clean_scorer_registry):
    import zero_shot.core.scorer as scorer_module

    _DummyScorer.instances = 0
    monkeypatch.setattr(scorer_module, "Scorer", _DummyScorer)
    monkeypatch.setattr(scorer_module, "resolve_device", lambda device, gpu: "cpu")
    monkeypatch.setattr(
        scorer_module, "resolve_precision", lambda device, quantize: ("float32", False)
    )

    get_scorer("acme/model", device="cpu")
    get_scorer("acme/model", device="cpu", max_image_pixels=401408)
    assert _DummyScorer.instances == 2


def _make_vision_oom_stub(monkeypatch, cap, *, cuda_available=True):
    """Bind ``Scorer._score_options_vision`` to a stub with OOM-raising helpers.

    Building a real Scorer would load torch + transformers + the actual model —
    we just need the OOM-guard code path to run. ``_score_options_vision`` is an
    unbound function on the real class; binding it to the stub gives us the real
    control flow against stubbed lower-level methods.

    ``cuda_available`` controls which device the OOM message names (``GPU memory``
    vs plain ``memory``); tests can flip it to exercise the CPU message path.
    """
    import torch

    from zero_shot.core.scorer import Scorer

    class _StubProcessor:
        image_processor = object()

    class _StubModel:
        pass

    _StubModel.__name__ = "StubVisionModel"

    class _StubScorer:
        multimodal = True
        max_image_pixels = cap
        _vision_kv_cache_ok = None
        device = "cuda" if cuda_available else "cpu"
        processor = _StubProcessor()
        _model = _StubModel()
        model_id = "acme/vision"

        def __init__(self):
            self.torch = torch  # set on instance, not class (name resolution)

        def _as_image(self, _image):
            from PIL import Image

            return Image.new("RGB", (8, 8), "red")

        def _cap_image(self, image):
            return image  # no-op so the cap path is skipped in OOM tests

        def _vision_cached(self, *_a, **_k):
            raise self.torch.cuda.OutOfMemoryError("oom in cache")

        def _vision_exact(self, *_a, **_k):
            raise self.torch.cuda.OutOfMemoryError("oom in exact")

    stub = _StubScorer()
    # Bind the real method so the OOM-guard control flow runs against the stub.
    stub._score_options_vision = Scorer._score_options_vision.__get__(stub)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    return stub


def _make_cpu_oom_stub(monkeypatch, cap):
    """Stub that raises a CPU-shaped OOM (plain ``RuntimeError``)."""
    import torch

    from zero_shot.core.scorer import Scorer

    class _StubProcessor:
        image_processor = object()

    class _StubModel:
        pass

    _StubModel.__name__ = "StubVisionCpuModel"

    class _StubScorer:
        multimodal = True
        max_image_pixels = cap
        _vision_kv_cache_ok = None
        device = "cpu"
        processor = _StubProcessor()
        _model = _StubModel()
        model_id = "acme/vision-cpu"

        def __init__(self):
            self.torch = torch

        def _as_image(self, _image):
            from PIL import Image

            return Image.new("RGB", (8, 8), "red")

        def _cap_image(self, image):
            return image

        def _vision_cached(self, *_a, **_k):
            raise RuntimeError(
                "DefaultCPUAllocator: can't allocate memory: you tried to allocate 512.00 MiB"
            )

        def _vision_exact(self, *_a, **_k):
            raise AssertionError("exact path must not be reached on cache OOM")

    stub = _StubScorer()
    stub._score_options_vision = Scorer._score_options_vision.__get__(stub)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return stub


def test_score_options_vision_oom_message_advertises_cap_when_set(monkeypatch):
    """The OOM guard must name the currently configured `max_image_pixels` cap."""
    boom = _make_vision_oom_stub(monkeypatch, cap=401_408)
    with pytest.raises(RuntimeError) as info:
        boom._score_options_vision("prompt", ["a", "b"], True, True, b"\x89PNG")
    msg = str(info.value)
    assert "Image scoring exhausted GPU memory" in msg
    assert "currently 401408" in msg
    assert "setting `max_image_pixels`" not in msg


def test_score_options_vision_oom_message_suggests_cap_when_unset(monkeypatch):
    """Without `max_image_pixels`, the message must suggest setting one."""
    boom = _make_vision_oom_stub(monkeypatch, cap=None)
    with pytest.raises(RuntimeError, match=r"setting `max_image_pixels`"):
        boom._score_options_vision("prompt", ["a", "b"], True, True, b"\x89PNG")


def test_score_options_vision_cpu_oom_message(monkeypatch):
    """CPU OOM (plain ``RuntimeError`` on memory exhaustion) surfaces the cap hint
    and names ``memory`` — not ``GPU memory``.
    """
    boom_unset = _make_cpu_oom_stub(monkeypatch, cap=None)
    with pytest.raises(RuntimeError) as info:
        boom_unset._score_options_vision(
            "prompt", ["a", "b"], True, True, b"\x89PNG"
        )
    msg = str(info.value)
    assert "Image scoring exhausted memory" in msg
    assert "GPU memory" not in msg
    assert "setting `max_image_pixels`" in msg

    boom_set = _make_cpu_oom_stub(monkeypatch, cap=401_408)
    with pytest.raises(RuntimeError) as info2:
        boom_set._score_options_vision(
            "prompt", ["a", "b"], True, True, b"\x89PNG"
        )
    msg2 = str(info2.value)
    assert "currently 401408" in msg2
    assert "setting `max_image_pixels`" not in msg2


def test_score_options_vision_cpu_oom_skips_cuda_empty_cache(monkeypatch):
    """On CPU we must not call ``torch.cuda.empty_cache`` (no CUDA allocator)."""
    import torch

    boom = _make_cpu_oom_stub(monkeypatch, cap=401_408)
    called = []

    def _spy():
        called.append(True)

    # The stub's torch already has empty_cache stubbed to no-op; replace with a
    # spy and confirm the CPU path leaves it untouched (is_available()=False).
    monkeypatch.setattr(torch.cuda, "empty_cache", _spy)
    with pytest.raises(RuntimeError):
        boom._score_options_vision("prompt", ["a", "b"], True, True, b"\x89PNG")
    assert called == []


def test_score_options_vision_oom_bubbles_through_kv_cache_fallback(monkeypatch):
    """OOM from the KV-cache path must reach the outer guard, not be swallowed.

    The inner `except oom: raise` short-circuits the fallback to the exact path,
    so the exact path must NOT have been tried and `_vision_kv_cache_ok` must
    stay ``None`` (NOT be set to ``False`` — that would disable the cache for
    this run, even though OOM is unrelated to cache correctness).
    """
    import torch

    from zero_shot.core.scorer import Scorer

    exact_called = []

    class _StubScorer:
        multimodal = True
        max_image_pixels = 401_408
        _vision_kv_cache_ok = None
        device = "cuda"
        processor = type("P", (), {"image_processor": object()})()
        _model = type("M", (), {"__name__": "StubVisionModel"})()
        model_id = "acme/vision"

        def __init__(self):
            self.torch = torch  # set on instance, not class (name resolution)

        def _as_image(self, _image):
            from PIL import Image

            return Image.new("RGB", (8, 8), "red")

        def _cap_image(self, image):
            return image

        def _vision_cached(self, *_a, **_k):
            raise self.torch.cuda.OutOfMemoryError("oom in cache")

        def _vision_exact(self, *_a, **_k):
            exact_called.append(True)
            raise AssertionError("exact path must not be reached on cache OOM")

    stub = _StubScorer()
    stub._score_options_vision = Scorer._score_options_vision.__get__(stub)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match=r"currently 401408"):
        stub._score_options_vision("prompt", ["a", "b"], True, True, b"\x89PNG")
    # Exact path was not invoked: the cache-path OOM propagated to the outer
    # guard before the KV-cache fallback could silently swallow it.
    assert exact_called == []
    # And the cache path was not flagged as bad (correct — OOM is not its fault).
    assert stub._vision_kv_cache_ok is None


def test_cap_image_returns_input_unchanged_when_cap_is_none():
    from PIL import Image

    from zero_shot.core.scorer import Scorer

    class _StubScorer:
        max_image_pixels = None

    image = Image.new("RGB", (1024, 1024), "red")
    out = Scorer._cap_image(_StubScorer(), image)
    assert out is image


def test_cap_image_returns_input_when_under_budget():
    from PIL import Image

    from zero_shot.core.scorer import Scorer

    image = Image.new("RGB", (200, 200), "blue")  # 40,000 px

    class _StubScorer:
        max_image_pixels = 40_000  # exactly at budget

    out = Scorer._cap_image(_StubScorer(), image)
    assert out is image


def test_cap_image_downsamples_over_budget_to_preserve_aspect_ratio():
    from PIL import Image

    from zero_shot.core.scorer import Scorer

    image = Image.new("RGB", (1600, 800), "red")  # 1,280,000 px, 2:1 aspect

    class _StubScorer:
        max_image_pixels = 100_000

    out = Scorer._cap_image(_StubScorer(), image)
    assert out.size[0] * out.size[1] <= 100_000
    # Aspect ratio preserved within rounding.
    assert abs(out.size[0] / out.size[1] - 2.0) < 0.01


def test_cap_image_handles_tiny_budget():
    from PIL import Image

    from zero_shot.core.scorer import Scorer

    image = Image.new("RGB", (10, 10), "red")

    class _StubScorer:
        max_image_pixels = 4  # smaller than a single pixel — must clamp to 1px

    out = Scorer._cap_image(_StubScorer(), image)
    # The contract is "total pixel count <= cap"; the old code only asserted
    # each side >= 1 and missed floor-rounding overflow on elongated inputs.
    assert out.size[0] * out.size[1] <= 4
    assert out.size[0] >= 1 and out.size[1] >= 1


def test_cap_image_clamps_elongated_input_to_fit_budget():
    """Floor-rounding both dims can push the product above the cap.

    A 10000x1 image with cap=4 used to produce 200x1 = 200 px (because
    int(10000*0.02)=200), well over the cap. The fix shrinks the longer
    side so the product actually fits.
    """
    from PIL import Image

    from zero_shot.core.scorer import Scorer

    image = Image.new("RGB", (10000, 1), "red")

    class _StubScorer:
        max_image_pixels = 4

    out = Scorer._cap_image(_StubScorer(), image)
    assert out.size[0] * out.size[1] <= 4
    # The longer side must shrink (here, 10000 -> 4); the short side stays 1.
    assert out.size[0] == 4
    assert out.size[1] == 1


def test_cap_image_clamps_tall_elongated_input_to_fit_budget():
    """Same as the wide-elongated case but with the orientation swapped."""
    from PIL import Image

    from zero_shot.core.scorer import Scorer

    image = Image.new("RGB", (1, 10000), "red")

    class _StubScorer:
        max_image_pixels = 4

    out = Scorer._cap_image(_StubScorer(), image)
    assert out.size[0] * out.size[1] <= 4
    assert out.size[0] == 1
    assert out.size[1] == 4


def test_cap_image_applies_when_score_options_vision_runs(monkeypatch):
    """End-to-end: an oversized image is downsampled before any scoring call.

    We bind the real ``_score_options_vision`` to a stub that records the image
    it received and returns canned results. The cap helper should have shrunk
    a 1000x1000 input down to ≤ max_image_pixels total pixels.
    """
    import torch

    from zero_shot.core.scorer import Scorer

    received = []

    class _StubScorer:
        multimodal = True
        max_image_pixels = 40_000
        _vision_kv_cache_ok = False  # take the exact path to keep this simple
        device = "cuda"
        processor = type("P", (), {"image_processor": object()})()
        _model = type("M", (), {"__name__": "StubVisionModel"})()
        model_id = "acme/vision"

        def __init__(self):
            self.torch = torch

        def _as_image(self, _image):
            from PIL import Image

            return Image.new("RGB", (1000, 1000), "green")

        def _vision_exact(self, prefix, options, add_space, image):
            received.append(image.size)
            return []  # one entry per option; not asserted on here

    stub = _StubScorer()
    stub._score_options_vision = Scorer._score_options_vision.__get__(stub)
    # Bind the real cap helper so we exercise it end-to-end.
    stub._cap_image = Scorer._cap_image.__get__(stub)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    stub._score_options_vision("prompt", ["a"], True, False, b"x")
    assert received == [(200, 200)]


def test_score_options_vision_validation_oom_returns_cached_result(monkeypatch):
    """If the cached path succeeds but validation OOMs, the cached result wins.

    Regression for the OOM-guard discarding a good KV-cache answer: when the
    memory-hungry exact-validation pass OOMs, the cached result already in
    hand must still come back to the caller instead of being thrown away by
    the outer guard. The cache is trusted without comparison this once;
    later calls will retry validation once memory pressure eases.
    """
    import torch

    from zero_shot.core.models import SequenceScore
    from zero_shot.core.scorer import Scorer

    cached_results = [
        SequenceScore(
            option=o,
            continuation=o,
            tokens=[],
            eos_token="<e>",
            eos_logprob=-0.1,
            total_logprob=-0.1,
        )
        for o in ("a", "b")
    ]

    class _StubScorer:
        multimodal = True
        max_image_pixels = 401_408
        _vision_kv_cache_ok = None
        device = "cuda"
        processor = type("P", (), {"image_processor": object()})()
        _model = type("M", (), {"__name__": "StubVisionModel"})()
        model_id = "acme/vision"

        def __init__(self):
            self.torch = torch  # instance attr: name resolution in scorer

        def _as_image(self, _image):
            from PIL import Image

            return Image.new("RGB", (8, 8), "red")

        def _cap_image(self, image):
            return image

        def _vision_cached(self, *_a, **_k):
            return cached_results

        def _vision_exact(self, *_a, **_k):
            raise torch.cuda.OutOfMemoryError("oom in validation")

    stub = _StubScorer()
    stub._score_options_vision = Scorer._score_options_vision.__get__(stub)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    out = stub._score_options_vision("prompt", ["a", "b"], True, True, b"\x89PNG")
    assert out is cached_results
    # Cached path succeeded, validation OOM'd -- trust the cache for next time.
    assert stub._vision_kv_cache_ok is True


def test_score_options_vision_memory_error_surfaces_cap_hint(monkeypatch):
    """A plain ``MemoryError`` from a tensor allocation must reach the OOM
    guard so the cap hint still surfaces.

    Regression for ``MemoryError`` bypassing the OOM hint: previously only
    ``torch.cuda.OutOfMemoryError`` and the C10 string were recognised, so a
    bare ``MemoryError`` from ``torch.empty`` re-raised raw and the user
    never saw the ``max_image_pixels`` advice.
    """
    import torch

    from zero_shot.core.scorer import Scorer

    class _StubScorer:
        multimodal = True
        max_image_pixels = None  # no cap set -- message must suggest one
        _vision_kv_cache_ok = False  # skip the cache path entirely
        device = "cpu"
        processor = type("P", (), {"image_processor": object()})()
        _model = type("M", (), {"__name__": "StubCpuVisionModel"})()
        model_id = "acme/vision-cpu"

        def __init__(self):
            self.torch = torch

        def _as_image(self, _image):
            from PIL import Image

            return Image.new("RGB", (8, 8), "red")

        def _cap_image(self, image):
            return image

        def _vision_cached(self, *_a, **_k):
            raise AssertionError("cache path must not be reached")

        def _vision_exact(self, *_a, **_k):
            raise MemoryError(  # noqa: TRY003 - exact exception we test for
                "CPU OOM: cannot allocate 512.00 MiB"
            )

    stub = _StubScorer()
    stub._score_options_vision = Scorer._score_options_vision.__get__(stub)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError) as info:
        stub._score_options_vision("prompt", ["a"], True, True, b"\x89PNG")
    msg = str(info.value)
    # The cap hint must surface even when the trigger is MemoryError, not
    # the CUDA OOM class or the C10 string.
    assert "MemoryError" in msg or "exhausted memory" in msg
    assert "setting `max_image_pixels`" in msg
    assert "GPU memory" not in msg


def test_score_options_vision_memory_error_in_cache_path_surfaces_cap_hint(monkeypatch):
    """``MemoryError`` from the KV-cache path must also reach the OOM guard.

    Companion to the exact-path case: the cache path's inner OOM guard
    must recognise ``MemoryError`` so it propagates to the outer handler
    instead of being misclassified as a cache-correctness failure.
    """
    import torch

    from zero_shot.core.scorer import Scorer

    class _StubScorer:
        multimodal = True
        max_image_pixels = 401_408
        _vision_kv_cache_ok = None
        device = "cpu"
        processor = type("P", (), {"image_processor": object()})()
        _model = type("M", (), {"__name__": "StubCpuVisionModel"})()
        model_id = "acme/vision-cpu"

        def __init__(self):
            self.torch = torch

        def _as_image(self, _image):
            from PIL import Image

            return Image.new("RGB", (8, 8), "red")

        def _cap_image(self, image):
            return image

        def _vision_cached(self, *_a, **_k):
            raise MemoryError("CPU OOM in cache")

        def _vision_exact(self, *_a, **_k):
            raise AssertionError("exact path must not be reached")

    stub = _StubScorer()
    stub._score_options_vision = Scorer._score_options_vision.__get__(stub)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"currently 401408"):
        stub._score_options_vision("prompt", ["a"], True, True, b"\x89PNG")
    # Cache path must NOT be marked bad: MemoryError isn't a cache-correctness
    # failure, and disabling the cache for the run would slow every future call.
    assert stub._vision_kv_cache_ok is None
