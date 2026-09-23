from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL_ID, SUPPORTED_GPUS

logger = logging.getLogger(__name__)

_MIN_AVAILABLE_GB = 9.0

# Cached and exact vision scoring differ only by attention/GEMM shapes (~0.05-0.4
# nats on near-tie tokens), so a correct position/rope setup stays far below this.
# A model whose multimodal positions we mishandle diverges by several nats.
_VISION_CACHE_TOLERANCE_NATS = 1.0

_QUANTIZE_MODES = ("auto", "bf16", "fp32", "int8")


def resolve_precision(device: str, quantize: str = "auto") -> tuple[str, bool]:
    """Return ``(dtype_name, quantized)`` for a run.

    - ``auto``: bf16 on CUDA, **fp32 on CPU**. CPU stays on fp32 even where the
      hardware supports bf16 (AVX512_BF16/AMX): the bf16 KV-cached path drifts
      slightly from the exact path, while fp32 keeps them identical, and the
      extra memory is small. It also avoids PyTorch's slow emulated bf16 on
      Intel consumer chips since 12th gen (AVX-512 fused off).
    - ``bf16`` / ``fp32``: force that dtype (set ``bf16`` for CPU speed).
    - ``int8``: dynamic int8 quantization on CPU (loaded as fp32). CUDA is never
      quantized.
    """
    mode = (quantize or "auto").strip().lower()
    if mode == "none":  # alias for the pre-existing default
        mode = "bf16"
    if mode not in _QUANTIZE_MODES:
        raise ValueError(f"Unknown quantize {quantize!r}; use one of {_QUANTIZE_MODES}.")

    import torch

    resolved = (device or "auto").strip().lower()
    is_cpu = resolved == "cpu" or (resolved == "auto" and not torch.cuda.is_available())

    if mode == "fp32":
        return ("float32", False)
    if mode == "bf16":
        return ("bfloat16", False)
    if mode == "int8":
        return ("float32", True) if is_cpu else ("bfloat16", False)
    # auto: bf16 on GPU, fp32 on CPU
    return ("float32", False) if is_cpu else ("bfloat16", False)


def _quantize_dynamic(model):
    """Quantize Linear layers to int8 (dynamic). No-op if torch.ao is missing."""
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )


# Files we need for text scoring (the big safetensors + configs/tokenizer).
_MODEL_PATTERNS = ["*.json", "*.safetensors", "*.jinja"]


def resolve_device(device: str = "cpu", gpu: str | None = None) -> str:
    """Map the config device to a torch device string, validating the GPU.

    `device` is "auto" (use CUDA if available, else CPU), "cpu", or
    "gpu"/"cuda". When GPU mode is requested, `gpu` names one of the supported
    GPUs in `config.SUPPORTED_GPUS`.
    """
    device = (device or "auto").strip().lower()

    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device not in ("gpu", "cuda"):
        raise ValueError(f"Unknown device {device!r}; use 'auto', 'cpu', 'gpu' or 'cuda'.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU mode requested but this PyTorch has no CUDA support. Install the "
            "GPU build with `uv pip install --reinstall -r requirements-gpu.txt`, then "
            "run WITHOUT `uv run`/`uv sync` (which revert to the CPU wheel): use "
            "`.venv/bin/zero-shot-serve` or `uv run --no-sync ...`."
        )
    actual = torch.cuda.get_device_name(0)
    if gpu:
        info = SUPPORTED_GPUS.get(gpu)
        if info is None:
            raise ValueError(f"Unknown gpu {gpu!r}. Supported: {sorted(SUPPORTED_GPUS)}.")
        if str(info["name"]).lower() not in actual.lower():
            raise RuntimeError(
                f"config gpu = {gpu!r} expects {info['name']!r} but found {actual!r}."
            )
    return "cuda"


@dataclass
class TokenScore:
    token: str
    token_id: int
    logprob: float

    def to_dict(self) -> dict:
        return {"token": self.token, "token_id": self.token_id, "logprob": self.logprob}


@dataclass
class SequenceScore:
    option: str
    continuation: str
    tokens: list[TokenScore] = field(default_factory=list)
    eos_token: str = ""
    eos_logprob: float = 0.0
    total_logprob: float = 0.0

    def to_dict(self) -> dict:
        return {
            "option": self.option,
            "continuation": self.continuation,
            "tokens": [t.to_dict() for t in self.tokens],
            "eos_token": self.eos_token,
            "eos_logprob": self.eos_logprob,
            "total_logprob": self.total_logprob,
        }


def resolve_model_dir(model_id: str, save_to: str | None) -> tuple[str, bool]:
    """Return (source, from_disk).

    If `save_to` is set and already contains weights, use it directly. Otherwise
    download the model there (once) and then use the local copy.
    """
    if not save_to:
        return model_id, False

    dest = Path(save_to).expanduser()
    if dest.exists() and any(dest.glob("*.safetensors")):
        return str(dest), True

    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, local_dir=str(dest), allow_patterns=_MODEL_PATTERNS)
    return str(dest), True


class Scorer:
    """Scores a continuation under the loaded model using the full-vocabulary
    next-token distribution. Exact token probabilities (not limited to any top-k)
    and the true EOS probability are read from the logits.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        save_to: str | None = None,
        device: str = "cpu",
        quantize: str = "auto",
    ) -> None:
        if device == "cpu":
            try:
                import psutil

                available_gb = psutil.virtual_memory().available / 1e9
            except Exception:  # noqa: BLE001
                available_gb = _MIN_AVAILABLE_GB
            if available_gb < _MIN_AVAILABLE_GB:
                raise RuntimeError(
                    f"Only {available_gb:.1f} GB RAM available; the model needs about "
                    f"{_MIN_AVAILABLE_GB:.0f} GB. Close other apps or use a smaller model."
                )

        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        source, from_disk = resolve_model_dir(model_id, save_to)

        self.torch = torch
        self.model_id = model_id
        self.device = device
        self.source = source
        self.from_disk = from_disk
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=from_disk)

        config = AutoConfig.from_pretrained(source, local_files_only=from_disk)
        arch = (getattr(config, "architectures", None) or [None])[0]
        model_cls = getattr(transformers, arch, None) if arch else None
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            model_cls = transformers.AutoModelForCausalLM

        # Trust the processor for multimodal detection: if it loads with an
        # image_processor the checkpoint is multimodal. We don't second-guess by
        # poking at per-architecture vision tower attributes (vision_tower /
        # visual / vision_model / embed_vision / vision_encoder / ...) — every
        # new VLM invents a new name. If the processor says multimodal but the
        # forward pass doesn't actually accept images, the runtime error in
        # `_score_options_vision` makes that clear.
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(source, local_files_only=from_disk)
        except Exception as exc:  # noqa: BLE001 - text-only models have no processor
            logger.info(
                "No processor for %s (%s: %s); treating as text-only.",
                model_id,
                type(exc).__name__,
                exc,
            )
            self.processor = None
        self.multimodal = getattr(self.processor, "image_processor", None) is not None

        # Dynamic int8 quantized layers expect float32 activations, so load fp32
        # (then quantize) instead of bf16 when quantizing.
        self.dtype_name, self.quantized = resolve_precision(device, quantize)
        load_dtype = (
            torch.float32 if self.dtype_name == "float32" else torch.bfloat16
        )
        model = model_cls.from_pretrained(
            source,
            dtype=load_dtype,
            device_map=device,
            local_files_only=from_disk,
        )

        base = getattr(model, "model", model)
        self._base = base
        # Multimodal-RoPE models (Qwen2.5/3-VL) need explicit positions during
        # cached decoding; see _vision_cached.
        self._uses_mrope = hasattr(base, "compute_3d_position_ids")

        for attr in ("mtp",):
            if getattr(model, attr, None) is not None:
                setattr(model, attr, None)

        if self.quantized:
            logger.info(
                "Quantizing %s Linear layers to int8 (dynamic).",
                type(model).__name__,
            )
            model = _quantize_dynamic(model)

        self.language_model = getattr(base, "language_model", base)
        self.lm_head = model.get_output_embeddings() or getattr(model, "lm_head", None)
        if self.lm_head is None:
            raise RuntimeError(f"Could not find an output LM head on {type(model).__name__}.")

        model.eval()
        self._model = model

        eos = self.tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        self.eos_token_id = int(eos)

        # None = untested, True = cached path works, False = fall back to exact.
        # Kept separate per modality: a vision-specific cache failure must not
        # also disable the text cache (and vice versa).
        self._kv_cache_ok: bool | None = None
        self._vision_kv_cache_ok: bool | None = None
        # Vision scoring relies on model-level mutable state (`rope_deltas` on
        # M-RoPE models), so serialize it across threads.
        self._vision_lock = threading.Lock()

    def score_options(
        self,
        prefix_text: str,
        options: list[str],
        *,
        add_leading_space: bool = True,
        use_kv_cache: bool = True,
        image: Any = None,
        chat_template: bool = False,
    ) -> list[SequenceScore]:
        """Return sequence log-probabilities (tokens + EOS) for each option.

        With ``use_kv_cache=True`` (the default) the shared prompt is prefilled once
        and its KV cache is reused for every option (~4x faster). That path is
        *approximate*: the cached continuation uses different attention/GEMM shapes,
        so a small number of near-tie tokens can shift by ~0.05-0.4 nats. Set
        ``use_kv_cache=False`` for the exact full-sequence path.

        If the model's cache cannot be reused (no ``crop`` support, no cache
        returned, forward rejects ``past_key_values``), this falls back to the exact
        path and logs a warning once.

        ``image`` (a PIL image, raw bytes, or a path) prepends the image to the
        prompt and scores the options through the same full-vocabulary logits.
        Requires a multimodal checkpoint; text-only models raise a clear error.

        ``chat_template`` renders a text-only ``prefix_text`` through the model's
        chat template before scoring. Used for the image calibration pass so its
        content-free prior is measured in the same format as the vision main pass.
        """
        if image is not None:
            # Callers should pass add_leading_space=False for vision: the chat
            # template already ends with the assistant generation prompt, so the
            # option continues immediately. It is honored here (not forced) so a
            # vision main pass and its text-only calibration pass tokenize the
            # options identically.
            with self._vision_lock:
                return self._score_options_vision(
                    prefix_text, options, add_leading_space, use_kv_cache, image
                )

        if chat_template:
            prefix_text = self._text_chat(prefix_text)

        if use_kv_cache and self._kv_cache_ok is not False:
            try:
                result = self._score_options_cached(prefix_text, options, add_leading_space)
            except Exception as exc:  # noqa: BLE001 - any failure => fall back
                self._kv_cache_ok = False
                logger.warning(
                    "KV cache unavailable for %s (%s: %s); falling back to exact "
                    "scoring.",
                    type(self._model).__name__,
                    type(exc).__name__,
                    exc,
                )
            else:
                self._kv_cache_ok = True
                return result

        torch = self.torch
        tok = self.tokenizer
        device = next(self.language_model.parameters()).device
        prefix_ids = tok(prefix_text)["input_ids"]

        results: list[SequenceScore] = []
        for option in options:
            continuation = (" " if add_leading_space else "") + option
            full_ids = tok(prefix_text + continuation)["input_ids"]
            cont_ids = full_ids[len(prefix_ids) :]

            seq = full_ids + [self.eos_token_id]
            # Positions whose logits predict each continuation token, plus EOS.
            positions = [len(prefix_ids) + k - 1 for k in range(len(cont_ids))]
            positions.append(len(seq) - 2)

            with torch.inference_mode():
                input_ids = torch.tensor([seq], device=device)
                index = torch.tensor(positions, device=device)
                hidden = self.language_model(input_ids).last_hidden_state
                hidden = hidden[:, index, :]
                logits = self.lm_head(hidden)[0]
                logprobs = torch.log_softmax(logits.float(), dim=-1)

            tokens = [
                TokenScore(
                    token=tok.decode([tid]),
                    token_id=int(tid),
                    logprob=logprobs[i, tid].item(),
                )
                for i, tid in enumerate(cont_ids)
            ]
            eos_logprob = logprobs[len(cont_ids), self.eos_token_id].item()
            total = sum(t.logprob for t in tokens) + eos_logprob

            results.append(
                SequenceScore(
                    option=option,
                    continuation=continuation,
                    tokens=tokens,
                    eos_token=tok.decode([self.eos_token_id]),
                    eos_logprob=eos_logprob,
                    total_logprob=total,
                )
            )
        return results

    def _score_options_cached(
        self,
        prefix_text: str,
        options: list[str],
        add_leading_space: bool,
    ) -> list[SequenceScore]:
        """Approximate but ~4x faster: prefill the prompt once, reuse its KV cache."""
        torch = self.torch
        tok = self.tokenizer
        device = next(self.language_model.parameters()).device
        prefix_ids = tok(prefix_text)["input_ids"]
        prefix_len = len(prefix_ids)

        with torch.inference_mode():
            prefill = self.language_model(
                torch.tensor([prefix_ids], device=device), use_cache=True
            )
            prefix_logprobs = torch.log_softmax(
                self.lm_head(prefill.last_hidden_state[:, -1:, :])[0].float(), dim=-1
            )[0]
            cache = prefill.past_key_values
            # Drop the full prefill output (last_hidden_state/logits + ModelOutput
            # wrapper) before the per-option loop. Without this, the full hidden
            # state for the whole prefix stays live for every option and the loop
            # allocations pile on top.
            del prefill
        if cache is None or not hasattr(cache, "crop"):
            raise RuntimeError(
                f"{type(self.language_model).__name__} did not return a reusable KV cache"
            )

        results: list[SequenceScore] = []
        for option in options:
            continuation = (" " if add_leading_space else "") + option
            full_ids = tok(prefix_text + continuation)["input_ids"]
            cont_ids = full_ids[prefix_len:]
            count = len(cont_ids)

            with torch.inference_mode():
                if count == 0:
                    logprob_values = [prefix_logprobs[self.eos_token_id].item()]
                else:
                    attention_mask = torch.ones((1, prefix_len + count), device=device, dtype=torch.long)
                    cache_position = torch.arange(prefix_len, prefix_len + count, device=device)
                    out = self.language_model(
                        torch.tensor([cont_ids], device=device),
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                    logprobs = torch.log_softmax(
                        self.lm_head(out.last_hidden_state[0]).float(), dim=-1
                    )
                    logprob_values = [prefix_logprobs[cont_ids[0]].item()]
                    logprob_values.extend(
                        logprobs[j, cont_ids[j + 1]].item() for j in range(count - 1)
                    )
                    logprob_values.append(logprobs[count - 1, self.eos_token_id].item())

            # Drop the tokens we just appended so the cache is reusable.
            if count:
                cache.crop(-count)

            tokens = [
                TokenScore(token=tok.decode([tid]), token_id=int(tid), logprob=logprob_values[i])
                for i, tid in enumerate(cont_ids)
            ]
            eos_logprob = logprob_values[-1]
            total = sum(t.logprob for t in tokens) + eos_logprob
            results.append(
                SequenceScore(
                    option=option,
                    continuation=continuation,
                    tokens=tokens,
                    eos_token=tok.decode([self.eos_token_id]),
                    eos_logprob=eos_logprob,
                    total_logprob=total,
                )
            )
        return results

    @staticmethod
    def _as_image(image: Any):
        """Coerce a PIL image, raw bytes, or a filesystem path into a PIL image."""
        from PIL import Image

        from .image_utils import decode_image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray, memoryview)):
            return decode_image(bytes(image))
        if isinstance(image, (str, os.PathLike)):
            if not os.path.exists(image):
                raise ValueError(f"Image file not found: {image}")
            with open(image, "rb") as fh:
                return decode_image(fh.read())
        raise TypeError(f"Unsupported image type {type(image).__name__}")

    def _apply_chat(self, messages: list[dict[str, Any]]) -> str:
        """Render `messages` through the chat template, assistant turn open.

        Hybrid "thinking" models (e.g. Qwen3-VL-*-Thinking) accept
        ``enable_thinking``; passing False keeps them in direct-answer mode, which
        is what we score. Processors that don't know the kwarg retry without it.
        """
        # Only pass `enable_thinking` when the template actually uses it; sending
        # it to a template that doesn't warns and does nothing.
        template = getattr(self.processor, "chat_template", None) or getattr(
            self.tokenizer, "chat_template", None
        )
        attempts = (
            ({"enable_thinking": False}, {})
            if template and "enable_thinking" in template
            else ({},)
        )

        last_error: Exception | None = None
        for extra in attempts:
            try:
                return self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **extra
                )
            except (TypeError, ValueError) as exc:  # processor rejects enable_thinking
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"{type(self.processor).__name__} cannot build a chat template "
                    f"({type(exc).__name__}: {exc})."
                ) from exc
        raise RuntimeError(
            f"{type(self.processor).__name__} cannot build a chat template ({last_error})."
        )

    def _vision_text(self, prefix_text: str) -> str:
        """Chat-template the prompt with a single image slot."""
        return self._apply_chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prefix_text},
                    ],
                }
            ]
        )

    def _text_chat(self, prefix_text: str) -> str:
        """Chat-template a text-only prompt (the image calibration pass).

        Keeps the content-free prior in the same format as the vision main pass
        while leaving the image out, so the image's own option priors are not
        subtracted away (the image is the signal, not noise to calibrate off).
        """
        if self.processor is None:
            return prefix_text
        return self._apply_chat(
            [{"role": "user", "content": [{"type": "text", "text": prefix_text}]}]
        )

    def _vision_batch(self, text: str, image: Any) -> Any:
        return self.processor(text=[text], images=[image], return_tensors="pt")

    def _vision_prefix(self, prefix_text: str, image: Any) -> tuple[str, int, Any]:
        wrapped = self._vision_text(prefix_text)
        inputs = self._vision_batch(wrapped, image)
        return wrapped, int(inputs["input_ids"].shape[1]), inputs

    def count_input_tokens(self, prefix_text: str, image: Any = None) -> int:
        """Number of input tokens the model sees for `prefix_text`.

        With an image this includes the expanded image placeholder tokens, so the
        reported usage matches what the vision model actually consumes.
        """
        if image is None or self.processor is None or not self.multimodal:
            return len(self.tokenizer(prefix_text)["input_ids"])
        _, prefix_len, _ = self._vision_prefix(prefix_text, self._as_image(image))
        return prefix_len

    def _score_options_vision(
        self,
        prefix_text: str,
        options: list[str],
        add_leading_space: bool,
        use_kv_cache: bool,
        image: Any,
    ) -> list[SequenceScore]:
        if self.processor is None or not self.multimodal:
            raise RuntimeError(
                f"{self.model_id!r} is not a multimodal model; image classification "
                "needs a vision-language checkpoint (e.g. Qwen/Qwen3-VL-4B-Instruct)."
            )
        image = self._as_image(image)

        if use_kv_cache and self._vision_kv_cache_ok is not False:
            try:
                result = self._vision_cached(prefix_text, options, add_leading_space, image)
            except Exception as exc:  # noqa: BLE001 - any failure => fall back
                self._vision_kv_cache_ok = False
                logger.warning(
                    "KV cache unavailable for vision with %s (%s: %s); falling back "
                    "to exact scoring.",
                    type(self._model).__name__,
                    type(exc).__name__,
                    exc,
                )
            else:
                if self._vision_kv_cache_ok is None:
                    # First vision call this run: confirm the cached path agrees
                    # with exact scoring before trusting it for the rest. A model
                    # whose multimodal positions we mishandle (e.g. an M-RoPE
                    # variant without `compute_3d_position_ids`) shows up here as
                    # a large gap instead of silently wrong logprobs.
                    exact = self._vision_exact(
                        prefix_text, options, add_leading_space, image
                    )
                    delta = self._vision_cache_delta(result, exact)
                    if delta > _VISION_CACHE_TOLERANCE_NATS:
                        self._vision_kv_cache_ok = False
                        logger.warning(
                            "Vision KV cache disagreed with exact scoring for %s "
                            "(max Δ%.3f nats > %.1f); using the exact path. This "
                            "usually means the model's multimodal position handling "
                            "is not supported.",
                            type(self._model).__name__,
                            delta,
                            _VISION_CACHE_TOLERANCE_NATS,
                        )
                        return exact
                self._vision_kv_cache_ok = True
                return result
        return self._vision_exact(prefix_text, options, add_leading_space, image)

    @staticmethod
    def _vision_cache_delta(
        cached: list[SequenceScore], exact: list[SequenceScore]
    ) -> float:
        """Largest per-token logprob gap between the cached and exact paths.

        Returns ``inf`` when the two disagree on tokenization, so the caller
        falls back to the exact path.
        """
        if len(cached) != len(exact):
            return float("inf")
        worst = 0.0
        for c, e in zip(cached, exact):
            if [t.token_id for t in c.tokens] != [t.token_id for t in e.tokens]:
                return float("inf")
            for ct, et in zip(c.tokens, e.tokens):
                worst = max(worst, abs(ct.logprob - et.logprob))
            worst = max(worst, abs(c.eos_logprob - e.eos_logprob))
        return worst

    def _vision_exact(
        self, prefix_text: str, options: list[str], add_leading_space: bool, image: Any
    ) -> list[SequenceScore]:
        torch = self.torch
        model = self._model
        device = next(model.parameters()).device
        wrapped, prefix_len, _ = self._vision_prefix(prefix_text, image)

        results: list[SequenceScore] = []
        for option in options:
            continuation = (" " if add_leading_space else "") + option
            inputs = self._vision_batch(wrapped + continuation, image)
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            full_ids = inputs["input_ids"][0].tolist()
            cont_ids = full_ids[prefix_len:]

            with torch.inference_mode():
                logits = model(**inputs).logits[0]
                logprobs = torch.log_softmax(logits.float(), dim=-1)

            tokens = [
                TokenScore(
                    token=self.tokenizer.decode([tid]),
                    token_id=int(tid),
                    logprob=logprobs[prefix_len + i - 1, tid].item(),
                )
                for i, tid in enumerate(cont_ids)
            ]
            eos_logprob = logprobs[len(full_ids) - 1, self.eos_token_id].item()
            total = sum(t.logprob for t in tokens) + eos_logprob
            results.append(
                SequenceScore(
                    option=option,
                    continuation=continuation,
                    tokens=tokens,
                    eos_token=self.tokenizer.decode([self.eos_token_id]),
                    eos_logprob=eos_logprob,
                    total_logprob=total,
                )
            )
        return results

    def _vision_cached(
        self, prefix_text: str, options: list[str], add_leading_space: bool, image: Any
    ) -> list[SequenceScore]:
        """Prefill the image + prompt once, then reuse its KV cache per option."""
        torch = self.torch
        model = self._model
        device = next(model.parameters()).device
        wrapped, prefix_len, prefix_inputs = self._vision_prefix(prefix_text, image)
        prefix_inputs = {
            k: (v.to(device) if hasattr(v, "to") else v) for k, v in prefix_inputs.items()
        }
        # The option continuation lives entirely after the shared prompt, so the
        # tokenizer alone yields the same ids as a full processor pass -- no need
        # to re-decode/resize the image for every option (the model's cached
        # forward doesn't need pixel_values again). The image placeholder expands
        # to `prefix_len`, so the text prefix is sliced by its own length.
        prefix_ids = self.tokenizer(wrapped)["input_ids"]

        with torch.inference_mode():
            prefill = model(**prefix_inputs, use_cache=True)
            prefix_logprobs = torch.log_softmax(
                prefill.logits[0, -1:, :].float(), dim=-1
            )[0]
            cache = prefill.past_key_values
            # Drop the full prefill output (logits + ModelOutput wrapper) before
            # the per-option loop. The full vocab logits for the whole prefix
            # otherwise stays live for every option.
            del prefill
        if cache is None or not hasattr(cache, "crop"):
            raise RuntimeError(
                f"{type(model).__name__} did not return a reusable KV cache"
            )

        results: list[SequenceScore] = []
        for option in options:
            continuation = (" " if add_leading_space else "") + option
            full_ids = self.tokenizer(wrapped + continuation)["input_ids"]
            if full_ids[: len(prefix_ids)] != prefix_ids:
                # Unusual processor/tokenizer mismatch: re-encode with the image
                # so the slice stays aligned with the expanded prefix.
                full_ids = self._vision_batch(wrapped + continuation, image)[
                    "input_ids"
                ][0].tolist()
                cont_ids = full_ids[prefix_len:]
            else:
                cont_ids = full_ids[len(prefix_ids):]
            count = len(cont_ids)

            with torch.inference_mode():
                if count == 0:
                    logprob_values = [prefix_logprobs[self.eos_token_id].item()]
                else:
                    attention_mask = torch.ones(
                        (1, prefix_len + count), device=device, dtype=torch.long
                    )
                    cache_position = torch.arange(
                        prefix_len, prefix_len + count, device=device
                    )
                    # Multimodal-RoPE models derive positions from the attention
                    # mask length, which is the full prefix here; pass the new
                    # tokens' positions explicitly (shifted by the image's rope
                    # delta), like GenerationMixin does during decoding.
                    position_ids = cache_position.unsqueeze(0)
                    if self._uses_mrope:
                        rope_deltas = getattr(self._base, "rope_deltas", None)
                        if rope_deltas is not None:
                            position_ids = position_ids + rope_deltas.to(device)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                    out = model(
                        input_ids=torch.tensor([cont_ids], device=device),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                    logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
                    logprob_values = [prefix_logprobs[cont_ids[0]].item()]
                    logprob_values.extend(
                        logprobs[j, cont_ids[j + 1]].item() for j in range(count - 1)
                    )
                    logprob_values.append(logprobs[count - 1, self.eos_token_id].item())

            if count:
                cache.crop(-count)

            tokens = [
                TokenScore(
                    token=self.tokenizer.decode([tid]),
                    token_id=int(tid),
                    logprob=logprob_values[i],
                )
                for i, tid in enumerate(cont_ids)
            ]
            eos_logprob = logprob_values[-1]
            total = sum(t.logprob for t in tokens) + eos_logprob
            results.append(
                SequenceScore(
                    option=option,
                    continuation=continuation,
                    tokens=tokens,
                    eos_token=self.tokenizer.decode([self.eos_token_id]),
                    eos_logprob=eos_logprob,
                    total_logprob=total,
                )
            )
        return results


_SCORERS: dict[tuple[str, str | None, str, str, bool], Scorer] = {}
_LOCK = threading.Lock()
_REGISTRY_LOCK = threading.Lock()


def get_scorer(
    model_id: str = DEFAULT_MODEL_ID,
    save_to: str | None = None,
    device: str = "cpu",
    gpu: str | None = None,
    quantize: str = "auto",
) -> Scorer:
    """Load (once) and cache a scorer per (model, save dir, resolved device, precision)."""
    resolved = resolve_device(device, gpu)
    dtype_name, quantized = resolve_precision(resolved, quantize)
    key = (model_id, str(save_to) if save_to else None, resolved, dtype_name, quantized)
    # `_LOCK` serializes (slow) construction so a model loads once.
    # `_REGISTRY_LOCK` guards only the dict structure, so is_loaded/loaded_scorer
    # stay responsive (and race-free) while a model is loading.
    with _LOCK:
        with _REGISTRY_LOCK:
            scorer = _SCORERS.get(key)
        if scorer is None:
            scorer = Scorer(model_id, save_to=save_to, device=resolved, quantize=quantize)
            with _REGISTRY_LOCK:
                _SCORERS[key] = scorer
        return scorer


def is_loaded(model_id: str | None = None) -> bool:
    with _REGISTRY_LOCK:
        if model_id is None:
            return bool(_SCORERS)
        return any(key[0] == model_id for key in _SCORERS)


def loaded_scorer(model_id: str | None = None) -> Scorer | None:
    """Return the cached scorer for `model_id` (any if None), or None if not loaded.

    Lets callers read model capabilities (e.g. `multimodal`) without loading it.
    """
    with _REGISTRY_LOCK:
        if model_id is None:
            return next(iter(_SCORERS.values()), None)
        return next((s for key, s in _SCORERS.items() if key[0] == model_id), None)
