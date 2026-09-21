from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_MODEL_ID, SUPPORTED_GPUS

logger = logging.getLogger(__name__)

_MIN_AVAILABLE_GB = 9.0

# Files we need for text scoring (the big safetensors + configs/tokenizer).
_MODEL_PATTERNS = ["*.json", "*.safetensors", "*.jinja"]


def resolve_device(device: str = "cpu", gpu: str | None = None) -> str:
    """Map the config device to a torch device string, validating the GPU.

    `device` is "cpu" (default) or "gpu"/"cuda". When GPU mode is requested,
    `gpu` names one of the supported GPUs in `config.SUPPORTED_GPUS`.
    """
    device = (device or "cpu").strip().lower()
    if device == "cpu":
        return "cpu"
    if device not in ("gpu", "cuda"):
        raise ValueError(f"Unknown device {device!r}; use 'cpu', 'gpu' or 'cuda'.")

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


class GemmaScorer:
    """Scores a continuation under Gemma using the full-vocabulary next-token
    distribution. Exact token probabilities (not limited to any top-k) and the
    true EOS probability are read from the logits.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        save_to: str | None = None,
        device: str = "cpu",
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
        from transformers import AutoConfig, AutoTokenizer

        if device.startswith("cuda"):
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 11.0:
                raise RuntimeError(
                    f"GPU has only {vram_gb:.1f} GB VRAM; the model needs about 11 GB."
                )

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
            model_cls = transformers.AutoModelForCausalLM

        model = model_cls.from_pretrained(
            source,
            dtype=torch.bfloat16,
            device_map=device,
            local_files_only=from_disk,
        )

        # Text-only scoring: pull the language decoder + output head out of the
        # (possibly multimodal) model and drop the unused towers to save memory.
        base = getattr(model, "model", model)
        self.language_model = getattr(base, "language_model", base)
        self.lm_head = model.get_output_embeddings() or getattr(model, "lm_head", None)
        if self.lm_head is None:
            raise RuntimeError(f"Could not find an output LM head on {type(model).__name__}.")

        for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "visual"):
            if getattr(base, attr, None) is not None:
                setattr(base, attr, None)
        for attr in ("mtp",):
            if getattr(model, attr, None) is not None:
                setattr(model, attr, None)
        model.eval()
        self._model = model

        eos = self.tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        self.eos_token_id = int(eos)

        # None = untested, True = cached path works, False = fall back to exact.
        self._kv_cache_ok: bool | None = None

    def score_options(
        self,
        prefix_text: str,
        options: list[str],
        *,
        add_leading_space: bool = True,
        use_kv_cache: bool = True,
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
        """
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


_SCORERS: dict[tuple[str, str | None, str], GemmaScorer] = {}
_LOCK = threading.Lock()


def get_scorer(
    model_id: str = DEFAULT_MODEL_ID,
    save_to: str | None = None,
    device: str = "cpu",
    gpu: str | None = None,
) -> GemmaScorer:
    """Load (once) and cache a scorer per (model, save dir, resolved device)."""
    resolved = resolve_device(device, gpu)
    key = (model_id, str(save_to) if save_to else None, resolved)
    with _LOCK:
        if key not in _SCORERS:
            _SCORERS[key] = GemmaScorer(model_id, save_to=save_to, device=resolved)
        return _SCORERS[key]


def is_loaded(model_id: str | None = None) -> bool:
    if model_id is None:
        return bool(_SCORERS)
    return any(key[0] == model_id for key in _SCORERS)
