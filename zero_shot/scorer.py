from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_MODEL_ID

_MIN_AVAILABLE_GB = 9.0

# Files we need for text scoring (the big safetensors + configs/tokenizer).
_MODEL_PATTERNS = ["*.json", "*.safetensors", "*.jinja"]


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
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

        source, from_disk = resolve_model_dir(model_id, save_to)

        self.torch = torch
        self.model_id = model_id
        self.source = source
        self.from_disk = from_disk
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=from_disk)

        model = Gemma4ForConditionalGeneration.from_pretrained(
            source,
            dtype=torch.bfloat16,
            device_map=device,
            local_files_only=from_disk,
        )
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

        # The vision/audio towers are unused for text scoring; drop to save memory.
        for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            if getattr(model.model, attr, None) is not None:
                setattr(model.model, attr, None)
        model.eval()
        self._model = model

        eos = self.tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        self.eos_token_id = int(eos)

    def score_options(
        self,
        prefix_text: str,
        options: list[str],
        *,
        add_leading_space: bool = True,
    ) -> list[SequenceScore]:
        """Return exact sequence log-probabilities (tokens + EOS) for each option."""
        torch = self.torch
        tok = self.tokenizer
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
                hidden = self.language_model(torch.tensor([seq])).last_hidden_state
                hidden = hidden[:, positions, :]
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


_SCORERS: dict[tuple[str, str | None, str], GemmaScorer] = {}
_LOCK = threading.Lock()


def get_scorer(
    model_id: str = DEFAULT_MODEL_ID,
    save_to: str | None = None,
    device: str = "cpu",
) -> GemmaScorer:
    """Load (once) and cache a scorer per (model, save dir, device)."""
    key = (model_id, str(save_to) if save_to else None, device)
    with _LOCK:
        if key not in _SCORERS:
            _SCORERS[key] = GemmaScorer(model_id, save_to=save_to, device=device)
        return _SCORERS[key]


def is_loaded(model_id: str | None = None) -> bool:
    if model_id is None:
        return bool(_SCORERS)
    return any(key[0] == model_id for key in _SCORERS)
