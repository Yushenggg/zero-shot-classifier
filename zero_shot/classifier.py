from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .scorer import DEFAULT_MODEL_ID, SequenceScore, TokenScore, get_scorer

SUPPORTED_TYPES = ("choice", "noul", "score")


@dataclass
class OptionScore:
    option: str
    continuation: str = ""
    tokens: list[TokenScore] = field(default_factory=list)
    eos_logprob: float | None = None
    logprob: float | None = None
    calibrated_logprob: float | None = None
    probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out = {
            "option": self.option,
            "continuation": self.continuation,
            "tokens": [t.to_dict() for t in self.tokens],
            "eos_logprob": self.eos_logprob,
            "logprob": self.logprob,
            "probability": self.probability,
        }
        if self.calibrated_logprob is not None:
            out["calibrated_logprob"] = self.calibrated_logprob
        return out


@dataclass
class Classification:
    """One answered question.

    Mirrors the TypeSafe answer shapes: `choice` for choice questions, `noul` for
    yes/no, and `score` (+ `legend`) for ordered scales. `scores` always carries
    the per-option detail; `confidence` is only set for choice/score.
    """

    name: str
    type: str
    prompt: str
    scores: list[OptionScore]
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    legend: dict[str, str] | None = None
    confidence: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "prompt": self.prompt,
            "probabilities": {s.option: s.probability for s in self.scores},
            "scores": [s.to_dict() for s in self.scores],
        }
        if self.type == "choice":
            out["choice"] = self.choice
        elif self.type == "noul":
            out["noul"] = self.noul
        elif self.type == "score":
            out["score"] = self.score
            out["legend"] = self.legend
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out


def _render(value: Any) -> str:
    """Render an instruction/criterion/state value (string, object, array, or null)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _build_question(name: str, spec: dict[str, Any], state: Any) -> tuple[str, list[tuple[str, str]]]:
    """Return (prompt, options) where options is a list of (option_id, continuation)."""
    qtype = spec.get("type")
    if qtype not in SUPPORTED_TYPES:
        raise ValueError(f"Question '{name}' has unsupported type {qtype!r}; use one of {SUPPORTED_TYPES}")

    header = (
        "You are a precise zero-shot classifier. Use only the context and the "
        "instructions.\n\n"
        f"# CONTEXT\n{_render(state)}\n\n"
        f"# TASK\n{_render(spec.get('instructions', ''))}\n\n"
    )

    if qtype == "choice":
        criteria = spec.get("criteria") or {}
        if not criteria:
            raise ValueError(f"Question '{name}' (choice) needs a non-empty criteria map")
        criteria_lines = [
            f"{key}: {_render(desc)}" for key, desc in criteria.items() if _render(desc)
        ]
        criteria_block = (
            f"# CRITERIA\n" + "\n".join(criteria_lines) + "\n\n" if criteria_lines else ""
        )
        prompt = (
            header
            + criteria_block
            + "# ANSWER\nRespond with exactly one option key and nothing else.\n"
            + f'The best option for "{name}" is:'
        )
        options = [(key, key) for key in criteria]

    elif qtype == "noul":
        prompt = (
            header
            + "# ANSWER\nRespond with exactly one word, yes or no, and nothing else.\n"
            + f'The answer to "{name}" is:'
        )
        options = [("true", "yes"), ("false", "no")]

    else:  # score
        levels = spec.get("criteria")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(f"Question '{name}' (score) needs 2-10 ordered criteria levels")
        if len(levels) > 10:
            raise ValueError(f"Question '{name}' (score) has {len(levels)} levels; the max is 10")
        scale = "\n".join(f"{i}: {_render(level)}" for i, level in enumerate(levels))
        prompt = (
            header
            + f"# RATING SCALE (0 = low, {len(levels) - 1} = high)\n{scale}\n\n"
            + "# ANSWER\nRespond with exactly one number and nothing else.\n"
            + "The rating is:"
        )
        options = [(str(i), str(i)) for i in range(len(levels))]

    return prompt, options


def _effective_logprob(score: OptionScore) -> float | None:
    return score.calibrated_logprob if score.calibrated_logprob is not None else score.logprob


def _softmax(scores: list[OptionScore], temperature: float) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    known = [s for s in scores if _effective_logprob(s) is not None]
    if not known:
        return
    peak = max(_effective_logprob(s) for s in known)
    weights = [math.exp((_effective_logprob(s) - peak) / temperature) for s in known]
    total = sum(weights)
    for score, weight in zip(known, weights):
        score.probability = weight / total


def classify_one(
    name: str,
    spec: dict[str, Any],
    state: Any,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    save_to: str | None = None,
    device: str = "cpu",
    gpu: str | None = None,
    quantize: str = "auto",
    temperature: float = 1.0,
    use_kv_cache: bool = False,
    calibrate: bool = False,
    calibration_context: str = "N/A",
    image: Any = None,
) -> Classification:
    prompt, options = _build_question(name, spec, state)
    scorer = get_scorer(model_id, save_to=save_to, device=device, gpu=gpu, quantize=quantize)
    texts = [text for _, text in options]
    # Vision prompts end at the assistant generation prompt, so the option
    # follows immediately (no synthetic leading space); text prompts end with
    # "... is:" and need the space. Use the same form for the calibration pass
    # so its priors are subtracted from identically-tokenized options.
    add_leading_space = image is None
    sequence_scores: list[SequenceScore] = scorer.score_options(
        prompt, texts, use_kv_cache=use_kv_cache, image=image,
        add_leading_space=add_leading_space,
    )
    by_text = {s.option: s for s in sequence_scores}

    # Contextual calibration: subtract each option's content-free prior. For
    # image questions the image is content, so the null pass runs text-only
    # (calibrating against the same image would cancel the image's signal).
    null_by_text: dict[str, float] = {}
    if calibrate:
        null_prompt, _ = _build_question(name, spec, calibration_context)
        null_scores = scorer.score_options(
            null_prompt, texts, use_kv_cache=use_kv_cache, image=None,
            add_leading_space=add_leading_space,
        )
        null_by_text = {s.option: s.total_logprob for s in null_scores}

    scores: list[OptionScore] = []
    for option_id, text in options:
        seq = by_text.get(text)
        if seq is None:
            scores.append(OptionScore(option=option_id, continuation=text))
        else:
            calibrated = seq.total_logprob - null_by_text[text] if text in null_by_text else None
            scores.append(
                OptionScore(
                    option=option_id,
                    continuation=text,
                    tokens=seq.tokens,
                    eos_logprob=seq.eos_logprob,
                    logprob=seq.total_logprob,
                    calibrated_logprob=calibrated,
                )
            )
    _softmax(scores, temperature)

    input_tokens = scorer.count_input_tokens(prompt, image)
    ranked = [s for s in scores if s.logprob is not None]
    winner = max(ranked, key=lambda s: s.probability) if ranked else None
    output_tokens = (len(winner.tokens) + 1) if winner else 0

    qtype = spec["type"]
    if qtype == "choice":
        choice = winner.option if winner else None
        confidence = sum(s.probability**2 for s in scores)
        return Classification(
            name, qtype, prompt, scores, choice=choice, confidence=confidence,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )

    if qtype == "noul":
        yes = next((s.probability for s in scores if s.option == "true"), 0.0)
        return Classification(
            name, qtype, prompt, scores, noul=yes,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )

    # score
    levels = spec["criteria"]
    value = sum(int(s.option) * s.probability for s in scores)
    legend = {str(i): _render(levels[i]) for i in range(len(levels))}
    confidence = sum(s.probability**2 for s in scores)
    return Classification(
        name, qtype, prompt, scores, score=value, legend=legend, confidence=confidence,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )


def classify(
    question: dict[str, Any],
    state: Any,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    save_to: str | None = None,
    device: str = "cpu",
    gpu: str | None = None,
    quantize: str = "auto",
    temperature: float = 1.0,
    use_kv_cache: bool = False,
    calibrate: bool = False,
    calibration_context: str = "N/A",
    image: Any = None,
) -> list[Classification]:
    """Evaluate `state` against a map of typed questions.

    `question` maps a name to a spec, where `type` is one of:

        choice: {"type": "choice", "instructions": "...",
                 "criteria": {"option1": "desc", ...}}          # max 255 options
        noul:   {"type": "noul", "instructions": "...",
                 "criteria": {"true": "...", "false": "..."}}     # criteria optional
        score:  {"type": "score", "instructions": "...",
                 "criteria": ["Calm", "Frustrated", "Very angry"]} # 2-10 ordered levels

    Candidate keys are scored directly. For `choice`, the prompt includes a
    `# CRITERIA` block listing each key and its description so the model sees
    what each key means, but only the key itself is used as the scored
    continuation. With `calibrate=True`, each option's content-free prior (from
    `calibration_context`) is subtracted before the softmax, removing
    surface-form/option bias.
    """
    if not isinstance(question, dict) or not question:
        raise ValueError("question must be a non-empty JSON object")
    return [
        classify_one(
            name,
            spec,
            state,
            model_id=model_id,
            save_to=save_to,
            device=device,
            gpu=gpu,
            quantize=quantize,
            temperature=temperature,
            use_kv_cache=use_kv_cache,
            calibrate=calibrate,
            calibration_context=calibration_context,
            image=image,
        )
        for name, spec in question.items()
    ]
