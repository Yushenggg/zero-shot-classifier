from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .scorer import DEFAULT_MODEL_ID, SequenceScore, TokenScore, get_scorer


@dataclass
class OptionScore:
    option: str
    tokens: list[TokenScore] = field(default_factory=list)
    eos_logprob: float | None = None
    logprob: float | None = None
    probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "tokens": [t.to_dict() for t in self.tokens],
            "eos_logprob": self.eos_logprob,
            "logprob": self.logprob,
            "probability": self.probability,
        }


@dataclass
class Classification:
    name: str
    choice: str
    prompt: str
    scores: list[OptionScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "choice": self.choice,
            "prompt": self.prompt,
            "scores": [s.to_dict() for s in self.scores],
        }


def _build_prompt(name: str, spec: dict[str, Any], state: Any) -> str:
    criteria: dict[str, str] = spec.get("criteria", {})
    instructions = spec.get("instructions", "")
    choices = "\n".join(f"- {key}: {desc}" for key, desc in criteria.items())
    context = json.dumps(state, indent=2, ensure_ascii=False, default=str)
    return (
        "You are a precise zero-shot classifier. Use only the CONTEXT and the "
        "INSTRUCTIONS to decide.\n\n"
        f"# CONTEXT (state)\n{context}\n\n"
        f"# TASK: {name}\n{instructions}\n\n"
        f"# CHOICES (choose exactly one key)\n{choices}\n\n"
        "# ANSWER\n"
        f"The best choice for {name} is:"
    )


def score_options(
    criteria: dict[str, str],
    sequence_scores: list[SequenceScore],
    *,
    temperature: float = 1.0,
) -> list[OptionScore]:
    """Turn sequence log-probabilities into a distribution over the options.

    Each option's score is the exact log P(option) = sum of its token log-probs
    plus the EOS log-prob (the probability that the answer ends there). Those
    scores are then softmaxed with the given temperature.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    by_option = {s.option: s for s in sequence_scores}
    scores: list[OptionScore] = []
    for option in criteria:
        seq = by_option.get(option)
        if seq is None:
            scores.append(OptionScore(option=option))
        else:
            scores.append(
                OptionScore(
                    option=option,
                    tokens=seq.tokens,
                    eos_logprob=seq.eos_logprob,
                    logprob=seq.total_logprob,
                )
            )

    known = [s for s in scores if s.logprob is not None]
    if known:
        peak = max(s.logprob for s in known)
        weights = [math.exp((s.logprob - peak) / temperature) for s in known]
        total = sum(weights)
        for score, weight in zip(known, weights):
            score.probability = weight / total
    return scores


def classify_one(
    name: str,
    spec: dict[str, Any],
    state: Any,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    save_to: str | None = None,
    device: str = "cpu",
    gpu: str | None = None,
    temperature: float = 1.0,
    use_kv_cache: bool = False,
) -> Classification:
    if spec.get("type") != "choice":
        raise ValueError(f"Question '{name}' must have type 'choice', got {spec.get('type')!r}")
    criteria: dict[str, str] = spec.get("criteria", {})
    if not criteria:
        raise ValueError(f"Question '{name}' has no criteria")

    prompt = _build_prompt(name, spec, state)
    scorer = get_scorer(model_id, save_to=save_to, device=device, gpu=gpu)
    sequence_scores = scorer.score_options(
        prompt, list(criteria.keys()), use_kv_cache=use_kv_cache
    )
    scores = score_options(criteria, sequence_scores, temperature=temperature)

    ranked = [s for s in scores if s.logprob is not None]
    choice = max(ranked, key=lambda s: s.probability).option if ranked else ""

    return Classification(name=name, choice=choice, prompt=prompt, scores=scores)


def classify(
    question: dict[str, Any],
    state: Any,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    save_to: str | None = None,
    device: str = "cpu",
    gpu: str | None = None,
    temperature: float = 1.0,
    use_kv_cache: bool = False,
) -> list[Classification]:
    """Classify `state` against every choice question in `question`.

    `question` maps a name to a spec:
        {"any_name": {"type": "choice", "instructions": "...",
                      "criteria": {"option1": "desc", ...}}}
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
            temperature=temperature,
            use_kv_cache=use_kv_cache,
        )
        for name, spec in question.items()
    ]
