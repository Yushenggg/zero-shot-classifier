from __future__ import annotations

import json
import math
from typing import Any

from .models import (
    SUPPORTED_TYPES,
    ChoiceQuestion,
    Classification,
    NoulQuestion,
    OptionScore,
    QuestionSpec,
    ScoreQuestion,
    SequenceScore,
    parse_question,
    parse_questions,
)
from .scorer import DEFAULT_MODEL_ID, get_scorer

__all__ = [
    "SUPPORTED_TYPES",
    "Classification",
    "OptionScore",
    "classify",
    "classify_one",
]


def _render(value: Any) -> str:
    """Render an instruction/criterion/state value (string, object, array, or null)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _build_question(
    name: str, spec: QuestionSpec, state: Any
) -> tuple[str, list[tuple[str, str]]]:
    """Return (prompt, options) where options is a list of (option_id, continuation)."""
    header = (
        "You are a precise zero-shot classifier. Use only the context and the "
        "instructions.\n\n"
        f"# CONTEXT\n{_render(state)}\n\n"
        f"# TASK\n{_render(spec.instructions)}\n\n"
    )

    if isinstance(spec, ChoiceQuestion):
        criteria_lines = [
            f"{key}: {_render(desc)}" for key, desc in spec.criteria.items() if _render(desc)
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
        options = [(key, key) for key in spec.criteria]

    elif isinstance(spec, NoulQuestion):
        prompt = (
            header
            + "# ANSWER\nRespond with exactly one word, yes or no, and nothing else.\n"
            + f'The answer to "{name}" is:'
        )
        options = [("true", "yes"), ("false", "no")]

    else:  # ScoreQuestion
        levels = spec.criteria
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
    spec: QuestionSpec | dict[str, Any],
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
    question = parse_question(spec)
    prompt, options = _build_question(name, question, state)
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
    # image questions the null pass stays text-only -- the image *is* the signal,
    # so calibrating against it would cancel it -- but it is rendered through the
    # same chat template as the vision main pass so only the context differs.
    null_by_text: dict[str, float] = {}
    if calibrate:
        null_prompt, _ = _build_question(name, question, calibration_context)
        null_scores = scorer.score_options(
            null_prompt, texts, use_kv_cache=use_kv_cache, image=None,
            add_leading_space=add_leading_space, chat_template=image is not None,
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

    # Vision scoring allocates a fresh KV cache (and large prefill outputs) per
    # request. PyTorch's CUDA caching allocator doesn't return freed blocks to
    # the GPU on its own, so nvidia-smi keeps showing the high-water mark across
    # requests and the next image can OOM. empty_cache() forces the return.
    if image is not None and scorer.device == "cuda":
        scorer.torch.cuda.empty_cache()

    input_tokens = scorer.count_input_tokens(prompt, image)
    ranked = [s for s in scores if s.logprob is not None]
    winner = max(ranked, key=lambda s: s.probability) if ranked else None
    output_tokens = (len(winner.tokens) + 1) if winner else 0

    qtype = question.type
    if qtype == "choice":
        choice = winner.option if winner else None
        confidence = sum(s.probability**2 for s in scores)
        return Classification(
            name=name, type=qtype, prompt=prompt, scores=scores, choice=choice,
            confidence=confidence, input_tokens=input_tokens, output_tokens=output_tokens,
        )

    if qtype == "noul":
        yes = next((s.probability for s in scores if s.option == "true"), 0.0)
        return Classification(
            name=name, type=qtype, prompt=prompt, scores=scores, noul=yes,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )

    # score
    levels = question.criteria
    value = sum(int(s.option) * s.probability for s in scores)
    legend = {str(i): _render(levels[i]) for i in range(len(levels))}
    confidence = sum(s.probability**2 for s in scores)
    return Classification(
        name=name, type=qtype, prompt=prompt, scores=scores, score=value, legend=legend,
        confidence=confidence, input_tokens=input_tokens, output_tokens=output_tokens,
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
    specs = parse_questions(question)
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
        for name, spec in specs.items()
    ]
