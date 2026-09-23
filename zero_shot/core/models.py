"""Typed domain models: question specs, scores, and classifications.

These are pydantic models so malformed requests and configs fail loudly at the
boundary instead of somewhere deep in prompt construction. The ``to_dict``
methods preserve the exact JSON wire shape clients (and the web UI) consume;
pydantic is used for parsing/validation, not for serialization.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

QuestionType = Literal["choice", "noul", "score"]

# Tuple of the supported question type names (kept for backwards compatibility).
SUPPORTED_TYPES: tuple[str, ...] = get_args(QuestionType)


class BaseQuestion(BaseModel):
    """Fields shared by every question spec.

    ``instructions`` is intentionally ``Any``: it is rendered by
    :func:`zero_shot.core.classifier._render` and may be a string, an object, an
    array, or null.
    """

    instructions: Any = ""


class ChoiceQuestion(BaseQuestion):
    """Pick exactly one key from a criteria map; only the key is scored."""

    type: Literal["choice"]
    criteria: dict[str, Any]

    @field_validator("criteria")
    @classmethod
    def _criteria_non_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("choice questions need a non-empty criteria map")
        return value


class NoulQuestion(BaseQuestion):
    """Yes/no question. ``criteria`` is accepted but ignored."""

    type: Literal["noul"]
    criteria: Any = None


class ScoreQuestion(BaseQuestion):
    """Ordered scale; ``criteria`` is 2-10 levels, scored as their index."""

    type: Literal["score"]
    criteria: list[Any]

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, value: list[Any]) -> list[Any]:
        if not 2 <= len(value) <= 10:
            raise ValueError(
                f"score questions need 2-10 ordered criteria levels, got {len(value)}"
            )
        return value


QuestionSpec = Annotated[
    ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")
]

_QUESTION_ADAPTER: TypeAdapter[QuestionSpec] = TypeAdapter(QuestionSpec)
_QUESTIONS_ADAPTER: TypeAdapter[dict[str, QuestionSpec]] = TypeAdapter(
    dict[str, QuestionSpec]
)


def _format_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}" if location else first["msg"]


def parse_question(spec: Any) -> QuestionSpec:
    """Validate a single question spec (a dict, or pass an already-typed spec)."""
    if isinstance(spec, (ChoiceQuestion, NoulQuestion, ScoreQuestion)):
        return spec
    try:
        return _QUESTION_ADAPTER.validate_python(spec)
    except ValidationError as exc:
        raise ValueError(f"invalid question: {_format_error(exc)}") from exc


def parse_questions(question: Any) -> dict[str, QuestionSpec]:
    """Validate a ``{name: spec}`` mapping, raising ``ValueError`` on any problem."""
    if not isinstance(question, dict) or not question:
        raise ValueError("question must be a non-empty JSON object")
    try:
        return _QUESTIONS_ADAPTER.validate_python(question)
    except ValidationError as exc:
        raise ValueError(f"invalid questions: {_format_error(exc)}") from exc


class TokenScore(BaseModel):
    """One continuation token and its log-probability under the model."""

    token: str
    token_id: int
    logprob: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "token_id": self.token_id,
            "logprob": self.logprob,
        }


class SequenceScore(BaseModel):
    """The exact score of one option sequence (tokens + EOS)."""

    option: str
    continuation: str
    tokens: list[TokenScore] = Field(default_factory=list)
    eos_token: str = ""
    eos_logprob: float = 0.0
    total_logprob: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "continuation": self.continuation,
            "tokens": [t.to_dict() for t in self.tokens],
            "eos_token": self.eos_token,
            "eos_logprob": self.eos_logprob,
            "total_logprob": self.total_logprob,
        }


class OptionScore(BaseModel):
    """One option of a question after softmax (plus its raw/calibrated logprob)."""

    option: str
    continuation: str = ""
    tokens: list[TokenScore] = Field(default_factory=list)
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


class Classification(BaseModel):
    """One answered question.

    Mirrors the TypeSafe answer shapes: ``choice`` for choice questions, ``noul``
    for yes/no, and ``score`` (+ ``legend``) for ordered scales. ``scores`` always
    carries the per-option detail; ``confidence`` is only set for choice/score.
    """

    name: str
    type: QuestionType
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
