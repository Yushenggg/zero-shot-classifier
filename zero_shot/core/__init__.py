"""Core classifier logic: config, scoring, and classification.

This package holds the model-independent, interface-independent pieces. Nothing
here imports an interface (CLI, server); interfaces depend on core, never the
other way around.
"""

from .classifier import classify, classify_one
from .config import Config, load_config
from .models import (
    SUPPORTED_TYPES,
    ChoiceQuestion,
    Classification,
    NoulQuestion,
    OptionScore,
    QuestionSpec,
    ScoreQuestion,
    SequenceScore,
    TokenScore,
    parse_question,
    parse_questions,
)
from .scorer import Scorer, get_scorer, is_loaded, loaded_scorer, unload

__all__ = [
    "SUPPORTED_TYPES",
    "ChoiceQuestion",
    "Classification",
    "Config",
    "NoulQuestion",
    "OptionScore",
    "QuestionSpec",
    "ScoreQuestion",
    "Scorer",
    "SequenceScore",
    "TokenScore",
    "classify",
    "classify_one",
    "get_scorer",
    "is_loaded",
    "load_config",
    "loaded_scorer",
    "parse_question",
    "parse_questions",
    "unload",
]
