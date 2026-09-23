"""Core classifier logic: config, scoring, and classification.

This package holds the model-independent, interface-independent pieces. Nothing
here imports an interface (CLI, server); interfaces depend on core, never the
other way around.
"""

from .classifier import SUPPORTED_TYPES, Classification, OptionScore, classify, classify_one
from .config import Config, load_config
from .scorer import (
    Scorer,
    SequenceScore,
    TokenScore,
    get_scorer,
    is_loaded,
    loaded_scorer,
)

__all__ = [
    "SUPPORTED_TYPES",
    "Classification",
    "OptionScore",
    "classify",
    "classify_one",
    "Config",
    "load_config",
    "Scorer",
    "SequenceScore",
    "TokenScore",
    "get_scorer",
    "is_loaded",
    "loaded_scorer",
]
