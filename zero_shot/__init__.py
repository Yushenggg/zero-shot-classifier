"""zero-shot: local zero-shot classifier over a model's next-token distribution.

The public API lives in :mod:`zero_shot.core`; user-facing entry points live in
:mod:`zero_shot.interfaces`. This module re-exports the core API for convenience.
"""

from .core import (
    SUPPORTED_TYPES,
    Classification,
    Config,
    OptionScore,
    Scorer,
    SequenceScore,
    TokenScore,
    classify,
    classify_one,
    get_scorer,
    is_loaded,
    load_config,
    loaded_scorer,
    unload,
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
    "unload",
]
