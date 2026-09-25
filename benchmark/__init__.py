"""Benchmark harness for the zero-shot classifier.

``zero-shot-bench`` scores a labelled CSV against a question set with a single
model load/unload and writes per-row predictions plus aggregate metrics.
See :mod:`benchmark.run` for the CLI and :mod:`benchmark.metrics` for the math.
"""

from .data import Example, load_examples, read_header, resolve_label_columns
from .metrics import (
    accuracy,
    canonical_choice,
    canonical_noul,
    confusion_matrix,
    macro_f1,
    normalize,
    score_metrics,
)
from .run import build_summary, main, metric_summary, question_keys

__all__ = [
    "Example",
    "accuracy",
    "build_summary",
    "canonical_choice",
    "canonical_noul",
    "confusion_matrix",
    "load_examples",
    "macro_f1",
    "main",
    "metric_summary",
    "normalize",
    "question_keys",
    "read_header",
    "resolve_label_columns",
    "score_metrics",
]
