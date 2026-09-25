"""Metric computation for the benchmark harness.

Pure functions over aligned ``(gold, predicted)`` lists -- no model or config
imports -- so the scoring math is unit-testable without loading any weights.

Ground-truth labels come from a CSV and rarely match the model's option keys
byte-for-byte (HF ClassLabel names, integer class ids, ``yes``/``True``/``1``).
``canonical_choice``/``canonical_noul`` fold those spellings onto the keys the
question actually offers, optionally through a user-supplied ``--label-map``.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "accuracy",
    "canonical_choice",
    "canonical_noul",
    "confusion_matrix",
    "macro_f1",
    "normalize",
    "score_metrics",
]

_TRUE_LABELS = frozenset({"true", "yes", "y", "1", "positive", "pos"})
_FALSE_LABELS = frozenset({"false", "no", "n", "0", "negative", "neg"})


def normalize(value: Any) -> str:
    """Stringify and strip a CSV/label cell; ``None`` becomes the empty string."""
    return "" if value is None else str(value).strip()


def _apply_aliases(text: str, aliases: dict[str, str] | None) -> str:
    return aliases[text] if aliases and text in aliases else text


def canonical_choice(
    value: Any, labels: list[str], aliases: dict[str, str] | None = None
) -> str | None:
    """Map a raw label onto one of ``labels`` (the question's option keys).

    Matching is exact first, then case-insensitive. ``aliases`` (from
    ``--label-map``) rewrites the raw value before matching, which is how
    integer class ids or letters reach a human-readable criteria key. Returns
    ``None`` when nothing matches.
    """
    text = _apply_aliases(normalize(value), aliases)
    if not text:
        return None
    for label in labels:
        if text == label:
            return label
    lowered = text.lower()
    for label in labels:
        if lowered == label.lower():
            return label
    return None


def canonical_noul(value: Any, aliases: dict[str, str] | None = None) -> str | None:
    """Map a raw yes/no label onto ``"true"``/``"false"``; ``None`` if unknown."""
    text = _apply_aliases(normalize(value), aliases).lower()
    if text in _TRUE_LABELS:
        return "true"
    if text in _FALSE_LABELS:
        return "false"
    return None


def confusion_matrix(
    gold: list[str], pred: list[str | None], labels: list[str] | None = None
) -> dict[str, dict[str, int]]:
    """Counts of ``gold -> predicted``. Predicted values outside ``labels``
    (including ``None``) still appear as extra columns so nothing is hidden.
    """
    columns = list(labels) if labels is not None else []
    for pred_value in pred:
        key = pred_value if pred_value is not None else "(none)"
        if key not in columns:
            columns.append(key)
    rows = list(labels) if labels is not None else []
    for gold_value in gold:
        if gold_value not in rows:
            rows.append(gold_value)
    matrix: dict[str, dict[str, int]] = {row: dict.fromkeys(columns, 0) for row in rows}
    for gold_value, pred_value in zip(gold, pred, strict=True):
        key = pred_value if pred_value is not None else "(none)"
        matrix[gold_value][key] += 1
    return matrix


def accuracy(gold: list[str], pred: list[str | None]) -> float | None:
    """Fraction of examples whose prediction exactly matches the gold label."""
    if not gold:
        return None
    correct = sum(g == p for g, p in zip(gold, pred, strict=True))
    return correct / len(gold)


def macro_f1(
    gold: list[str], pred: list[str | None], labels: list[str]
) -> float | None:
    """Unweighted mean F1 across ``labels`` (sklearn's ``macro`` averaging)."""
    if not gold:
        return None
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall:
            scores.append(2 * precision * recall / (precision + recall))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def score_metrics(gold: list[float], pred: list[float]) -> dict[str, float | int | None]:
    """MAE/RMSE and rounded exact-match for an ordered ``score`` question."""
    n = len(gold)
    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "exact_match": None,
            "mean_prediction": None,
        }
    errors = [p - g for g, p in zip(gold, pred, strict=True)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    exact = sum(round(p) == round(g) for g, p in zip(gold, pred, strict=True)) / n
    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "exact_match": exact,
        "mean_prediction": sum(pred) / n,
    }
