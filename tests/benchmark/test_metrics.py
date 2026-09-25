"""Regression tests for benchmark metric helpers."""

from __future__ import annotations

import pytest
from benchmark import metrics


def test_normalize_none_and_whitespace():
    assert metrics.normalize(None) == ""
    assert metrics.normalize("  red ") == "red"
    assert metrics.normalize(2) == "2"


def test_canonical_choice_exact_and_case_insensitive():
    labels = ["billing", "technical", "sales"]
    assert metrics.canonical_choice("technical", labels) == "technical"
    assert metrics.canonical_choice("TECHNICAL", labels) == "technical"
    assert metrics.canonical_choice("unknown", labels) is None
    assert metrics.canonical_choice("", labels) is None


def test_canonical_choice_applies_aliases():
    labels = ["airplane", "automobile"]
    aliases = {"0": "airplane", "1": "automobile"}
    assert metrics.canonical_choice("1", labels, aliases) == "automobile"


@pytest.mark.parametrize("value", ["true", "TRUE", "yes", "Y", "1", "positive"])
def test_canonical_noul_true_variants(value):
    assert metrics.canonical_noul(value) == "true"


@pytest.mark.parametrize("value", ["false", "no", "N", "0", "negative"])
def test_canonical_noul_false_variants(value):
    assert metrics.canonical_noul(value) == "false"


def test_canonical_noul_unknown_and_alias():
    assert metrics.canonical_noul("maybe") is None
    assert metrics.canonical_noul("oui", {"oui": "yes"}) == "true"


def test_accuracy_and_confusion():
    gold = ["a", "b", "a", "b"]
    pred = ["a", "a", "a", "b"]
    assert metrics.accuracy(gold, pred) == 0.75
    matrix = metrics.confusion_matrix(gold, pred, ["a", "b"])
    assert matrix == {"a": {"a": 2, "b": 0}, "b": {"a": 1, "b": 1}}


def test_confusion_includes_none_predictions_as_extra_column():
    matrix = metrics.confusion_matrix(["a"], [None], ["a"])
    assert matrix["a"]["(none)"] == 1


def test_accuracy_empty_is_none():
    assert metrics.accuracy([], []) is None


def test_macro_f1_perfect_and_partial():
    assert metrics.macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    # One class entirely missed: F1(a)=2/3, F1(b)=0 -> macro 1/3.
    assert metrics.macro_f1(["a", "b"], ["a", "a"], ["a", "b"]) == pytest.approx(1 / 3)
    assert metrics.macro_f1([], [], ["a"]) is None


def test_score_metrics_mae_rmse_exact():
    out = metrics.score_metrics([0.0, 2.0], [1.0, 2.0])
    assert out["n"] == 2
    assert out["mae"] == pytest.approx(0.5)
    assert out["rmse"] == pytest.approx((0.5) ** 0.5)
    # Only the second prediction rounds to its level.
    assert out["exact_match"] == pytest.approx(0.5)


def test_score_metrics_empty():
    out = metrics.score_metrics([], [])
    assert out["n"] == 0
    assert out["mae"] is None
