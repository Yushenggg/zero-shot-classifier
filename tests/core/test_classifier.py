"""Regression tests for prompt building, scoring math, and classify."""

from __future__ import annotations

import math

import pytest

from zero_shot.core.classifier import _build_question, _render, classify, classify_one
from zero_shot.core.models import ChoiceQuestion, NoulQuestion, ScoreQuestion


def test_render_handles_strings_objects_arrays_and_null():
    assert _render(None) == ""
    assert _render("plain") == "plain"
    assert _render({"a": 1}) == '{\n  "a": 1\n}'
    assert "1" in _render([1, 2])



def test_build_question_choice_includes_criteria():
    spec = ChoiceQuestion(
        type="choice",
        instructions="Pick one",
        criteria={"red": "the color red", "green": ""},
    )
    prompt, options = _build_question("color", spec, {"input": "a fruit"})
    assert "# CONTEXT" in prompt
    assert "# TASK" in prompt
    assert "# CRITERIA" in prompt
    assert "red: the color red" in prompt
    assert "# ANSWER" in prompt
    assert options == [("red", "red"), ("green", "green")]


def test_build_question_noul_and_score():
    noul_prompt, noul_options = _build_question("q", NoulQuestion(type="noul"), None)
    assert "yes or no" in noul_prompt
    assert noul_options == [("true", "yes"), ("false", "no")]

    score_prompt, score_options = _build_question(
        "q", ScoreQuestion(type="score", criteria=["Calm", "Angry"]), None
    )
    assert "# RATING SCALE" in score_prompt
    assert "0: Calm" in score_prompt
    assert score_options == [("0", "0"), ("1", "1")]


def test_classify_one_choice_softmax(fake_scorer):
    fake_scorer.logprobs = {"red": -1.0, "green": -2.0, "blue": -3.0}
    result = classify_one(
        "color",
        {
            "type": "choice",
            "instructions": "?",
            "criteria": {"red": "", "green": "", "blue": ""},
        },
        {},
    )
    assert result.type == "choice"
    assert result.choice == "red"
    probs = {s.option: s.probability for s in result.scores}
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
    assert probs["red"] > probs["green"] > probs["blue"]
    assert result.confidence == pytest.approx(sum(p**2 for p in probs.values()))
    assert result.output_tokens == 2  # one continuation token + EOS


def test_classify_one_noul_reports_yes_probability(fake_scorer):
    fake_scorer.logprobs = {"yes": -0.1, "no": -1.0}
    result = classify_one("q", {"type": "noul", "instructions": "?"}, {})
    assert result.type == "noul"
    assert result.noul == pytest.approx(0.7109495, abs=1e-4)
    assert result.choice is None


def test_classify_one_score_is_probability_weighted_mean(fake_scorer):
    fake_scorer.logprobs = {"0": 0.0, "1": -1.0, "2": -2.0}
    result = classify_one("q", {"type": "score", "criteria": ["lo", "mid", "hi"]}, {})
    assert result.type == "score"
    probs = {s.option: s.probability for s in result.scores}
    expected = sum(int(option) * p for option, p in probs.items())
    assert result.score == pytest.approx(expected)
    assert result.legend == {"0": "lo", "1": "mid", "2": "hi"}
    assert result.confidence is not None


def test_calibration_subtracts_content_free_prior(fake_scorer):
    # Uncalibrated: red (0.0) beats green (-0.1).
    fake_scorer.logprobs = {"red": 0.0, "green": -0.1}
    # Priors penalize red, so calibration flips the winner to green.
    fake_scorer.prior = {"red": 1.0, "green": -1.0}
    spec = {"type": "choice", "instructions": "?", "criteria": {"red": "", "green": ""}}

    uncalibrated = classify_one("c", spec, {"input": "state"}, calibrate=False)
    assert uncalibrated.choice == "red"

    calibrated = classify_one("c", spec, {"input": "state"}, calibrate=True)
    assert calibrated.choice == "green"
    by_option = {s.option: s for s in calibrated.scores}
    assert by_option["red"].calibrated_logprob == pytest.approx(-1.0)
    assert by_option["green"].calibrated_logprob == pytest.approx(0.9)
    # Raw logprob is still reported alongside the calibrated one.
    assert by_option["red"].logprob == pytest.approx(0.0)


def test_temperature_sharpens_and_smooths(fake_scorer):
    fake_scorer.logprobs = {"a": 0.0, "b": -1.0}
    spec = {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}}

    sharp = classify_one("c", spec, {}, temperature=0.1)
    smooth = classify_one("c", spec, {}, temperature=10.0)
    sharp_top = max(s.probability for s in sharp.scores)
    smooth_top = max(s.probability for s in smooth.scores)
    assert sharp_top > smooth_top
    assert smooth_top < 0.6


def test_classify_validates_every_question_before_scoring(monkeypatch):
    calls = []
    monkeypatch.setattr("zero_shot.core.classifier.get_scorer", lambda *a, **k: calls.append(1))
    question = {
        "good": {"type": "noul", "instructions": "?"},
        "bad": {"type": "choice", "criteria": {}},
    }
    with pytest.raises(ValueError):
        classify(question, {})
    assert calls == []


def test_classify_returns_one_result_per_question(fake_scorer):
    fake_scorer.logprobs = {"yes": 0.0, "no": -1.0}
    results = classify(
        {"a": {"type": "noul", "instructions": "?"}, "b": {"type": "noul", "instructions": "?"}},
        {},
    )
    assert [r.name for r in results] == ["a", "b"]
    assert all(r.to_dict()["type"] == "noul" for r in results)


def test_unknown_and_all_unknown_options_are_reported(fake_scorer):
    spec = {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}}

    # One option did not come back from the scorer: it is listed with no logprob.
    fake_scorer.logprobs = {"a": 0.0}
    fake_scorer.omit = {"b"}
    result = classify_one("c", spec, {})
    by_option = {s.option: s for s in result.scores}
    assert by_option["a"].probability == pytest.approx(1.0)
    assert by_option["b"].logprob is None
    assert by_option["b"].probability == 0.0
    assert result.choice == "a"

    # Nothing scored: softmax is skipped and there is no winner.
    fake_scorer.omit = {"a", "b"}
    empty = classify_one("c", spec, {})
    assert all(s.probability == 0.0 for s in empty.scores)
    assert empty.choice is None


def test_temperature_must_be_positive(fake_scorer):
    fake_scorer.logprobs = {"a": 0.0, "b": -1.0}
    spec = {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}}
    with pytest.raises(ValueError):
        classify_one("c", spec, {}, temperature=0.0)


def test_text_scoring_uses_leading_space(fake_scorer):
    fake_scorer.logprobs = {"yes": 0.0, "no": -1.0}
    classify_one("q", {"type": "noul", "instructions": "?"}, {})
    main = fake_scorer.calls[0]
    assert main["add_leading_space"] is True
    assert main["has_image"] is False
    assert main["chat_template"] is False


def test_image_calibration_contract(fake_scorer):
    fake_scorer.logprobs = {"yes": 0.0, "no": -1.0}
    fake_scorer.prior = {"yes": -0.5, "no": -0.5}
    classify_one("q", {"type": "noul", "instructions": "?"}, {}, image=b"fake", calibrate=True)

    main, null = fake_scorer.calls
    # Main pass: the image is the leading signal, no synthetic space.
    assert main["has_image"] is True
    assert main["add_leading_space"] is False
    assert main["chat_template"] is False
    # Calibration pass: text-only, rendered through the chat template.
    assert null["has_image"] is False
    assert null["add_leading_space"] is False
    assert null["chat_template"] is True

