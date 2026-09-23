"""Regression tests for the pydantic domain models and their wire shape."""

from __future__ import annotations

import pytest

from zero_shot.core.models import (
    SUPPORTED_TYPES,
    ChoiceQuestion,
    Classification,
    NoulQuestion,
    OptionScore,
    ScoreQuestion,
    SequenceScore,
    TokenScore,
    parse_question,
    parse_questions,
)


def test_supported_types():
    assert SUPPORTED_TYPES == ("choice", "noul", "score")


def test_parse_questions_returns_typed_specs():
    specs = parse_questions(
        {
            "a": {"type": "choice", "instructions": "pick", "criteria": {"x": "d"}},
            "b": {"type": "noul"},
            "c": {"type": "score", "criteria": ["lo", "hi"]},
        }
    )
    assert isinstance(specs["a"], ChoiceQuestion)
    assert isinstance(specs["b"], NoulQuestion)
    assert isinstance(specs["c"], ScoreQuestion)
    assert specs["a"].criteria == {"x": "d"}
    assert specs["b"].instructions == ""
    assert specs["c"].criteria == ["lo", "hi"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        None,
        {"q": {"type": "choice", "criteria": {}}},
        {"q": {"type": "score", "criteria": ["only-one"]}},
        {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}},
        {"q": {"type": "bogus"}},
        {"q": {"criteria": {"x": "d"}}},
    ],
)
def test_parse_questions_rejects_bad_payloads(payload):
    with pytest.raises(ValueError):
        parse_questions(payload)


def test_parse_question_accepts_typed_and_dict():
    typed = ChoiceQuestion(type="choice", criteria={"x": "d"})
    assert parse_question(typed) is typed
    assert isinstance(parse_question({"type": "noul"}), NoulQuestion)


def test_noul_criteria_is_ignored_and_flexible():
    # Historically `criteria` is accepted but ignored for noul; any shape is fine.
    assert parse_question({"type": "noul", "criteria": ["true", "false"]}).criteria == [
        "true",
        "false",
    ]


def test_token_and_sequence_score_to_dict():
    token = TokenScore(token=" hi", token_id=42, logprob=-1.25)
    assert token.to_dict() == {"token": " hi", "token_id": 42, "logprob": -1.25}

    seq = SequenceScore(
        option="a",
        continuation="a",
        tokens=[token],
        eos_token="<eos>",
        eos_logprob=-0.5,
        total_logprob=-1.75,
    )
    assert seq.to_dict() == {
        "option": "a",
        "continuation": "a",
        "tokens": [{"token": " hi", "token_id": 42, "logprob": -1.25}],
        "eos_token": "<eos>",
        "eos_logprob": -0.5,
        "total_logprob": -1.75,
    }


def test_option_score_to_dict_omits_unset_calibrated_logprob():
    plain = OptionScore(option="a", continuation="a", probability=0.6)
    keys = plain.to_dict().keys()
    assert "calibrated_logprob" not in keys
    # logprob/eos_logprob are always present, as null when unknown.
    assert plain.to_dict()["logprob"] is None
    assert plain.to_dict()["eos_logprob"] is None

    calibrated = OptionScore(
        option="a",
        continuation="a",
        probability=0.6,
        logprob=-1.0,
        calibrated_logprob=-0.25,
    )
    assert calibrated.to_dict()["calibrated_logprob"] == -0.25


def test_classification_to_dict_shapes():
    score = OptionScore(option="a", continuation="a", probability=1.0)

    choice = Classification(
        name="q", type="choice", prompt="p", scores=[score], choice="a", confidence=1.0
    )
    choice_dict = choice.to_dict()
    assert list(choice_dict) == [
        "name",
        "type",
        "prompt",
        "probabilities",
        "scores",
        "choice",
        "confidence",
    ]
    assert choice_dict["probabilities"] == {"a": 1.0}

    noul = Classification(name="q", type="noul", prompt="p", scores=[score], noul=0.3)
    assert list(noul.to_dict()) == [
        "name",
        "type",
        "prompt",
        "probabilities",
        "scores",
        "noul",
    ]

    level = Classification(
        name="q",
        type="score",
        prompt="p",
        scores=[score],
        score=1.5,
        legend={"0": "lo", "1": "hi"},
    )
    level_dict = level.to_dict()
    assert list(level_dict) == [
        "name",
        "type",
        "prompt",
        "probabilities",
        "scores",
        "score",
        "legend",
    ]
    assert level_dict["score"] == 1.5
    assert level_dict["legend"] == {"0": "lo", "1": "hi"}
