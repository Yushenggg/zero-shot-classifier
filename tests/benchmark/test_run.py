"""Regression tests for the zero-shot-bench CLI and its pure helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from benchmark import run

from zero_shot.core.models import Classification, OptionScore, parse_questions


def _scorer() -> SimpleNamespace:
    return SimpleNamespace(
        model_id="test/model",
        source="test/model",
        from_disk=False,
        device="cpu",
        dtype_name="float32",
        quantized=False,
        multimodal=False,
    )


def _choice(name, choice, probabilities) -> Classification:
    scores = [
        OptionScore(option=option, probability=probability)
        for option, probability in probabilities.items()
    ]
    return Classification(
        name=name, type="choice", prompt="", scores=scores, choice=choice
    )


def _write(path, text):
    path.write_text(text)
    return path


def test_question_keys():
    specs = parse_questions(
        {
            "topic": {"type": "choice", "criteria": {"a": "", "b": ""}},
            "urgent": {"type": "noul"},
            "rating": {"type": "score", "criteria": ["low", "high"]},
        }
    )
    assert run.question_keys(specs["topic"]) == ["a", "b"]
    assert run.question_keys(specs["urgent"]) == ["true", "false"]
    assert run.question_keys(specs["rating"]) == []


def test_parse_label_map():
    assert run._parse_label_map(["0=no", "1=yes"]) == {"0": "no", "1": "yes"}


def test_canonical_gold_and_is_correct():
    specs = parse_questions(
        {
            "topic": {"type": "choice", "criteria": {"a": "", "b": ""}},
            "urgent": {"type": "noul"},
            "rating": {"type": "score", "criteria": ["low", "high"]},
        }
    )
    assert run.canonical_gold(specs["topic"], "A", None) == "a"
    assert run.canonical_gold(specs["urgent"], "yes", None) == "true"
    assert run.canonical_gold(specs["rating"], "1", None) == 1.0
    assert run.canonical_gold(specs["rating"], "nope", None) is None

    assert run.is_correct(specs["topic"], "a", "a") is True
    assert run.is_correct(specs["topic"], "a", "b") is False
    assert run.is_correct(specs["rating"], 1.0, 0.9) is True
    assert run.is_correct(specs["rating"], 1.0, 0.4) is False
    assert run.is_correct(specs["topic"], None, "a") is False


def test_prediction_of_each_type():
    specs = parse_questions(
        {
            "topic": {"type": "choice", "criteria": {"a": "", "b": ""}},
            "urgent": {"type": "noul"},
            "rating": {"type": "score", "criteria": ["low", "high"]},
        }
    )
    choice = _choice("topic", "a", {"a": 0.8, "b": 0.2})
    assert run.prediction_of(specs["topic"], choice) == ("a", 0.8)

    noul = Classification(
        name="urgent", type="noul", prompt="", scores=[], noul=0.7
    )
    assert run.prediction_of(specs["urgent"], noul) == ("true", 0.7)

    score = Classification(
        name="rating", type="score", prompt="", scores=[], score=0.25, confidence=0.6
    )
    assert run.prediction_of(specs["rating"], score) == (0.25, 0.6)


def test_metric_summary_choice_and_score():
    specs = parse_questions(
        {
            "topic": {"type": "choice", "criteria": {"a": "", "b": ""}},
            "rating": {"type": "score", "criteria": ["low", "high"]},
        }
    )
    accs = {
        "topic": run._Accumulator(gold=["a", "b"], pred=["a", "a"]),
        "rating": run._Accumulator(gold=[0.0, 1.0], pred=[0.0, 0.0]),
    }
    metrics, overall = run.metric_summary(specs, accs)
    assert metrics["topic"]["accuracy"] == 0.5
    assert metrics["topic"]["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["rating"]["mae"] == 0.5
    assert overall["mean_accuracy"] == 0.5
    assert overall["mean_mae"] == 0.5


def test_main_end_to_end(monkeypatch, tmp_path):
    config = _write(
        tmp_path / "config.toml", 'model = "test/model"\ndevice = "cpu"\n'
    )
    questions = _write(
        tmp_path / "q.json",
        json.dumps(
            {
                "color": {
                    "type": "choice",
                    "instructions": "What color?",
                    "criteria": {"red": "", "green": ""},
                }
            }
        ),
    )
    data_csv = _write(
        tmp_path / "data.csv",
        "text,label\nreddish apple,red\ngreenish apple,green\nboom,red\n",
    )

    monkeypatch.setattr(run, "get_scorer", lambda *a, **k: _scorer())
    monkeypatch.setattr(run, "unload", lambda *a, **k: None)

    def fake_classify(question, state, **kwargs):
        if state == "boom":
            raise RuntimeError("scoring blew up")
        choice = "red" if "red" in state else "green"
        return [_choice("color", choice, {"red": 0.9, "green": 0.1})]

    monkeypatch.setattr(run, "classify", fake_classify)

    out = tmp_path / "out"
    code = run.main(
        [
            "--data", str(data_csv),
            "--questions", str(questions),
            "--config", str(config),
            "--out", str(out),
            "--quiet",
        ]
    )
    assert code == 0

    summary = json.loads((out / "summary.json").read_text())
    assert summary["dataset"] == {
        "path": str(data_csv),
        "examples": 3,
        "evaluated": 2,
        "errors": 1,
    }
    assert summary["metrics"]["color"]["accuracy"] == 1.0
    assert summary["model"]["id"] == "test/model"
    assert summary["timing"]["load_seconds"] >= 0

    predictions = (out / "predictions.csv").read_text().splitlines()
    assert predictions[0].startswith("text,label,color.prediction")
    assert "red,red,1" in predictions[1]
    assert "scoring blew up" in predictions[3]


def test_main_reports_bad_input(monkeypatch, tmp_path, capsys):
    config = _write(tmp_path / "config.toml", 'model = "test/model"\n')
    questions = _write(
        tmp_path / "q.json",
        json.dumps({"color": {"type": "choice", "criteria": {"red": ""}}}),
    )
    data_csv = _write(tmp_path / "data.csv", "sentence,label\nx,red\n")

    monkeypatch.setattr(run, "get_scorer", lambda *a, **k: _scorer())
    monkeypatch.setattr(run, "unload", lambda *a, **k: None)

    code = run.main(
        [
            "--data", str(data_csv),
            "--questions", str(questions),
            "--config", str(config),
            "--out", str(tmp_path / "out"),
            "--quiet",
        ]
    )
    assert code == 1
    assert "text column" in capsys.readouterr().err


def test_main_custom_text_columns(monkeypatch, tmp_path):
    config = _write(tmp_path / "config.toml", 'model = "test/model"\n')
    questions = _write(
        tmp_path / "q.json",
        json.dumps(
            {"topic": {"type": "choice", "criteria": {"a": "", "b": ""}}}
        ),
    )
    data_csv = _write(
        tmp_path / "data.csv", "title,content,label\nA,B,a\nB,A,b\n"
    )

    seen = []

    def fake_classify(question, state, **kwargs):
        seen.append(state)
        return [_choice("topic", "a", {"a": 0.9, "b": 0.1})]

    monkeypatch.setattr(run, "get_scorer", lambda *a, **k: _scorer())
    monkeypatch.setattr(run, "unload", lambda *a, **k: None)
    monkeypatch.setattr(run, "classify", fake_classify)

    code = run.main(
        [
            "--data", str(data_csv),
            "--questions", str(questions),
            "--config", str(config),
            "--out", str(tmp_path / "out"),
            "--text-column", "title",
            "--text-column", "content",
            "--state-key", "input",
            "--quiet",
        ]
    )
    assert code == 0
    assert seen == [{"input": "A\n\nB"}, {"input": "B\n\nA"}]
