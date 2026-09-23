"""Regression tests for the zero-shot CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero_shot.core.models import Classification, OptionScore, TokenScore
from zero_shot.interfaces.cli import main as cli


def _classification() -> Classification:
    return Classification(
        name="color",
        type="choice",
        prompt="PROMPT TEXT",
        scores=[
            OptionScore(
                option="red",
                continuation="red",
                tokens=[TokenScore(token=" red", token_id=7, logprob=-0.3)],
                eos_logprob=-1.0,
                logprob=-1.3,
                probability=0.7,
            ),
            OptionScore(
                option="green",
                continuation="green",
                tokens=[TokenScore(token=" green", token_id=8, logprob=-0.9)],
                eos_logprob=-1.0,
                logprob=-1.9,
                probability=0.3,
            ),
        ],
        choice="red",
        confidence=0.7,
    )


def _write(tmp_path: Path, name: str, payload) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


@pytest.fixture
def cli_files(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('model = "test/model"\ndevice = "cpu"\n')
    question = _write(tmp_path, "q.json", {"color": {"type": "noul", "instructions": "?"}})
    state = _write(tmp_path, "s.json", {"input": "hello"})
    return {"config": str(config), "question": question, "state": state}


def test_load_json_from_file(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"a": 1}')
    assert cli._load_json(str(path), "Question") == {"a": 1}


def test_load_json_missing_file_exits():
    with pytest.raises(SystemExit):
        cli._load_json("/does/not/exist.json", "Question")


def test_load_json_invalid_json_exits(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(SystemExit):
        cli._load_json(str(path), "Question")


def test_load_image_missing_file_exits():
    with pytest.raises(SystemExit):
        cli._load_image("/does/not/exist.png")


def test_format_tokens():
    score = _classification().scores[0]
    assert cli._format_tokens(score) == "' red'(-0.300) <eos>(-1.000)"


def test_main_json_output(monkeypatch, capsys, cli_files):
    monkeypatch.setattr(cli, "classify", lambda *a, **k: [_classification()])
    code = cli.main(
        [
            "-q", cli_files["question"],
            "-s", cli_files["state"],
            "--config", cli_files["config"],
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "color"
    assert payload[0]["type"] == "choice"
    assert payload[0]["choice"] == "red"


def test_main_table_output_and_show_prompt(monkeypatch, capsys, cli_files):
    monkeypatch.setattr(cli, "classify", lambda *a, **k: [_classification()])
    code = cli.main(
        [
            "-q", cli_files["question"],
            "-s", cli_files["state"],
            "--config", cli_files["config"],
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Choice: red" in out
    assert "PROMPT TEXT" not in out

    code = cli.main(
        [
            "-q", cli_files["question"],
            "-s", cli_files["state"],
            "--config", cli_files["config"],
            "--show-prompt",
        ]
    )
    assert code == 0
    assert "PROMPT TEXT" in capsys.readouterr().out


def test_main_forwards_overrides_to_classify(monkeypatch, cli_files):
    captured = {}

    def fake_classify(question, state, **kwargs):
        captured.update(kwargs)
        return [_classification()]

    monkeypatch.setattr(cli, "classify", fake_classify)
    cli.main(
        [
            "-q", cli_files["question"],
            "-s", cli_files["state"],
            "--config", cli_files["config"],
            "--device", "cpu",
            "--temperature", "0.5",
            "--no-calibrate",
            "--no-kv-cache",
        ]
    )
    assert captured["temperature"] == 0.5
    assert captured["device"] == "cpu"
    assert captured["calibrate"] is False
    assert captured["use_kv_cache"] is False


def test_main_reports_classification_errors(monkeypatch, capsys, cli_files):
    def boom(*a, **k):
        raise ValueError("bad question")

    monkeypatch.setattr(cli, "classify", boom)
    code = cli.main(
        ["-q", cli_files["question"], "-s", cli_files["state"], "--config", cli_files["config"]]
    )
    assert code == 1
    assert "bad question" in capsys.readouterr().err


def test_two_stdin_sources_rejected(cli_files):
    with pytest.raises(SystemExit):
        cli.main(["-q", "-", "-s", "-", "--config", cli_files["config"]])
