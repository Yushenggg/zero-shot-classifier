"""Regression tests for benchmark CSV loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmark import data


def _write_csv(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_read_header(tmp_path):
    path = _write_csv(tmp_path, "d.csv", "text,label\na,1\n")
    assert data.read_header(str(path)) == ["text", "label"]
    empty = _write_csv(tmp_path, "e.csv", "")
    assert data.read_header(str(empty)) == []


def test_resolve_label_columns_single_question():
    columns = data.resolve_label_columns(["text", "label"], ["topic"], "label")
    assert columns == {"topic": "label"}


def test_resolve_label_columns_per_question_columns():
    columns = data.resolve_label_columns(["text", "a", "b"], ["a", "b"], "label")
    assert columns == {"a": "a", "b": "b"}


def test_resolve_label_columns_error_when_ambiguous():
    with pytest.raises(ValueError, match="more than one question"):
        data.resolve_label_columns(["text", "label"], ["a", "b"], "label")


def test_resolve_label_columns_error_when_missing():
    with pytest.raises(ValueError, match="not found"):
        data.resolve_label_columns(["text"], ["topic"], "label")


def test_load_examples_basic(tmp_path):
    path = _write_csv(tmp_path, "d.csv", "text,label\nalpha,1\nbeta,0\n")
    examples = data.load_examples(str(path), question_names=["q"])
    assert len(examples) == 2
    assert examples[0].state == "alpha"
    assert examples[0].gold == {"q": "1"}
    assert examples[0].row == {"text": "alpha", "label": "1"}


def test_load_examples_joins_multiple_text_columns(tmp_path):
    path = _write_csv(tmp_path, "d.csv", "title,content,label\nT,C,1\n,only,0\n")
    examples = data.load_examples(
        str(path),
        question_names=["q"],
        text_columns=["title", "content"],
        text_separator=" | ",
    )
    assert examples[0].state == "T | C"
    # Empty cells are dropped rather than leaving a dangling separator.
    assert examples[1].state == "only"


def test_load_examples_state_key_wraps_text(tmp_path):
    path = _write_csv(tmp_path, "d.csv", "text,label\nhello,1\n")
    examples = data.load_examples(str(path), question_names=["q"], state_key="input")
    assert examples[0].state == {"input": "hello"}


def test_load_examples_limit(tmp_path):
    path = _write_csv(tmp_path, "d.csv", "text,label\na,1\nb,0\nc,1\n")
    examples = data.load_examples(str(path), question_names=["q"], limit=2)
    assert [e.index for e in examples] == [0, 1]


def test_load_examples_resolves_image_paths(tmp_path):
    (tmp_path / "cat.png").write_bytes(b"not-a-real-png")
    path = _write_csv(tmp_path, "d.csv", "text,img,label\na,cat.png,1\n")
    examples = data.load_examples(str(path), question_names=["q"], image_column="img")
    assert examples[0].image == str(tmp_path / "cat.png")


def test_load_examples_missing_image_raises(tmp_path):
    path = _write_csv(tmp_path, "d.csv", "text,img,label\na,nope.png,1\n")
    with pytest.raises(ValueError, match="image not found"):
        data.load_examples(str(path), question_names=["q"], image_column="img")


def test_load_examples_missing_text_column_raises(tmp_path):
    path = _write_csv(tmp_path, "d.csv", "text,label\na,1\n")
    with pytest.raises(ValueError, match="text column"):
        data.load_examples(str(path), question_names=["q"], text_columns=["sentence"])


def test_load_examples_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="data file not found"):
        data.load_examples(str(tmp_path / "nope.csv"), question_names=["q"])
