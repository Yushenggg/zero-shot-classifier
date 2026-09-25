"""CSV loading for the benchmark harness.

The harness accepts the shape most HF datasets are dumped to -- one row per
example with a text field and a label field -- while staying flexible about the
column names (``text``, ``sentence``, ``content``/``title``, ``query``, ...) and
about images (``img``, ``jpg``, ``image``). A row maps to the classifier as:

    state  = the joined text column(s), or ``{state_key: text}`` if set
    image  = one path resolved relative to --data-dir (for VLM rows)
    gold   = the label column (or one column per question name)

Nothing model-specific lives here, so it is unit-testable with plain CSV files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Example", "load_examples", "read_header", "resolve_label_columns"]


@dataclass(frozen=True)
class Example:
    """One CSV row, pre-resolved for the classifier."""

    index: int
    state: Any
    gold: dict[str, str]
    row: dict[str, str]
    image: str | None = None


def read_header(path: str | Path) -> list[str]:
    """Return a CSV's column names (empty list for an empty file/header)."""
    with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), [])


def resolve_label_columns(
    fieldnames: list[str],
    question_names: list[str],
    label_column: str,
) -> dict[str, str]:
    """Decide which CSV column holds the gold label for each question.

    Two conventions are supported: a column named after every question (the
    multi-question case), or a single shared label column when there is exactly
    one question. Anything else is a configuration error, not a silent skip.
    """
    if question_names and all(name in fieldnames for name in question_names):
        return {name: name for name in question_names}

    if len(question_names) == 1:
        name = question_names[0]
        if label_column not in fieldnames:
            raise ValueError(
                f"label column {label_column!r} not found in CSV; columns are "
                f"{fieldnames}"
            )
        return {name: label_column}

    missing = [name for name in question_names if name not in fieldnames]
    raise ValueError(
        "could not resolve label columns: "
        f"questions {missing} have no matching CSV column and there is more "
        f"than one question, so a single --label-column cannot be used. "
        f"Columns are {fieldnames}."
    )


def load_examples(
    path: str | Path,
    *,
    question_names: list[str],
    text_columns: list[str] | None = None,
    label_column: str = "label",
    image_column: str | None = None,
    data_dir: str | Path | None = None,
    state_key: str | None = None,
    text_separator: str = "\n\n",
    limit: int | None = None,
) -> list[Example]:
    """Read a CSV into :class:`Example` rows.

    ``text_columns`` are joined with ``text_separator`` (empty cells dropped), so
    a dataset split across ``title`` + ``content`` still becomes one state.
    ``state_key`` wraps the text in ``{state_key: text}`` to match the project's
    ``{"input": ...}`` convention; leave it unset to pass the bare string.
    ``image_column`` is resolved against ``data_dir`` (default: the CSV's
    directory) and must exist on disk.
    """
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise ValueError(f"data file not found: {csv_path}")

    text_columns = list(text_columns or ["text"])
    base = Path(data_dir).expanduser() if data_dir else csv_path.parent

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header row: {csv_path}")

        missing_text = [name for name in text_columns if name not in fieldnames]
        if missing_text:
            raise ValueError(
                f"text column(s) {missing_text} not found in CSV; columns are "
                f"{fieldnames}"
            )
        label_columns = resolve_label_columns(fieldnames, question_names, label_column)

        if image_column is not None and image_column not in fieldnames:
            raise ValueError(
                f"image column {image_column!r} not found in CSV; columns are "
                f"{fieldnames}"
            )

        examples: list[Example] = []
        for index, row in enumerate(reader):
            if limit is not None and len(examples) >= limit:
                break
            parts = [(row.get(name) or "").strip() for name in text_columns]
            text = text_separator.join(part for part in parts if part)
            state: Any = {state_key: text} if state_key else text

            image: str | None = None
            if image_column is not None:
                raw = (row.get(image_column) or "").strip()
                if raw:
                    candidate = Path(raw)
                    if not candidate.is_absolute():
                        candidate = base / candidate
                    if not candidate.is_file():
                        raise ValueError(
                            f"row {index}: image not found: {raw} "
                            f"(resolved to {candidate})"
                        )
                    image = str(candidate)

            gold = {
                name: (row.get(column) or "").strip()
                for name, column in label_columns.items()
            }
            examples.append(
                Example(index=index, state=state, gold=gold, row=dict(row), image=image)
            )
        return examples
