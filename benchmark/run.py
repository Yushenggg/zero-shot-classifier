"""``zero-shot-bench``: score a labelled CSV with the zero-shot classifier.

Flow: parse the config and questions, load the model **once**, classify every
row against the same question set, compute metrics, write ``predictions.csv``
and ``summary.json``, then unload the model. The load/score/unload split keeps a
long benchmark from pinning VRAM/RAM after it finishes.

Inputs:
    --data       CSV of examples, one labelled row each (``text`` + ``label`` by
                 default; column names are configurable, and multiple text or
                 image columns are supported -- see ``--help``).
    --questions  the same question JSON the CLI and server take; for a single
                 question the shared ``--label-column`` supplies the gold, for
                 several questions each needs a column named after it.

Outputs (under --out):
    predictions.csv  one row per example: original columns + per-question
                     ``<name>.prediction`` / ``.gold`` / ``.correct`` /
                     ``.probability`` (plus ``.error`` for score questions).
    summary.json     run metadata (model, settings, timing) and per-question
                     metrics (accuracy + macro-F1 + confusion for choice/noul,
                     MAE/RMSE/exact-match for score).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zero_shot.core.classifier import Classification, classify
from zero_shot.core.config import load_config
from zero_shot.core.models import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    parse_questions,
)
from zero_shot.core.scorer import get_scorer, unload

from .data import load_examples, read_header, resolve_label_columns
from .metrics import (
    accuracy,
    canonical_choice,
    canonical_noul,
    confusion_matrix,
    macro_f1,
    score_metrics,
)

__all__ = ["build_summary", "main", "metric_summary"]


@dataclass
class _Accumulator:
    """Per-question gold/prediction lists plus counters for the summary."""

    gold: list[Any] = field(default_factory=list)
    pred: list[Any] = field(default_factory=list)
    probabilities: list[float | None] = field(default_factory=list)
    unmatched_gold: int = 0
    errors: int = 0


def _load_json(path: str, label: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise SystemExit(f"{label} file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {exc}") from None


def _parse_label_map(entries: list[str]) -> dict[str, str]:
    """Parse repeatable ``--label-map RAW=CANONICAL`` args into a dict."""
    mapping: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--label-map needs RAW=CANONICAL, got {entry!r}")
        raw, canonical = entry.split("=", 1)
        mapping[raw.strip()] = canonical.strip()
    return mapping


def question_keys(spec: ChoiceQuestion | NoulQuestion | ScoreQuestion) -> list[str]:
    """The option keys a question can score (used as metric class labels)."""
    if isinstance(spec, ChoiceQuestion):
        return list(spec.criteria)
    if isinstance(spec, NoulQuestion):
        return ["true", "false"]
    return []


def canonical_gold(
    spec: ChoiceQuestion | NoulQuestion | ScoreQuestion,
    raw: str,
    aliases: dict[str, str] | None,
) -> Any:
    """Parse a raw CSV gold cell into a canonical label, or ``None`` if invalid."""
    if isinstance(spec, ChoiceQuestion):
        return canonical_choice(raw, question_keys(spec), aliases)
    if isinstance(spec, NoulQuestion):
        return canonical_noul(raw, aliases)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def prediction_of(
    spec: ChoiceQuestion | NoulQuestion | ScoreQuestion, classification: Classification
) -> tuple[Any, float | None]:
    """The model's answer for one question plus the probability of that answer."""
    if isinstance(spec, ChoiceQuestion):
        probability = None
        if classification.choice is not None:
            probability = max(
                (s.probability for s in classification.scores), default=None
            )
        return classification.choice, probability
    if isinstance(spec, NoulQuestion):
        yes = classification.noul or 0.0
        return ("true" if yes >= 0.5 else "false"), max(yes, 1.0 - yes)
    return classification.score, classification.confidence


def is_correct(
    spec: ChoiceQuestion | NoulQuestion | ScoreQuestion, gold: Any, pred: Any
) -> bool:
    if gold is None or pred is None:
        return False
    if isinstance(spec, ScoreQuestion):
        return round(float(pred)) == round(float(gold))
    return gold == pred


def metric_summary(
    specs: dict[str, Any], accs: dict[str, _Accumulator]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build per-question metrics and an overall roll-up from the accumulators."""
    metrics: dict[str, Any] = {}
    accuracies: list[float] = []
    maes: list[float] = []
    for name, spec in specs.items():
        acc = accs[name]
        if isinstance(spec, ScoreQuestion):
            summary = score_metrics(
                [float(g) for g in acc.gold], [float(p) for p in acc.pred]
            )
            summary["type"] = "score"
        else:
            labels = question_keys(spec)
            summary = {
                "type": spec.type,
                "n": len(acc.gold),
                "accuracy": accuracy(acc.gold, acc.pred),
                "macro_f1": macro_f1(acc.gold, acc.pred, labels),
                "confusion": confusion_matrix(acc.gold, acc.pred, labels),
                "labels": labels,
            }
        summary["unmatched_gold"] = acc.unmatched_gold
        summary["errors"] = acc.errors
        metrics[name] = summary
        if summary.get("accuracy") is not None:
            accuracies.append(summary["accuracy"])
        if summary.get("mae") is not None:
            maes.append(summary["mae"])

    overall = {
        "questions": len(specs),
        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "mean_mae": sum(maes) / len(maes) if maes else None,
    }
    return metrics, overall


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _prediction_columns(
    specs: dict[str, Any], row_values: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Flatten one example's per-question results into CSV columns."""
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        values = row_values.get(name, {})
        out[f"{name}.prediction"] = _fmt(values.get("prediction"))
        out[f"{name}.gold"] = _fmt(values.get("gold"))
        out[f"{name}.correct"] = _fmt(values.get("correct"))
        out[f"{name}.probability"] = _fmt(values.get("probability"))
        if isinstance(spec, ScoreQuestion):
            out[f"{name}.error"] = _fmt(values.get("error"))
    return out


def build_summary(
    *,
    dataset_path: str,
    out_dir: Path,
    total: int,
    evaluated: int,
    errors: int,
    metrics: dict[str, Any],
    overall: dict[str, Any],
    questions: dict[str, Any],
    scorer: Any,
    settings: dict[str, Any],
    label_map: dict[str, str],
    label_columns: dict[str, str],
    timing: dict[str, float],
) -> dict[str, Any]:
    """Assemble the machine-readable run report."""
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "examples": total,
            "evaluated": evaluated,
            "errors": errors,
        },
        "model": {
            "id": scorer.model_id,
            "source": scorer.source,
            "from_disk": scorer.from_disk,
            "device": scorer.device,
            "dtype": getattr(scorer, "dtype_name", None),
            "quantized": getattr(scorer, "quantized", None),
            "multimodal": getattr(scorer, "multimodal", None),
        },
        "settings": settings,
        "label_map": label_map,
        "label_columns": label_columns,
        "questions": questions,
        "metrics": metrics,
        "overall": overall,
        "timing": timing,
        "output_dir": str(out_dir),
    }


def _print_summary(metrics: dict[str, Any], overall: dict[str, Any]) -> None:
    print("\n=== benchmark summary ===")
    for name, summary in metrics.items():
        kind = summary.get("type")
        if kind == "score":
            print(
                f"{name} ({kind}, n={summary['n']}): "
                f"MAE={_fmt(summary['mae'])} RMSE={_fmt(summary['rmse'])} "
                f"exact={_fmt(summary['exact_match'])}"
            )
        else:
            print(
                f"{name} ({kind}, n={summary['n']}): "
                f"accuracy={_fmt(summary['accuracy'])} "
                f"macro_f1={_fmt(summary['macro_f1'])}"
            )
        if summary["unmatched_gold"]:
            print(f"  unmatched gold labels skipped: {summary['unmatched_gold']}")
        if summary["errors"]:
            print(f"  scoring errors: {summary['errors']}")
    print(
        f"overall: mean_accuracy={_fmt(overall['mean_accuracy'])} "
        f"mean_mae={_fmt(overall['mean_mae'])}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zero-shot-bench",
        description=(
            "Benchmark the zero-shot classifier on a labelled CSV. Loads the model "
            "once, scores every row against --questions, then unloads it."
        ),
    )
    parser.add_argument(
        "--data", "-d", required=True, help="CSV of labelled examples (one row each)."
    )
    parser.add_argument(
        "--questions",
        "-q",
        required=True,
        help="Question JSON (same format as the CLI/server), or - for stdin.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output directory (default: benchmark/results/<UTC timestamp>).",
    )
    parser.add_argument(
        "--text-column",
        action="append",
        default=None,
        help="CSV column(s) holding the state text; repeat to join several "
        "(e.g. --text-column title --text-column content). Default: text.",
    )
    parser.add_argument(
        "--text-separator",
        default="\n\n",
        help="Separator when joining multiple --text-column values (default: blank line).",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="CSV column with the gold label when there is a single question "
        "(default: label). Ignored when a column named after each question exists.",
    )
    parser.add_argument(
        "--image-column",
        default=None,
        help="CSV column with an image path (resolved against --data-dir) for "
        "vision-language models. Empty cells fall back to text-only.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Base directory for relative image paths (default: the CSV's directory).",
    )
    parser.add_argument(
        "--state-key",
        default=None,
        help='Wrap the text in {"<state-key>": text} (matches the {"input": ...} '
        "convention). Default: pass the bare string as state.",
    )
    parser.add_argument(
        "--label-map",
        action="append",
        default=None,
        metavar="RAW=CANONICAL",
        help="Rewrite a raw gold label before matching option keys (repeatable), "
        "e.g. --label-map 2=Business. Handy for integer class ids and letters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N rows.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-example progress output."
    )

    parser.add_argument(
        "--config", default=None, help="Config file (default: $ZERO_SHOT_CONFIG)."
    )
    parser.add_argument("--model-id", default=None, help="Override the model id.")
    parser.add_argument("--save-to", default=None, help="Override the model cache dir.")
    parser.add_argument("--device", default=None, help="Override the device.")
    parser.add_argument("--gpu", default=None, help="Override the GPU label.")
    parser.add_argument(
        "--quantize",
        default=None,
        choices=("auto", "bf16", "fp32", "int8"),
        help="Override the compute precision.",
    )
    parser.add_argument(
        "--max-image-pixels", type=int, default=None, help="Override the image pixel cap."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override the option softmax temperature."
    )
    parser.add_argument(
        "--calibrate",
        dest="calibrate",
        action="store_true",
        default=None,
        help="Enable contextual calibration (default from config).",
    )
    parser.add_argument(
        "--no-calibrate", dest="calibrate", action="store_false", help="Disable calibration."
    )
    parser.add_argument(
        "--calibration-context", default=None, help="Content-free calibration context."
    )
    parser.add_argument(
        "--kv-cache",
        dest="kv_cache",
        action="store_true",
        default=None,
        help="Reuse one KV cache across options (default from config).",
    )
    parser.add_argument(
        "--no-kv-cache", dest="kv_cache", action="store_false", help="Force exact scoring."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        model_id = args.model_id or config.model
        save_to = config.resolve_save_to(args.save_to)
        device = args.device or config.device
        gpu = args.gpu or config.gpu
        quantize = args.quantize or config.quantize
        max_image_pixels = (
            args.max_image_pixels
            if args.max_image_pixels is not None
            else config.max_image_pixels
        )
        use_kv_cache = config.kv_cache if args.kv_cache is None else args.kv_cache
        temperature = (
            args.temperature if args.temperature is not None else config.temperature
        )
        calibrate = config.calibrate if args.calibrate is None else args.calibrate
        calibration_context = (
            args.calibration_context
            if args.calibration_context is not None
            else config.calibration_context
        )

        questions = _load_json(args.questions, "Question")
        specs = parse_questions(questions)
        aliases = _parse_label_map(args.label_map or [])
        text_columns = args.text_column or ["text"]
        examples = load_examples(
            args.data,
            question_names=list(specs),
            text_columns=text_columns,
            label_column=args.label_column,
            image_column=args.image_column,
            data_dir=args.data_dir,
            state_key=args.state_key,
            text_separator=args.text_separator,
            limit=args.limit,
        )
        label_columns = resolve_label_columns(
            read_header(args.data), list(specs), args.label_column
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not examples:
        print("error: no rows to evaluate", file=sys.stderr)
        return 1

    out_dir = (
        Path(args.out)
        if args.out
        else Path("benchmark/results") / datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    if not args.quiet:
        print(f"loading {model_id} (device={device}) ...", file=sys.stderr)
    try:
        scorer = get_scorer(
            model_id,
            save_to=save_to,
            device=device,
            gpu=gpu,
            quantize=quantize,
            max_image_pixels=max_image_pixels,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    load_seconds = time.perf_counter() - load_start
    if not args.quiet:
        print(
            f"model ready on {scorer.device} in {load_seconds:.1f}s", file=sys.stderr
        )

    accs = {name: _Accumulator() for name in specs}
    prediction_rows: list[dict[str, Any]] = []
    evaluated = 0
    errors = 0
    eval_start = time.perf_counter()
    try:
        for position, example in enumerate(examples, start=1):
            row_out: dict[str, Any] = dict(example.row)
            row_values: dict[str, dict[str, Any]] = {}
            try:
                results = classify(
                    questions,
                    example.state,
                    model_id=model_id,
                    save_to=save_to,
                    device=device,
                    gpu=gpu,
                    quantize=quantize,
                    max_image_pixels=max_image_pixels,
                    temperature=temperature,
                    use_kv_cache=use_kv_cache,
                    calibrate=calibrate,
                    calibration_context=calibration_context,
                    image=example.image,
                )
            except (ValueError, RuntimeError) as exc:
                errors += 1
                row_out["error"] = str(exc)
                for name, spec in specs.items():
                    accs[name].errors += 1
                    row_values[name] = {
                        "prediction": None,
                        "gold": example.gold.get(name),
                        "correct": 0,
                        "probability": None,
                    }
                    if isinstance(spec, ScoreQuestion):
                        row_values[name]["error"] = None
                prediction_rows.append(row_out | _prediction_columns(specs, row_values))
                if not args.quiet:
                    print(f"[{position}/{len(examples)}] error: {exc}", file=sys.stderr)
                continue

            evaluated += 1
            by_name = {result.name: result for result in results}
            for name, spec in specs.items():
                classification = by_name.get(name)
                raw_gold = example.gold.get(name, "")
                acc = accs[name]
                gold = canonical_gold(spec, raw_gold, aliases)
                if classification is None:
                    acc.errors += 1
                    row_values[name] = {
                        "prediction": None,
                        "gold": gold,
                        "correct": 0,
                        "probability": None,
                    }
                    if isinstance(spec, ScoreQuestion):
                        row_values[name]["error"] = None
                    continue
                if gold is None:
                    acc.unmatched_gold += 1
                pred, probability = prediction_of(spec, classification)
                correct = is_correct(spec, gold, pred)
                if gold is not None:
                    acc.gold.append(gold)
                    acc.pred.append(pred)
                    acc.probabilities.append(probability)
                entry = {
                    "prediction": pred,
                    "gold": gold,
                    "correct": int(correct),
                    "probability": probability,
                }
                if isinstance(spec, ScoreQuestion):
                    entry["error"] = (
                        abs(float(pred) - float(gold))
                        if gold is not None and pred is not None
                        else None
                    )
                row_values[name] = entry

            prediction_rows.append(row_out | _prediction_columns(specs, row_values))
            if not args.quiet:
                winners = ", ".join(
                    f"{name}={_fmt(row_values[name]['prediction'])}"
                    for name in specs
                )
                print(
                    f"[{position}/{len(examples)}] {winners} "
                    f"(gold: {', '.join(f'{n}={example.gold.get(n, '')}' for n in specs)})",
                    file=sys.stderr,
                )
    finally:
        unload()
        if not args.quiet:
            print("model unloaded", file=sys.stderr)
    eval_seconds = time.perf_counter() - eval_start

    metrics, overall = metric_summary(specs, accs)
    settings = {
        "temperature": temperature,
        "calibrate": calibrate,
        "calibration_context": calibration_context,
        "kv_cache": use_kv_cache,
        "max_image_pixels": max_image_pixels,
        "text_columns": text_columns,
        "text_separator": args.text_separator,
        "state_key": args.state_key,
        "image_column": args.image_column,
        "limit": args.limit,
    }
    summary = build_summary(
        dataset_path=args.data,
        out_dir=out_dir,
        total=len(examples),
        evaluated=evaluated,
        errors=errors,
        metrics=metrics,
        overall=overall,
        questions=questions,
        scorer=scorer,
        settings=settings,
        label_map=aliases,
        label_columns=label_columns,
        timing={
            "load_seconds": load_seconds,
            "eval_seconds": eval_seconds,
            "seconds_per_example": eval_seconds / evaluated if evaluated else None,
        },
    )

    predictions_path = out_dir / "predictions.csv"
    fieldnames = list(examples[0].row.keys())
    for name, spec in specs.items():
        fieldnames += [
            f"{name}.prediction",
            f"{name}.gold",
            f"{name}.correct",
            f"{name}.probability",
        ]
        if isinstance(spec, ScoreQuestion):
            fieldnames.append(f"{name}.error")
    fieldnames.append("error")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    _print_summary(metrics, overall)
    print(f"\npredictions: {predictions_path}")
    print(f"summary:     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
