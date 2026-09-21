from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .classifier import Classification, classify
from .config import load_config


def _load_json(path: str, label: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise SystemExit(f"{label} file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {exc}")


def _format_tokens(score) -> str:
    parts = [f"{t.token!r}({t.logprob:.3f})" for t in score.tokens]
    parts.append(f"<eos>({score.eos_logprob:.3f})")
    return " ".join(parts)


def _print_text(results: list[Classification], show_prompt: bool) -> None:
    for result in results:
        print(f"=== {result.name} ({result.type}) ===")
        if show_prompt:
            print("prompt:")
            for line in result.prompt.splitlines():
                print(f"  | {line}")
            print()
        print("score = sum(token logprob) + EOS logprob, then softmax over options")
        for score in result.scores:
            label = score.option
            if score.continuation and score.continuation != score.option:
                label = f"{score.option} ({score.continuation})"
            if score.logprob is None:
                print(f"  {label:<22} (not scored)")
            else:
                print(
                    f"  {label:<22} p={score.probability:7.4f}  "
                    f"logp={score.logprob:9.3f}   {_format_tokens(score)}"
                )

        if result.type == "choice":
            print(f"Choice: {result.choice or '(none)'}")
        elif result.type == "noul":
            yes = result.noul or 0.0
            print(f"Noul: {yes:.4f}   (yes={yes:.4f}, no={1 - yes:.4f})")
        elif result.type == "score":
            print(f"Score: {result.score:.4f}   (levels 0..{len(result.scores) - 1})")
            for key, desc in (result.legend or {}).items():
                print(f"  {key}: {desc}")
        if result.confidence is not None:
            print(f"Confidence: {result.confidence:.4f}")
        print(f"Usage: input_tokens={result.input_tokens} output_tokens={result.output_tokens}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zero-shot",
        description=(
            "Zero-shot classifier using the exact next-token distribution (including "
            "EOS) of a local model. Supports choice, noul (yes/no) and score questions."
        ),
    )
    parser.add_argument("--question", "-q", required=True, help="Question JSON file, or - for stdin.")
    parser.add_argument("--state", "-s", default="-", help="State JSON file, or - for stdin (default).")
    parser.add_argument(
        "--config",
        default=None,
        help="Config file (default: $ZERO_SHOT_CONFIG or ./config.toml).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model id. Overrides the config file.",
    )
    parser.add_argument(
        "--save-to",
        default=None,
        help="Directory to download/cache the model. Overrides the config file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'cpu' (default) or 'gpu'/'cuda'. Overrides the config file.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU id (e.g. rtx_5060_ti) used to validate GPU mode. Overrides the config file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature over the options; <1 sharpens, >1 smooths (default 1.0).",
    )
    parser.add_argument(
        "--kv-cache",
        dest="kv_cache",
        action="store_true",
        default=None,
        help="Reuse one KV cache for the shared prompt (default; ~4x faster).",
    )
    parser.add_argument(
        "--no-kv-cache",
        dest="kv_cache",
        action="store_false",
        help="Disable KV-cache reuse and use the exact full-sequence path.",
    )
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt sent to the model.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of a table.")
    args = parser.parse_args(argv)

    if args.question == "-" and args.state == "-":
        raise SystemExit("Only one of --question/--state can read from stdin.")

    config = load_config(args.config)
    model_id = args.model_id or config.model
    save_to = config.resolve_save_to(args.save_to)
    device = args.device or config.device
    gpu = args.gpu or config.gpu
    use_kv_cache = config.kv_cache if args.kv_cache is None else args.kv_cache

    question = _load_json(args.question, "Question")
    state = _load_json(args.state, "State")

    try:
        results = classify(
            question,
            state,
            model_id=model_id,
            save_to=save_to,
            device=device,
            gpu=gpu,
            temperature=args.temperature,
            use_kv_cache=use_kv_cache,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
    else:
        _print_text(results, args.show_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
