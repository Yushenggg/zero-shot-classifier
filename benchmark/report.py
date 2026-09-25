"""Build a self-contained HTML report from zero-shot-bench summary.json files.

Usage:
    .venv/bin/python -m benchmark.report
    .venv/bin/python -m benchmark.report path/to/summary.json -o report.html

The output is a single HTML file with every summary inlined, so it opens
straight from disk -- no server and no fetch/CORS problems. Layout and styling
live in ``report_template.html`` next to this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TEMPLATE_NAME = "report_template.html"

__all__ = ["build_html", "discover", "load_reports", "main"]


def discover(paths: list[str]) -> list[Path]:
    """Find ``summary.json`` files under the given files/directories."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
            continue
        if path.is_dir():
            for pattern in (
                "*/*/out*/summary.json",
                "*/out*/summary.json",
                "out*/summary.json",
            ):
                found.extend(sorted(path.glob(pattern)))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    """Load each summary and tag it with its dataset and run.

    Handles the native layout (``<dataset>/out/summary.json``) and variant run
    dirs (``<dataset>/out-nocal/summary.json``): the dataset name comes from the
    grandparent whenever the parent looks like an output dir. The run label is
    taken from the summary's ``calibrate`` setting so calibrated and
    uncalibrated runs compare clearly.
    """
    reports: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text())
        parent = path.parent
        if parent.name.startswith("out"):
            report["name"] = parent.parent.name
        else:
            report["name"] = parent.name
        report["run_dir"] = parent.name
        calibrate = report.get("settings", {}).get("calibrate")
        if calibrate is True:
            report["run"] = "calibrated"
        elif calibrate is False:
            report["run"] = "uncalibrated"
        else:
            report["run"] = parent.name
        reports.append(report)
    return reports


def build_html(reports: list[dict[str, Any]]) -> str:
    """Render the template with the reports embedded as JSON."""
    template = Path(__file__).with_name(_TEMPLATE_NAME).read_text()
    payload = json.dumps(reports, ensure_ascii=False).replace("</", "<\\/")
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return template.replace("__REPORT_DATA__", payload).replace("__GENERATED_AT__", generated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zero-shot-bench-report",
        description="Build a self-contained HTML report from benchmark summary.json files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["benchmark/data"],
        help="summary.json files or directories to scan (default: benchmark/data).",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="benchmark/data/report.html",
        help="Output HTML path (default: benchmark/data/report.html).",
    )
    args = parser.parse_args(argv)

    files = discover(args.paths)
    if not files:
        print(f"no summary.json found under: {', '.join(args.paths)}", file=sys.stderr)
        return 1

    reports = load_reports(files)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(reports))
    print(f"wrote {out} ({len(reports)} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
