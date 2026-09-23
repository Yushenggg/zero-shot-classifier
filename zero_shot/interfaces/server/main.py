"""``zero-shot-serve`` entry point: serve the web UI and HTTP API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from ...core.config import load_config
from .app import app, configure


def cpu_low_config_path() -> Path:
    """Locate the bundled ``config.cpu.toml`` (project root, else cwd).

    ``parents[3]`` is the project root for a source/editable checkout; a wheel
    install falls back to the current working directory (Docker runs from
    ``/app``, where the file is copied).
    """
    project_root = Path(__file__).resolve().parents[3]
    candidate = project_root / "config.cpu.toml"
    return candidate if candidate.exists() else Path.cwd() / "config.cpu.toml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zero-shot-serve",
        description="Serve the zero-shot classifier web UI and HTTP API.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file to serve, e.g. config.cpu.toml for the SmolVLM-500M CPU path.",
    )
    parser.add_argument(
        "--cpu-low",
        action="store_true",
        help="Serve the CPU model (config.cpu.toml, SmolVLM-500M) instead of the configured one.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: $ZERO_SHOT_PORT or 8000).",
    )
    args = parser.parse_args(argv)

    if args.config:
        configure(load_config(args.config))
    elif args.cpu_low:
        configure(load_config(cpu_low_config_path()))

    host = os.environ.get("ZERO_SHOT_HOST", "127.0.0.1")
    port = args.port if args.port is not None else int(os.environ.get("ZERO_SHOT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
