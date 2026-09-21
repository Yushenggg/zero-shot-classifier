from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ID = os.environ.get("ZERO_SHOT_MODEL", "google/gemma-4-E2B-it")
DEFAULT_SAVE_TO: str | None = os.environ.get("ZERO_SHOT_SAVE_TO", "models/gemma-4-E2B-it")


def _default_config_path() -> Path:
    """Locate config.toml without depending on the current working directory."""
    env = os.environ.get("ZERO_SHOT_CONFIG")
    if env:
        return Path(env)
    project_root = Path(__file__).resolve().parent.parent
    candidate = project_root / "config.toml"
    if candidate.exists():
        return candidate
    return Path.cwd() / "config.toml"


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text.strip().lower() in ("", "null", "none") else text


@dataclass
class Config:
    """Model settings loaded from a TOML config file.

    Paths in `save_to` are resolved relative to `base_dir` (the directory that
    contains the config file), so the model always lands inside the project and
    never depends on the current working directory.
    """

    model: str = DEFAULT_MODEL_ID
    save_to: str | None = DEFAULT_SAVE_TO
    base_dir: Path = field(default_factory=Path.cwd)

    @property
    def local_dir(self) -> Path | None:
        resolved = self.resolve_save_to()
        return Path(resolved) if resolved else None

    def resolve_save_to(self, override: str | None = None) -> str | None:
        value = _clean(self.save_to if override is None else override)
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        return str(path)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from `path` (default: $ZERO_SHOT_CONFIG or the project's config.toml).

    Missing file or keys fall back to the built-in defaults.
    """
    config_path = Path(path) if path else _default_config_path()
    if not config_path.exists():
        return Config(base_dir=Path.cwd())

    data = tomllib.loads(config_path.read_text())
    return Config(
        model=str(data.get("model", DEFAULT_MODEL_ID)),
        save_to=_clean(data.get("save_to", DEFAULT_SAVE_TO)),
        base_dir=config_path.resolve().parent,
    )
