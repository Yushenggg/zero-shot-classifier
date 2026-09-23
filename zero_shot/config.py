from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_SAVE_TO: str | None = "models/qwen3-vl-4b-instruct"
DEFAULT_DEVICE = "auto"
DEFAULT_GPU = "rtx_5060_ti"
DEFAULT_QUANTIZE = "auto"
QUANTIZE_MODES = ("auto", "bf16", "fp32", "int8")

# Env vars override the config file (and the built-in defaults), matching the
# README. Precedence: $ZERO_SHOT_* > config file > default.
_ENV_KEYS = {
    "model": "ZERO_SHOT_MODEL",
    "save_to": "ZERO_SHOT_SAVE_TO",
    "device": "ZERO_SHOT_DEVICE",
    "gpu": "ZERO_SHOT_GPU",
    "quantize": "ZERO_SHOT_QUANTIZE",
}

# GPUs we know how to run on. Compute capability / CUDA are informational; the
# name is checked against torch.cuda.get_device_name() when device = "gpu".
SUPPORTED_GPUS: dict[str, dict[str, object]] = {
    "rtx_5060_ti": {
        "name": "NVIDIA GeForce RTX 5060 Ti",
        "compute_capability": "12.0",
        "cuda": "13.0",
        "vram_gb": 16,
    },
}


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


def _env_override(key: str) -> str | None:
    """Return the $ZERO_SHOT_* value for `key`, or None if unset/empty."""
    return _clean(os.environ.get(_ENV_KEYS[key]))


@dataclass
class Config:
    """Model settings loaded from a TOML config file.

    Paths in `save_to` are resolved relative to `base_dir` (the directory that
    contains the config file), so the model always lands inside the project and
    never depends on the current working directory.
    """

    model: str = DEFAULT_MODEL_ID
    save_to: str | None = DEFAULT_SAVE_TO
    device: str = DEFAULT_DEVICE
    gpu: str = DEFAULT_GPU
    quantize: str = DEFAULT_QUANTIZE
    kv_cache: bool = True
    temperature: float = 1.0
    calibrate: bool = True
    calibration_context: str = "N/A"
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

    def save_to_for(self, model_id: str) -> str | None:
        """Local dir for `model_id`: the configured one for the default model,
        otherwise a sibling directory named after the model. None disables disk
        caching (use the Hugging Face cache)."""
        if model_id == self.model:
            return self.resolve_save_to()
        if not self.save_to:
            return None
        base = self.local_dir
        parent = base.parent if base else (self.base_dir / "models")
        return str(parent / model_id.replace("/", "--"))


def load_config(path: str | Path | None = None) -> Config:
    """Load config from `path` (default: $ZERO_SHOT_CONFIG or the project's config.toml).

    Missing file or keys fall back to the built-in defaults.
    """
    config_path = Path(path) if path else _default_config_path()
    if config_path.exists():
        data = tomllib.loads(config_path.read_text())
        base_dir = config_path.resolve().parent
    else:
        data = {}
        base_dir = Path.cwd()

    def pick(key: str, default: object) -> object:
        env = _env_override(key)
        if env is not None:
            return env
        if data.get(key) is not None:
            return data[key]
        return default

    temperature = float(data.get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError(f"config temperature must be > 0, got {temperature}")
    quantize = str(pick("quantize", DEFAULT_QUANTIZE)).strip().lower()
    if quantize not in QUANTIZE_MODES:
        raise ValueError(
            f"config quantize must be one of {QUANTIZE_MODES}, got {quantize!r}"
        )
    return Config(
        model=str(pick("model", DEFAULT_MODEL_ID)),
        save_to=_clean(pick("save_to", DEFAULT_SAVE_TO)),
        device=str(pick("device", DEFAULT_DEVICE)).strip().lower(),
        gpu=str(pick("gpu", DEFAULT_GPU)),
        quantize=quantize,
        kv_cache=bool(data.get("kv_cache", True)),
        temperature=temperature,
        calibrate=bool(data.get("calibrate", True)),
        calibration_context=str(data.get("calibration_context", "N/A")),
        base_dir=base_dir,
    )
