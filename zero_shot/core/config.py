from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_SAVE_TO: str | None = "models/qwen3-vl-4b-instruct"
DEFAULT_DEVICE = "auto"
DEFAULT_GPU = "rtx_5060_ti"
DEFAULT_QUANTIZE = "auto"
QUANTIZE_MODES = ("auto", "bf16", "fp32", "int8")
DEVICE_MODES = ("auto", "cpu", "gpu", "cuda")

# Env vars override the config file (and the built-in defaults), matching the
# README. Precedence: $ZERO_SHOT_* > config file > default.
_ENV_KEYS = {
    "model": "ZERO_SHOT_MODEL",
    "save_to": "ZERO_SHOT_SAVE_TO",
    "device": "ZERO_SHOT_DEVICE",
    "gpu": "ZERO_SHOT_GPU",
    "quantize": "ZERO_SHOT_QUANTIZE",
    "max_image_pixels": "ZERO_SHOT_MAX_IMAGE_PIXELS",
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
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate = project_root / "config.toml"
    if candidate.exists():
        return candidate
    return Path.cwd() / "config.toml"


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text.strip().lower() in ("", "null", "none") else text


def _int_or_none(value: object) -> int | None:
    text = _clean(value)
    return None if text is None else int(text)


def _env_override(key: str) -> str | None:
    """Return the $ZERO_SHOT_* value for `key`, or None if unset/empty."""
    return _clean(os.environ.get(_ENV_KEYS[key]))


class Config(BaseModel):
    """Model settings loaded from a TOML config file.

    Paths in `save_to` are resolved relative to `base_dir` (the directory that
    contains the config file), so the model always lands inside the project and
    never depends on the current working directory.
    """

    model_config = ConfigDict(validate_assignment=False)

    model: str = DEFAULT_MODEL_ID
    save_to: str | None = DEFAULT_SAVE_TO
    device: str = DEFAULT_DEVICE
    gpu: str = DEFAULT_GPU
    quantize: str = DEFAULT_QUANTIZE
    max_image_pixels: int | None = None
    kv_cache: bool = True
    temperature: float = 1.0
    calibrate: bool = True
    calibration_context: str = "N/A"
    base_dir: Path = Field(default_factory=Path.cwd)

    @field_validator("temperature")
    @classmethod
    def _temperature_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"config temperature must be > 0, got {value}")
        return value

    @field_validator("quantize")
    @classmethod
    def _quantize_known(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in QUANTIZE_MODES:
            raise ValueError(
                f"config quantize must be one of {QUANTIZE_MODES}, got {mode!r}"
            )
        return mode

    @field_validator("max_image_pixels")
    @classmethod
    def _max_image_pixels_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError(f"config max_image_pixels must be > 0, got {value}")
        return value

    @field_validator("device")
    @classmethod
    def _device_known(cls, value: str) -> str:
        device = value.strip().lower()
        if device not in DEVICE_MODES:
            raise ValueError(
                f"config device must be one of {DEVICE_MODES}, got {device!r}"
            )
        return device

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
    if config_path.is_file():
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

    return Config(
        model=str(pick("model", DEFAULT_MODEL_ID)),
        save_to=_clean(pick("save_to", DEFAULT_SAVE_TO)),
        device=str(pick("device", DEFAULT_DEVICE)),
        gpu=str(pick("gpu", DEFAULT_GPU)),
        quantize=str(pick("quantize", DEFAULT_QUANTIZE)),
        max_image_pixels=_int_or_none(pick("max_image_pixels", None)),
        kv_cache=bool(data.get("kv_cache", True)),
        temperature=data.get("temperature", 1.0),
        calibrate=bool(data.get("calibrate", True)),
        calibration_context=str(data.get("calibration_context", "N/A")),
        base_dir=base_dir,
    )
