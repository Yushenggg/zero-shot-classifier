"""Regression tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from zero_shot.core.config import Config, load_config


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


def test_load_config_reads_values_and_resolves_paths(tmp_path):
    path = _write_config(
        tmp_path,
        """
model = "acme/model"
save_to = "models/acme"
device = "cpu"
gpu = "test-gpu"
quantize = "fp32"
kv_cache = false
temperature = 0.5
calibrate = false
calibration_context = "EMPTY"
""",
    )
    config = load_config(path)

    assert config.model == "acme/model"
    assert config.device == "cpu"
    assert config.quantize == "fp32"
    assert config.kv_cache is False
    assert config.temperature == 0.5
    assert config.calibrate is False
    assert config.calibration_context == "EMPTY"
    # Relative save_to resolves against the config file's directory.
    assert config.resolve_save_to() == str(tmp_path / "models/acme")
    assert config.local_dir == tmp_path / "models/acme"


def test_save_to_for_other_model_uses_sibling_directory(tmp_path):
    path = _write_config(tmp_path, 'model = "acme/model"\nsave_to = "models/acme"\n')
    config = load_config(path)

    assert config.save_to_for("acme/model") == str(tmp_path / "models/acme")
    assert config.save_to_for("other/model") == str(tmp_path / "models/other--model")


def test_empty_save_to_disables_disk_cache(tmp_path):
    path = _write_config(tmp_path, 'save_to = ""\n')
    config = load_config(path)
    assert config.save_to is None
    assert config.resolve_save_to() is None
    assert config.local_dir is None
    assert config.save_to_for("other/model") is None


@pytest.mark.parametrize(
    "body",
    [
        "temperature = 0\n",
        "temperature = -1\n",
        'quantize = "nope"\n',
        'device = "tpu"\n',
        "max_image_pixels = 0\n",
        "max_image_pixels = -1\n",
    ],
)
def test_invalid_config_raises_value_error(tmp_path, body):
    path = _write_config(tmp_path, body)
    # pydantic.ValidationError subclasses ValueError, which the interfaces catch.
    with pytest.raises(ValueError):
        load_config(path)


def test_defaults_when_keys_missing(tmp_path):
    path = _write_config(tmp_path, "")
    config = load_config(path)
    assert isinstance(config, Config)
    assert config.device == "auto"
    assert config.quantize == "auto"
    assert config.temperature == 1.0


def test_env_overrides_file(tmp_path, monkeypatch):
    path = _write_config(tmp_path, 'model = "file/model"\ndevice = "cpu"\n')
    monkeypatch.setenv("ZERO_SHOT_MODEL", "env/model")
    monkeypatch.setenv("ZERO_SHOT_QUANTIZE", "bf16")

    config = load_config(path)
    assert config.model == "env/model"
    assert config.quantize == "bf16"
    # device is not overridden by env here, so the file wins.
    assert config.device == "cpu"


def test_config_normalizes_quantize_and_device(tmp_path):
    path = _write_config(tmp_path, 'quantize = "INT8"\ndevice = "CPU"\n')
    config = load_config(path)
    assert config.quantize == "int8"
    assert config.device == "cpu"


def test_max_image_pixels_parses_and_defaults(tmp_path):
    path = _write_config(tmp_path, "max_image_pixels = 401408\n")
    assert load_config(path).max_image_pixels == 401408
    assert load_config(_write_config(tmp_path, "")).max_image_pixels is None


def test_default_config_path_honors_env_override(tmp_path, monkeypatch):
    path = _write_config(tmp_path, 'model = "env-path/model"\n')
    monkeypatch.setenv("ZERO_SHOT_CONFIG", str(path))
    # No explicit path: load_config must resolve $ZERO_SHOT_CONFIG.
    assert load_config().model == "env-path/model"


def test_load_config_ignores_directory_at_path(tmp_path):
    """A directory at the config path (Docker compose bind-mount of a
    missing host file) must be skipped, not read with ``read_text()``.

    Regression: a host-side ``config.toml`` that doesn't exist makes Docker
    create a directory at the mount point. ``Path.exists()`` returns True for
    directories, so the old code crashed with ``IsADirectoryError`` before
    the server could even start.
    """
    config_path = tmp_path / "config.toml"
    config_path.mkdir()  # directory, not file
    config = load_config(config_path)
    # Falls back to defaults rather than crashing; the directory's parent is
    # the project root, so relative save_to paths still resolve from there.
    assert isinstance(config, Config)
    assert config.model  # populated from defaults
    assert config.device == "auto"
    assert config.base_dir == tmp_path


def test_load_config_explicit_missing_file_uses_defaults(tmp_path):
    """The missing-file branch must still work after switching to is_file()."""
    config = load_config(tmp_path / "does-not-exist.toml")
    assert config.device == "auto"
