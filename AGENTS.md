# zero-shot

Local zero-shot classifier that scores options by exact next-token
log-probabilities (including EOS) from a local `transformers` model.
Vision-language is the default; text and image requests share one scoring path.

## Layout
- `zero_shot/core/` — interface-independent logic (`classifier.py`, `scorer.py`, `config.py`, `models.py`, `image_utils.py`)
- `zero_shot/interfaces/cli/` — `zero-shot` command
- `zero_shot/interfaces/server/` — `zero-shot-serve` command, FastAPI app (`app.py`), HTTP schemas (`schemas.py`), `static/` UI
- `config.toml.example` — blank template (tracked); copy to `config.toml`
- `config.toml` — your machine profile (gitignored, per-host)
- `config.cpu.toml` — CPU preset (SmolVLM-500M-Instruct, ships ready-to-run)
- `examples/` — request bodies
- `docs/index.html` — design write-up
- `.opencode/skills/model-advisor/SKILL.md` — model selection + bias-table measurement recipe

## Commands
- `uv sync` — install deps (CUDA build of PyTorch by default)
- `uv pip install --reinstall -r requirements-cpu.txt` — switch to CPU-only PyTorch
- `.venv/bin/zero-shot-serve` — serve the UI/API on `127.0.0.1:8000`
- `.venv/bin/zero-shot-serve --cpu-low` — serve the CPU preset
- `.venv/bin/zero-shot-serve --config <path>` — serve an arbitrary config
- `.venv/bin/zero-shot -q <questions.json> -s <state.json> [-i <image>]` — classify from the CLI
- `.venv/bin/python -m pytest` — run the regression suite (`tests/core`, `tests/interfaces`); no model is loaded
- `uvx ruff@0.16.8 check` — lint the tree (config in `pyproject.toml`; CI pins the same version and runs on PRs and pushes to `main`)

## Conventions
- Model weights live in `./models/<short-name>/` and are loaded with
  `local_files_only=True` after the first download.
- `config.toml` is your machine profile (gitignored); copy from
  `config.toml.example` or have the model-advisor skill generate one.
  `config.cpu.toml` is the universal CPU fallback and ships ready-to-run.
- Vision-language is the default mode; the UI disables Image mode for text-only checkpoints.

## Tests
`.venv/bin/python -m pytest` runs the regression suite in `tests/` (mirrors the
package: `tests/core`, `tests/interfaces`). A `FakeScorer` replaces the real
model, so the suite is deterministic, offline, and fast — it covers prompt
building, calibration/softmax, config + pydantic validation, the scorer cache,
image utils, the CLI, and the HTTP API via `TestClient`.

For a real end-to-end check, start the server and poll `/api/health` for
`model_loaded: true` (60–120 s), dumping the last 30 lines of the uvicorn log on
failure.
