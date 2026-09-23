# zero-shot

Local zero-shot classifier that scores options by exact next-token
log-probabilities (including EOS) from a local `transformers` model.
Vision-language is the default; text and image requests share one scoring path.

## Layout
- `zero_shot/core/` — interface-independent logic (`classifier.py`, `scorer.py`, `config.py`, `models.py`, `image_utils.py`)
- `zero_shot/interfaces/cli/` — `zero-shot` command
- `zero_shot/interfaces/server/` — `zero-shot-serve` command, FastAPI app (`app.py`), HTTP schemas (`schemas.py`), `static/` UI
- `config.toml` — default config (GPU, vision-language)
- `config.cpu.toml` — CPU preset (SmolVLM-500M-Instruct)
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
- `uvx ruff@0.16.8 check` — lint the tree (config in `pyproject.toml`; CI pins the same version and runs on PRs and pushes to `main`)

## Conventions
- Model weights live in `./models/<short-name>/` and are loaded with
  `local_files_only=True` after the first download.
- `config.toml` is the GPU/quality default; `config.cpu.toml` is the universal CPU fallback.
- Vision-language is the default mode; the UI disables Image mode for text-only checkpoints.

## Tests
None. Smoke-test: start the server, poll `/api/health` for `model_loaded: true`
(60–120 s), and dump the last 30 lines of the uvicorn log on failure.
