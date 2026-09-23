# zero-shot

Local zero-shot classifier that scores options by exact next-token
log-probabilities (including EOS) from a local `transformers` model.
Vision-language is the default; text and image requests share one scoring path.

## Layout
- `zero_shot/` — package (`server.py`, `scorer.py`, `classifier.py`, `config.py`, `cli.py`, `static/`)
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

## Conventions
- Model weights live in `./models/<short-name>/` and are loaded with
  `local_files_only=True` after the first download.
- `config.toml` is the GPU/quality default; `config.cpu.toml` is the universal CPU fallback.
- Vision-language is the default mode; the UI disables Image mode for text-only checkpoints.

## Tests
None. Smoke-test: start the server, poll `/api/health` for `model_loaded: true`
(60–120 s), and dump the last 30 lines of the uvicorn log on failure.
