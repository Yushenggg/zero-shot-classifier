# zero-shot

A local zero-shot classifier using next-token logprobs. Default is a vision-language model (Qwen3-VL-4B-Instruct); CPU preset is SmolVLM-500M-Instruct.

## Layout
- `zero_shot/` — package (`server.py`, `scorer.py`, `classifier.py`, `config.py`)
- `config.toml` — default (GPU, vision-language)
- `config.cpu.toml` — CPU preset (int8, ~500 MB RAM)
- `examples/` — request bodies (`payouts_*.json` is the bias-table reference)
- `docs/index.html` — blog-style design write-up

## Commands
- `uv sync` — install deps (CUDA build of PyTorch by default)
- `uv pip install --reinstall -r requirements-cpu.txt` — switch to CPU-only torch
- `.venv/bin/zero-shot-serve` — start server on `127.0.0.1:8000`
- `.venv/bin/zero-shot-serve --cpu-low` — SmolVLM-500M on CPU from `config.cpu.toml`
- `.venv/bin/zero-shot-serve --config <path>` — serve from an arbitrary config

## Conventions
- Model weights go in `./models/<short-name>/`; loaded with `local_files_only=True` after the first download
- `config.toml` is the GPU/quality default; `config.cpu.toml` is the universal CPU fallback
- Vision-language is the default mode; the web UI auto-disables Image mode for text-only checkpoints
- `multimodal` detection in `zero_shot/scorer.py` trusts the processor (`processor.image_processor is not None`); do not reintroduce a per-architecture vision-tower attribute check
- The bias table in `README.md` must be re-measured if `config.toml` (default model) changes — see `.opencode/skills/model-advisor/SKILL.md` for the measurement recipe

## Tests
None yet. Smoke-test pattern: start server with a config, poll `/api/health` for `model_loaded: true` with a 60–120 s timeout, dump the last 30 lines of the uvicorn log on failure.
