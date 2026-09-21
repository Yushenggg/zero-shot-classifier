# zero-shot

Inspired by [jev.ai](https://jev.ai) and its ability to turn an arbitrary "state"
plus a set of choices into a decision with real probabilities, no training
required. See [docs/context.md](docs/context.md) for the design narrative and the
reasoning behind the current shape of the project.

## How it works (roughly)

1. Build a prompt from the state (arbitrary JSON context), the task instructions,
   and the choice criteria, ending right at the decision point.
2. Run a forward pass with a local Hugging Face model (Qwen3-4B-Instruct-2507 by
   default) and read the next-token distribution from the full-vocabulary logits.
3. For each option, score its exact continuation token-by-token, plus the EOS
   token, giving `log P(option)`.
4. Softmax those scores into probabilities and pick the highest.

Including EOS is what lets a longer but more complete answer win over a short one.

## Setup

```bash
uv sync
```

## Configuration

Model settings live in `config.toml`:

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"
save_to = "models/qwen3-4b-instruct-2507"   # loaded from disk after first download
device = "gpu"                        # "cpu" or "gpu"
gpu = "rtx_5060_ti"                   # which GPU when device = "gpu"
kv_cache = true                       # reuse one KV cache for the shared prompt
```

Paths in `save_to` are resolved from the project directory. Once populated the
model is loaded with `local_files_only=True` (no network). Overridable with
`$ZERO_SHOT_CONFIG`, `$ZERO_SHOT_MODEL`, `$ZERO_SHOT_SAVE_TO`, `$ZERO_SHOT_DEVICE`,
`$ZERO_SHOT_GPU`, or the CLI flags `--config`, `--model-id`, `--save-to`,
`--device`, `--gpu`.

### KV cache

`kv_cache = true` (default) prefills the shared prompt once and reuses its KV cache
across options (~3-4x faster). Because it uses the cached attention path it is
slightly approximate: rare near-tie tokens can shift, which can occasionally change
a prediction. If a model's cache cannot be reused (no `crop` support, no cache
returned, forward rejects `past_key_values`), the scorer logs a warning and
automatically falls back to the exact path. Use `--no-kv-cache` (CLI/API) to force
exact scoring.

## CPU vs GPU

The project installs the small CPU-only PyTorch wheel by default. Two files switch
builds (use `--reinstall`, because `torch==2.14.0` is otherwise considered already
satisfied by the installed build):

```bash
uv pip install --reinstall -r requirements-cpu.txt   # CPU-only (default)
uv pip install --reinstall -r requirements-gpu.txt   # CUDA 13, supports RTX 5060 Ti
```

Currently supported GPU: `rtx_5060_ti` (Blackwell, sm_120, CUDA 13). With GPU mode,
`resolve_device` checks that CUDA is available and that
`torch.cuda.get_device_name()` matches the configured GPU.

After installing the GPU build, run without `uv sync` reverting it — use
`.venv/bin/...` or `uv run --no-sync ...`.

## Usage

```bash
# device comes from config.toml
.venv/bin/zero-shot -q examples/color_question.json -s examples/color_state.json

# force the exact (non-KV) path
.venv/bin/zero-shot -q examples/color_question.json -s examples/color_state.json --no-kv-cache

.venv/bin/zero-shot-serve   # web UI at http://127.0.0.1:8000
```

The server preloads the configured model at startup and logs the device, so you can
confirm whether it is running on CPU or GPU.

## License

MIT. See [LICENSE](LICENSE).
