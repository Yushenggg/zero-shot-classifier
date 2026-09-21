# zero-shot

Inspired by [jev.ai](https://jev.ai) and its ability to turn an arbitrary "state"
plus a set of choices into a decision with real probabilities, no training
required. See [docs/context.md](docs/context.md) for the design narrative and the
reasoning behind the current shape of the project.

## How it works (roughly)

1. Build a prompt from the state (arbitrary JSON context) and the task
   instructions, ending right at the decision point.
2. Run a forward pass with a local Hugging Face model (Qwen3-4B-Instruct-2507 by
   default) and read the next-token distribution from the full-vocabulary logits.
3. For each option, score its exact continuation token-by-token, plus the EOS
   token, giving `log P(option)`.
4. Softmax those scores into probabilities and pick the highest.

Including EOS is what lets a longer but more complete answer win over a short one.

## Question types

`question` is a map of named, typed questions (all three types can be mixed in one
call):

- **`choice`** — pick one option. `criteria` is a map of `option → rubric`.
- **`noul`** — yes/no. Returns the probability of "yes" (`noul`). `criteria`
  (`{"true": ..., "false": ...}`) is optional.
- **`score`** — rate on an ordered scale. `criteria` is an ordered array of 2–10
  level descriptions. Returns `score` (a probability-weighted value that can land
  between levels), the per-level `probabilities`, and a `legend`.

Candidate keys are scored directly; they are **not** enumerated in the prompt (only
the `score` rating scale is shown). `instructions` and `state` may be plain strings
or structured objects/arrays (rendered as JSON in the prompt). `choice` and `score`
answers also include a `confidence` (the sum of squared probabilities).

Examples: [examples/payouts_questions.json](examples/payouts_questions.json) (all
three types) with [examples/payouts_state.json](examples/payouts_state.json), and
the apple example in [examples/color_question.json](examples/color_question.json).

## Setup

```bash
uv sync
```

## Configuration

Model settings live in `config.toml`:

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"
save_to = "models/qwen3-4b-instruct-2507"   # loaded from disk after first download
device = "auto"                       # auto | cpu | gpu
gpu = "rtx_5060_ti"                   # which GPU when device = "gpu"
kv_cache = true                       # reuse one KV cache for the shared prompt
temperature = 1.0                     # option softmax; >1 less certain, <1 sharper
calibrate = true                      # subtract content-free option priors
calibration_context = "N/A"           # the content-free context used for that
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

### Calibration

Because the options are compared by their token log-probabilities, the model's
surface-form priors leak in (e.g. `sales` or the digit `1` are simply more likely
tokens, so they win more often). With `calibrate = true` (default) each option is
also scored against a content-free context (`calibration_context`, default `"N/A"`)
and that prior is subtracted before the softmax:

```
score(option) = log P(option | context, cue) − log P(option | "N/A", cue)
```

This removes digital/yes-no/token-length bias; it roughly doubles scoring time
(two passes). Disable with `--no-calibrate` or `calibrate = false`. It does not
change the chosen option's *order* bias (we don't list options) nor correct
length-preference between differently long keys.

## CPU vs GPU

`uv sync` installs the **CUDA** build of PyTorch by default (CUDA 13, supports the
RTX 5060 Ti / Blackwell sm_120). On a machine without a GPU, switch to the CPU-only
build:

```bash
uv pip install --reinstall -r requirements-cpu.txt   # CPU fallback
uv pip install --reinstall -r requirements-gpu.txt   # restore the CUDA build
```

`device = "auto"` uses the GPU when available and falls back to CPU otherwise. Set
`device = "gpu"` to require the GPU (validated against `gpu`, currently
`rtx_5060_ti`), or `device = "cpu"` to force CPU.

## Usage

```bash
# device comes from config.toml
.venv/bin/zero-shot -q examples/color_question.json -s examples/color_state.json

# choice + noul + score in one call
.venv/bin/zero-shot -q examples/payouts_questions.json -s examples/payouts_state.json

# force the exact (non-KV) path
.venv/bin/zero-shot -q examples/color_question.json -s examples/color_state.json --no-kv-cache

.venv/bin/zero-shot-serve   # web UI at http://127.0.0.1:8000
```

The server preloads the configured model at startup and logs the device, so you can
confirm whether it is running on CPU or GPU.

### HTTP API

`POST /api/classify` (also mounted at the TypeSafe-compatible `POST /v1/systemone`)
accepts a TypeSafe-style body (all three question types can be mixed):

```json
{
  "state": { "input": "Help! My payouts have been failing for 3 days." },
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
    "frustration": { "type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"] },
    "department": { "type": "choice", "instructions": "Which team should handle this?", "criteria": { "billing": "Payments", "technical": "Bugs", "sales": "Pricing" } }
  }
}
```

Returns `{"model": ..., "answers": { "<id>": <answer> }, "results": [ ... ], "usage":
{"input_tokens": ..., "output_tokens": ...}}`, with answer shapes matching TypeSafe
(`noul`, `choice` + `probabilities` + `confidence`, `score` + `legend` +
`probabilities` + `confidence`).

`model` is optional and defaults to `config.toml`; if a different model is given it
is loaded from (or downloaded into) a sibling directory under `models/`.
`GET /api/health` reports the configured model and device.

## License

MIT. See [LICENSE](LICENSE).
