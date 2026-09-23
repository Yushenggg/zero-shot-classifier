# zero-shot

A local, from-scratch recreation of the [Jev](https://typesafe.ai/) decision
API: a server that turns an arbitrary "state" plus a set of choices into a
decision with real probabilities, no training required.

Read more about the thought process behind it
[here](https://yushenggg.github.io/zero-shot-classifier/).

## Setup

```bash
uv sync
```

## Configuration

Two configs ship by default; pick whichever fits the host.

| Config | Model | Hardware | Notes |
|---|---|---|---|
| `config.toml` *(default)* | `Qwen/Qwen3-VL-4B-Instruct` | GPU (RTX 5060 Ti) | Vision-language: serves text and image from one checkpoint. ~8 GB bf16. |
| `config.cpu.toml` | `HuggingFaceTB/SmolVLM-500M-Instruct` | CPU | Idefics3 VLM, ~500M params, int8 → ~500 MB RAM. Lower quality, runs anywhere, no `trust_remote_code` needed. |

Pass a config to the CLI or set `ZERO_SHOT_CONFIG`:

```bash
zero-shot-serve                          # config.toml (Qwen3-VL-4B on GPU)
zero-shot-serve --cpu-low                # config.cpu.toml (SmolVLM-500M on CPU)
zero-shot-serve --config config.cpu.toml # explicit
```

`config.toml` in detail:

```toml
model = "Qwen/Qwen3-VL-4B-Instruct"
save_to = "models/qwen3-vl-4b-instruct"  # loaded from disk after first download
device = "auto"                          # auto | cpu | gpu
gpu = "rtx_5060_ti"                      # which GPU when device = "gpu"
quantize = "auto"                        # auto | bf16 | fp32 | int8
kv_cache = true                          # reuse one KV cache for the shared prompt
temperature = 1.0                        # option softmax; >1 less certain, <1 sharper
calibrate = true                         # subtract content-free option priors
calibration_context = "N/A"              # the content-free context used for that
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
correct length-preference between differently long keys.

### Criteria-order bias

For `choice` questions the prompt includes a `# CRITERIA` block listing each key
with its description, so the model sees what each option means before it scores
the key as a continuation. The keys themselves are still scored independently,
so there is no positional bias *between options* — but the order in which the
criteria are listed can still prime the model.

Measured on the payouts example (state: "Help! My payouts have been failing for 3
days.", `criteria` for `department`):

**Qwen3-VL-4B-Instruct** (`config.toml`, GPU, bf16):

| option | calibrated logprob, order A (billing, technical, sales) | order B (sales, technical, billing) | Δ |
|---|---|---|---|
| billing  | +0.609 | +2.660 | +2.05 |
| technical | -4.171 | -1.118 | +3.05 |
| sales    | -6.743 | -8.635 | -1.89 |

Winner is unchanged (`billing`) and confidence stays high (0.98 → 0.96). The
absolute deltas are smaller than the previous text-only Qwen3-4B baseline (which
saw a 4.36 swing on `technical`); VL calibration appears less order-sensitive
in practice.

**SmolVLM-500M-Instruct** (`config.cpu.toml`, CPU, int8):

| option | calibrated logprob, order A | order B | Δ |
|---|---|---|---|
| billing  | -0.068 | -0.029 | +0.04 |
| technical | -0.034 | +0.389 | **+0.42** |
| sales    | +0.411 | -0.044 | -0.46 |

The winner *does* change between orders (`sales` vs `technical`), and confidence
is much lower (≈0.35) — the small model is genuinely unsure and order tips it.
Take note if you are near the decision boundary, especially on CPU.

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

The `config.cpu.toml` preset forces `device = "cpu"` and uses
**SmolVLM-500M-Instruct** with `quantize = "int8"`: ~500M params, ~500 MB RAM,
runs on a laptop. Quality is lower than Qwen3-VL-4B (the winner can flip
between criteria orderings); prefer `config.toml` whenever a GPU is available.

### CPU precision (`quantize`)

`quantize = "auto"` uses **bf16** where it is hardware-accelerated — CUDA, or a
CPU advertising `AVX512_BF16`/`AMX_BF16` (e.g. AMD Zen 4/5) — and **fp32**
otherwise. This matters on Intel consumer CPUs since 12th gen (Alder Lake on),
where AVX-512 is fused off: PyTorch then *emulates* bf16, which is several times
slower than fp32 (the SmolLM2 CPU image can take seconds per request there).
`quantize = "fp32"` or `"bf16"` forces a dtype, and `"int8"` applies dynamic
quantization on CPU — faster on AVX2+VNNI and ~4x less RAM, but it perturbs
logprobs, so near-tie predictions can shift. CUDA is never quantized. Overridable
with `--quantize` or `$ZERO_SHOT_QUANTIZE`.

## Usage

```bash
# device comes from config.toml
.venv/bin/zero-shot -q examples/color_question.json -s examples/color_state.json

# choice + noul + score in one call
.venv/bin/zero-shot -q examples/payouts_questions.json -s examples/payouts_state.json

# force the exact (non-KV) path
.venv/bin/zero-shot -q examples/color_question.json -s examples/color_state.json --no-kv-cache

.venv/bin/zero-shot-serve            # web UI at http://127.0.0.1:8000
.venv/bin/zero-shot-serve --cpu-low  # CPU model (config.cpu.toml, SmolVLM-500M)
```

`--cpu-low` serves SmolVLM-500M on CPU from `config.cpu.toml` — handy on a
laptop with no GPU, where the default Qwen3-VL-4B would be slow. The server preloads
the configured model at startup and logs the device, so you can confirm whether it
is running on CPU or GPU.

### HTTP API

`POST /v1/classify/text` (also available at the TypeSafe-compatible
`POST /v1/systemone`) accepts a TypeSafe-style body (all three question types can be
mixed):

```json
{
  "state": { "input": "Help! My payouts have been failing for 3 days." },
  "model": "Qwen/Qwen3-VL-4B-Instruct",
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
`GET /api/health` reports the configured model, device, and `multimodal`
(`true`/`false` once the model is loaded, `null` before) so clients know whether
image input is available.

### Image classification

The same questions can be grounded in an image, using the identical logit
extraction (full-vocabulary `log P(token)` + EOS, corrected softmax) — the image is
just prepended to the prompt through the model's chat template. The default
`config.toml` already points at a vision-language checkpoint
(`Qwen/Qwen3-VL-4B-Instruct`, ~8 GB in bf16), so both text and image requests
work without swapping configs. For a reasoning variant, set
`model = "Qwen/Qwen3-VL-4B-Thinking"`.

CLI (`--image`/`-i` takes a file, or `-` for raw bytes on stdin):

```bash
.venv/bin/zero-shot \
  -q examples/color_question.json -s examples/color_state.json -i photo.jpg
```

HTTP API: `POST /v1/classify/image` accepts `multipart/form-data` with a `file`
image stream and JSON `questions`/`state` strings. Text requests use
`/v1/classify/text`; both go through the exact same scoring path:

```bash
curl -F file=@photo.jpg \
     -F 'questions={"color": {"type": "choice", "instructions": "What color is the fruit?", "criteria": {"red": "", "green": "", "yellow": ""}}}' \
     http://127.0.0.1:8000/v1/classify/image
```

In the web UI, switch the **Text / Image** toggle to Image, attach a file, and
Classify — the UI posts to `/v1/classify/image` (the State editor is hidden,
since the image is the context). The Text/Image toggle is automatically disabled
when the loaded model is text-only. The image + prompt are prefilled once and
the KV cache is reused across options, so the calibration pass shares the same
image prefill. Passing `--image` to a text-only model fails with a clear error.

### Web UI

Besides the CLI and the HTTP API, the server ships a browser UI: build a request
from the example templates (or paste your own JSON), run it, and inspect the
result — per-option probabilities, log-probabilities, and the token-level tables.
Start the server and open http://127.0.0.1:8000:

```bash
.venv/bin/zero-shot-serve
```

![The local web UI after a mixed noul + score + choice call on the payouts example](docs/img/ui-mixed-result.png)

## Docker

A `Dockerfile` and `docker-compose.yml` are included. The standard service runs
**CPU + SmolVLM-500M-Instruct** and serves the same web UI:

```bash
docker compose up --build        # http://127.0.0.1:8000
```

Model weights are bind-mounted from `./models`, so they persist across rebuilds and
are never re-downloaded. The service mounts `config.cpu.toml` read-only and points
`ZERO_SHOT_CONFIG` at it, so config edits apply without a rebuild.

The CPU image is the portable option: it has no CUDA, driver, or toolkit
dependencies, so it should build and run on essentially any machine Docker supports
(amd64 or arm64, Linux or Docker Desktop on macOS/Windows) with no GPU and no model
download. The trade-off is reasoning quality — SmolVLM-500M is noticeably below
Qwen3-VL-4B on edge cases (see the criteria-order bias table; the winner can
flip between orderings on CPU).

### Prebuilt image

A prebuilt CPU image is published on Docker Hub, so you can skip the build
entirely:

```bash
docker pull yushenggg/zero-shot:cpu
docker run -p 8000:8000 yushenggg/zero-shot:cpu    # http://127.0.0.1:8000
```

Mount `./models` and `config.cpu.toml` the same way `docker-compose.yml` does if
you want the weights to persist across container recreation.

### GPU

The GPU service (CUDA + `config.toml`, i.e. Qwen3-VL-4B) is **commented out** in
`docker-compose.yml`: building it downloads the CUDA PyTorch wheels plus several GB
of NVIDIA/CUDA packages. To use it, uncomment the `zero-shot-gpu` service and run:

```bash
docker compose --profile gpu up --build zero-shot-gpu   # http://127.0.0.1:8001
```

This requires the NVIDIA Container Toolkit on the host (`nvidia-ctk runtime configure
--runtime=docker`, then restart Docker) and a driver supporting CUDA 13 / Blackwell
`sm_120`.

Inside the container the server binds `0.0.0.0` (set by the image); override with
`ZERO_SHOT_HOST` / `ZERO_SHOT_PORT`. Outside Docker it still defaults to
`127.0.0.1:8000`.

## License

MIT. See [LICENSE](LICENSE).
