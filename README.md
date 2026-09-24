# zero-shot

A local, from-scratch recreation of the [Jev](https://typesafe.ai/) decision
API: a server that turns an arbitrary "state" plus a set of choices into a
decision with real probabilities, no training required. Use any model as is.

Read more about the thought process behind it
[here](https://yushenggg.github.io/zero-shot-classifier/).

## Setup

```bash
uv sync
```

## Configuration

One config ships ready-to-run; one is a blank template.

| Config | Model | Hardware | Notes |
|---|---|---|---|
| `config.cpu.toml` *(shipped)* | `HuggingFaceTB/SmolVLM-500M-Instruct` | CPU | Idefics3 VLM, ~500M params, fp32 → ~2 GB RAM. Lower quality, runs anywhere, no `trust_remote_code` needed. |
| `config.toml.example` *(shipped)* | — | — | Blank template. Copy to `config.toml` and edit, or let the model-advisor skill generate one for your GPU. |

`config.toml` is your machine profile — it is **gitignored** so per-host
customizations (model id, GPU key, `max_image_pixels`, etc.) never leak back
to the repo. To set one up:

```bash
cp config.toml.example config.toml        # then edit the keys, or...
                                          # ...use the model-advisor skill:
                                          #   .opencode/skills/model-advisor/SKILL.md
```

The skill detects your VRAM, recommends a model that fits, downloads the
weights, and writes `config.toml` with the right `max_image_pixels` cap.
Every key is optional; anything you leave unset falls back to the built-in
default.

Pass a config to the CLI or set `ZERO_SHOT_CONFIG`:

```bash
zero-shot-serve                          # config.toml (your machine profile)
zero-shot-serve --cpu-low                # config.cpu.toml (SmolVLM-500M on CPU)
zero-shot-serve --config config.cpu.toml # explicit
```

Available keys:

```toml
model = "Qwen/Qwen3-VL-4B-Instruct"      # any native-transformers VLM
save_to = "models/qwen3-vl-4b-instruct"  # loaded from disk after first download
device = "auto"                          # auto | cpu | gpu
gpu = "rtx_5060_ti"                      # which GPU when device = "gpu"
quantize = "auto"                        # auto | bf16 | fp32 | int8
max_image_pixels = ""                    # omit / "" = model native; cap helps on small GPUs
kv_cache = true                          # reuse one KV cache for the shared prompt
temperature = 1.0                        # option softmax; >1 less certain, <1 sharper
calibrate = true                         # subtract content-free option priors
calibration_context = "N/A"              # the content-free context used for that
```

Paths in `save_to` are resolved from the project directory. Once populated the
model is loaded with `local_files_only=True` (no network). Overridable with
`$ZERO_SHOT_CONFIG`, `$ZERO_SHOT_MODEL`, `$ZERO_SHOT_SAVE_TO`, `$ZERO_SHOT_DEVICE`,
`$ZERO_SHOT_GPU`, `$ZERO_SHOT_MAX_IMAGE_PIXELS`, or the CLI flags `--config`,
`--model-id`, `--save-to`, `--device`, `--gpu`.

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

**Qwen3-VL-4B-Instruct** (your `config.toml`, GPU, bf16):

| option | calibrated logprob, order A (billing, technical, sales) | order B (sales, technical, billing) | Δ |
|---|---|---|---|
| billing  | +0.609 | +2.660 | +2.05 |
| technical | -4.171 | -1.118 | +3.05 |
| sales    | -6.743 | -8.635 | -1.89 |

Winner is unchanged (`billing`) and confidence stays high (0.98 → 0.96). The
absolute deltas are smaller than the previous text-only Qwen3-4B baseline (which
saw a 4.36 swing on `technical`); VL calibration appears less order-sensitive
in practice.

**SmolVLM-500M-Instruct** (`config.cpu.toml`, CPU, fp32):

| option | calibrated logprob, order A | order B | Δ |
|---|---|---|---|
| billing  | -0.225 | -0.126 | +0.10 |
| technical | -0.078 | +0.678 | **+0.76** |
| sales    | +1.034 | +0.038 | -1.00 |

The winner *does* change between orders (`sales` vs `technical`), and the model
is unsure either way (confidence 0.62 vs 0.51) — the small model is genuinely
near the boundary and order tips it. Take note if you are near the decision
boundary, especially on CPU.

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
**SmolVLM-500M-Instruct** with `quantize = "fp32"`: ~500M params, ~2 GB RAM,
runs on a laptop. fp32 keeps the KV-cached scores identical to the exact path
and sidesteps emulated bf16 on mobile/consumer Intel CPUs. Quality is lower than
Qwen3-VL-4B (the winner can flip between criteria orderings); prefer a GPU
profile whenever a GPU is available.

### CPU precision (`quantize`)

`quantize = "auto"` uses **bf16** on CUDA and **fp32** on CPU. CPU stays on fp32
even where the hardware supports bf16 (`AVX512_BF16`/`AMX_BF16`, e.g. AMD Zen
4/5): bf16 drifts slightly from the exact path under KV-cache reuse, whereas
fp32 keeps the cached and exact scores identical, and the extra memory is small.
This is also why the CPU presets no longer quantize — it avoids PyTorch's
*emulated* bf16 on Intel consumer CPUs since 12th gen (Alder Lake on), where
AVX-512 is fused off and bf16 is several times slower than fp32. Set
`quantize = "bf16"` explicitly for CPU speed, or `"fp32"` to force it. `"int8"`
applies dynamic quantization on CPU — faster on AVX2+VNNI and ~4x less RAM, but
it perturbs logprobs, so near-tie predictions can shift. CUDA is never
quantized. Overridable with `--quantize` or `$ZERO_SHOT_QUANTIZE`.

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
laptop with no GPU, where a large GPU profile would be slow or OOM. The
server preloads the configured model at startup and logs the device, so
you can confirm whether it is running on CPU or GPU.

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

`model` is optional and defaults to the `model` value in your `config.toml`; if a
different model is given it is loaded from (or downloaded into) a sibling
directory under `models/`.
`GET /api/health` reports the configured model, device, and `multimodal`
(`true`/`false` once the model is loaded, `null` before) so clients know whether
image input is available.

### Image classification

The same questions can be grounded in an image, using the identical logit
extraction (full-vocabulary `log P(token)` + EOS, corrected softmax) — the image is
just prepended to the prompt through the model's chat template. Both the
template (`config.toml.example`) and the CPU preset (`config.cpu.toml`) use
vision-language checkpoints, so text and image requests work without
swapping configs. For a reasoning variant, set
`model = "Qwen/Qwen3-VL-4B-Thinking"`.

CLI (`--image`/`-i` takes a file, or `-` for raw bytes on stdin):

```bash
.venv/bin/zero-shot \
  -q examples/color_question.json -s examples/color_state.json -i photo.jpg
```

HTTP API: `POST /v1/classify/image` (also at the TypeSafe-compatible
`POST /v1/systemone/image`) accepts `multipart/form-data` with a `file` image
stream and JSON `questions`/`state` strings. Text requests use
`/v1/classify/text`; both go through the exact same scoring path:

```bash
curl -F file=@photo.jpg \
     -F 'questions={"color": {"type": "choice", "instructions": "What color is the fruit?", "criteria": {"red": "", "green": "", "yellow": ""}}}' \
     http://127.0.0.1:8000/v1/classify/image
```

Uploads up to 64 MB are accepted. Anything above 16 MB is automatically
downscaled and re-encoded as JPEG until it fits, so you don't have to resize by
hand (`ZERO_SHOT_MAX_IMAGE_MB` sets the target, `ZERO_SHOT_MAX_UPLOAD_MB` the hard
ceiling that is rejected with HTTP 413). The model processor then downscales to
its own resolution.

In the web UI, switch the **Text / Image** toggle to Image, attach a file, and
Classify. The State editor is replaced by an optional **Add text context**
field — attach a `state` JSON there only when the image alone is not enough
(the API accepts `state` alongside the file either way). The Text/Image toggle is
automatically disabled when the loaded model is text-only. The image + prompt are
prefilled once and the KV cache is reused across options. The calibration pass
deliberately runs text-only — the image *is* the content, so calibrating against
it would cancel the signal. Passing `--image` to a text-only model fails with
a clear error.

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

The GPU service (CUDA + your `config.toml`) is **commented out** in
`docker-compose.yml`: building it downloads the CUDA PyTorch wheels plus
several GB of NVIDIA/CUDA packages. The image does **not** ship a
`config.toml` (it is per-host and gitignored), so generate one first —
`cp config.toml.example config.toml` then edit, or let the
[model-advisor skill](.opencode/skills/model-advisor/SKILL.md) generate
one for you — before uncommenting. Then:

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
