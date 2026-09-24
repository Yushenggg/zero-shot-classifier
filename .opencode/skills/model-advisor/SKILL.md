---
name: model-advisor
description: Use when the user asks which model to run for this project, wants to switch the default model, or asks what hardware they need. Primarily for GPU hosts — detects VRAM, websearches for current vision-language checkpoints, recommends one with alternatives, and — after the user picks — downloads the weights and updates the relevant config. CPU hosts get pointed at the existing config.cpu.toml (SmolVLM-500M-Instruct, fp32) with brief tuning guidance. Default-mode is vision-language unless the user explicitly asks for text-only.
---

# Model advisor

The project ships `config.cpu.toml` (universal CPU fallback) and a blank
`config.toml.example`. `config.toml` is gitignored — every user generates
their own machine profile, and this skill is how you do it: detection →
search → recommendation → download → configure (`config.toml` written
from `config.toml.example` with the right model, `gpu` key, and
`max_image_pixels` cap).

Do not skip steps. Do not auto-download before the user picks.

## 1. Detect host specs

Run these in parallel:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>&1
echo "---"
uname -m
echo "---"
lscpu | grep -E "Model name|^CPU\(s\):" | head -2
echo "---"
free -h | head -2
echo "---"
df -h . | head -2
```

If `nvidia-smi` errors out (no GPU, no driver, no NVIDIA toolkit) treat the
host as CPU-only. Note the architecture (`x86_64` / `arm64`); most popular VLMs
ship arm64 wheels but a few do not.

For CPU hosts, also grab SIMD capability (needed in Step 4b to pick
quantization):

```bash
grep -oE 'avx2|avx512f|avx512_bf16|avx_vnni|amx_tile' /proc/cpuinfo | sort -u
```

## 2. Confirm the use case

If not already stated, ask one question:

> "Vision-language (text + image) or text-only? — default is vision"

Vision is the default. Only switch to text-only if the user explicitly asks.

## 3. Search

Websearch for recent candidates that match the hardware tier. Useful queries:

- "best small vision-language model {current_year} CPU"
- "vision language model {vram_size}GB benchmark {current_year}"
- "transformers native VLM architecture list" — to filter out models that
  require `trust_remote_code`

Hard requirement: **the model must have native transformers support** (no
`auto_map` in `config.json`). Models like `vikhyatk/moondream2` whose config
points at a custom class must be excluded — they force arbitrary code
execution at load time.

## 4. Recommend — GPU path

GPU is the primary use case: more VRAM = a bigger model = better decisions,
with latency roughly constant. Match by **available VRAM after subtracting
~1 GB for CUDA runtime + activations**. bf16 = 2 bytes/param; int8 is not
used on CUDA (it would only slow things down).

Default short-list (verify via websearch; newer models may deserve a spot):

| Free VRAM after overhead | Pick | bf16 footprint |
|---|---|---|
| ≥9 GB | `Qwen/Qwen3-VL-4B-Instruct` (or `Qwen3-VL-8B-Instruct` if ≥14 GB free) | ~8 GB |
| 5–9 GB | `Qwen/Qwen2.5-VL-3B-Instruct` | ~6 GB |
| 3–5 GB | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (still bf16 here) | ~4.4 GB |

Always present 2–3 alternatives with a one-line tradeoff (size vs. quality
vs. reasoning vs. multimodal coverage). If the user already has one in mind,
skip the recommendation but still confirm it fits.

## 4b. Recommend — CPU path

**CPU is not where you go for quality** — it's where you go because you
have no GPU. On CPU the bottleneck is **compute speed**, not RAM: a 4 GB
model on a laptop CPU takes ~30 s per request, a 500 MB model takes ~1 s.
Bigger models just waste RAM and slow everything down without giving you
better decisions on most prompts.

Default rule: **use the smallest model that meets your quality bar**. The
project already ships `config.cpu.toml` (SmolVLM-500M-Instruct, fp32,
~2 GB RAM) — that's the recommended starting point. Move down to
`SmolVLM-256M-Instruct` (~1 GB fp32) if your CPU is very slow or you need to
fit alongside a browser; move up to `SmolVLM2-2.2B-Instruct` only if you've
measured the smaller one and it isn't good enough.

Quick CPU capability check (run before deciding quantization):

```bash
grep -oE 'avx2|avx512f|avx512_bf16|avx_vnni|amx_tile' /proc/cpuinfo | sort -u
```

- `quantize = "auto"` is fp32 on CPU (bf16 only on GPU). fp32 keeps the
  KV-cached scores identical to the exact path and avoids emulated bf16; on a
  500M model the ~2 GB RAM cost is negligible, so this is the default.
- `avx512_bf16` or `amx_tile` (Zen 4/5, Sapphire Rapids): set
  `quantize = "bf16"` explicitly for ~2x speed (auto won't pick it on CPU); the
  logprob drift is tiny (<0.1 nats). Only if speed matters more than
  bit-exactness.
- `avx2` + `avx_vnni` and tight on RAM or slow: `quantize = "int8"` is ~2x
  faster and ~0.5 GB, but it perturbs logprobs (near-tie predictions can
  shift), so only use it if the accuracy trade-off is acceptable.
- Older CPUs without AVX2: stay on fp32; expect slow load times and several
  seconds per request.

If the user wants text-only on CPU (smaller model, no vision tower cost),
point them at `HuggingFaceTB/SmolLM2-360M-Instruct` or
`HuggingFaceTB/SmolLM2-1.7B-Instruct` — these are not VLMs but score
significantly faster than the equivalent-size VLMs on text-only prompts.

## 5. Ask the user to pick

Show the recommendation and the alternatives with the per-model VRAM and
image cap so the tradeoff is visible at pick time. **Do not download
until the user picks.** If the user says "use the default", confirm which
config file — they almost certainly mean `config.cpu.toml` for the shipped
CPU preset, since `config.toml` is per-host. Then skip to step 7.

For each candidate compute `footprint ≈ 2 × params` and
`headroom ≈ total VRAM − footprint − 1 GB`, then look up the cap from the
"Estimating `max_image_pixels`" section in step 7. Present:

| Model | Footprint | Headroom | `max_image_pixels` | Effective image (square) |
|---|---|---|---|---|
| `<model>` | ~X GB | ~Y GB | `<cap>` / omit | ≈ Z MB (~S×S px, ~N tokens) |

"Effective image" is what the processor resamples any upload to,
**assuming a square**: `√max_image_pixels` px on a side,
`≈ max_image_pixels × 3 / 1024²` MB of raw RGB data (3 bytes/pixel) — the
data the vision encoder actually sees. The number of *vision tokens* this
expands to depends on the model's patch size; for Qwen-VL it's
`≈ cap / 784`. Users can always upload larger originals — we downsample
to fit. The cap bounds the model's view, not the upload size: a tighter
cap means less fine detail (small text, subtle features) but never a
rejected upload.

## 6. Download

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '<model_id>',
    local_dir='models/<short-name>',
    allow_patterns=['*.json', '*.txt', '*.safetensors', '*.bin',
                    '*.py', '*.model', '*.tiktoken'],
)
print('downloaded to models/<short-name>')
"
```

Check available disk before pulling — most VLMs are 1–9 GB.

## 7. Configure

Edit or create the relevant `config.toml`. The scorer reads:

```toml
model = "<model_id>"
save_to = "models/<short-name>"   # relative to project root
device = "auto" | "cpu" | "gpu"
gpu = "<GPU label from nvidia-smi>"  # e.g. "NVIDIA GeForce RTX 5060 Ti" — free-form, just for documentation
quantize = "auto" | "bf16" | "fp32" | "int8"
max_image_pixels = ""   # omit / "" = model's native image budget
kv_cache = true
temperature = 1.0
calibrate = true
calibration_context = "N/A"
```

### Estimating `max_image_pixels`

The cap is applied **before** the model's processor sees the image: we
resize the PIL image so its total pixel count is at most
`max_image_pixels` (aspect ratio preserved). This is model-agnostic —
Qwen-VL, Idefics3, SmolVLM, LLaVA, InternVL, ... all flow through the
same code path, so one cap table covers the whole VLM landscape.

1. **Footprint.** bf16 size ≈ 2 × params in billions (vision tower included).
2. **Headroom.** `total VRAM (nvidia-smi memory.total) − footprint − ~1 GB
   overhead` — what's left for image activations and per-option KV cache.
3. **Cap.** Pick from the headroom-based table. Halve the cap value if the
   scorer still OOMs — activations track the cap roughly linearly with
   image area.

| Headroom after the chosen model | `max_image_pixels` | ≈ effective image (square) |
|---|---|---|
| ≥ 4 GB | omit | ~48 MB raw (~4096×4096 px) |
| 2 – 4 GB | `1605632` | ~4.6 MB raw (~1267×1267 px) |
| 1 – 2 GB | `802816` | ~2.3 MB raw (~896×896 px) |
| < 1 GB | `401408` | ~1.1 MB raw (~633×633 px) |

"Effective image" assumes a square: `√max_image_pixels` px on a side, with
the raw RGB size being `max_image_pixels × 3` bytes. That's the data the
vision encoder actually processes after downsampling. Users can still
upload larger originals — the cap bounds the model's view, not the upload.

Vision-token count depends on the model's patch architecture (Qwen-VL ≈
`cap / 784`); the cap itself is in pixels and stays portable across
families.

Concrete picks for the project's reference tiers:

| Model (from step 5) | Card | Footprint | Headroom | Cap | Effective image (square) |
|---|---|---|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | 16 GB (RTX 5060 Ti) | ~8 GB | ~7 GB | omit | ~48 MB raw |
| `Qwen/Qwen3-VL-2B-Instruct` | 8 GB | ~4 GB | ~3 GB | `1605632` | ~4.6 MB raw |
| `Qwen/Qwen3-VL-2B-Instruct` | 6 GB (RTX 3060 Laptop) | ~4 GB | ~0.7 GB | `401408` | ~1.1 MB raw |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | CPU | ~2 GB RAM | n/a | omit | already tiny — set only if you see RAM pressure |

If the scorer still OOMs at the chosen cap, halve the pixel value (tokens
halve, activations roughly quarter). The OOM message itself names the
current cap and a sensible next step.

If switching the **default model** (the one `config.toml.example` ships
with and `DEFAULT_MODEL_ID` points at), also update:

- `README.md` — Configuration table + bias-table placeholder
- `Dockerfile` / `docker-compose.yml` — only if the default's CPU image changes
- `docs/index.html` — "Choosing a model" / "Trying it on a vision model" sections
- `zero_shot/core/config.py` — `DEFAULT_MODEL_ID` / `DEFAULT_SAVE_TO`
- `config.toml.example` — keep the tracked template's `model = "..."` line
  in sync with the code default so a fresh `cp` lands on the same model

## 8. Smoke test

Confirm the new config actually loads. Start the server and poll `/api/health`:

```bash
.venv/bin/python -m uvicorn zero_shot.interfaces.server.app:app \
    --host 127.0.0.1 --port 8766 --log-level info \
    > /tmp/opencode/smoke.log 2>&1 &
SERVER_PID=$!
for i in {1..60}; do
    sleep 2
    if curl -sf http://127.0.0.1:8766/api/health | grep -q '"model_loaded": true'; then
        echo "loaded after ${i} polls"
        break
    fi
done
kill -INT $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
```

If it fails to load, dump the last 30 lines of `/tmp/opencode/smoke.log` and do
not declare success. Common failures: missing `trust_remote_code` (re-pick a
native model), disk full (clean `models/` of unused weights), or
architecture mismatch (arm64 wheels unavailable for that repo).

## 9. Report

Summarise what changed: config file edited, weights downloaded (path + size),
expected runtime memory, the command to serve (`zero-shot-serve` vs
`--config <path>`), and a one-line caveat about quality.
