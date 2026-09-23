---
name: model-advisor
description: Use when the user asks which model to run for this project, wants to switch the default model, or asks what hardware they need. Primarily for GPU hosts — detects VRAM, websearches for current vision-language checkpoints, recommends one with alternatives, and — after the user picks — downloads the weights and updates the relevant config. CPU hosts get pointed at the existing config.cpu.toml (SmolVLM-500M-Instruct, int8) with brief tuning guidance. Default-mode is vision-language unless the user explicitly asks for text-only.
---

# Model advisor

The project ships two configs (`config.toml` for GPU, `config.cpu.toml` for CPU)
but the user may want a different model based on hardware or use case. This skill
walks through detection → search → recommendation → download → configure.

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
project already ships `config.cpu.toml` (SmolVLM-500M-Instruct, int8,
~500 MB RAM) — that's the recommended starting point. Move down to
`SmolVLM-256M-Instruct` (~250 MB) if your CPU is very slow or you need to
fit alongside a browser; move up to `SmolVLM2-2.2B-Instruct` only if you've
measured the smaller one and it isn't good enough.

Quick CPU capability check (run before deciding quantization):

```bash
grep -oE 'avx2|avx512f|avx512_bf16|avx_vnni|amx_tile' /proc/cpuinfo | sort -u
```

- `avx2` + `avx_vnni` (most Intel/AMD since ~2017): int8 dynamic quant is
  ~2× faster than bf16. Stick with `quantize = "int8"`.
- `avx512_bf16` or `amx_tile` (Zen 4/5, Sapphire Rapids): bf16 is hardware-
  accelerated; int8 is still slightly faster but bf16 keeps the
  probabilities cleaner.
- Older CPUs without AVX2: int8 is still your best bet but expect slow
  load times and ~5–10 s per request.

If the user wants text-only on CPU (smaller model, no vision tower cost),
point them at `HuggingFaceTB/SmolLM2-360M-Instruct` or
`HuggingFaceTB/SmolLM2-1.7B-Instruct` — these are not VLMs but score
significantly faster than the equivalent-size VLMs on text-only prompts.

## 5. Ask the user to pick

Show the recommendation and the alternatives. **Do not download until the
user picks.** If the user says "use the default", confirm which config file
(`config.toml` GPU vs `config.cpu.toml` CPU) and skip to step 7.

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
gpu = "<gpu key from nvidia-smi>"  # e.g. "rtx_5060_ti"
quantize = "auto" | "bf16" | "fp32" | "int8"
kv_cache = true
temperature = 1.0
calibrate = true
calibration_context = "N/A"
```

If switching the **default** (changes to `config.toml`), also update:

- `README.md` — Configuration table + bias-table placeholder
- `Dockerfile` / `docker-compose.yml` — only if the default's CPU image changes
- `docs/index.html` — "Choosing a model" / "Trying it on a vision model" sections
- `zero_shot/config.py` — `DEFAULT_MODEL_ID` / `DEFAULT_SAVE_TO`

## 8. Smoke test

Confirm the new config actually loads. Start the server and poll `/api/health`:

```bash
.venv/bin/python -m uvicorn zero_shot.server:app \
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
