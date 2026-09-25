#!/usr/bin/env bash
# Full image-classification benchmark: MNIST, Fashion-MNIST, KMNIST, USPS.
#
# Loads Qwen3-VL-4B once per dataset on the GPU and writes each dataset's
# predictions.csv + summary.json under <DATA_DIR>/<name>/out/.
#
# Prepare the data first:
#   .venv/bin/python benchmark/datasets/prepare_image_classification.py
#
# Then run:
#   benchmark/datasets/run_popular_image_datasets.sh [DATA_DIR]
#
# Overrides (env vars):
#   MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
#   SAVE_TO=models/qwen3-vl-4b-instruct
#   DEVICE=gpu
#   LIMIT=20           # smoke test: only the first N rows per dataset
#   OUT_SUFFIX=out     # output dir name; use e.g. out-nocal to keep runs apart
#   EXTRA_ARGS=""      # extra flags passed through, e.g. "--no-calibrate"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
DATA_DIR="${1:-$REPO_ROOT/benchmark/data/popular}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
SAVE_TO="${SAVE_TO:-models/qwen3-vl-4b-instruct}"
DEVICE="${DEVICE:-gpu}"
OUT_SUFFIX="${OUT_SUFFIX:-out}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS+=(--limit "$LIMIT")
fi

EXTRA=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA <<< "${EXTRA_ARGS}"
fi

for name in mnist fashion_mnist kmnist usps; do
    data="$DATA_DIR/$name/data.csv"
    questions="$DATA_DIR/$name/q.json"
    if [[ ! -f "$data" || ! -f "$questions" ]]; then
        echo "missing $data or $questions" >&2
        echo "run: $PY benchmark/datasets/prepare_image_classification.py" >&2
        exit 1
    fi
    echo "=== $name ==="
    "$PY" -m benchmark \
        -d "$data" \
        -q "$questions" \
        --model-id "$MODEL_ID" \
        --save-to "$SAVE_TO" \
        --device "$DEVICE" \
        --image-column image \
        -o "$DATA_DIR/$name/$OUT_SUFFIX" \
        "${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}" \
        "${EXTRA[@]+"${EXTRA[@]}"}"
done

echo
echo "Done. Per-dataset results:"
for name in mnist fashion_mnist kmnist usps; do
    echo "  $DATA_DIR/$name/$OUT_SUFFIX/{predictions.csv,summary.json}"
done
