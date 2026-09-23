# syntax=docker/dockerfile:1

# One Dockerfile, two builds:
#   docker build -t zero-shot:cpu .                                     # CPU + SmolVLM (default)
#   docker build --build-arg TORCH_BACKEND=cu130 -t zero-shot:gpu .     # CUDA + Qwen
#
# TORCH_BACKEND is passed straight to `uv pip install --torch-backend`, so the
# image gets the matching PyTorch wheel (+ its CUDA deps only in GPU builds).
# The CUDA userspace libs come from the wheel; only the host driver is injected
# by the NVIDIA Container Toolkit at run time.
FROM python:3.12-slim

ARG TORCH_BACKEND=cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    ZERO_SHOT_CONFIG=/app/config.cpu.toml \
    ZERO_SHOT_HOST=0.0.0.0 \
    ZERO_SHOT_PORT=8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY zero_shot ./zero_shot
RUN uv venv .venv \
 && uv pip install --python .venv --torch-backend="${TORCH_BACKEND}" .

# Triton (used by the GPU path) JIT-compiles a C launcher on first use, so GPU
# builds need a C compiler + libc headers at runtime. CPU builds skip this.
RUN if [ "${TORCH_BACKEND}" != "cpu" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends gcc libc6-dev \
      && rm -rf /var/lib/apt/lists/*; \
    fi

COPY config.toml config.cpu.toml ./

EXPOSE 8000
CMD ["zero-shot-serve"]
