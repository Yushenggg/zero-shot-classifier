"""Regression tests for image decoding and size-limited compression."""

from __future__ import annotations

import io
import random

import pytest
from PIL import Image

from zero_shot.core.image_utils import decode_image, downscale_to_byte_limit


def _noise_png(size: int, seed: int = 0) -> bytes:
    """A seeded, poorly-compressible PNG (deterministic across runs)."""
    rng = random.Random(seed)
    raw = rng.randbytes(size * size * 3)
    image = Image.frombytes("RGB", (size, size), raw)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_returns_rgb(png_bytes):
    image = decode_image(png_bytes)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_decode_image_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"definitely not an image")


def test_downscale_returns_small_data_unchanged(png_bytes):
    assert downscale_to_byte_limit(png_bytes, max_bytes=10_000_000) is png_bytes


def test_downscale_compresses_large_image_below_limit():
    data = _noise_png(256)
    assert len(data) > 40_000

    out = downscale_to_byte_limit(data, max_bytes=40_000, min_edge=64)
    assert len(out) <= 40_000
    # The compressed result is still a decodable image.
    assert decode_image(out).mode == "RGB"


def test_downscale_raises_when_limit_unreachable():
    data = _noise_png(256)
    with pytest.raises(ValueError):
        downscale_to_byte_limit(data, max_bytes=100, min_edge=200)
