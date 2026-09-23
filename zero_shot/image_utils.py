from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

_DEFAULT_QUALITY = 85
_DEFAULT_MIN_EDGE = 256
_SCALE_STEP = 0.75


def decode_image(data: bytes):
    """Decode ``data`` into an RGB PIL image, raising ``ValueError`` on failure.

    EXIF is deliberately ignored: the pixels are used as stored, with no
    orientation transpose. (It would otherwise only be applied on some code
    paths and not others.)
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as src:
            src.load()
            im = src.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError(f"Image is too large to decode safely ({exc}).") from exc
    except (Image.UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc
    return im


def downscale_to_byte_limit(
    data: bytes,
    max_bytes: int,
    *,
    quality: int = _DEFAULT_QUALITY,
    min_edge: int = _DEFAULT_MIN_EDGE,
) -> bytes:
    """Return JPEG bytes at or below ``max_bytes``.

    ``data`` is returned unchanged when it already fits. Otherwise it is decoded,
    converted to RGB, and repeatedly scaled down by 25% and re-encoded as JPEG
    until it fits. Raises ``ValueError`` if the image cannot be decoded or cannot
    be brought under the limit.
    """
    if len(data) <= max_bytes:
        return data

    from PIL import Image

    im = decode_image(data)

    scale = 1.0
    encoded = data
    while True:
        width = max(1, int(im.width * scale))
        height = max(1, int(im.height * scale))
        frame = (
            im
            if (width == im.width and height == im.height)
            else im.resize((width, height), Image.Resampling.LANCZOS)
        )
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=quality)
        encoded = buf.getvalue()
        if len(encoded) <= max_bytes:
            logger.info(
                "Compressed image %dx%d (%d bytes) to %dx%d JPEG q%d (%d bytes).",
                im.width,
                im.height,
                len(data),
                width,
                height,
                quality,
                len(encoded),
            )
            return encoded
        if width <= min_edge or height <= min_edge:
            break
        scale *= _SCALE_STEP

    raise ValueError(
        f"Could not compress image below {max_bytes} bytes "
        f"(still {len(encoded)} bytes at {min_edge}px)."
    )
