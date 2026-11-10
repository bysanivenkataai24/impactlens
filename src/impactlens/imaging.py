"""Safe decoding of untrusted image uploads.

The bytes are sniffed (the client's Content-Type and file name are ignored),
decoded with only the JPEG/PNG/WebP codecs enabled, checked against a pixel
budget *before* the pixels are decompressed, rotated to honour EXIF
orientation and converted to RGB. Only the pixel array leaves this module, so
EXIF (GPS position, device serial numbers, timestamps) is dropped by
construction. Nothing is written to disk.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

_EXIF_ORIENTATION_TAG = 0x0112
SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")
MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class ImageRejected(Exception):
    """The upload cannot be used. Carries the HTTP status and a stable error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DecodedImage:
    #: ``(height, width, 3)`` uint8 RGB pixels, already EXIF-oriented.
    rgb: np.ndarray
    width: int
    height: int
    format: str
    exif_orientation: int | None
    orientation_applied: bool
    sha256: str


def sniff_format(data: bytes) -> str | None:
    """Identify JPEG / PNG / WebP from magic bytes. Anything else is ``None``."""
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def configure_pillow(max_pixels: int) -> None:
    """Set Pillow's process-wide decompression-bomb threshold to the service budget."""
    Image.MAX_IMAGE_PIXELS = int(max_pixels)


def decode_image(data: bytes, *, max_pixels: int, min_side: int = 32) -> DecodedImage:
    """Decode ``data`` or raise :class:`ImageRejected` (415 / 413 / 422)."""
    if not data:
        raise ImageRejected(422, "malformed_image", "The uploaded file is empty.")

    fmt = sniff_format(data)
    if fmt is None:
        raise ImageRejected(
            415,
            "unsupported_media_type",
            "The file is not a JPEG, PNG or WebP image (detected from its content, not its name).",
        )

    digest = hashlib.sha256(data).hexdigest()
    malformed = ImageRejected(
        422, "malformed_image", f"The file starts like a {fmt} image but could not be decoded."
    )
    too_many_pixels = ImageRejected(
        413,
        "image_pixel_limit_exceeded",
        f"The image has more pixels than this service accepts ({max_pixels:,}).",
    )

    try:
        with warnings.catch_warnings():
            # Pillow only *warns* between 1x and 2x its limit; treat that as a hard stop.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[fmt]) as image:
                # ``size`` comes from the header: no pixel data is decompressed yet.
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise malformed
                if width * height > max_pixels:
                    raise too_many_pixels
                if min(width, height) < min_side:
                    raise ImageRejected(
                        422,
                        "image_too_small",
                        f"The image is {width}x{height} px; at least {min_side} px per side "
                        "is required.",
                    )
                if getattr(image, "is_animated", False):
                    image.seek(0)  # first frame only
                orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
                image.load()
                oriented = ImageOps.exif_transpose(image)
                rgb = np.array(oriented.convert("RGB"), dtype=np.uint8)
    except ImageRejected:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise too_many_pixels from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError, MemoryError) as exc:
        raise malformed from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise malformed
    orientation = int(orientation) if isinstance(orientation, int) else None
    return DecodedImage(
        rgb=np.ascontiguousarray(rgb),
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        format=fmt,
        exif_orientation=orientation,
        orientation_applied=orientation is not None and 2 <= orientation <= 8,
        sha256=digest,
    )
