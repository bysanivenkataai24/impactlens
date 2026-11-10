"""Shared fixtures.

IMPORTANT - what these tests do and do not show
------------------------------------------------
Most tests here are *contract tests*. They replace the model with
:class:`SyntheticAdapter`, a few lines of numpy that "detect" saturated magenta
pixels. That makes the HTTP contract, validation, coordinate handling and
error paths deterministic and fast. It says nothing about whether any real
model can find vehicle damage. Model quality is measured only by
``impactlens evaluate`` on a held-out, licensed dataset.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from impactlens.adapters.base import AdapterInfo, RawInstance
from impactlens.config import Settings
from impactlens.service import create_app
from impactlens.service.runtime import LoadedModel
from impactlens.taxonomy import MODEL_DAMAGE_TYPES

API_KEY = "test-key-0123456789abcdefghij"  # 30 chars, test-only value
SYNTHETIC_SHA256 = "0" * 64
MAGENTA = (255, 0, 255)


class SyntheticAdapter:
    """NOT a model. Returns one 'dent' covering the bounding box of magenta pixels.

    Hooks let a test hold the adapter busy, make it fail, or make it return
    arbitrary instances.
    """

    def __init__(
        self,
        class_names: tuple[str, ...] = MODEL_DAMAGE_TYPES,
        task: str = "segment",
    ) -> None:
        self.info = AdapterInfo(
            task=task,
            class_names=class_names,
            device="cpu",
            inference_image_size=640,
            runtime={"adapter": "SyntheticAdapter (test double, not a model)"},
        )
        self.calls = 0
        self.gate: threading.Event | None = None
        self.entered = threading.Event()
        self.fail_with: Exception | None = None
        self.override: Callable[[np.ndarray], list[RawInstance]] | None = None
        self.closed = False

    def predict(self, image_rgb, *, score_threshold, max_detections):
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(timeout=30)
        if self.fail_with is not None:
            raise self.fail_with
        if self.override is not None:
            return self.override(image_rgb)[:max_detections]
        r, g, b = (image_rgb[:, :, i].astype(np.int16) for i in range(3))
        mask = (r > 200) & (g < 80) & (b > 200)  # tolerant, so lossy JPEG/WebP still match
        if not mask.any() or score_threshold > 0.9:
            return []
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        return [
            RawInstance(
                class_name="dent",
                score=0.9,
                box_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                polygon_xy=polygon,
            )
        ]

    def close(self) -> None:
        self.closed = True


def make_settings(**overrides) -> Settings:
    """Ready-by-default settings. The checkpoint path is never opened: a loader is injected."""
    values: dict = {
        "api_key": API_KEY,
        "model_id": "synthetic-test-double",
        "model_path": Path("injected-by-test.pt"),
        "rate_limit_per_minute": 1000,
    }
    values.update(overrides)
    return Settings(**values)


def synthetic_loader(adapter: SyntheticAdapter):
    def load(_settings: Settings) -> LoadedModel:
        return LoadedModel(adapter=adapter, revision="test", checkpoint_sha256=SYNTHETIC_SHA256)

    return load


def encode(image: Image.Image, fmt: str = "JPEG", **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    if fmt == "JPEG":
        save_kwargs.setdefault("quality", 100)
        save_kwargs.setdefault("subsampling", 0)
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def canvas_with_block(
    width: int, height: int, box: tuple[int, int, int, int], fmt: str = "PNG", **save_kwargs
) -> bytes:
    """Grey image with one magenta rectangle at ``box`` = (x0, y0, x1, y1)."""
    pixels = np.full((height, width, 3), 128, dtype=np.uint8)
    x0, y0, x1, y1 = box
    pixels[y0:y1, x0:x1] = MAGENTA
    return encode(Image.fromarray(pixels), fmt, **save_kwargs)


def with_exif_orientation(image: Image.Image, orientation: int, fmt: str = "PNG") -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return encode(image, fmt, exif=exif)


@pytest.fixture
def adapter() -> SyntheticAdapter:
    return SyntheticAdapter()


@pytest.fixture
def client(adapter: SyntheticAdapter):
    app = create_app(make_settings(), model_loader=synthetic_loader(adapter))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def post_image(client: TestClient, data: bytes, headers: dict[str, str], **fields):
    return client.post(
        "/predict",
        headers=headers,
        files={"image": ("photo.bin", data, "application/octet-stream")},
        data={k: str(v) for k, v in fields.items()},
    )
