"""Serve the built UI behind the real FastAPI service with a SYNTHETIC adapter.

    NOT A MODEL. The adapter "detects" saturated magenta pixels. This server
    exists so browser tests can exercise upload, EXIF orientation, overlay
    alignment, review and export against the real HTTP contract. It says
    nothing about damage-detection quality.

Usage (from the repository root):
    .venv/bin/python tests/e2e_server.py --port 7861 --fixtures frontend/e2e/.fixtures
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np
import uvicorn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from impactlens.service import create_app  # noqa: E402
from tests.conftest import API_KEY, MAGENTA, SyntheticAdapter, make_settings, synthetic_loader  # noqa: E402


def _canvas(
    width: int, height: int, box: tuple[int, int, int, int] | None, base=(120, 128, 136)
) -> np.ndarray:
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = base
    # A little texture so the photos are not flat.
    pixels[::8, :] = (110, 118, 126)
    if box:
        x0, y0, x1, y1 = box
        pixels[y0:y1, x0:x1] = MAGENTA
    return pixels


def write_fixtures(directory: Path) -> None:
    """Deterministic test photos. Expected normalized boxes are in the file names' docs below."""
    directory.mkdir(parents=True, exist_ok=True)

    # Portrait as a phone stores it: sensor pixels landscape (900x600) + EXIF orientation 6.
    # Displayed (and reported by the service) as 600x900 with the block at [0.1, 0.1, 0.4, 0.3].
    display = _canvas(600, 900, (60, 90, 240, 270))
    stored = np.rot90(display, k=1)  # orientation 6 = viewer rotates 90 deg clockwise
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(stored)).save(
        buffer, "JPEG", quality=95, subsampling=0, exif=exif
    )
    (directory / "portrait-exif6.jpg").write_bytes(buffer.getvalue())

    # Landscape PNG, 1200x800, block at [0.5, 0.5, 0.75, 0.75].
    Image.fromarray(_canvas(1200, 800, (600, 400, 900, 600))).save(directory / "landscape.png")

    # No magenta: the synthetic adapter returns nothing.
    Image.fromarray(_canvas(800, 600, None, base=(90, 140, 100))).save(directory / "no-region.png")

    # Lossless WebP square with block at [0.25, 0.25, 0.5, 0.5].
    Image.fromarray(_canvas(640, 640, (160, 160, 320, 320))).save(
        directory / "square.webp", lossless=True
    )

    # Not an image at all, despite its name.
    (directory / "not-a-photo.jpg").write_bytes(b"This is a text file pretending to be a JPEG.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--dist", type=Path, default=ROOT / "frontend" / "dist")
    args = parser.parse_args()

    write_fixtures(args.fixtures)
    settings = make_settings(
        frontend_dist=args.dist, model_id="synthetic-test-double", rate_limit_per_minute=10_000
    )
    app = create_app(settings, model_loader=synthetic_loader(SyntheticAdapter()))
    logging.basicConfig(level="WARNING")
    print(
        f"E2E server (SYNTHETIC adapter, not a model) on http://127.0.0.1:{args.port} key={API_KEY}",
        flush=True,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
