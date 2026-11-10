"""``impactlens robustness`` - re-evaluate a split under synthetic degradations.

Each corruption writes a degraded copy of the split and runs the normal
evaluation on it. The report is the change relative to the clean split.

These are *synthetic* stand-ins (a drawn glare blob is not real sun glare).
They show sensitivity trends and catch regressions; they do not replace
evaluating on real photographs taken in rain, at night or through glass.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import yaml

from .evaluate import evaluate
from .provenance import utc_now

Corruption = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _gaussian_blur(image, rng):
    return cv2.GaussianBlur(image, (0, 0), sigmaX=max(1.5, 0.004 * max(image.shape[:2])))


def _motion_blur(image, rng):
    size = max(7, int(0.015 * max(image.shape[:2])) | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(image, -1, kernel)


def _jpeg_compression(image, rng):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 15])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image


def _glare(image, rng):
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = rng.uniform(0.2, 0.8) * width, rng.uniform(0.2, 0.8) * height
    radius = 0.25 * max(height, width)
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2)))
    return np.clip(image.astype(np.float32) + 200.0 * blob[..., None], 0, 255).astype(np.uint8)


def _low_light(image, rng):
    dark = (image.astype(np.float32) / 255.0) ** 2.2 * 255.0 * 0.6
    noise = rng.normal(0, 8, size=image.shape)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)


def _rain(image, rng):
    height, width = image.shape[:2]
    layer = np.zeros((height, width), dtype=np.uint8)
    length = max(8, height // 40)
    for _ in range(int(0.0015 * height * width)):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        cv2.line(layer, (x, y), (x + length // 4, y + length), 255, 1)
    layer = cv2.blur(layer, (3, 3))
    hazy = cv2.addWeighted(image, 0.85, np.full_like(image, 180), 0.15, 0)
    return np.clip(
        hazy.astype(np.int16) + (layer[..., None] * 0.5).astype(np.int16), 0, 255
    ).astype(np.uint8)


def _occlusion(image, rng):
    out = image.copy()
    height, width = image.shape[:2]
    for _ in range(3):
        w, h = int(rng.uniform(0.08, 0.2) * width), int(rng.uniform(0.08, 0.2) * height)
        x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
        out[y : y + h, x : x + w] = rng.integers(40, 200)
    return out


CORRUPTIONS: dict[str, Corruption] = {
    "gaussian_blur": _gaussian_blur,
    "motion_blur": _motion_blur,
    "jpeg_q15": _jpeg_compression,
    "glare": _glare,
    "low_light_noise": _low_light,
    "rain_streaks": _rain,
    "occlusion_patches": _occlusion,
}
#: Geometric corruption handled separately because labels must be clipped too.
PARTIAL_VIEW = "partial_view_crop70"


def _clip_polygon(ring: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman clipping of a ring against an axis-aligned rectangle."""

    def clip(points, inside, intersect):
        out = []
        for i, current in enumerate(points):
            previous = points[i - 1]
            if inside(current):
                if not inside(previous):
                    out.append(intersect(previous, current))
                out.append(current)
            elif inside(previous):
                out.append(intersect(previous, current))
        return out

    def x_cross(x):
        return lambda p, q: (x, p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0]))

    def y_cross(y):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]), y)

    points = [tuple(p) for p in ring]
    for inside, intersect in (
        (lambda p: p[0] >= x0, x_cross(x0)),
        (lambda p: p[0] <= x1, x_cross(x1)),
        (lambda p: p[1] >= y0, y_cross(y0)),
        (lambda p: p[1] <= y1, y_cross(y1)),
    ):
        if not points:
            break
        points = clip(points, inside, intersect)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _area(ring: np.ndarray) -> float:
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def crop_labels(
    lines: list[str], x0: float, y0: float, x1: float, y1: float, min_visible: float = 0.25
) -> list[str]:
    """Clip normalized label polygons to a normalized crop window and re-normalize.

    An instance whose visible part is under ``min_visible`` of its area is dropped:
    it would be unreasonable to require a detector to find it.
    """
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        ring = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
        clipped = _clip_polygon(ring, x0, y0, x1, y1)
        if len(clipped) < 3 or _area(clipped) < min_visible * _area(ring):
            continue
        clipped = (clipped - (x0, y0)) / (x1 - x0, y1 - y0)
        out.append(parts[0] + " " + " ".join(f"{v:.6f}" for v in np.clip(clipped, 0, 1).flatten()))
    return out


def _build_variant(dataset_dir: Path, split: str, name: str, out_dir: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for path in sorted((dataset_dir / "images" / split).iterdir()):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        label_file = dataset_dir / "labels" / split / f"{path.stem}.txt"
        lines = label_file.read_text(encoding="utf-8").splitlines() if label_file.is_file() else []
        if name == PARTIAL_VIEW:
            height, width = image.shape[:2]
            x0, y0 = rng.uniform(0, 0.3), rng.uniform(0, 0.3)
            x1, y1 = x0 + 0.7, y0 + 0.7
            image = image[int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)]
            lines = crop_labels(lines, x0, y0, x1, y1)
        else:
            image = CORRUPTIONS[name](image, rng)
        cv2.imwrite(str(out_dir / "images" / split / f"{path.stem}.png"), image)
        (out_dir / "labels" / split / f"{path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    base = yaml.safe_load((dataset_dir / "dataset.yaml").read_text(encoding="utf-8"))
    variant = {
        "path": str(out_dir.resolve()),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "names": base["names"],
    }
    (out_dir / "dataset.yaml").write_text(
        yaml.safe_dump(variant, sort_keys=False), encoding="utf-8"
    )
    revision = json.loads((dataset_dir / "dataset_revision.json").read_text(encoding="utf-8"))
    revision["dataset_revision"] = f"{revision['dataset_revision']}+{name}"
    revision["derived_from_split"] = split
    (out_dir / "dataset_revision.json").write_text(
        json.dumps(revision, indent=2) + "\n", encoding="utf-8"
    )
    # Keep the join to manifest metadata, with the new file extension.
    index = dataset_dir / "image_index.csv"
    if index.is_file():
        with (
            index.open(newline="", encoding="utf-8") as source,
            (out_dir / "image_index.csv").open("w", newline="", encoding="utf-8") as sink,
        ):
            reader = csv.DictReader(source)
            writer = csv.DictWriter(sink, fieldnames=reader.fieldnames, lineterminator="\n")
            writer.writeheader()
            for record in reader:
                record["file"] = Path(record["file"]).stem + ".png"
                writer.writerow(record)
    return out_dir / "dataset.yaml"


def run_robustness(
    weights: Path,
    data_yaml: Path,
    *,
    out_dir: Path,
    score_threshold: float,
    corruptions: list[str] | None = None,
    seed: int = 0,
    keep_images: bool = False,
    **evaluate_kwargs,
) -> dict:
    """Evaluate the validation split clean and under each corruption."""
    dataset_dir = Path(data_yaml).resolve().parent
    out_dir = Path(out_dir)
    names = corruptions or [*CORRUPTIONS, PARTIAL_VIEW]
    unknown = [n for n in names if n not in CORRUPTIONS and n != PARTIAL_VIEW]
    if unknown:
        raise ValueError(
            f"unknown corruption(s) {unknown}; available: {[*CORRUPTIONS, PARTIAL_VIEW]}"
        )

    def headline(result: dict) -> dict:
        return {
            "mask_map50_95": result["mask"]["map50_95"],
            "mask_map50": result["mask"]["map50"],
            "precision": result["operating_point"]["instances"]["precision"],
            "recall": result["operating_point"]["instances"]["recall"],
            "negative_image_fp_rate": result["operating_point"]["negative_images"][
                "image_level_false_positive_rate"
            ],
        }

    clean = evaluate(
        weights,
        data_yaml,
        split="val",
        out_dir=out_dir / "clean",
        score_threshold=score_threshold,
        **evaluate_kwargs,
    )
    report = {
        "kind": "impactlens.robustness",
        "created_utc": utc_now(),
        "smoke_test": clean["smoke_test"],
        "score_threshold": score_threshold,
        "clean": headline(clean),
        "corruptions": {},
    }
    for name in names:
        variant_dir = out_dir / "variants" / name
        variant_yaml = _build_variant(dataset_dir, "val", name, variant_dir, seed)
        result = evaluate(
            weights,
            variant_yaml,
            split="val",
            out_dir=out_dir / name,
            score_threshold=score_threshold,
            **evaluate_kwargs,
        )
        row = headline(result)
        row["delta_mask_map50_95"] = (
            None
            if None in (row["mask_map50_95"], report["clean"]["mask_map50_95"])
            else round(row["mask_map50_95"] - report["clean"]["mask_map50_95"], 4)
        )
        report["corruptions"][name] = row
        if not keep_images:
            shutil.rmtree(variant_dir, ignore_errors=True)
    report["note"] = (
        "Synthetic degradations. Trends only; confirm on real adverse-condition photographs."
    )
    (out_dir / "robustness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
