"""Materialize a split manifest as a YOLO segmentation dataset.

Images are re-encoded with their EXIF orientation applied and all metadata
removed (no GPS position, serial numbers or timestamps reach the training
set), and are named by content hash so original file names do not leak
either. Label coordinates are normalized to those oriented pixels.

Only annotated polygons become labels. A box without a mask is counted and
skipped: a mask is never invented from a bounding box.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from ..taxonomy import MODEL_DAMAGE_TYPES, normalize_class_name
from .manifest import SPLITS, ManifestError, Row, manifest_revision, split_report

CLASS_INDEX = {name: index for index, name in enumerate(MODEL_DAMAGE_TYPES)}


def _merge_parts(parts: list[np.ndarray]) -> np.ndarray:
    """One instance annotated as several polygons -> a single ring (Ultralytics convention)."""
    if len(parts) == 1:
        return parts[0]
    from ultralytics.data.converter import merge_multi_segment

    merged = merge_multi_segment([part.tolist() for part in parts])
    return np.concatenate(merged, axis=0).reshape(-1, 2)


def _label_lines(
    annotations: list[dict], categories: dict, width: int, height: int, counters: Counter
):
    lines: list[str] = []
    for annotation in annotations:
        name = categories.get(annotation.get("category_id"))
        if name not in CLASS_INDEX or annotation.get("iscrowd"):
            continue
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            counters["instances_without_polygon_skipped"] += 1
            continue
        parts = [
            np.asarray(part, dtype=np.float64).reshape(-1, 2)
            for part in segmentation
            if isinstance(part, list) and len(part) >= 6 and len(part) % 2 == 0
        ]
        if not parts:
            counters["instances_without_polygon_skipped"] += 1
            continue
        if len(parts) > 1:
            counters["multi_part_instances_merged"] += 1
        ring = _merge_parts(parts)
        ring[:, 0] = np.clip(ring[:, 0] / width, 0.0, 1.0)
        ring[:, 1] = np.clip(ring[:, 1] / height, 0.0, 1.0)
        if not np.all(np.isfinite(ring)) or len(ring) < 3:
            counters["invalid_polygons_skipped"] += 1
            continue
        coordinates = " ".join(f"{value:.6f}" for value in ring.flatten())
        lines.append(f"{CLASS_INDEX[name]} {coordinates}")
        counters[f"instances:{name}"] += 1
    return lines


def repeat_factors(train_rows: list[Row], threshold: float) -> dict[str, float]:
    """Repeat-factor sampling weights (Gupta et al., LVIS 2019).

    ``f_c`` is the fraction of training images containing class ``c``. A class
    rarer than ``threshold`` gets ``r_c = sqrt(threshold / f_c) > 1`` and an
    image is repeated by the largest factor among its classes.
    """
    total = max(1, len(train_rows))
    frequency = Counter()
    for row in train_rows:
        frequency.update(name for name, count in row.class_counts.items() if count)
    factor = {
        name: max(1.0, math.sqrt(threshold / (count / total))) for name, count in frequency.items()
    }
    return {
        row.image_id: max((factor[n] for n, c in row.class_counts.items() if c), default=1.0)
        for row in train_rows
    }


def export_yolo(
    rows: list[Row],
    *,
    data_root: Path,
    out_dir: Path,
    jpeg_quality: int = 95,
    rfs_threshold: float | None = None,
    seed: int = 0,
) -> dict:
    """Write images, labels, ``dataset.yaml`` and ``dataset_revision.json`` to ``out_dir``."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    kept = [row for row in rows if row.kept]
    if not kept or any(row.split not in SPLITS for row in kept):
        raise ManifestError("every kept image needs a split; run `impactlens data split` first")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ManifestError(f"{out_dir} is not empty; export into a new directory per revision")

    coco_cache: dict[str, tuple[dict, dict]] = {}

    def annotations_for(row: Row) -> tuple[list[dict], dict]:
        if not row.annotation_file:
            return [], {}
        if row.annotation_file not in coco_cache:
            coco = json.loads((data_root / row.annotation_file).read_text(encoding="utf-8"))
            by_image: dict[str, list[dict]] = defaultdict(list)
            for annotation in coco.get("annotations", []):
                by_image[str(annotation["image_id"])].append(annotation)
            categories = {
                c["id"]: normalize_class_name(c["name"]) for c in coco.get("categories", [])
            }
            coco_cache[row.annotation_file] = (by_image, categories)
        by_image, categories = coco_cache[row.annotation_file]
        return by_image.get(row.coco_image_id, []), categories

    counters: Counter = Counter()
    written: dict[str, str] = {}
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in sorted(kept, key=lambda r: r.image_id):
        with Image.open(data_root / row.image_path) as image:
            oriented = ImageOps.exif_transpose(image)
            lossless = (image.format or "").upper() == "PNG"
            rgb = oriented.convert("RGB")
        if (rgb.width, rgb.height) != (row.width, row.height):
            raise ManifestError(f"{row.image_id}: image changed since the manifest was built")
        stem = row.sha256[:20]
        target = out_dir / "images" / row.split / f"{stem}.{'png' if lossless else 'jpg'}"
        # A fresh Image without ``info``/exif: nothing but pixels is written.
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        if lossless:
            clean.save(target, "PNG")
        else:
            clean.save(target, "JPEG", quality=jpeg_quality, subsampling=0)
        annotations, categories = annotations_for(row)
        lines = _label_lines(annotations, categories, row.width, row.height, counters)
        # An empty label file marks a background (negative) image for the trainer.
        (out_dir / "labels" / row.split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        written[row.image_id] = f"./images/{row.split}/{target.name}"
        counters[f"images:{row.split}"] += 1

    # Lets evaluation join an exported file back to its manifest row (negatives, slices)
    # without copying source paths or original file names into the dataset directory.
    meta_columns = sorted({key for row in kept for key in row.meta})
    with (out_dir / "image_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "file",
                "split",
                "image_id",
                "leakage_group",
                "is_negative",
                "n_instances",
                *meta_columns,
            ]
        )
        for row in sorted(kept, key=lambda r: r.image_id):
            writer.writerow(
                [
                    Path(written[row.image_id]).name,
                    row.split,
                    row.image_id,
                    row.leakage_group,
                    "1" if row.is_negative else "0",
                    row.n_instances,
                    *(row.meta.get(column, "") for column in meta_columns),
                ]
            )

    names = dict(enumerate(MODEL_DAMAGE_TYPES))
    base_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (out_dir / "dataset.yaml").write_text(
        yaml.safe_dump(base_yaml, sort_keys=False), encoding="utf-8"
    )

    rfs_info = None
    if rfs_threshold is not None:
        train_rows = [row for row in kept if row.split == "train"]
        factors = repeat_factors(train_rows, rfs_threshold)
        rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
        listing: list[str] = []
        for row in sorted(train_rows, key=lambda r: r.image_id):
            factor = factors[row.image_id]
            copies = int(factor) + (1 if rng.random() < factor - int(factor) else 0)
            listing.extend([written[row.image_id]] * copies)
        (out_dir / "train_rfs.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
        (out_dir / "dataset_rfs.yaml").write_text(
            yaml.safe_dump({**base_yaml, "train": "train_rfs.txt"}, sort_keys=False),
            encoding="utf-8",
        )
        rfs_info = {
            "threshold": rfs_threshold,
            "train_images": len(train_rows),
            "train_entries_after_repeats": len(listing),
        }

    revision = {
        "dataset_revision": manifest_revision(rows),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "class_names": list(MODEL_DAMAGE_TYPES),
        "export": {
            "format": "ultralytics-yolo-segmentation",
            "images": "EXIF-oriented, metadata stripped, named by content hash",
            "jpeg_quality": jpeg_quality,
            "repeat_factor_sampling": rfs_info,
        },
        "export_counters": dict(sorted(counters.items())),
        "split_report": split_report(rows),
    }
    (out_dir / "dataset_revision.json").write_text(
        json.dumps(revision, indent=2) + "\n", encoding="utf-8"
    )
    return revision
