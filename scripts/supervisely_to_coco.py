"""Convert a Supervisely-format folder (per-image ann/*.json + img/*) to one COCO JSON.

Polygons are kept as polygons; nothing is derived from boxes. Class names are kept
exactly as the source wrote them (renaming is a separate step, see
scripts/map_coco_categories.py). The downloaded files are only read.

    python scripts/supervisely_to_coco.py --data-root data/raw/hitl \
        --folder "Car parts dataset" --out data/raw/hitl/coco/hitl_damage_original.json

Note for the Humans in the Loop "Car Parts and Car Damages" Kaggle upload: the folder
named "Car parts dataset" holds the DAMAGE annotations and "Car damages dataset"
holds the PART annotations (check each folder's meta.json). Use --expect-class to
fail fast if the folder is not the one you meant.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def shoelace(points: list[list[float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def convert(data_root: Path, folder: str, expect: list[str]) -> tuple[dict, dict]:
    base = data_root / folder
    meta_file = base / "meta.json"
    if not meta_file.is_file():
        raise SystemExit(f"{meta_file} not found")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    class_names = [c["title"] for c in meta.get("classes", [])]
    missing = [name for name in expect if name not in class_names]
    if missing:
        raise SystemExit(f"{meta_file} lacks expected classes {missing}; found {class_names}")
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    category_id = {c["name"]: c["id"] for c in categories}

    coco = {
        "info": {
            "description": f"Converted from Supervisely folder '{folder}' by scripts/supervisely_to_coco.py"
        },
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    report: Counter = Counter()
    for ann_file in sorted(base.glob("*/ann/*.json")):
        image_name = ann_file.name[: -len(".json")]
        image_path = ann_file.parent.parent / "img" / image_name
        if Path(image_name).suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            report["annotation_files_without_image"] += 1
            continue
        ann = json.loads(ann_file.read_text(encoding="utf-8"))
        image_id = len(coco["images"]) + 1
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.relative_to(data_root).as_posix(),
                "width": int(ann["size"]["width"]),
                "height": int(ann["size"]["height"]),
            }
        )
        report["images"] += 1
        for obj in ann.get("objects", []):
            if obj.get("geometryType") != "polygon":
                report[f"skipped_geometry_{obj.get('geometryType')}"] += 1
                continue
            name = obj.get("classTitle")
            if name not in category_id:
                report["skipped_unknown_class"] += 1
                continue
            exterior = [[float(x), float(y)] for x, y in obj["points"]["exterior"]]
            if len(exterior) < 3 or shoelace(exterior) <= 0:
                report["skipped_degenerate_polygon"] += 1
                continue
            if obj["points"].get("interior"):
                # COCO polygon lists cannot express holes; the outer boundary is kept.
                report["polygons_with_holes_kept_as_outer_boundary"] += 1
            xs, ys = [p[0] for p in exterior], [p[1] for p in exterior]
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category_id[name],
                    "segmentation": [[round(v, 2) for point in exterior for v in point]],
                    "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "area": round(shoelace(exterior), 2),
                    "iscrowd": 0,
                }
            )
            report[f"instances:{name}"] += 1
    return coco, dict(sorted(report.items()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--folder", required=True, help="Supervisely project folder under --data-root"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expect-class", action="append", default=[], help="class that meta.json must list"
    )
    args = parser.parse_args()
    if args.out.exists():
        print(f"error: {args.out} exists; choose a new file", file=sys.stderr)
        return 1
    coco, report = convert(args.data_root, args.folder, args.expect_class)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(coco), encoding="utf-8")
    print(json.dumps({"out": str(args.out), **report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
