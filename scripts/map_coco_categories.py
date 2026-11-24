"""Write a COPY of a COCO JSON with category names mapped to the ImpactLens taxonomy.

The input file is never modified. Categories without a mapping keep their original
name, so the manifest step must drop them explicitly with --ignore-category; nothing
is merged into a damage type that it is not.

    python scripts/map_coco_categories.py --mapping hitl \
        --in data/raw/hitl/coco/hitl_damage_original.json \
        --out data/raw/hitl/coco/hitl_damage_impactlens.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Humans in the Loop "Car Parts and Car Damages" (CC0-1.0), damage classes.
# Kept: the three classes that mean the same thing as an ImpactLens type.
# Dropped (no ImpactLens equivalent; pass each to --ignore-category):
#   Missing part, Broken part, Flaking, Paint chip, Corrosion.
#   "Broken part" is not mapped to lamp_broken or crack: it covers any broken
#   component (bumper clips, trims, lamps), so the type cannot be told from the label.
MAPPINGS: dict[str, dict[str, str]] = {
    "hitl": {"Dent": "dent", "Scratch": "scratch", "Cracked": "crack"},
}
DROPPED: dict[str, list[str]] = {
    "hitl": ["Missing part", "Broken part", "Flaking", "Paint chip", "Corrosion"],
}


def map_categories(coco: dict, mapping: dict[str, str]) -> tuple[dict, dict]:
    old_name = {c["id"]: c["name"] for c in coco["categories"]}
    unknown = sorted(set(mapping) - set(old_name.values()))
    if unknown:
        raise SystemExit(f"mapping names not found in the file: {unknown}")
    new_names: list[str] = []
    for name in old_name.values():
        target = mapping.get(name, name)
        if target not in new_names:
            new_names.append(target)
    new_id = {name: i + 1 for i, name in enumerate(new_names)}
    remap = {cid: new_id[mapping.get(name, name)] for cid, name in old_name.items()}

    out = {**coco, "categories": [{"id": new_id[n], "name": n} for n in new_names]}
    out["annotations"] = [
        {**a, "category_id": remap[a["category_id"]]} for a in coco["annotations"]
    ]
    counts = Counter(new_names[a["category_id"] - 1] for a in out["annotations"])
    return out, {"renamed": dict(mapping), "instances_by_category": dict(counts.most_common())}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mapping", choices=sorted(MAPPINGS), required=True)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() == args.source.resolve():
        print(
            "error: --out must be a different file; the original is never modified", file=sys.stderr
        )
        return 1
    if args.out.exists():
        print(f"error: {args.out} exists; choose a new file", file=sys.stderr)
        return 1
    coco = json.loads(args.source.read_text(encoding="utf-8"))
    out, report = map_categories(coco, MAPPINGS[args.mapping])
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(
        json.dumps(
            {"out": str(args.out), **report, "drop_with_ignore_category": DROPPED[args.mapping]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
