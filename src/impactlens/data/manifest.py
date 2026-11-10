"""Dataset manifest: provenance, de-duplication and leakage-safe splitting.

The manifest is one CSV row per image file. It is the single source of truth
for *which* image is in *which* split and *why*, and it is what a dataset
revision id is computed from.

Pipeline (each step is a CLI command, see ``impactlens data --help``):

1. ``manifest``  - read COCO annotations + image files, record provenance,
   hashes, EXIF-oriented size, class counts and grouping ids.
2. ``dedup``     - drop exact duplicates (SHA-256) and near-duplicates
   (perceptual hash), across all sources, *before* any split exists.
3. ``split``     - assign whole leakage groups to train/val/test. A leakage
   group is the connected set of images that share an incident, a vehicle, or
   a near-duplicate relation. No group is ever divided.
4. ``stats``     - print the counts that actually resulted.

Nothing here invents labels, and nothing here downloads or redistributes data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..taxonomy import MODEL_DAMAGE_TYPES, normalize_class_name

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NEGATIVE_LABEL = "__negative__"

CORE_COLUMNS = [
    "image_id",
    "source",
    "license",
    "annotation_revision",
    "image_path",
    "annotation_file",
    "coco_image_id",
    "sha256",
    "phash",
    "width",
    "height",
    "exif_orientation",
    "incident_id",
    "vehicle_group_id",
    "group_source",
    "leakage_group",
    "n_instances",
    "class_counts",
    "is_negative",
    "status",
    "duplicate_of",
    "split",
]

STATUS_KEPT = "kept"
STATUS_EXACT_DUP = "dropped_exact_duplicate"
STATUS_NEAR_DUP = "dropped_near_duplicate"
STATUS_ORIENTATION = "excluded_orientation_conflict"
STATUS_SIZE_MISMATCH = "excluded_size_mismatch"


class ManifestError(ValueError):
    """The data cannot be turned into a trustworthy manifest as given."""


@dataclass
class Row:
    image_id: str
    source: str
    license: str
    annotation_revision: str
    image_path: str
    annotation_file: str = ""
    coco_image_id: str = ""
    sha256: str = ""
    phash: str = ""
    width: int = 0
    height: int = 0
    exif_orientation: int = 1
    incident_id: str = ""
    vehicle_group_id: str = ""
    group_source: str = ""
    leakage_group: str = ""
    n_instances: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    is_negative: bool = False
    status: str = STATUS_KEPT
    duplicate_of: str = ""
    split: str = ""
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def kept(self) -> bool:
        return self.status == STATUS_KEPT


# --------------------------------------------------------------------------- io
def write_manifest(rows: list[Row], path: Path) -> None:
    meta_columns = sorted({key for row in rows for key in row.meta})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CORE_COLUMNS + meta_columns)
        for row in sorted(rows, key=lambda r: r.image_id):
            values = []
            for column in CORE_COLUMNS:
                value = getattr(row, column)
                if column == "class_counts":
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"))
                elif isinstance(value, bool):
                    value = "1" if value else "0"
                values.append(value)
            values.extend(row.meta.get(column, "") for column in meta_columns)
            writer.writerow(values)


def read_manifest(path: Path) -> list[Row]:
    rows: list[Row] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in CORE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"{path}: missing manifest columns {missing}")
        for record in reader:
            rows.append(
                Row(
                    image_id=record["image_id"],
                    source=record["source"],
                    license=record["license"],
                    annotation_revision=record["annotation_revision"],
                    image_path=record["image_path"],
                    annotation_file=record["annotation_file"],
                    coco_image_id=record["coco_image_id"],
                    sha256=record["sha256"],
                    phash=record["phash"],
                    width=int(record["width"]),
                    height=int(record["height"]),
                    exif_orientation=int(record["exif_orientation"] or 1),
                    incident_id=record["incident_id"],
                    vehicle_group_id=record["vehicle_group_id"],
                    group_source=record["group_source"],
                    leakage_group=record["leakage_group"],
                    n_instances=int(record["n_instances"]),
                    class_counts=json.loads(record["class_counts"] or "{}"),
                    is_negative=record["is_negative"] == "1",
                    status=record["status"],
                    duplicate_of=record["duplicate_of"],
                    split=record["split"],
                    meta={k: v for k, v in record.items() if k.startswith("meta_") and v},
                )
            )
    return rows


def manifest_revision(rows: list[Row]) -> str:
    """Content hash of everything that defines the dataset (not of file paths)."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.image_id):
        digest.update(
            "|".join(
                [
                    row.image_id,
                    row.sha256,
                    row.annotation_revision,
                    row.status,
                    row.split,
                    json.dumps(row.class_counts, sort_keys=True),
                ]
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()[:16]


# ------------------------------------------------------------------- hashing
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    """Perceptual signature of the EXIF-oriented image: 64-bit dHash + mean RGB (22 hex chars).

    The dHash part is robust to re-encoding and resizing; it is *not* robust to
    crops or new viewpoints, so two different photos of one dent will usually
    not collide. That is why incident / vehicle ids are still required for a
    leakage-free split. The mean-colour part exists because dHash degenerates
    to all-zero on featureless images (plain walls, blank documents), which
    would otherwise all look like duplicates of each other.
    """
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    mean = np.asarray(
        image.convert("RGB").resize((1, 1), Image.Resampling.BOX), dtype=np.uint8
    ).reshape(3)
    return f"{value:016x}" + "".join(f"{int(channel):02x}" for channel in mean)


def inspect_image(path: Path) -> tuple[int, int, int, str]:
    """Return ``(oriented_width, oriented_height, exif_orientation, phash)``."""
    try:
        with Image.open(path) as image:
            orientation = image.getexif().get(0x0112, 1)
            orientation = (
                orientation if isinstance(orientation, int) and 1 <= orientation <= 8 else 1
            )
            oriented = ImageOps.exif_transpose(image)
            return oriented.width, oriented.height, orientation, difference_hash(oriented)
    except Exception as exc:
        raise ManifestError(f"cannot read image {path}: {type(exc).__name__}") from exc


# ------------------------------------------------------------------ building
def _load_image_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    """Optional CSV keyed by ``file_name``: incident_id, vehicle_group_id, meta_* columns."""
    if path is None:
        return {}
    table: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "file_name" not in (reader.fieldnames or []):
            raise ManifestError(f"{path}: a 'file_name' column is required")
        for record in reader:
            table[record["file_name"]] = {k: (v or "").strip() for k, v in record.items()}
    return table


def build_rows(
    *,
    data_root: Path,
    images_dir: str,
    coco_file: str | None,
    source: str,
    license_text: str,
    annotation_revision: str,
    image_metadata: Path | None = None,
    negatives_dir: str | None = None,
    ignore_categories: Iterable[str] = (),
) -> tuple[list[Row], dict]:
    """Create manifest rows for one source. Returns ``(rows, report)``."""
    data_root = Path(data_root)
    if not source or "/" in source:
        raise ManifestError("--source must be a short name without '/'")
    if not license_text.strip():
        raise ManifestError("--license is required: record the terms the data was obtained under")
    ignore = {normalize_class_name(c) for c in ignore_categories}
    metadata = _load_image_metadata(image_metadata)
    rows: list[Row] = []
    report: dict = {"source": source, "ignored_category_instances": Counter(), "crowd_instances": 0}

    def make_row(
        relative: Path, file_name: str, *, coco_id: str, counts: Counter, negative_type: str = ""
    ):
        absolute = data_root / relative
        if not absolute.is_file():
            raise ManifestError(f"image listed but not found: {relative}")
        width, height, orientation, phash = inspect_image(absolute)
        extra = metadata.get(file_name, {})
        meta = {k: v for k, v in extra.items() if k.startswith("meta_") and v}
        if negative_type:
            meta["meta_negative_type"] = negative_type
        incident, vehicle = extra.get("incident_id", ""), extra.get("vehicle_group_id", "")
        return Row(
            image_id=f"{source}/{Path(file_name).with_suffix('').as_posix()}",
            source=source,
            license=license_text.strip(),
            annotation_revision=annotation_revision,
            image_path=relative.as_posix(),
            annotation_file=coco_file or "",
            coco_image_id=coco_id,
            sha256=file_sha256(absolute),
            phash=phash,
            width=width,
            height=height,
            exif_orientation=orientation,
            incident_id=incident,
            vehicle_group_id=vehicle,
            group_source="provided" if (incident or vehicle) else "missing",
            n_instances=sum(counts.values()),
            class_counts=dict(sorted(counts.items())),
            is_negative=sum(counts.values()) == 0,
            meta=meta,
        )

    if coco_file:
        coco = json.loads((data_root / coco_file).read_text(encoding="utf-8"))
        categories = {c["id"]: normalize_class_name(c["name"]) for c in coco.get("categories", [])}
        unknown = sorted(set(categories.values()) - set(MODEL_DAMAGE_TYPES) - ignore)
        if unknown:
            raise ManifestError(
                f"{coco_file}: categories outside the supported taxonomy: {unknown}. Map them "
                "in the annotation tool, or pass --ignore-category to drop those instances."
            )
        per_image: dict[object, Counter] = defaultdict(Counter)
        for annotation in coco.get("annotations", []):
            name = categories.get(annotation.get("category_id"))
            if name is None:
                raise ManifestError(f"{coco_file}: annotation with unknown category id")
            if name in ignore:
                report["ignored_category_instances"][name] += 1
                continue
            if annotation.get("iscrowd"):
                report["crowd_instances"] += 1
                continue
            per_image[annotation["image_id"]][name] += 1
        for image in coco.get("images", []):
            file_name = image["file_name"]
            row = make_row(
                Path(images_dir) / file_name,
                file_name,
                coco_id=str(image["id"]),
                counts=per_image.get(image["id"], Counter()),
            )
            # Annotation coordinates must refer to the EXIF-oriented pixels, because that
            # is what the service, the UI and the exported training images all use. If the
            # annotation file was written against the raw sensor grid we cannot tell which
            # way to rotate safely, so the image is excluded and reported, never guessed.
            declared = (image.get("width"), image.get("height"))
            if all(declared) and declared != (row.width, row.height):
                rotated = declared == (row.height, row.width) and row.exif_orientation >= 5
                row.status = STATUS_ORIENTATION if rotated else STATUS_SIZE_MISMATCH
                report.setdefault("excluded", []).append([row.image_id, row.status])
            if row.exif_orientation != 1:
                report.setdefault("exif_oriented_images", []).append(row.image_id)
            rows.append(row)

    if negatives_dir:
        base = data_root / negatives_dir
        if not base.is_dir():
            raise ManifestError(f"negatives directory not found: {negatives_dir}")
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                relative = path.relative_to(data_root)
                inner = path.relative_to(base)
                negative_type = inner.parts[0] if len(inner.parts) > 1 else "unspecified"
                rows.append(
                    make_row(
                        relative,
                        f"negatives/{inner.as_posix()}",
                        coco_id="",
                        counts=Counter(),
                        negative_type=negative_type,
                    )
                )

    ids = Counter(row.image_id for row in rows)
    repeated = [image_id for image_id, n in ids.items() if n > 1]
    if repeated:
        raise ManifestError(f"duplicate image ids within source {source!r}: {repeated[:5]}")
    report["ignored_category_instances"] = dict(report["ignored_category_instances"])
    report["images"] = len(rows)
    return rows, report


# --------------------------------------------------------------- union-find
class _DisjointSet:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        parent = self._parent.setdefault(item, item)
        while parent != self._parent[parent]:
            self._parent[parent] = self._parent[self._parent[parent]]
            parent = self._parent[parent]
        self._parent[item] = parent
        return parent

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Deterministic: the lexicographically smaller root wins.
            low, high = sorted((root_a, root_b))
            self._parent[high] = low


# -------------------------------------------------------------------- dedup
def deduplicate(rows: list[Row], *, hamming_threshold: int = 4, colour_tolerance: int = 12) -> dict:
    """Mark exact and near-duplicates in place. Returns a report with reviewable pairs.

    The representative kept for a cluster is the image with the most annotated
    instances, then the most pixels, then the smallest id. Dropped images stay
    in the manifest (with ``status`` and ``duplicate_of``) so the decision is
    auditable and so their group ids still link leakage groups together.
    """
    for row in rows:
        if row.status in (STATUS_EXACT_DUP, STATUS_NEAR_DUP):
            row.status, row.duplicate_of = STATUS_KEPT, ""
    candidates = [row for row in rows if row.kept]
    by_id = {row.image_id: row for row in candidates}
    clusters = _DisjointSet()
    exact_pairs: list[tuple[str, str]] = []
    near_pairs: list[tuple[str, str, int]] = []

    by_hash: dict[str, list[Row]] = defaultdict(list)
    for row in candidates:
        by_hash[row.sha256].append(row)
    for group in by_hash.values():
        for other in group[1:]:
            clusters.union(group[0].image_id, other.image_id)
            exact_pairs.append((group[0].image_id, other.image_id))

    if hamming_threshold >= 0 and len(candidates) > 1:
        hashes = np.array([int(row.phash[:16], 16) for row in candidates], dtype=np.uint64)
        colours = np.array(
            [[int(row.phash[16 + 2 * k : 18 + 2 * k], 16) for k in range(3)] for row in candidates],
            dtype=np.int16,
        )
        for i in range(len(candidates) - 1):
            distance = np.bitwise_count(hashes[i + 1 :] ^ hashes[i])
            colour_gap = np.abs(colours[i + 1 :] - colours[i]).max(axis=1)
            close = (distance <= hamming_threshold) & (colour_gap <= colour_tolerance)
            for offset in np.nonzero(close)[0]:
                a, b = candidates[i], candidates[i + 1 + int(offset)]
                if a.sha256 != b.sha256:
                    near_pairs.append((a.image_id, b.image_id, int(distance[offset])))
                clusters.union(a.image_id, b.image_id)

    members: dict[str, list[Row]] = defaultdict(list)
    for row in candidates:
        members[clusters.find(row.image_id)].append(row)
    dropped_exact = dropped_near = 0
    for group in members.values():
        if len(group) == 1:
            continue
        keeper = min(group, key=lambda r: (-r.n_instances, -(r.width * r.height), r.image_id))
        for row in group:
            if row is keeper:
                continue
            row.duplicate_of = keeper.image_id
            if row.sha256 == keeper.sha256:
                row.status = STATUS_EXACT_DUP
                dropped_exact += 1
            else:
                row.status = STATUS_NEAR_DUP
                dropped_near += 1

    # Duplicates across differing annotations deserve a look by a human.
    conflicts = [
        (a, b)
        for a, b, *_ in [*exact_pairs, *near_pairs]
        if by_id[a].class_counts != by_id[b].class_counts
    ]
    return {
        "hamming_threshold": hamming_threshold,
        "colour_tolerance": colour_tolerance,
        "candidates": len(candidates),
        "dropped_exact_duplicates": dropped_exact,
        "dropped_near_duplicates": dropped_near,
        "near_duplicate_pairs": [list(pair) for pair in near_pairs],
        "pairs_with_different_annotations": [list(pair) for pair in conflicts],
        "note": (
            "Near-duplicate detection is a perceptual-hash heuristic. Review "
            "near_duplicate_pairs before trusting the result."
        ),
    }


# ----------------------------------------------------------------- grouping
def assign_leakage_groups(rows: list[Row], *, allow_ungrouped: bool) -> dict:
    """Compute ``leakage_group`` for every row from incident, vehicle and duplicate links."""
    ungrouped = [r for r in rows if r.kept and not r.incident_id and not r.vehicle_group_id]
    if ungrouped and not allow_ungrouped:
        examples = [r.image_id for r in ungrouped[:5]]
        raise ManifestError(
            f"{len(ungrouped)} image(s) have neither incident_id nor vehicle_group_id "
            f"(e.g. {examples}). Views of one vehicle or accident could then land in "
            "different splits. Provide ids via --image-metadata, or pass --allow-ungrouped "
            "to accept that risk explicitly (it is recorded in the split report)."
        )
    links = _DisjointSet()
    for row in rows:
        node = f"img:{row.image_id}"
        links.find(node)
        if row.incident_id:
            links.union(node, f"incident:{row.source}:{row.incident_id}")
        if row.vehicle_group_id:
            links.union(node, f"vehicle:{row.source}:{row.vehicle_group_id}")
        if row.duplicate_of:
            links.union(node, f"img:{row.duplicate_of}")
    roots: dict[str, str] = {}
    for row in rows:
        root = links.find(f"img:{row.image_id}")
        row.leakage_group = roots.setdefault(
            root, "g" + hashlib.sha256(root.encode()).hexdigest()[:12]
        )
        if row.kept and not row.incident_id and not row.vehicle_group_id:
            row.group_source = "singleton_unverified"
    return {
        "leakage_groups": len({r.leakage_group for r in rows if r.kept}),
        "images_without_group_ids": len(ungrouped),
    }


# ------------------------------------------------------------------ splitting
def _labels_of(row: Row) -> Counter:
    counts = Counter(row.class_counts)
    if row.is_negative:
        counts[NEGATIVE_LABEL] += 1
    return counts


def split_groups(
    rows: list[Row],
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
    previous: list[Row] | None = None,
) -> dict:
    """Assign every kept image a split, moving whole leakage groups only.

    Iterative stratification (Sechidis et al., 2011) over groups: the rarest
    remaining label is placed first, each group going to the split that still
    wants that label most. ``previous`` pins groups that already have a split in
    an earlier manifest, so an existing test set neither shrinks nor absorbs
    images that were ever used for training or validation.
    """
    if abs(sum(fractions) - 1.0) > 1e-6 or min(fractions) < 0:
        raise ManifestError("split fractions must be non-negative and sum to 1")
    kept = [row for row in rows if row.kept]
    if any(not row.leakage_group for row in kept):
        raise ManifestError("leakage groups are missing; run group assignment first")
    rng = random.Random(seed)  # noqa: S311 - reproducible shuffling, not security

    groups: dict[str, list[Row]] = defaultdict(list)
    for row in kept:
        groups[row.leakage_group].append(row)
    group_labels = {
        g: sum((_labels_of(r) for r in members), Counter()) for g, members in groups.items()
    }

    # --- pin groups that already had a split -------------------------------
    pinned: dict[str, str] = {}
    if previous:
        earlier = {r.sha256: r.split for r in previous if r.split in SPLITS and r.kept}
        for group, members in groups.items():
            seen = {earlier[r.sha256] for r in members if r.sha256 in earlier}
            if "test" in seen and len(seen) > 1:
                raise ManifestError(
                    f"leakage group {group} links images previously in test with images "
                    f"previously in {sorted(seen - {'test'})}. Resolve this by hand: the test "
                    "set must stay disjoint from everything used for development."
                )
            if seen:
                order = Counter(earlier[r.sha256] for r in members if r.sha256 in earlier)
                pinned[group] = order.most_common(1)[0][0]

    assignment: dict[str, str] = dict(pinned)
    totals = sum(group_labels.values(), Counter())
    total_images = len(kept)
    desire = {
        split: {label: totals[label] * fraction for label in totals}
        for split, fraction in zip(SPLITS, fractions, strict=True)
    }
    image_desire = {s: total_images * f for s, f in zip(SPLITS, fractions, strict=True)}

    def place(group: str, split: str) -> None:
        assignment[group] = split
        for label, count in group_labels[group].items():
            desire[split][label] -= count
        image_desire[split] -= len(groups[group])

    for group, split in pinned.items():
        place(group, split)

    free = [g for g in sorted(groups) if g not in assignment]
    rng.shuffle(free)
    remaining = set(free)
    allowed = [s for s, f in zip(SPLITS, fractions, strict=True) if f > 0]
    while remaining:
        left = sum((group_labels[g] for g in remaining), Counter())
        if left:
            label = min(left, key=lambda name: (left[name], name))
            batch = [g for g in free if g in remaining and group_labels[g][label] > 0]
            batch.sort(key=lambda g: -group_labels[g][label])
        else:
            label, batch = None, [g for g in free if g in remaining]
        for group in batch:
            split = max(
                allowed,
                key=lambda s: (
                    desire[s].get(label, 0.0) if label else 0.0,
                    image_desire[s],
                    -SPLITS.index(s),
                ),
            )
            place(group, split)
            remaining.discard(group)

    for row in rows:
        row.split = assignment[row.leakage_group] if row.kept else ""
    return {
        "seed": seed,
        "fractions": dict(zip(SPLITS, fractions, strict=True)),
        "pinned_groups": len(pinned),
    }


# --------------------------------------------------------------------- stats
def split_report(rows: list[Row], *, min_eval_instances: int = 30) -> dict:
    """Counts that actually resulted, plus explicit coverage warnings."""
    kept = [row for row in rows if row.kept]
    per_split: dict[str, dict] = {}
    for split in SPLITS:
        members = [row for row in kept if row.split == split]
        classes = sum((Counter(row.class_counts) for row in members), Counter())
        per_split[split] = {
            "images": len(members),
            "leakage_groups": len({row.leakage_group for row in members}),
            "incidents": len({(row.source, row.incident_id) for row in members if row.incident_id}),
            "vehicle_groups": len(
                {(row.source, row.vehicle_group_id) for row in members if row.vehicle_group_id}
            ),
            "negative_images": sum(row.is_negative for row in members),
            "negative_types": dict(
                Counter(
                    row.meta.get("meta_negative_type", "unspecified")
                    for row in members
                    if row.is_negative
                )
            ),
            "instances": {name: classes.get(name, 0) for name in MODEL_DAMAGE_TYPES},
            "images_with_class": {
                name: sum(1 for row in members if row.class_counts.get(name))
                for name in MODEL_DAMAGE_TYPES
            },
        }

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in kept:
        group_splits[row.leakage_group].add(row.split)
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)

    warnings: list[str] = []
    if leaking:
        warnings.append(f"LEAKAGE: {len(leaking)} group(s) span more than one split")
    unverified = sum(1 for row in kept if row.group_source == "singleton_unverified")
    if unverified:
        warnings.append(
            f"{unverified} image(s) have no incident/vehicle id. Cross-split leakage of the "
            "same vehicle cannot be ruled out for them; reported metrics may be optimistic."
        )
    for split in ("val", "test"):
        for name in MODEL_DAMAGE_TYPES:
            count = per_split[split]["instances"][name]
            if count == 0:
                warnings.append(f"{split}: no '{name}' instances - this class cannot be evaluated")
            elif count < min_eval_instances:
                warnings.append(
                    f"{split}: only {count} '{name}' instances (< {min_eval_instances}); "
                    "per-class AP will be unstable"
                )
        if per_split[split]["negative_images"] == 0:
            warnings.append(
                f"{split}: no negative images (intact vehicles / out-of-domain scenes), so "
                "false-positive rates on undamaged input cannot be measured"
            )
    statuses = Counter(row.status for row in rows)
    return {
        "dataset_revision": manifest_revision(rows),
        "images_total": len(rows),
        "images_kept": len(kept),
        "status_counts": dict(statuses),
        "sources": dict(Counter(row.source for row in kept)),
        "licenses": sorted({row.license for row in kept}),
        "annotation_revisions": sorted({row.annotation_revision for row in kept}),
        "splits": per_split,
        "groups_spanning_splits": leaking,
        "warnings": warnings,
    }
