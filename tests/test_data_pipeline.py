"""Data pipeline tests: manifest, de-duplication, leakage-free splitting, label export.

Uses small generated images. No real dataset is needed or bundled.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from impactlens.data import manifest as m
from impactlens.data.synthetic import SMOKE_LICENSE, SMOKE_SOURCE, generate_smoke_dataset
from impactlens.data.yolo_export import export_yolo, repeat_factors
from impactlens.taxonomy import MODEL_DAMAGE_TYPES


@pytest.fixture(scope="module")
def smoke_rows(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    generate_smoke_dataset(root, incidents=24, seed=3)
    rows, report = m.build_rows(
        data_root=root / "raw",
        images_dir="images",
        coco_file="annotations.json",
        source=SMOKE_SOURCE,
        license_text=SMOKE_LICENSE,
        annotation_revision="r1",
        image_metadata=root / "raw" / "groups.csv",
        negatives_dir="negatives",
    )
    return root, rows, report


def _fresh(rows):
    """Independent copies, so one test's split cannot leak into another's."""
    return [
        m.Row(**{**r.__dict__, "class_counts": dict(r.class_counts), "meta": dict(r.meta)})
        for r in rows
    ]


def _pipeline(rows, seed=0, fractions=(0.6, 0.2, 0.2)):
    rows = _fresh(rows)
    m.deduplicate(rows)
    m.assign_leakage_groups(rows, allow_ungrouped=True)
    m.split_groups(rows, fractions=fractions, seed=seed)
    return rows


# ----------------------------------------------------------------- manifest
def test_manifest_records_provenance_and_oriented_size(smoke_rows):
    _, rows, report = smoke_rows
    assert report["images"] == len(rows)
    for row in rows:
        assert row.source == SMOKE_SOURCE and row.license == SMOKE_LICENSE
        assert row.annotation_revision == "r1" and len(row.sha256) == 64
        assert row.width > 0 and row.height > 0
    portrait = next(r for r in rows if r.image_id.endswith("portrait_exif6"))
    assert (portrait.width, portrait.height, portrait.exif_orientation) == (256, 320, 6)
    negatives = [r for r in rows if r.is_negative]
    assert negatives and all(r.meta["meta_negative_type"] == "plain_background" for r in negatives)


def test_manifest_round_trip_preserves_revision(smoke_rows, tmp_path):
    _, rows, _ = smoke_rows
    rows = _pipeline(rows)
    m.write_manifest(rows, tmp_path / "manifest.csv")
    again = m.read_manifest(tmp_path / "manifest.csv")
    assert m.manifest_revision(again) == m.manifest_revision(rows)
    assert {r.image_id: r.split for r in again} == {r.image_id: r.split for r in rows}


def test_unknown_categories_are_refused(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "images" / "a.jpg")
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        "annotations": [],
        "categories": [{"id": 1, "name": "dent"}, {"id": 2, "name": "rust"}],
    }
    (tmp_path / "c.json").write_text(json.dumps(coco))
    kwargs = {
        "data_root": tmp_path,
        "images_dir": "images",
        "coco_file": "c.json",
        "source": "s",
        "license_text": "x",
        "annotation_revision": "r",
    }
    with pytest.raises(m.ManifestError, match="rust"):
        m.build_rows(**kwargs)
    rows, _ = m.build_rows(**kwargs, ignore_categories=["rust"])
    assert len(rows) == 1


def test_license_is_mandatory(tmp_path):
    with pytest.raises(m.ManifestError, match="license"):
        m.build_rows(
            data_root=tmp_path,
            images_dir="",
            coco_file=None,
            source="s",
            license_text=" ",
            annotation_revision="r",
        )


def test_annotation_size_mismatch_from_exif_rotation_is_excluded_not_guessed(tmp_path):
    (tmp_path / "images").mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (120, 80)).save(
        tmp_path / "images" / "rot.jpg", exif=exif
    )  # displays as 80x120
    coco = {
        # Annotated against the raw sensor grid (120x80), not the displayed image.
        "images": [{"id": 1, "file_name": "rot.jpg", "width": 120, "height": 80}],
        "annotations": [],
        "categories": [{"id": 1, "name": "dent"}],
    }
    (tmp_path / "c.json").write_text(json.dumps(coco))
    rows, report = m.build_rows(
        data_root=tmp_path,
        images_dir="images",
        coco_file="c.json",
        source="s",
        license_text="x",
        annotation_revision="r",
    )
    assert rows[0].status == m.STATUS_ORIENTATION and not rows[0].kept
    assert report["excluded"] == [["s/rot", m.STATUS_ORIENTATION]]


# -------------------------------------------------------------------- dedup
def test_exact_and_near_duplicates_are_dropped_but_kept_in_manifest(smoke_rows):
    _, rows, _ = smoke_rows
    rows = _fresh(rows)
    report = m.deduplicate(rows)
    by_id = {r.image_id: r for r in rows}
    copy, original, reencoded = (
        by_id[f"{SMOKE_SOURCE}/copy_of_inc-000_view0"],
        by_id[f"{SMOKE_SOURCE}/inc-000_view0"],
        by_id[f"{SMOKE_SOURCE}/reencoded_inc-000_view0"],
    )
    statuses = {copy.status, original.status, reencoded.status}
    assert (
        m.STATUS_KEPT in statuses
        and m.STATUS_EXACT_DUP in statuses
        and m.STATUS_NEAR_DUP in statuses
    )
    assert report["dropped_exact_duplicates"] >= 1 and report["dropped_near_duplicates"] >= 1
    assert all(r.duplicate_of for r in rows if not r.kept)


def test_featureless_images_of_different_colour_are_not_duplicates(tmp_path):
    rows = []
    for index, colour in enumerate([(20, 20, 20), (200, 200, 200), (30, 120, 200)]):
        Image.new("RGB", (64, 64), colour).save(tmp_path / f"{index}.png")
        width, height, _orientation, phash = m.inspect_image(tmp_path / f"{index}.png")
        rows.append(
            m.Row(
                image_id=f"s/{index}",
                source="s",
                license="x",
                annotation_revision="r",
                image_path=f"{index}.png",
                sha256=str(index) * 64,
                phash=phash,
                width=width,
                height=height,
            )
        )
    report = m.deduplicate(rows)
    assert report["dropped_near_duplicates"] == 0 and all(r.kept for r in rows)


# ------------------------------------------------------------------- splits
def test_no_leakage_group_spans_two_splits(smoke_rows):
    _, rows, _ = smoke_rows
    for seed in range(5):
        split_rows = _pipeline(rows, seed=seed)
        report = m.split_report(split_rows)
        assert report["groups_spanning_splits"] == []
        incident_splits: dict = {}
        for row in split_rows:
            if row.kept and row.incident_id:
                incident_splits.setdefault(row.incident_id, set()).add(row.split)
        assert all(len(s) == 1 for s in incident_splits.values())


def test_views_of_one_incident_stay_together(smoke_rows):
    _, rows, _ = smoke_rows
    split_rows = {r.image_id: r for r in _pipeline(rows)}
    for index in range(1, 24):
        a, b = (split_rows[f"{SMOKE_SOURCE}/inc-{index:03d}_view{v}"] for v in (0, 1))
        assert a.split == b.split and a.leakage_group == b.leakage_group


def test_duplicate_links_merge_groups_across_incidents(smoke_rows):
    """A duplicate filed under another incident must still land with its original."""
    _, rows, _ = smoke_rows
    split_rows = {r.image_id: r for r in _pipeline(rows)}
    original = split_rows[f"{SMOKE_SOURCE}/inc-000_view0"]
    for duplicate in ("copy_of_inc-000_view0", "reencoded_inc-000_view0"):
        assert split_rows[f"{SMOKE_SOURCE}/{duplicate}"].leakage_group == original.leakage_group


def test_split_is_deterministic_for_a_seed(smoke_rows):
    _, rows, _ = smoke_rows
    first = {r.image_id: r.split for r in _pipeline(rows, seed=7)}
    second = {r.image_id: r.split for r in _pipeline(rows, seed=7)}
    assert first == second


def test_every_class_reaches_val_and_test_when_possible(smoke_rows):
    _, rows, _ = smoke_rows
    report = m.split_report(_pipeline(rows), min_eval_instances=1)
    for split in ("val", "test"):
        assert all(report["splits"][split]["instances"][name] > 0 for name in MODEL_DAMAGE_TYPES), (
            split
        )


def test_ungrouped_images_need_explicit_consent(smoke_rows):
    _, rows, _ = smoke_rows
    rows = _fresh(rows)
    for row in rows:
        row.incident_id = row.vehicle_group_id = ""
    with pytest.raises(m.ManifestError, match="allow-ungrouped"):
        m.assign_leakage_groups(rows, allow_ungrouped=False)
    m.assign_leakage_groups(rows, allow_ungrouped=True)
    m.split_groups(rows, seed=0)
    assert any("no incident/vehicle id" in w for w in m.split_report(rows)["warnings"])


def test_previous_test_set_is_preserved_when_data_is_added(smoke_rows, tmp_path):
    root, rows, _ = smoke_rows
    first = _pipeline(rows, seed=1)
    test_hashes = {r.sha256 for r in first if r.split == "test"}

    # New data arrives (here: same rows plus extra singletons), and the split is re-run with another seed.
    extra = []
    for index in range(6):
        path = root / "raw" / "images" / f"late_{index}.jpg"
        Image.fromarray(
            np.random.default_rng(index).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        ).save(path)
        width, height, _orientation, phash = m.inspect_image(path)
        extra.append(
            m.Row(
                image_id=f"late/{index}",
                source="late",
                license="x",
                annotation_revision="r",
                image_path=f"images/late_{index}.jpg",
                sha256=m.file_sha256(path),
                phash=phash,
                width=width,
                height=height,
                incident_id=f"late-{index}",
                group_source="provided",
                is_negative=True,
            )
        )
    second = _fresh(rows) + extra
    m.deduplicate(second)
    m.assign_leakage_groups(second, allow_ungrouped=True)
    m.split_groups(second, seed=99, previous=first)
    assert {
        r.sha256 for r in second if r.split == "test" and r.sha256 in test_hashes
    } == test_hashes
    previously_dev = {r.sha256 for r in first if r.split in ("train", "val")}
    assert not any(r.split == "test" and r.sha256 in previously_dev for r in second)


def test_conflicting_previous_assignment_is_an_error():
    rows = [
        m.Row(
            image_id=f"s/{i}",
            source="s",
            license="x",
            annotation_revision="r",
            image_path=f"{i}.jpg",
            sha256=str(i) * 64,
            phash="0" * 22,
            incident_id="same-incident",
            group_source="provided",
        )
        for i in range(2)
    ]
    previous = [
        m.Row(**{**r.__dict__, "split": split})
        for r, split in zip(rows, ("test", "train"), strict=True)
    ]
    m.assign_leakage_groups(rows, allow_ungrouped=False)
    with pytest.raises(m.ManifestError, match="test set must stay disjoint"):
        m.split_groups(rows, previous=previous)


def test_report_flags_missing_classes_and_negatives():
    rows = [
        m.Row(
            image_id=f"s/{i}",
            source="s",
            license="x",
            annotation_revision="r",
            image_path=f"{i}.jpg",
            sha256=str(i) * 64,
            phash="0" * 22,
            incident_id=str(i),
            group_source="provided",
            class_counts={"dent": 1},
            n_instances=1,
        )
        for i in range(10)
    ]
    m.assign_leakage_groups(rows, allow_ungrouped=False)
    m.split_groups(rows, seed=0)
    warnings = m.split_report(rows)["warnings"]
    assert any("no 'tire_flat' instances" in w for w in warnings)
    assert any("no negative images" in w for w in warnings)


# ------------------------------------------------------------------- export
def test_yolo_export_writes_oriented_clean_images_and_matching_labels(smoke_rows, tmp_path):
    root, rows, _ = smoke_rows
    rows = _pipeline(rows)
    out = tmp_path / "dataset"
    revision = export_yolo(rows, data_root=root / "raw", out_dir=out, rfs_threshold=0.3, seed=0)
    assert revision["dataset_revision"] == m.manifest_revision(rows)

    kept = [r for r in rows if r.kept]
    images = sorted((out / "images").rglob("*.*"))
    assert len(images) == len(kept)
    for path in images:
        with Image.open(path) as image:
            assert not image.getexif(), f"{path.name} still carries EXIF"
        label = out / "labels" / path.parent.name / f"{path.stem}.txt"
        assert label.is_file()
        for line in label.read_text().splitlines():
            values = [float(v) for v in line.split()[1:]]
            assert int(line.split()[0]) in range(6) and len(values) % 2 == 0
            assert all(0.0 <= v <= 1.0 for v in values)

    portrait = next(r for r in kept if r.image_id.endswith("portrait_exif6"))
    with Image.open(out / "images" / portrait.split / f"{portrait.sha256[:20]}.jpg") as image:
        assert image.size == (256, 320)  # rotated to display orientation

    negatives = [r for r in kept if r.is_negative]
    for row in negatives:
        assert (out / "labels" / row.split / f"{row.sha256[:20]}.txt").read_text() == ""

    total_labels = Counter()
    for label in (out / "labels").rglob("*.txt"):
        total_labels.update(line.split()[0] for line in label.read_text().splitlines())
    expected = sum(sum(r.class_counts.values()) for r in kept)
    assert sum(total_labels.values()) == expected

    assert (out / "dataset.yaml").is_file() and (out / "dataset_rfs.yaml").is_file()
    index_lines = (out / "image_index.csv").read_text().splitlines()
    assert len(index_lines) == len(kept) + 1 and "meta_negative_type" in index_lines[0]
    assert not any(r.image_path in (out / "image_index.csv").read_text() for r in kept)


def test_export_refuses_a_non_empty_directory(smoke_rows, tmp_path):
    root, rows, _ = smoke_rows
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "old.txt").write_text("x")
    with pytest.raises(m.ManifestError, match="not empty"):
        export_yolo(_pipeline(rows), data_root=root / "raw", out_dir=tmp_path / "dataset")


def test_repeat_factors_upweight_rare_classes_only():
    rows = [
        m.Row(
            image_id=f"s/{i}",
            source="s",
            license="x",
            annotation_revision="r",
            image_path="",
            class_counts={"dent": 1},
        )
        for i in range(9)
    ]
    rows.append(
        m.Row(
            image_id="s/rare",
            source="s",
            license="x",
            annotation_revision="r",
            image_path="",
            class_counts={"tire_flat": 1},
        )
    )
    factors = repeat_factors(rows, threshold=0.3)
    assert factors["s/0"] == 1.0
    assert factors["s/rare"] == pytest.approx((0.3 / 0.1) ** 0.5)


def test_training_config_rejects_unknown_keys(tmp_path):
    from impactlens.training.config import TrainConfigError, load_train_config

    config = tmp_path / "c.yaml"
    config.write_text("experiment: e\ndata: d.yaml\nlearning_rate: 0.1\n")
    with pytest.raises(TrainConfigError, match="learning_rate"):
        load_train_config(config)
    config.write_text("experiment: e\ndata: d.yaml\noptimizer: auto\n")
    with pytest.raises(TrainConfigError, match="optimizer"):
        load_train_config(config)


def test_shipped_training_configs_are_valid():
    from impactlens.training.config import load_train_config

    configs = sorted(Path(__file__).resolve().parents[1].glob("configs/train/*.yaml"))
    assert configs, "no training configs found"
    digests = set()
    for path in configs:
        config = load_train_config(path)
        digests.add(config.digest())
        assert config.optimizer != "auto" and config.seed
    assert len(digests) == len(configs)
