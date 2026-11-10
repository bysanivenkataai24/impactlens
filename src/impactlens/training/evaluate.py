"""``impactlens evaluate`` - measure a checkpoint on one split of an exported dataset.

Two passes, both written to ``evaluation.json``:

A. **Ranking metrics** (Ultralytics validator, score threshold 0.001): box and
   mask AP averaged over IoU 0.50:0.95, AP at each IoU threshold, per-class AP,
   and precision / recall / F1 as a function of the score threshold.

B. **Operating-point analysis** at one score threshold, run through the *same
   adapter the service uses*, so it measures what a reviewer would be shown:
   instance precision / recall (mask IoU >= 0.5), recall by region size,
   cross-class confusions, image-level false positives on negative images
   (intact vehicles, out-of-domain scenes) and breakdowns by any ``meta_*``
   column of the manifest (lighting, colour, angle, camera, ...).

Rules that are enforced, not just documented:

* The threshold is *selected* only on the validation split. Evaluating the
  test split requires passing the threshold explicitly.
* The test split needs ``--final-test`` and every use is appended to
  ``test_evaluations.jsonl`` in the dataset directory, so repeated peeking is
  visible.
* Classes with no ground truth in the split are reported as not evaluated,
  never as a number.

Scores are uncalibrated. Nothing here turns them into probabilities.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from ..adapters.base import RawInstance
from ..adapters.ultralytics_seg import (
    UltralyticsSegAdapter,
    configure_ultralytics_environment,
    disable_ultralytics_analytics,
)
from ..taxonomy import MODEL_DAMAGE_TYPES
from .provenance import PURPOSE_SMOKE, environment_record, sha256_file, utc_now

MATCH_IOU = 0.5
#: Region size as a fraction of the image area.
SIZE_BUCKETS = (
    ("small_lt_0.1pct", 0.0, 0.001),
    ("medium_0.1_to_1pct", 0.001, 0.01),
    ("large_ge_1pct", 0.01, 1.01),
)
_RASTER_LONG_SIDE = 640


class EvaluationError(RuntimeError):
    pass


# ------------------------------------------------------------------ matching
@dataclass
class GroundTruth:
    class_name: str
    polygon: np.ndarray  # (K, 2) pixels
    area_fraction: float


def read_labels(label_file: Path, width: int, height: int) -> list[GroundTruth]:
    """Parse one YOLO segmentation label file into pixel-space polygons."""
    truths: list[GroundTruth] = []
    if not label_file.is_file():
        return truths
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        ring = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2) * (width, height)
        x, y = ring[:, 0], ring[:, 1]
        area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0
        truths.append(
            GroundTruth(MODEL_DAMAGE_TYPES[int(parts[0])], ring, float(area / (width * height)))
        )
    return truths


def _rasterize(polygon: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(1.0, _RASTER_LONG_SIDE / max(width, height))
    canvas = np.zeros((max(1, round(height * scale)), max(1, round(width * scale))), dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(canvas, [np.round(polygon * scale).astype(np.int32)], 1)
    return canvas.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def match_image(
    predictions: list[RawInstance], truths: list[GroundTruth], width: int, height: int
) -> dict:
    """Greedy, score-ordered matching at mask IoU >= 0.5.

    A prediction is a true positive if it matches an unmatched ground truth of
    the same class. Otherwise, if it overlaps an unmatched ground truth of a
    *different* class, the pair is recorded as a class confusion (it still
    counts as one false positive and one false negative).
    """
    pred_masks = [_rasterize(np.asarray(p.polygon_xy), width, height) for p in predictions]
    truth_masks = [_rasterize(t.polygon, width, height) for t in truths]
    iou = np.zeros((len(predictions), len(truths)))
    for i, pm in enumerate(pred_masks):
        for j, tm in enumerate(truth_masks):
            iou[i, j] = mask_iou(pm, tm)

    order = sorted(range(len(predictions)), key=lambda i: -predictions[i].score)
    truth_taken = [False] * len(truths)
    pred_true = [False] * len(predictions)
    for i in order:
        same = [
            j
            for j in range(len(truths))
            if not truth_taken[j] and truths[j].class_name == predictions[i].class_name
        ]
        if same:
            best = max(same, key=lambda j: iou[i, j])
            if iou[i, best] >= MATCH_IOU:
                truth_taken[best], pred_true[i] = True, True

    confusions: list[tuple[str, str]] = []
    confused_truth = [False] * len(truths)
    for i in order:
        if pred_true[i]:
            continue
        others = [j for j in range(len(truths)) if not truth_taken[j] and not confused_truth[j]]
        if others:
            best = max(others, key=lambda j: iou[i, j])
            if iou[i, best] >= MATCH_IOU:
                confused_truth[best] = True
                confusions.append((truths[best].class_name, predictions[i].class_name))
    return {"pred_true": pred_true, "truth_matched": truth_taken, "confusions": confusions}


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }


# ---------------------------------------------------------------- pass B
def operating_point_analysis(
    adapter: UltralyticsSegAdapter,
    dataset_dir: Path,
    split: str,
    score_threshold: float,
    *,
    max_detections: int = 50,
    per_image_csv: Path | None = None,
) -> dict:
    index: dict[str, dict] = {}
    index_file = dataset_dir / "image_index.csv"
    if index_file.is_file():
        with index_file.open(newline="", encoding="utf-8") as handle:
            index = {record["file"]: record for record in csv.DictReader(handle)}

    images = sorted(
        p
        for p in (dataset_dir / "images" / split).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise EvaluationError(f"no images in {dataset_dir / 'images' / split}")

    per_class: dict[str, Counter] = defaultdict(Counter)
    size_counts: dict[str, Counter] = defaultdict(Counter)
    confusion_pairs: Counter = Counter()
    slices: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    negatives: dict[str, Counter] = defaultdict(Counter)
    rows: list[dict] = []

    for path in images:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        truths = read_labels(dataset_dir / "labels" / split / f"{path.stem}.txt", width, height)
        predictions = [
            p
            for p in adapter.predict(
                rgb, score_threshold=score_threshold, max_detections=max_detections
            )
            if len(p.polygon_xy) >= 3
        ]
        result = match_image(predictions, truths, width, height)

        tp = sum(result["pred_true"])
        fp, fn = len(predictions) - tp, len(truths) - sum(result["truth_matched"])
        for prediction, is_true in zip(predictions, result["pred_true"], strict=True):
            per_class[prediction.class_name]["tp" if is_true else "fp"] += 1
        for truth, matched in zip(truths, result["truth_matched"], strict=True):
            if not matched:
                per_class[truth.class_name]["fn"] += 1
            for name, low, high in SIZE_BUCKETS:
                if low <= truth.area_fraction < high:
                    size_counts[name]["total"] += 1
                    size_counts[name]["found"] += int(matched)
        confusion_pairs.update(result["confusions"])

        record = index.get(path.name, {})
        if not truths:
            kind = record.get("meta_negative_type") or "unspecified"
            negatives[kind]["images"] += 1
            negatives[kind]["images_with_detections"] += int(bool(predictions))
            negatives[kind]["detections"] += len(predictions)
        for column, value in record.items():
            if column.startswith("meta_") and value:
                slices[column][value].update({"images": 1, "tp": tp, "fp": fp, "fn": fn})
        rows.append(
            {
                "file": path.name,
                "ground_truth": len(truths),
                "predictions": len(predictions),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                **{k: v for k, v in record.items() if k.startswith("meta_")},
            }
        )

    if per_image_csv is not None:
        columns = sorted(
            {key for row in rows for key in row}, key=lambda k: (k.startswith("meta_"), k)
        )
        with per_image_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    totals = Counter()
    for counts in per_class.values():
        totals.update(counts)
    negative_total = sum((c for c in negatives.values()), Counter())
    return {
        "score_threshold": score_threshold,
        "match_rule": f"greedy by score, same class, mask IoU >= {MATCH_IOU} (masks rasterized at <= {_RASTER_LONG_SIDE}px)",
        "images": len(images),
        "instances": _prf(totals["tp"], totals["fp"], totals["fn"]),
        "per_class": {
            name: _prf(per_class[name]["tp"], per_class[name]["fp"], per_class[name]["fn"])
            for name in MODEL_DAMAGE_TYPES
        },
        "recall_by_region_size": {
            name: {
                "ground_truth": size_counts[name]["total"],
                "found": size_counts[name]["found"],
                "recall": _ratio(size_counts[name]["found"], size_counts[name]["total"]),
            }
            for name, _, _ in SIZE_BUCKETS
        },
        "class_confusions": [
            {"ground_truth": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in sorted(
                confusion_pairs.items(), key=lambda item: -item[1]
            )
        ],
        "negative_images": {
            "images": negative_total["images"],
            "images_with_any_detection": negative_total["images_with_detections"],
            "image_level_false_positive_rate": _ratio(
                negative_total["images_with_detections"], negative_total["images"]
            ),
            "by_type": {
                kind: {
                    **dict(c),
                    "image_level_false_positive_rate": _ratio(
                        c["images_with_detections"], c["images"]
                    ),
                }
                for kind, c in sorted(negatives.items())
            },
            "note": None
            if negative_total["images"]
            else "No negative images in this split: false positives on undamaged or out-of-domain input were NOT measured.",
        },
        "slices": {
            column: {
                value: {"images": c["images"], **_prf(c["tp"], c["fp"], c["fn"])}
                for value, c in sorted(values.items())
            }
            for column, values in sorted(slices.items())
        },
    }


# ---------------------------------------------------------------- pass A
def _metric_block(metric, names: dict[int, str], instances: dict[str, int]) -> dict:
    evaluated = [names[int(i)] for i in metric.ap_class_index]
    all_ap = np.asarray(metric.all_ap, dtype=np.float64).reshape(len(evaluated), -1)
    thresholds = [round(0.5 + 0.05 * i, 2) for i in range(all_ap.shape[1])]
    per_class = {}
    for name in MODEL_DAMAGE_TYPES:
        if name in evaluated:
            row = all_ap[evaluated.index(name)]
            per_class[name] = {
                "ap50_95": round(float(row.mean()), 4),
                "ap50": round(float(row[0]), 4),
                "instances": instances.get(name, 0),
            }
        else:
            per_class[name] = {
                "ap50_95": None,
                "ap50": None,
                "instances": 0,
                "status": "not_evaluated_no_ground_truth_in_split",
            }
    return {
        "map50_95": round(float(all_ap.mean()), 4) if len(evaluated) else None,
        "map50": round(float(all_ap[:, 0].mean()), 4) if len(evaluated) else None,
        "map75": round(float(all_ap[:, 5].mean()), 4) if len(evaluated) else None,
        "map_by_iou_threshold": {
            f"{t:.2f}": round(float(all_ap[:, i].mean()), 4) for i, t in enumerate(thresholds)
        }
        if len(evaluated)
        else {},
        "classes_evaluated": evaluated,
        "per_class": per_class,
    }


def _threshold_sweep(metric) -> tuple[float | None, list[dict]]:
    """Mean-over-classes F1 / precision / recall against the score threshold."""
    if not len(metric.ap_class_index):
        return None, []
    px = np.asarray(metric.px, dtype=np.float64)
    f1 = np.asarray(metric.f1_curve, dtype=np.float64).mean(axis=0)
    precision = np.asarray(metric.p_curve, dtype=np.float64).mean(axis=0)
    recall = np.asarray(metric.r_curve, dtype=np.float64).mean(axis=0)
    best = float(px[int(np.argmax(f1))])
    sweep = []
    for target in np.round(np.arange(0.05, 0.96, 0.05), 2):
        i = int(np.argmin(np.abs(px - target)))
        sweep.append(
            {
                "score_threshold": float(target),
                "mean_f1": round(float(f1[i]), 4),
                "mean_precision": round(float(precision[i]), 4),
                "mean_recall": round(float(recall[i]), 4),
            }
        )
    return round(min(max(best, 0.05), 0.95), 3), sweep


def evaluate(
    weights: Path,
    data_yaml: Path,
    *,
    split: str = "val",
    out_dir: Path,
    image_size: int | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    workers: int = 4,
    nms_iou: float = 0.7,
    score_threshold: float | None = None,
    tile: int | None = None,
    final_test: bool = False,
    plots: bool = False,
) -> dict:
    weights, data_yaml, out_dir = Path(weights), Path(data_yaml).resolve(), Path(out_dir)
    if split not in ("val", "test"):
        raise EvaluationError(
            "split must be 'val' or 'test' (never evaluate on the training split)"
        )
    dataset_dir = data_yaml.parent
    revision_file = dataset_dir / "dataset_revision.json"
    if not revision_file.is_file():
        raise EvaluationError(
            f"{revision_file} not found; evaluate only on `impactlens data export-yolo` output"
        )
    revision = json.loads(revision_file.read_text(encoding="utf-8"))

    if split == "test":
        if not final_test:
            raise EvaluationError(
                "The test split is reserved for the final evaluation of a frozen model. "
                "Pass --final-test to confirm; the use is recorded in test_evaluations.jsonl."
            )
        if score_threshold is None:
            raise EvaluationError(
                "Pass --score-threshold with the value selected on the validation split. "
                "Thresholds are never tuned on test data."
            )

    configure_ultralytics_environment(offline=True)
    from ultralytics import YOLO

    disable_ultralytics_analytics()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_sha256 = sha256_file(weights)

    # The adapter applies the same task / class checks as the service.
    adapter = UltralyticsSegAdapter(
        weights, device=device, image_size=image_size, nms_iou=nms_iou, allow_smoke_checkpoint=True
    )
    image_size = adapter.info.inference_image_size
    tag = adapter.checkpoint_tag or {}

    model = YOLO(str(weights))
    arguments: dict = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": image_size,
        "batch": batch_size,
        "conf": 0.001,
        "iou": nms_iou,
        "device": device,
        "workers": workers,
        "plots": plots,
        "verbose": False,
        "project": str(out_dir.resolve()),
        "name": "ultralytics_val",
        "exist_ok": True,
    }
    if tile:
        arguments["tile"] = int(tile)
    metrics = model.val(**arguments)

    names = {int(k): v for k, v in metrics.names.items()}
    counts = revision.get("split_report", {}).get("splits", {}).get(split, {}).get("instances", {})
    selected, sweep = _threshold_sweep(metrics.seg)
    if split == "val" and score_threshold is None:
        if selected is None:
            raise EvaluationError(
                "the validation split has no ground truth; cannot select a threshold"
            )
        score_threshold, selection = (
            selected,
            "argmax of mean-over-classes mask F1 on the validation split",
        )
    else:
        selection = "provided by caller"

    operating = operating_point_analysis(
        adapter,
        dataset_dir,
        split,
        float(score_threshold),
        per_image_csv=out_dir / f"per_image_{split}.csv",
    )
    operating["selection"] = selection

    confusion = metrics.confusion_matrix.matrix
    labels = [*(names[i] for i in sorted(names)), "background"]
    result = {
        "kind": "impactlens.evaluation",
        "format_version": 1,
        "created_utc": utc_now(),
        "smoke_test": tag.get("purpose") == PURPOSE_SMOKE,
        "weights": {"file": weights.name, "sha256": weights_sha256, "provenance": tag or None},
        "dataset": {
            "dataset_revision": revision.get("dataset_revision"),
            "split": split,
            "images": operating["images"],
            "instances": counts,
        },
        "settings": {
            "image_size": image_size,
            "nms_iou": nms_iou,
            "tile": tile,
            "device": device,
            "ranking_score_threshold": 0.001,
            "batch_size": batch_size,
        },
        "environment": environment_record(device),
        "box": _metric_block(metrics.box, names, counts),
        "mask": _metric_block(metrics.seg, names, counts),
        "threshold_sweep_mask": sweep,
        "f1_optimal_threshold_this_split": selected,
        "operating_point": operating,
        "confusion_matrix_at_0.001": {
            "labels": labels,
            "rows_predicted_columns_true": np.round(confusion, 1).tolist(),
        },
        "validator_speed_ms_per_image": {k: round(float(v), 2) for k, v in metrics.speed.items()},
        "caveats": [
            "Scores are uncalibrated model scores, not probabilities.",
            "Per-class numbers with few instances are unstable; check 'instances' before quoting them.",
            "Speed here is measured inside the validator on this machine and is not a service latency benchmark.",
        ],
    }
    if result["smoke_test"]:
        result["caveats"].insert(
            0,
            "SMOKE TEST: this checkpoint was trained on abstract synthetic shapes. "
            "These numbers verify that the pipeline runs. They say nothing about vehicle damage.",
        )
    if split == "test":
        ledger = dataset_dir / "test_evaluations.jsonl"
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "utc": utc_now(),
                        "weights_sha256": weights_sha256,
                        "score_threshold": score_threshold,
                        "mask_map50_95": result["mask"]["map50_95"],
                    }
                )
                + "\n"
            )
        result["test_set_uses_recorded"] = len(ledger.read_text(encoding="utf-8").splitlines())

    (out_dir / f"evaluation_{split}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if split == "val":
        (out_dir / "operating_point.json").write_text(
            json.dumps(
                {
                    "score_threshold": score_threshold,
                    "selection": selection,
                    "selected_on_split": "val",
                    "nms_iou": nms_iou,
                    "image_size": image_size,
                    "weights_sha256": weights_sha256,
                    "dataset_revision": revision.get("dataset_revision"),
                    "created_utc": utc_now(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    adapter.close()
    return result


def summarize(result: dict) -> str:
    """Short plain-text summary for the terminal."""
    lines = []
    if result.get("smoke_test"):
        lines.append("*** SMOKE TEST - pipeline mechanics only, NOT a model validation ***")
    dataset, mask, box, op = (
        result["dataset"],
        result["mask"],
        result["box"],
        result["operating_point"],
    )
    lines.append(
        f"dataset {dataset['dataset_revision']} split={dataset['split']} images={dataset['images']}"
    )
    lines.append(
        f"mask AP50-95={mask['map50_95']} AP50={mask['map50']} | box AP50-95={box['map50_95']} AP50={box['map50']}"
    )
    for name, values in mask["per_class"].items():
        lines.append(
            f"  {name:<14} mask AP50-95={values['ap50_95']} instances={values['instances']}"
        )
    inst = op["instances"]
    lines.append(
        f"operating point score>={op['score_threshold']} ({op['selection']}): precision={inst['precision']} recall={inst['recall']}"
    )
    neg = op["negative_images"]
    lines.append(
        f"negative images: {neg['images']} | with any detection: {neg['images_with_any_detection']} | rate={neg['image_level_false_positive_rate']}"
    )
    return "\n".join(lines)


def load_dataset_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")).get("names", {})
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
