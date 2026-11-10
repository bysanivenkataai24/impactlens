"""``impactlens export`` - package a trained checkpoint for the inference service.

The deployable artifact is a directory::

    models/<model_id>/
        <model_id>.pt          the checkpoint (provenance embedded)
        model_manifest.json    identity, hash, classes, operating point, metrics, licences
        SHA256SUMS             for `sha256sum -c`
        service.env            the exact environment variables for the service

The checkpoint stays in PyTorch ``.pt`` form, which is a pickle. It is safe to
load only because the operator pins its SHA-256 (``MODEL_SHA256``) and the
service verifies that hash before deserializing. Never load a ``.pt`` file of
unknown origin.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ..taxonomy import MODEL_DAMAGE_TYPES
from .provenance import PURPOSE_SMOKE, sha256_file, utc_now

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{2,63}$")


class ExportError(RuntimeError):
    pass


def export_artifact(
    run_dir: Path,
    *,
    model_id: str,
    out_root: Path = Path("models"),
    which: str = "best",
    evaluation_file: Path | None = None,
    operating_point_file: Path | None = None,
    allow_smoke: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    if not _MODEL_ID.match(model_id):
        raise ExportError("model id: 3-64 characters from letters, digits, '.', '_' and '-'")
    summary_file = run_dir / "train_summary.json"
    source = run_dir / "weights" / f"{which}.pt"
    if not summary_file.is_file() or not source.is_file():
        raise ExportError(
            f"{run_dir} is not a completed `impactlens train` run ({which}.pt missing)"
        )
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    recorded = summary["weights"][which]["sha256"]
    actual = sha256_file(source)
    if actual != recorded:
        raise ExportError(f"{source} changed after training (hash differs from train_summary.json)")

    is_smoke = summary.get("purpose") == PURPOSE_SMOKE
    if is_smoke and not allow_smoke:
        raise ExportError("this run is a pipeline smoke test, not a model. Nothing was exported.")
    if is_smoke and not model_id.startswith("smoke-"):
        raise ExportError("a smoke-test artifact must have a model id starting with 'smoke-'")

    evaluation = None
    if evaluation_file is not None:
        evaluation = json.loads(Path(evaluation_file).read_text(encoding="utf-8"))
        if evaluation["weights"]["sha256"] != actual:
            raise ExportError("the evaluation file belongs to a different checkpoint")
    operating_point = None
    if operating_point_file is not None:
        operating_point = json.loads(Path(operating_point_file).read_text(encoding="utf-8"))
        if operating_point["weights_sha256"] != actual:
            raise ExportError("the operating point file belongs to a different checkpoint")
        if operating_point.get("selected_on_split") != "val":
            raise ExportError("the operating point must have been selected on the validation split")

    target_dir = Path(out_root) / model_id
    if target_dir.exists():
        raise ExportError(f"{target_dir} already exists; model ids are immutable, choose a new one")
    target_dir.mkdir(parents=True)
    target = target_dir / f"{model_id}.pt"
    shutil.copyfile(source, target)

    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    dataset = json.loads((run_dir / "dataset_revision.json").read_text(encoding="utf-8"))
    manifest = {
        "kind": "impactlens.model_manifest",
        "format_version": 1,
        "model_id": model_id,
        "purpose": summary.get("purpose"),
        "created_utc": utc_now(),
        "checkpoint": {
            "file": target.name,
            "sha256": actual,
            "format": "pytorch-pt (pickle; load only with a pinned hash)",
        },
        "task": "segment",
        "class_names": list(MODEL_DAMAGE_TYPES),
        "image_size": summary["image_size"],
        "operating_point": operating_point,
        "training": {
            k: summary.get(k)
            for k in (
                "experiment",
                "run",
                "config_digest",
                "init_checkpoint",
                "init_checkpoint_sha256",
                "epochs_run",
                "trained_utc",
                "git_commit",
            )
        },
        "dataset": {
            "dataset_revision": dataset.get("dataset_revision"),
            "sources": dataset.get("split_report", {}).get("sources"),
            "licenses": dataset.get("split_report", {}).get("licenses"),
            "annotation_revisions": dataset.get("split_report", {}).get("annotation_revisions"),
        },
        "environment": {
            k: environment.get(k) for k in ("python", "torch", "ultralytics", "platform")
        },
        "evaluation": None
        if evaluation is None
        else {
            "split": evaluation["dataset"]["split"],
            "mask_map50_95": evaluation["mask"]["map50_95"],
            "mask_map50": evaluation["mask"]["map50"],
            "box_map50_95": evaluation["box"]["map50_95"],
            "per_class_mask_ap50_95": {
                k: v["ap50_95"] for k, v in evaluation["mask"]["per_class"].items()
            },
            "operating_point_instances": evaluation["operating_point"]["instances"],
            "negative_image_false_positive_rate": evaluation["operating_point"]["negative_images"][
                "image_level_false_positive_rate"
            ],
        },
        "licenses": {
            "checkpoint": "AGPL-3.0 (trained with Ultralytics from Ultralytics pretrained weights; "
            "an Ultralytics Enterprise licence is the alternative)",
            "note": "The training data licences listed under 'dataset' also govern how this checkpoint may be used.",
        },
        "limitations": [
            "Decision support only; every region needs human review.",
            "Scores are uncalibrated.",
            "No part, severity or cost prediction.",
        ],
    }
    if evaluation is None:
        manifest["limitations"].insert(
            0, "NOT EVALUATED: no evaluation file was attached at export."
        )
    if is_smoke:
        manifest["limitations"].insert(
            0, "SMOKE TEST ARTIFACT: trained on abstract shapes, not on vehicles. Do not deploy."
        )
    (target_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (target_dir / "SHA256SUMS").write_text(f"{actual}  {target.name}\n", encoding="utf-8")

    env_lines = [
        "# Generated by `impactlens export`. Add API_KEY as a secret; never store it here.",
        f"MODEL_PATH={target.as_posix()}",
        f"MODEL_ID={model_id}",
        f"MODEL_REVISION={summary.get('run')}",
        f"MODEL_SHA256={actual}",
        f"INFERENCE_IMAGE_SIZE={summary['image_size']}",
        "DEVICE=cpu",
    ]
    if operating_point:
        env_lines += [
            f"NMS_IOU={operating_point['nms_iou']}",
            f"DEFAULT_SCORE_THRESHOLD={operating_point['score_threshold']}",
        ]
    (target_dir / "service.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    return target_dir
