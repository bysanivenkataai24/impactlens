"""``impactlens smoke`` - run the whole pipeline once on synthetic shapes.

    SMOKE TEST. It proves that data loading, splitting, training, evaluation,
    export and the service adapter all execute on this machine. It is NOT a
    model validation: the data are abstract shapes, the run lasts seconds, and
    the resulting checkpoint knows nothing about vehicles.

The checkpoint is tagged ``smoke_test_not_a_model`` and the inference service
refuses to load it.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..data import manifest as manifest_module
from ..data.synthetic import SMOKE_LICENSE, SMOKE_SOURCE, generate_smoke_dataset
from ..data.yolo_export import export_yolo
from .config import Augmentation, TrainConfig, validate
from .evaluate import evaluate
from .export import export_artifact
from .provenance import PURPOSE_SMOKE, utc_now
from .train import run_training

BANNER = "SMOKE TEST - verifies pipeline mechanics only. NOT a model validation."


def run_smoke(
    work_dir: Path | None = None,
    *,
    init_checkpoint: str = "yolo11n-seg.pt",
    epochs: int = 3,
    image_size: int = 320,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    work = Path(work_dir) if work_dir else Path("runs") / "smoke" / stamp
    work.mkdir(parents=True, exist_ok=False)
    steps: list[dict] = []

    def step(name: str, function):
        started = time.perf_counter()
        value = function()
        steps.append(
            {"step": name, "status": "ok", "seconds": round(time.perf_counter() - started, 2)}
        )
        return value

    generated = step("generate synthetic shapes", lambda: generate_smoke_dataset(work, seed=seed))
    rows, build_report = step(
        "build manifest",
        lambda: manifest_module.build_rows(
            data_root=work / "raw",
            images_dir="images",
            coco_file="annotations.json",
            source=SMOKE_SOURCE,
            license_text=SMOKE_LICENSE,
            annotation_revision="smoke-r1",
            image_metadata=work / "raw" / "groups.csv",
            negatives_dir="negatives",
        ),
    )
    dedup = step(
        "remove duplicates", lambda: manifest_module.deduplicate(rows, hamming_threshold=4)
    )
    step(
        "assign leakage groups",
        lambda: manifest_module.assign_leakage_groups(rows, allow_ungrouped=True),
    )
    step(
        "group-aware split",
        lambda: manifest_module.split_groups(rows, fractions=(0.6, 0.2, 0.2), seed=seed),
    )
    split = manifest_module.split_report(rows, min_eval_instances=5)
    if split["groups_spanning_splits"]:
        raise RuntimeError("smoke test failed: a leakage group spans splits")
    manifest_module.write_manifest(rows, work / "manifest.csv")
    step(
        "export YOLO dataset",
        lambda: export_yolo(
            rows, data_root=work / "raw", out_dir=work / "dataset", rfs_threshold=0.3, seed=seed
        ),
    )

    config = TrainConfig(
        experiment="smoke",
        data=str(work / "dataset" / "dataset.yaml"),
        init_checkpoint=init_checkpoint,
        seed=seed,
        image_size=image_size,
        batch_size=8,
        epochs=epochs,
        patience=0,
        device=device,
        workers=0,
        amp=False,
        plots=False,
        augment=Augmentation(mosaic=0.5, close_mosaic=0),
        notes=BANNER,
    )
    validate(config)
    run_dir = step(
        "train", lambda: run_training(config, runs_dir=work / "runs", purpose=PURPOSE_SMOKE)
    )
    best = run_dir / "weights" / "best.pt"
    evaluation = step(
        "evaluate (validation split)",
        lambda: evaluate(
            best,
            work / "dataset" / "dataset.yaml",
            split="val",
            out_dir=run_dir / "evaluation",
            batch_size=8,
            device=device,
            workers=0,
        ),
    )
    artifact = step(
        "export artifact",
        lambda: export_artifact(
            run_dir,
            model_id="smoke-not-a-model",
            out_root=work / "models",
            evaluation_file=run_dir / "evaluation" / "evaluation_val.json",
            operating_point_file=run_dir / "evaluation" / "operating_point.json",
            allow_smoke=True,
        ),
    )

    def service_guard() -> dict:
        """The service path must refuse this checkpoint; the explicit test path must load it."""
        from ..adapters.base import ModelUnavailable
        from ..adapters.ultralytics_seg import UltralyticsSegAdapter

        checkpoint = artifact / "smoke-not-a-model.pt"
        try:
            UltralyticsSegAdapter(checkpoint, device=device)
        except ModelUnavailable as exc:
            refused = exc.code
        else:
            raise RuntimeError("smoke test failed: the service adapter accepted a smoke checkpoint")
        adapter = UltralyticsSegAdapter(checkpoint, device=device, allow_smoke_checkpoint=True)
        labels = work / "dataset" / "labels" / "val"
        image_file = next(
            path
            for path in sorted((work / "dataset" / "images" / "val").iterdir())
            if (labels / f"{path.stem}.txt").read_text(encoding="utf-8").strip()
        )
        with Image.open(image_file) as image:
            instances = adapter.predict(
                np.asarray(image.convert("RGB")), score_threshold=0.05, max_detections=50
            )
        adapter.close()
        return {
            "service_adapter_refusal": refused,
            "instances_on_one_validation_image": len(instances),
        }

    guard = step("adapter load + refusal check", service_guard)

    report = {
        "kind": "impactlens.smoke_report",
        "banner": BANNER,
        "created_utc": utc_now(),
        "work_dir": str(work),
        "steps": steps,
        "data": {
            "generated": generated,
            "manifest_images": build_report["images"],
            "dropped_exact_duplicates": dedup["dropped_exact_duplicates"],
            "dropped_near_duplicates": dedup["dropped_near_duplicates"],
            "split_images": {name: values["images"] for name, values in split["splits"].items()},
            "split_groups": {
                name: values["leakage_groups"] for name, values in split["splits"].items()
            },
            "groups_spanning_splits": len(split["groups_spanning_splits"]),
        },
        "training": json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8")),
        "evaluation_on_synthetic_shapes": {
            "mask_map50_95": evaluation["mask"]["map50_95"],
            "mask_map50": evaluation["mask"]["map50"],
            "meaning": "Mechanics only. Not evidence of any ability to detect vehicle damage.",
        },
        "adapter": guard,
        "artifact": str(artifact),
    }
    (work / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
