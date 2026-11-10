"""``impactlens train`` - fine-tune a segmentation checkpoint from a config file.

Runs on the machine you start it on (ideally one training GPU). It is never
reachable from the HTTP service.

Each run directory is self-describing::

    runs/<experiment>/<utc>-<config digest>/
        config.resolved.yaml     every hyper-parameter actually used
        dataset_revision.json    the data it was trained on (copied from the export)
        environment.json         library versions, hardware, git commit
        results.csv              per-epoch losses and validation metrics
        weights/best.pt          best validation fitness (tagged with provenance)
        weights/last.pt          final epoch (tagged with provenance)
        train_summary.json       outcome of this run
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import time
from pathlib import Path

import yaml

from ..adapters.ultralytics_seg import (
    configure_ultralytics_environment,
    disable_ultralytics_analytics,
)
from ..taxonomy import MODEL_DAMAGE_TYPES, class_mapping_problem
from .config import TrainConfig
from .provenance import (
    PURPOSE_TRAINED,
    environment_record,
    sha256_file,
    tag_checkpoint,
    utc_now,
)

_TRACKERS = ("tensorboard", "mlflow", "wandb", "comet", "clearml", "dvc", "raytune")


class TrainingError(RuntimeError):
    pass


def load_dataset_revision(data_yaml: Path) -> dict:
    """The dataset must come from `impactlens data export-yolo`, which records its revision."""
    revision_file = Path(data_yaml).parent / "dataset_revision.json"
    if not revision_file.is_file():
        raise TrainingError(
            f"{revision_file} not found. Train only on datasets exported with "
            "`impactlens data export-yolo`, so every run records what it was trained on."
        )
    revision = json.loads(revision_file.read_text(encoding="utf-8"))
    names = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")).get("names", {})
    ordered = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    problem = class_mapping_problem(ordered)
    if problem:
        raise TrainingError(f"{data_yaml}: {problem}")
    return revision


def resolve_init_checkpoint(config: TrainConfig) -> Path:
    """Locate (downloading official release assets if needed) and verify the init weights.

    The hash is checked before the file is deserialized.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    path = Path(config.init_checkpoint)
    if not path.is_file():
        path = Path(attempt_download_asset(config.init_checkpoint))
    if not path.is_file():
        raise TrainingError(f"initialization checkpoint not found: {config.init_checkpoint}")
    if config.init_checkpoint_sha256:
        actual = sha256_file(path)
        if actual != config.init_checkpoint_sha256.lower():
            raise TrainingError(
                f"{path}: SHA-256 {actual} does not match init_checkpoint_sha256. "
                "Refusing to load an unexpected file."
            )
    return path


def run_training(
    config: TrainConfig,
    *,
    runs_dir: Path = Path("runs"),
    purpose: str = PURPOSE_TRAINED,
) -> Path:
    """Train and return the run directory."""
    configure_ultralytics_environment(offline=False)
    from ultralytics import YOLO
    from ultralytics import settings as ultralytics_settings

    disable_ultralytics_analytics()
    ultralytics_settings.update({name: name == config.tracker for name in _TRACKERS})

    data_yaml = Path(config.data).resolve()
    if not data_yaml.is_file():
        raise TrainingError(f"dataset file not found: {config.data}")
    dataset_revision = load_dataset_revision(data_yaml)
    init_checkpoint = resolve_init_checkpoint(config)

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (Path(runs_dir) / config.experiment / f"{stamp}-{config.digest()[:8]}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.resolved.yaml").write_text(config.to_yaml(), encoding="utf-8")
    shutil.copyfile(data_yaml.parent / "dataset_revision.json", run_dir / "dataset_revision.json")
    environment = environment_record(config.device)
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )

    arguments = config.trainer_arguments()
    arguments["data"] = str(data_yaml)
    started = time.perf_counter()
    model = YOLO(str(init_checkpoint))
    if model.task != "segment":
        raise TrainingError(
            f"{init_checkpoint} is a {model.task!r} checkpoint; 'segment' is required"
        )
    model.train(project=str(run_dir.parent), name=run_dir.name, exist_ok=True, **arguments)
    elapsed = time.perf_counter() - started

    weights = {name: run_dir / "weights" / f"{name}.pt" for name in ("best", "last")}
    missing = [str(p) for p in weights.values() if not p.is_file()]
    if missing:
        raise TrainingError(f"training finished without writing {missing}")

    provenance = {
        "purpose": purpose,
        "experiment": config.experiment,
        "run": run_dir.name,
        "config_digest": config.digest(),
        "dataset_revision": dataset_revision.get("dataset_revision"),
        "class_names": list(MODEL_DAMAGE_TYPES),
        "image_size": config.image_size,
        "init_checkpoint": Path(config.init_checkpoint).name,
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "trained_utc": utc_now(),
        "git_commit": environment.get("git_commit"),
    }
    hashes = {
        name: tag_checkpoint(path, {**provenance, "which": name}) for name, path in weights.items()
    }

    epochs_run = None
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        epochs_run = max(0, len(results_csv.read_text(encoding="utf-8").strip().splitlines()) - 1)
    summary = {
        **provenance,
        "status": "completed",
        "epochs_requested": config.epochs,
        "epochs_run": epochs_run,
        "stopped_early": epochs_run is not None and epochs_run < config.epochs,
        "wall_time_s": round(elapsed, 1),
        "weights": {
            name: {"path": str(path.relative_to(run_dir)), "sha256": hashes[name]}
            for name, path in weights.items()
        },
        "note": (
            "Validation numbers in results.csv were computed during training on the validation "
            "split. Report metrics from `impactlens evaluate`, not from this file."
        ),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir
