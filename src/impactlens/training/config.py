"""Training configuration: one YAML file, validated, saved with every run.

Every hyper-parameter that influences a result lives here, not in a notebook
and not in library defaults that can change between versions. Unknown keys are
an error, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


class TrainConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Augmentation:
    """Training-only augmentation. Geometry is applied to images and masks together
    by the trainer, so annotations stay aligned. Validation/test images are never augmented."""

    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    close_mosaic: int = 10
    mixup: float = 0.0
    cutmix: float = 0.0
    copy_paste: float = 0.0
    multi_scale: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    experiment: str
    data: str
    # --- initialization ----------------------------------------------------
    init_checkpoint: str = "yolo11n-seg.pt"
    #: When set, the file is hashed and compared before it is deserialized.
    init_checkpoint_sha256: str | None = None
    # --- schedule ------------------------------------------------------------
    seed: int = 20261110
    image_size: int = 640
    batch_size: int = 16
    epochs: int = 100
    patience: int = 20
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    cos_lr: bool = True
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    # --- loss ----------------------------------------------------------------
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    #: Inverse-frequency class weighting power (0 = off). An imbalance ablation knob.
    cls_pw: float = 0.0
    overlap_mask: bool = True
    mask_ratio: int = 4
    # --- runtime -------------------------------------------------------------
    device: str = "0"
    workers: int = 8
    amp: bool = True
    deterministic: bool = True
    cache: bool = False
    plots: bool = True
    #: Fraction of the training split to use (smoke tests and learning curves).
    fraction: float = 1.0
    #: Validation NMS IoU. Keep equal to the service's NMS_IOU.
    nms_iou: float = 0.7
    #: 'none' or one of the trackers Ultralytics integrates (tensorboard, mlflow, wandb, ...).
    tracker: str = "none"
    augment: Augmentation = field(default_factory=Augmentation)
    notes: str = ""

    def digest(self) -> str:
        """Stable hash of the resolved configuration, recorded with the run."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)

    def trainer_arguments(self) -> dict:
        """The exact keyword arguments handed to ``ultralytics`` ``model.train``."""
        return {
            "data": self.data,
            "seed": self.seed,
            "imgsz": self.image_size,
            "batch": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "cos_lr": self.cos_lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "box": self.box_gain,
            "cls": self.cls_gain,
            "dfl": self.dfl_gain,
            "cls_pw": self.cls_pw,
            "overlap_mask": self.overlap_mask,
            "mask_ratio": self.mask_ratio,
            "device": self.device,
            "workers": self.workers,
            "amp": self.amp,
            "deterministic": self.deterministic,
            "cache": self.cache,
            "plots": self.plots,
            "fraction": self.fraction,
            "iou": self.nms_iou,
            "pretrained": True,
            "val": True,
            "save": True,
            "verbose": True,
            **asdict(self.augment),
        }


def _build(cls, values: dict, where: str):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise TrainConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    return cls(**values)


def load_train_config(path: Path, overrides: dict | None = None) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TrainConfigError(f"{path}: expected a mapping at the top level")
    raw.update(overrides or {})
    # YAML and --set may give `device: 0`; the trainer accepts both, the record keeps one form.
    if "device" in raw:
        raw["device"] = str(raw["device"])
    augment = raw.pop("augment", {}) or {}
    if not isinstance(augment, dict):
        raise TrainConfigError(f"{path}: 'augment' must be a mapping")
    try:
        config = _build(
            TrainConfig,
            {**raw, "augment": _build(Augmentation, augment, f"{path}: augment")},
            str(path),
        )
    except TypeError as exc:
        raise TrainConfigError(f"{path}: {exc}") from exc
    validate(config, str(path))
    return config


def validate(config: TrainConfig, where: str = "config") -> None:
    def check(condition: bool, message: str) -> None:
        if not condition:
            raise TrainConfigError(f"{where}: {message}")

    check(
        bool(config.experiment) and "/" not in config.experiment, "experiment must be a simple name"
    )
    check(
        config.image_size % 32 == 0 and 160 <= config.image_size <= 2048,
        "image_size must be a multiple of 32 in [160, 2048]",
    )
    check(config.batch_size >= 1, "batch_size must be >= 1")
    check(config.epochs >= 1 and config.patience >= 0, "epochs must be >= 1 and patience >= 0")
    check(0 < config.lr0 < 1 and 0 < config.lrf <= 1, "lr0 must be in (0, 1) and lrf in (0, 1]")
    check(0 < config.fraction <= 1, "fraction must be in (0, 1]")
    check(0 <= config.cls_pw <= 1, "cls_pw must be in [0, 1]")
    check(
        config.optimizer
        in {"SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"},
        "optimizer must be named explicitly ('auto' would let the library choose lr0 for you)",
    )
    check(
        config.tracker in {"none", "tensorboard", "mlflow", "wandb", "comet", "clearml", "dvc"},
        "unknown tracker",
    )
    if config.init_checkpoint_sha256 is not None:
        check(
            len(config.init_checkpoint_sha256) == 64,
            "init_checkpoint_sha256 must be 64 hex characters",
        )
    for name, value in asdict(config.augment).items():
        check(value >= 0, f"augment.{name} must be >= 0")
    check(
        config.augment.flipud == 0.0 or config.augment.flipud <= 0.5,
        "augment.flipud above 0.5 is almost certainly a mistake",
    )
