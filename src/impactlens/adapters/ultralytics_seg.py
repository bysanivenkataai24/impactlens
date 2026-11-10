"""Ultralytics YOLO instance-segmentation adapter.

Verified against ultralytics 8.4.208 / torch 2.15.0 (see docs/EXECUTION_RECORD.md).

Things this adapter is deliberately strict about:

* A ``.pt`` checkpoint is a pickle. It is only ever loaded from the path the
  *operator* configured, after its SHA-256 was computed (and compared, when
  ``MODEL_SHA256`` is set) by :mod:`impactlens.model_store`. Nothing a visitor
  uploads can reach this code.
* The checkpoint must be a ``segment`` task whose class names are exactly the
  supported taxonomy. Generic COCO weights (80 classes) are rejected: they are
  an initialization for training, not a damage detector.
* The requested device is used or loading fails. There is no silent CPU fallback.
* Ultralytics usage analytics, auto-install and update checks are switched off.
"""

from __future__ import annotations

import os
import platform
import threading
from pathlib import Path

import numpy as np

from ..taxonomy import class_mapping_problem, normalize_class_name
from .base import AdapterInfo, ModelUnavailable, RawInstance

SUPPORTED_SUFFIXES = (".pt",)


def configure_ultralytics_environment(*, offline: bool) -> None:
    """Set process-wide switches. Must run before ``ultralytics`` is first imported."""
    # Never let a request path pip-install packages into a running service.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    if offline:
        # No update checks, no font/weight downloads, no analytics at serve time.
        os.environ.setdefault("YOLO_OFFLINE", "true")


def disable_ultralytics_analytics() -> None:
    """Turn off anonymous usage events for this process (belt and braces)."""
    try:
        from ultralytics.utils.events import events

        events.enabled = False
    except Exception:  # noqa: BLE001,S110 - analytics layout is not part of our contract
        pass


class UltralyticsSegAdapter:
    """Loads one checkpoint once and serves predictions from it."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "cpu",
        image_size: int | None = None,
        nms_iou: float = 0.7,
        warmup: bool = True,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ModelUnavailable(
                "unsupported_checkpoint_format",
                f"expected one of {SUPPORTED_SUFFIXES}, got {checkpoint_path.suffix!r}",
            )
        if not checkpoint_path.is_file():
            raise ModelUnavailable("checkpoint_not_found", "configured checkpoint does not exist")

        configure_ultralytics_environment(offline=True)
        try:
            import torch
            import ultralytics
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - broken installation
            raise ModelUnavailable("model_load_failed", f"model runtime import failed: {exc}") from exc
        disable_ultralytics_analytics()

        self._device = self._resolve_device(torch, device)

        try:
            model = YOLO(str(checkpoint_path))
        except Exception as exc:
            raise ModelUnavailable(
                "model_load_failed", f"{type(exc).__name__} while reading the checkpoint"
            ) from exc

        task = str(getattr(model, "task", None))
        if task != "segment":
            raise ModelUnavailable(
                "unsupported_task", f"checkpoint task is {task!r}; 'segment' is required"
            )

        raw_names = model.names
        if isinstance(raw_names, dict):
            ordered = [raw_names[i] for i in sorted(raw_names)]
            self._index_to_name = {int(i): normalize_class_name(n) for i, n in raw_names.items()}
        else:
            ordered = list(raw_names)
            self._index_to_name = {i: normalize_class_name(n) for i, n in enumerate(ordered)}
        problem = class_mapping_problem(ordered)
        if problem:
            raise ModelUnavailable("class_mapping_mismatch", problem)

        if image_size is None:
            train_args = model.ckpt.get("train_args", {}) if isinstance(model.ckpt, dict) else {}
            trained = train_args.get("imgsz")
            image_size = int(trained) if isinstance(trained, int) and trained > 0 else 640

        model.model.eval()
        if model.model.training:  # pragma: no cover - defensive
            raise ModelUnavailable("model_load_failed", "model did not enter evaluation mode")

        self._model = model
        self._image_size = int(image_size)
        self._nms_iou = float(nms_iou)
        self._lock = threading.Lock()
        self._ops = __import__("ultralytics.utils.ops", fromlist=["ops"])
        self._cv2 = __import__("cv2")
        self.info = AdapterInfo(
            task=task,
            class_names=tuple(normalize_class_name(n) for n in ordered),
            device=self._device,
            inference_image_size=self._image_size,
            runtime={
                "adapter": type(self).__name__,
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
        )

        if warmup:
            try:
                blank = np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
                self.predict(blank, score_threshold=0.5, max_detections=1)
            except Exception as exc:
                raise ModelUnavailable(
                    "model_load_failed", f"{type(exc).__name__} during warm-up inference"
                ) from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(torch, device: str) -> str:
        if device == "cpu":
            return "cpu"
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ModelUnavailable(
                    "device_unavailable", "DEVICE requests CUDA but no CUDA device is available"
                )
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            if index >= torch.cuda.device_count():
                raise ModelUnavailable("device_unavailable", f"CUDA device {index} does not exist")
            return f"cuda:{index}"
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise ModelUnavailable("device_unavailable", "DEVICE requests MPS but it is absent")
            return "mps"
        raise ModelUnavailable("device_unavailable", f"unsupported DEVICE {device!r}")

    # ------------------------------------------------------------------
    def predict(
        self, image_rgb: np.ndarray, *, score_threshold: float, max_detections: int
    ) -> list[RawInstance]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be an (H, W, 3) uint8 array")
        # Ultralytics treats numpy input as BGR (OpenCV convention).
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        # One predictor instance is not safe for concurrent calls.
        with self._lock:
            result = self._model.predict(
                source=bgr,
                conf=float(score_threshold),
                iou=self._nms_iou,
                imgsz=self._image_size,
                device=self._device,
                max_det=int(max_detections),
                retina_masks=False,
                half=False,
                save=False,
                stream=False,
                verbose=False,
            )[0]
            return self._extract(result)

    def _extract(self, result) -> list[RawInstance]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or result.masks is None:
            return []
        cv2, ops = self._cv2, self._ops
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float64)
        classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
        masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
        mask_hw = masks.shape[1:]
        orig_hw = tuple(int(v) for v in result.orig_shape[:2])

        instances: list[RawInstance] = []
        for i in range(len(xyxy)):
            contours, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if len(c) >= 3]
            if contours:
                largest = max(contours, key=cv2.contourArea)
                # Mask space (letterboxed network input) -> oriented image pixels.
                polygon = ops.scale_coords(
                    mask_hw, largest.reshape(-1, 2).astype(np.float32), orig_hw, normalize=False
                )
                polygon = np.asarray(polygon, dtype=np.float64)
            else:
                polygon = np.zeros((0, 2), dtype=np.float64)
            instances.append(
                RawInstance(
                    class_name=self._index_to_name.get(int(classes[i]), f"class_{int(classes[i])}"),
                    score=float(scores[i]),
                    box_xyxy=tuple(float(v) for v in xyxy[i]),
                    polygon_xy=polygon,
                    fragment_count=max(1, len(contours)),
                )
            )
        return instances

    def close(self) -> None:
        self._model = None
