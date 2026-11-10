"""The single seam between a model runtime and everything else.

Nothing in this module (or in any adapter) imports web code, and the web
service imports only this module. Swapping Ultralytics for another runtime
means writing one class that satisfies :class:`SegmentationAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


class ModelUnavailable(Exception):
    """The checkpoint cannot be used. ``code`` is a ``NotReadyReason`` from the schema."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class RawInstance:
    """One predicted instance in *pixel* coordinates of the oriented input image."""

    class_name: str
    score: float
    #: ``(x_min, y_min, x_max, y_max)`` in pixels.
    box_xyxy: tuple[float, float, float, float]
    #: ``(K, 2)`` float array of the largest mask outline, in pixels. May be empty.
    polygon_xy: np.ndarray
    #: Number of disconnected pieces the predicted mask had.
    fragment_count: int = 1


@dataclass(frozen=True)
class AdapterInfo:
    task: str
    #: Normalized class names in checkpoint index order.
    class_names: tuple[str, ...]
    device: str
    inference_image_size: int
    #: Library versions and similar provenance, for logs and the model card.
    runtime: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class SegmentationAdapter(Protocol):
    """What the service needs from a model runtime."""

    info: AdapterInfo

    def predict(
        self, image_rgb: np.ndarray, *, score_threshold: float, max_detections: int
    ) -> list[RawInstance]:
        """Run inference on one ``(H, W, 3)`` uint8 RGB image.

        Implementations must be safe to call from a worker thread and must not
        keep a reference to ``image_rgb`` after returning.
        """
        ...

    def close(self) -> None:
        """Release model resources."""
        ...
