"""Turn raw adapter output into contract-conforming detections.

This is the only place where pixel coordinates become normalized coordinates,
so the convention is enforced (and unit-tested) once: values are finite,
inside ``[0, 1]``, relative to the EXIF-oriented image, and the number of
detections and polygon vertices is capped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from .adapters.base import RawInstance
from .schemas import Detection
from .taxonomy import MODEL_DAMAGE_TYPES

#: A bbox edge closer than this (normalized) to the image edge sets ``touches_image_border``.
BORDER_TOLERANCE = 0.002
#: Regions covering less than this fraction of the image set ``small_region``.
SMALL_REGION_FRACTION = 0.001
_DECIMALS = 6


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area of a closed ring given as ``(K, 2)`` points."""
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def simplify_polygon(points: np.ndarray, max_points: int) -> tuple[np.ndarray, bool]:
    """Reduce a ring to at most ``max_points`` vertices. Returns ``(points, was_simplified)``."""
    if len(points) <= max_points:
        return points, False
    ring = points.astype(np.float32).reshape(-1, 1, 2)
    epsilon = max(cv2.arcLength(ring, True) * 0.0005, 1e-3)
    for _ in range(32):
        approx = cv2.approxPolyDP(ring, epsilon, True)
        if 3 <= len(approx) <= max_points:
            return approx.reshape(-1, 2).astype(np.float64), True
        if len(approx) < 3:
            break
        epsilon *= 1.5
    # Fallback: evenly spaced vertices. Coarser, but always within the cap.
    index = np.unique(np.linspace(0, len(points) - 1, num=max_points).round().astype(int))
    return points[index].astype(np.float64), True


def _drop_repeated(points: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicate vertices (including a repeated closing vertex)."""
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    points = points[keep]
    if len(points) > 1 and np.all(points[0] == points[-1]):
        points = points[:-1]
    return points


def build_detections(
    instances: Sequence[RawInstance],
    *,
    width: int,
    height: int,
    max_detections: int,
    max_polygon_points: int,
) -> tuple[list[Detection], list[str]]:
    """Validate, normalize, cap and order instances. Returns ``(detections, warnings)``."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    candidates: list[tuple[float, dict]] = []
    dropped = 0
    for instance in instances:
        if instance.class_name not in MODEL_DAMAGE_TYPES:
            dropped += 1
            continue
        score = float(instance.score)
        box = np.asarray(instance.box_xyxy, dtype=np.float64)
        polygon = np.asarray(instance.polygon_xy, dtype=np.float64).reshape(-1, 2)
        if (
            not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            or box.shape != (4,)
            or not np.all(np.isfinite(box))
            or not np.all(np.isfinite(polygon))
        ):
            dropped += 1
            continue

        # Clamp to the image, then normalize.
        x_min, x_max = np.clip(box[[0, 2]], 0.0, float(width)) / width
        y_min, y_max = np.clip(box[[1, 3]], 0.0, float(height)) / height
        if not (x_max > x_min and y_max > y_min):
            dropped += 1
            continue

        polygon[:, 0] = np.clip(polygon[:, 0], 0.0, float(width))
        polygon[:, 1] = np.clip(polygon[:, 1], 0.0, float(height))
        polygon = _drop_repeated(polygon)
        polygon, simplified = simplify_polygon(polygon, max_polygon_points)
        area_px = polygon_area(polygon)
        if len(polygon) < 3 or area_px <= 0.0:
            dropped += 1
            continue

        normalized = np.round(polygon / np.array([width, height], dtype=np.float64), _DECIMALS)
        normalized = _drop_repeated(np.clip(normalized, 0.0, 1.0))
        if len(normalized) < 3 or polygon_area(normalized) <= 0.0:
            dropped += 1
            continue

        flags: list[str] = []
        if instance.fragment_count > 1:
            flags.append("mask_fragmented")
        if simplified:
            flags.append("polygon_simplified")
        if min(x_min, y_min) <= BORDER_TOLERANCE or max(x_max, y_max) >= 1.0 - BORDER_TOLERANCE:
            flags.append("touches_image_border")
        if area_px / float(width * height) < SMALL_REGION_FRACTION:
            flags.append("small_region")

        candidates.append(
            (
                score,
                {
                    "damage_type": instance.class_name,
                    "score": round(score, _DECIMALS),
                    "bbox": [round(float(v), _DECIMALS) for v in (x_min, y_min, x_max, y_max)],
                    "polygon": normalized.tolist(),
                    "flags": flags,
                },
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    warnings: list[str] = []
    # The adapter already limits itself to ``max_detections``; hitting the cap means
    # lower-scoring regions may have been cut off, which the reviewer should know.
    if len(instances) >= max_detections:
        warnings.append("detection_cap_reached")
    if dropped:
        warnings.append("invalid_detections_dropped")

    detections = [
        Detection(id=f"det-{rank}", **payload)
        for rank, (_, payload) in enumerate(candidates[:max_detections], start=1)
    ]
    return detections, warnings
