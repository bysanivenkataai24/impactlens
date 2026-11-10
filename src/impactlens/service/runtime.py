"""Model state and bounded execution of inference.

Concurrency model
-----------------
* One inference runs at a time, on one dedicated worker thread. The event loop
  never runs model code.
* At most ``max_pending_requests`` further requests may wait. Anything beyond
  that is refused immediately (HTTP 429 ``server_busy``) *before* its body is
  read, so memory use is bounded by configuration.
* Each request has a time budget covering queueing and inference.

What a timeout does and does not do
-----------------------------------
When the budget expires the client gets HTTP 504 and stops waiting. The model
call that is already running is **not** interrupted: Python threads and
CPU/GPU kernels cannot be cancelled from outside. The slot stays occupied
until that call returns, so a new request cannot start a second model call on
top of it. The same holds when a client disconnects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..adapters.base import ModelUnavailable, SegmentationAdapter
from ..config import Settings
from ..imaging import DecodedImage, ImageRejected, decode_image
from ..postprocess import build_detections
from ..schemas import Detection
from ..taxonomy import class_mapping_problem

log = logging.getLogger("impactlens.runtime")


@dataclass(frozen=True)
class LoadedModel:
    adapter: SegmentationAdapter
    revision: str
    checkpoint_sha256: str


ModelLoader = Callable[[Settings], LoadedModel]


@dataclass
class ModelState:
    ready: bool
    reasons: list[str] = field(default_factory=list)
    adapter: SegmentationAdapter | None = None
    model_id: str | None = None
    model_revision: str | None = None
    checkpoint_sha256: str | None = None


def default_model_loader(settings: Settings) -> LoadedModel:
    """Resolve the configured checkpoint and load it with the Ultralytics adapter."""
    from ..adapters.ultralytics_seg import UltralyticsSegAdapter
    from ..model_store import resolve_checkpoint

    checkpoint = resolve_checkpoint(settings)
    adapter = UltralyticsSegAdapter(
        checkpoint.path,
        device=settings.device,
        image_size=settings.inference_image_size,
        nms_iou=settings.nms_iou,
    )
    return LoadedModel(
        adapter=adapter, revision=checkpoint.revision, checkpoint_sha256=checkpoint.sha256
    )


_MODEL_CONFIG_REASONS = {
    "model_not_configured",
    "model_source_ambiguous",
    "model_source_incomplete",
    "model_revision_not_pinned",
}


def build_model_state(settings: Settings, loader: ModelLoader) -> ModelState:
    """Load the model once. Never raises: problems become not-ready reasons."""
    problems = settings.configuration_problems()
    reasons = [code for code, _ in problems]
    for code, explanation in problems:
        log.warning("not ready: %s - %s", code, explanation)

    state = ModelState(ready=False, reasons=reasons, model_id=settings.model_id)
    if _MODEL_CONFIG_REASONS.intersection(reasons):
        return state

    try:
        loaded = loader(settings)
        adapter = loaded.adapter
        # Checked again here so the guarantee does not depend on any one adapter.
        if adapter.info.task != "segment":
            raise ModelUnavailable("unsupported_task", f"adapter task is {adapter.info.task!r}")
        problem = class_mapping_problem(adapter.info.class_names)
        if problem:
            raise ModelUnavailable("class_mapping_mismatch", problem)
    except ModelUnavailable as exc:
        log.error("model unavailable: %s - %s", exc.code, exc.detail)
        state.reasons.append(exc.code)
        return state
    except Exception as exc:
        log.exception("model load failed: %s", type(exc).__name__)
        state.reasons.append("model_load_failed")
        return state

    state.adapter = adapter
    state.model_revision = loaded.revision
    state.checkpoint_sha256 = loaded.checkpoint_sha256
    state.ready = not state.reasons
    log.info(
        "model loaded: id=%s revision=%s sha256=%s device=%s imgsz=%s runtime=%s ready=%s",
        settings.model_id,
        loaded.revision,
        loaded.checkpoint_sha256,
        adapter.info.device,
        adapter.info.inference_image_size,
        adapter.info.runtime,
        state.ready,
    )
    return state


class InferenceTimeout(Exception):
    """The request's time budget ran out (while queued or while the model was running)."""


class InferenceFailed(Exception):
    """The model call raised. The original exception is chained, never sent to clients."""


@dataclass(frozen=True)
class InferenceOutput:
    image: DecodedImage
    detections: list[Detection]
    warnings: list[str]
    queue_ms: float
    decode_ms: float
    inference_ms: float
    postprocess_ms: float


class InferenceService:
    """Runs decode -> predict -> post-process with bounded concurrency."""

    def __init__(self, adapter: SegmentationAdapter, settings: Settings) -> None:
        self._adapter = adapter
        self._settings = settings
        self._slot = asyncio.Semaphore(1)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="impactlens-infer")
        self._capacity = 1 + settings.max_pending_requests
        self._admitted = 0

    # --- admission ----------------------------------------------------
    def try_admit(self) -> bool:
        """Reserve a place (running or waiting). Pair every success with :meth:`leave`."""
        if self._admitted >= self._capacity:
            return False
        self._admitted += 1
        return True

    def leave(self) -> None:
        self._admitted = max(0, self._admitted - 1)

    # --- execution ----------------------------------------------------
    async def run(self, data: bytes, score_threshold: float) -> InferenceOutput:
        budget = self._settings.inference_timeout_s
        started = time.perf_counter()
        try:
            await asyncio.wait_for(self._slot.acquire(), timeout=budget)
        except TimeoutError as exc:
            raise InferenceTimeout("timed out waiting for an inference slot") from exc
        queue_s = time.perf_counter() - started

        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._executor, self._work, data, score_threshold)
        except BaseException:
            self._slot.release()
            raise
        # The slot is released when the worker actually finishes, not when the
        # client stops waiting. See the module docstring.
        future.add_done_callback(self._release_slot)

        remaining = max(0.001, budget - queue_s)
        try:
            decoded, detections, warnings, timings = await asyncio.wait_for(
                asyncio.shield(future), timeout=remaining
            )
        except TimeoutError as exc:
            raise InferenceTimeout("inference exceeded the time budget") from exc
        return InferenceOutput(
            image=decoded,
            detections=detections,
            warnings=warnings,
            queue_ms=queue_s * 1000.0,
            decode_ms=timings[0],
            inference_ms=timings[1],
            postprocess_ms=timings[2],
        )

    def _release_slot(self, future: asyncio.Future) -> None:
        self._slot.release()
        if not future.cancelled():
            future.exception()  # mark as retrieved; abandoned work must not warn at GC

    def _work(self, data: bytes, score_threshold: float):
        """Runs on the worker thread."""
        settings = self._settings
        t0 = time.perf_counter()
        decoded = decode_image(
            data, max_pixels=settings.max_image_pixels, min_side=settings.min_image_side
        )
        t1 = time.perf_counter()
        try:
            instances = self._adapter.predict(
                decoded.rgb,
                score_threshold=score_threshold,
                max_detections=settings.max_detections,
            )
        except Exception as exc:
            raise InferenceFailed(type(exc).__name__) from exc
        t2 = time.perf_counter()
        try:
            detections, warnings = build_detections(
                instances,
                width=decoded.width,
                height=decoded.height,
                max_detections=settings.max_detections,
                max_polygon_points=settings.max_polygon_points,
            )
        except Exception as exc:
            raise InferenceFailed(type(exc).__name__) from exc
        t3 = time.perf_counter()
        timings = ((t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0)
        return decoded, detections, warnings, timings

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "ImageRejected",
    "InferenceFailed",
    "InferenceOutput",
    "InferenceService",
    "InferenceTimeout",
    "LoadedModel",
    "ModelLoader",
    "ModelState",
    "build_model_state",
    "default_model_loader",
]
