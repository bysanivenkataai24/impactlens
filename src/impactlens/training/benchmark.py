"""``impactlens benchmark`` - measure latency, memory and throughput. Never estimates.

Two modes:

``model``  in-process: cold initialization (fresh interpreter, imports included),
           warm inference p50/p95 and peak resident memory.
``http``   end-to-end against a running service: per-request latency p50/p95,
           throughput and status codes at several client concurrencies.

Numbers are valid only for the machine, checkpoint, image and settings they
were measured with; the report records all of them. Run the ``http`` mode
against the *deployed* service before setting any acceptance target.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .provenance import environment_record, sha256_file, utc_now

_COLD_START_SNIPPET = """
import json, sys, time, resource
t0 = time.perf_counter()
from pathlib import Path
from impactlens.adapters.ultralytics_seg import UltralyticsSegAdapter
t1 = time.perf_counter()
adapter = UltralyticsSegAdapter(Path(sys.argv[1]), device=sys.argv[2], image_size=int(sys.argv[3]), allow_smoke_checkpoint=True)
t2 = time.perf_counter()
print(json.dumps({"import_s": t1 - t0, "load_and_warmup_s": t2 - t1, "total_s": t2 - t0,
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))
"""


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _latency_summary(values_ms: list[float]) -> dict:
    return {
        "count": len(values_ms),
        "mean_ms": round(statistics.fmean(values_ms), 2),
        "p50_ms": round(_percentile(values_ms, 50), 2),
        "p95_ms": round(_percentile(values_ms, 95), 2),
        "max_ms": round(max(values_ms), 2),
    }


def _test_image(path: Path | None, width: int, height: int) -> tuple[np.ndarray, str]:
    if path is not None:
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            return np.asarray(
                ImageOps.exif_transpose(image).convert("RGB")
            ), f"file {Path(path).name}"
    rng = np.random.default_rng(0)
    return rng.integers(
        0, 255, (height, width, 3), dtype=np.uint8
    ), f"uniform noise {width}x{height} (no detections expected)"


def benchmark_model(
    weights: Path,
    *,
    device: str = "cpu",
    image_size: int = 640,
    runs: int = 30,
    cold_starts: int = 3,
    image: Path | None = None,
    image_width: int = 1920,
    image_height: int = 1080,
    score_threshold: float = 0.25,
) -> dict:
    import resource

    from ..adapters.ultralytics_seg import UltralyticsSegAdapter

    weights = Path(weights)
    cold = []
    for _ in range(cold_starts):
        out = subprocess.run(  # noqa: S603 - fixed interpreter and script
            [sys.executable, "-c", _COLD_START_SNIPPET, str(weights), device, str(image_size)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "YOLO_OFFLINE": "true"},
        )
        cold.append(json.loads(out.stdout.strip().splitlines()[-1]))

    adapter = UltralyticsSegAdapter(
        weights, device=device, image_size=image_size, allow_smoke_checkpoint=True
    )
    pixels, description = _test_image(image, image_width, image_height)
    for _ in range(3):
        adapter.predict(pixels, score_threshold=score_threshold, max_detections=50)
    timings, detections = [], 0
    for _ in range(runs):
        started = time.perf_counter()
        detections = len(
            adapter.predict(pixels, score_threshold=score_threshold, max_detections=50)
        )
        timings.append((time.perf_counter() - started) * 1000.0)
    tag = adapter.checkpoint_tag or {}
    return {
        "kind": "impactlens.benchmark.model",
        "created_utc": utc_now(),
        "smoke_checkpoint": tag.get("purpose") == "smoke_test_not_a_model",
        "weights": {"file": weights.name, "sha256": sha256_file(weights)},
        "settings": {
            "device": device,
            "image_size": image_size,
            "score_threshold": score_threshold,
            "input": description,
            "input_shape_hw": list(pixels.shape[:2]),
        },
        "environment": environment_record(device),
        "cold_initialization": {
            "runs": cold_starts,
            "import_s_median": round(statistics.median(c["import_s"] for c in cold), 2),
            "load_and_warmup_s_median": round(
                statistics.median(c["load_and_warmup_s"] for c in cold), 2
            ),
            "total_s_median": round(statistics.median(c["total_s"] for c in cold), 2),
            "peak_rss_mb_median": round(statistics.median(c["peak_rss_mb"] for c in cold), 1),
        },
        "warm_inference_adapter_only": {
            **_latency_summary(timings),
            "detections_last_run": detections,
            "includes": "RGB->BGR, letterbox, forward pass, NMS, mask polygons; excludes HTTP and image decoding",
        },
        "peak_rss_mb_this_process": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "caveat": "Measured on the machine in 'environment'. Not transferable to other hardware.",
    }


async def _http_level(
    client, url: str, headers: dict, payload: bytes, file_name: str, requests: int, concurrency: int
) -> dict:
    latencies: list[float] = []
    server_ms: list[float] = []
    statuses: dict[str, int] = {}
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(i)

    async def worker() -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            started = time.perf_counter()
            try:
                response = await client.post(
                    f"{url}/predict",
                    headers=headers,
                    files={"image": (file_name, payload, "application/octet-stream")},
                )
                key = str(response.status_code)
                if response.status_code == 200:
                    latencies.append((time.perf_counter() - started) * 1000.0)
                    server_ms.append(response.json()["latency_ms"]["total"])
            except Exception as exc:
                key = f"client_error:{type(exc).__name__}"
            statuses[key] = statuses.get(key, 0) + 1

    wall = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - wall
    return {
        "concurrency": concurrency,
        "requests": requests,
        "status_counts": dict(sorted(statuses.items())),
        "successful": len(latencies),
        "end_to_end": _latency_summary(latencies) if latencies else None,
        "server_reported_total": _latency_summary(server_ms) if server_ms else None,
        "throughput_successful_rps": round(len(latencies) / wall, 2) if wall > 0 else None,
        "wall_time_s": round(wall, 2),
    }


def benchmark_http(
    url: str,
    image: Path,
    *,
    api_key_env: str = "API_KEY",
    requests: int = 30,
    concurrency: tuple[int, ...] = (1, 2, 4),
    timeout_s: float = 120.0,
) -> dict:
    import httpx

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"set the service key in the {api_key_env} environment variable (it is never taken from the command line)"
        )
    url = url.rstrip("/")
    payload = Path(image).read_bytes()

    async def main() -> dict:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            health = (await client.get(f"{url}/health")).json()
            if not health.get("ready"):
                raise RuntimeError(f"service is not ready: {health.get('reasons')}")
            headers = {"Authorization": f"Bearer {api_key}"}
            levels = [
                await _http_level(client, url, headers, payload, Path(image).name, requests, c)
                for c in concurrency
            ]
            return {"health": health, "levels": levels}

    measured = asyncio.run(main())
    health = measured["health"]
    return {
        "kind": "impactlens.benchmark.http",
        "created_utc": utc_now(),
        "target": {
            "url": url,
            "model_id": health.get("model_id"),
            "model_revision": health.get("model_revision"),
            "checkpoint_sha256": health.get("checkpoint_sha256"),
            "device": health.get("device"),
        },
        "input": {
            "file": Path(image).name,
            "bytes": len(payload),
            "sha256": sha256_file(Path(image)),
        },
        "client_environment": environment_record(),
        "levels": measured["levels"],
        "caveat": (
            "Latency includes the network between this client and the service. 429 responses are the "
            "service's rate or capacity limits working as configured; they are reported, not retried."
        ),
    }
