"""Provenance carried with runs and embedded in exported checkpoints."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

#: Key under which ImpactLens metadata is stored inside a ``.pt`` checkpoint dict.
CHECKPOINT_KEY = "impactlens"
PURPOSE_TRAINED = "trained_model"
#: Checkpoints from `impactlens smoke`. The service refuses to serve them.
PURPOSE_SMOKE = "smoke_test_not_a_model"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def environment_record(device: str | None = None) -> dict:
    """Versions and hardware that produced a result. Saved next to every run."""
    import numpy
    import torch
    import ultralytics

    record: dict = {
        "recorded_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": numpy.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": device,
        "git_commit": git_commit(),
    }
    try:
        import psutil

        record["cpu_count_logical"] = psutil.cpu_count()
        record["memory_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: S110
        pass
    if torch.cuda.is_available():
        record["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        record["cuda_runtime"] = torch.version.cuda
    return record


def tag_checkpoint(path: Path, metadata: dict) -> str:
    """Embed ``metadata`` in a checkpoint we produced ourselves. Returns the new SHA-256.

    Only ever called on checkpoints written by this process's own training run.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a checkpoint dictionary")
    checkpoint[CHECKPOINT_KEY] = json.loads(json.dumps(metadata))  # plain JSON types only
    torch.save(checkpoint, path)
    return sha256_file(path)


def read_checkpoint_tag(checkpoint: object) -> dict | None:
    """Metadata from an already-loaded checkpoint dict, or ``None``."""
    if isinstance(checkpoint, dict):
        tag = checkpoint.get(CHECKPOINT_KEY)
        if isinstance(tag, dict):
            return tag
    return None
