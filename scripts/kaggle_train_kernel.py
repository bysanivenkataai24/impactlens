"""Kaggle GPU kernel: train, validate and export an ImpactLens model from a private bundle.

The attached Kaggle dataset ("bundle") must contain:
    impactlens-v1/            output of `impactlens data export-yolo` (may arrive zipped)
    impactlens_code.zip       `git archive` of the ImpactLens commit to run
    CODE_COMMIT.txt           that commit's SHA

Steps: verify the dataset revision, install ultralytics==8.4.208 on top of Kaggle's
CUDA PyTorch plus the ImpactLens code (--no-deps), run `impactlens train` with the
baseline config, evaluate on the VALIDATION split only (this selects the operating
threshold), and export the artifact. The test split is not touched here.

Set EXPECTED_REVISION / MODEL_ID / CONFIG below before pushing a new run.
"""

from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

EXPECTED_REVISION = "b5ceb7781ab3e3e4"
MODEL_ID = "impactlens-yolo11n-seg-hitl-v1"
CONFIG = "configs/train/baseline_yolo11n_640.yaml"
ULTRALYTICS = "ultralytics==8.4.208"

WORK = Path("/kaggle/working")
PY = sys.executable


def run(cmd: list, cwd: Path | None = None) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)  # noqa: S603 - fixed commands


def check_machine() -> None:
    """Stop early, with a clear message, if Kaggle gave this session no GPU or no internet."""
    import urllib.request

    import torch

    gpu = torch.cuda.is_available()
    print("CUDA available:", gpu, "| torch", torch.__version__, flush=True)
    if gpu:
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    try:
        urllib.request.urlopen("https://pypi.org/simple/ultralytics/", timeout=20)
        online = True
    except Exception as exc:
        online = False
        print("internet check failed:", type(exc).__name__, flush=True)
    print("internet:", online, flush=True)
    if not gpu or not online:
        raise SystemExit(
            "STOP: this Kaggle session has no GPU and/or no internet. Enable both in the "
            "notebook settings (they need a phone-verified Kaggle account) and re-run."
        )


def unpack(bundle: Path, name: str, target: Path) -> Path:
    """Copy or extract `name` (folder or name.zip) from the read-only bundle into `target`."""
    if (bundle / name).is_dir():
        shutil.copytree(bundle / name, target)
    elif (bundle / f"{name}.zip").is_file():
        with zipfile.ZipFile(bundle / f"{name}.zip") as archive:
            archive.extractall(target)
    else:
        raise SystemExit(
            f"{name} not found in {bundle}: {sorted(p.name for p in bundle.iterdir())}"
        )
    return target


def main() -> None:
    started = time.time()
    hits = glob.glob("/kaggle/input/**/CODE_COMMIT.txt", recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"expected one bundle with CODE_COMMIT.txt, found {hits}")
    bundle = Path(hits[0]).parent
    commit = (bundle / "CODE_COMMIT.txt").read_text().strip()
    print("bundle:", bundle, "| code commit:", commit, flush=True)
    check_machine()

    code = unpack(bundle, "impactlens_code", WORK / "impactlens")
    if not (code / "pyproject.toml").is_file():  # archive extracted one level deeper
        code = next(p.parent for p in code.rglob("pyproject.toml"))
    data_root = unpack(bundle, "impactlens-v1", WORK / "datasets" / "impactlens-v1")
    dataset_yaml = next(data_root.rglob("dataset.yaml"))
    data_dir = dataset_yaml.parent

    revision = json.loads((data_dir / "dataset_revision.json").read_text())["dataset_revision"]
    if revision != EXPECTED_REVISION:
        raise SystemExit(f"dataset revision {revision} != expected {EXPECTED_REVISION}")
    # The export wrote the local absolute path; point it at this machine's copy.
    text = dataset_yaml.read_text().splitlines()
    text = [f"path: {data_dir}" if line.startswith("path:") else line for line in text]
    dataset_yaml.write_text("\n".join(text) + "\n")
    print(dataset_yaml.read_text(), flush=True)

    # Kaggle's preinstalled CUDA PyTorch is kept; the repo lock pins CPU wheels.
    run([PY, "-m", "pip", "install", "-q", ULTRALYTICS])
    run([PY, "-m", "pip", "install", "-q", "--no-deps", "-e", code])
    run(
        [
            PY,
            "-c",
            "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.version.cuda, "
            "'available', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('ultralytics', ultralytics.__version__)",
        ]
    )

    runs_dir = WORK / "runs"
    train_started = time.time()
    run(
        [
            PY,
            "-m",
            "impactlens.cli",
            "train",
            "--config",
            code / CONFIG,
            "--set",
            f"data={dataset_yaml}",
            "--runs-dir",
            runs_dir,
        ],
        cwd=code,
    )
    train_seconds = time.time() - train_started
    run_dir = max(runs_dir.glob("*/*/train_summary.json"), key=lambda p: p.stat().st_mtime).parent

    evaluation = run_dir / "evaluation"
    run(
        [
            PY,
            "-m",
            "impactlens.cli",
            "evaluate",
            "--weights",
            run_dir / "weights" / "best.pt",
            "--data",
            dataset_yaml,
            "--split",
            "val",
            "--out",
            evaluation,
            # The ImpactLens adapter takes "cuda:0" (Ultralytics accepts it too).
            "--device",
            "cuda:0",
            "--workers",
            "4",
            "--batch-size",
            "16",
            "--plots",
        ],
        cwd=code,
    )

    run(
        [
            PY,
            "-m",
            "impactlens.cli",
            "export",
            "--run",
            run_dir,
            "--model-id",
            MODEL_ID,
            "--out-root",
            WORK / "models",
            "--evaluation",
            evaluation / "evaluation_val.json",
            "--operating-point",
            evaluation / "operating_point.json",
        ],
        cwd=code,
    )

    summary = {
        "code_commit": commit,
        "dataset_revision": revision,
        "run_dir": str(run_dir.relative_to(WORK)),
        "model_id": MODEL_ID,
        "train_wall_time_s": round(train_seconds, 1),
        "kernel_wall_time_s": round(time.time() - started, 1),
        "note": "Validation split only. The test split was not evaluated in this kernel.",
    }
    (WORK / "kernel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    # Keep the outputs small: the data copy and the code checkout are inputs, not results.
    shutil.rmtree(WORK / "datasets", ignore_errors=True)
    shutil.rmtree(WORK / "impactlens", ignore_errors=True)


if __name__ == "__main__":
    main()
