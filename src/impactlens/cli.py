"""``impactlens`` command line. Heavy imports happen inside each command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


# ------------------------------------------------------------------------ data
def _data_manifest(args: argparse.Namespace) -> int:
    from .data import manifest as m

    rows, report = m.build_rows(
        data_root=args.data_root,
        images_dir=args.images or "",
        coco_file=args.coco,
        source=args.source,
        license_text=args.license,
        annotation_revision=args.annotation_revision,
        image_metadata=args.image_metadata,
        negatives_dir=args.negatives_dir,
        ignore_categories=args.ignore_category or (),
    )
    existing = m.read_manifest(args.out) if args.out.is_file() else []
    if any(row.source == args.source for row in existing):
        raise m.ManifestError(
            f"source {args.source!r} is already in {args.out}; use a new manifest per revision"
        )
    m.write_manifest(existing + rows, args.out)
    _print_json({**report, "manifest": str(args.out), "rows_total": len(existing) + len(rows)})
    return 0


def _data_dedup(args: argparse.Namespace) -> int:
    from .data import manifest as m

    rows = m.read_manifest(args.manifest)
    report = m.deduplicate(rows, hamming_threshold=args.hamming_threshold)
    m.write_manifest(rows, args.manifest)
    report_file = args.manifest.with_suffix(".dedup.json")
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_json(
        {
            k: (v if not isinstance(v, list) else f"{len(v)} pair(s), see {report_file}")
            for k, v in report.items()
        }
    )
    return 0


def _data_split(args: argparse.Namespace) -> int:
    from .data import manifest as m

    rows = m.read_manifest(args.manifest)
    previous = m.read_manifest(args.previous) if args.previous else None
    groups = m.assign_leakage_groups(rows, allow_ungrouped=args.allow_ungrouped)
    info = m.split_groups(rows, fractions=tuple(args.fractions), seed=args.seed, previous=previous)
    report = {**m.split_report(rows, min_eval_instances=args.min_eval_instances), **groups, **info}
    if report["groups_spanning_splits"]:
        raise m.ManifestError("internal error: a leakage group spans splits; nothing was written")
    m.write_manifest(rows, args.manifest)
    args.manifest.with_suffix(".split.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _print_json(report)
    return 0


def _data_stats(args: argparse.Namespace) -> int:
    from .data import manifest as m

    _print_json(
        m.split_report(m.read_manifest(args.manifest), min_eval_instances=args.min_eval_instances)
    )
    return 0


def _data_export(args: argparse.Namespace) -> int:
    from .data import manifest as m
    from .data.yolo_export import export_yolo

    revision = export_yolo(
        m.read_manifest(args.manifest),
        data_root=args.data_root,
        out_dir=args.out,
        jpeg_quality=args.jpeg_quality,
        rfs_threshold=args.rfs_threshold,
        seed=args.seed,
    )
    _print_json(
        {
            "dataset": str(args.out / "dataset.yaml"),
            "dataset_revision": revision["dataset_revision"],
            "export_counters": revision["export_counters"],
            "warnings": revision["split_report"]["warnings"],
        }
    )
    return 0


# -------------------------------------------------------------------- training
def _train(args: argparse.Namespace) -> int:
    from .training.config import load_train_config
    from .training.train import run_training

    overrides = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value) if value[:1] in '[{0123456789tfn-"' else value
    config = load_train_config(args.config, overrides)
    run_dir = run_training(config, runs_dir=args.runs_dir)
    print(f"run directory: {run_dir}")
    print((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .training.evaluate import evaluate, summarize

    result = evaluate(
        args.weights,
        args.data,
        split=args.split,
        out_dir=args.out,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
        nms_iou=args.nms_iou,
        score_threshold=args.score_threshold,
        tile=args.tile,
        final_test=args.final_test,
        plots=args.plots,
    )
    print(summarize(result))
    print(f"written: {args.out / f'evaluation_{args.split}.json'}")
    return 0


def _robustness(args: argparse.Namespace) -> int:
    from .training.robustness import run_robustness

    report = run_robustness(
        args.weights,
        args.data,
        out_dir=args.out,
        score_threshold=args.score_threshold,
        corruptions=args.corruption,
        seed=args.seed,
        keep_images=args.keep_images,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
        nms_iou=args.nms_iou,
    )
    _print_json(report)
    return 0


def _export(args: argparse.Namespace) -> int:
    from .training.export import export_artifact

    target = export_artifact(
        args.run,
        model_id=args.model_id,
        out_root=args.out_root,
        which=args.which,
        evaluation_file=args.evaluation,
        operating_point_file=args.operating_point,
        allow_smoke=args.allow_smoke,
    )
    print(f"artifact: {target}")
    print((target / "service.env").read_text(encoding="utf-8"))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .training import benchmark

    if args.mode == "model":
        report = benchmark.benchmark_model(
            args.weights,
            device=args.device,
            image_size=args.image_size,
            runs=args.runs,
            cold_starts=args.cold_starts,
            image=args.image,
            score_threshold=args.score_threshold,
        )
    else:
        report = benchmark.benchmark_http(
            args.url,
            args.image,
            api_key_env=args.api_key_env,
            requests=args.requests,
            concurrency=tuple(args.concurrency),
            timeout_s=args.timeout,
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_json(report)
    return 0


def _smoke(args: argparse.Namespace) -> int:
    from .training.smoke import BANNER, run_smoke

    print(f"\n{BANNER}\n")
    report = run_smoke(
        args.work_dir,
        init_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        image_size=args.image_size,
        device=args.device,
        seed=args.seed,
    )
    _print_json({k: v for k, v in report.items() if k != "training"})
    print(f"\n{BANNER}\n")
    return 0


# ---------------------------------------------------------------------- others
def _serve(_args: argparse.Namespace) -> int:
    from .service.__main__ import main as serve

    serve()
    return 0


SCHEMA_FILES = {
    "predict-response": "PredictResponse",
    "health-response": "HealthResponse",
    "error-response": "ErrorResponse",
}


def render_schemas() -> dict[str, str]:
    from . import schemas

    rendered = {}
    for stem, model_name in SCHEMA_FILES.items():
        schema = getattr(schemas, model_name).model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"impactlens/{stem}/{schemas.SCHEMA_VERSION}"
        if stem == "predict-response":
            schema["x-detection-flags"] = schemas.DETECTION_FLAG_DOCS
        rendered[f"{stem}-{schemas.SCHEMA_VERSION}.schema.json"] = (
            json.dumps(schema, indent=2, sort_keys=True) + "\n"
        )
    return rendered


def _schema(args: argparse.Namespace) -> int:
    stale = []
    for name, text in render_schemas().items():
        target = args.dir / name
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            args.dir.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print(f"wrote {target}")
    if stale:
        print(
            f"out of date: {stale}. Run `impactlens schema` and review the diff.", file=sys.stderr
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="impactlens", description="ImpactLens data, training and service commands."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "serve", help="run the inference service (configuration from environment variables)"
    ).set_defaults(run=_serve)

    schema = commands.add_parser(
        "schema", help="write or check the JSON Schema files of the API contract"
    )
    schema.add_argument("--dir", type=Path, default=Path("schemas"))
    schema.add_argument("--check", action="store_true")
    schema.set_defaults(run=_schema)

    # --- data ---------------------------------------------------------------
    data = commands.add_parser(
        "data", help="manifest, de-duplication, splits, label export"
    ).add_subparsers(dest="data_command", required=True)

    manifest = data.add_parser("manifest", help="add one data source to a manifest")
    manifest.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="directory all paths in the manifest are relative to",
    )
    manifest.add_argument("--images", help="image directory, relative to --data-root")
    manifest.add_argument("--coco", help="COCO instance-segmentation JSON, relative to --data-root")
    manifest.add_argument(
        "--negatives-dir",
        help="directory of unannotated negative images; first sub-folder = negative type",
    )
    manifest.add_argument(
        "--source", required=True, help="short provenance name, e.g. 'cardd' or 'workshop-2026q4'"
    )
    manifest.add_argument(
        "--license", required=True, help="licence / consent terms the data was obtained under"
    )
    manifest.add_argument("--annotation-revision", required=True)
    manifest.add_argument(
        "--image-metadata",
        type=Path,
        help="CSV: file_name, incident_id, vehicle_group_id, meta_* columns",
    )
    manifest.add_argument(
        "--ignore-category", action="append", help="drop instances of this annotation category"
    )
    manifest.add_argument("--out", type=Path, required=True)
    manifest.set_defaults(run=_data_manifest)

    dedup = data.add_parser("dedup", help="drop exact and near-duplicates (before splitting)")
    dedup.add_argument("--manifest", type=Path, required=True)
    dedup.add_argument(
        "--hamming-threshold",
        type=int,
        default=4,
        help="max dHash distance (0-64) treated as near-duplicate",
    )
    dedup.set_defaults(run=_data_dedup)

    split = data.add_parser("split", help="assign whole incident/vehicle groups to train/val/test")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    split.add_argument("--seed", type=int, default=20261110)
    split.add_argument(
        "--previous", type=Path, help="earlier manifest whose split assignments must be preserved"
    )
    split.add_argument(
        "--allow-ungrouped",
        action="store_true",
        help="accept images without incident/vehicle ids (leakage risk is recorded)",
    )
    split.add_argument("--min-eval-instances", type=int, default=30)
    split.set_defaults(run=_data_split)

    stats = data.add_parser("stats", help="print the counts that actually resulted")
    stats.add_argument("--manifest", type=Path, required=True)
    stats.add_argument("--min-eval-instances", type=int, default=30)
    stats.set_defaults(run=_data_stats)

    export_yolo = data.add_parser(
        "export-yolo", help="write images/labels/dataset.yaml for training"
    )
    export_yolo.add_argument("--manifest", type=Path, required=True)
    export_yolo.add_argument("--data-root", type=Path, required=True)
    export_yolo.add_argument("--out", type=Path, required=True)
    export_yolo.add_argument("--jpeg-quality", type=int, default=95)
    export_yolo.add_argument(
        "--rfs-threshold",
        type=float,
        help="also write a repeat-factor-sampled training list (imbalance ablation)",
    )
    export_yolo.add_argument("--seed", type=int, default=20261110)
    export_yolo.set_defaults(run=_data_export)

    # --- training -----------------------------------------------------------
    train = commands.add_parser(
        "train",
        help="fine-tune from a config file (run on a training machine, never in the service)",
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--runs-dir", type=Path, default=Path("runs"))
    train.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a top-level config key, e.g. --set device=cpu",
    )
    train.set_defaults(run=_train)

    def evaluation_arguments(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--weights", type=Path, required=True)
        sub.add_argument(
            "--data",
            type=Path,
            required=True,
            help="dataset.yaml from `impactlens data export-yolo`",
        )
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument(
            "--image-size", type=int, help="default: the size the checkpoint was trained at"
        )
        sub.add_argument("--batch-size", type=int, default=16)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--workers", type=int, default=4)
        sub.add_argument("--nms-iou", type=float, default=0.7)

    evaluate = commands.add_parser(
        "evaluate", help="AP, per-class metrics, operating-point analysis"
    )
    evaluation_arguments(evaluate)
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument(
        "--score-threshold",
        type=float,
        help="operating threshold; selected on validation when omitted",
    )
    evaluate.add_argument(
        "--tile",
        type=int,
        help="tiled inference with this tile size in source pixels (comparison experiments)",
    )
    evaluate.add_argument(
        "--final-test", action="store_true", help="required for --split test; the use is logged"
    )
    evaluate.add_argument("--plots", action="store_true")
    evaluate.set_defaults(run=_evaluate)

    robustness = commands.add_parser(
        "robustness", help="re-evaluate the validation split under synthetic degradations"
    )
    evaluation_arguments(robustness)
    robustness.add_argument(
        "--score-threshold",
        type=float,
        required=True,
        help="the operating threshold selected on validation",
    )
    robustness.add_argument("--corruption", action="append", help="repeatable; default: all")
    robustness.add_argument("--seed", type=int, default=0)
    robustness.add_argument("--keep-images", action="store_true")
    robustness.set_defaults(run=_robustness)

    export = commands.add_parser("export", help="package a trained checkpoint for the service")
    export.add_argument(
        "--run", type=Path, required=True, help="run directory from `impactlens train`"
    )
    export.add_argument("--model-id", required=True)
    export.add_argument("--out-root", type=Path, default=Path("models"))
    export.add_argument("--which", choices=["best", "last"], default="best")
    export.add_argument("--evaluation", type=Path, help="evaluation_val.json to attach")
    export.add_argument(
        "--operating-point", type=Path, help="operating_point.json selected on validation"
    )
    export.add_argument("--allow-smoke", action="store_true", help=argparse.SUPPRESS)
    export.set_defaults(run=_export)

    benchmark = commands.add_parser(
        "benchmark", help="measure latency, memory and throughput"
    ).add_subparsers(dest="mode", required=True)
    bench_model = benchmark.add_parser("model", help="in-process: cold start, warm latency, memory")
    bench_model.add_argument("--weights", type=Path, required=True)
    bench_model.add_argument("--device", default="cpu")
    bench_model.add_argument("--image-size", type=int, default=640)
    bench_model.add_argument("--runs", type=int, default=30)
    bench_model.add_argument("--cold-starts", type=int, default=3)
    bench_model.add_argument("--image", type=Path, help="a real photo; default is a noise image")
    bench_model.add_argument("--score-threshold", type=float, default=0.25)
    bench_model.add_argument("--out", type=Path)
    bench_model.set_defaults(run=_benchmark)
    bench_http = benchmark.add_parser("http", help="end-to-end against a running service")
    bench_http.add_argument("--url", required=True)
    bench_http.add_argument("--image", type=Path, required=True)
    bench_http.add_argument(
        "--api-key-env",
        default="API_KEY",
        help="NAME of the environment variable holding the service key",
    )
    bench_http.add_argument("--requests", type=int, default=30)
    bench_http.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    bench_http.add_argument("--timeout", type=float, default=120.0)
    bench_http.add_argument("--out", type=Path)
    bench_http.set_defaults(run=_benchmark)

    smoke = commands.add_parser(
        "smoke", help="tiny end-to-end run on synthetic shapes (mechanics only, not validation)"
    )
    smoke.add_argument("--work-dir", type=Path)
    smoke.add_argument("--init-checkpoint", default="yolo11n-seg.pt")
    smoke.add_argument("--epochs", type=int, default=3)
    smoke.add_argument("--image-size", type=int, default=320)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(run=_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.run(args) or 0)
    except (ValueError, RuntimeError) as exc:
        # Our own error types derive from these; show the message, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
