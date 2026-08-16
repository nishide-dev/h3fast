"""H3Fast command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from h3fast import __version__
from h3fast.benchmarks import (
    apply_quality_registry_review,
    build_quality_reference,
    build_singularity_launch,
    check_formal_quality_set,
    check_human_pairwise_ballot,
    check_quality,
    check_quality_metric_plan,
    compile_quality_registry,
    load_runtime_settings,
    prepare_human_pairwise_ballot,
    prepare_quality_registry_review,
    record_human_pairwise_selection,
    run_case,
    run_preflight,
    run_suite,
    score_perceptual_video,
    serve_guarded,
    stage_human_pairwise_presentation,
    validate_protocol,
)
from h3fast.benchmarks.perceptual_video import ALEXNET_BACKBONE_SHA256
from h3fast.compliance import check_territory_inventory
from h3fast.diagnostics import run_doctor
from h3fast.exceptions import H3FastError
from h3fast.manifest import inspect_snapshot, verify_model_artifact
from h3fast.release import check_release_gate

CommandHandler = Callable[[argparse.Namespace], int]


def _default_ffprobe_adapter() -> str:
    bundled = Path(__file__).parent / "runtime" / "ffprobe.py"
    return str(bundled if bundled.is_file() else Path("runtime/ffprobe.py"))


def _write_json(value: object) -> None:
    sys.stdout.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _doctor(args: argparse.Namespace) -> int:
    report = run_doctor()
    if args.json:
        _write_json(report.to_dict())
    else:
        for check in report.checks:
            sys.stdout.write(
                f"[{check.status.upper():7}] {check.name}: {check.message}\n"
            )
    return 0 if report.healthy else 1


def _inspect_snapshot(args: argparse.Namespace) -> int:
    report = inspect_snapshot(
        Path(args.path),
        variant=args.variant,
        base_revision=args.base_revision,
        base_model=args.base_model,
        include_hashes=args.hash,
    )
    data = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(data)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(data, encoding="utf-8")
        sys.stdout.write(f"Wrote snapshot report to {output}\n")
    return 0


def _verify_model(args: argparse.Namespace) -> int:
    report = verify_model_artifact(Path(args.path))
    _write_json(report.to_dict())
    return 0


def _release_check(args: argparse.Namespace) -> int:
    report = check_release_gate(Path(args.record))
    _write_json(report.to_dict())
    return 0 if report.ready else 1


def _compliance_check_territories(args: argparse.Namespace) -> int:
    report = check_territory_inventory(Path(args.record))
    _write_json(report.to_dict())
    return 0 if report.ready else 1


def _validate_benchmark_protocol(args: argparse.Namespace) -> int:
    report = validate_protocol(Path(args.path))
    _write_json(report.to_dict())
    return 0


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        message = "GPU IDs must be comma-separated integers"
        raise argparse.ArgumentTypeError(message) from error
    if (
        not result
        or any(index < 0 for index in result)
        or len(set(result)) != len(result)
    ):
        message = "GPU IDs must be distinct non-negative integers"
        raise argparse.ArgumentTypeError(message)
    return result


def _benchmark_preflight(args: argparse.Namespace) -> int:
    output = Path(args.output)
    report = run_preflight(
        Path(args.protocol),
        snapshot_path=Path(args.snapshot),
        selected_gpus=args.gpus,
        output_path=output,
        sglang_source=Path(args.sglang_source),
        runtime_image=Path(args.runtime_image),
        ffprobe_adapter=Path(args.ffprobe_adapter),
    )
    data = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(output)
    sys.stdout.write(data)
    return 0 if report.ready else 1


def _benchmark_plan_launch(args: argparse.Namespace) -> int:
    runtime_settings = load_runtime_settings(Path(args.protocol))
    plan = build_singularity_launch(
        snapshot_path=Path(args.snapshot),
        runtime_image=Path(args.runtime_image),
        sglang_source=Path(args.sglang_source),
        ffprobe_adapter=Path(args.ffprobe_adapter),
        output_path=Path(args.server_output),
        selected_gpus=args.gpus,
        dit_layerwise_resident_layers=(runtime_settings.dit_layerwise_resident_layers),
        port=args.port,
    )
    _write_json(plan.to_dict())
    return 0


def _benchmark_serve_guarded(args: argparse.Namespace) -> int:
    preflight_output = Path(args.preflight_output)
    report = run_preflight(
        Path(args.protocol),
        snapshot_path=Path(args.snapshot),
        selected_gpus=args.gpus,
        output_path=preflight_output,
        sglang_source=Path(args.sglang_source),
        runtime_image=Path(args.runtime_image),
        ffprobe_adapter=Path(args.ffprobe_adapter),
    )
    data = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    preflight_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = preflight_output.with_suffix(preflight_output.suffix + ".partial")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(preflight_output)
    if not report.ready:
        sys.stdout.write(data)
        return 1

    runtime_settings = load_runtime_settings(Path(args.protocol))
    plan = build_singularity_launch(
        snapshot_path=Path(args.snapshot),
        runtime_image=Path(args.runtime_image),
        sglang_source=Path(args.sglang_source),
        ffprobe_adapter=Path(args.ffprobe_adapter),
        output_path=Path(args.server_output),
        selected_gpus=args.gpus,
        dit_layerwise_resident_layers=(runtime_settings.dit_layerwise_resident_layers),
        port=args.port,
    )
    serve_guarded(
        plan,
        endpoint=f"http://127.0.0.1:{args.port}",
        report_path=Path(args.guard_report),
        lifecycle_path=(
            Path(args.lifecycle_report) if args.lifecycle_report is not None else None
        ),
        startup_timeout=args.startup_timeout,
        poll_interval=args.poll_interval,
    )
    return 0


def _benchmark_run_case(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    try:
        result = run_case(
            Path(args.protocol),
            case_id=args.case_id,
            endpoint=args.endpoint,
            output_dir=output_dir,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
    except H3FastError as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "1.0",
            "status": "failed",
            "recorded_at": datetime.now(UTC).isoformat(),
            "case_id": args.case_id,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        path = output_dir / f"{args.case_id}-failure.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        raise
    (output_dir / f"{args.case_id}-failure.json").unlink(missing_ok=True)
    _write_json(result.to_dict())
    return 0


def _benchmark_run_suite(args: argparse.Namespace) -> int:
    suite = run_suite(
        Path(args.protocol),
        case_id=args.case_id,
        endpoint=args.endpoint,
        output_dir=Path(args.output_dir),
        server_output_dir=Path(args.server_output),
        server_lifecycle_path=Path(args.server_lifecycle_report),
        server_guard_report_path=Path(args.server_guard_report),
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    _write_json(suite.to_dict())
    return 0


def _benchmark_build_quality_reference(args: argparse.Namespace) -> int:
    reference = build_quality_reference(
        Path(args.suite),
        Path(args.protocol),
        Path(args.output),
        reference_id=args.reference_id,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    _write_json(reference)
    return 0


def _benchmark_check_quality(args: argparse.Namespace) -> int:
    report = check_quality(
        Path(args.reference),
        Path(args.suite),
        Path(args.protocol),
        Path(args.output),
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    _write_json(report)
    return 0 if report["status"] == "passed" else 1


def _benchmark_check_quality_set(args: argparse.Namespace) -> int:
    report = check_formal_quality_set(Path(args.record))
    _write_json(report.to_dict())
    return 0 if report.ready else 1


def _benchmark_check_quality_metric_plan(args: argparse.Namespace) -> int:
    report = check_quality_metric_plan(Path(args.plan))
    _write_json(report.to_dict())
    return 0 if report.ready else 1


def _benchmark_compile_quality_registry(args: argparse.Namespace) -> int:
    report = compile_quality_registry(
        Path(args.registry),
        Path(args.template),
        Path(args.output),
        registry_uri=args.registry_uri,
    )
    _write_json(report.to_dict())
    return 0


def _benchmark_prepare_quality_review(args: argparse.Namespace) -> int:
    report = prepare_quality_registry_review(
        Path(args.registry),
        Path(args.output),
        reviewer=args.reviewer,
    )
    _write_json(report.to_dict())
    return 0


def _benchmark_apply_quality_review(args: argparse.Namespace) -> int:
    report = apply_quality_registry_review(
        Path(args.registry),
        Path(args.review),
        Path(args.output),
    )
    _write_json(report.to_dict())
    return 0 if report.ready else 1


def _benchmark_prepare_human_pairwise(args: argparse.Namespace) -> int:
    report = prepare_human_pairwise_ballot(
        Path(args.formal_set),
        Path(args.ballot),
        Path(args.assignment),
        ballot_id=args.ballot_id,
        reviewer=args.reviewer,
        randomization_seed_file=Path(args.randomization_seed_file),
    )
    _write_json(report.to_dict())
    return 0


def _benchmark_check_human_pairwise(args: argparse.Namespace) -> int:
    report = check_human_pairwise_ballot(
        Path(args.formal_set), Path(args.ballot), Path(args.assignment)
    )
    _write_json(report.to_dict())
    return 0 if report.complete else 1


def _benchmark_stage_human_pairwise(args: argparse.Namespace) -> int:
    report = stage_human_pairwise_presentation(
        Path(args.formal_set),
        Path(args.ballot),
        Path(args.assignment),
        Path(args.media_manifest),
        Path(args.staging_dir),
    )
    _write_json(report.to_dict())
    return 0


def _benchmark_record_human_pairwise(args: argparse.Namespace) -> int:
    report = record_human_pairwise_selection(
        Path(args.ballot),
        case_id=args.case,
        selection=args.selection,
        overwrite=args.overwrite,
    )
    _write_json(report.to_dict())
    return 0


def _benchmark_score_perceptual_video(args: argparse.Namespace) -> int:
    report = score_perceptual_video(
        Path(args.baseline),
        Path(args.candidate),
        backbone_dir=Path(args.backbone_dir),
        expected_backbone_sha256=(
            args.expected_backbone_sha256 or ALEXNET_BACKBONE_SHA256
        ),
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    _write_json(report.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(
        prog="h3fast",
        description="Reproducible optimization tooling for local MiniMax H3-Base",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect the local environment")
    doctor.add_argument("--json", action="store_true", help="Emit JSON")
    doctor.set_defaults(handler=_doctor)

    snapshot = subparsers.add_parser(
        "inspect-snapshot",
        help="Validate an explicitly supplied local H3 snapshot",
    )
    snapshot.add_argument("path")
    snapshot.add_argument("--variant", choices=("fl2va", "ref2va"), required=True)
    snapshot.add_argument(
        "--base-model",
        default="MiniMaxAI/MiniMax-H3",
        help="Expected Hugging Face repository ID",
    )
    snapshot.add_argument(
        "--base-revision",
        required=True,
        help="Immutable lowercase 40-character Hugging Face commit SHA",
    )
    snapshot.add_argument(
        "--hash",
        action="store_true",
        help="Calculate SHA-256 for every local file",
    )
    snapshot.add_argument("--output", help="Write the JSON report to this path")
    snapshot.set_defaults(handler=_inspect_snapshot)

    verify_model = subparsers.add_parser(
        "verify-model",
        help="Verify an H3Fast artifact manifest and checksums",
    )
    verify_model.add_argument("path")
    verify_model.set_defaults(handler=_verify_model)

    release = subparsers.add_parser(
        "release",
        help="Validate fail-closed release readiness records",
    )
    release_subparsers = release.add_subparsers(dest="release_command", required=True)
    release_check = release_subparsers.add_parser(
        "check",
        help="Exit successfully only when every required release gate is approved",
    )
    release_check.add_argument("--record", required=True)
    release_check.set_defaults(handler=_release_check)

    compliance = subparsers.add_parser(
        "compliance",
        help="Validate compliance evidence without inferring legal approval",
    )
    compliance_subparsers = compliance.add_subparsers(
        dest="compliance_command", required=True
    )
    territory_check = compliance_subparsers.add_parser(
        "check-territories",
        help="Exit successfully only when every required territory flow is approved",
    )
    territory_check.add_argument("--record", required=True)
    territory_check.set_defaults(handler=_compliance_check_territories)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Validate or run reproducible benchmarks",
    )
    benchmark_subparsers = benchmark.add_subparsers(
        dest="benchmark_command", required=True
    )
    protocol = benchmark_subparsers.add_parser(
        "validate-protocol",
        help="Validate a JSON-compatible YAML benchmark protocol",
    )
    protocol.add_argument("path")
    protocol.set_defaults(handler=_validate_benchmark_protocol)

    preflight = benchmark_subparsers.add_parser(
        "preflight",
        help="Fail closed unless the pinned local benchmark environment is ready",
    )
    preflight.add_argument("--protocol", default="benchmarks/protocol.yaml")
    preflight.add_argument("--snapshot", required=True)
    preflight.add_argument("--gpus", required=True, type=_gpu_ids)
    preflight.add_argument("--sglang-source", required=True)
    preflight.add_argument("--runtime-image", required=True)
    preflight.add_argument("--ffprobe-adapter", default=_default_ffprobe_adapter())
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(handler=_benchmark_preflight)

    launch = benchmark_subparsers.add_parser(
        "plan-launch",
        help="Emit the pinned Singularity SGLang launch argv",
    )
    launch.add_argument("--protocol", default="benchmarks/protocol.yaml")
    launch.add_argument("--snapshot", required=True)
    launch.add_argument("--gpus", required=True, type=_gpu_ids)
    launch.add_argument("--sglang-source", required=True)
    launch.add_argument("--runtime-image", required=True)
    launch.add_argument("--ffprobe-adapter", default=_default_ffprobe_adapter())
    launch.add_argument("--server-output", required=True)
    launch.add_argument("--port", type=int, default=30010)
    launch.set_defaults(handler=_benchmark_plan_launch)

    guarded = benchmark_subparsers.add_parser(
        "serve-guarded",
        help="Preflight, launch, and continuously guard the pinned server",
    )
    guarded.add_argument("--protocol", default="benchmarks/protocol.yaml")
    guarded.add_argument("--snapshot", required=True)
    guarded.add_argument("--gpus", required=True, type=_gpu_ids)
    guarded.add_argument("--sglang-source", required=True)
    guarded.add_argument("--runtime-image", required=True)
    guarded.add_argument("--ffprobe-adapter", default=_default_ffprobe_adapter())
    guarded.add_argument("--server-output", required=True)
    guarded.add_argument("--preflight-output", required=True)
    guarded.add_argument("--guard-report", required=True)
    guarded.add_argument(
        "--lifecycle-report",
        required=True,
        help="Write model-load-to-ready lifecycle metadata to this JSON path",
    )
    guarded.add_argument("--port", type=int, default=30010)
    guarded.add_argument("--startup-timeout", type=float, default=3600.0)
    guarded.add_argument("--poll-interval", type=float, default=2.0)
    guarded.set_defaults(handler=_benchmark_serve_guarded)

    run = benchmark_subparsers.add_parser(
        "run-case",
        help="Run one asynchronous local video benchmark case",
    )
    run.add_argument("--protocol", default="benchmarks/protocol.yaml")
    run.add_argument("--case-id", required=True)
    run.add_argument("--endpoint", default="http://127.0.0.1:30010")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--poll-interval", type=float, default=1.0)
    run.add_argument("--timeout", type=float, default=7200.0)
    run.set_defaults(handler=_benchmark_run_case)

    suite = benchmark_subparsers.add_parser(
        "run-suite",
        help="Run and aggregate the protocol warmup and measured cases",
    )
    suite.add_argument("--protocol", default="benchmarks/protocol.yaml")
    suite.add_argument("--case-id", required=True)
    suite.add_argument("--endpoint", default="http://127.0.0.1:30010")
    suite.add_argument("--output-dir", required=True)
    suite.add_argument(
        "--server-output",
        required=True,
        help="Host directory mounted by serve-guarded at /outputs",
    )
    suite.add_argument("--server-lifecycle-report", required=True)
    suite.add_argument("--server-guard-report", required=True)
    suite.add_argument("--poll-interval", type=float, default=1.0)
    suite.add_argument("--timeout", type=float, default=7200.0)
    suite.set_defaults(handler=_benchmark_run_suite)

    quality_reference = benchmark_subparsers.add_parser(
        "build-quality-reference",
        help="Build a redacted exact reference from a measured suite",
    )
    quality_reference.add_argument("--suite", required=True)
    quality_reference.add_argument("--protocol", default="benchmarks/protocol.yaml")
    quality_reference.add_argument("--reference-id", required=True)
    quality_reference.add_argument("--output", required=True)
    quality_reference.add_argument("--ffmpeg", default="ffmpeg")
    quality_reference.add_argument("--ffprobe", default="ffprobe")
    quality_reference.set_defaults(handler=_benchmark_build_quality_reference)

    quality_check = benchmark_subparsers.add_parser(
        "check-quality",
        help="Check a measured suite against an exact quality reference",
    )
    quality_check.add_argument("--reference", required=True)
    quality_check.add_argument("--suite", required=True)
    quality_check.add_argument("--protocol", default="benchmarks/protocol.yaml")
    quality_check.add_argument("--output", required=True)
    quality_check.add_argument("--ffmpeg", default="ffmpeg")
    quality_check.add_argument("--ffprobe", default="ffprobe")
    quality_check.set_defaults(handler=_benchmark_check_quality)

    quality_set_check = benchmark_subparsers.add_parser(
        "check-quality-set",
        help="Exit successfully only when the formal quality-set record is approved",
    )
    quality_set_check.add_argument("--record", required=True)
    quality_set_check.set_defaults(handler=_benchmark_check_quality_set)

    quality_metric_plan_check = benchmark_subparsers.add_parser(
        "check-quality-metric-plan",
        help="Exit successfully only when every formal metric plan is approved",
    )
    quality_metric_plan_check.add_argument("--plan", required=True)
    quality_metric_plan_check.set_defaults(handler=_benchmark_check_quality_metric_plan)

    quality_registry_compile = benchmark_subparsers.add_parser(
        "compile-quality-registry",
        help="Compile a private registry into redacted formal quality metadata",
    )
    quality_registry_compile.add_argument("--registry", required=True)
    quality_registry_compile.add_argument(
        "--template",
        default="benchmarks/quality/formal-quality-set.json",
    )
    quality_registry_compile.add_argument("--registry-uri", required=True)
    quality_registry_compile.add_argument("--output", required=True)
    quality_registry_compile.set_defaults(handler=_benchmark_compile_quality_registry)

    quality_review_prepare = benchmark_subparsers.add_parser(
        "prepare-quality-review",
        help="Create a local-only rights and selection review checklist",
    )
    quality_review_prepare.add_argument("--registry", required=True)
    quality_review_prepare.add_argument("--reviewer", required=True)
    quality_review_prepare.add_argument("--output", required=True)
    quality_review_prepare.set_defaults(handler=_benchmark_prepare_quality_review)

    quality_review_apply = benchmark_subparsers.add_parser(
        "apply-quality-review",
        help="Apply a complete local review to a new private registry",
    )
    quality_review_apply.add_argument("--registry", required=True)
    quality_review_apply.add_argument("--review", required=True)
    quality_review_apply.add_argument("--output", required=True)
    quality_review_apply.set_defaults(handler=_benchmark_apply_quality_review)

    human_pairwise_prepare = benchmark_subparsers.add_parser(
        "prepare-human-pairwise",
        help="Create a private blind ballot and separate assignment key",
    )
    human_pairwise_prepare.add_argument("--formal-set", required=True)
    human_pairwise_prepare.add_argument("--ballot-id", required=True)
    human_pairwise_prepare.add_argument("--reviewer", required=True)
    human_pairwise_prepare.add_argument("--randomization-seed-file", required=True)
    human_pairwise_prepare.add_argument("--ballot", required=True)
    human_pairwise_prepare.add_argument("--assignment", required=True)
    human_pairwise_prepare.set_defaults(handler=_benchmark_prepare_human_pairwise)

    human_pairwise_check = benchmark_subparsers.add_parser(
        "check-human-pairwise",
        help="Verify and score a complete private human-pairwise ballot",
    )
    human_pairwise_check.add_argument("--formal-set", required=True)
    human_pairwise_check.add_argument("--ballot", required=True)
    human_pairwise_check.add_argument("--assignment", required=True)
    human_pairwise_check.set_defaults(handler=_benchmark_check_human_pairwise)

    human_pairwise_stage = benchmark_subparsers.add_parser(
        "stage-human-pairwise",
        help="Stage digest-verified blinded A/B media with a local index page",
    )
    human_pairwise_stage.add_argument("--formal-set", required=True)
    human_pairwise_stage.add_argument("--ballot", required=True)
    human_pairwise_stage.add_argument("--assignment", required=True)
    human_pairwise_stage.add_argument("--media-manifest", required=True)
    human_pairwise_stage.add_argument("--staging-dir", required=True)
    human_pairwise_stage.set_defaults(handler=_benchmark_stage_human_pairwise)

    human_pairwise_record = benchmark_subparsers.add_parser(
        "record-human-pairwise",
        help="Record one reviewer selection on a pending private ballot",
    )
    human_pairwise_record.add_argument("--ballot", required=True)
    human_pairwise_record.add_argument("--case", required=True)
    human_pairwise_record.add_argument("--selection", required=True)
    human_pairwise_record.add_argument("--overwrite", action="store_true")
    human_pairwise_record.set_defaults(handler=_benchmark_record_human_pairwise)

    perceptual_video = benchmark_subparsers.add_parser(
        "score-perceptual-video",
        help="Score frame-aligned LPIPS between a baseline and a candidate video",
    )
    perceptual_video.add_argument("--baseline", required=True)
    perceptual_video.add_argument("--candidate", required=True)
    perceptual_video.add_argument("--backbone-dir", required=True)
    perceptual_video.add_argument("--expected-backbone-sha256", default=None)
    perceptual_video.add_argument("--ffmpeg", default="ffmpeg")
    perceptual_video.add_argument("--ffprobe", default="ffprobe")
    perceptual_video.set_defaults(handler=_benchmark_score_perceptual_video)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: CommandHandler = args.handler
    try:
        return handler(args)
    except H3FastError as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
