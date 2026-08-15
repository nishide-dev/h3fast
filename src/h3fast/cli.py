"""H3Fast command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from h3fast import __version__
from h3fast.benchmarks import validate_protocol
from h3fast.diagnostics import run_doctor
from h3fast.exceptions import H3FastError
from h3fast.manifest import inspect_snapshot, verify_model_artifact

CommandHandler = Callable[[argparse.Namespace], int]


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


def _validate_benchmark_protocol(args: argparse.Namespace) -> int:
    report = validate_protocol(Path(args.path))
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
