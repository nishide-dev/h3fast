"""Repeated guarded benchmark measurement and deterministic aggregation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from h3fast.benchmarks.client import BenchmarkResult, run_case
from h3fast.benchmarks.protocol import load_runtime_settings, validate_protocol
from h3fast.exceptions import H3FastError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class _MeasuredMetrics(TypedDict):
    client_elapsed_seconds: float
    server_inference_seconds: float
    peak_gpu_memory_mib: float
    pipeline_total_seconds: float
    stages: dict[str, float]


@dataclass(frozen=True, slots=True)
class SummaryStatistics:
    """Protocol statistics for one numeric measurement."""

    minimum: float
    p50: float
    p95: float
    maximum: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable statistics."""
        return {
            "min": self.minimum,
            "p50": self.p50,
            "p95": self.p95,
            "max": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    """One completed warmup and measured benchmark suite."""

    protocol_id: str
    case_id: str
    started_at: str
    completed_at: str
    warmup_runs: int
    measured_runs: int
    server_lifecycle: dict[str, object]
    guard: dict[str, object]
    runs: tuple[dict[str, object], ...]
    aggregate: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return the machine-readable benchmark bundle."""
        return {
            "schema_version": "1.0",
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "status": "completed",
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "server_lifecycle": self.server_lifecycle,
            "guard": self.guard,
            "runs": list(self.runs),
            "aggregate": self.aggregate,
        }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values: list[float]) -> SummaryStatistics:
    """Calculate the protocol's min, p50, p95, and max statistics."""
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        message = "benchmark statistics require finite non-negative values"
        raise ValidationError(message)
    return SummaryStatistics(
        minimum=min(values),
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
        maximum=max(values),
    )


def _measurement_plan(
    protocol_path: Path,
) -> tuple[str, int, int, tuple[str, ...]]:
    report = validate_protocol(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    measurement = protocol.get("measurement")
    if not isinstance(measurement, dict):
        message = "benchmark protocol measurement must be an object"
        raise ValidationError(message)
    warmups = measurement.get("warmup_runs")
    measured = measurement.get("measured_runs")
    if (
        not isinstance(warmups, int)
        or isinstance(warmups, bool)
        or warmups < 0
        or not isinstance(measured, int)
        or isinstance(measured, bool)
        or measured <= 0
    ):
        message = "benchmark protocol run counts are invalid"
        raise ValidationError(message)
    if measurement.get("statistics") != ["min", "p50", "p95", "max"]:
        message = "benchmark suite supports the fixed min/p50/p95/max statistics"
        raise ValidationError(message)
    stages = measurement.get("stages")
    if (
        not isinstance(stages, list)
        or not stages
        or not all(isinstance(stage, str) and stage for stage in stages)
        or len(set(stages)) != len(stages)
    ):
        message = "benchmark protocol stages must be non-empty unique names"
        raise ValidationError(message)
    return report.protocol_id, warmups, measured, tuple(stages)


def _required_number(mapping: dict[str, object], field: str) -> float:
    value = mapping.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        message = f"completed benchmark result is missing server metric {field!r}"
        raise ValidationError(message)
    return float(value)


def _measured_metrics(result: BenchmarkResult) -> _MeasuredMetrics:
    server = result.server
    performance = server.get("performance")
    media_contract = server.get("media_contract")
    if not isinstance(performance, dict) or not isinstance(media_contract, dict):
        message = "measured run requires performance and media contract metadata"
        raise ValidationError(message)
    stages_raw = performance.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        message = "measured run requires server stage timings"
        raise ValidationError(message)
    stages: dict[str, float] = {}
    for stage in stages_raw:
        if not isinstance(stage, dict) or not isinstance(stage.get("name"), str):
            message = "measured run contains an invalid server stage"
            raise ValidationError(message)
        stages[str(stage["name"])] = _required_number(stage, "seconds")
    return {
        "client_elapsed_seconds": result.elapsed_seconds,
        "server_inference_seconds": _required_number(server, "inference_time_seconds"),
        "peak_gpu_memory_mib": _required_number(server, "peak_memory_mib"),
        "pipeline_total_seconds": _required_number(
            performance, "pipeline_total_seconds"
        ),
        "stages": stages,
    }


def _aggregate(
    results: list[BenchmarkResult], expected_stages: tuple[str, ...]
) -> dict[str, object]:
    metrics = [_measured_metrics(result) for result in results]
    stage_names = set(metrics[0]["stages"])
    if any(set(metric["stages"]) != stage_names for metric in metrics[1:]):
        message = "measured runs returned inconsistent stage sets"
        raise ValidationError(message)
    if stage_names != set(expected_stages):
        message = "server stage timings do not match the benchmark protocol"
        raise ValidationError(message)

    aggregate: dict[str, object] = {}
    for field in (
        "client_elapsed_seconds",
        "server_inference_seconds",
        "pipeline_total_seconds",
        "peak_gpu_memory_mib",
    ):
        aggregate[field] = summarize(
            [float(metric[field]) for metric in metrics]
        ).to_dict()
    stage_aggregate = {
        name: summarize([float(metric["stages"][name]) for metric in metrics]).to_dict()
        for name in sorted(stage_names)
    }
    aggregate["stages_seconds"] = stage_aggregate
    bottleneck = max(
        stage_aggregate,
        key=lambda name: float(stage_aggregate[name]["p50"]),
    )
    aggregate["dominant_stage"] = {
        "name": bottleneck,
        "p50_seconds": stage_aggregate[bottleneck]["p50"],
    }
    return aggregate


def _write_bundle(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_lifecycle(
    path: Path, endpoint: str, expected_runtime_settings: dict[str, object]
) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"could not read server lifecycle report {path}: {error}"
        raise ValidationError(message) from error
    if (
        not isinstance(value, dict)
        or value.get("status") != "ready"
        or value.get("endpoint") != endpoint.rstrip("/")
    ):
        message = "server lifecycle report does not match the ready endpoint"
        raise ValidationError(message)
    _required_number(value, "startup_seconds")
    # Synchronized stage profiling is a launch-time diagnostic switch, not a
    # protocol-owned setting: it changes timing attribution, not the compute
    # graph, so it is recorded but excluded from the protocol match.
    observed = dict(value.get("runtime_settings") or {})
    observed.pop("synchronized_stage_profiling", None)
    # The GPU topology is a launch-time fact recorded for reproducibility, not
    # a protocol-owned setting; the protocol pins compute, not placement.
    observed.pop("tensor_parallel_size", None)
    observed.pop("ulysses_degree", None)
    observed.pop("text_encoder_override", None)
    if observed != expected_runtime_settings:
        message = "server lifecycle runtime settings do not match the protocol"
        raise ValidationError(message)
    return value


def _load_guard_failure(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"could not read GPU guard failure report {path}: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict) or value.get("status") != "failed":
        message = "GPU guard failure report is invalid"
        raise ValidationError(message)
    return value


def run_suite(
    protocol_path: Path,
    *,
    case_id: str,
    endpoint: str,
    output_dir: Path,
    server_output_dir: Path,
    server_lifecycle_path: Path,
    server_guard_report_path: Path,
    poll_interval: float = 1.0,
    timeout: float = 7200.0,
    case_runner: Callable[..., BenchmarkResult] = run_case,
) -> BenchmarkSuiteResult:
    """Run the protocol warmups and measurements against one ready server."""
    protocol_id, warmup_count, measured_count, expected_stages = _measurement_plan(
        protocol_path
    )
    if not server_output_dir.is_dir():
        message = f"server output directory is missing: {server_output_dir}"
        raise ValidationError(message)
    runtime_settings = load_runtime_settings(protocol_path)
    lifecycle = _load_lifecycle(
        server_lifecycle_path, endpoint, runtime_settings.to_dict()
    )
    started = datetime.now(UTC)
    runs: list[dict[str, object]] = []
    measured_results: list[BenchmarkResult] = []
    bundle_path = output_dir / f"{protocol_id}-{case_id}-suite.json"
    plan = [("warmup", index + 1) for index in range(warmup_count)] + [
        ("measured", index + 1) for index in range(measured_count)
    ]
    for kind, index in plan:
        label = f"{kind}-{index:03d}"
        host_perf_path = (
            server_output_dir
            / "h3fast-metrics"
            / f"{protocol_id}-{case_id}-{label}.json"
        )
        container_perf_path = f"/outputs/h3fast-metrics/{host_perf_path.name}"
        try:
            result = case_runner(
                protocol_path,
                case_id=case_id,
                endpoint=endpoint,
                output_dir=output_dir / "runs" / label,
                poll_interval=poll_interval,
                timeout=timeout,
                server_perf_dump_path=container_perf_path,
                performance_dump_path=host_perf_path,
            )
        except H3FastError as error:
            runs.append(
                {
                    "label": label,
                    "kind": kind,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            failure_bundle: dict[str, object] = {
                "schema_version": "1.0",
                "protocol_id": protocol_id,
                "case_id": case_id,
                "status": "failed",
                "started_at": started.isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
                "warmup_runs": warmup_count,
                "measured_runs": measured_count,
                "server_lifecycle": lifecycle,
                "error_type": type(error).__name__,
                "error": str(error),
                "runs": runs,
            }
            guard_failure = _load_guard_failure(server_guard_report_path)
            if guard_failure is not None:
                failure_bundle["guard"] = guard_failure
            _write_bundle(
                bundle_path,
                failure_bundle,
            )
            raise
        runs.append(
            {
                "label": label,
                "kind": kind,
                "status": "completed",
                "result": result.to_dict(),
            }
        )
        if kind == "measured":
            measured_results.append(result)

    guard_failure: dict[str, object] | None = None
    try:
        guard_failure = _load_guard_failure(server_guard_report_path)
        if guard_failure is not None:
            message = f"GPU guard failed: {guard_failure.get('error', 'unknown error')}"
            raise ValidationError(message)
        aggregate = _aggregate(measured_results, expected_stages)
    except H3FastError as error:
        failure_bundle: dict[str, object] = {
            "schema_version": "1.0",
            "protocol_id": protocol_id,
            "case_id": case_id,
            "status": "failed",
            "started_at": started.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "warmup_runs": warmup_count,
            "measured_runs": measured_count,
            "server_lifecycle": lifecycle,
            "error_type": type(error).__name__,
            "error": str(error),
            "runs": runs,
        }
        if guard_failure is not None:
            failure_bundle["guard"] = guard_failure
        _write_bundle(bundle_path, failure_bundle)
        raise

    suite = BenchmarkSuiteResult(
        protocol_id=protocol_id,
        case_id=case_id,
        started_at=started.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        warmup_runs=warmup_count,
        measured_runs=measured_count,
        server_lifecycle=lifecycle,
        guard={"status": "passed", "failure_report_present": False},
        runs=tuple(runs),
        aggregate=aggregate,
    )
    _write_bundle(bundle_path, suite.to_dict())
    return suite
