"""Tests for repeated baseline measurement and aggregation."""

import json
from pathlib import Path

import pytest

from h3fast.benchmarks.client import BenchmarkResult
from h3fast.benchmarks.suite import _load_guard_failure, run_suite, summarize
from h3fast.exceptions import ValidationError


def _lifecycle(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "ready",
                "started_at": "2026-08-15T00:00:00+00:00",
                "ready_at": "2026-08-15T00:10:00+00:00",
                "startup_seconds": 600.0,
                "selected_gpus": [1, 2],
                "server_pid": 100,
                "endpoint": "http://127.0.0.1:30010",
                "runtime_settings": {
                    "dit_layerwise_resident_layers": 40,
                    "attention_backend": "auto",
                    "model_variant": "fl2va",
                    "lora": None,
                    "quantization": None,
                    "synchronized_stage_profiling": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _result(index: int) -> BenchmarkResult:
    stages = [
        "InputValidationStage",
        "MiniMaxH3PartitionAdmissionStage",
        "MiniMaxH3TextEncodingStage",
        "MiniMaxH3VisualEncodingStage",
        "MiniMaxH3AudioEncodingStage",
        "MiniMaxH3LatentPreparationStage",
        "MiniMaxH3TimestepPreparationStage",
        "MiniMaxH3DenoisingStage",
        "MiniMaxH3DecodingStage",
    ]
    return BenchmarkResult(
        protocol_id="h3fast-phase1b-resident40-v1",
        case_id="smoke-001",
        job_id=f"job-{index}",
        started_at="2026-08-15T00:00:00+00:00",
        completed_at="2026-08-15T00:01:00+00:00",
        elapsed_seconds=float(index + 10),
        prompt_sha256="a" * 64,
        request={},
        artifact_path=f"/tmp/{index}.mp4",
        artifact_size=10,
        artifact_sha256="b" * 64,
        server={
            "inference_time_seconds": float(index + 9),
            "peak_memory_mib": float(index + 1000),
            "media_contract": {"size": "1344x768", "seconds": "5.166667"},
            "performance": {
                "sglang_commit": "6eb941a34cb100b708a42ed1d26d2bdefafbd01e",
                "pipeline_total_seconds": float(index + 8),
                "stages": [
                    {
                        "name": name,
                        "seconds": (
                            float(index + 7)
                            if name == "MiniMaxH3DenoisingStage"
                            else 1.0
                        ),
                    }
                    for name in stages
                ],
                "denoise_steps_seconds": [1.0],
            },
        },
    )


def test_summarize_uses_linear_percentiles() -> None:
    result = summarize([10.0, 20.0, 30.0]).to_dict()

    assert result == {"min": 10.0, "p50": 20.0, "p95": 29.0, "max": 30.0}


@pytest.mark.parametrize("values", [[], [-1.0], [float("inf")]])
def test_summarize_rejects_invalid_values(values: list[float]) -> None:
    with pytest.raises(ValidationError, match="finite non-negative"):
        summarize(values)


def test_run_suite_executes_protocol_plan_and_aggregates(tmp_path: Path) -> None:
    lifecycle = tmp_path / "lifecycle.json"
    _lifecycle(lifecycle)
    server_output = tmp_path / "server"
    server_output.mkdir()
    calls: list[dict[str, object]] = []

    def runner(*_args, **kwargs):
        calls.append(kwargs)
        return _result(len(calls))

    result = run_suite(
        Path("benchmarks/protocol.yaml"),
        case_id="smoke-001",
        endpoint="http://127.0.0.1:30010",
        output_dir=tmp_path / "suite",
        server_output_dir=server_output,
        server_lifecycle_path=lifecycle,
        server_guard_report_path=tmp_path / "guard.json",
        case_runner=runner,
    )

    assert len(calls) == 4
    assert result.warmup_runs == 1
    assert result.measured_runs == 3
    assert result.aggregate["dominant_stage"]["name"] == "MiniMaxH3DenoisingStage"
    assert result.aggregate["client_elapsed_seconds"]["p50"] == 13.0
    saved = json.loads(
        (
            tmp_path / "suite" / "h3fast-phase1b-resident40-v1-smoke-001-suite.json"
        ).read_text(encoding="utf-8")
    )
    assert saved["status"] == "completed"
    assert saved["server_lifecycle"]["startup_seconds"] == 600.0
    assert saved["server_lifecycle"]["runtime_settings"] == {
        "dit_layerwise_resident_layers": 40,
        "attention_backend": "auto",
        "model_variant": "fl2va",
        "lora": None,
        "quantization": None,
        "synchronized_stage_profiling": False,
    }


def test_run_suite_records_failure_bundle(tmp_path: Path) -> None:
    lifecycle = tmp_path / "lifecycle.json"
    _lifecycle(lifecycle)
    server_output = tmp_path / "server"
    server_output.mkdir()
    guard = tmp_path / "guard.json"
    guard.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "failed",
                "error": "foreign GPU process",
            }
        ),
        encoding="utf-8",
    )

    def fail(*_args, **_kwargs):
        message = "GPU guard failed"
        raise ValidationError(message)

    with pytest.raises(ValidationError, match="GPU guard"):
        run_suite(
            Path("benchmarks/protocol.yaml"),
            case_id="smoke-001",
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "suite",
            server_output_dir=server_output,
            server_lifecycle_path=lifecycle,
            server_guard_report_path=guard,
            case_runner=fail,
        )

    saved = json.loads(
        (
            tmp_path / "suite" / "h3fast-phase1b-resident40-v1-smoke-001-suite.json"
        ).read_text(encoding="utf-8")
    )
    assert saved["status"] == "failed"
    assert saved["runs"][0]["error"] == "GPU guard failed"
    assert saved["guard"]["error"] == "foreign GPU process"


def test_run_suite_requires_server_output_and_matching_lifecycle(
    tmp_path: Path,
) -> None:
    lifecycle = tmp_path / "lifecycle.json"
    _lifecycle(lifecycle)

    with pytest.raises(ValidationError, match="server output directory"):
        run_suite(
            Path("benchmarks/protocol.yaml"),
            case_id="smoke-001",
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "suite",
            server_output_dir=tmp_path / "missing",
            server_lifecycle_path=lifecycle,
            server_guard_report_path=tmp_path / "guard.json",
        )

    server_output = tmp_path / "server"
    server_output.mkdir()
    with pytest.raises(ValidationError, match="ready endpoint"):
        run_suite(
            Path("benchmarks/protocol.yaml"),
            case_id="smoke-001",
            endpoint="http://localhost:30010",
            output_dir=tmp_path / "suite",
            server_output_dir=server_output,
            server_lifecycle_path=lifecycle,
            server_guard_report_path=tmp_path / "guard.json",
        )


def test_run_suite_rejects_lifecycle_runtime_mismatch(tmp_path: Path) -> None:
    lifecycle = tmp_path / "lifecycle.json"
    _lifecycle(lifecycle)
    lifecycle_value = json.loads(lifecycle.read_text(encoding="utf-8"))
    lifecycle_value["runtime_settings"] = {
        "dit_layerwise_resident_layers": 20,
        "attention_backend": "auto",
    }
    lifecycle.write_text(json.dumps(lifecycle_value), encoding="utf-8")
    server_output = tmp_path / "server"
    server_output.mkdir()

    with pytest.raises(ValidationError, match="runtime settings"):
        run_suite(
            Path("benchmarks/protocol.yaml"),
            case_id="smoke-001",
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "suite",
            server_output_dir=server_output,
            server_lifecycle_path=lifecycle,
            server_guard_report_path=tmp_path / "guard.json",
        )


def test_guard_failure_loader_handles_absent_and_invalid_reports(
    tmp_path: Path,
) -> None:
    path = tmp_path / "guard.json"
    assert _load_guard_failure(path) is None

    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValidationError, match="could not read GPU guard"):
        _load_guard_failure(path)

    path.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
    with pytest.raises(ValidationError, match="report is invalid"):
        _load_guard_failure(path)


def test_run_suite_records_post_run_stage_mismatch(tmp_path: Path) -> None:
    lifecycle = tmp_path / "lifecycle.json"
    _lifecycle(lifecycle)
    server_output = tmp_path / "server"
    server_output.mkdir()

    def runner(*_args, **_kwargs):
        result = _result(1)
        result.server["performance"]["stages"].pop()
        return result

    output = tmp_path / "suite"
    with pytest.raises(ValidationError, match="do not match"):
        run_suite(
            Path("benchmarks/protocol.yaml"),
            case_id="smoke-001",
            endpoint="http://127.0.0.1:30010",
            output_dir=output,
            server_output_dir=server_output,
            server_lifecycle_path=lifecycle,
            server_guard_report_path=tmp_path / "guard.json",
            case_runner=runner,
        )

    saved = json.loads(
        (output / "h3fast-phase1b-resident40-v1-smoke-001-suite.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["status"] == "failed"
    assert "do not match" in saved["error"]
