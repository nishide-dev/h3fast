"""Tests for the public command-line interface."""

import argparse
import json
from importlib import metadata
from pathlib import Path

import pytest

from h3fast.cli import _gpu_ids, main
from h3fast.exceptions import ValidationError
from h3fast.manifest.snapshot import REQUIRED_COMPONENTS
from tests.test_model_manifest import _write_artifact


def _write_snapshot(root: Path) -> None:
    (root / "model_index.json").write_text("{}\n", encoding="utf-8")
    variant_root = root / "FL2VA"
    variant_root.mkdir()
    (variant_root / "model_index.json").write_text("{}\n", encoding="utf-8")
    for component in REQUIRED_COMPONENTS:
        component_root = variant_root / component
        component_root.mkdir()
        (component_root / "config.json").write_text("{}\n", encoding="utf-8")


def test_doctor_json_is_cpu_safe(monkeypatch, capsys) -> None:
    def missing_distribution(_name: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", missing_distribution)
    monkeypatch.setattr("h3fast.diagnostics.doctor.shutil.which", lambda _name: None)

    status = main(["doctor", "--json"])

    output = json.loads(capsys.readouterr().out)
    assert status == 0
    assert output["healthy"] is True
    assert {check["name"] for check in output["checks"]} == {
        "h3fast",
        "python",
        "sglang",
        "accelerator",
    }


def test_invalid_snapshot_returns_user_error(tmp_path, capsys) -> None:
    status = main(
        [
            "inspect-snapshot",
            str(tmp_path),
            "--variant",
            "fl2va",
            "--base-revision",
            "main",
        ]
    )

    assert status == 2
    assert "40-character commit SHA" in capsys.readouterr().err


def test_doctor_human_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr("h3fast.diagnostics.doctor.shutil.which", lambda _name: None)

    assert main(["doctor"]) == 0
    assert "[PASS" in capsys.readouterr().out


def test_inspect_snapshot_writes_report(tmp_path: Path, capsys) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _write_snapshot(snapshot)
    output = tmp_path / "reports" / "snapshot.json"

    status = main(
        [
            "inspect-snapshot",
            str(snapshot),
            "--variant",
            "fl2va",
            "--base-revision",
            "a" * 40,
            "--hash",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is True
    assert "Wrote snapshot report" in capsys.readouterr().out


def test_verify_model_command(tmp_path: Path, capsys) -> None:
    _write_artifact(tmp_path)

    assert main(["verify-model", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["artifact_id"] == "h3fast-test"


def test_benchmark_protocol_command(capsys) -> None:
    status = main(["benchmark", "validate-protocol", "benchmarks/protocol.yaml"])

    assert status == 0
    assert json.loads(capsys.readouterr().out)["status"] == "draft"


def test_gpu_ids_parser_rejects_invalid_values() -> None:
    assert _gpu_ids("1,2") == (1, 2)
    with pytest.raises(argparse.ArgumentTypeError):
        _gpu_ids("one,two")
    with pytest.raises(argparse.ArgumentTypeError):
        _gpu_ids("1,1")


def test_benchmark_preflight_command_writes_report(
    tmp_path, monkeypatch, capsys
) -> None:
    class Report:
        ready = True

        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"ready": True}

    monkeypatch.setattr("h3fast.cli.run_preflight", lambda *_args, **_kwargs: Report())
    output = tmp_path / "preflight.json"

    status = main(
        [
            "benchmark",
            "preflight",
            "--snapshot",
            "snapshot",
            "--gpus",
            "1,2",
            "--sglang-source",
            "source",
            "--runtime-image",
            "runtime.sif",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["ready"] is True
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_benchmark_plan_command(monkeypatch, capsys) -> None:
    class Value:
        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"ok": True}

    arguments: list[dict[str, object]] = []

    def build(**kwargs):
        arguments.append(kwargs)
        return Value()

    monkeypatch.setattr("h3fast.cli.build_singularity_launch", build)
    assert (
        main(
            [
                "benchmark",
                "plan-launch",
                "--protocol",
                "benchmarks/protocol.yaml",
                "--snapshot",
                "snapshot",
                "--gpus",
                "1,2",
                "--sglang-source",
                "source",
                "--runtime-image",
                "runtime.sif",
                "--server-output",
                "output",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert arguments[0]["dit_layerwise_resident_layers"] == 40


def test_benchmark_serve_guarded_runs_preflight_before_launch(
    tmp_path, monkeypatch
) -> None:
    class Report:
        ready = True

        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"ready": True}

    class Plan:
        pass

    guarded: list[object] = []
    monkeypatch.setattr("h3fast.cli.run_preflight", lambda *_args, **_kwargs: Report())
    launch_arguments: list[dict[str, object]] = []

    def build(**kwargs):
        launch_arguments.append(kwargs)
        return Plan()

    monkeypatch.setattr("h3fast.cli.build_singularity_launch", build)
    monkeypatch.setattr(
        "h3fast.cli.serve_guarded",
        lambda plan, **_kwargs: guarded.append(plan),
    )
    preflight = tmp_path / "preflight.json"

    status = main(
        [
            "benchmark",
            "serve-guarded",
            "--protocol",
            "benchmarks/protocol.yaml",
            "--snapshot",
            "snapshot",
            "--gpus",
            "1,2",
            "--sglang-source",
            "source",
            "--runtime-image",
            "runtime.sif",
            "--server-output",
            "server",
            "--preflight-output",
            str(preflight),
            "--guard-report",
            str(tmp_path / "guard.json"),
            "--lifecycle-report",
            str(tmp_path / "lifecycle.json"),
        ]
    )

    assert status == 0
    assert json.loads(preflight.read_text(encoding="utf-8"))["ready"] is True
    assert len(guarded) == 1
    assert launch_arguments[0]["dit_layerwise_resident_layers"] == 40


def test_benchmark_serve_guarded_requires_lifecycle_report() -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "benchmark",
                "serve-guarded",
                "--snapshot",
                "snapshot",
                "--gpus",
                "1,2",
                "--sglang-source",
                "source",
                "--runtime-image",
                "runtime.sif",
                "--server-output",
                "server",
                "--preflight-output",
                "preflight.json",
                "--guard-report",
                "guard.json",
            ]
        )

    assert error.value.code == 2


def test_benchmark_run_case_command(tmp_path, monkeypatch, capsys) -> None:
    class Value:
        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"ok": True}

    monkeypatch.setattr("h3fast.cli.run_case", lambda *_args, **_kwargs: Value())
    output = tmp_path / "output"
    output.mkdir()
    failure = output / "smoke-001-failure.json"
    failure.write_text("stale", encoding="utf-8")
    assert (
        main(
            [
                "benchmark",
                "run-case",
                "--case-id",
                "smoke-001",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert not failure.exists()


def test_benchmark_run_case_records_failure(tmp_path, monkeypatch, capsys) -> None:
    def fail(*_args, **_kwargs):
        message = "server failed"
        raise ValidationError(message)

    monkeypatch.setattr("h3fast.cli.run_case", fail)
    output = tmp_path / "results"

    status = main(
        [
            "benchmark",
            "run-case",
            "--case-id",
            "smoke-001",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 2
    failure = json.loads(
        (output / "smoke-001-failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["error"] == "server failed"
    assert "server failed" in capsys.readouterr().err


def test_benchmark_run_suite_command(tmp_path, monkeypatch, capsys) -> None:
    class Value:
        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"status": "completed"}

    monkeypatch.setattr("h3fast.cli.run_suite", lambda *_args, **_kwargs: Value())

    status = main(
        [
            "benchmark",
            "run-suite",
            "--case-id",
            "smoke-001",
            "--output-dir",
            str(tmp_path / "suite"),
            "--server-output",
            str(tmp_path / "server"),
            "--server-lifecycle-report",
            str(tmp_path / "lifecycle.json"),
            "--server-guard-report",
            str(tmp_path / "guard.json"),
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_benchmark_quality_reference_command(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "reference.json"

    def build(*_args, **_kwargs):
        output.write_text('{"reference_id":"reference-v1"}\n', encoding="utf-8")
        return {"reference_id": "reference-v1"}

    monkeypatch.setattr("h3fast.cli.build_quality_reference", build)

    status = main(
        [
            "benchmark",
            "build-quality-reference",
            "--suite",
            "suite.json",
            "--reference-id",
            "reference-v1",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["reference_id"] == "reference-v1"


@pytest.mark.parametrize(
    ("quality_status", "exit_status"), [("passed", 0), ("failed", 1)]
)
def test_benchmark_quality_check_command(
    tmp_path, monkeypatch, capsys, quality_status: str, exit_status: int
) -> None:
    output = tmp_path / "quality-report.json"

    def check(*_args, **_kwargs):
        output.write_text(
            json.dumps({"status": quality_status}) + "\n", encoding="utf-8"
        )
        return {"status": quality_status}

    monkeypatch.setattr("h3fast.cli.check_quality", check)

    status = main(
        [
            "benchmark",
            "check-quality",
            "--reference",
            "reference.json",
            "--suite",
            "suite.json",
            "--output",
            str(output),
        ]
    )

    assert status == exit_status
    assert json.loads(capsys.readouterr().out)["status"] == quality_status
