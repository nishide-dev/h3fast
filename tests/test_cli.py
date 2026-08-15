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


def test_benchmark_plan_and_run_case_commands(monkeypatch, capsys) -> None:
    class Value:
        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"ok": True}

    monkeypatch.setattr(
        "h3fast.cli.build_singularity_launch", lambda **_kwargs: Value()
    )
    assert (
        main(
            [
                "benchmark",
                "plan-launch",
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

    monkeypatch.setattr("h3fast.cli.run_case", lambda *_args, **_kwargs: Value())
    assert (
        main(
            [
                "benchmark",
                "run-case",
                "--case-id",
                "smoke-001",
                "--output-dir",
                "output",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True


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
