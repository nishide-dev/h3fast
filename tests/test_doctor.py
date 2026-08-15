"""Tests for environment diagnostics."""

import subprocess

from h3fast.backends.sglang import SGLangStatus
from h3fast.diagnostics.doctor import (
    DiagnosticCheck,
    DoctorReport,
    _accelerator_check,
    _sglang_check,
    run_doctor,
)


def test_doctor_report_fails_only_on_failed_checks() -> None:
    warning = DoctorReport(
        checks=(DiagnosticCheck("optional", "warning", "not installed"),)
    )
    failure = DoctorReport(checks=(DiagnosticCheck("python", "fail", "unsupported"),))

    assert warning.healthy is True
    assert failure.healthy is False
    assert failure.to_dict()["healthy"] is False


def test_sglang_check_reports_match_and_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3fast.diagnostics.doctor.inspect_sglang",
        lambda: SGLangStatus("0.5.15.post1", "0.5.15.post1", True),
    )
    assert _sglang_check().status == "pass"

    monkeypatch.setattr(
        "h3fast.diagnostics.doctor.inspect_sglang",
        lambda: SGLangStatus("0.5.14", "0.5.15.post1", False),
    )
    assert _sglang_check().status == "warning"


def test_accelerator_check_handles_success_and_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3fast.diagnostics.doctor.shutil.which", lambda _name: "/usr/bin/nvidia-smi"
    )
    monkeypatch.setattr(
        "h3fast.diagnostics.doctor.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="NVIDIA H100, 81920 MiB, 580.0, 9.0\n",
            stderr="",
        ),
    )
    assert _accelerator_check().status == "pass"

    monkeypatch.setattr(
        "h3fast.diagnostics.doctor.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="driver unavailable"
        ),
    )
    assert _accelerator_check().status == "warning"


def test_accelerator_check_handles_execution_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3fast.diagnostics.doctor.shutil.which", lambda _name: "/usr/bin/nvidia-smi"
    )

    def fail(*_args, **_kwargs):
        error = OSError("not executable")
        raise error

    monkeypatch.setattr("h3fast.diagnostics.doctor.subprocess.run", fail)

    assert _accelerator_check().status == "warning"


def test_run_doctor_collects_checks(monkeypatch) -> None:
    monkeypatch.setattr("h3fast.diagnostics.doctor.shutil.which", lambda _name: None)

    report = run_doctor()

    assert report.healthy is True
    assert len(report.checks) == 4
