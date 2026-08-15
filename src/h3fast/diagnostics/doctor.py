"""CPU-safe environment diagnostics."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

from h3fast import __version__
from h3fast.backends.sglang import inspect_sglang

CheckStatus = Literal["pass", "info", "warning", "fail"]


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    """One diagnostic observation."""

    name: str
    status: CheckStatus
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-serializable check data."""
        return {"name": self.name, "status": self.status, "message": self.message}


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Collection of environment diagnostics."""

    checks: tuple[DiagnosticCheck, ...]

    @property
    def healthy(self) -> bool:
        """Return whether no mandatory check failed."""
        return all(check.status != "fail" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data."""
        return {
            "healthy": self.healthy,
            "checks": [check.to_dict() for check in self.checks],
        }


def _python_check() -> DiagnosticCheck:
    supported = sys.version_info[:2] == (3, 12)
    version = platform.python_version()
    if supported:
        return DiagnosticCheck("python", "pass", f"Python {version} is supported")
    return DiagnosticCheck(
        "python",
        "fail",
        f"Python {version} is unsupported; H3Fast currently requires Python 3.12",
    )


def _sglang_check() -> DiagnosticCheck:
    status = inspect_sglang()
    if status.installed_version is None:
        return DiagnosticCheck(
            "sglang",
            "info",
            f"SGLang is not installed; reference version is {status.reference_version}",
        )
    if status.compatible:
        return DiagnosticCheck(
            "sglang",
            "pass",
            f"SGLang {status.installed_version} matches the reference candidate",
        )
    return DiagnosticCheck(
        "sglang",
        "warning",
        f"SGLang {status.installed_version} is untested; expected "
        f"{status.reference_version}",
    )


def _accelerator_check() -> DiagnosticCheck:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return DiagnosticCheck(
            "accelerator",
            "info",
            "nvidia-smi is unavailable; CPU-only tooling remains supported",
        )

    try:
        result = subprocess.run(  # noqa: S603
            [
                executable,
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return DiagnosticCheck(
            "accelerator",
            "warning",
            f"nvidia-smi could not be executed: {error}",
        )

    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown nvidia-smi failure"
        return DiagnosticCheck("accelerator", "warning", detail)
    devices = result.stdout.strip() or "no NVIDIA devices reported"
    return DiagnosticCheck("accelerator", "pass", devices)


def run_doctor() -> DoctorReport:
    """Collect diagnostics without importing optional GPU frameworks."""
    checks = (
        DiagnosticCheck("h3fast", "pass", f"H3Fast {__version__}"),
        _python_check(),
        _sglang_check(),
        _accelerator_check(),
    )
    return DoctorReport(checks=checks)
