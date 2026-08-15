"""Fail-closed release readiness checks."""

from h3fast.release.gates import ReleaseGateReport, check_release_gate

__all__ = ["ReleaseGateReport", "check_release_gate"]
