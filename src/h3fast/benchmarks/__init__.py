"""Benchmark protocol, preflight, launch, and execution helpers."""

from h3fast.benchmarks.client import BenchmarkResult, run_case
from h3fast.benchmarks.guard import ForeignGpuProcess, serve_guarded
from h3fast.benchmarks.launch import LaunchPlan, build_singularity_launch
from h3fast.benchmarks.preflight import PreflightReport, run_preflight
from h3fast.benchmarks.protocol import ProtocolReport, validate_protocol
from h3fast.benchmarks.suite import BenchmarkSuiteResult, run_suite

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "ForeignGpuProcess",
    "LaunchPlan",
    "PreflightReport",
    "ProtocolReport",
    "build_singularity_launch",
    "run_case",
    "run_preflight",
    "run_suite",
    "serve_guarded",
    "validate_protocol",
]
