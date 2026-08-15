"""Benchmark protocol, preflight, launch, and execution helpers."""

from h3fast.benchmarks.client import BenchmarkResult, run_case
from h3fast.benchmarks.guard import ForeignGpuProcess, serve_guarded
from h3fast.benchmarks.launch import LaunchPlan, build_singularity_launch
from h3fast.benchmarks.preflight import PreflightReport, run_preflight
from h3fast.benchmarks.protocol import (
    ProtocolReport,
    RuntimeSettings,
    load_runtime_settings,
    validate_protocol,
)
from h3fast.benchmarks.quality import build_quality_reference, check_quality
from h3fast.benchmarks.quality_registry import (
    QualityRegistryCompileReport,
    QualityRegistryReviewReport,
    apply_quality_registry_review,
    compile_quality_registry,
    prepare_quality_registry_review,
)
from h3fast.benchmarks.quality_sets import (
    FormalQualitySetReport,
    check_formal_quality_set,
)
from h3fast.benchmarks.suite import BenchmarkSuiteResult, run_suite

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "ForeignGpuProcess",
    "FormalQualitySetReport",
    "LaunchPlan",
    "PreflightReport",
    "ProtocolReport",
    "QualityRegistryCompileReport",
    "QualityRegistryReviewReport",
    "RuntimeSettings",
    "apply_quality_registry_review",
    "build_quality_reference",
    "build_singularity_launch",
    "check_formal_quality_set",
    "check_quality",
    "compile_quality_registry",
    "load_runtime_settings",
    "prepare_quality_registry_review",
    "run_case",
    "run_preflight",
    "run_suite",
    "serve_guarded",
    "validate_protocol",
]
