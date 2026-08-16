"""Benchmark protocol, preflight, launch, and execution helpers."""

from h3fast.benchmarks.client import BenchmarkResult, run_case
from h3fast.benchmarks.guard import ForeignGpuProcess, serve_guarded
from h3fast.benchmarks.human_pairwise import (
    HumanPairwisePreparationReport,
    HumanPairwiseReport,
    check_human_pairwise_ballot,
    prepare_human_pairwise_ballot,
)
from h3fast.benchmarks.human_pairwise_runner import (
    HumanPairwiseRecordReport,
    HumanPairwiseStagingReport,
    record_human_pairwise_selection,
    stage_human_pairwise_presentation,
)
from h3fast.benchmarks.launch import LaunchPlan, build_singularity_launch
from h3fast.benchmarks.preflight import PreflightReport, run_preflight
from h3fast.benchmarks.protocol import (
    ProtocolReport,
    RuntimeSettings,
    load_runtime_settings,
    validate_protocol,
)
from h3fast.benchmarks.quality import build_quality_reference, check_quality
from h3fast.benchmarks.quality_metrics import (
    QualityMetricPlanReport,
    check_quality_metric_plan,
)
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
    "HumanPairwisePreparationReport",
    "HumanPairwiseRecordReport",
    "HumanPairwiseReport",
    "HumanPairwiseStagingReport",
    "LaunchPlan",
    "PreflightReport",
    "ProtocolReport",
    "QualityMetricPlanReport",
    "QualityRegistryCompileReport",
    "QualityRegistryReviewReport",
    "RuntimeSettings",
    "apply_quality_registry_review",
    "build_quality_reference",
    "build_singularity_launch",
    "check_formal_quality_set",
    "check_human_pairwise_ballot",
    "check_quality",
    "check_quality_metric_plan",
    "compile_quality_registry",
    "load_runtime_settings",
    "prepare_human_pairwise_ballot",
    "prepare_quality_registry_review",
    "record_human_pairwise_selection",
    "run_case",
    "run_preflight",
    "run_suite",
    "serve_guarded",
    "stage_human_pairwise_presentation",
    "validate_protocol",
]
