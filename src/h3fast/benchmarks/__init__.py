"""Benchmark protocol, preflight, launch, and execution helpers."""

from h3fast.benchmarks.backend_verification import (
    BackendVerificationReport,
    verify_attention_backend,
)
from h3fast.benchmarks.client import BenchmarkResult, run_case, run_supplied_case
from h3fast.benchmarks.formal_runner import FormalRunReport, run_formal_cases
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
from h3fast.benchmarks.perceptual_video import (
    PerceptualVideoReport,
    score_perceptual_video,
)
from h3fast.benchmarks.preflight import PreflightReport, run_preflight
from h3fast.benchmarks.profiles import (
    DEFAULT_GENERATION_PROFILE,
    GENERATION_PROFILES,
    GenerationProfile,
    resolve_generation_profile,
)
from h3fast.benchmarks.prompt_adherence import (
    PromptAdherenceReport,
    score_prompt_adherence,
)
from h3fast.benchmarks.protocol import (
    ProtocolReport,
    RuntimeSettings,
    load_runtime_settings,
    load_sglang_revision,
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
from h3fast.benchmarks.temporal_consistency import (
    TemporalConsistencyReport,
    score_temporal_consistency,
)

__all__ = [
    "DEFAULT_GENERATION_PROFILE",
    "GENERATION_PROFILES",
    "BackendVerificationReport",
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "ForeignGpuProcess",
    "FormalQualitySetReport",
    "FormalRunReport",
    "GenerationProfile",
    "HumanPairwisePreparationReport",
    "HumanPairwiseRecordReport",
    "HumanPairwiseReport",
    "HumanPairwiseStagingReport",
    "LaunchPlan",
    "PerceptualVideoReport",
    "PreflightReport",
    "PromptAdherenceReport",
    "ProtocolReport",
    "QualityMetricPlanReport",
    "QualityRegistryCompileReport",
    "QualityRegistryReviewReport",
    "RuntimeSettings",
    "TemporalConsistencyReport",
    "apply_quality_registry_review",
    "build_quality_reference",
    "build_singularity_launch",
    "check_formal_quality_set",
    "check_human_pairwise_ballot",
    "check_quality",
    "check_quality_metric_plan",
    "compile_quality_registry",
    "load_runtime_settings",
    "load_sglang_revision",
    "prepare_human_pairwise_ballot",
    "prepare_quality_registry_review",
    "record_human_pairwise_selection",
    "resolve_generation_profile",
    "run_case",
    "run_formal_cases",
    "run_preflight",
    "run_suite",
    "run_supplied_case",
    "score_perceptual_video",
    "score_prompt_adherence",
    "score_temporal_consistency",
    "serve_guarded",
    "stage_human_pairwise_presentation",
    "validate_protocol",
    "verify_attention_backend",
]
