"""Fail-closed validation for formal quality metric plans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

PLAN_ID = "h3fast-phase0-formal-quality-metrics-v1"
REQUIRED_METRIC_FAMILIES = frozenset(
    {
        "audio-quality",
        "av-sync",
        "human-pairwise",
        "perceptual-video",
        "prompt-adherence",
        "temporal-consistency",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "plan_id",
        "status",
        "updated_at",
        "source_issue",
        "quality_profile",
        "evaluation",
        "metrics",
        "limitations",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "baseline_repetitions",
        "candidate_repetitions",
        "statistics",
        "comparison",
        "family_aggregation",
        "missing_observation_policy",
    }
)
_METRIC_FIELDS = frozenset(
    {"family", "state", "owner", "implementation", "budget", "evidence"}
)
_IMPLEMENTATION_FIELDS = frozenset(
    {
        "name",
        "version",
        "revision",
        "entrypoint",
        "dependencies",
        "inputs",
        "score_direction",
    }
)
_BUDGET_FIELDS = frozenset(
    {
        "method",
        "unit",
        "absolute_tolerance",
        "relative_tolerance",
        "minimum_case_coverage",
        "aggregation",
        "failure_policy",
        "notes",
    }
)
_INPUTS = frozenset(
    {
        "baseline-audio",
        "baseline-video",
        "candidate-audio",
        "candidate-video",
        "human-ballot",
        "prompt",
        "reference-media",
    }
)
_REQUIRED_INPUTS_BY_FAMILY = {
    "audio-quality": frozenset({"baseline-audio", "candidate-audio"}),
    "av-sync": frozenset(
        {
            "baseline-audio",
            "baseline-video",
            "candidate-audio",
            "candidate-video",
        }
    ),
    "human-pairwise": frozenset(
        {
            "baseline-audio",
            "baseline-video",
            "candidate-audio",
            "candidate-video",
            "human-ballot",
            "prompt",
        }
    ),
    "perceptual-video": frozenset({"baseline-video", "candidate-video"}),
    "prompt-adherence": frozenset({"candidate-video", "prompt"}),
    "temporal-consistency": frozenset({"baseline-video", "candidate-video"}),
}
_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_DEPENDENCY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]*")
_PLACEHOLDER_PATTERN = re.compile(
    r"(?:^|[=:@/_-])(?:head|latest|main|master|stable|tested-version)(?:$|[=:@/_-])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class QualityMetricPlanReport:
    """Quality metric-plan readiness result."""

    plan_id: str
    status: str
    ready: bool
    approved_metrics: int
    planned_metrics: int
    unassigned_metrics: int
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable readiness metadata."""
        return {
            "schema_version": "1.0",
            "plan_id": self.plan_id,
            "status": self.status,
            "ready": self.ready,
            "approved_metrics": self.approved_metrics,
            "planned_metrics": self.planned_metrics,
            "unassigned_metrics": self.unassigned_metrics,
            "blockers": list(self.blockers),
        }


def _load(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        message = f"quality metric plan is missing: {path}"
        raise ValidationError(message) from error
    except (OSError, UnicodeDecodeError) as error:
        message = "quality metric plan could not be read as UTF-8"
        raise ValidationError(message) from error
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        message = f"quality metric plan is not valid JSON: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict):
        message = "quality metric plan root must be an object"
        raise ValidationError(message)
    return value


def _fields(value: dict[str, object], expected: frozenset[str], name: str) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing:
        message = f"{name} is missing required fields: {', '.join(missing)}"
        raise ValidationError(message)
    if unknown:
        message = f"{name} contains unknown fields: {', '.join(unknown)}"
        raise ValidationError(message)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        message = f"{name} must be a non-empty string"
        raise ValidationError(message)
    return value


def _nullable_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _choice(value: object, name: str, choices: frozenset[str]) -> str:
    text = _string(value, name)
    if text not in choices:
        message = f"{name} has unsupported value: {text}"
        raise ValidationError(message)
    return text


def _strings(value: object, name: str, *, minimum: int = 0) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        message = f"{name} must be an array of non-empty strings"
        raise ValidationError(message)
    if len(value) < minimum:
        message = f"{name} must contain at least {minimum} item(s)"
        raise ValidationError(message)
    if len(set(value)) != len(value):
        message = f"{name} must not contain duplicates"
        raise ValidationError(message)
    return tuple(value)


def _https_url(value: object, name: str) -> str:
    text = _string(value, name)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        message = f"{name} must be an HTTPS URL"
        raise ValidationError(message)
    return text


def _validate_evaluation(value: object) -> None:
    if not isinstance(value, dict):
        message = "quality metric evaluation must be an object"
        raise ValidationError(message)
    _fields(value, _EVALUATION_FIELDS, "quality metric evaluation")
    fixed_values: dict[str, object] = {
        "baseline_repetitions": 3,
        "candidate_repetitions": 3,
        "comparison": "baseline-self-envelope-v1",
        "family_aggregation": "independent-all-must-pass",
        "missing_observation_policy": "fail",
    }
    for field, expected in fixed_values.items():
        actual = value[field]
        if isinstance(actual, bool) or actual != expected:
            message = f"quality metric evaluation.{field} must equal {expected!r}"
            raise ValidationError(message)
    statistics = _strings(
        value["statistics"], "quality metric evaluation.statistics", minimum=4
    )
    if statistics != ("p5", "p50", "p95", "worst-case"):
        message = "quality metric evaluation.statistics must use fixed order p5/p50/p95/worst-case"
        raise ValidationError(message)


def _validate_implementation(value: object, family: str) -> None:
    if not isinstance(value, dict):
        message = f"metric {family!r} implementation must be an object"
        raise ValidationError(message)
    _fields(value, _IMPLEMENTATION_FIELDS, f"metric {family!r} implementation")
    _string(value["name"], f"metric {family!r} implementation.name")
    version = _string(value["version"], f"metric {family!r} implementation.version")
    revision = _string(value["revision"], f"metric {family!r} implementation.revision")
    if not _REVISION_PATTERN.fullmatch(revision):
        message = f"metric {family!r} implementation.revision must be an immutable 40- or 64-character digest"
        raise ValidationError(message)
    if not _VERSION_PATTERN.fullmatch(version) or _PLACEHOLDER_PATTERN.search(version):
        message = f"metric {family!r} implementation.version must use an exact non-moving identifier"
        raise ValidationError(message)
    _string(value["entrypoint"], f"metric {family!r} implementation.entrypoint")
    dependencies = _strings(
        value["dependencies"],
        f"metric {family!r} implementation.dependencies",
        minimum=1,
    )
    for dependency in dependencies:
        valid = False
        if "==" in dependency:
            name, version_pin = dependency.split("==", maxsplit=1)
            valid = bool(
                _DEPENDENCY_NAME_PATTERN.fullmatch(name)
                and _VERSION_PATTERN.fullmatch(version_pin)
                and not _PLACEHOLDER_PATTERN.search(version_pin)
            )
        elif "@" in dependency:
            name, revision_pin = dependency.split("@", maxsplit=1)
            valid = bool(
                _DEPENDENCY_NAME_PATTERN.fullmatch(name)
                and _REVISION_PATTERN.fullmatch(revision_pin)
            )
        if not valid:
            message = f"metric {family!r} dependency must use exact name==version or name@40/64-character-revision pin"
            raise ValidationError(message)
    inputs = _strings(
        value["inputs"], f"metric {family!r} implementation.inputs", minimum=1
    )
    unknown_inputs = sorted(set(inputs).difference(_INPUTS))
    if unknown_inputs:
        message = f"metric {family!r} implementation.inputs contains unsupported values: {', '.join(unknown_inputs)}"
        raise ValidationError(message)
    missing_inputs = sorted(_REQUIRED_INPUTS_BY_FAMILY[family].difference(inputs))
    if missing_inputs:
        message = f"metric {family!r} implementation.inputs is missing required values: {', '.join(missing_inputs)}"
        raise ValidationError(message)
    _choice(
        value["score_direction"],
        f"metric {family!r} implementation.score_direction",
        frozenset({"higher-is-better", "lower-is-better", "two-sided"}),
    )


def _zero(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
        message = f"{name} must equal zero for the exact profile"
        raise ValidationError(message)


def _validate_budget(value: object, family: str) -> None:
    if not isinstance(value, dict):
        message = f"metric {family!r} budget must be an object"
        raise ValidationError(message)
    _fields(value, _BUDGET_FIELDS, f"metric {family!r} budget")
    fixed_values = {
        "method": "baseline-self-envelope-v1",
        "aggregation": "per-case-all-runs",
        "failure_policy": "any-family-fails",
    }
    for field, expected in fixed_values.items():
        if value[field] != expected:
            message = f"metric {family!r} budget.{field} must equal {expected!r}"
            raise ValidationError(message)
    _string(value["unit"], f"metric {family!r} budget.unit")
    _zero(
        value["absolute_tolerance"],
        f"metric {family!r} budget.absolute_tolerance",
    )
    _zero(
        value["relative_tolerance"],
        f"metric {family!r} budget.relative_tolerance",
    )
    coverage = value["minimum_case_coverage"]
    if isinstance(coverage, bool) or coverage != 1:
        message = f"metric {family!r} budget.minimum_case_coverage must equal 1"
        raise ValidationError(message)
    _string(value["notes"], f"metric {family!r} budget.notes")


def _validate_metric(value: object, index: int) -> tuple[str, str]:
    if not isinstance(value, dict):
        message = f"quality metric at index {index} must be an object"
        raise ValidationError(message)
    _fields(value, _METRIC_FIELDS, f"quality metric at index {index}")
    family = _choice(
        value["family"],
        f"quality metric at index {index} family",
        REQUIRED_METRIC_FAMILIES,
    )
    state = _choice(
        value["state"],
        f"metric {family!r} state",
        frozenset({"approved", "planned", "unassigned"}),
    )
    owner = _nullable_string(value["owner"], f"metric {family!r} owner")
    implementation = value["implementation"]
    budget = value["budget"]
    evidence = _strings(value["evidence"], f"metric {family!r} evidence")

    if state == "unassigned":
        if (
            owner is not None
            or implementation is not None
            or budget is not None
            or evidence
        ):
            message = f"unassigned metric {family!r} cannot contain disposition data"
            raise ValidationError(message)
    else:
        if owner is None:
            message = f"{state} metric {family!r} requires an owner"
            raise ValidationError(message)
        _validate_implementation(implementation, family)
        if budget is not None:
            _validate_budget(budget, family)
        if state == "planned" and evidence:
            message = f"planned metric {family!r} cannot contain approval evidence"
            raise ValidationError(message)
        if state == "approved":
            if budget is None or not evidence:
                message = f"approved metric {family!r} requires budget and evidence"
                raise ValidationError(message)
            for item in evidence:
                _https_url(item, f"metric {family!r} evidence")
    return family, state


def check_quality_metric_plan(path: Path) -> QualityMetricPlanReport:
    """Validate a metric plan and report every unresolved family."""
    record = _load(path)
    _fields(record, _TOP_LEVEL_FIELDS, "quality metric plan")
    if record["schema_version"] != "1.0":
        message = "unsupported quality metric-plan schema_version; expected '1.0'"
        raise ValidationError(message)
    plan_id = _string(record["plan_id"], "quality metric plan_id")
    if plan_id != PLAN_ID:
        message = f"unsupported quality metric plan_id: {plan_id}"
        raise ValidationError(message)
    status = _choice(
        record["status"], "quality metric status", frozenset({"approved", "draft"})
    )
    updated_at = _string(record["updated_at"], "quality metric updated_at")
    try:
        date.fromisoformat(updated_at)
    except ValueError as error:
        message = "quality metric updated_at must be an ISO 8601 date"
        raise ValidationError(message) from error
    _https_url(record["source_issue"], "quality metric source_issue")
    if record["quality_profile"] != "exact":
        message = "quality metric plan supports only the exact profile"
        raise ValidationError(message)
    _validate_evaluation(record["evaluation"])
    _strings(record["limitations"], "quality metric limitations", minimum=1)

    metrics = record["metrics"]
    if not isinstance(metrics, list):
        message = "quality metrics must be an array"
        raise ValidationError(message)
    states: dict[str, str] = {}
    for index, metric in enumerate(metrics):
        family, state = _validate_metric(metric, index)
        if family in states:
            message = f"duplicate quality metric family: {family}"
            raise ValidationError(message)
        states[family] = state
    if set(states) != REQUIRED_METRIC_FAMILIES:
        missing = sorted(REQUIRED_METRIC_FAMILIES.difference(states))
        unknown = sorted(set(states).difference(REQUIRED_METRIC_FAMILIES))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        message = (
            f"quality metrics do not match required families ({'; '.join(details)})"
        )
        raise ValidationError(message)

    blockers = tuple(
        f"metric:{family}:{state}"
        for family, state in sorted(states.items())
        if state != "approved"
    )
    ready = not blockers
    if status == "approved" and not ready:
        message = "quality metric plan claims approval while blockers remain"
        raise ValidationError(message)
    if status == "draft" and ready:
        message = "quality metric plan is draft even though every family is approved"
        raise ValidationError(message)
    return QualityMetricPlanReport(
        plan_id=plan_id,
        status=status,
        ready=ready,
        approved_metrics=sum(state == "approved" for state in states.values()),
        planned_metrics=sum(state == "planned" for state in states.values()),
        unassigned_metrics=sum(state == "unassigned" for state in states.values()),
        blockers=blockers,
    )
