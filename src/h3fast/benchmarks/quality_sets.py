"""Fail-closed validation for formal H3Fast quality-set metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

SMOKE_MIN_CASES = 10
REGRESSION_MIN_CASES = 50
REQUIRED_APPROVAL_ROLES = frozenset({"quality_owner", "rights_reviewer"})
REQUIRED_TASKS = frozenset({"fl2va", "ref2va", "t2va"})
REQUIRED_DURATIONS = frozenset({4.0, 5.0, 10.0, 15.0})
REQUIRED_ASPECT_RATIOS = frozenset({"landscape", "portrait", "square"})
REQUIRED_LANGUAGES = frozenset({"ja"})
MINIMUM_DISTINCT_LANGUAGES = 2
REQUIRED_SUBJECT_TAGS = frozenset(
    {"face", "hands", "multiple-people", "product", "text"}
)
REQUIRED_MOTION_TAGS = frozenset({"camera-movement", "dynamic", "static"})
REQUIRED_AUDIO_TAGS = frozenset({"dialogue", "environment", "music", "near-silent"})
REQUIRED_REFERENCE_MODALITIES = frozenset({"audio", "image", "mixed", "none", "video"})
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
        "set_id",
        "status",
        "updated_at",
        "source_issue",
        "requirements",
        "selection",
        "approvals",
        "metrics",
        "cases",
        "limitations",
    }
)
_REQUIREMENT_FIELDS = frozenset(
    {
        "smoke_min_cases",
        "regression_min_cases",
        "minimum_distinct_languages",
        "required_tasks",
        "required_durations_seconds",
        "required_aspect_ratios",
        "required_languages",
        "required_subject_tags",
        "required_motion_tags",
        "required_audio_tags",
        "required_reference_modalities",
        "required_metric_families",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "method",
        "registry_uri",
        "registry_sha256",
        "exclusions_reviewed",
        "exclusions",
        "known_failures_reviewed",
        "known_failures",
    }
)
_APPROVAL_FIELDS = frozenset({"state", "owner", "deadline", "approved_at", "evidence"})
_METRIC_FIELDS = frozenset(
    {"family", "state", "owner", "implementation", "version", "budget", "evidence"}
)
_CASE_FIELDS = frozenset(
    {
        "id",
        "split",
        "prompt_sha256",
        "seed",
        "task",
        "duration_seconds",
        "aspect_ratio",
        "languages",
        "subject_tags",
        "motion_tags",
        "audio_tags",
        "reference_modalities",
        "reference_asset_sha256s",
        "rights_status",
        "rights_evidence",
        "notes",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CASE_ID_PATTERN = re.compile(r"(?:smoke|regression)-[0-9]{3}")
_LANGUAGE_PATTERN = re.compile(r"[a-z]{2,3}(?:-[A-Z][a-z]{3})?(?:-[A-Z]{2})?")


@dataclass(frozen=True, slots=True)
class FormalQualitySetReport:
    """Formal quality-set readiness result."""

    set_id: str
    status: str
    ready: bool
    smoke_cases: int
    regression_cases: int
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable readiness data."""
        return {
            "schema_version": "1.0",
            "set_id": self.set_id,
            "status": self.status,
            "ready": self.ready,
            "smoke_cases": self.smoke_cases,
            "regression_cases": self.regression_cases,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class _CaseCoverage:
    split: str
    task: str
    duration: float
    aspect_ratio: str
    languages: tuple[str, ...]
    subject_tags: tuple[str, ...]
    motion_tags: tuple[str, ...]
    audio_tags: tuple[str, ...]
    reference_modalities: tuple[str, ...]


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        message = f"formal quality-set record is missing: {path}"
        raise ValidationError(message) from error
    except json.JSONDecodeError as error:
        message = f"formal quality-set record is not valid JSON: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict):
        message = "formal quality-set record root must be an object"
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


def _date(value: object, name: str) -> str:
    text = _string(value, name)
    try:
        date.fromisoformat(text)
    except ValueError as error:
        message = f"{name} must be an ISO 8601 date"
        raise ValidationError(message) from error
    return text


def _date_time(value: object, name: str) -> str | None:
    text = _nullable_string(value, name)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        message = f"{name} must be an ISO 8601 date-time"
        raise ValidationError(message) from error
    if parsed.tzinfo is None:
        message = f"{name} must include a UTC offset"
        raise ValidationError(message)
    return text


def _https_url(value: object, name: str) -> str:
    text = _string(value, name)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        message = f"{name} must be an HTTPS URL"
        raise ValidationError(message)
    return text


def _sha256(value: object, name: str, *, nullable: bool) -> str | None:
    if value is None and nullable:
        return None
    text = _string(value, name)
    if not _SHA256_PATTERN.fullmatch(text):
        message = f"{name} must be a lowercase SHA-256"
        raise ValidationError(message)
    return text


def _string_array(value: object, name: str, *, minimum: int = 0) -> tuple[str, ...]:
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


def _choice(value: object, name: str, choices: frozenset[str]) -> str:
    text = _string(value, name)
    if text not in choices:
        message = f"{name} has unsupported value: {text}"
        raise ValidationError(message)
    return text


def _exact_requirement_set(
    requirements: dict[str, object], field: str, expected: frozenset[object]
) -> None:
    value = requirements[field]
    if not isinstance(value, list) or any(isinstance(item, bool) for item in value):
        message = f"requirements.{field} must be an array"
        raise ValidationError(message)
    try:
        actual = frozenset(value)
    except TypeError as error:
        message = f"requirements.{field} contains an unsupported value"
        raise ValidationError(message) from error
    if len(actual) != len(value) or actual != expected:
        message = f"requirements.{field} does not match the formal quality contract"
        raise ValidationError(message)


def _validate_requirements(value: object) -> None:
    if not isinstance(value, dict):
        message = "requirements must be an object"
        raise ValidationError(message)
    _fields(value, _REQUIREMENT_FIELDS, "requirements")
    exact_scalars = {
        "smoke_min_cases": SMOKE_MIN_CASES,
        "regression_min_cases": REGRESSION_MIN_CASES,
        "minimum_distinct_languages": MINIMUM_DISTINCT_LANGUAGES,
    }
    for field, expected in exact_scalars.items():
        actual = value[field]
        if isinstance(actual, bool) or actual != expected:
            message = f"requirements.{field} must equal {expected}"
            raise ValidationError(message)
    _exact_requirement_set(value, "required_tasks", frozenset(REQUIRED_TASKS))
    _exact_requirement_set(
        value, "required_durations_seconds", frozenset(REQUIRED_DURATIONS)
    )
    _exact_requirement_set(
        value, "required_aspect_ratios", frozenset(REQUIRED_ASPECT_RATIOS)
    )
    _exact_requirement_set(value, "required_languages", frozenset(REQUIRED_LANGUAGES))
    _exact_requirement_set(
        value, "required_subject_tags", frozenset(REQUIRED_SUBJECT_TAGS)
    )
    _exact_requirement_set(
        value, "required_motion_tags", frozenset(REQUIRED_MOTION_TAGS)
    )
    _exact_requirement_set(value, "required_audio_tags", frozenset(REQUIRED_AUDIO_TAGS))
    _exact_requirement_set(
        value,
        "required_reference_modalities",
        frozenset(REQUIRED_REFERENCE_MODALITIES),
    )
    _exact_requirement_set(
        value, "required_metric_families", frozenset(REQUIRED_METRIC_FAMILIES)
    )


def _validate_selection(value: object) -> tuple[str, ...]:
    if not isinstance(value, dict):
        message = "selection must be an object"
        raise ValidationError(message)
    _fields(value, _SELECTION_FIELDS, "selection")
    method = _nullable_string(value["method"], "selection.method")
    registry_uri = _nullable_string(value["registry_uri"], "selection.registry_uri")
    if registry_uri is not None:
        _https_url(registry_uri, "selection.registry_uri")
    registry_sha = _sha256(
        value["registry_sha256"], "selection.registry_sha256", nullable=True
    )
    exclusions_reviewed = value["exclusions_reviewed"]
    failures_reviewed = value["known_failures_reviewed"]
    if not isinstance(exclusions_reviewed, bool):
        message = "selection.exclusions_reviewed must be boolean"
        raise ValidationError(message)
    if not isinstance(failures_reviewed, bool):
        message = "selection.known_failures_reviewed must be boolean"
        raise ValidationError(message)
    _string_array(value["exclusions"], "selection.exclusions")
    _string_array(value["known_failures"], "selection.known_failures")

    blockers: list[str] = []
    if method is None:
        blockers.append("selection:method-unassigned")
    if registry_uri is None:
        blockers.append("selection:registry-uri-unassigned")
    if registry_sha is None:
        blockers.append("selection:registry-digest-unassigned")
    if not exclusions_reviewed:
        blockers.append("selection:exclusions-unreviewed")
    if not failures_reviewed:
        blockers.append("selection:known-failures-unreviewed")
    return tuple(blockers)


def _validate_approval(role: str, value: object) -> str:
    if not isinstance(value, dict):
        message = f"approval {role!r} must be an object"
        raise ValidationError(message)
    _fields(value, _APPROVAL_FIELDS, f"approval {role!r}")
    state = _choice(
        value["state"],
        f"approval {role!r} state",
        frozenset({"approved", "pending", "unassigned"}),
    )
    owner = _nullable_string(value["owner"], f"approval {role!r} owner")
    _date(value["deadline"], f"approval {role!r} deadline")
    approved_at = _date_time(value["approved_at"], f"approval {role!r} approved_at")
    evidence = _string_array(value["evidence"], f"approval {role!r} evidence")
    if state == "unassigned" and (
        owner is not None or approved_at is not None or evidence
    ):
        message = f"unassigned approval {role!r} cannot contain disposition data"
        raise ValidationError(message)
    if state == "pending" and (owner is None or approved_at is not None or evidence):
        message = f"pending approval {role!r} requires owner without approval evidence"
        raise ValidationError(message)
    if state == "approved" and (owner is None or approved_at is None or not evidence):
        message = f"approved approval {role!r} requires owner, time, and evidence"
        raise ValidationError(message)
    return state


def _validate_approvals(value: object) -> tuple[str, ...]:
    if not isinstance(value, dict):
        message = "approvals must be an object"
        raise ValidationError(message)
    roles = set(value)
    if roles != REQUIRED_APPROVAL_ROLES:
        missing = sorted(REQUIRED_APPROVAL_ROLES.difference(roles))
        unknown = sorted(roles.difference(REQUIRED_APPROVAL_ROLES))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        message = (
            f"quality approvals do not match required roles ({'; '.join(details)})"
        )
        raise ValidationError(message)
    blockers = []
    for role in sorted(REQUIRED_APPROVAL_ROLES):
        state = _validate_approval(role, value[role])
        if state != "approved":
            blockers.append(f"approval:{role}:{state}")
    return tuple(blockers)


def _validate_metric(value: object, index: int) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        message = f"metric at index {index} must be an object"
        raise ValidationError(message)
    _fields(value, _METRIC_FIELDS, f"metric at index {index}")
    family = _choice(
        value["family"], f"metric at index {index} family", REQUIRED_METRIC_FAMILIES
    )
    state = _choice(
        value["state"],
        f"metric {family!r} state",
        frozenset({"approved", "planned", "unassigned"}),
    )
    owner = _nullable_string(value["owner"], f"metric {family!r} owner")
    implementation = _nullable_string(
        value["implementation"], f"metric {family!r} implementation"
    )
    version = _nullable_string(value["version"], f"metric {family!r} version")
    budget = _nullable_string(value["budget"], f"metric {family!r} budget")
    evidence = _string_array(value["evidence"], f"metric {family!r} evidence")
    details = (owner, implementation, version, budget)
    if state == "unassigned" and (
        any(item is not None for item in details) or evidence
    ):
        message = f"unassigned metric {family!r} cannot contain implementation data"
        raise ValidationError(message)
    if state == "planned" and owner is None:
        message = f"planned metric {family!r} requires an owner"
        raise ValidationError(message)
    if state == "approved" and (any(item is None for item in details) or not evidence):
        message = f"approved metric {family!r} requires owner, versioned method, budget, and evidence"
        raise ValidationError(message)
    blockers = () if state == "approved" else (f"metric:{family}:{state}",)
    return family, blockers


def _validate_metrics(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        message = "metrics must be an array"
        raise ValidationError(message)
    seen: set[str] = set()
    blockers: list[str] = []
    for index, metric in enumerate(value):
        family, metric_blockers = _validate_metric(metric, index)
        if family in seen:
            message = f"duplicate metric family: {family}"
            raise ValidationError(message)
        seen.add(family)
        blockers.extend(metric_blockers)
    if seen != REQUIRED_METRIC_FAMILIES:
        missing = sorted(REQUIRED_METRIC_FAMILIES.difference(seen))
        unknown = sorted(seen.difference(REQUIRED_METRIC_FAMILIES))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        message = f"metrics do not match required families ({'; '.join(details)})"
        raise ValidationError(message)
    return tuple(blockers)


def _case_tags(
    value: dict[str, object], field: str, allowed: frozenset[str], case_id: str
) -> tuple[str, ...]:
    tags = _string_array(value[field], f"case {case_id!r} {field}", minimum=1)
    unknown = sorted(set(tags).difference(allowed))
    if unknown:
        message = f"case {case_id!r} {field} contains unsupported values: {', '.join(unknown)}"
        raise ValidationError(message)
    return tags


def _validate_case(
    value: object, index: int
) -> tuple[str, _CaseCoverage, tuple[str, ...]]:
    if not isinstance(value, dict):
        message = f"quality case at index {index} must be an object"
        raise ValidationError(message)
    _fields(value, _CASE_FIELDS, f"quality case at index {index}")
    case_id = _string(value["id"], f"quality case at index {index} id")
    if not _CASE_ID_PATTERN.fullmatch(case_id):
        message = f"quality case id has unsupported format: {case_id}"
        raise ValidationError(message)
    split = _choice(
        value["split"], f"case {case_id!r} split", frozenset({"regression", "smoke"})
    )
    if not case_id.startswith(f"{split}-"):
        message = f"quality case {case_id!r} id does not match split {split!r}"
        raise ValidationError(message)
    prompt_sha = _sha256(
        value["prompt_sha256"], f"case {case_id!r} prompt_sha256", nullable=True
    )
    seed = value["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        message = f"case {case_id!r} seed must be a non-negative integer"
        raise ValidationError(message)
    task = _choice(value["task"], f"case {case_id!r} task", REQUIRED_TASKS)
    duration = value["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        message = f"case {case_id!r} duration_seconds must be numeric"
        raise ValidationError(message)
    duration_value = float(duration)
    if duration_value not in REQUIRED_DURATIONS:
        message = f"case {case_id!r} has unsupported duration_seconds: {duration}"
        raise ValidationError(message)
    aspect_ratio = _choice(
        value["aspect_ratio"], f"case {case_id!r} aspect_ratio", REQUIRED_ASPECT_RATIOS
    )
    languages = _string_array(
        value["languages"], f"case {case_id!r} languages", minimum=1
    )
    if any(not _LANGUAGE_PATTERN.fullmatch(language) for language in languages):
        message = f"case {case_id!r} languages must use supported BCP 47 language tags"
        raise ValidationError(message)
    subject_tags = _case_tags(value, "subject_tags", REQUIRED_SUBJECT_TAGS, case_id)
    motion_tags = _case_tags(value, "motion_tags", REQUIRED_MOTION_TAGS, case_id)
    audio_tags = _case_tags(value, "audio_tags", REQUIRED_AUDIO_TAGS, case_id)
    modalities = _case_tags(
        value, "reference_modalities", REQUIRED_REFERENCE_MODALITIES, case_id
    )
    asset_hashes = _string_array(
        value["reference_asset_sha256s"], f"case {case_id!r} reference_asset_sha256s"
    )
    for digest in asset_hashes:
        _sha256(digest, f"case {case_id!r} reference asset digest", nullable=False)
    if task == "t2va" and (modalities != ("none",) or asset_hashes):
        message = f"T2VA case {case_id!r} must use only the none reference modality"
        raise ValidationError(message)
    if task != "t2va" and ("none" in modalities or not asset_hashes):
        message = (
            f"referenced case {case_id!r} requires asset digests without none modality"
        )
        raise ValidationError(message)
    rights_status = _choice(
        value["rights_status"],
        f"case {case_id!r} rights_status",
        frozenset({"approved", "unreviewed"}),
    )
    rights_evidence = _string_array(
        value["rights_evidence"], f"case {case_id!r} rights_evidence"
    )
    if rights_status == "unreviewed" and rights_evidence:
        message = f"unreviewed case {case_id!r} cannot contain rights evidence"
        raise ValidationError(message)
    if rights_status == "approved" and not rights_evidence:
        message = f"approved case {case_id!r} requires rights evidence"
        raise ValidationError(message)
    _string(value["notes"], f"case {case_id!r} notes")

    blockers: list[str] = []
    if prompt_sha is None:
        blockers.append(f"case:{case_id}:prompt-digest-missing")
    if rights_status != "approved":
        blockers.append(f"case:{case_id}:rights-unreviewed")
    coverage = _CaseCoverage(
        split=split,
        task=task,
        duration=duration_value,
        aspect_ratio=aspect_ratio,
        languages=languages,
        subject_tags=subject_tags,
        motion_tags=motion_tags,
        audio_tags=audio_tags,
        reference_modalities=modalities,
    )
    return case_id, coverage, tuple(blockers)


def _coverage_blockers(cases: list[_CaseCoverage]) -> tuple[str, ...]:
    smoke_count = sum(case.split == "smoke" for case in cases)
    regression_count = sum(case.split == "regression" for case in cases)
    blockers: list[str] = []
    if smoke_count < SMOKE_MIN_CASES:
        blockers.append(f"cases:smoke-count:{smoke_count}/{SMOKE_MIN_CASES}")
    if regression_count < REGRESSION_MIN_CASES:
        blockers.append(
            f"cases:regression-count:{regression_count}/{REGRESSION_MIN_CASES}"
        )
    observed: dict[str, set[object]] = {
        "tasks": {case.task for case in cases},
        "durations": {case.duration for case in cases},
        "aspect-ratios": {case.aspect_ratio for case in cases},
        "languages": {item for case in cases for item in case.languages},
        "subject-tags": {item for case in cases for item in case.subject_tags},
        "motion-tags": {item for case in cases for item in case.motion_tags},
        "audio-tags": {item for case in cases for item in case.audio_tags},
        "reference-modalities": {
            item for case in cases for item in case.reference_modalities
        },
    }
    required: dict[str, frozenset[object]] = {
        "tasks": frozenset(REQUIRED_TASKS),
        "durations": frozenset(REQUIRED_DURATIONS),
        "aspect-ratios": frozenset(REQUIRED_ASPECT_RATIOS),
        "languages": frozenset(REQUIRED_LANGUAGES),
        "subject-tags": frozenset(REQUIRED_SUBJECT_TAGS),
        "motion-tags": frozenset(REQUIRED_MOTION_TAGS),
        "audio-tags": frozenset(REQUIRED_AUDIO_TAGS),
        "reference-modalities": frozenset(REQUIRED_REFERENCE_MODALITIES),
    }
    for dimension, expected in required.items():
        missing = sorted(
            str(value) for value in expected.difference(observed[dimension])
        )
        if missing:
            blockers.append(f"coverage:{dimension}:missing:{','.join(missing)}")
    language_count = len(observed["languages"])
    if language_count < MINIMUM_DISTINCT_LANGUAGES:
        blockers.append(
            f"coverage:languages:distinct:{language_count}/{MINIMUM_DISTINCT_LANGUAGES}"
        )
    return tuple(blockers)


def check_formal_quality_set(path: Path) -> FormalQualitySetReport:
    """Validate formal set metadata and report all unresolved release blockers."""
    record = _load(path)
    _fields(record, _TOP_LEVEL_FIELDS, "formal quality-set record")
    if record["schema_version"] != "1.0":
        message = "unsupported formal quality-set schema_version; expected '1.0'"
        raise ValidationError(message)
    set_id = _string(record["set_id"], "set_id")
    if set_id != "h3fast-phase0-formal-quality-v1":
        message = f"unsupported formal quality set_id: {set_id}"
        raise ValidationError(message)
    status = _choice(record["status"], "status", frozenset({"approved", "incomplete"}))
    _date(record["updated_at"], "updated_at")
    _https_url(record["source_issue"], "source_issue")
    _validate_requirements(record["requirements"])

    blockers = list(_validate_selection(record["selection"]))
    blockers.extend(_validate_approvals(record["approvals"]))
    blockers.extend(_validate_metrics(record["metrics"]))
    limitations = _string_array(record["limitations"], "limitations", minimum=1)
    if not limitations:
        message = "limitations must not be empty"
        raise ValidationError(message)

    raw_cases = record["cases"]
    if not isinstance(raw_cases, list):
        message = "cases must be an array"
        raise ValidationError(message)
    seen: set[str] = set()
    cases: list[_CaseCoverage] = []
    for index, case in enumerate(raw_cases):
        case_id, coverage, case_blockers = _validate_case(case, index)
        if case_id in seen:
            message = f"duplicate quality case id: {case_id}"
            raise ValidationError(message)
        seen.add(case_id)
        cases.append(coverage)
        blockers.extend(case_blockers)
    blockers.extend(_coverage_blockers(cases))

    smoke_count = sum(case.split == "smoke" for case in cases)
    regression_count = sum(case.split == "regression" for case in cases)
    ready = not blockers
    if status == "approved" and not ready:
        message = "formal quality-set record claims approval while blockers remain"
        raise ValidationError(message)
    if status == "incomplete" and ready:
        message = (
            "formal quality-set record is incomplete even though every gate passed"
        )
        raise ValidationError(message)
    return FormalQualitySetReport(
        set_id=set_id,
        status=status,
        ready=ready,
        smoke_cases=smoke_count,
        regression_cases=regression_count,
        blockers=tuple(blockers),
    )
