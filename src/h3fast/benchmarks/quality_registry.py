"""Compile private quality registries into redacted formal-set metadata."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlparse

from h3fast.benchmarks.quality_sets import check_formal_quality_set
from h3fast.exceptions import ValidationError

_REGISTRY_ID = "h3fast-phase0-formal-quality-v1"
_REGISTRY_FIELDS = frozenset(
    {"schema_version", "registry_id", "updated_at", "selection", "cases"}
)
_SELECTION_FIELDS = frozenset(
    {
        "method",
        "exclusions_reviewed",
        "public_exclusions",
        "known_failures_reviewed",
        "public_known_failures",
    }
)
_CASE_FIELDS = frozenset(
    {
        "id",
        "split",
        "prompt",
        "seed",
        "task",
        "duration_seconds",
        "aspect_ratio",
        "languages",
        "subject_tags",
        "motion_tags",
        "audio_tags",
        "references",
        "rights_status",
        "rights_evidence",
        "public_notes",
    }
)
_REFERENCE_FIELDS = frozenset({"path", "modality"})
_CASE_ID_PATTERN = re.compile(r"(?:smoke|regression)-[0-9]{3}")
_LANGUAGE_PATTERN = re.compile(r"[a-z]{2,3}(?:-[A-Z][a-z]{3})?(?:-[A-Z]{2})?")
_TASKS = frozenset({"fl2va", "ref2va", "t2va"})
_DURATIONS = frozenset({4.0, 5.0, 10.0, 15.0})
_ASPECT_RATIOS = frozenset({"landscape", "portrait", "square"})
_SUBJECT_TAGS = frozenset({"face", "hands", "multiple-people", "product", "text"})
_MOTION_TAGS = frozenset({"camera-movement", "dynamic", "static"})
_AUDIO_TAGS = frozenset({"dialogue", "environment", "music", "near-silent"})
_REFERENCE_MODALITIES = frozenset({"audio", "image", "video"})
_REVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "registry_id",
        "source_registry_sha256",
        "registry_content_sha256",
        "prepared_at",
        "reviewer",
        "reviewed_at",
        "selection",
        "cases",
    }
)
_REVIEW_SELECTION_FIELDS = frozenset(
    {
        "method_decision",
        "exclusions_decision",
        "known_failures_decision",
        "notes",
    }
)
_REVIEW_CASE_FIELDS = frozenset(
    {
        "id",
        "prompt_sha256",
        "reference_asset_sha256s",
        "rights_decision",
        "selection_decision",
        "rights_evidence",
        "notes",
    }
)
_REVIEW_DECISIONS = frozenset({"approved", "pending", "rejected"})


@dataclass(frozen=True, slots=True)
class QualityRegistryCompileReport:
    """Redacted result of a private quality-registry compilation."""

    registry_id: str
    registry_sha256: str
    smoke_cases: int
    regression_cases: int
    ready: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe compilation metadata without private inputs."""
        return {
            "schema_version": "1.0",
            "registry_id": self.registry_id,
            "registry_sha256": self.registry_sha256,
            "smoke_cases": self.smoke_cases,
            "regression_cases": self.regression_cases,
            "ready": self.ready,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class QualityRegistryReviewReport:
    """Redacted status for a local registry-review operation."""

    registry_id: str
    source_registry_sha256: str
    registry_content_sha256: str
    total_cases: int
    approved_cases: int
    pending_cases: int
    rejected_cases: int
    ready: bool
    output_registry_sha256: str | None
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return review status without prompt, asset, path, or per-case metadata."""
        return {
            "schema_version": "1.0",
            "registry_id": self.registry_id,
            "source_registry_sha256": self.source_registry_sha256,
            "registry_content_sha256": self.registry_content_sha256,
            "total_cases": self.total_cases,
            "approved_cases": self.approved_cases,
            "pending_cases": self.pending_cases,
            "rejected_cases": self.rejected_cases,
            "ready": self.ready,
            "output_registry_sha256": self.output_registry_sha256,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class _ParsedPrivateRegistry:
    record: dict[str, object]
    raw: bytes
    registry_id: str
    updated_at: str
    method: str
    exclusions_reviewed: bool
    exclusions: tuple[str, ...]
    known_failures_reviewed: bool
    known_failures: tuple[str, ...]
    cases: tuple[dict[str, object], ...]


def _load_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        message = f"{name} is missing"
        raise ValidationError(message) from error
    except OSError as error:
        message = f"{name} could not be read"
        raise ValidationError(message) from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"{name} is not valid UTF-8 JSON: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict):
        message = f"{name} root must be an object"
        raise ValidationError(message)
    return value, raw


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


def _choice(value: object, name: str, allowed: frozenset[str]) -> str:
    text = _string(value, name)
    if text not in allowed:
        message = f"{name} has unsupported value: {text}"
        raise ValidationError(message)
    return text


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        message = f"{name} must be boolean"
        raise ValidationError(message)
    return value


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


def _public_strings(value: object, name: str) -> tuple[str, ...]:
    values = _strings(value, name)
    for item in values:
        if Path(item).is_absolute() or item.startswith(("./", "../", "file:")):
            message = f"{name} must not expose local paths"
            raise ValidationError(message)
    return values


def _https_url(value: object, name: str) -> str:
    text = _string(value, name)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        message = f"{name} must be an HTTPS URL"
        raise ValidationError(message)
    return text


def _tags(value: object, name: str, allowed: frozenset[str]) -> tuple[str, ...]:
    tags = _strings(value, name, minimum=1)
    unknown = sorted(set(tags).difference(allowed))
    if unknown:
        message = f"{name} contains unsupported values: {', '.join(unknown)}"
        raise ValidationError(message)
    return tags


def _sha256_file(path: Path, *, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        message = f"{label} is missing"
        raise ValidationError(message) from error
    except IsADirectoryError as error:
        message = f"{label} must be a file"
        raise ValidationError(message) from error
    except OSError as error:
        message = f"{label} could not be read"
        raise ValidationError(message) from error
    return digest.hexdigest()


def _compile_references(
    value: object, *, registry_root: Path, case_id: str, task: str
) -> tuple[list[str], list[str]]:
    if not isinstance(value, list):
        message = f"private case {case_id!r} references must be an array"
        raise ValidationError(message)
    if task == "t2va":
        if value:
            message = f"private T2VA case {case_id!r} must not contain references"
            raise ValidationError(message)
        return ["none"], []
    if not value:
        message = f"private referenced case {case_id!r} requires at least one asset"
        raise ValidationError(message)
    paths: set[str] = set()
    modalities: set[str] = set()
    digests: list[str] = []
    for index, reference in enumerate(value):
        if not isinstance(reference, dict):
            message = (
                f"private case {case_id!r} reference at index {index} must be an object"
            )
            raise ValidationError(message)
        _fields(reference, _REFERENCE_FIELDS, f"private case {case_id!r} reference")
        path_text = _string(reference["path"], f"private case {case_id!r} path")
        if path_text in paths:
            message = f"private case {case_id!r} contains duplicate reference paths"
            raise ValidationError(message)
        paths.add(path_text)
        modality = _choice(
            reference["modality"],
            f"private case {case_id!r} reference modality",
            _REFERENCE_MODALITIES,
        )
        modalities.add(modality)
        path = Path(path_text)
        if not path.is_absolute():
            path = registry_root / path
        digests.append(
            _sha256_file(
                path,
                label=f"private case {case_id!r} reference at index {index}",
            )
        )

    public_modalities = sorted(modalities)
    if len(modalities) > 1:
        public_modalities.append("mixed")
    return public_modalities, digests


def _compile_case(
    value: object, *, registry_root: Path, index: int
) -> dict[str, object]:
    if not isinstance(value, dict):
        message = f"private quality case at index {index} must be an object"
        raise ValidationError(message)
    _fields(value, _CASE_FIELDS, f"private quality case at index {index}")
    case_id = _string(value["id"], f"private quality case at index {index} id")
    if not _CASE_ID_PATTERN.fullmatch(case_id):
        message = f"private quality case id has unsupported format: {case_id}"
        raise ValidationError(message)
    split = _choice(
        value["split"],
        f"private case {case_id!r} split",
        frozenset({"regression", "smoke"}),
    )
    if not case_id.startswith(f"{split}-"):
        message = f"private quality case {case_id!r} id does not match split"
        raise ValidationError(message)
    prompt = _string(value["prompt"], f"private case {case_id!r} prompt")
    seed = value["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        message = f"private case {case_id!r} seed must be a non-negative integer"
        raise ValidationError(message)
    task = _choice(value["task"], f"private case {case_id!r} task", _TASKS)
    duration = value["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        message = f"private case {case_id!r} duration_seconds must be numeric"
        raise ValidationError(message)
    duration_value = float(duration)
    if duration_value not in _DURATIONS:
        message = f"private case {case_id!r} has unsupported duration_seconds"
        raise ValidationError(message)
    aspect_ratio = _choice(
        value["aspect_ratio"],
        f"private case {case_id!r} aspect_ratio",
        _ASPECT_RATIOS,
    )
    languages = _strings(
        value["languages"], f"private case {case_id!r} languages", minimum=1
    )
    if any(not _LANGUAGE_PATTERN.fullmatch(language) for language in languages):
        message = f"private case {case_id!r} languages must use BCP 47 tags"
        raise ValidationError(message)
    subject_tags = _tags(
        value["subject_tags"], f"private case {case_id!r} subject_tags", _SUBJECT_TAGS
    )
    motion_tags = _tags(
        value["motion_tags"], f"private case {case_id!r} motion_tags", _MOTION_TAGS
    )
    audio_tags = _tags(
        value["audio_tags"], f"private case {case_id!r} audio_tags", _AUDIO_TAGS
    )
    modalities, asset_digests = _compile_references(
        value["references"], registry_root=registry_root, case_id=case_id, task=task
    )
    rights_status = _choice(
        value["rights_status"],
        f"private case {case_id!r} rights_status",
        frozenset({"approved", "unreviewed"}),
    )
    rights_evidence = _strings(
        value["rights_evidence"], f"private case {case_id!r} rights_evidence"
    )
    if rights_status == "approved" and not rights_evidence:
        message = f"approved private case {case_id!r} requires rights evidence"
        raise ValidationError(message)
    if rights_status == "unreviewed" and rights_evidence:
        message = f"unreviewed private case {case_id!r} cannot contain rights evidence"
        raise ValidationError(message)
    for evidence in rights_evidence:
        _https_url(evidence, f"private case {case_id!r} rights evidence")
    public_notes = _string(
        value["public_notes"], f"private case {case_id!r} public_notes"
    )
    _public_strings([public_notes], f"private case {case_id!r} public_notes")

    return {
        "id": case_id,
        "split": split,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "seed": seed,
        "task": task,
        "duration_seconds": duration,
        "aspect_ratio": aspect_ratio,
        "languages": list(languages),
        "subject_tags": list(subject_tags),
        "motion_tags": list(motion_tags),
        "audio_tags": list(audio_tags),
        "reference_modalities": modalities,
        "reference_asset_sha256s": asset_digests,
        "rights_status": rights_status,
        "rights_evidence": list(rights_evidence),
        "notes": public_notes,
    }


def _parse_private_registry(registry_path: Path) -> _ParsedPrivateRegistry:
    registry, registry_bytes = _load_json(registry_path, "private quality registry")
    _fields(registry, _REGISTRY_FIELDS, "private quality registry")
    if registry["schema_version"] != "1.0":
        message = "unsupported private quality registry schema_version; expected '1.0'"
        raise ValidationError(message)
    registry_id = _string(registry["registry_id"], "private registry_id")
    if registry_id != _REGISTRY_ID:
        message = f"unsupported private quality registry_id: {registry_id}"
        raise ValidationError(message)
    updated_at = _string(registry["updated_at"], "private registry updated_at")
    try:
        date.fromisoformat(updated_at)
    except ValueError as error:
        message = "private registry updated_at must be an ISO 8601 date"
        raise ValidationError(message) from error

    selection = registry["selection"]
    if not isinstance(selection, dict):
        message = "private registry selection must be an object"
        raise ValidationError(message)
    _fields(selection, _SELECTION_FIELDS, "private registry selection")
    method = _string(selection["method"], "private registry selection.method")
    exclusions_reviewed = _boolean(
        selection["exclusions_reviewed"], "private registry exclusions_reviewed"
    )
    known_failures_reviewed = _boolean(
        selection["known_failures_reviewed"],
        "private registry known_failures_reviewed",
    )
    exclusions = _public_strings(
        selection["public_exclusions"], "private registry public_exclusions"
    )
    known_failures = _public_strings(
        selection["public_known_failures"],
        "private registry public_known_failures",
    )

    raw_cases = registry["cases"]
    if not isinstance(raw_cases, list):
        message = "private registry cases must be an array"
        raise ValidationError(message)
    cases = tuple(
        _compile_case(case, registry_root=registry_path.parent, index=index)
        for index, case in enumerate(raw_cases)
    )
    case_ids = [case["id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        message = "private quality registry contains duplicate case ids"
        raise ValidationError(message)
    return _ParsedPrivateRegistry(
        record=registry,
        raw=registry_bytes,
        registry_id=registry_id,
        updated_at=updated_at,
        method=method,
        exclusions_reviewed=exclusions_reviewed,
        exclusions=exclusions,
        known_failures_reviewed=known_failures_reviewed,
        known_failures=known_failures,
        cases=cases,
    )


def _registry_sha256(parsed: _ParsedPrivateRegistry) -> str:
    return hashlib.sha256(parsed.raw).hexdigest()


def _registry_content_sha256(parsed: _ParsedPrivateRegistry) -> str:
    cases = [
        {
            key: value
            for key, value in case.items()
            if key not in {"rights_status", "rights_evidence"}
        }
        for case in parsed.cases
    ]
    content = {
        "registry_id": parsed.registry_id,
        "selection": {
            "method": parsed.method,
            "public_exclusions": list(parsed.exclusions),
            "public_known_failures": list(parsed.known_failures),
        },
        "cases": cases,
    }
    canonical = json.dumps(
        content, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_atomic_new(output_path: Path, value: dict[str, object], name: str) -> None:
    if output_path.exists():
        message = f"{name} output already exists"
        raise ValidationError(message)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        message = f"{name} partial output already exists"
        raise ValidationError(message)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _reset_set_approvals(template: dict[str, object]) -> None:
    approvals = template["approvals"]
    if not isinstance(approvals, dict):  # pragma: no cover - template is prevalidated
        message = "formal quality-set template approvals must be an object"
        raise ValidationError(message)
    reset: dict[str, object] = {}
    for role in ("quality_owner", "rights_reviewer"):
        approval = approvals[role]
        if not isinstance(approval, dict):  # pragma: no cover - prevalidated
            message = f"formal quality-set template approval {role!r} must be an object"
            raise ValidationError(message)
        reset[role] = {
            "state": "unassigned",
            "owner": None,
            "deadline": approval["deadline"],
            "approved_at": None,
            "evidence": [],
        }
    template["approvals"] = reset


def compile_quality_registry(
    registry_path: Path,
    template_path: Path,
    output_path: Path,
    *,
    registry_uri: str,
) -> QualityRegistryCompileReport:
    """Compile private prompt/asset data into an atomic redacted formal record."""
    if output_path.resolve() in {registry_path.resolve(), template_path.resolve()}:
        message = "quality-set output must differ from registry and template inputs"
        raise ValidationError(message)
    uri = _https_url(registry_uri, "registry_uri")
    parsed = _parse_private_registry(registry_path)

    template, _ = _load_json(template_path, "formal quality-set template")
    check_formal_quality_set(template_path)
    template["status"] = "incomplete"
    template["updated_at"] = parsed.updated_at
    template["selection"] = {
        "method": parsed.method,
        "registry_uri": uri,
        "registry_sha256": _registry_sha256(parsed),
        "exclusions_reviewed": parsed.exclusions_reviewed,
        "exclusions": list(parsed.exclusions),
        "known_failures_reviewed": parsed.known_failures_reviewed,
        "known_failures": list(parsed.known_failures),
    }
    template["cases"] = list(parsed.cases)
    _reset_set_approvals(template)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        formal_report = check_formal_quality_set(temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return QualityRegistryCompileReport(
        registry_id=parsed.registry_id,
        registry_sha256=_registry_sha256(parsed),
        smoke_cases=formal_report.smoke_cases,
        regression_cases=formal_report.regression_cases,
        ready=formal_report.ready,
        blockers=formal_report.blockers,
    )


def prepare_quality_registry_review(
    registry_path: Path,
    output_path: Path,
    *,
    reviewer: str,
) -> QualityRegistryReviewReport:
    """Create a local-only review checklist bound to exact registry content."""
    if output_path.resolve() == registry_path.resolve():
        message = "quality-review output must differ from registry input"
        raise ValidationError(message)
    reviewer_name = _string(reviewer, "quality-review reviewer")
    parsed = _parse_private_registry(registry_path)
    source_sha256 = _registry_sha256(parsed)
    content_sha256 = _registry_content_sha256(parsed)
    prepared_at = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    review_cases = [
        {
            "id": case["id"],
            "prompt_sha256": case["prompt_sha256"],
            "reference_asset_sha256s": case["reference_asset_sha256s"],
            "rights_decision": "pending",
            "selection_decision": "pending",
            "rights_evidence": [],
            "notes": "",
        }
        for case in parsed.cases
    ]
    review: dict[str, object] = {
        "schema_version": "1.0",
        "registry_id": parsed.registry_id,
        "source_registry_sha256": source_sha256,
        "registry_content_sha256": content_sha256,
        "prepared_at": prepared_at,
        "reviewer": reviewer_name,
        "reviewed_at": None,
        "selection": {
            "method_decision": "pending",
            "exclusions_decision": "pending",
            "known_failures_decision": "pending",
            "notes": "",
        },
        "cases": review_cases,
    }
    _write_atomic_new(output_path, review, "quality-review")
    return QualityRegistryReviewReport(
        registry_id=parsed.registry_id,
        source_registry_sha256=source_sha256,
        registry_content_sha256=content_sha256,
        total_cases=len(review_cases),
        approved_cases=0,
        pending_cases=len(review_cases),
        rejected_cases=0,
        ready=False,
        output_registry_sha256=None,
        blockers=("selection:pending", "cases:pending"),
    )


def _review_datetime(value: object, name: str) -> datetime | None:
    if value is None:
        return None
    text = _string(value, name)
    if "T" not in text:
        message = f"{name} must be an ISO 8601 date-time"
        raise ValidationError(message)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        message = f"{name} must be an ISO 8601 date-time"
        raise ValidationError(message) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        message = f"{name} must include a UTC offset"
        raise ValidationError(message)
    return parsed


def _review_note(value: object, name: str) -> str:
    if not isinstance(value, str):
        message = f"{name} must be a string"
        raise ValidationError(message)
    if Path(value).is_absolute() or value.startswith(("./", "../", "file:")):
        message = f"{name} must not expose local paths"
        raise ValidationError(message)
    return value


def apply_quality_registry_review(
    registry_path: Path,
    review_path: Path,
    output_path: Path,
) -> QualityRegistryReviewReport:
    """Apply a complete review to a new private registry without self-approval."""
    resolved_output = output_path.resolve()
    if resolved_output in {registry_path.resolve(), review_path.resolve()}:
        message = "reviewed registry output must differ from registry and review inputs"
        raise ValidationError(message)
    parsed = _parse_private_registry(registry_path)
    review, _ = _load_json(review_path, "private quality review")
    _fields(review, _REVIEW_FIELDS, "private quality review")
    if review["schema_version"] != "1.0":
        message = "unsupported private quality-review schema_version; expected '1.0'"
        raise ValidationError(message)
    if review["registry_id"] != parsed.registry_id:
        message = "private quality review registry_id does not match registry"
        raise ValidationError(message)

    source_sha256 = _registry_sha256(parsed)
    content_sha256 = _registry_content_sha256(parsed)
    if review["source_registry_sha256"] != source_sha256:
        message = "private quality review source registry digest is stale"
        raise ValidationError(message)
    if review["registry_content_sha256"] != content_sha256:
        message = "private quality review content digest is stale"
        raise ValidationError(message)
    if _review_datetime(review["prepared_at"], "quality-review prepared_at") is None:
        message = "quality-review prepared_at is required"
        raise ValidationError(message)
    _string(review["reviewer"], "quality-review reviewer")
    reviewed_at = _review_datetime(review["reviewed_at"], "quality-review reviewed_at")

    selection = review["selection"]
    if not isinstance(selection, dict):
        message = "private quality review selection must be an object"
        raise ValidationError(message)
    _fields(selection, _REVIEW_SELECTION_FIELDS, "private quality review selection")
    selection_decisions = tuple(
        _choice(
            selection[field],
            f"private quality review selection.{field}",
            _REVIEW_DECISIONS,
        )
        for field in (
            "method_decision",
            "exclusions_decision",
            "known_failures_decision",
        )
    )
    _review_note(selection["notes"], "private quality review selection.notes")

    raw_review_cases = review["cases"]
    if not isinstance(raw_review_cases, list):
        message = "private quality review cases must be an array"
        raise ValidationError(message)
    expected_cases = {str(case["id"]): case for case in parsed.cases}
    decisions_by_id: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    approved_cases = 0
    pending_cases = 0
    rejected_cases = 0
    for index, item in enumerate(raw_review_cases):
        if not isinstance(item, dict):
            message = f"private quality review case at index {index} must be an object"
            raise ValidationError(message)
        _fields(
            item, _REVIEW_CASE_FIELDS, f"private quality review case at index {index}"
        )
        case_id = _string(
            item["id"], f"private quality review case at index {index} id"
        )
        if case_id in decisions_by_id:
            message = "private quality review contains duplicate case ids"
            raise ValidationError(message)
        expected = expected_cases.get(case_id)
        if expected is None:
            message = "private quality review case ids do not match registry"
            raise ValidationError(message)
        if item["prompt_sha256"] != expected["prompt_sha256"]:
            message = f"private quality review prompt digest is stale for {case_id!r}"
            raise ValidationError(message)
        if item["reference_asset_sha256s"] != expected["reference_asset_sha256s"]:
            message = f"private quality review asset digests are stale for {case_id!r}"
            raise ValidationError(message)
        rights_decision = _choice(
            item["rights_decision"],
            f"private quality review case {case_id!r} rights_decision",
            _REVIEW_DECISIONS,
        )
        selection_decision = _choice(
            item["selection_decision"],
            f"private quality review case {case_id!r} selection_decision",
            _REVIEW_DECISIONS,
        )
        evidence = _strings(
            item["rights_evidence"],
            f"private quality review case {case_id!r} rights_evidence",
        )
        if rights_decision == "approved":
            if not evidence:
                message = (
                    f"approved private quality review case {case_id!r} requires "
                    "rights evidence"
                )
                raise ValidationError(message)
            for uri in evidence:
                _https_url(uri, f"private quality review case {case_id!r} evidence")
        elif evidence:
            message = (
                f"non-approved private quality review case {case_id!r} cannot "
                "contain rights evidence"
            )
            raise ValidationError(message)
        _review_note(item["notes"], f"private quality review case {case_id!r} notes")
        decisions_by_id[case_id] = (
            rights_decision,
            selection_decision,
            evidence,
        )
        decisions = {rights_decision, selection_decision}
        if "rejected" in decisions:
            rejected_cases += 1
        elif "pending" in decisions:
            pending_cases += 1
        else:
            approved_cases += 1

    if set(decisions_by_id) != set(expected_cases):
        message = "private quality review case ids do not match registry"
        raise ValidationError(message)

    blockers: list[str] = []
    if any(decision == "pending" for decision in selection_decisions):
        blockers.append("selection:pending")
    if any(decision == "rejected" for decision in selection_decisions):
        blockers.append("selection:rejected")
    if pending_cases:
        blockers.append(f"cases:pending:{pending_cases}")
    if rejected_cases:
        blockers.append(f"cases:rejected:{rejected_cases}")
    if reviewed_at is None:
        blockers.append("reviewed_at:missing")

    ready = not blockers
    output_registry_sha256: str | None = None
    if ready:
        if reviewed_at is None:  # pragma: no cover - guarded by blockers
            message = "private quality review is missing reviewed_at"
            raise ValidationError(message)
        reviewed_registry = deepcopy(parsed.record)
        registry_selection = reviewed_registry["selection"]
        if not isinstance(registry_selection, dict):  # pragma: no cover - prevalidated
            message = "private registry selection must be an object"
            raise ValidationError(message)
        registry_selection["exclusions_reviewed"] = True
        registry_selection["known_failures_reviewed"] = True
        registry_cases = reviewed_registry["cases"]
        if not isinstance(registry_cases, list):  # pragma: no cover - prevalidated
            message = "private registry cases must be an array"
            raise ValidationError(message)
        for case in registry_cases:
            if not isinstance(case, dict):  # pragma: no cover - prevalidated
                message = "private registry case must be an object"
                raise ValidationError(message)
            case_id = str(case["id"])
            rights_decision, _, evidence = decisions_by_id[case_id]
            if rights_decision != "approved":  # pragma: no cover - guarded by ready
                message = "private quality review is not complete"
                raise ValidationError(message)
            case["rights_status"] = "approved"
            case["rights_evidence"] = list(evidence)
        reviewed_registry["updated_at"] = reviewed_at.date().isoformat()
        _write_atomic_new(output_path, reviewed_registry, "reviewed registry")
        try:
            reviewed = _parse_private_registry(output_path)
        except ValidationError:
            output_path.unlink(missing_ok=True)
            raise
        output_registry_sha256 = _registry_sha256(reviewed)

    return QualityRegistryReviewReport(
        registry_id=parsed.registry_id,
        source_registry_sha256=source_sha256,
        registry_content_sha256=content_sha256,
        total_cases=len(parsed.cases),
        approved_cases=approved_cases,
        pending_cases=pending_cases,
        rejected_cases=rejected_cases,
        ready=ready,
        output_registry_sha256=output_registry_sha256,
        blockers=tuple(blockers),
    )
