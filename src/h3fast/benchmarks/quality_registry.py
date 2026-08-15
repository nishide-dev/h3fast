"""Compile private quality registries into redacted formal-set metadata."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
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
    cases = [
        _compile_case(case, registry_root=registry_path.parent, index=index)
        for index, case in enumerate(raw_cases)
    ]
    case_ids = [case["id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        message = "private quality registry contains duplicate case ids"
        raise ValidationError(message)

    template, _ = _load_json(template_path, "formal quality-set template")
    check_formal_quality_set(template_path)
    template["status"] = "incomplete"
    template["updated_at"] = updated_at
    template["selection"] = {
        "method": method,
        "registry_uri": uri,
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "exclusions_reviewed": exclusions_reviewed,
        "exclusions": list(exclusions),
        "known_failures_reviewed": known_failures_reviewed,
        "known_failures": list(known_failures),
    }
    template["cases"] = cases
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
        registry_id=registry_id,
        registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        smoke_cases=formal_report.smoke_cases,
        regression_cases=formal_report.regression_cases,
        ready=formal_report.ready,
        blockers=formal_report.blockers,
    )
