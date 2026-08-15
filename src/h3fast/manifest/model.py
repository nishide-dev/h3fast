"""Validation for H3Fast derivative model artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from h3fast.exceptions import ValidationError
from h3fast.manifest.checksums import validate_relative_path, verify_checksums

if TYPE_CHECKING:
    from pathlib import Path

IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_id",
    "artifact_type",
    "base_model",
    "base_revision",
    "task_family",
    "runtime",
    "components",
    "license",
    "build",
}


@dataclass(frozen=True, slots=True)
class ModelVerificationReport:
    """Successful model artifact verification result."""

    artifact_id: str
    base_revision: str
    verified_files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data."""
        return {
            "valid": True,
            "artifact_id": self.artifact_id,
            "base_revision": self.base_revision,
            "verified_files": list(self.verified_files),
        }


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        msg = f"model manifest is missing: {path}"
        raise ValidationError(msg) from error
    except json.JSONDecodeError as error:
        msg = f"model manifest is not valid JSON: {error}"
        raise ValidationError(msg) from error
    if not isinstance(value, dict):
        msg = "model manifest root must be an object"
        raise ValidationError(msg)
    return value


def _require_string(manifest: dict[str, object], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value:
        msg = f"manifest field {field!r} must be a non-empty string"
        raise ValidationError(msg)
    return value


def _validate_runtime(value: object) -> None:
    if not isinstance(value, dict):
        msg = "manifest field 'runtime' must be an object"
        raise ValidationError(msg)
    if value.get("name") != "h3fast":
        msg = "manifest runtime.name must be 'h3fast'"
        raise ValidationError(msg)
    requires = value.get("requires")
    tested = value.get("tested_versions")
    if not isinstance(requires, str) or not requires:
        msg = "manifest runtime.requires must be a non-empty version specifier"
        raise ValidationError(msg)
    if (
        not isinstance(tested, list)
        or not tested
        or not all(isinstance(item, str) and item for item in tested)
    ):
        msg = "manifest runtime.tested_versions must contain tested versions"
        raise ValidationError(msg)


def _validate_components(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        msg = "manifest components must be a non-empty array"
        raise ValidationError(msg)
    indexes: list[str] = []
    for component in value:
        if not isinstance(component, dict):
            msg = "each manifest component must be an object"
            raise ValidationError(msg)
        for field in ("name", "format", "dtype", "index"):
            if not isinstance(component.get(field), str) or not component[field]:
                msg = f"component field {field!r} must be a non-empty string"
                raise ValidationError(msg)
        index = str(component["index"])
        validate_relative_path(index)
        indexes.append(index)
    return tuple(indexes)


def validate_model_manifest(manifest: dict[str, object]) -> tuple[str, ...]:
    """Validate the required Phase 1 manifest contract."""
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS.difference(manifest))
    if missing:
        msg = f"manifest is missing required fields: {', '.join(missing)}"
        raise ValidationError(msg)
    if manifest.get("schema_version") != "1.0":
        msg = "unsupported manifest schema_version; expected '1.0'"
        raise ValidationError(msg)

    for field in ("artifact_id", "artifact_type", "base_model", "task_family"):
        _require_string(manifest, field)
    revision = _require_string(manifest, "base_revision")
    if not IMMUTABLE_REVISION_PATTERN.fullmatch(revision):
        msg = "manifest base_revision must be a lowercase 40-character commit SHA"
        raise ValidationError(msg)

    _validate_runtime(manifest.get("runtime"))
    indexes = _validate_components(manifest.get("components"))
    for field in ("license", "build"):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            msg = f"manifest field {field!r} must be a non-empty object"
            raise ValidationError(msg)
    return indexes


def verify_model_artifact(root: Path) -> ModelVerificationReport:
    """Validate a model manifest and all files in its checksum inventory."""
    if not root.is_dir():
        msg = f"model artifact directory does not exist: {root}"
        raise ValidationError(msg)
    manifest = _load_manifest(root / "h3fast_manifest.json")
    component_indexes = validate_model_manifest(manifest)
    verified = verify_checksums(root, root / "checksums.sha256")
    missing_indexes = sorted(set(component_indexes).difference(verified))
    if missing_indexes:
        msg = "component indexes are absent from checksums.sha256: " + ", ".join(
            missing_indexes
        )
        raise ValidationError(msg)
    return ModelVerificationReport(
        artifact_id=_require_string(manifest, "artifact_id"),
        base_revision=_require_string(manifest, "base_revision"),
        verified_files=verified,
    )
