"""Validation for the JSON-compatible Phase 0 benchmark protocol."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class ProtocolReport:
    """Benchmark protocol readiness report."""

    protocol_id: str
    status: str
    unresolved: tuple[str, ...]

    @property
    def ready(self) -> bool:
        """Return whether the protocol is complete enough for a baseline claim."""
        return self.status == "ready" and not self.unresolved

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data."""
        return {
            "valid": True,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "ready": self.ready,
            "unresolved": list(self.unresolved),
        }


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Protocol-owned settings that may change benchmark runtime behavior."""

    dit_layerwise_resident_layers: int

    def to_dict(self) -> dict[str, int]:
        """Return JSON-serializable effective runtime settings."""
        return {
            "dit_layerwise_resident_layers": self.dit_layerwise_resident_layers,
        }


def _load_protocol(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        msg = f"benchmark protocol is missing: {path}"
        raise ValidationError(msg) from error
    except json.JSONDecodeError as error:
        msg = "benchmark protocol must use JSON-compatible YAML syntax: " + str(error)
        raise ValidationError(msg) from error
    if not isinstance(value, dict):
        msg = "benchmark protocol root must be an object"
        raise ValidationError(msg)
    return value


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"benchmark protocol field {field!r} must be a non-empty string"
        raise ValidationError(msg)
    return value


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"benchmark protocol field {field!r} must be an object"
        raise ValidationError(msg)
    return value


def validate_protocol(path: Path) -> ProtocolReport:
    """Validate a draft or ready benchmark protocol."""
    protocol = _load_protocol(path)
    if protocol.get("schema_version") != "1.0":
        msg = "unsupported benchmark protocol schema_version"
        raise ValidationError(msg)

    protocol_id = _non_empty_string(protocol.get("protocol_id"), "protocol_id")
    status = _non_empty_string(protocol.get("status"), "status")
    if status not in {"draft", "ready"}:
        msg = "benchmark protocol status must be 'draft' or 'ready'"
        raise ValidationError(msg)

    unresolved_raw = protocol.get("unresolved", [])
    if not isinstance(unresolved_raw, list) or not all(
        isinstance(item, str) and item for item in unresolved_raw
    ):
        msg = "benchmark protocol unresolved must be an array of strings"
        raise ValidationError(msg)
    unresolved = tuple(unresolved_raw)

    base_model = _object(protocol.get("base_model"), "base_model")
    environment = _object(protocol.get("environment"), "environment")
    runtime = _object(protocol.get("runtime"), "runtime")
    cases = protocol.get("cases")
    _object(protocol.get("measurement"), "measurement")
    if not isinstance(cases, list) or not cases:
        msg = "benchmark protocol cases must be a non-empty array"
        raise ValidationError(msg)
    resident_layers = runtime.get("dit_layerwise_resident_layers")
    if (
        not isinstance(resident_layers, int)
        or isinstance(resident_layers, bool)
        or not 1 <= resident_layers <= 50
    ):
        msg = (
            "benchmark protocol runtime.dit_layerwise_resident_layers "
            "must be an integer between 1 and 50"
        )
        raise ValidationError(msg)

    quality_raw = protocol.get("quality")
    if quality_raw is not None:
        quality = _object(quality_raw, "quality")
        for field in (
            "reference_id",
            "reference_path",
            "method",
            "profile",
            "video_decode_format",
            "audio_decode_format",
            "scope",
        ):
            _non_empty_string(quality.get(field), f"quality.{field}")
        if quality.get("method") != "exact-decoded-artifact-v1":
            msg = "benchmark protocol quality method is unsupported"
            raise ValidationError(msg)
        if quality.get("profile") != "exact":
            msg = "benchmark protocol quality profile must be 'exact'"
            raise ValidationError(msg)
        baseline_runs = quality.get("baseline_measured_runs")
        if (
            not isinstance(baseline_runs, int)
            or isinstance(baseline_runs, bool)
            or baseline_runs < 3
        ):
            msg = "benchmark protocol quality baseline requires at least three runs"
            raise ValidationError(msg)
        if not isinstance(quality.get("formal_quality_set_ready"), bool):
            msg = "benchmark protocol formal_quality_set_ready must be boolean"
            raise ValidationError(msg)

    if status == "ready":
        revision = base_model.get("revision")
        if not isinstance(revision, str) or not IMMUTABLE_REVISION_PATTERN.fullmatch(
            revision
        ):
            msg = "ready protocol requires an immutable base model revision"
            raise ValidationError(msg)
        if unresolved:
            msg = "ready protocol cannot contain unresolved items"
            raise ValidationError(msg)
        accelerator = environment.get("accelerator")
        software = environment.get("software")
        if not isinstance(accelerator, dict) or not accelerator.get("model"):
            msg = "ready protocol requires an accelerator model"
            raise ValidationError(msg)
        if not isinstance(software, dict) or not software.get("sglang"):
            msg = "ready protocol requires a pinned SGLang version"
            raise ValidationError(msg)

    return ProtocolReport(
        protocol_id=protocol_id,
        status=status,
        unresolved=unresolved,
    )


def load_runtime_settings(path: Path) -> RuntimeSettings:
    """Load validated runtime settings from a benchmark protocol."""
    validate_protocol(path)
    protocol = _load_protocol(path)
    runtime = _object(protocol.get("runtime"), "runtime")
    resident_layers = cast("int", runtime["dit_layerwise_resident_layers"])
    return RuntimeSettings(dit_layerwise_resident_layers=resident_layers)
