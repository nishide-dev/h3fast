"""Validation for machine-readable H3Fast release gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

from h3fast.benchmarks.quality_sets import check_formal_quality_set
from h3fast.exceptions import ValidationError

REQUIRED_APPROVAL_ROLES = frozenset({"release_approver", "schema_owner"})
INITIAL_RUNTIME_REQUIRED_CHECKS = frozenset(
    {
        "artifact-notices",
        "byow-converter",
        "clean-machine-reproduction",
        "code-boundary-classification",
        "code-boundary-engineering",
        "formal-quality-set",
        "gpu-e2e",
        "license-evidence",
        "public-benchmark",
        "supply-chain",
        "support-target",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "status",
        "updated_at",
        "source_issue",
        "approvals",
        "checks",
    }
)
_APPROVAL_FIELDS = frozenset({"state", "owner", "deadline", "approved_at", "evidence"})
_CHECK_FIELDS = frozenset(
    {"id", "description", "status", "owner", "deadline", "evidence"}
)
_FORMAL_QUALITY_SET_PATH = "benchmarks/quality/formal-quality-set.json"


@dataclass(frozen=True, slots=True)
class ReleaseGateReport:
    """Validated release readiness result."""

    release_id: str
    status: str
    ready: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable release readiness data."""
        return {
            "schema_version": "1.1",
            "release_id": self.release_id,
            "status": self.status,
            "ready": self.ready,
            "blockers": list(self.blockers),
        }


def _load_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        message = f"release gate record is missing: {path}"
        raise ValidationError(message) from error
    except json.JSONDecodeError as error:
        message = f"release gate record is not valid JSON: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict):
        message = "release gate record root must be an object"
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


def _https_url(value: object, name: str) -> str:
    text = _string(value, name)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        message = f"{name} must be an HTTPS URL"
        raise ValidationError(message)
    return text


def _date(value: object, name: str) -> str | None:
    text = _nullable_string(value, name)
    if text is None:
        return None
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


def _evidence(value: object, name: str, *, required: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        message = f"{name} must be an array of non-empty strings"
        raise ValidationError(message)
    if required and not value:
        message = f"{name} must contain at least one item"
        raise ValidationError(message)
    return tuple(value)


def _validate_approval(role: str, value: object) -> str:
    if not isinstance(value, dict):
        message = f"approval {role!r} must be an object"
        raise ValidationError(message)
    _fields(value, _APPROVAL_FIELDS, f"approval {role!r}")
    state = _string(value["state"], f"approval {role!r} state")
    if state not in {"unassigned", "pending", "approved"}:
        message = f"approval {role!r} has unsupported state: {state}"
        raise ValidationError(message)
    owner = _nullable_string(value["owner"], f"approval {role!r} owner")
    deadline = _date(value["deadline"], f"approval {role!r} deadline")
    if deadline is None:
        message = f"approval {role!r} requires a decision deadline"
        raise ValidationError(message)
    approved_at = _date_time(value["approved_at"], f"approval {role!r} approved_at")
    evidence = _evidence(
        value["evidence"],
        f"approval {role!r} evidence",
        required=state == "approved",
    )
    if state == "unassigned" and (owner is not None or approved_at is not None):
        message = f"unassigned approval {role!r} cannot have an owner or approval time"
        raise ValidationError(message)
    if state == "pending" and (owner is None or deadline is None or approved_at):
        message = (
            f"pending approval {role!r} requires owner and deadline without approved_at"
        )
        raise ValidationError(message)
    if state == "approved" and (owner is None or approved_at is None or not evidence):
        message = (
            f"approved approval {role!r} requires owner, approved_at, and evidence"
        )
        raise ValidationError(message)
    return state


def _validate_check(value: object, index: int) -> tuple[str, str, str | None]:
    if not isinstance(value, dict):
        message = f"release check at index {index} must be an object"
        raise ValidationError(message)
    _fields(value, _CHECK_FIELDS, f"release check at index {index}")
    check_id = _string(value["id"], f"release check at index {index} id")
    _string(value["description"], f"release check {check_id!r} description")
    status = _string(value["status"], f"release check {check_id!r} status")
    if status not in {"blocked", "passed", "not_applicable"}:
        message = f"release check {check_id!r} has unsupported status: {status}"
        raise ValidationError(message)
    owner = _nullable_string(value["owner"], f"release check {check_id!r} owner")
    deadline = _date(value["deadline"], f"release check {check_id!r} deadline")
    if deadline is None:
        message = f"release check {check_id!r} requires a disposition deadline"
        raise ValidationError(message)
    _evidence(
        value["evidence"],
        f"release check {check_id!r} evidence",
        required=True,
    )
    if status != "blocked" and owner is None:
        message = f"resolved release check {check_id!r} requires an owner"
        raise ValidationError(message)
    return check_id, status, owner


def check_release_gate(path: Path) -> ReleaseGateReport:
    """Validate a release record and fail closed until every gate is approved."""
    record = _load_record(path)
    _fields(record, _TOP_LEVEL_FIELDS, "release gate record")
    if record["schema_version"] != "1.1":
        message = "unsupported release gate schema_version; expected '1.1'"
        raise ValidationError(message)
    release_id = _string(record["release_id"], "release_id")
    if release_id != "initial-runtime":
        message = f"unsupported release_id: {release_id}"
        raise ValidationError(message)
    status = _string(record["status"], "status")
    if status not in {"blocked", "approved"}:
        message = f"unsupported release status: {status}"
        raise ValidationError(message)
    if _date(record["updated_at"], "updated_at") is None:
        message = "updated_at must be an ISO 8601 date"
        raise ValidationError(message)
    _https_url(record["source_issue"], "source_issue")

    approvals = record["approvals"]
    if not isinstance(approvals, dict):
        message = "approvals must be an object"
        raise ValidationError(message)
    approval_roles = set(approvals)
    if approval_roles != REQUIRED_APPROVAL_ROLES:
        missing = sorted(REQUIRED_APPROVAL_ROLES.difference(approval_roles))
        unknown = sorted(approval_roles.difference(REQUIRED_APPROVAL_ROLES))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        message = (
            f"release approvals do not match required roles ({'; '.join(details)})"
        )
        raise ValidationError(message)

    blockers: list[str] = []
    for role in sorted(REQUIRED_APPROVAL_ROLES):
        state = _validate_approval(role, approvals[role])
        if state != "approved":
            blockers.append(f"approval:{role}:{state}")

    checks = record["checks"]
    if not isinstance(checks, list):
        message = "checks must be an array"
        raise ValidationError(message)
    seen: set[str] = set()
    for index, value in enumerate(checks):
        check_id, check_status, owner = _validate_check(value, index)
        if check_id in seen:
            message = f"duplicate release check id: {check_id}"
            raise ValidationError(message)
        seen.add(check_id)
        if (
            check_id == "formal-quality-set"
            and check_status == "passed"
            and isinstance(value, dict)
            and _FORMAL_QUALITY_SET_PATH in value["evidence"]
            and not check_formal_quality_set(Path(_FORMAL_QUALITY_SET_PATH)).ready
        ):
            message = (
                "formal-quality-set release check cannot pass while its committed "
                "record is incomplete"
            )
            raise ValidationError(message)
        if check_status == "blocked":
            blockers.append(f"check:{check_id}:blocked")
            if owner is None:
                blockers.append(f"check:{check_id}:owner-unassigned")

    if seen != INITIAL_RUNTIME_REQUIRED_CHECKS:
        missing = sorted(INITIAL_RUNTIME_REQUIRED_CHECKS.difference(seen))
        unknown = sorted(seen.difference(INITIAL_RUNTIME_REQUIRED_CHECKS))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        message = (
            f"initial runtime checks do not match required set ({'; '.join(details)})"
        )
        raise ValidationError(message)

    ready = not blockers
    if status == "approved" and not ready:
        message = "release record claims approval while blockers remain"
        raise ValidationError(message)
    if status == "blocked" and ready:
        message = "release record is blocked even though every gate is approved"
        raise ValidationError(message)
    return ReleaseGateReport(
        release_id=release_id,
        status=status,
        ready=ready,
        blockers=tuple(blockers),
    )
