"""Fail-closed validation for H3 territory evidence inventories."""

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

REQUIRED_FLOW_IDS = frozenset(
    {
        "benchmark-output-storage",
        "ci-artifact-storage",
        "development-host",
        "github-actions",
        "gpu-benchmark-host",
        "initial-user-access",
        "output-use",
        "public-source-distribution",
        "runtime-execution",
        "source-storage",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "inventory_id",
        "status",
        "updated_at",
        "source_issue",
        "license_evidence",
        "legal_approval",
        "flows",
    }
)
_LICENSE_FIELDS = frozenset(
    {"base_revision", "license_sha256", "applicable_territory", "evidence"}
)
_APPROVAL_FIELDS = frozenset({"state", "owner", "deadline", "approved_at", "evidence"})
_FLOW_FIELDS = frozenset(
    {
        "id",
        "description",
        "location_scope",
        "country_codes",
        "territory_assessment",
        "h3_relation",
        "decision",
        "written_license",
        "operator",
        "owner",
        "deadline",
        "evidence",
        "notes",
    }
)
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COUNTRY_CODE_PATTERN = re.compile(r"[A-Z]{2}")


@dataclass(frozen=True, slots=True)
class TerritoryInventoryReport:
    """Validated H3-use territory inventory readiness result."""

    inventory_id: str
    status: str
    ready: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable territory readiness data."""
        return {
            "schema_version": "1.0",
            "inventory_id": self.inventory_id,
            "status": self.status,
            "ready": self.ready,
            "blockers": list(self.blockers),
        }


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        message = f"territory inventory is missing: {path}"
        raise ValidationError(message) from error
    except json.JSONDecodeError as error:
        message = f"territory inventory is not valid JSON: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict):
        message = "territory inventory root must be an object"
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


def _validate_license(value: object) -> None:
    if not isinstance(value, dict):
        message = "license_evidence must be an object"
        raise ValidationError(message)
    _fields(value, _LICENSE_FIELDS, "license_evidence")
    revision = _string(value["base_revision"], "license_evidence.base_revision")
    if not _REVISION_PATTERN.fullmatch(revision):
        message = "license_evidence.base_revision must be a lowercase 40-character SHA"
        raise ValidationError(message)
    digest = _string(value["license_sha256"], "license_evidence.license_sha256")
    if not _SHA256_PATTERN.fullmatch(digest):
        message = "license_evidence.license_sha256 must be a lowercase SHA-256"
        raise ValidationError(message)
    _string(value["applicable_territory"], "license_evidence.applicable_territory")
    _evidence(value["evidence"], "license_evidence.evidence", required=True)


def _validate_approval(value: object) -> str:
    if not isinstance(value, dict):
        message = "legal_approval must be an object"
        raise ValidationError(message)
    _fields(value, _APPROVAL_FIELDS, "legal_approval")
    state = _string(value["state"], "legal_approval.state")
    if state not in {"unassigned", "pending", "approved"}:
        message = f"legal_approval has unsupported state: {state}"
        raise ValidationError(message)
    owner = _nullable_string(value["owner"], "legal_approval.owner")
    _date(value["deadline"], "legal_approval.deadline")
    approved_at = _date_time(value["approved_at"], "legal_approval.approved_at")
    evidence = _evidence(
        value["evidence"], "legal_approval.evidence", required=state == "approved"
    )
    if state == "unassigned" and (owner is not None or approved_at is not None):
        message = "unassigned legal_approval cannot have an owner or approval time"
        raise ValidationError(message)
    if state == "pending" and (owner is None or approved_at is not None):
        message = "pending legal_approval requires an owner without approved_at"
        raise ValidationError(message)
    if state == "approved" and (owner is None or approved_at is None or not evidence):
        message = "approved legal_approval requires owner, approved_at, and evidence"
        raise ValidationError(message)
    return state


def _country_codes(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and _COUNTRY_CODE_PATTERN.fullmatch(item)
        for item in value
    ):
        message = f"{name} must be an array of ISO 3166-1 alpha-2 country codes"
        raise ValidationError(message)
    if len(set(value)) != len(value):
        message = f"{name} must not contain duplicates"
        raise ValidationError(message)
    return tuple(value)


def _choice(value: object, name: str, choices: set[str]) -> str:
    result = _string(value, name)
    if result not in choices:
        message = f"{name} has unsupported value: {result}"
        raise ValidationError(message)
    return result


def _validate_flow(value: object, index: int) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        message = f"territory flow at index {index} must be an object"
        raise ValidationError(message)
    _fields(value, _FLOW_FIELDS, f"territory flow at index {index}")
    flow_id = _string(value["id"], f"territory flow at index {index} id")
    _string(value["description"], f"territory flow {flow_id!r} description")
    location_scope = _choice(
        value["location_scope"],
        f"territory flow {flow_id!r} location_scope",
        {"unknown", "countries", "global"},
    )
    country_codes = _country_codes(
        value["country_codes"], f"territory flow {flow_id!r} country_codes"
    )
    if location_scope == "countries" and not country_codes:
        message = f"territory flow {flow_id!r} countries scope requires country codes"
        raise ValidationError(message)
    if location_scope != "countries" and country_codes:
        message = (
            f"territory flow {flow_id!r} country codes require location_scope countries"
        )
        raise ValidationError(message)

    assessment = _choice(
        value["territory_assessment"],
        f"territory flow {flow_id!r} territory_assessment",
        {"unknown", "within-applicable", "includes-excluded", "not-applicable"},
    )
    h3_relation = _choice(
        value["h3_relation"],
        f"territory flow {flow_id!r} h3_relation",
        {"none", "materials", "h3-works", "output", "both", "undetermined"},
    )
    decision = _choice(
        value["decision"],
        f"territory flow {flow_id!r} decision",
        {
            "unknown",
            "approved-under-community-license",
            "approved-with-written-license",
            "not-applicable",
        },
    )
    if (
        location_scope == "global"
        and decision != "not-applicable"
        and assessment != "includes-excluded"
    ):
        message = f"global territory flow {flow_id!r} must include excluded territories"
        raise ValidationError(message)
    written_license = _nullable_string(
        value["written_license"], f"territory flow {flow_id!r} written_license"
    )
    if written_license is not None:
        _https_url(written_license, f"territory flow {flow_id!r} written_license")
    if decision == "approved-with-written-license" and (
        assessment != "includes-excluded" or written_license is None
    ):
        message = (
            f"territory flow {flow_id!r} written-license approval requires "
            "excluded-territory assessment and evidence"
        )
        raise ValidationError(message)
    if decision != "approved-with-written-license" and written_license is not None:
        message = f"territory flow {flow_id!r} has unused written-license evidence"
        raise ValidationError(message)
    if (
        decision == "approved-under-community-license"
        and assessment != "within-applicable"
    ):
        message = (
            f"territory flow {flow_id!r} community-license approval requires "
            "within-applicable assessment"
        )
        raise ValidationError(message)
    if decision == "not-applicable":
        if h3_relation != "none":
            message = (
                f"territory flow {flow_id!r} not-applicable decision requires "
                "no H3 relation"
            )
            raise ValidationError(message)
        if assessment != "not-applicable":
            message = (
                f"territory flow {flow_id!r} not-applicable decision requires "
                "not-applicable assessment"
            )
            raise ValidationError(message)

    operator = _nullable_string(
        value["operator"], f"territory flow {flow_id!r} operator"
    )
    owner = _nullable_string(value["owner"], f"territory flow {flow_id!r} owner")
    _date(value["deadline"], f"territory flow {flow_id!r} deadline")
    _evidence(value["evidence"], f"territory flow {flow_id!r} evidence", required=True)
    _string(value["notes"], f"territory flow {flow_id!r} notes")
    if decision != "unknown" and owner is None:
        message = f"resolved territory flow {flow_id!r} requires an owner"
        raise ValidationError(message)

    blockers: list[str] = []
    if location_scope == "unknown" and decision != "not-applicable":
        blockers.append(f"flow:{flow_id}:location-unknown")
    if assessment == "unknown":
        blockers.append(f"flow:{flow_id}:territory-assessment-unknown")
    if h3_relation == "undetermined":
        blockers.append(f"flow:{flow_id}:h3-relation-undetermined")
    if decision == "unknown":
        blockers.append(f"flow:{flow_id}:decision-unknown")
    if operator is None:
        blockers.append(f"flow:{flow_id}:operator-unassigned")
    if owner is None:
        blockers.append(f"flow:{flow_id}:owner-unassigned")
    return flow_id, tuple(blockers)


def check_territory_inventory(path: Path) -> TerritoryInventoryReport:
    """Validate territory evidence and report unresolved H3-use blockers."""
    inventory = _load(path)
    _fields(inventory, _TOP_LEVEL_FIELDS, "territory inventory")
    if inventory["schema_version"] != "1.0":
        message = "unsupported territory inventory schema_version; expected '1.0'"
        raise ValidationError(message)
    inventory_id = _string(inventory["inventory_id"], "inventory_id")
    if inventory_id != "initial-runtime-territories":
        message = f"unsupported territory inventory_id: {inventory_id}"
        raise ValidationError(message)
    status = _choice(inventory["status"], "status", {"incomplete", "approved"})
    _date(inventory["updated_at"], "updated_at")
    _https_url(inventory["source_issue"], "source_issue")
    _validate_license(inventory["license_evidence"])

    blockers: list[str] = []
    approval_state = _validate_approval(inventory["legal_approval"])
    if approval_state != "approved":
        blockers.append(f"legal-approval:{approval_state}")

    flows = inventory["flows"]
    if not isinstance(flows, list):
        message = "flows must be an array"
        raise ValidationError(message)
    seen: set[str] = set()
    for index, value in enumerate(flows):
        flow_id, flow_blockers = _validate_flow(value, index)
        if flow_id in seen:
            message = f"duplicate territory flow id: {flow_id}"
            raise ValidationError(message)
        seen.add(flow_id)
        blockers.extend(flow_blockers)
    if seen != REQUIRED_FLOW_IDS:
        missing = sorted(REQUIRED_FLOW_IDS.difference(seen))
        unknown = sorted(seen.difference(REQUIRED_FLOW_IDS))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        message = f"territory flows do not match required set ({'; '.join(details)})"
        raise ValidationError(message)

    ready = not blockers
    if status == "approved" and not ready:
        message = "territory inventory claims approval while blockers remain"
        raise ValidationError(message)
    if status == "incomplete" and ready:
        message = "territory inventory is incomplete even though every flow is approved"
        raise ValidationError(message)
    return TerritoryInventoryReport(
        inventory_id=inventory_id,
        status=status,
        ready=ready,
        blockers=tuple(blockers),
    )
