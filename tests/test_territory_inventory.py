"""Tests for fail-closed territory evidence inventories."""

import copy
import json
from pathlib import Path

import pytest

from h3fast.compliance import check_territory_inventory
from h3fast.exceptions import ValidationError

INVENTORY_PATH = Path("compliance/territories/initial-runtime.json")


def _inventory() -> dict[str, object]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _write_inventory(tmp_path: Path, inventory: object) -> Path:
    path = tmp_path / "territories.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    return path


def _approved_inventory() -> dict[str, object]:
    inventory = copy.deepcopy(_inventory())
    inventory["status"] = "approved"
    approval = inventory["legal_approval"]
    assert isinstance(approval, dict)
    approval.update(
        {
            "state": "approved",
            "owner": "legal-owner",
            "approved_at": "2026-08-20T00:00:00+09:00",
            "evidence": ["https://example.test/legal-approval"],
        }
    )
    flows = inventory["flows"]
    assert isinstance(flows, list)
    for flow in flows:
        assert isinstance(flow, dict)
        flow["owner"] = "legal-owner"
        if flow["operator"] is None:
            flow["operator"] = "declared-operator"
        if flow["location_scope"] == "unknown":
            flow["location_scope"] = "countries"
            flow["country_codes"] = ["JP"]
        if flow["h3_relation"] == "undetermined":
            flow["h3_relation"] = "none"
        if flow["h3_relation"] == "none":
            flow["decision"] = "not-applicable"
            if flow["territory_assessment"] == "unknown":
                flow["territory_assessment"] = "not-applicable"
        else:
            flow["decision"] = "approved-under-community-license"
            flow["territory_assessment"] = "within-applicable"
    return inventory


def _flow(inventory: dict[str, object], flow_id: str) -> dict[str, object]:
    flows = inventory["flows"]
    assert isinstance(flows, list)
    for flow in flows:
        assert isinstance(flow, dict)
        if flow["id"] == flow_id:
            return flow
    message = f"missing flow: {flow_id}"
    raise AssertionError(message)


def test_committed_territory_inventory_is_explicitly_incomplete() -> None:
    report = check_territory_inventory(INVENTORY_PATH)

    assert report.status == "incomplete"
    assert report.ready is False
    assert "legal-approval:unassigned" in report.blockers
    assert "flow:development-host:location-unknown" in report.blockers
    assert "flow:development-host:operator-unassigned" in report.blockers
    assert "flow:source-storage:h3-relation-undetermined" not in report.blockers
    assert "flow:public-source-distribution:decision-unknown" not in report.blockers


def test_fully_approved_territory_inventory_is_ready(tmp_path: Path) -> None:
    report = check_territory_inventory(
        _write_inventory(tmp_path, _approved_inventory())
    )

    assert report.ready is True
    assert report.blockers == ()
    assert report.to_dict()["status"] == "approved"


def test_approved_inventory_rejects_remaining_blockers(tmp_path: Path) -> None:
    inventory = _approved_inventory()
    flow = _flow(inventory, "development-host")
    flow["decision"] = "unknown"

    with pytest.raises(ValidationError, match="claims approval while blockers remain"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_incomplete_inventory_rejects_no_remaining_blockers(tmp_path: Path) -> None:
    inventory = _approved_inventory()
    inventory["status"] = "incomplete"

    with pytest.raises(ValidationError, match="incomplete even though every flow"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_global_flow_requires_excluded_assessment(tmp_path: Path) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "public-source-distribution")
    flow["h3_relation"] = "h3-works"
    flow["decision"] = "unknown"
    flow["territory_assessment"] = "unknown"

    with pytest.raises(ValidationError, match="must include excluded territories"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_country_scope_requires_codes_and_rejects_duplicates(tmp_path: Path) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "source-storage")
    flow["country_codes"] = []
    with pytest.raises(ValidationError, match="requires country codes"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    flow = _flow(inventory, "source-storage")
    flow["country_codes"] = ["US", "US"]
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_unknown_scope_rejects_country_codes(tmp_path: Path) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "development-host")
    flow["country_codes"] = ["JP"]

    with pytest.raises(ValidationError, match="require location_scope countries"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_written_license_decision_requires_excluded_scope_and_url(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "development-host")
    flow["decision"] = "approved-with-written-license"
    flow["owner"] = "legal-owner"
    with pytest.raises(ValidationError, match="requires excluded-territory"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    flow["territory_assessment"] = "includes-excluded"
    flow["written_license"] = "license-11"
    with pytest.raises(ValidationError, match="must be an HTTPS URL"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_unused_written_license_is_rejected(tmp_path: Path) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "development-host")
    flow["written_license"] = "https://example.test/license"

    with pytest.raises(ValidationError, match="unused written-license evidence"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_community_license_decision_requires_applicable_assessment(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "development-host")
    flow["decision"] = "approved-under-community-license"
    flow["owner"] = "legal-owner"

    with pytest.raises(ValidationError, match="requires within-applicable"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_not_applicable_decision_requires_no_h3_relation(tmp_path: Path) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "development-host")
    flow["decision"] = "not-applicable"
    flow["owner"] = "legal-owner"

    with pytest.raises(ValidationError, match="requires no H3 relation"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_not_applicable_decision_requires_matching_assessment(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "github-actions")
    flow["territory_assessment"] = "unknown"

    with pytest.raises(ValidationError, match="requires not-applicable assessment"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_resolved_flow_requires_owner(tmp_path: Path) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "github-actions")
    flow["decision"] = "not-applicable"
    flow["territory_assessment"] = "not-applicable"
    flow["owner"] = None

    with pytest.raises(ValidationError, match="requires an owner"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_inventory_rejects_missing_and_duplicate_flows(tmp_path: Path) -> None:
    inventory = _inventory()
    flows = inventory["flows"]
    assert isinstance(flows, list)
    flows.pop()
    with pytest.raises(ValidationError, match="missing: output-use"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    flows = inventory["flows"]
    assert isinstance(flows, list)
    assert isinstance(flows[0], dict)
    assert isinstance(flows[1], dict)
    flows[1]["id"] = flows[0]["id"]
    with pytest.raises(ValidationError, match="duplicate territory flow id"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_inventory_rejects_invalid_identity_and_license(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory["inventory_id"] = "other"
    with pytest.raises(ValidationError, match="unsupported territory inventory_id"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    license_evidence = inventory["license_evidence"]
    assert isinstance(license_evidence, dict)
    license_evidence["license_sha256"] = "invalid"
    with pytest.raises(ValidationError, match="must be a lowercase SHA-256"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_inventory_rejects_invalid_legal_approval(tmp_path: Path) -> None:
    inventory = _inventory()
    approval = inventory["legal_approval"]
    assert isinstance(approval, dict)
    approval["state"] = "pending"

    with pytest.raises(ValidationError, match="pending legal_approval requires"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_inventory_rejects_missing_and_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="inventory is missing"):
        check_territory_inventory(tmp_path / "missing.json")

    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError, match="not valid JSON"):
        check_territory_inventory(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValidationError, match="root must be an object"):
        check_territory_inventory(path)


def test_inventory_rejects_missing_and_unknown_fields(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory.pop("updated_at")
    with pytest.raises(ValidationError, match="missing required fields: updated_at"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    inventory["unexpected"] = True
    with pytest.raises(ValidationError, match="unknown fields: unexpected"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0", "unsupported territory inventory schema_version"),
        ("status", "ready", "status has unsupported value"),
        ("updated_at", "tomorrow", "must be an ISO 8601 date"),
        ("source_issue", "", "source_issue must be a non-empty string"),
        ("source_issue", "http://example.test/11", "must be an HTTPS URL"),
    ],
)
def test_inventory_rejects_invalid_top_level_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    inventory = _inventory()
    inventory[field] = value

    with pytest.raises(ValidationError, match=message):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_inventory_rejects_invalid_license_evidence(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory["license_evidence"] = []
    with pytest.raises(ValidationError, match="license_evidence must be an object"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    license_evidence = inventory["license_evidence"]
    assert isinstance(license_evidence, dict)
    license_evidence["base_revision"] = "main"
    with pytest.raises(ValidationError, match="lowercase 40-character SHA"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    license_evidence = inventory["license_evidence"]
    assert isinstance(license_evidence, dict)
    license_evidence["evidence"] = []
    with pytest.raises(ValidationError, match="evidence must contain"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_inventory_rejects_inconsistent_legal_approval(tmp_path: Path) -> None:
    inventory = _inventory()
    approval = inventory["legal_approval"]
    assert isinstance(approval, dict)
    approval["state"] = "accepted"
    with pytest.raises(ValidationError, match="unsupported state"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    approval = inventory["legal_approval"]
    assert isinstance(approval, dict)
    approval["owner"] = "legal-owner"
    with pytest.raises(ValidationError, match="unassigned legal_approval"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    approval = inventory["legal_approval"]
    assert isinstance(approval, dict)
    approval.update(
        {
            "state": "approved",
            "owner": "legal-owner",
            "approved_at": "2026-08-20T00:00:00",
            "evidence": ["https://example.test/approval"],
        }
    )
    with pytest.raises(ValidationError, match="must include a UTC offset"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_inventory_rejects_invalid_flow_shapes(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory["flows"] = {}
    with pytest.raises(ValidationError, match="flows must be an array"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    flows = inventory["flows"]
    assert isinstance(flows, list)
    flows[0] = "invalid"
    with pytest.raises(ValidationError, match="flow at index 0 must be an object"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    flow = _flow(inventory, "development-host")
    flow["location_scope"] = "local"
    with pytest.raises(ValidationError, match="location_scope has unsupported value"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))

    inventory = _inventory()
    flow = _flow(inventory, "development-host")
    flow["country_codes"] = ["japan"]
    with pytest.raises(ValidationError, match="ISO 3166-1 alpha-2"):
        check_territory_inventory(_write_inventory(tmp_path, inventory))


def test_inventory_rejects_unknown_flow_id(tmp_path: Path) -> None:
    inventory = _inventory()
    flow = _flow(inventory, "output-use")
    flow["id"] = "unexpected-flow"

    with pytest.raises(
        ValidationError,
        match="missing: output-use; unknown: unexpected-flow",
    ):
        check_territory_inventory(_write_inventory(tmp_path, inventory))
