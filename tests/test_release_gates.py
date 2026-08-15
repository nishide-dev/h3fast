"""Tests for fail-closed release readiness records."""

import copy
import json
from pathlib import Path

import pytest

from h3fast.exceptions import ValidationError
from h3fast.release import check_release_gate

RECORD_PATH = Path("compliance/release-gates/initial-runtime.json")


def _record() -> dict[str, object]:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def _write_record(tmp_path: Path, record: dict[str, object]) -> Path:
    path = tmp_path / "release-gate.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _approved_record() -> dict[str, object]:
    record = copy.deepcopy(_record())
    record["status"] = "approved"
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    for role, approval in approvals.items():
        assert isinstance(role, str)
        assert isinstance(approval, dict)
        approval.update(
            {
                "state": "approved",
                "owner": f"{role}-owner",
                "deadline": "2026-08-31",
                "approved_at": "2026-08-20T00:00:00+09:00",
                "evidence": [f"https://example.test/approvals/{role}"],
            }
        )
    checks = record["checks"]
    assert isinstance(checks, list)
    for check in checks:
        assert isinstance(check, dict)
        check["status"] = "passed"
        check["owner"] = "release-owner"
        check["evidence"] = ["https://example.test/evidence"]
    return record


def test_committed_initial_runtime_gate_is_explicitly_blocked() -> None:
    report = check_release_gate(RECORD_PATH)

    assert report.status == "blocked"
    assert report.ready is False
    assert "approval:legal_reviewer:unassigned" in report.blockers
    assert "check:territory-approval:blocked" in report.blockers
    assert "check:territory-approval:owner-unassigned" in report.blockers


def test_formal_quality_release_check_rejects_incomplete_record(
    tmp_path: Path,
) -> None:
    record = _record()
    checks = record["checks"]
    assert isinstance(checks, list)
    formal = next(
        check
        for check in checks
        if isinstance(check, dict) and check.get("id") == "formal-quality-set"
    )
    assert isinstance(formal, dict)
    formal["status"] = "passed"

    with pytest.raises(ValidationError, match="committed record is incomplete"):
        check_release_gate(_write_record(tmp_path, record))


def test_fully_approved_release_gate_is_ready(tmp_path: Path) -> None:
    report = check_release_gate(_write_record(tmp_path, _approved_record()))

    assert report.status == "approved"
    assert report.ready is True
    assert report.blockers == ()


def test_approved_release_gate_rejects_remaining_blocker(tmp_path: Path) -> None:
    record = _approved_record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    legal = approvals["legal_reviewer"]
    assert isinstance(legal, dict)
    legal.update(
        {
            "state": "pending",
            "owner": "legal-owner",
            "approved_at": None,
            "evidence": [],
        }
    )

    with pytest.raises(ValidationError, match="claims approval while blockers remain"):
        check_release_gate(_write_record(tmp_path, record))


def test_release_gate_rejects_missing_required_check(tmp_path: Path) -> None:
    record = _record()
    checks = record["checks"]
    assert isinstance(checks, list)
    checks.pop()

    with pytest.raises(ValidationError, match="missing: public-benchmark"):
        check_release_gate(_write_record(tmp_path, record))


def test_release_gate_rejects_duplicate_check(tmp_path: Path) -> None:
    record = _record()
    checks = record["checks"]
    assert isinstance(checks, list)
    assert isinstance(checks[0], dict)
    assert isinstance(checks[1], dict)
    checks[1]["id"] = checks[0]["id"]

    with pytest.raises(ValidationError, match="duplicate release check id"):
        check_release_gate(_write_record(tmp_path, record))


def test_release_gate_rejects_pending_approval_without_owner(tmp_path: Path) -> None:
    record = _record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    legal = approvals["legal_reviewer"]
    assert isinstance(legal, dict)
    legal["state"] = "pending"

    with pytest.raises(ValidationError, match="requires owner and deadline"):
        check_release_gate(_write_record(tmp_path, record))


def test_release_gate_rejects_naive_approval_timestamp(tmp_path: Path) -> None:
    record = _approved_record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    legal = approvals["legal_reviewer"]
    assert isinstance(legal, dict)
    legal["approved_at"] = "2026-08-20T00:00:00"

    with pytest.raises(ValidationError, match="must include a UTC offset"):
        check_release_gate(_write_record(tmp_path, record))


def test_release_gate_rejects_missing_and_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="record is missing"):
        check_release_gate(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError, match="not valid JSON"):
        check_release_gate(invalid)

    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValidationError, match="root must be an object"):
        check_release_gate(invalid)


def test_release_gate_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    record = _record()
    record["unexpected"] = True

    with pytest.raises(ValidationError, match="unknown fields: unexpected"):
        check_release_gate(_write_record(tmp_path, record))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0", "unsupported release gate schema_version"),
        ("release_id", "nightly", "unsupported release_id"),
        ("status", "ready", "unsupported release status"),
        ("updated_at", "tomorrow", "must be an ISO 8601 date"),
        ("updated_at", None, "must be an ISO 8601 date"),
        ("source_issue", "", "source_issue must be a non-empty string"),
        ("source_issue", "issue-11", "source_issue must be an HTTPS URL"),
    ],
)
def test_release_gate_rejects_invalid_identity_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    record = _record()
    record[field] = value

    with pytest.raises(ValidationError, match=message):
        check_release_gate(_write_record(tmp_path, record))


def test_release_gate_rejects_approval_role_mismatch(tmp_path: Path) -> None:
    record = _record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    approvals["unexpected"] = approvals.pop("schema_owner")

    with pytest.raises(
        ValidationError, match="missing: schema_owner; unknown: unexpected"
    ):
        check_release_gate(_write_record(tmp_path, record))


def test_release_gate_rejects_invalid_approval_states(tmp_path: Path) -> None:
    record = _record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    legal = approvals["legal_reviewer"]
    assert isinstance(legal, dict)
    legal["state"] = "accepted"
    with pytest.raises(ValidationError, match="unsupported state"):
        check_release_gate(_write_record(tmp_path, record))

    record = _record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    legal = approvals["legal_reviewer"]
    assert isinstance(legal, dict)
    legal["owner"] = "someone"
    with pytest.raises(ValidationError, match="unassigned approval"):
        check_release_gate(_write_record(tmp_path, record))

    record = _record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    legal = approvals["legal_reviewer"]
    assert isinstance(legal, dict)
    legal["deadline"] = None
    with pytest.raises(ValidationError, match="requires a decision deadline"):
        check_release_gate(_write_record(tmp_path, record))


def test_release_gate_rejects_invalid_check_states(tmp_path: Path) -> None:
    record = _record()
    checks = record["checks"]
    assert isinstance(checks, list)
    first = checks[0]
    assert isinstance(first, dict)
    first["status"] = "complete"
    with pytest.raises(ValidationError, match="unsupported status"):
        check_release_gate(_write_record(tmp_path, record))

    record = _record()
    checks = record["checks"]
    assert isinstance(checks, list)
    first = checks[0]
    assert isinstance(first, dict)
    first["owner"] = None
    with pytest.raises(ValidationError, match="resolved release check"):
        check_release_gate(_write_record(tmp_path, record))

    record = _record()
    checks = record["checks"]
    assert isinstance(checks, list)
    blocked = checks[1]
    assert isinstance(blocked, dict)
    blocked["deadline"] = None
    with pytest.raises(ValidationError, match="requires a disposition deadline"):
        check_release_gate(_write_record(tmp_path, record))

    record = _record()
    checks = record["checks"]
    assert isinstance(checks, list)
    blocked = checks[1]
    assert isinstance(blocked, dict)
    blocked["evidence"] = []
    with pytest.raises(ValidationError, match="evidence must contain"):
        check_release_gate(_write_record(tmp_path, record))


def test_blocked_release_gate_rejects_no_remaining_blockers(tmp_path: Path) -> None:
    record = _approved_record()
    record["status"] = "blocked"

    with pytest.raises(ValidationError, match="blocked even though every gate"):
        check_release_gate(_write_record(tmp_path, record))


def test_not_applicable_check_can_be_approved_with_evidence(tmp_path: Path) -> None:
    record = _approved_record()
    checks = record["checks"]
    assert isinstance(checks, list)
    first = checks[0]
    assert isinstance(first, dict)
    first["status"] = "not_applicable"

    assert check_release_gate(_write_record(tmp_path, record)).ready is True
