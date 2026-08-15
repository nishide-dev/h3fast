"""Tests for formal quality-set readiness metadata."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from h3fast.benchmarks.quality_sets import check_formal_quality_set
from h3fast.exceptions import ValidationError

RECORD_PATH = Path("benchmarks/quality/formal-quality-set.json")


def _record() -> dict[str, object]:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def _write_record(tmp_path: Path, record: object) -> Path:
    path = tmp_path / "formal-quality-set.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(split: str, index: int) -> dict[str, object]:
    task = ("t2va", "fl2va", "ref2va")[index % 3]
    if task == "t2va":
        modalities = ["none"]
        assets: list[str] = []
    elif task == "fl2va":
        modalities = ["image"]
        assets = [_digest(f"{split}-{index}-image")]
    else:
        modalities = [["video"], ["audio"], ["mixed"]][(index // 3) % 3]
        assets = [_digest(f"{split}-{index}-reference")]
    return {
        "id": f"{split}-{index:03d}",
        "split": split,
        "prompt_sha256": _digest(f"{split}-{index}-prompt"),
        "seed": 1000 + index,
        "task": task,
        "duration_seconds": (4, 5, 10, 15)[index % 4],
        "aspect_ratio": ("landscape", "square", "portrait")[index % 3],
        "languages": [("ja", "en")[index % 2]],
        "subject_tags": [
            ("face", "hands", "text", "product", "multiple-people")[index % 5]
        ],
        "motion_tags": [("dynamic", "static", "camera-movement")[index % 3]],
        "audio_tags": [("dialogue", "environment", "music", "near-silent")[index % 4]],
        "reference_modalities": modalities,
        "reference_asset_sha256s": assets,
        "rights_status": "approved",
        "rights_evidence": [f"https://example.test/rights/{split}-{index:03d}"],
        "notes": "Synthetic metadata used only to validate the contract.",
    }


def _approved_record() -> dict[str, object]:
    record = copy.deepcopy(_record())
    record["status"] = "approved"
    selection = record["selection"]
    assert isinstance(selection, dict)
    selection.update(
        {
            "method": "Stratified selection version 1",
            "registry_uri": "https://example.test/quality/cases-v1.json",
            "registry_sha256": _digest("registry-v1"),
            "exclusions_reviewed": True,
            "known_failures_reviewed": True,
        }
    )
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    for role, approval in approvals.items():
        assert isinstance(role, str)
        assert isinstance(approval, dict)
        approval.update(
            {
                "state": "approved",
                "owner": f"{role}-owner",
                "approved_at": "2026-08-20T00:00:00+09:00",
                "evidence": [f"https://example.test/approvals/{role}"],
            }
        )
    metrics = record["metrics"]
    assert isinstance(metrics, list)
    for metric in metrics:
        assert isinstance(metric, dict)
        metric.update(
            {
                "state": "approved",
                "owner": "quality-owner",
                "implementation": f"{metric['family']}-implementation",
                "version": "1.0.0",
                "budget": "baseline envelope version 1",
                "evidence": [f"https://example.test/metrics/{metric['family']}"],
            }
        )
    record["cases"] = [
        *(_case("smoke", index) for index in range(1, 11)),
        *(_case("regression", index) for index in range(1, 51)),
    ]
    return record


def _cases(record: dict[str, object]) -> list[dict[str, object]]:
    value = record["cases"]
    assert isinstance(value, list)
    assert all(isinstance(case, dict) for case in value)
    return value  # type: ignore[return-value]


def test_committed_formal_quality_set_is_explicitly_incomplete() -> None:
    report = check_formal_quality_set(RECORD_PATH)

    assert report.status == "incomplete"
    assert report.ready is False
    assert report.smoke_cases == 0
    assert report.regression_cases == 0
    assert "approval:quality_owner:unassigned" in report.blockers
    assert "cases:smoke-count:0/10" in report.blockers
    assert "cases:regression-count:0/50" in report.blockers
    assert "coverage:languages:distinct:0/2" in report.blockers


def test_approved_formal_quality_set_is_ready(tmp_path: Path) -> None:
    report = check_formal_quality_set(_write_record(tmp_path, _approved_record()))

    assert report.ready is True
    assert report.smoke_cases == 10
    assert report.regression_cases == 50
    assert report.blockers == ()


def test_approved_status_rejects_remaining_blockers(tmp_path: Path) -> None:
    record = _approved_record()
    _cases(record).pop()

    with pytest.raises(ValidationError, match="claims approval while blockers remain"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_incomplete_status_rejects_completed_record(tmp_path: Path) -> None:
    record = _approved_record()
    record["status"] = "incomplete"

    with pytest.raises(ValidationError, match="incomplete even though every gate"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_requires_smoke_and_regression_counts(tmp_path: Path) -> None:
    record = _approved_record()
    record["status"] = "incomplete"
    record["cases"] = _cases(record)[:9]

    report = check_formal_quality_set(_write_record(tmp_path, record))

    assert "cases:smoke-count:9/10" in report.blockers
    assert "cases:regression-count:0/50" in report.blockers


def test_formal_set_reports_missing_coverage(tmp_path: Path) -> None:
    record = _approved_record()
    record["status"] = "incomplete"
    for case in _cases(record):
        case["languages"] = ["ja"]

    report = check_formal_quality_set(_write_record(tmp_path, record))

    assert "coverage:languages:distinct:1/2" in report.blockers


def test_formal_set_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    record = _approved_record()
    cases = _cases(record)
    cases[1]["id"] = cases[0]["id"]
    cases[1]["split"] = cases[0]["split"]

    with pytest.raises(ValidationError, match="duplicate quality case id"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_requires_prompt_digest_and_rights_review(tmp_path: Path) -> None:
    record = _approved_record()
    record["status"] = "incomplete"
    case = _cases(record)[0]
    case["prompt_sha256"] = None
    case["rights_status"] = "unreviewed"
    case["rights_evidence"] = []

    report = check_formal_quality_set(_write_record(tmp_path, record))

    assert "case:smoke-001:prompt-digest-missing" in report.blockers
    assert "case:smoke-001:rights-unreviewed" in report.blockers


def test_formal_set_rejects_reference_contract_mismatches(tmp_path: Path) -> None:
    record = _approved_record()
    case = _cases(record)[0]
    assert case["task"] == "fl2va"
    case["reference_modalities"] = ["none"]

    with pytest.raises(
        ValidationError, match=r"referenced case .* requires asset digests"
    ):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _approved_record()
    case = next(case for case in _cases(record) if case["task"] == "t2va")
    case["reference_asset_sha256s"] = [_digest("unexpected")]
    with pytest.raises(ValidationError, match=r"T2VA case .* must use only"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_invalid_selection_evidence(tmp_path: Path) -> None:
    record = _record()
    selection = record["selection"]
    assert isinstance(selection, dict)
    selection["registry_uri"] = "file:///private/cases.json"

    with pytest.raises(ValidationError, match="must be an HTTPS URL"):
        check_formal_quality_set(_write_record(tmp_path, record))

    selection["registry_uri"] = "https://example.test/cases.json"
    selection["registry_sha256"] = "invalid"
    with pytest.raises(ValidationError, match="must be a lowercase SHA-256"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_inconsistent_approval(tmp_path: Path) -> None:
    record = _record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    approval = approvals["quality_owner"]
    assert isinstance(approval, dict)
    approval["owner"] = "quality-owner"

    with pytest.raises(ValidationError, match="unassigned approval"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_duplicate_metric_family(tmp_path: Path) -> None:
    record = _record()
    metrics = record["metrics"]
    assert isinstance(metrics, list)
    assert isinstance(metrics[0], dict)
    assert isinstance(metrics[1], dict)
    metrics[1]["family"] = metrics[0]["family"]

    with pytest.raises(ValidationError, match="duplicate metric family"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_unassigned_metric_details(tmp_path: Path) -> None:
    record = _record()
    metrics = record["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[0]
    assert isinstance(metric, dict)
    metric["owner"] = "quality-owner"

    with pytest.raises(ValidationError, match=r"unassigned metric .* cannot contain"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_modified_requirements(tmp_path: Path) -> None:
    record = _record()
    requirements = record["requirements"]
    assert isinstance(requirements, dict)
    requirements["smoke_min_cases"] = 1

    with pytest.raises(ValidationError, match="smoke_min_cases must equal 10"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_missing_invalid_and_nonobject_json(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="record is missing"):
        check_formal_quality_set(tmp_path / "missing.json")

    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError, match="not valid JSON"):
        check_formal_quality_set(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValidationError, match="root must be an object"):
        check_formal_quality_set(path)


def test_formal_set_rejects_missing_and_unknown_top_level_fields(
    tmp_path: Path,
) -> None:
    record = _record()
    record.pop("updated_at")
    with pytest.raises(ValidationError, match="missing required fields: updated_at"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    record["unexpected"] = True
    with pytest.raises(ValidationError, match="unknown fields: unexpected"):
        check_formal_quality_set(_write_record(tmp_path, record))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0", "unsupported formal quality-set schema_version"),
        ("set_id", "other", "unsupported formal quality set_id"),
        ("status", "ready", "status has unsupported value"),
        ("updated_at", "tomorrow", "must be an ISO 8601 date"),
        ("source_issue", "http://example.test/14", "must be an HTTPS URL"),
    ],
)
def test_formal_set_rejects_invalid_top_level_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    record = _record()
    record[field] = value

    with pytest.raises(ValidationError, match=message):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_invalid_requirement_shapes(tmp_path: Path) -> None:
    record = _record()
    record["requirements"] = []
    with pytest.raises(ValidationError, match="requirements must be an object"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    requirements = record["requirements"]
    assert isinstance(requirements, dict)
    requirements["required_tasks"] = "t2va"
    with pytest.raises(ValidationError, match="required_tasks must be an array"):
        check_formal_quality_set(_write_record(tmp_path, record))

    requirements["required_tasks"] = [["t2va"]]
    with pytest.raises(ValidationError, match="contains an unsupported value"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_invalid_selection_shapes(tmp_path: Path) -> None:
    record = _record()
    record["selection"] = []
    with pytest.raises(ValidationError, match="selection must be an object"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    selection = record["selection"]
    assert isinstance(selection, dict)
    selection["exclusions_reviewed"] = "yes"
    with pytest.raises(ValidationError, match="exclusions_reviewed must be boolean"):
        check_formal_quality_set(_write_record(tmp_path, record))

    selection["exclusions_reviewed"] = False
    selection["known_failures_reviewed"] = 1
    with pytest.raises(
        ValidationError, match="known_failures_reviewed must be boolean"
    ):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_invalid_approval_roles_and_states(tmp_path: Path) -> None:
    record = _record()
    record["approvals"] = []
    with pytest.raises(ValidationError, match="approvals must be an object"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    approvals.pop("quality_owner")
    with pytest.raises(ValidationError, match="missing: quality_owner"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    approvals = record["approvals"]
    assert isinstance(approvals, dict)
    approval = approvals["quality_owner"]
    assert isinstance(approval, dict)
    approval.update({"state": "pending", "owner": "quality-owner"})
    report = check_formal_quality_set(_write_record(tmp_path, record))
    assert "approval:quality_owner:pending" in report.blockers

    approval["approved_at"] = "2026-08-20T00:00:00"
    with pytest.raises(ValidationError, match="must include a UTC offset"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_invalid_metric_shapes_and_states(tmp_path: Path) -> None:
    record = _record()
    record["metrics"] = {}
    with pytest.raises(ValidationError, match="metrics must be an array"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    metrics = record["metrics"]
    assert isinstance(metrics, list)
    metrics[0] = "invalid"
    with pytest.raises(ValidationError, match="metric at index 0 must be an object"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    metrics = record["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[0]
    assert isinstance(metric, dict)
    metric.update({"state": "planned", "owner": None})
    with pytest.raises(ValidationError, match=r"planned metric .* requires an owner"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    metrics = record["metrics"]
    assert isinstance(metrics, list)
    metrics.pop()
    with pytest.raises(ValidationError, match="metrics do not match required families"):
        check_formal_quality_set(_write_record(tmp_path, record))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "case-1", "unsupported format"),
        ("seed", -1, "seed must be a non-negative integer"),
        ("duration_seconds", "5", "duration_seconds must be numeric"),
        ("duration_seconds", 6, "unsupported duration_seconds"),
        ("languages", ["Japanese"], "supported BCP 47"),
        ("subject_tags", ["landscape"], "contains unsupported values"),
    ],
)
def test_formal_set_rejects_invalid_case_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    record = _approved_record()
    _cases(record)[0][field] = value

    with pytest.raises(ValidationError, match=message):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_invalid_case_shapes_and_rights(tmp_path: Path) -> None:
    record = _approved_record()
    _cases(record)[0] = "invalid"  # type: ignore[assignment]
    with pytest.raises(ValidationError, match="case at index 0 must be an object"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _approved_record()
    case = _cases(record)[0]
    case["split"] = "regression"
    with pytest.raises(ValidationError, match="id does not match split"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _approved_record()
    case = _cases(record)[0]
    case["rights_status"] = "unreviewed"
    with pytest.raises(ValidationError, match=r"unreviewed case .* cannot contain"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _approved_record()
    case = _cases(record)[0]
    case["rights_evidence"] = []
    with pytest.raises(ValidationError, match=r"approved case .* requires rights"):
        check_formal_quality_set(_write_record(tmp_path, record))


def test_formal_set_rejects_nonarray_cases_and_empty_limitations(
    tmp_path: Path,
) -> None:
    record = _record()
    record["cases"] = {}
    with pytest.raises(ValidationError, match="cases must be an array"):
        check_formal_quality_set(_write_record(tmp_path, record))

    record = _record()
    record["limitations"] = []
    with pytest.raises(ValidationError, match="limitations must contain at least"):
        check_formal_quality_set(_write_record(tmp_path, record))
