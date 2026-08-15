"""Tests for private blind human-pairwise ballots."""

import hashlib
import json
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from h3fast.benchmarks import (
    check_human_pairwise_ballot,
    prepare_human_pairwise_ballot,
)
from h3fast.exceptions import ValidationError

FORMAL_SET = Path("benchmarks/quality/formal-quality-set.json")


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ballot = tmp_path / "ballot.json"
    assignment = tmp_path / "assignment.json"
    seed = tmp_path / "seed.txt"
    seed.write_text("test-only-secret-seed-with-32-characters", encoding="utf-8")
    seed.chmod(0o600)
    prepare_human_pairwise_ballot(
        FORMAL_SET,
        ballot,
        assignment,
        ballot_id="human-pairwise-pilot-001",
        reviewer="reviewer-001",
        randomization_seed_file=seed,
    )
    return ballot, assignment


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _cases(value: dict[str, object]) -> list[dict[str, object]]:
    cases = value["cases"]
    assert isinstance(cases, list)
    assert all(isinstance(case, dict) for case in cases)
    return cases  # type: ignore[return-value]


def _complete(
    ballot_path: Path,
    assignment_path: Path,
    *,
    candidate_wins: int = 20,
    baseline_wins: int = 20,
) -> None:
    ballot = _read(ballot_path)
    assignment = _read(assignment_path)
    assignment_cases = {str(case["case_id"]): case for case in _cases(assignment)}
    for index, ballot_case in enumerate(_cases(ballot)):
        if index >= candidate_wins + baseline_wins:
            ballot_case["selection"] = "tie"
            continue
        winner = "candidate" if index < candidate_wins else "baseline"
        assignment_case = assignment_cases[str(ballot_case["case_id"])]
        ballot_case["selection"] = "a" if assignment_case["a_source"] == winner else "b"
    ballot["status"] = "completed"
    ballot["completed_at"] = ballot["created_at"]
    _write(ballot_path, ballot)


def _refresh_assignment_digest(ballot_path: Path, assignment_path: Path) -> None:
    ballot = _read(ballot_path)
    ballot["assignment_sha256"] = hashlib.sha256(
        assignment_path.read_bytes()
    ).hexdigest()
    _write(ballot_path, ballot)


def test_prepare_creates_private_schema_valid_records(tmp_path: Path) -> None:
    ballot = tmp_path / "ballot.json"
    assignment = tmp_path / "assignment.json"
    seed = tmp_path / "seed.txt"
    seed.write_text("test-only-secret-seed-with-32-characters", encoding="utf-8")
    seed.chmod(0o600)
    report = prepare_human_pairwise_ballot(
        FORMAL_SET,
        ballot,
        assignment,
        ballot_id="human-pairwise-pilot-001",
        reviewer="reviewer-001",
        randomization_seed_file=seed,
    )
    ballot_schema = _read(Path("schemas/private-human-pairwise-ballot.schema.json"))
    assignment_schema = _read(
        Path("schemas/private-human-pairwise-assignment.schema.json")
    )

    Draft202012Validator(
        ballot_schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(_read(ballot))
    Draft202012Validator(
        assignment_schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(_read(assignment))
    assert len(_cases(_read(ballot))) == 60
    assert len(_cases(_read(assignment))) == 60
    assert stat.S_IMODE(ballot.stat().st_mode) == 0o600
    assert stat.S_IMODE(assignment.stat().st_mode) == 0o600
    serialized = ballot.read_text(encoding="utf-8") + assignment.read_text(
        encoding="utf-8"
    )
    assert "prompt_sha256" not in serialized
    assert "reference_asset" not in serialized
    assert "local_path" not in serialized
    assert report.to_dict() == {
        "schema_version": "1.0",
        "ballot_id": "human-pairwise-pilot-001",
        "case_count": 60,
        "formal_set_sha256": hashlib.sha256(FORMAL_SET.read_bytes()).hexdigest(),
        "assignment_sha256": hashlib.sha256(assignment.read_bytes()).hexdigest(),
    }


def test_complete_ballot_scores_candidate_minus_baseline_rate(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    _complete(ballot, assignment, candidate_wins=25, baseline_wins=15)
    ballot_schema = _read(Path("schemas/private-human-pairwise-ballot.schema.json"))

    Draft202012Validator(
        ballot_schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(_read(ballot))

    report = check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)

    assert report.complete is True
    assert report.case_count == 60
    assert report.candidate_wins == 25
    assert report.baseline_wins == 15
    assert report.ties == 20
    assert report.score == pytest.approx(1 / 6)
    assert set(report.to_dict()) == {
        "schema_version",
        "ballot_id",
        "complete",
        "case_count",
        "baseline_wins",
        "candidate_wins",
        "ties",
        "score",
    }


def test_pending_or_missing_observation_fails_closed(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    with pytest.raises(ValidationError, match="not completed"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)

    _complete(ballot, assignment)
    value = _read(ballot)
    _cases(value)[0]["selection"] = None
    _write(ballot, value)
    with pytest.raises(ValidationError, match="missing a valid selection"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)

    value = _read(ballot)
    _cases(value)[0]["selection"] = []
    _write(ballot, value)
    with pytest.raises(ValidationError, match="missing a valid selection"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)


def test_assignment_digest_and_commitment_tampering_fail(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    _complete(ballot, assignment)
    assignment_value = _read(assignment)
    assignment_case = _cases(assignment_value)[0]
    assignment_case["a_source"] = (
        "candidate" if assignment_case["a_source"] == "baseline" else "baseline"
    )
    _write(assignment, assignment_value)

    with pytest.raises(ValidationError, match="assignment digest"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)

    _refresh_assignment_digest(ballot, assignment)
    with pytest.raises(ValidationError, match="assignment commitment mismatch"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)


def test_case_coverage_order_and_duplicate_fail(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    _complete(ballot, assignment)
    ballot_value = _read(ballot)
    assignment_value = _read(assignment)
    _cases(ballot_value).reverse()
    _cases(assignment_value).reverse()
    _write(ballot, ballot_value)
    _write(assignment, assignment_value)
    _refresh_assignment_digest(ballot, assignment)
    with pytest.raises(ValidationError, match="fixed order"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)

    ballot, assignment = _prepare(tmp_path / "duplicate")
    _complete(ballot, assignment)
    ballot_value = _read(ballot)
    _cases(ballot_value)[1]["case_id"] = _cases(ballot_value)[0]["case_id"]
    _write(ballot, ballot_value)
    with pytest.raises(ValidationError, match="duplicate human-pairwise ballot"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)


def test_prepare_rejects_existing_or_colliding_outputs(tmp_path: Path) -> None:
    ballot, _ = _prepare(tmp_path)
    seed = tmp_path / "seed.txt"
    other_assignment = tmp_path / "other-assignment.json"
    with pytest.raises(ValidationError, match="already exists"):
        prepare_human_pairwise_ballot(
            FORMAL_SET,
            ballot,
            other_assignment,
            ballot_id="other",
            reviewer="reviewer",
            randomization_seed_file=seed,
        )
    assert not other_assignment.exists()
    with pytest.raises(ValidationError, match="different paths"):
        prepare_human_pairwise_ballot(
            FORMAL_SET,
            tmp_path / "same.json",
            tmp_path / "same.json",
            ballot_id="other",
            reviewer="reviewer",
            randomization_seed_file=seed,
        )


def test_prepare_rejects_exposed_or_short_seed_file(tmp_path: Path) -> None:
    seed = tmp_path / "seed.txt"
    seed.write_text("short", encoding="utf-8")
    seed.chmod(0o600)
    with pytest.raises(ValidationError, match="at least 32"):
        prepare_human_pairwise_ballot(
            FORMAL_SET,
            tmp_path / "ballot.json",
            tmp_path / "assignment.json",
            ballot_id="pilot",
            reviewer="reviewer",
            randomization_seed_file=seed,
        )

    seed.write_text("long-enough-test-seed-with-32-characters", encoding="utf-8")
    seed.chmod(0o644)
    with pytest.raises(ValidationError, match="group or other"):
        prepare_human_pairwise_ballot(
            FORMAL_SET,
            tmp_path / "ballot.json",
            tmp_path / "assignment.json",
            ballot_id="pilot",
            reviewer="reviewer",
            randomization_seed_file=seed,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda ballot, _assignment: ballot.update(schema_version="2.0"),
            "schema_version",
        ),
        (
            lambda _ballot, assignment: assignment.update(ballot_id="different-ballot"),
            "IDs do not match",
        ),
        (lambda ballot, _assignment: ballot.update(reviewer=""), "reviewer"),
        (
            lambda ballot, _assignment: ballot.update(created_at="not-a-time"),
            "ISO 8601",
        ),
        (
            lambda ballot, _assignment: ballot.update(
                completed_at="2020-01-01T00:00:00Z"
            ),
            "must not precede",
        ),
        (
            lambda ballot, _assignment: ballot["protocol"].update(tie_policy="discard"),
            "tie_policy",
        ),
        (
            lambda ballot, _assignment: ballot.update(formal_set_sha256="0" * 64),
            "formal-set digest",
        ),
    ],
)
def test_completed_ballot_rejects_invalid_metadata(
    tmp_path: Path, mutation, message: str
) -> None:
    ballot, assignment = _prepare(tmp_path)
    _complete(ballot, assignment)
    ballot_value = _read(ballot)
    assignment_value = _read(assignment)
    mutation(ballot_value, assignment_value)
    _write(assignment, assignment_value)
    _write(ballot, ballot_value)
    if "assignment" not in message and message != "IDs do not match":
        _refresh_assignment_digest(ballot, assignment)

    with pytest.raises(ValidationError, match=message):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)


def test_ballot_commitment_and_assignment_shape_fail_closed(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    _complete(ballot, assignment)
    ballot_value = _read(ballot)
    _cases(ballot_value)[0]["assignment_commitment_sha256"] = "0" * 64
    _write(ballot, ballot_value)
    with pytest.raises(ValidationError, match="ballot commitment mismatch"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)

    assignment_value = _read(assignment)
    _cases(assignment_value)[0]["unexpected"] = True
    _write(assignment, assignment_value)
    _refresh_assignment_digest(ballot, assignment)
    with pytest.raises(ValidationError, match="unknown fields"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)

    assignment_value = _read(assignment)
    _cases(assignment_value)[0].pop("unexpected")
    _cases(assignment_value)[0]["a_source"] = []
    _write(assignment, assignment_value)
    _refresh_assignment_digest(ballot, assignment)
    with pytest.raises(ValidationError, match="invalid a_source"):
        check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)


def test_invalid_json_and_missing_parent_fail_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError, match="not valid UTF-8 JSON"):
        check_human_pairwise_ballot(FORMAL_SET, invalid, invalid)

    seed = tmp_path / "seed.txt"
    seed.write_text("test-only-secret-seed-with-32-characters", encoding="utf-8")
    seed.chmod(0o600)
    with pytest.raises(ValidationError, match="parent is missing"):
        prepare_human_pairwise_ballot(
            FORMAL_SET,
            tmp_path / "missing" / "ballot.json",
            tmp_path / "assignment.json",
            ballot_id="pilot",
            reviewer="reviewer",
            randomization_seed_file=seed,
        )
