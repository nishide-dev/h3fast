"""Private blind human-pairwise ballot preparation and scoring."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from h3fast.benchmarks.quality_sets import check_formal_quality_set
from h3fast.exceptions import ValidationError

_BALLOT_FIELDS = frozenset(
    {
        "schema_version",
        "ballot_id",
        "status",
        "formal_set_sha256",
        "assignment_sha256",
        "reviewer",
        "created_at",
        "completed_at",
        "protocol",
        "cases",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "schema_version",
        "ballot_id",
        "formal_set_sha256",
        "randomization_seed_sha256",
        "cases",
    }
)
_PROTOCOL = {
    "blinding": "a-b-hidden-assignment-v1",
    "randomization": "per-case-sha256-order-v1",
    "missing_observation_policy": "fail",
    "tie_policy": "neutral",
    "score": "candidate-minus-baseline-win-rate-v1",
}
_PROTOCOL_FIELDS = frozenset(_PROTOCOL)
_BALLOT_CASE_FIELDS = frozenset(
    {"case_id", "presentations", "assignment_commitment_sha256", "selection"}
)
_ASSIGNMENT_CASE_FIELDS = frozenset(
    {"case_id", "a_source", "salt", "assignment_commitment_sha256"}
)
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class HumanPairwisePreparationReport:
    """Metadata for newly prepared private ballot files."""

    ballot_id: str
    case_count: int
    formal_set_sha256: str
    assignment_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return non-content preparation metadata."""
        return {
            "schema_version": "1.0",
            "ballot_id": self.ballot_id,
            "case_count": self.case_count,
            "formal_set_sha256": self.formal_set_sha256,
            "assignment_sha256": self.assignment_sha256,
        }


@dataclass(frozen=True, slots=True)
class HumanPairwiseReport:
    """Verified two-sided human-pairwise score."""

    ballot_id: str
    complete: bool
    case_count: int
    baseline_wins: int
    candidate_wins: int
    ties: int
    score: float

    def to_dict(self) -> dict[str, object]:
        """Return aggregate results without per-case reviewer decisions."""
        return {
            "schema_version": "1.0",
            "ballot_id": self.ballot_id,
            "complete": self.complete,
            "case_count": self.case_count,
            "baseline_wins": self.baseline_wins,
            "candidate_wins": self.candidate_wins,
            "ties": self.ties,
            "score": self.score,
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_commitment(
    ballot_id: str, case_id: str, a_source: str, salt: str
) -> str:
    payload = f"{ballot_id}\0{case_id}\0{a_source}\0{salt}".encode()
    return _sha256(payload)


def _load_object(path: Path, name: str) -> tuple[dict[str, object], bytes]:
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
        message = f"{name} is not valid UTF-8 JSON"
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


def _sha(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in _HEX_DIGITS for character in text):
        message = f"{name} must be a lowercase SHA-256 digest"
        raise ValidationError(message)
    return text


def _timestamp(value: object, name: str) -> datetime:
    text = _string(value, name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        message = f"{name} must be an ISO 8601 date-time"
        raise ValidationError(message) from error
    if parsed.tzinfo is None:
        message = f"{name} must include a timezone"
        raise ValidationError(message)
    return parsed


def _load_private_seed(path: Path) -> str:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        message = "private randomization seed file is missing"
        raise ValidationError(message) from error
    except (OSError, UnicodeDecodeError) as error:
        message = "private randomization seed file could not be read as UTF-8"
        raise ValidationError(message) from error
    if mode & 0o077:
        message = "private randomization seed file must not be accessible by group or other users"
        raise ValidationError(message)
    seed = raw.rstrip("\r\n")
    if len(seed) < 32:
        message = "private randomization seed must contain at least 32 characters"
        raise ValidationError(message)
    return seed


def _case_ids(formal_set: dict[str, object]) -> tuple[str, ...]:
    cases = formal_set.get("cases")
    if not isinstance(cases, list):
        message = "formal quality set cases must be an array"
        raise ValidationError(message)
    result: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            message = f"formal quality case at index {index} must be an object"
            raise ValidationError(message)
        result.append(_string(case.get("id"), f"formal quality case {index} id"))
    if not result or len(set(result)) != len(result):
        message = "formal quality set must contain distinct case IDs"
        raise ValidationError(message)
    return tuple(result)


def _write_new_private_json(path: Path, value: dict[str, object]) -> bytes:
    if path.exists():
        message = "private output already exists"
        raise ValidationError(message)
    if not path.parent.is_dir():
        message = "private output parent is missing"
        raise ValidationError(message)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        message = "private output already exists"
        raise ValidationError(message) from error
    except OSError as error:
        message = "private output could not be written"
        raise ValidationError(message) from error
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
    return data


def _validate_new_private_output(path: Path) -> None:
    if path.exists():
        message = "private output already exists"
        raise ValidationError(message)
    if not path.parent.is_dir():
        message = "private output parent is missing"
        raise ValidationError(message)


def prepare_human_pairwise_ballot(
    formal_set_path: Path,
    ballot_path: Path,
    assignment_path: Path,
    *,
    ballot_id: str,
    reviewer: str,
    randomization_seed_file: Path,
) -> HumanPairwisePreparationReport:
    """Create a pending ballot and a separate private assignment key."""
    ballot_id = _string(ballot_id, "ballot_id")
    reviewer = _string(reviewer, "reviewer")
    if ballot_path == assignment_path:
        message = "ballot and assignment outputs must be different paths"
        raise ValidationError(message)
    _validate_new_private_output(ballot_path)
    _validate_new_private_output(assignment_path)
    randomization_seed = _load_private_seed(randomization_seed_file)
    formal_set, formal_raw = _load_object(formal_set_path, "formal quality set")
    check_formal_quality_set(formal_set_path)
    formal_sha = _sha256(formal_raw)
    case_ids = _case_ids(formal_set)

    assignment_cases: list[dict[str, object]] = []
    ballot_cases: list[dict[str, object]] = []
    for case_id in case_ids:
        order_digest = hashlib.sha256(
            f"{randomization_seed}\0{ballot_id}\0order\0{case_id}".encode()
        ).digest()
        a_source = "baseline" if order_digest[0] % 2 == 0 else "candidate"
        salt = _sha256(f"{randomization_seed}\0{ballot_id}\0salt\0{case_id}".encode())
        commitment = _canonical_commitment(ballot_id, case_id, a_source, salt)
        assignment_cases.append(
            {
                "case_id": case_id,
                "a_source": a_source,
                "salt": salt,
                "assignment_commitment_sha256": commitment,
            }
        )
        ballot_cases.append(
            {
                "case_id": case_id,
                "presentations": ["a", "b"],
                "assignment_commitment_sha256": commitment,
                "selection": None,
            }
        )

    assignment: dict[str, object] = {
        "schema_version": "1.0",
        "ballot_id": ballot_id,
        "formal_set_sha256": formal_sha,
        "randomization_seed_sha256": _sha256(randomization_seed.encode()),
        "cases": assignment_cases,
    }
    assignment_data = _write_new_private_json(assignment_path, assignment)
    assignment_sha = _sha256(assignment_data)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    ballot: dict[str, object] = {
        "schema_version": "1.0",
        "ballot_id": ballot_id,
        "status": "pending",
        "formal_set_sha256": formal_sha,
        "assignment_sha256": assignment_sha,
        "reviewer": reviewer,
        "created_at": now,
        "completed_at": None,
        "protocol": dict(_PROTOCOL),
        "cases": ballot_cases,
    }
    try:
        _write_new_private_json(ballot_path, ballot)
    except ValidationError:
        assignment_path.unlink(missing_ok=True)
        raise
    return HumanPairwisePreparationReport(
        ballot_id=ballot_id,
        case_count=len(case_ids),
        formal_set_sha256=formal_sha,
        assignment_sha256=assignment_sha,
    )


def _validate_protocol(value: object) -> None:
    if not isinstance(value, dict):
        message = "human-pairwise protocol must be an object"
        raise ValidationError(message)
    _fields(value, _PROTOCOL_FIELDS, "human-pairwise protocol")
    for field, expected in _PROTOCOL.items():
        if value[field] != expected:
            message = f"human-pairwise protocol.{field} must equal {expected!r}"
            raise ValidationError(message)


def check_human_pairwise_ballot(
    formal_set_path: Path, ballot_path: Path, assignment_path: Path
) -> HumanPairwiseReport:
    """Verify and score a complete private human-pairwise ballot."""
    formal_set, formal_raw = _load_object(formal_set_path, "formal quality set")
    check_formal_quality_set(formal_set_path)
    ballot, _ = _load_object(ballot_path, "human-pairwise ballot")
    assignment, assignment_raw = _load_object(
        assignment_path, "human-pairwise assignment"
    )
    _fields(ballot, _BALLOT_FIELDS, "human-pairwise ballot")
    _fields(assignment, _ASSIGNMENT_FIELDS, "human-pairwise assignment")
    if ballot["schema_version"] != "1.0" or assignment["schema_version"] != "1.0":
        message = "human-pairwise records require schema_version '1.0'"
        raise ValidationError(message)
    ballot_id = _string(ballot["ballot_id"], "human-pairwise ballot_id")
    if assignment["ballot_id"] != ballot_id:
        message = "human-pairwise ballot and assignment IDs do not match"
        raise ValidationError(message)
    if ballot["status"] != "completed":
        message = "human-pairwise ballot is not completed"
        raise ValidationError(message)
    _string(ballot["reviewer"], "human-pairwise reviewer")
    created_at = _timestamp(ballot["created_at"], "human-pairwise created_at")
    completed_at = _timestamp(ballot["completed_at"], "human-pairwise completed_at")
    if completed_at < created_at:
        message = "human-pairwise completed_at must not precede created_at"
        raise ValidationError(message)
    _validate_protocol(ballot["protocol"])
    formal_sha = _sha256(formal_raw)
    if ballot["formal_set_sha256"] != formal_sha:
        message = "human-pairwise ballot formal-set digest does not match"
        raise ValidationError(message)
    if assignment["formal_set_sha256"] != formal_sha:
        message = "human-pairwise assignment formal-set digest does not match"
        raise ValidationError(message)
    _sha(assignment["randomization_seed_sha256"], "randomization_seed_sha256")
    if ballot["assignment_sha256"] != _sha256(assignment_raw):
        message = "human-pairwise assignment digest does not match ballot"
        raise ValidationError(message)

    expected_ids = _case_ids(formal_set)
    ballot_cases = ballot["cases"]
    assignment_cases = assignment["cases"]
    if not isinstance(ballot_cases, list) or not isinstance(assignment_cases, list):
        message = "human-pairwise cases must be arrays"
        raise ValidationError(message)
    ballot_by_id: dict[str, dict[str, object]] = {}
    for index, case in enumerate(ballot_cases):
        if not isinstance(case, dict):
            message = f"human-pairwise ballot case {index} must be an object"
            raise ValidationError(message)
        _fields(case, _BALLOT_CASE_FIELDS, f"human-pairwise ballot case {index}")
        case_id = _string(case["case_id"], f"human-pairwise ballot case {index} id")
        if case_id in ballot_by_id:
            message = f"duplicate human-pairwise ballot case: {case_id}"
            raise ValidationError(message)
        if case["presentations"] != ["a", "b"]:
            message = (
                f"human-pairwise ballot case {case_id} presentations must be ['a', 'b']"
            )
            raise ValidationError(message)
        _sha(case["assignment_commitment_sha256"], f"ballot case {case_id} commitment")
        selection = case["selection"]
        if not isinstance(selection, str) or selection not in {"a", "b", "tie"}:
            message = (
                f"human-pairwise ballot case {case_id} is missing a valid selection"
            )
            raise ValidationError(message)
        ballot_by_id[case_id] = case

    assignment_by_id: dict[str, dict[str, object]] = {}
    for index, case in enumerate(assignment_cases):
        if not isinstance(case, dict):
            message = f"human-pairwise assignment case {index} must be an object"
            raise ValidationError(message)
        _fields(
            case, _ASSIGNMENT_CASE_FIELDS, f"human-pairwise assignment case {index}"
        )
        case_id = _string(case["case_id"], f"human-pairwise assignment case {index} id")
        if case_id in assignment_by_id:
            message = f"duplicate human-pairwise assignment case: {case_id}"
            raise ValidationError(message)
        a_source = case["a_source"]
        if not isinstance(a_source, str) or a_source not in {"baseline", "candidate"}:
            message = f"human-pairwise assignment case {case_id} has invalid a_source"
            raise ValidationError(message)
        salt = _sha(case["salt"], f"assignment case {case_id} salt")
        commitment = _sha(
            case["assignment_commitment_sha256"],
            f"assignment case {case_id} commitment",
        )
        if commitment != _canonical_commitment(ballot_id, case_id, a_source, salt):
            message = f"human-pairwise assignment commitment mismatch for {case_id}"
            raise ValidationError(message)
        assignment_by_id[case_id] = case

    if tuple(ballot_by_id) != expected_ids or tuple(assignment_by_id) != expected_ids:
        message = "human-pairwise records must cover every formal case in fixed order"
        raise ValidationError(message)

    baseline_wins = 0
    candidate_wins = 0
    ties = 0
    for case_id in expected_ids:
        ballot_case = ballot_by_id[case_id]
        assignment_case = assignment_by_id[case_id]
        if (
            ballot_case["assignment_commitment_sha256"]
            != assignment_case["assignment_commitment_sha256"]
        ):
            message = f"human-pairwise ballot commitment mismatch for {case_id}"
            raise ValidationError(message)
        selection = ballot_case["selection"]
        if selection == "tie":
            ties += 1
            continue
        a_source = assignment_case["a_source"]
        selected_source = (
            a_source
            if selection == "a"
            else ("candidate" if a_source == "baseline" else "baseline")
        )
        if selected_source == "candidate":
            candidate_wins += 1
        else:
            baseline_wins += 1
    case_count = len(expected_ids)
    score = (candidate_wins - baseline_wins) / case_count
    return HumanPairwiseReport(
        ballot_id=ballot_id,
        complete=True,
        case_count=case_count,
        baseline_wins=baseline_wins,
        candidate_wins=candidate_wins,
        ties=ties,
        score=score,
    )
