"""Tests for the offline blind human-pairwise presentation runner."""

import hashlib
import json
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from h3fast.benchmarks import (
    check_human_pairwise_ballot,
    prepare_human_pairwise_ballot,
    record_human_pairwise_selection,
    stage_human_pairwise_presentation,
)
from h3fast.exceptions import ValidationError

FORMAL_SET = Path("benchmarks/quality/formal-quality-set.json")


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _cases(value: dict[str, object]) -> list[dict[str, object]]:
    cases = value["cases"]
    assert isinstance(cases, list)
    assert all(isinstance(case, dict) for case in cases)
    return cases  # type: ignore[return-value]


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
        ballot_id="runner-pilot-001",
        reviewer="reviewer-001",
        randomization_seed_file=seed,
    )
    return ballot, assignment


def _build_media(tmp_path: Path) -> Path:
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    formal_sha = hashlib.sha256(FORMAL_SET.read_bytes()).hexdigest()
    cases = []
    for case in _cases(_read(FORMAL_SET)):
        case_id = str(case["id"])
        entry: dict[str, object] = {"case_id": case_id}
        for source in ("baseline", "candidate"):
            path = media_dir / f"{case_id}-{source}.mp4"
            data = f"{source} media for {case_id}\n".encode()
            path.write_bytes(data)
            entry[source] = {
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        cases.append(entry)
    manifest = tmp_path / "media-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "formal_set_sha256": formal_sha,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    return manifest


def _stage(tmp_path: Path) -> tuple[Path, Path, Path]:
    ballot, assignment = _prepare(tmp_path)
    manifest = _build_media(tmp_path)
    staging = tmp_path / "staging"
    stage_human_pairwise_presentation(FORMAL_SET, ballot, assignment, manifest, staging)
    return ballot, assignment, staging


def test_media_manifest_matches_private_schema(tmp_path: Path) -> None:
    manifest = _build_media(tmp_path)
    schema = _read(Path("schemas/private-human-pairwise-media.schema.json"))

    Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(_read(manifest))


def test_stage_creates_blinded_presentation(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    manifest = _build_media(tmp_path)
    staging = tmp_path / "staging"

    report = stage_human_pairwise_presentation(
        FORMAL_SET, ballot, assignment, manifest, staging
    )

    assert report.ballot_id == "runner-pilot-001"
    assert report.case_count == 60
    assert report.staged_file_count == 120
    assert report.to_dict() == {
        "schema_version": "1.0",
        "ballot_id": "runner-pilot-001",
        "case_count": 60,
        "staged_file_count": 120,
    }
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700
    for case in _cases(_read(assignment)):
        case_id = str(case["case_id"])
        a_data = (staging / case_id / "a.mp4").read_bytes()
        b_data = (staging / case_id / "b.mp4").read_bytes()
        a_source = str(case["a_source"])
        b_source = "candidate" if a_source == "baseline" else "baseline"
        assert a_data == f"{a_source} media for {case_id}\n".encode()
        assert b_data == f"{b_source} media for {case_id}\n".encode()


def test_stage_index_reveals_no_sources_or_paths(tmp_path: Path) -> None:
    _ballot, _assignment, staging = _stage(tmp_path)

    html = (staging / "index.html").read_text(encoding="utf-8")

    assert "smoke-001" in html
    assert "regression-050" in html
    assert 'src="smoke-001/a.mp4"' in html
    assert "baseline" not in html
    assert "candidate" not in html
    assert str(tmp_path) not in html
    assert "http://" not in html
    assert "https://" not in html


def test_stage_rejects_media_digest_mismatch_and_cleans_up(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    manifest = _build_media(tmp_path)
    tampered = _read(manifest)
    first = _cases(tampered)[0]
    assert isinstance(first["baseline"], dict)
    Path(str(first["baseline"]["path"])).write_bytes(b"tampered media\n")
    staging = tmp_path / "staging"

    with pytest.raises(ValidationError, match="digest"):
        stage_human_pairwise_presentation(
            FORMAL_SET, ballot, assignment, manifest, staging
        )
    assert not staging.exists()


def test_stage_rejects_incomplete_or_mismatched_manifest(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    manifest = _build_media(tmp_path)
    staging = tmp_path / "staging"

    value = _read(manifest)
    removed = _cases(value).pop()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValidationError, match="every formal case"):
        stage_human_pairwise_presentation(
            FORMAL_SET, ballot, assignment, manifest, staging
        )

    value = _read(manifest)
    _cases(value).append(removed)
    value["formal_set_sha256"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValidationError, match="formal-set digest"):
        stage_human_pairwise_presentation(
            FORMAL_SET, ballot, assignment, manifest, staging
        )


def test_stage_rejects_unblinding_extension_mismatch(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    manifest = _build_media(tmp_path)
    value = _read(manifest)
    first = _cases(value)[0]
    assert isinstance(first["candidate"], dict)
    old = Path(str(first["candidate"]["path"]))
    renamed = old.with_suffix(".webm")
    old.rename(renamed)
    first["candidate"]["path"] = str(renamed)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    staging = tmp_path / "staging"

    with pytest.raises(ValidationError, match="suffix"):
        stage_human_pairwise_presentation(
            FORMAL_SET, ballot, assignment, manifest, staging
        )


def test_stage_rejects_existing_staging_or_completed_ballot(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    manifest = _build_media(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValidationError, match="already exists"):
        stage_human_pairwise_presentation(
            FORMAL_SET, ballot, assignment, manifest, staging
        )

    staging.rmdir()
    for case in _cases(_read(FORMAL_SET)):
        record_human_pairwise_selection(
            ballot, case_id=str(case["id"]), selection="tie"
        )
    with pytest.raises(ValidationError, match="not pending"):
        stage_human_pairwise_presentation(
            FORMAL_SET, ballot, assignment, manifest, staging
        )


def test_record_selection_updates_pending_ballot(tmp_path: Path) -> None:
    ballot, _assignment = _prepare(tmp_path)

    report = record_human_pairwise_selection(ballot, case_id="smoke-001", selection="a")

    assert report.ballot_id == "runner-pilot-001"
    assert report.case_id == "smoke-001"
    assert report.recorded_count == 1
    assert report.case_count == 60
    assert report.completed is False
    assert report.to_dict() == {
        "schema_version": "1.0",
        "ballot_id": "runner-pilot-001",
        "case_id": "smoke-001",
        "recorded_count": 1,
        "case_count": 60,
        "completed": False,
    }
    value = _read(ballot)
    assert value["status"] == "pending"
    assert value["completed_at"] is None
    assert _cases(value)[0]["selection"] == "a"
    assert stat.S_IMODE(ballot.stat().st_mode) == 0o600


def test_record_final_selection_completes_valid_ballot(tmp_path: Path) -> None:
    ballot, assignment = _prepare(tmp_path)
    ballot_schema = _read(Path("schemas/private-human-pairwise-ballot.schema.json"))

    last = None
    for index, case in enumerate(_cases(_read(FORMAL_SET))):
        selection = ("a", "b", "tie")[index % 3]
        last = record_human_pairwise_selection(
            ballot, case_id=str(case["id"]), selection=selection
        )

    assert last is not None
    assert last.completed is True
    assert last.recorded_count == 60
    value = _read(ballot)
    assert value["status"] == "completed"
    assert isinstance(value["completed_at"], str)
    Draft202012Validator(
        ballot_schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(value)
    report = check_human_pairwise_ballot(FORMAL_SET, ballot, assignment)
    assert report.complete is True
    assert report.ties == 20


def test_record_rejects_unknown_case_and_invalid_selection(tmp_path: Path) -> None:
    ballot, _assignment = _prepare(tmp_path)

    with pytest.raises(ValidationError, match="unknown case"):
        record_human_pairwise_selection(ballot, case_id="smoke-999", selection="a")
    with pytest.raises(ValidationError, match="selection"):
        record_human_pairwise_selection(ballot, case_id="smoke-001", selection="c")


def test_record_requires_overwrite_to_change_a_selection(tmp_path: Path) -> None:
    ballot, _assignment = _prepare(tmp_path)
    record_human_pairwise_selection(ballot, case_id="smoke-001", selection="a")

    with pytest.raises(ValidationError, match="already recorded"):
        record_human_pairwise_selection(ballot, case_id="smoke-001", selection="b")

    report = record_human_pairwise_selection(
        ballot, case_id="smoke-001", selection="b", overwrite=True
    )
    assert report.recorded_count == 1
    assert _cases(_read(ballot))[0]["selection"] == "b"


def test_record_fails_closed_when_ballot_update_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ballot, _assignment = _prepare(tmp_path)

    def raise_oserror(*_args: object, **_kwargs: object) -> object:
        message = "temporary file could not be created"
        raise OSError(message)

    monkeypatch.setattr(
        "h3fast.benchmarks.human_pairwise_runner.tempfile.NamedTemporaryFile",
        raise_oserror,
    )
    with pytest.raises(ValidationError, match="could not be updated"):
        record_human_pairwise_selection(ballot, case_id="smoke-001", selection="a")
    assert _cases(_read(ballot))[0]["selection"] is None


def test_record_rejects_completed_ballot(tmp_path: Path) -> None:
    ballot, _assignment = _prepare(tmp_path)
    for case in _cases(_read(FORMAL_SET)):
        record_human_pairwise_selection(
            ballot, case_id=str(case["id"]), selection="tie"
        )

    with pytest.raises(ValidationError, match="not pending"):
        record_human_pairwise_selection(
            ballot, case_id="smoke-001", selection="a", overwrite=True
        )
