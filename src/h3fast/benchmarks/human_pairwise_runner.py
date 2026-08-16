"""Offline blind A/B presentation staging and selection recording."""

from __future__ import annotations

import contextlib
import hashlib
import html
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from h3fast.benchmarks.human_pairwise import (
    _ASSIGNMENT_CASE_FIELDS,
    _ASSIGNMENT_FIELDS,
    _BALLOT_CASE_FIELDS,
    _BALLOT_FIELDS,
    _canonical_commitment,
    _case_ids,
    _fields,
    _load_object,
    _sha,
    _sha256,
    _string,
    _validate_protocol,
)
from h3fast.benchmarks.quality_sets import check_formal_quality_set
from h3fast.exceptions import ValidationError

_MEDIA_MANIFEST_FIELDS = frozenset({"schema_version", "formal_set_sha256", "cases"})
_MEDIA_CASE_FIELDS = frozenset({"case_id", "baseline", "candidate"})
_MEDIA_SOURCE_FIELDS = frozenset({"path", "sha256"})
_SELECTIONS = frozenset({"a", "b", "tie"})
_SUFFIX_PATTERN = re.compile(r"\.[A-Za-z0-9]+")
_COPY_CHUNK_BYTES = 1 << 20


def _require_private_file(path: Path, name: str) -> None:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError as error:
        message = f"{name} is missing"
        raise ValidationError(message) from error
    except OSError as error:
        message = f"{name} could not be read"
        raise ValidationError(message) from error
    if mode & 0o077:
        message = f"{name} must not be accessible by group or other users"
        raise ValidationError(message)


@dataclass(frozen=True, slots=True)
class HumanPairwiseStagingReport:
    """Metadata for a newly staged blind presentation directory."""

    ballot_id: str
    case_count: int
    staged_file_count: int

    def to_dict(self) -> dict[str, object]:
        """Return non-content staging metadata."""
        return {
            "schema_version": "1.0",
            "ballot_id": self.ballot_id,
            "case_count": self.case_count,
            "staged_file_count": self.staged_file_count,
        }


@dataclass(frozen=True, slots=True)
class HumanPairwiseRecordReport:
    """Progress metadata for one recorded reviewer selection."""

    ballot_id: str
    case_id: str
    recorded_count: int
    case_count: int
    completed: bool

    def to_dict(self) -> dict[str, object]:
        """Return recording progress without any selection values."""
        return {
            "schema_version": "1.0",
            "ballot_id": self.ballot_id,
            "case_id": self.case_id,
            "recorded_count": self.recorded_count,
            "case_count": self.case_count,
            "completed": self.completed,
        }


def _validate_ballot_case(
    case: object, index: int
) -> tuple[dict[str, object], str, str]:
    if not isinstance(case, dict):
        message = f"human-pairwise ballot case {index} must be an object"
        raise ValidationError(message)
    _fields(case, _BALLOT_CASE_FIELDS, f"human-pairwise ballot case {index}")
    case_id = _string(case["case_id"], f"human-pairwise ballot case {index} id")
    commitment = _sha(
        case["assignment_commitment_sha256"], f"ballot case {case_id} commitment"
    )
    selection = case["selection"]
    if selection is not None and (
        not isinstance(selection, str) or selection not in _SELECTIONS
    ):
        message = f"human-pairwise ballot case {case_id} has an invalid selection"
        raise ValidationError(message)
    return case, case_id, commitment


def _validate_pending_ballot(
    ballot: dict[str, object], expected_ids: tuple[str, ...], formal_sha: str
) -> tuple[str, dict[str, str]]:
    _fields(ballot, _BALLOT_FIELDS, "human-pairwise ballot")
    if ballot["schema_version"] != "1.0":
        message = "human-pairwise ballot requires schema_version '1.0'"
        raise ValidationError(message)
    ballot_id = _string(ballot["ballot_id"], "human-pairwise ballot_id")
    if ballot["status"] != "pending":
        message = "human-pairwise ballot is not pending"
        raise ValidationError(message)
    _validate_protocol(ballot["protocol"])
    if ballot["formal_set_sha256"] != formal_sha:
        message = "human-pairwise ballot formal-set digest does not match"
        raise ValidationError(message)
    cases = ballot["cases"]
    if not isinstance(cases, list):
        message = "human-pairwise ballot cases must be an array"
        raise ValidationError(message)
    commitments: dict[str, str] = {}
    for index, case in enumerate(cases):
        _, case_id, commitment = _validate_ballot_case(case, index)
        if case_id in commitments:
            message = f"duplicate human-pairwise ballot case: {case_id}"
            raise ValidationError(message)
        commitments[case_id] = commitment
    if tuple(commitments) != expected_ids:
        message = "human-pairwise ballot must cover every formal case in fixed order"
        raise ValidationError(message)
    return ballot_id, commitments


def _validate_assignment(
    assignment: dict[str, object],
    assignment_raw: bytes,
    ballot: dict[str, object],
    *,
    ballot_id: str,
    ballot_commitments: dict[str, str],
    expected_ids: tuple[str, ...],
    formal_sha: str,
) -> dict[str, str]:
    _fields(assignment, _ASSIGNMENT_FIELDS, "human-pairwise assignment")
    if assignment["schema_version"] != "1.0":
        message = "human-pairwise assignment requires schema_version '1.0'"
        raise ValidationError(message)
    if assignment["ballot_id"] != ballot_id:
        message = "human-pairwise ballot and assignment IDs do not match"
        raise ValidationError(message)
    if assignment["formal_set_sha256"] != formal_sha:
        message = "human-pairwise assignment formal-set digest does not match"
        raise ValidationError(message)
    if ballot["assignment_sha256"] != _sha256(assignment_raw):
        message = "human-pairwise assignment digest does not match ballot"
        raise ValidationError(message)
    cases = assignment["cases"]
    if not isinstance(cases, list):
        message = "human-pairwise assignment cases must be an array"
        raise ValidationError(message)
    a_sources: dict[str, str] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            message = f"human-pairwise assignment case {index} must be an object"
            raise ValidationError(message)
        _fields(
            case, _ASSIGNMENT_CASE_FIELDS, f"human-pairwise assignment case {index}"
        )
        case_id = _string(case["case_id"], f"human-pairwise assignment case {index} id")
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
        if ballot_commitments.get(case_id) != commitment:
            message = f"human-pairwise ballot commitment mismatch for {case_id}"
            raise ValidationError(message)
        a_sources[case_id] = a_source
    if tuple(a_sources) != expected_ids:
        message = (
            "human-pairwise assignment must cover every formal case in fixed order"
        )
        raise ValidationError(message)
    return a_sources


def _validate_media_source(
    case: dict[str, object], source: str, case_id: str, manifest_dir: Path
) -> tuple[Path, str]:
    value = case[source]
    if not isinstance(value, dict):
        message = f"media manifest case {case_id} {source} must be an object"
        raise ValidationError(message)
    _fields(value, _MEDIA_SOURCE_FIELDS, f"media manifest case {case_id} {source}")
    path = Path(_string(value["path"], f"media manifest case {case_id} {source} path"))
    if not path.is_absolute():
        path = manifest_dir / path
    digest = _sha(value["sha256"], f"media manifest case {case_id} {source} sha256")
    return path, digest


def _validate_media_manifest(
    manifest: dict[str, object],
    manifest_dir: Path,
    expected_ids: tuple[str, ...],
    formal_sha: str,
) -> dict[str, dict[str, tuple[Path, str]]]:
    _fields(manifest, _MEDIA_MANIFEST_FIELDS, "human-pairwise media manifest")
    if manifest["schema_version"] != "1.0":
        message = "human-pairwise media manifest requires schema_version '1.0'"
        raise ValidationError(message)
    if manifest["formal_set_sha256"] != formal_sha:
        message = "human-pairwise media manifest formal-set digest does not match"
        raise ValidationError(message)
    cases = manifest["cases"]
    if not isinstance(cases, list):
        message = "human-pairwise media manifest cases must be an array"
        raise ValidationError(message)
    media: dict[str, dict[str, tuple[Path, str]]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            message = f"media manifest case {index} must be an object"
            raise ValidationError(message)
        _fields(case, _MEDIA_CASE_FIELDS, f"media manifest case {index}")
        case_id = _string(case["case_id"], f"media manifest case {index} id")
        if case_id in media:
            message = f"duplicate media manifest case: {case_id}"
            raise ValidationError(message)
        baseline = _validate_media_source(case, "baseline", case_id, manifest_dir)
        candidate = _validate_media_source(case, "candidate", case_id, manifest_dir)
        if baseline[0].suffix != candidate[0].suffix or not _SUFFIX_PATTERN.fullmatch(
            baseline[0].suffix
        ):
            message = (
                f"media pair for {case_id} must share one alphanumeric file suffix"
            )
            raise ValidationError(message)
        media[case_id] = {"baseline": baseline, "candidate": candidate}
    if tuple(media) != expected_ids:
        message = (
            "human-pairwise media manifest must cover every formal case in fixed order"
        )
        raise ValidationError(message)
    return media


def _copy_verified_media(source: Path, digest: str, target: Path, name: str) -> None:
    hasher = hashlib.sha256()
    try:
        with source.open("rb") as reader, target.open("xb") as writer:
            while chunk := reader.read(_COPY_CHUNK_BYTES):
                hasher.update(chunk)
                writer.write(chunk)
        target.chmod(0o600)
    except OSError as error:
        message = f"{name} could not be copied"
        raise ValidationError(message) from error
    if hasher.hexdigest() != digest:
        message = f"{name} digest does not match the media manifest"
        raise ValidationError(message)


def _presentation_index(ballot_id: str, staged: list[tuple[str, str]]) -> str:
    title = html.escape(f"H3Fast human-pairwise ballot {ballot_id}")
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 2rem; }",
        "section { border-top: 1px solid #999; padding: 1rem 0; }",
        "video { max-width: 46%; margin-right: 2%; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        (
            "<p>Watch A and B for each case, then record a, b, or tie with"
            " <code>h3fast benchmark record-human-pairwise</code>.</p>"
        ),
    ]
    for case_id, suffix in staged:
        escaped = html.escape(case_id)
        lines.append(f'<section id="{escaped}">')
        lines.append(f"<h2>{escaped}</h2>")
        lines.extend(
            f"<figure><figcaption>{presentation.upper()}</figcaption>"
            f'<video controls preload="none"'
            f' src="{escaped}/{presentation}{suffix}"></video></figure>'
            for presentation in ("a", "b")
        )
        lines.append("</section>")
    lines.extend(["</body>", "</html>"])
    return "\n".join(lines) + "\n"


def stage_human_pairwise_presentation(
    formal_set_path: Path,
    ballot_path: Path,
    assignment_path: Path,
    media_manifest_path: Path,
    staging_dir: Path,
) -> HumanPairwiseStagingReport:
    """Stage digest-verified blinded A/B media and a local index page."""
    if staging_dir.exists():
        message = "staging directory already exists"
        raise ValidationError(message)
    if not staging_dir.parent.is_dir():
        message = "staging directory parent is missing"
        raise ValidationError(message)
    formal_set, formal_raw = _load_object(formal_set_path, "formal quality set")
    check_formal_quality_set(formal_set_path)
    formal_sha = _sha256(formal_raw)
    expected_ids = _case_ids(formal_set)
    _require_private_file(ballot_path, "human-pairwise ballot")
    ballot, _ = _load_object(ballot_path, "human-pairwise ballot")
    ballot_id, ballot_commitments = _validate_pending_ballot(
        ballot, expected_ids, formal_sha
    )
    _require_private_file(assignment_path, "human-pairwise assignment")
    assignment, assignment_raw = _load_object(
        assignment_path, "human-pairwise assignment"
    )
    a_sources = _validate_assignment(
        assignment,
        assignment_raw,
        ballot,
        ballot_id=ballot_id,
        ballot_commitments=ballot_commitments,
        expected_ids=expected_ids,
        formal_sha=formal_sha,
    )
    manifest, _ = _load_object(media_manifest_path, "human-pairwise media manifest")
    media = _validate_media_manifest(
        manifest, media_manifest_path.parent, expected_ids, formal_sha
    )

    staged: list[tuple[str, str]] = []
    staged_file_count = 0
    try:
        staging_dir.mkdir(mode=0o700)
        for case_id in expected_ids:
            a_source = a_sources[case_id]
            b_source = "candidate" if a_source == "baseline" else "baseline"
            case_dir = staging_dir / case_id
            case_dir.mkdir(mode=0o700)
            suffix = media[case_id]["baseline"][0].suffix
            for presentation, source in (("a", a_source), ("b", b_source)):
                path, digest = media[case_id][source]
                _copy_verified_media(
                    path,
                    digest,
                    case_dir / f"{presentation}{suffix}",
                    f"media file for case {case_id} presentation {presentation}",
                )
                staged_file_count += 1
            staged.append((case_id, suffix))
        index_path = staging_dir / "index.html"
        index_path.write_text(_presentation_index(ballot_id, staged), encoding="utf-8")
        index_path.chmod(0o600)
    except BaseException as error:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if staging_dir.exists():
            sys.stderr.write(
                "warning: the staging directory could not be fully removed;"
                " delete it manually before retrying\n"
            )
        if isinstance(error, OSError):
            message = "staging directory could not be written"
            raise ValidationError(message) from error
        raise
    return HumanPairwiseStagingReport(
        ballot_id=ballot_id,
        case_count=len(expected_ids),
        staged_file_count=staged_file_count,
    )


def _replace_private_json(path: Path, value: dict[str, object]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).chmod(0o600)
        Path(temporary).replace(path)
    except OSError as error:
        if temporary is not None:
            with contextlib.suppress(OSError):
                Path(temporary).unlink(missing_ok=True)
        message = "private ballot could not be updated"
        raise ValidationError(message) from error


def record_human_pairwise_selection(
    ballot_path: Path,
    *,
    case_id: str,
    selection: str,
    overwrite: bool = False,
) -> HumanPairwiseRecordReport:
    """Record one reviewer selection and complete the ballot when full."""
    case_id = _string(case_id, "case_id")
    if selection not in _SELECTIONS:
        message = "selection must be one of: a, b, tie"
        raise ValidationError(message)
    _require_private_file(ballot_path, "human-pairwise ballot")
    ballot, _ = _load_object(ballot_path, "human-pairwise ballot")
    _fields(ballot, _BALLOT_FIELDS, "human-pairwise ballot")
    if ballot["schema_version"] != "1.0":
        message = "human-pairwise ballot requires schema_version '1.0'"
        raise ValidationError(message)
    ballot_id = _string(ballot["ballot_id"], "human-pairwise ballot_id")
    if ballot["status"] != "pending":
        message = "human-pairwise ballot is not pending"
        raise ValidationError(message)
    cases = ballot["cases"]
    if not isinstance(cases, list):
        message = "human-pairwise ballot cases must be an array"
        raise ValidationError(message)
    target: dict[str, object] | None = None
    seen: set[str] = set()
    for index, case in enumerate(cases):
        validated, current_id, _ = _validate_ballot_case(case, index)
        if current_id in seen:
            message = f"duplicate human-pairwise ballot case: {current_id}"
            raise ValidationError(message)
        seen.add(current_id)
        if current_id == case_id:
            target = validated
    if target is None:
        message = f"human-pairwise ballot has unknown case: {case_id}"
        raise ValidationError(message)
    if target["selection"] is not None and not overwrite:
        message = f"human-pairwise selection already recorded for {case_id}"
        raise ValidationError(message)
    target["selection"] = selection

    recorded_count = sum(
        1
        for case in cases
        if isinstance(case, dict) and case.get("selection") is not None
    )
    completed = recorded_count == len(cases)
    if completed:
        ballot["status"] = "completed"
        ballot["completed_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    _replace_private_json(ballot_path, ballot)
    return HumanPairwiseRecordReport(
        ballot_id=ballot_id,
        case_id=case_id,
        recorded_count=recorded_count,
        case_count=len(cases),
        completed=completed,
    )
