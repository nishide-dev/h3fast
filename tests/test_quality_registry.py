"""Tests for private quality-registry compilation."""

import hashlib
import json
from pathlib import Path

import pytest

from h3fast.benchmarks import (
    apply_quality_registry_review,
    compile_quality_registry,
    prepare_quality_registry_review,
)
from h3fast.exceptions import ValidationError

TEMPLATE_PATH = Path("benchmarks/quality/formal-quality-set.json")


def _case(
    case_id: str,
    *,
    split: str,
    task: str = "t2va",
    references: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "id": case_id,
        "split": split,
        "prompt": f"Private prompt for {case_id}",
        "seed": 7,
        "task": task,
        "duration_seconds": 5,
        "aspect_ratio": "landscape",
        "languages": ["en"],
        "subject_tags": ["product"],
        "motion_tags": ["static"],
        "audio_tags": ["near-silent"],
        "references": references or [],
        "rights_status": "approved",
        "rights_evidence": [f"https://example.test/rights/{case_id}"],
        "public_notes": "Project-authored evaluation case.",
    }


def _registry(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "registry_id": "h3fast-phase0-formal-quality-v1",
        "updated_at": "2026-08-16",
        "selection": {
            "method": "Project-authored stratified cases.",
            "exclusions_reviewed": True,
            "public_exclusions": ["No customer or scraped inputs."],
            "known_failures_reviewed": True,
            "public_known_failures": ["Formal metric plans remain incomplete."],
        },
        "cases": cases,
    }


def _write_registry(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


def test_compile_registry_redacts_prompts_and_paths(tmp_path: Path) -> None:
    asset = tmp_path / "private-reference.bin"
    asset.write_bytes(b"private reference bytes")
    registry_path = tmp_path / "quality.private-quality-registry.json"
    registry = _registry(
        [
            _case("smoke-001", split="smoke"),
            _case(
                "regression-001",
                split="regression",
                task="ref2va",
                references=[
                    {"path": asset.name, "modality": "image"},
                    {"path": asset.name + ".audio", "modality": "audio"},
                ],
            ),
        ]
    )
    audio = tmp_path / "private-reference.bin.audio"
    audio.write_bytes(b"private audio bytes")
    registry_bytes = _write_registry(registry_path, registry)
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    for role in ("quality_owner", "rights_reviewer"):
        template["approvals"][role] = {
            "state": "approved",
            "owner": f"test-{role}",
            "deadline": "2026-08-31",
            "approved_at": "2026-08-16T00:00:00Z",
            "evidence": [f"https://example.test/approval/{role}"],
        }
    template_path = tmp_path / "formal-quality-template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    output = tmp_path / "formal-quality-set.json"

    report = compile_quality_registry(
        registry_path,
        template_path,
        output,
        registry_uri="https://example.test/private/quality-v1",
    )

    compiled = json.loads(output.read_text(encoding="utf-8"))
    serialized = output.read_text(encoding="utf-8")
    assert report.registry_sha256 == hashlib.sha256(registry_bytes).hexdigest()
    assert report.smoke_cases == 1
    assert report.regression_cases == 1
    assert report.ready is False
    assert compiled["selection"]["registry_sha256"] == report.registry_sha256
    assert compiled["approvals"]["quality_owner"]["state"] == "unassigned"
    assert compiled["approvals"]["rights_reviewer"]["state"] == "unassigned"
    assert (
        compiled["cases"][0]["prompt_sha256"]
        == hashlib.sha256(b"Private prompt for smoke-001").hexdigest()
    )
    assert compiled["cases"][1]["reference_asset_sha256s"] == [
        hashlib.sha256(b"private reference bytes").hexdigest(),
        hashlib.sha256(b"private audio bytes").hexdigest(),
    ]
    assert compiled["cases"][1]["reference_modalities"] == [
        "audio",
        "image",
        "mixed",
    ]
    assert "Private prompt" not in serialized
    assert str(tmp_path) not in serialized
    assert asset.name not in serialized


def test_compile_registry_keeps_existing_output_on_failure(tmp_path: Path) -> None:
    registry_path = tmp_path / "quality.private-quality-registry.json"
    _write_registry(
        registry_path,
        _registry(
            [
                _case(
                    "smoke-001",
                    split="smoke",
                    task="fl2va",
                    references=[{"path": "missing.png", "modality": "image"}],
                )
            ]
        ),
    )
    output = tmp_path / "formal-quality-set.json"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(
        ValidationError, match="reference at index 0 is missing"
    ) as error:
        compile_quality_registry(
            registry_path,
            TEMPLATE_PATH,
            output,
            registry_uri="https://example.test/private/quality-v1",
        )

    assert str(tmp_path) not in str(error.value)
    assert output.read_text(encoding="utf-8") == "existing\n"
    assert not output.with_suffix(".json.partial").exists()


def test_compile_registry_rejects_sensitive_or_inconsistent_input(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "quality.private-quality-registry.json"
    value = _registry([_case("smoke-001", split="smoke")])
    selection = value["selection"]
    assert isinstance(selection, dict)
    selection["public_exclusions"] = ["/private/path"]
    _write_registry(registry_path, value)

    with pytest.raises(ValidationError, match="must not expose local paths"):
        compile_quality_registry(
            registry_path,
            TEMPLATE_PATH,
            tmp_path / "output.json",
            registry_uri="https://example.test/private/quality-v1",
        )

    value = _registry([_case("smoke-001", split="smoke")])
    case = value["cases"][0]  # type: ignore[index]
    assert isinstance(case, dict)
    case["references"] = [{"path": "unexpected.png", "modality": "image"}]
    _write_registry(registry_path, value)
    with pytest.raises(ValidationError, match=r"T2VA case.*must not contain"):
        compile_quality_registry(
            registry_path,
            TEMPLATE_PATH,
            tmp_path / "output.json",
            registry_uri="https://example.test/private/quality-v1",
        )


def test_compile_registry_rejects_invalid_identity_and_destination(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "quality.private-quality-registry.json"
    value = _registry([_case("smoke-001", split="smoke")])
    value["registry_id"] = "other"
    _write_registry(registry_path, value)

    with pytest.raises(
        ValidationError, match="unsupported private quality registry_id"
    ):
        compile_quality_registry(
            registry_path,
            TEMPLATE_PATH,
            tmp_path / "output.json",
            registry_uri="https://example.test/private/quality-v1",
        )

    value["registry_id"] = "h3fast-phase0-formal-quality-v1"
    _write_registry(registry_path, value)
    with pytest.raises(ValidationError, match="output must differ"):
        compile_quality_registry(
            registry_path,
            TEMPLATE_PATH,
            registry_path,
            registry_uri="https://example.test/private/quality-v1",
        )
    with pytest.raises(ValidationError, match="registry_uri must be an HTTPS URL"):
        compile_quality_registry(
            registry_path,
            TEMPLATE_PATH,
            tmp_path / "output.json",
            registry_uri="file:///private/registry.json",
        )


def test_compile_registry_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    registry_path = tmp_path / "quality.private-quality-registry.json"
    _write_registry(
        registry_path,
        _registry(
            [
                _case("smoke-001", split="smoke"),
                _case("smoke-001", split="smoke"),
            ]
        ),
    )

    with pytest.raises(ValidationError, match="duplicate case ids"):
        compile_quality_registry(
            registry_path,
            TEMPLATE_PATH,
            tmp_path / "output.json",
            registry_uri="https://example.test/private/quality-v1",
        )


def test_prepare_review_binds_registry_without_copying_private_inputs(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "private-reference.bin"
    asset.write_bytes(b"private reference bytes")
    case = _case(
        "smoke-001",
        split="smoke",
        task="fl2va",
        references=[{"path": asset.name, "modality": "image"}],
    )
    case["rights_status"] = "unreviewed"
    case["rights_evidence"] = []
    registry_path = tmp_path / "quality.private-quality-registry.json"
    raw = _write_registry(registry_path, _registry([case]))
    output = tmp_path / "quality.private-quality-review.json"

    report = prepare_quality_registry_review(
        registry_path,
        output,
        reviewer="test-reviewer",
    )

    review = json.loads(output.read_text(encoding="utf-8"))
    serialized = output.read_text(encoding="utf-8")
    assert report.source_registry_sha256 == hashlib.sha256(raw).hexdigest()
    assert report.total_cases == 1
    assert report.pending_cases == 1
    assert report.ready is False
    assert review["cases"][0]["rights_decision"] == "pending"
    assert review["cases"][0]["selection_decision"] == "pending"
    assert review["reviewed_at"] is None
    assert "Private prompt" not in serialized
    assert asset.name not in serialized
    assert str(tmp_path) not in serialized


def test_apply_complete_review_writes_new_approved_private_registry(
    tmp_path: Path,
) -> None:
    case = _case("smoke-001", split="smoke")
    case["rights_status"] = "unreviewed"
    case["rights_evidence"] = []
    registry = _registry([case])
    selection = registry["selection"]
    assert isinstance(selection, dict)
    selection["exclusions_reviewed"] = False
    selection["known_failures_reviewed"] = False
    registry_path = tmp_path / "quality.private-quality-registry.json"
    _write_registry(registry_path, registry)
    review_path = tmp_path / "quality.private-quality-review.json"
    prepare_quality_registry_review(
        registry_path,
        review_path,
        reviewer="test-reviewer",
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewed_at"] = "2026-08-16T12:00:00Z"
    review["selection"]["method_decision"] = "approved"
    review["selection"]["exclusions_decision"] = "approved"
    review["selection"]["known_failures_decision"] = "approved"
    review["cases"][0]["rights_decision"] = "approved"
    review["cases"][0]["selection_decision"] = "approved"
    review["cases"][0]["rights_evidence"] = ["https://example.test/reviews/quality-v1"]
    review_path.write_text(json.dumps(review), encoding="utf-8")
    output = tmp_path / "reviewed.private-quality-registry.json"

    report = apply_quality_registry_review(registry_path, review_path, output)

    reviewed = json.loads(output.read_text(encoding="utf-8"))
    assert report.ready is True
    assert report.approved_cases == 1
    assert (
        report.output_registry_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert reviewed["selection"]["exclusions_reviewed"] is True
    assert reviewed["selection"]["known_failures_reviewed"] is True
    assert reviewed["cases"][0]["rights_status"] == "approved"
    assert reviewed["cases"][0]["rights_evidence"] == [
        "https://example.test/reviews/quality-v1"
    ]
    assert json.loads(registry_path.read_text(encoding="utf-8")) == registry


def test_apply_review_fails_closed_for_pending_or_stale_review(tmp_path: Path) -> None:
    case = _case("smoke-001", split="smoke")
    case["rights_status"] = "unreviewed"
    case["rights_evidence"] = []
    registry_path = tmp_path / "quality.private-quality-registry.json"
    registry = _registry([case])
    _write_registry(registry_path, registry)
    review_path = tmp_path / "quality.private-quality-review.json"
    prepare_quality_registry_review(
        registry_path,
        review_path,
        reviewer="test-reviewer",
    )
    output = tmp_path / "reviewed.private-quality-registry.json"

    pending = apply_quality_registry_review(registry_path, review_path, output)

    assert pending.ready is False
    assert pending.pending_cases == 1
    assert "reviewed_at:missing" in pending.blockers
    assert not output.exists()

    case["prompt"] = "Changed private prompt"
    _write_registry(registry_path, _registry([case]))
    with pytest.raises(ValidationError, match="source registry digest is stale"):
        apply_quality_registry_review(registry_path, review_path, output)
