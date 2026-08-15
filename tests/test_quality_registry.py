"""Tests for private quality-registry compilation."""

import hashlib
import json
from pathlib import Path

import pytest

from h3fast.benchmarks import compile_quality_registry
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
