"""Tests for benchmark protocol validation."""

import json
from pathlib import Path

import pytest

from h3fast.benchmarks.protocol import validate_protocol
from h3fast.exceptions import ValidationError


def test_repository_protocol_is_valid_draft() -> None:
    report = validate_protocol(Path("benchmarks/protocol.yaml"))

    assert report.status == "draft"
    assert report.ready is False
    assert "immutable base model revision" in report.unresolved


def test_ready_protocol_requires_immutable_revision(tmp_path: Path) -> None:
    protocol = {
        "schema_version": "1.0",
        "protocol_id": "test",
        "status": "ready",
        "unresolved": [],
        "base_model": {"revision": "main"},
        "environment": {
            "accelerator": {"model": "H100"},
            "software": {"sglang": "0.5.15.post1"},
        },
        "measurement": {},
        "cases": [{}],
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValidationError, match="immutable base model revision"):
        validate_protocol(path)


def _ready_protocol() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "protocol_id": "ready-test",
        "status": "ready",
        "unresolved": [],
        "base_model": {"revision": "a" * 40},
        "environment": {
            "accelerator": {"model": "H100"},
            "software": {"sglang": "0.5.15.post1"},
        },
        "measurement": {},
        "cases": [{}],
    }


def _write_protocol(tmp_path: Path, protocol: object) -> Path:
    path = tmp_path / "protocol.yaml"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def test_ready_protocol_is_reported_ready(tmp_path: Path) -> None:
    report = validate_protocol(_write_protocol(tmp_path, _ready_protocol()))

    assert report.ready is True
    assert report.to_dict() == {
        "valid": True,
        "protocol_id": "ready-test",
        "status": "ready",
        "ready": True,
        "unresolved": [],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0", "unsupported"),
        ("protocol_id", "", "protocol_id"),
        ("status", "unknown", "status must"),
        ("unresolved", [1], "array of strings"),
        ("base_model", [], "base_model"),
        ("environment", [], "environment"),
        ("measurement", [], "measurement"),
        ("cases", [], "non-empty array"),
    ],
)
def test_protocol_rejects_invalid_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    protocol = _ready_protocol()
    protocol[field] = value

    with pytest.raises(ValidationError, match=message):
        validate_protocol(_write_protocol(tmp_path, protocol))


def test_ready_protocol_rejects_unresolved_items(tmp_path: Path) -> None:
    protocol = _ready_protocol()
    protocol["unresolved"] = ["GPU"]

    with pytest.raises(ValidationError, match="cannot contain unresolved"):
        validate_protocol(_write_protocol(tmp_path, protocol))


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {"accelerator": {}, "software": {"sglang": "0.5.15.post1"}},
            "accelerator model",
        ),
        ({"accelerator": {"model": "H100"}, "software": {}}, "SGLang"),
    ],
)
def test_ready_protocol_requires_environment_details(
    tmp_path: Path, environment: object, message: str
) -> None:
    protocol = _ready_protocol()
    protocol["environment"] = environment

    with pytest.raises(ValidationError, match=message):
        validate_protocol(_write_protocol(tmp_path, protocol))


def test_protocol_rejects_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="missing"):
        validate_protocol(tmp_path / "missing.yaml")

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ValidationError, match="JSON-compatible"):
        validate_protocol(malformed)

    with pytest.raises(ValidationError, match="root must be an object"):
        validate_protocol(_write_protocol(tmp_path, []))
