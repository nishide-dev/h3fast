"""Tests for model artifact verification."""

import json
from pathlib import Path

import pytest

from h3fast.exceptions import ValidationError
from h3fast.manifest.checksums import parse_checksum_file, sha256_file
from h3fast.manifest.model import validate_model_manifest, verify_model_artifact


def _write_artifact(root: Path) -> Path:
    index = root / "transformer" / "model.safetensors.index.json"
    index.parent.mkdir(parents=True)
    index.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "artifact_id": "h3fast-test",
        "artifact_type": "model_derivative",
        "base_model": "MiniMaxAI/MiniMax-H3",
        "base_revision": "a" * 40,
        "task_family": "fl2va",
        "runtime": {
            "name": "h3fast",
            "requires": ">=0.1,<0.2",
            "tested_versions": ["0.1.0"],
        },
        "components": [
            {
                "name": "transformer",
                "format": "safetensors",
                "dtype": "bf16",
                "index": "transformer/model.safetensors.index.json",
            }
        ],
        "license": {"name": "MiniMax H3 Community License Agreement"},
        "build": {"source_revision": "b" * 40},
    }
    (root / "h3fast_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "checksums.sha256").write_text(
        f"{sha256_file(index)}  transformer/model.safetensors.index.json\n",
        encoding="utf-8",
    )
    return index


def test_verify_model_artifact(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = verify_model_artifact(tmp_path)

    assert report.artifact_id == "h3fast-test"
    assert report.verified_files == ("transformer/model.safetensors.index.json",)


def test_verify_model_artifact_detects_tampering(tmp_path: Path) -> None:
    index = _write_artifact(tmp_path)
    index.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="checksum mismatch"):
        verify_model_artifact(tmp_path)


def test_verify_model_artifact_rejects_path_traversal(tmp_path: Path) -> None:
    _write_artifact(tmp_path)
    (tmp_path / "checksums.sha256").write_text(
        f"{'a' * 64}  ../outside\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="unsafe artifact-relative path"):
        verify_model_artifact(tmp_path)


def test_checksum_inventory_rejects_missing_empty_and_duplicate(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="missing"):
        parse_checksum_file(tmp_path / "missing.sha256")

    inventory = tmp_path / "checksums.sha256"
    inventory.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="no entries"):
        parse_checksum_file(inventory)

    digest = "a" * 64
    inventory.write_text(f"{digest}  file\n{digest}  file\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate"):
        parse_checksum_file(inventory)


def test_verify_model_artifact_rejects_missing_checksummed_file(
    tmp_path: Path,
) -> None:
    _write_artifact(tmp_path)
    (tmp_path / "transformer" / "model.safetensors.index.json").unlink()

    with pytest.raises(ValidationError, match="checksummed file is missing"):
        verify_model_artifact(tmp_path)


def test_validate_model_manifest_rejects_missing_and_mutable_revision() -> None:
    with pytest.raises(ValidationError, match="missing required fields"):
        validate_model_manifest({})

    manifest = {
        "schema_version": "1.0",
        "artifact_id": "test",
        "artifact_type": "model_derivative",
        "base_model": "MiniMaxAI/MiniMax-H3",
        "base_revision": "main",
        "task_family": "fl2va",
        "runtime": {
            "name": "h3fast",
            "requires": ">=0.1",
            "tested_versions": ["0.1.0"],
        },
        "components": [
            {
                "name": "transformer",
                "format": "safetensors",
                "dtype": "bf16",
                "index": "transformer/index.json",
            }
        ],
        "license": {"name": "test"},
        "build": {"source_revision": "test"},
    }

    with pytest.raises(ValidationError, match="base_revision"):
        validate_model_manifest(manifest)
