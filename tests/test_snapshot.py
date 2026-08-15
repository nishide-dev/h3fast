"""Tests for local snapshot inspection."""

from pathlib import Path

import pytest

from h3fast.exceptions import ValidationError
from h3fast.manifest.snapshot import REQUIRED_COMPONENTS, inspect_snapshot


def _write_snapshot(root: Path, variant: str = "FL2VA") -> None:
    (root / "model_index.json").write_text("{}\n", encoding="utf-8")
    variant_root = root / variant
    variant_root.mkdir()
    (variant_root / "model_index.json").write_text("{}\n", encoding="utf-8")
    for component in REQUIRED_COMPONENTS:
        component_root = variant_root / component
        component_root.mkdir()
        (component_root / "config.json").write_text("{}\n", encoding="utf-8")


def test_inspect_snapshot_with_hashes(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)

    report = inspect_snapshot(
        tmp_path,
        variant="fl2va",
        base_revision="a" * 40,
        include_hashes=True,
    )

    assert report.variant == "fl2va"
    assert report.files
    assert all(file.sha256 is not None for file in report.files)


def test_inspect_snapshot_rejects_missing_component(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    (tmp_path / "FL2VA" / "audio_vae" / "config.json").unlink()
    (tmp_path / "FL2VA" / "audio_vae").rmdir()

    with pytest.raises(ValidationError, match="missing components: audio_vae"):
        inspect_snapshot(
            tmp_path,
            variant="fl2va",
            base_revision="a" * 40,
        )


def test_inspect_snapshot_rejects_mutable_revision(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)

    with pytest.raises(ValidationError, match="40-character commit SHA"):
        inspect_snapshot(tmp_path, variant="fl2va", base_revision="main")


def test_snapshot_report_is_json_serializable(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)

    report = inspect_snapshot(tmp_path, variant="fl2va", base_revision="a" * 40)

    assert report.to_dict()["base_model"] == "MiniMaxAI/MiniMax-H3"
    assert report.files[0].to_dict()["sha256"] is None


def test_inspect_snapshot_rejects_unsupported_model(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unsupported base model"):
        inspect_snapshot(
            tmp_path,
            variant="fl2va",
            base_revision="a" * 40,
            base_model="other/model",
        )


def test_inspect_snapshot_requires_root_and_variant_indexes(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match=r"model_index\.json"):
        inspect_snapshot(tmp_path, variant="fl2va", base_revision="a" * 40)

    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match=r"FL2VA/model_index\.json"):
        inspect_snapshot(tmp_path, variant="fl2va", base_revision="a" * 40)


def test_inspect_snapshot_rejects_symlink(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    (tmp_path / "FL2VA" / "processor" / "link.json").symlink_to(target)

    with pytest.raises(ValidationError, match="symlinks"):
        inspect_snapshot(tmp_path, variant="fl2va", base_revision="a" * 40)
