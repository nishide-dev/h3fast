"""Tests for optional backend metadata."""

from importlib import metadata

from h3fast.backends.sglang import REFERENCE_SGLANG_VERSION, inspect_sglang


def test_inspect_sglang_without_distribution(monkeypatch) -> None:
    """Inspecting SGLang must not require importing it."""

    def missing_distribution(_name: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", missing_distribution)

    status = inspect_sglang()

    assert status.installed_version is None
    assert status.reference_version == REFERENCE_SGLANG_VERSION
    assert status.compatible is False
    assert status.to_dict()["installed_version"] is None


def test_inspect_sglang_with_reference_distribution(monkeypatch) -> None:
    monkeypatch.setattr(metadata, "version", lambda _name: REFERENCE_SGLANG_VERSION)

    status = inspect_sglang()

    assert status.compatible is True
    assert status.to_dict()["reference_version"] == REFERENCE_SGLANG_VERSION
