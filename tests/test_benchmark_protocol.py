"""Tests for benchmark protocol validation."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from h3fast.benchmarks.protocol import load_runtime_settings, validate_protocol
from h3fast.exceptions import ValidationError


def test_repository_protocol_is_valid_draft() -> None:
    report = validate_protocol(Path("benchmarks/protocol.yaml"))

    assert report.status == "draft"
    assert report.ready is False
    assert "immutable base model revision" not in report.unresolved
    assert (
        "formal 10-case smoke and 50-case regression quality sets" in report.unresolved
    )


def test_repository_protocol_pins_exact_quality_gate() -> None:
    protocol = json.loads(Path("benchmarks/protocol.yaml").read_text(encoding="utf-8"))

    assert protocol["quality"] == {
        "reference_id": "h3fast-phase1a-exact-smoke-001-v1",
        "reference_path": "benchmarks/quality/exact-smoke-001-reference.json",
        "method": "exact-decoded-artifact-v1",
        "profile": "exact",
        "baseline_measured_runs": 3,
        "video_decode_format": "rgb24",
        "audio_decode_format": "pcm_s16le",
        "scope": "single-case placement-only regression gate",
        "formal_quality_set_path": "benchmarks/quality/formal-quality-set.json",
        "formal_quality_set_ready": False,
    }
    assert Path(protocol["quality"]["formal_quality_set_path"]).is_file()


def test_repository_protocols_change_only_dit_residency() -> None:
    baseline = json.loads(
        Path("benchmarks/protocol-baseline20.yaml").read_text(encoding="utf-8")
    )
    candidate = json.loads(Path("benchmarks/protocol.yaml").read_text(encoding="utf-8"))

    assert baseline["runtime"]["dit_layerwise_resident_layers"] == 20
    assert candidate["runtime"]["dit_layerwise_resident_layers"] == 40
    assert (
        load_runtime_settings(
            Path("benchmarks/protocol.yaml")
        ).dit_layerwise_resident_layers
        == 40
    )
    baseline["protocol_id"] = candidate["protocol_id"]
    baseline["runtime"] = candidate["runtime"]
    assert baseline == candidate


@pytest.mark.parametrize(
    "protocol_path",
    [Path("benchmarks/protocol-baseline20.yaml"), Path("benchmarks/protocol.yaml")],
)
def test_repository_protocols_match_schema(protocol_path: Path) -> None:
    schema = json.loads(
        Path("schemas/benchmark-protocol.schema.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(protocol)


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
        "runtime": {"dit_layerwise_resident_layers": 20},
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
        "runtime": {"dit_layerwise_resident_layers": 20},
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
        ("runtime", [], "runtime"),
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


@pytest.mark.parametrize("value", [0, 51, True, "40"])
def test_protocol_rejects_invalid_resident_layers(
    tmp_path: Path, value: object
) -> None:
    protocol = _ready_protocol()
    protocol["runtime"] = {"dit_layerwise_resident_layers": value}

    with pytest.raises(ValidationError, match="between 1 and 50"):
        validate_protocol(_write_protocol(tmp_path, protocol))


def test_protocol_rejects_unknown_runtime_fields(tmp_path: Path) -> None:
    protocol = _ready_protocol()
    protocol["runtime"] = {
        "dit_layerwise_resident_layers": 40,
        "dit_layerwise_resident_layer": 40,
    }

    with pytest.raises(ValidationError, match="unsupported fields"):
        validate_protocol(_write_protocol(tmp_path, protocol))


def test_protocol_accepts_supported_attention_backends(tmp_path: Path) -> None:
    for backend in ("auto", "fa", "sage_attn"):
        protocol = _ready_protocol()
        protocol["runtime"] = {
            "dit_layerwise_resident_layers": 40,
            "attention_backend": backend,
        }

        report = validate_protocol(_write_protocol(tmp_path, protocol))

        assert report.ready is True


def test_protocol_carries_the_served_model_variant(tmp_path: Path) -> None:
    """A partition serves only its own families, so the variant is reproducible."""
    protocol = _ready_protocol()
    protocol["runtime"] = {
        "dit_layerwise_resident_layers": 40,
        "model_variant": "ref2va",
    }
    path = _write_protocol(tmp_path, protocol)

    assert validate_protocol(path).ready is True
    assert load_runtime_settings(path).model_variant == "ref2va"

    protocol["runtime"] = {
        "dit_layerwise_resident_layers": 40,
        "model_variant": "t2va",
    }

    with pytest.raises(ValidationError, match="model_variant"):
        validate_protocol(_write_protocol(tmp_path, protocol))


def test_protocol_defaults_attention_backend_to_auto(tmp_path: Path) -> None:
    from h3fast.benchmarks import load_runtime_settings

    protocol = _ready_protocol()
    protocol["runtime"] = {"dit_layerwise_resident_layers": 40}
    path = _write_protocol(tmp_path, protocol)

    settings = load_runtime_settings(path)

    assert settings.attention_backend == "auto"
    assert settings.to_dict() == {
        "dit_layerwise_resident_layers": 40,
        "attention_backend": "auto",
        "model_variant": "fl2va",
    }


def test_protocol_rejects_unsupported_attention_backend(tmp_path: Path) -> None:
    protocol = _ready_protocol()
    protocol["runtime"] = {
        "dit_layerwise_resident_layers": 40,
        "attention_backend": "sage_attn_3",
    }

    with pytest.raises(ValidationError, match="attention_backend"):
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "unknown", "method is unsupported"),
        ("profile", "balanced", "profile must"),
        ("baseline_measured_runs", 2, "at least three"),
        ("formal_quality_set_ready", "no", "must be boolean"),
    ],
)
def test_protocol_rejects_invalid_quality_gate(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    protocol = _ready_protocol()
    quality: dict[str, object] = {
        "reference_id": "reference-v1",
        "reference_path": "reference.json",
        "method": "exact-decoded-artifact-v1",
        "profile": "exact",
        "baseline_measured_runs": 3,
        "video_decode_format": "rgb24",
        "audio_decode_format": "pcm_s16le",
        "scope": "test",
        "formal_quality_set_path": "formal-quality-set.json",
        "formal_quality_set_ready": False,
    }
    quality[field] = value
    protocol["quality"] = quality

    with pytest.raises(ValidationError, match=message):
        validate_protocol(_write_protocol(tmp_path, protocol))


def test_protocol_rejects_ready_flag_for_incomplete_formal_set(tmp_path: Path) -> None:
    protocol = _ready_protocol()
    protocol["quality"] = {
        "reference_id": "reference-v1",
        "reference_path": "reference.json",
        "method": "exact-decoded-artifact-v1",
        "profile": "exact",
        "baseline_measured_runs": 3,
        "video_decode_format": "rgb24",
        "audio_decode_format": "pcm_s16le",
        "scope": "test",
        "formal_quality_set_path": "benchmarks/quality/formal-quality-set.json",
        "formal_quality_set_ready": True,
    }

    with pytest.raises(ValidationError, match="requires an approved formal"):
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
