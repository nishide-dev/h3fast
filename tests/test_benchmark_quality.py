"""Tests for exact decoded-artifact quality references and reports."""

import hashlib
import io
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from h3fast.benchmarks.quality import (
    ArtifactObservation,
    MediaObservation,
    _decoded_stream_sha256,
    _load_measured_artifacts,
    _probe_media,
    _protocol_identity,
    _statistics,
    build_quality_reference,
    check_quality,
    inspect_artifact,
    tool_version,
)
from h3fast.exceptions import ValidationError
from h3fast.manifest.checksums import sha256_file

_PROMPT_SHA256 = "eeb2aac52b698c1d76031d2cd4755f9429e08a5fa614da07812bb185cbebccae"
_SGLANG_COMMIT = "6eb941a34cb100b708a42ed1d26d2bdefafbd01e"


def _media() -> MediaObservation:
    return MediaObservation(
        video_codec="h264",
        video_pixel_format="yuv420p",
        width=1344,
        height=768,
        frame_rate="24/1",
        video_duration_seconds=5.166667,
        video_frames=124,
        audio_codec="aac",
        audio_sample_rate_hz=32000,
        audio_channels=2,
        audio_duration_seconds=5.175,
        audio_frames=163,
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        container_duration_seconds=5.207,
    )


def _inspector(path: Path, _ffmpeg: str, _ffprobe: str) -> ArtifactObservation:
    return ArtifactObservation(
        artifact_sha256=sha256_file(path),
        artifact_size=path.stat().st_size,
        video_decoded_sha256="c" * 64,
        audio_decoded_sha256="d" * 64,
        media=_media(),
    )


def _version(executable: str) -> str:
    return f"{executable} version 1"


def _write_protocol(path: Path, protocol_id: str) -> None:
    protocol = json.loads(Path("benchmarks/protocol.yaml").read_text(encoding="utf-8"))
    protocol["protocol_id"] = protocol_id
    path.write_text(json.dumps(protocol), encoding="utf-8")


def _write_suite(root: Path, protocol_id: str, *, content: bytes = b"artifact") -> Path:
    root.mkdir(parents=True)
    runs: list[dict[str, object]] = []
    for index in range(3):
        artifact = root / f"measured-{index + 1:03d}.mp4"
        artifact.write_bytes(content)
        runs.append(
            {
                "label": f"measured-{index + 1:03d}",
                "kind": "measured",
                "status": "completed",
                "result": {
                    "schema_version": "1.0",
                    "protocol_id": protocol_id,
                    "case_id": "smoke-001",
                    "prompt_sha256": _PROMPT_SHA256,
                    "request": {"seed": 1101, "task": "t2va"},
                    "artifact": {
                        "path": str(artifact),
                        "size": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    },
                    "server": {"performance": {"sglang_commit": _SGLANG_COMMIT}},
                },
            }
        )
    suite = {
        "schema_version": "1.0",
        "protocol_id": protocol_id,
        "case_id": "smoke-001",
        "status": "completed",
        "completed_at": "2026-08-15T10:46:05+00:00",
        "measured_runs": 3,
        "runs": runs,
    }
    path = root / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    return path


def test_build_and_check_exact_quality_reference(tmp_path: Path) -> None:
    baseline_protocol = tmp_path / "baseline-protocol.json"
    candidate_protocol = tmp_path / "candidate-protocol.json"
    _write_protocol(baseline_protocol, "baseline")
    _write_protocol(candidate_protocol, "candidate")
    baseline_suite = _write_suite(tmp_path / "baseline", "baseline")
    candidate_suite = _write_suite(tmp_path / "candidate", "candidate")
    reference_path = tmp_path / "reference.json"

    reference = build_quality_reference(
        baseline_suite,
        baseline_protocol,
        reference_path,
        reference_id="exact-smoke-001-v1",
        artifact_inspector=_inspector,
        version_reader=_version,
    )

    assert reference["method"]["id"] == "exact-decoded-artifact-v1"
    assert reference["source"]["measured_runs"] == 3
    assert reference["baseline_statistics"]["av_duration_drift_seconds"] == {
        "min": pytest.approx(0.008333),
        "p5": pytest.approx(0.008333),
        "p50": pytest.approx(0.008333),
        "p95": pytest.approx(0.008333),
        "max": pytest.approx(0.008333),
    }
    serialized = reference_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "A quiet street" not in serialized

    report_path = tmp_path / "report.json"
    report = check_quality(
        reference_path,
        candidate_suite,
        candidate_protocol,
        report_path,
        artifact_inspector=_inspector,
        version_reader=_version,
    )

    assert report["status"] == "passed"
    assert report["candidate"]["protocol_id"] == "candidate"
    assert report["worst_case"] == {
        "scope": "run",
        "label": "measured-001",
        "failed_checks": 0,
    }
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "passed"

    for schema_path, value in (
        (Path("schemas/quality-reference.schema.json"), reference),
        (Path("schemas/quality-report.schema.json"), report),
    ):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)


def test_quality_check_fails_audio_regression_and_tool_mismatch(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.json"
    _write_protocol(protocol, "baseline")
    suite = _write_suite(tmp_path / "suite", "baseline")
    reference_path = tmp_path / "reference.json"
    build_quality_reference(
        suite,
        protocol,
        reference_path,
        reference_id="exact-smoke-001-v1",
        artifact_inspector=_inspector,
        version_reader=_version,
    )

    def changed_audio(path: Path, ffmpeg: str, ffprobe: str) -> ArtifactObservation:
        return replace(_inspector(path, ffmpeg, ffprobe), audio_decoded_sha256="e" * 64)

    def changed_version(executable: str) -> str:
        return f"{executable} version 2"

    report = check_quality(
        reference_path,
        suite,
        protocol,
        tmp_path / "failed.json",
        artifact_inspector=changed_audio,
        version_reader=changed_version,
    )

    assert report["status"] == "failed"
    assert report["worst_case"] == {
        "scope": "environment",
        "failed_checks": 2,
    }
    audio_check = next(
        check
        for check in report["runs"][0]["checks"]
        if check["name"] == "audio_decoded_sha256"
    )
    assert audio_check["passed"] is False


def test_reference_rejects_unstable_or_tampered_baseline(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    _write_protocol(protocol, "baseline")
    suite = _write_suite(tmp_path / "suite", "baseline")
    suite_value = json.loads(suite.read_text(encoding="utf-8"))
    second = Path(suite_value["runs"][1]["result"]["artifact"]["path"])
    second.write_bytes(b"different")
    suite_value["runs"][1]["result"]["artifact"].update(
        {"size": second.stat().st_size, "sha256": sha256_file(second)}
    )
    suite.write_text(json.dumps(suite_value), encoding="utf-8")

    with pytest.raises(ValidationError, match="bitwise-stable"):
        build_quality_reference(
            suite,
            protocol,
            tmp_path / "reference.json",
            reference_id="exact-smoke-001-v1",
            artifact_inspector=_inspector,
            version_reader=_version,
        )

    first = Path(suite_value["runs"][0]["result"]["artifact"]["path"])
    first.write_bytes(b"tampered")
    with pytest.raises(ValidationError, match="hash does not match"):
        build_quality_reference(
            suite,
            protocol,
            tmp_path / "reference.json",
            reference_id="exact-smoke-001-v1",
            artifact_inspector=_inspector,
            version_reader=_version,
        )


def test_inspect_artifact_hashes_normalized_decoded_streams(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"container")
    probe = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1344,
                "height": 768,
                "r_frame_rate": "24/1",
                "duration": "5.166667",
                "nb_frames": "124",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "32000",
                "channels": 2,
                "duration": "5.175000",
                "nb_frames": "163",
            },
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "5.207000",
            "size": "912408",
        },
    }
    monkeypatch.setattr(
        "h3fast.benchmarks.quality.shutil.which", lambda value: f"/usr/bin/{value}"
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.quality.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(probe), stderr=""
        ),
    )

    class Process:
        def __init__(self, command, **_kwargs) -> None:
            payload = b"video pixels" if "0:v:0" in command else b"audio samples"
            self.stdout = io.BytesIO(payload)

        @staticmethod
        def wait(**_kwargs) -> int:
            return 0

    monkeypatch.setattr("h3fast.benchmarks.quality.subprocess.Popen", Process)

    observation = inspect_artifact(artifact, "ffmpeg", "ffprobe")

    assert (
        observation.video_decoded_sha256 == hashlib.sha256(b"video pixels").hexdigest()
    )
    assert (
        observation.audio_decoded_sha256 == hashlib.sha256(b"audio samples").hexdigest()
    )
    assert observation.media.av_duration_drift_seconds == pytest.approx(0.008333)


def test_media_tools_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("h3fast.benchmarks.quality.shutil.which", lambda _value: None)
    with pytest.raises(ValidationError, match="unavailable"):
        tool_version("ffmpeg")

    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"invalid")
    monkeypatch.setattr(
        "h3fast.benchmarks.quality.shutil.which", lambda value: f"/usr/bin/{value}"
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.quality.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps({"streams": [], "format": {}}), stderr=""
        ),
    )
    with pytest.raises(ValidationError, match="exactly one video and one audio"):
        _probe_media(artifact, "ffprobe")


def test_suite_loader_rejects_inconsistent_metadata(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path / "suite", "baseline")
    baseline = json.loads(suite_path.read_text(encoding="utf-8"))

    mutations = [
        (lambda value: value.update(status="failed"), "completed schema"),
        (lambda value: value.update(runs={}), "runs must be an array"),
        (
            lambda value: value["runs"][0].update(status="failed"),
            "cannot include failed",
        ),
        (lambda value: value["runs"][0].update(result=None), "missing its result"),
        (
            lambda value: value["runs"][0]["result"].update(protocol_id="other"),
            "suite identity",
        ),
        (
            lambda value: value["runs"][0]["result"].update(prompt_sha256="bad"),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["runs"][0]["result"].update(request=[]),
            "request metadata",
        ),
        (
            lambda value: value["runs"][0]["result"].update(artifact=[]),
            "missing artifact",
        ),
        (
            lambda value: value["runs"][0]["result"]["server"].update(performance=[]),
            "performance provenance",
        ),
        (lambda value: value.update(measured_runs=4), "run count"),
        (
            lambda value: value["runs"][1].update(label="measured-001"),
            "labels must be unique",
        ),
    ]
    for mutate, message in mutations:
        value = deepcopy(baseline)
        mutate(value)
        suite_path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValidationError, match=message):
            _load_measured_artifacts(suite_path)


def test_suite_loader_rejects_missing_artifact_and_invalid_numbers(
    tmp_path: Path,
) -> None:
    suite_path = _write_suite(tmp_path / "suite", "baseline")
    baseline = json.loads(suite_path.read_text(encoding="utf-8"))

    missing = deepcopy(baseline)
    missing["runs"][0]["result"]["artifact"]["path"] = "missing.mp4"
    suite_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValidationError, match="unavailable"):
        _load_measured_artifacts(suite_path)

    invalid = deepcopy(baseline)
    invalid["runs"][0]["result"]["artifact"]["size"] = True
    suite_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValidationError, match="integer"):
        _load_measured_artifacts(suite_path)


def test_protocol_identity_rejects_missing_case_details(tmp_path: Path) -> None:
    protocol = json.loads(Path("benchmarks/protocol.yaml").read_text(encoding="utf-8"))
    path = tmp_path / "protocol.json"

    protocol["base_model"] = []
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValidationError, match="missing base_model"):
        _protocol_identity(path, "smoke-001")

    protocol = json.loads(Path("benchmarks/protocol.yaml").read_text(encoding="utf-8"))
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValidationError, match="does not define case"):
        _protocol_identity(path, "missing")

    protocol["cases"][0]["prompt"] = ""
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValidationError, match="prompt must"):
        _protocol_identity(path, "smoke-001")


def test_reference_builder_rejects_invalid_inputs(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    _write_protocol(protocol, "baseline")
    suite = _write_suite(tmp_path / "suite", "baseline")

    with pytest.raises(ValidationError, match="contain no whitespace"):
        build_quality_reference(
            suite,
            protocol,
            tmp_path / "reference.json",
            reference_id="bad id",
            artifact_inspector=_inspector,
            version_reader=_version,
        )

    suite_value = json.loads(suite.read_text(encoding="utf-8"))
    suite_value["measured_runs"] = 2
    suite_value["runs"] = suite_value["runs"][:2]
    suite.write_text(json.dumps(suite_value), encoding="utf-8")
    with pytest.raises(ValidationError, match="at least three"):
        build_quality_reference(
            suite,
            protocol,
            tmp_path / "reference.json",
            reference_id="reference-v1",
            artifact_inspector=_inspector,
            version_reader=_version,
        )

    suite = _write_suite(tmp_path / "suite-2", "other")
    with pytest.raises(ValidationError, match="protocol id"):
        build_quality_reference(
            suite,
            protocol,
            tmp_path / "reference.json",
            reference_id="reference-v1",
            artifact_inspector=_inspector,
            version_reader=_version,
        )


def test_reference_builder_rejects_inspector_and_provenance_mismatch(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.json"
    _write_protocol(protocol, "baseline")
    suite = _write_suite(tmp_path / "suite", "baseline")

    def wrong_size(path: Path, ffmpeg: str, ffprobe: str) -> ArtifactObservation:
        observation = _inspector(path, ffmpeg, ffprobe)
        return replace(observation, artifact_size=observation.artifact_size + 1)

    with pytest.raises(ValidationError, match="inspector disagrees"):
        build_quality_reference(
            suite,
            protocol,
            tmp_path / "reference.json",
            reference_id="reference-v1",
            artifact_inspector=wrong_size,
            version_reader=_version,
        )

    value = json.loads(suite.read_text(encoding="utf-8"))
    for run in value["runs"]:
        run["result"]["prompt_sha256"] = "a" * 64
    suite.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValidationError, match="prompt digest"):
        build_quality_reference(
            suite,
            protocol,
            tmp_path / "reference.json",
            reference_id="reference-v1",
            artifact_inspector=_inspector,
            version_reader=_version,
        )

    value = json.loads(
        _write_suite(tmp_path / "suite-2", "baseline").read_text(encoding="utf-8")
    )
    suite = tmp_path / "suite-2" / "suite.json"
    value["runs"][1]["result"]["request"]["seed"] = 42
    suite.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValidationError, match="inconsistent quality provenance"):
        build_quality_reference(
            suite,
            protocol,
            tmp_path / "reference.json",
            reference_id="reference-v1",
            artifact_inspector=_inspector,
            version_reader=_version,
        )


def test_quality_check_rejects_corrupt_reference_and_candidate(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.json"
    _write_protocol(protocol, "baseline")
    suite = _write_suite(tmp_path / "suite", "baseline")
    reference_path = tmp_path / "reference.json"
    build_quality_reference(
        suite,
        protocol,
        reference_path,
        reference_id="reference-v1",
        artifact_inspector=_inspector,
        version_reader=_version,
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    corrupt = deepcopy(reference)
    corrupt["method"]["id"] = "other"
    reference_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValidationError, match="unsupported quality reference method"):
        check_quality(
            reference_path,
            suite,
            protocol,
            tmp_path / "report.json",
            artifact_inspector=_inspector,
            version_reader=_version,
        )

    corrupt = deepcopy(reference)
    corrupt["method"]["video_decode_format"] = "gray"
    reference_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValidationError, match="decode format"):
        check_quality(
            reference_path,
            suite,
            protocol,
            tmp_path / "report.json",
            artifact_inspector=_inspector,
            version_reader=_version,
        )

    corrupt = deepcopy(reference)
    corrupt["baseline_statistics"]["artifact_size"]["p5"] = 999999
    reference_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValidationError, match="not ordered"):
        check_quality(
            reference_path,
            suite,
            protocol,
            tmp_path / "report.json",
            artifact_inspector=_inspector,
            version_reader=_version,
        )

    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    candidate_protocol = json.loads(protocol.read_text(encoding="utf-8"))
    candidate_protocol["protocol_id"] = "other"
    protocol.write_text(json.dumps(candidate_protocol), encoding="utf-8")
    with pytest.raises(ValidationError, match="protocol id"):
        check_quality(
            reference_path,
            suite,
            protocol,
            tmp_path / "report.json",
            artifact_inspector=_inspector,
            version_reader=_version,
        )


def test_media_subprocess_failures_are_validation_errors(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"artifact")
    monkeypatch.setattr(
        "h3fast.benchmarks.quality.shutil.which", lambda value: f"/usr/bin/{value}"
    )

    monkeypatch.setattr(
        "h3fast.benchmarks.quality.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, stdout="", stderr="probe error"
        ),
    )
    with pytest.raises(ValidationError, match="probe error"):
        _probe_media(artifact, "ffprobe")
    with pytest.raises(ValidationError, match="version query failed"):
        tool_version("ffmpeg")

    class StartFailure:
        def __init__(self, *_args, **_kwargs) -> None:
            message = "cannot start"
            raise OSError(message)

    monkeypatch.setattr("h3fast.benchmarks.quality.subprocess.Popen", StartFailure)
    with pytest.raises(ValidationError, match="could not start"):
        _decoded_stream_sha256(artifact, "ffmpeg", stream="video", media=_media())
    with pytest.raises(ValidationError, match="unsupported decoded"):
        _decoded_stream_sha256(artifact, "ffmpeg", stream="data", media=_media())


def test_quality_statistics_reject_invalid_values() -> None:
    with pytest.raises(ValidationError, match="finite non-negative"):
        _statistics([])
    with pytest.raises(ValidationError, match="finite non-negative"):
        _statistics([float("nan")])
