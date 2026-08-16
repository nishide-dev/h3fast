"""Tests for the formal 60-case generation runner."""

import hashlib
import json
from pathlib import Path

import pytest

from h3fast.benchmarks import run_formal_cases
from h3fast.benchmarks.client import BenchmarkResult
from h3fast.exceptions import ValidationError

FORMAL_SET = Path("benchmarks/quality/formal-quality-set.json")
PROTOCOL = Path("benchmarks/protocol.yaml")


def _test_prompt(case_id: str) -> str:
    return f"test prompt for {case_id}"


@pytest.fixture(scope="session")
def formal_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A synthetic formal set + private registry with matching prompt digests."""
    root = tmp_path_factory.mktemp("formal")
    formal = json.loads(FORMAL_SET.read_text(encoding="utf-8"))
    registry_cases = []
    for case in formal["cases"]:
        prompt = _test_prompt(case["id"])
        case["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        registry_cases.append(
            {
                "id": case["id"],
                "split": case["split"],
                "prompt": prompt,
                "seed": case["seed"],
                "task": case["task"],
                "duration_seconds": case["duration_seconds"],
                "aspect_ratio": case["aspect_ratio"],
            }
        )
    formal_path = root / "formal-quality-set.json"
    formal_path.write_text(
        json.dumps(formal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry_path = root / "registry.json"
    registry_path.write_text(
        json.dumps({"schema_version": "1.0", "cases": registry_cases}),
        encoding="utf-8",
    )
    registry_path.chmod(0o600)
    return formal_path, registry_path


def _fake_runner(record: list[dict[str, object]]):
    def run(_protocol_path: Path, case: dict[str, object], **kwargs) -> BenchmarkResult:
        output_dir = kwargs["output_dir"]
        case_id = str(case["id"])
        artifact = Path(output_dir) / f"proto-{case_id}.mp4"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        data = f"media {case_id}".encode()
        artifact.write_bytes(data)
        record.append({"case": dict(case), "kwargs": dict(kwargs)})
        return BenchmarkResult(
            protocol_id="proto",
            case_id=case_id,
            job_id=f"job-{case_id}",
            started_at="2026-08-16T00:00:00Z",
            completed_at="2026-08-16T00:01:00Z",
            elapsed_seconds=60.0,
            prompt_sha256=hashlib.sha256(
                str(case["prompt"]).encode("utf-8")
            ).hexdigest(),
            request={},
            artifact_path=str(artifact.resolve()),
            artifact_size=len(data),
            artifact_sha256=hashlib.sha256(data).hexdigest(),
            server={},
        )

    return run


def test_runs_all_cases_and_writes_manifest(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner(calls)
    )
    output_dir = tmp_path / "outputs"

    report = run_formal_cases(
        PROTOCOL,
        registry_path,
        formal_path,
        endpoint="http://127.0.0.1:30010",
        output_dir=output_dir,
        repetition_id="baseline-rep1",
    )

    assert report.case_count == 60
    assert report.completed_count == 60
    assert report.skipped_count == 0
    assert len(calls) == 60
    payload = report.to_dict()
    assert str(tmp_path) not in json.dumps(payload)
    manifest = json.loads(
        (output_dir / "baseline-rep1" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["repetition_id"] == "baseline-rep1"
    assert (
        manifest["formal_set_sha256"]
        == hashlib.sha256(formal_path.read_bytes()).hexdigest()
    )
    assert len(manifest["cases"]) == 60
    first = manifest["cases"][0]
    assert first["case_id"] == "smoke-001"
    assert first["artifact_sha256"]
    assert "prompt" not in json.dumps(manifest)
    assert str(output_dir) not in json.dumps(manifest["cases"])

    first_case = calls[0]["case"]
    assert isinstance(first_case, dict)
    assert first_case["prompt"] == _test_prompt("smoke-001")
    assert first_case["aspect_ratio"] in {"16:9", "9:16", "1:1"}
    assert first_case["short_edge"] == 768
    assert first_case["sigma_points"] == 50
    assert first_case["duration_seconds"] == 4


def test_split_filter_and_resume_skip(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner(calls)
    )
    output_dir = tmp_path / "outputs"

    report = run_formal_cases(
        PROTOCOL,
        registry_path,
        formal_path,
        endpoint="http://127.0.0.1:30010",
        output_dir=output_dir,
        repetition_id="rep1",
        split="smoke",
    )
    assert report.case_count == 10
    assert report.completed_count == 10
    assert len(calls) == 10

    calls.clear()
    resumed = run_formal_cases(
        PROTOCOL,
        registry_path,
        formal_path,
        endpoint="http://127.0.0.1:30010",
        output_dir=output_dir,
        repetition_id="rep1",
        split="smoke",
    )
    assert resumed.completed_count == 10
    assert resumed.skipped_count == 10
    assert calls == []


def test_rejects_prompt_digest_mismatch_and_metadata_drift(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner([])
    )

    tampered = json.loads(registry_path.read_text(encoding="utf-8"))
    tampered["cases"][0]["prompt"] = "a different prompt"
    bad_registry = tmp_path / "bad-registry.json"
    bad_registry.write_text(json.dumps(tampered), encoding="utf-8")
    bad_registry.chmod(0o600)
    with pytest.raises(ValidationError, match="prompt digest"):
        run_formal_cases(
            PROTOCOL,
            bad_registry,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "o1",
            repetition_id="rep1",
        )

    drifted = json.loads(registry_path.read_text(encoding="utf-8"))
    drifted["cases"][1]["seed"] = 999999
    drift_registry = tmp_path / "drift-registry.json"
    drift_registry.write_text(json.dumps(drifted), encoding="utf-8")
    drift_registry.chmod(0o600)
    with pytest.raises(ValidationError, match="seed"):
        run_formal_cases(
            PROTOCOL,
            drift_registry,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "o2",
            repetition_id="rep1",
        )


def test_rejects_exposed_registry_and_unknown_split(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner([])
    )

    registry_path.chmod(0o644)
    try:
        with pytest.raises(ValidationError, match="group or other"):
            run_formal_cases(
                PROTOCOL,
                registry_path,
                formal_path,
                endpoint="http://127.0.0.1:30010",
                output_dir=tmp_path / "o3",
                repetition_id="rep1",
            )
    finally:
        registry_path.chmod(0o600)

    with pytest.raises(ValidationError, match="split"):
        run_formal_cases(
            PROTOCOL,
            registry_path,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "o4",
            repetition_id="rep1",
            split="warmup",
        )


def test_cli_runs_formal_cases(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from h3fast.cli import main

    formal_path, registry_path = formal_pair
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner([])
    )
    output_dir = tmp_path / "outputs"

    status = main(
        [
            "benchmark",
            "run-formal-cases",
            "--registry",
            str(registry_path),
            "--formal-set",
            str(formal_path),
            "--endpoint",
            "http://127.0.0.1:30010",
            "--output-dir",
            str(output_dir),
            "--repetition",
            "cli-rep1",
            "--split",
            "smoke",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert status == 0
    assert output["case_count"] == 10
    assert output["completed_count"] == 10
    assert str(tmp_path) not in json.dumps(output)


def test_failure_stops_and_keeps_partial_progress(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    calls: list[dict[str, object]] = []
    inner = _fake_runner(calls)

    def failing(protocol_path: Path, case: dict[str, object], **kwargs):
        if case["id"] == "smoke-003":
            message = "video job failed"
            raise ValidationError(message)
        return inner(protocol_path, case, **kwargs)

    monkeypatch.setattr("h3fast.benchmarks.formal_runner._run_single_case", failing)
    output_dir = tmp_path / "outputs"

    with pytest.raises(ValidationError, match="video job failed"):
        run_formal_cases(
            PROTOCOL,
            registry_path,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=output_dir,
            repetition_id="rep1",
            split="smoke",
        )
    assert len(calls) == 2
    assert not (output_dir / "rep1" / "run-manifest.json").exists()

    calls.clear()
    monkeypatch.setattr("h3fast.benchmarks.formal_runner._run_single_case", inner)
    report = run_formal_cases(
        PROTOCOL,
        registry_path,
        formal_path,
        endpoint="http://127.0.0.1:30010",
        output_dir=output_dir,
        repetition_id="rep1",
        split="smoke",
    )
    assert report.completed_count == 10
    assert report.skipped_count == 2
    assert len(calls) == 8
