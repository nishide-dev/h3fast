"""Tests for the formal 60-case generation runner."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from h3fast.benchmarks import run_formal_cases, run_supplied_case
from h3fast.benchmarks.client import BenchmarkResult
from h3fast.benchmarks.formal_runner import _ASPECT_RATIO_MAP
from h3fast.exceptions import ValidationError

FORMAL_SET = Path("benchmarks/quality/formal-quality-set.json")
PROTOCOL = Path("benchmarks/protocol.yaml")
T2VA_TOTAL = 20
T2VA_SMOKE = 4
SMOKE_T2VA_IDS = ("smoke-001", "smoke-004", "smoke-007", "smoke-010")


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
        references = []
        if case["task"] == "fl2va":
            references = [
                {"modality": "image", "path": "assets/frame-first.png"},
                {"modality": "image", "path": "assets/frame-last.png"},
            ]
        elif case["task"] == "ref2va":
            modality = case["reference_modalities"][0]
            names = {
                "image": "assets/frame-first.png",
                "video": "assets/reference.mp4",
                "audio": "assets/reference.wav",
            }
            references = [
                {"modality": modality, "path": names.get(modality, names["image"])}
            ]
        registry_cases.append(
            {
                "id": case["id"],
                "split": case["split"],
                "prompt": prompt,
                "seed": case["seed"],
                "task": case["task"],
                "duration_seconds": case["duration_seconds"],
                "aspect_ratio": case["aspect_ratio"],
                "references": references,
            }
        )
    formal_path = root / "formal-quality-set.json"
    formal_path.write_text(
        json.dumps(formal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("frame-first.png", "frame-last.png", "reference.mp4", "reference.wav"):
        (assets / name).write_bytes(f"asset {name}".encode())
    registry_path = root / "registry.json"
    registry_path.write_text(
        json.dumps({"schema_version": "1.0", "cases": registry_cases}),
        encoding="utf-8",
    )
    registry_path.chmod(0o600)
    return formal_path, registry_path


def _copy_pair(pair: tuple[Path, Path], destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    formal_copy = destination / "formal-quality-set.json"
    registry_copy = destination / "registry.json"
    shutil.copyfile(pair[0], formal_copy)
    shutil.copyfile(pair[1], registry_copy)
    registry_copy.chmod(0o600)
    return formal_copy, registry_copy


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
            protocol_id="h3fast-phase1b-resident40-v1",
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


def test_aspect_ratio_map_is_pinned_exactly() -> None:
    assert _ASPECT_RATIO_MAP == {
        "landscape": "16:9",
        "portrait": "9:16",
        "square": "1:1",
    }


def test_runs_selected_t2va_cases_and_writes_manifest(
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

    assert report.case_count == T2VA_TOTAL
    assert report.generated_count == T2VA_TOTAL
    assert report.skipped_count == 0
    assert len(calls) == T2VA_TOTAL
    payload = report.to_dict()
    assert str(tmp_path) not in json.dumps(payload)
    manifest = json.loads(
        (output_dir / "baseline-rep1" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["repetition_id"] == "baseline-rep1"
    assert manifest["task"] == "t2va"
    assert (
        manifest["formal_set_sha256"]
        == hashlib.sha256(formal_path.read_bytes()).hexdigest()
    )
    assert len(manifest["cases"]) == T2VA_TOTAL
    first = manifest["cases"][0]
    assert first["case_id"] == "smoke-001"
    assert first["artifact_sha256"]
    assert "prompt" not in json.dumps(manifest)
    assert str(output_dir) not in json.dumps(manifest["cases"])
    assert (
        report.manifest_sha256
        == hashlib.sha256(
            (output_dir / "baseline-rep1" / "run-manifest.json").read_bytes()
        ).hexdigest()
    )

    first_case = calls[0]["case"]
    assert isinstance(first_case, dict)
    assert first_case["prompt"] == _test_prompt("smoke-001")
    assert first_case["aspect_ratio"] == "16:9"
    assert first_case["short_edge"] == 768
    assert first_case["sigma_points"] == 50
    assert first_case["duration_seconds"] == 4


def test_rejects_unsupported_task_family(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner([])
    )

    with pytest.raises(ValidationError, match="unsupported formal task family"):
        run_formal_cases(
            PROTOCOL,
            registry_path,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "outputs",
            repetition_id="rep1",
            task="i2va",
        )


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
    assert report.case_count == T2VA_SMOKE
    assert report.generated_count == T2VA_SMOKE
    first_manifest = json.loads(
        (output_dir / "rep1" / "run-manifest.json").read_text(encoding="utf-8")
    )

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
    assert resumed.generated_count == 0
    assert resumed.skipped_count == T2VA_SMOKE
    assert calls == []
    resumed_manifest = json.loads(
        (output_dir / "rep1" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert resumed_manifest["cases"] == first_manifest["cases"]


def test_resume_rejects_stale_results_for_changed_prompts(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = _copy_pair(formal_pair, tmp_path / "pair")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner(calls)
    )
    output_dir = tmp_path / "outputs"
    run_formal_cases(
        PROTOCOL,
        registry_path,
        formal_path,
        endpoint="http://127.0.0.1:30010",
        output_dir=output_dir,
        repetition_id="rep1",
        split="smoke",
    )

    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    new_prompt = "a revised prompt"
    for case in formal["cases"]:
        if case["id"] == "smoke-001":
            case["prompt_sha256"] = hashlib.sha256(
                new_prompt.encode("utf-8")
            ).hexdigest()
    for case in registry["cases"]:
        if case["id"] == "smoke-001":
            case["prompt"] = new_prompt
    formal_path.write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    registry_path.write_text(json.dumps(registry))
    registry_path.chmod(0o600)

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
    assert resumed.generated_count == 1
    assert resumed.skipped_count == T2VA_SMOKE - 1
    assert len(calls) == 1
    regenerated = calls[0]["case"]
    assert isinstance(regenerated, dict)
    assert regenerated["id"] == "smoke-001"
    assert regenerated["prompt"] == new_prompt


def test_resume_reruns_corrupt_or_missing_artifacts(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner(calls)
    )
    output_dir = tmp_path / "outputs"
    run_formal_cases(
        PROTOCOL,
        registry_path,
        formal_path,
        endpoint="http://127.0.0.1:30010",
        output_dir=output_dir,
        repetition_id="rep1",
        split="smoke",
    )
    repetition_dir = output_dir / "rep1"
    (repetition_dir / "smoke-001.result.json").write_text("{", encoding="utf-8")
    (repetition_dir / "proto-smoke-004.mp4").unlink()
    (repetition_dir / "proto-smoke-007.mp4").write_bytes(b"tampered")

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
    assert resumed.generated_count == 3
    assert resumed.skipped_count == T2VA_SMOKE - 3


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
    formal_path, registry_path = _copy_pair(formal_pair, tmp_path / "pair")
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner([])
    )

    registry_path.chmod(0o644)
    with pytest.raises(ValidationError, match="group or other"):
        run_formal_cases(
            PROTOCOL,
            registry_path,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "o3",
            repetition_id="rep1",
        )

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


@pytest.mark.parametrize("repetition_id", ["../x", "a/b", "", ".hidden", "x x"])
def test_rejects_unsafe_repetition_ids(
    tmp_path: Path, formal_pair, repetition_id: str
) -> None:
    formal_path, registry_path = formal_pair

    with pytest.raises(ValidationError, match="repetition_id"):
        run_formal_cases(
            PROTOCOL,
            registry_path,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "outputs",
            repetition_id=repetition_id,
        )


def test_executed_digest_defense_blocks_persisting_results(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    inner = _fake_runner([])

    def wrong_digest(protocol_path: Path, case: dict[str, object], **kwargs):
        import dataclasses

        result = inner(protocol_path, case, **kwargs)
        return dataclasses.replace(result, prompt_sha256="0" * 64)

    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", wrong_digest
    )
    output_dir = tmp_path / "outputs"

    with pytest.raises(ValidationError, match="executed prompt digest"):
        run_formal_cases(
            PROTOCOL,
            registry_path,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=output_dir,
            repetition_id="rep1",
            split="smoke",
        )
    assert not (output_dir / "rep1" / "smoke-001.result.json").exists()


def test_failure_stops_and_keeps_partial_progress(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    calls: list[dict[str, object]] = []
    inner = _fake_runner(calls)

    def failing(protocol_path: Path, case: dict[str, object], **kwargs):
        if case["id"] == SMOKE_T2VA_IDS[2]:
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
    assert report.generated_count == 2
    assert report.skipped_count == 2
    assert len(calls) == 2


def test_run_supplied_case_submits_runner_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[dict[str, object]] = []
    responses = iter(({"id": "job-9"}, {"status": "completed"}))

    def request(_method: str, _url: str, payload, _timeout: float):
        if payload is not None:
            submitted.append(payload)
        return next(responses)

    monkeypatch.setattr("h3fast.benchmarks.client._request_json", request)
    monkeypatch.setattr(
        "h3fast.benchmarks.client._download_content",
        lambda _url, destination, _timeout: destination.write_bytes(b"video"),
    )
    monkeypatch.setattr("h3fast.benchmarks.client.time.sleep", lambda _seconds: None)

    result = run_supplied_case(
        PROTOCOL,
        {
            "id": "smoke-001",
            "prompt": "test prompt for smoke-001",
            "seed": 12000,
            "conditions": [],
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 4,
            "sigma_points": 50,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
        endpoint="http://127.0.0.1:30010",
        output_dir=tmp_path / "results",
    )

    assert result.case_id == "smoke-001"
    assert len(submitted) == 1
    payload = submitted[0]
    assert payload["prompt"] == "test prompt for smoke-001"
    assert payload["seed"] == 12000
    assert payload["seconds"] == 4
    assert payload["num_inference_steps"] == 50
    target = payload["target"]
    assert isinstance(target, dict)
    assert target["aspect_ratio"] == "16:9"
    assert target["short_edge"] == 768
    saved = json.loads(
        (
            tmp_path / "results" / "h3fast-phase1b-resident40-v1-smoke-001.json"
        ).read_text(encoding="utf-8")
    )
    assert "prompt" not in saved["request"]
    assert "test prompt for smoke-001" not in json.dumps(saved)

    with pytest.raises(ValidationError, match="non-empty id"):
        run_supplied_case(
            PROTOCOL,
            {"id": "", "prompt": "x"},
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "results",
        )


def test_fl2va_builds_keyframe_conditions(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner(calls)
    )

    report = run_formal_cases(
        PROTOCOL,
        registry_path,
        formal_path,
        endpoint="http://127.0.0.1:30010",
        output_dir=tmp_path / "outputs",
        repetition_id="fl2va-rep1",
        split="smoke",
        task="fl2va",
    )

    assert report.case_count == 3
    case = calls[0]["case"]
    assert isinstance(case, dict)
    conditions = case["conditions"]
    assert isinstance(conditions, list)
    assert len(conditions) == 2
    assert conditions[0]["type"] == "image"
    assert conditions[0]["role"] == "keyframe"
    assert conditions[0]["frame_index"] == 0
    assert conditions[1]["frame_index"] == -1
    assert conditions[0]["uri"].startswith("file:///")
    assert conditions[0]["uri"].endswith("frame-first.png")


def test_ref2va_builds_reference_conditions(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = formal_pair
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner(calls)
    )

    run_formal_cases(
        PROTOCOL,
        registry_path,
        formal_path,
        endpoint="http://127.0.0.1:30010",
        output_dir=tmp_path / "outputs",
        repetition_id="ref2va-rep1",
        split="smoke",
        task="ref2va",
    )

    seen_types = set()
    for call in calls:
        case = call["case"]
        assert isinstance(case, dict)
        for condition in case["conditions"]:
            assert condition["role"] == "reference"
            assert "frame_index" not in condition
            assert condition["uri"].startswith("file:///")
            seen_types.add(condition["type"])
    assert seen_types <= {"image", "video", "audio", "video_audio"}
    assert seen_types


def test_reference_tasks_reject_missing_or_miscounted_assets(
    tmp_path: Path, formal_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_path, registry_path = _copy_pair(formal_pair, tmp_path / "pair")
    monkeypatch.setattr(
        "h3fast.benchmarks.formal_runner._run_single_case", _fake_runner([])
    )

    # assets were not copied alongside the registry
    with pytest.raises(ValidationError, match="reference asset"):
        run_formal_cases(
            PROTOCOL,
            registry_path,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "o1",
            repetition_id="rep1",
            split="smoke",
            task="fl2va",
        )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for case in registry["cases"]:
        if case["task"] == "fl2va":
            case["references"] = case["references"][:1]
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    registry_path.chmod(0o600)
    shutil.copytree(formal_pair[0].parent / "assets", tmp_path / "pair" / "assets")
    with pytest.raises(ValidationError, match="two keyframe"):
        run_formal_cases(
            PROTOCOL,
            registry_path,
            formal_path,
            endpoint="http://127.0.0.1:30010",
            output_dir=tmp_path / "o2",
            repetition_id="rep1",
            split="smoke",
            task="fl2va",
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
    assert output["case_count"] == T2VA_SMOKE
    assert output["generated_count"] == T2VA_SMOKE
    assert str(tmp_path) not in json.dumps(output)
