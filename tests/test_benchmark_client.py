"""Tests for the asynchronous local video benchmark client."""

import io
import json
from pathlib import Path

import pytest

from h3fast.benchmarks.client import (
    _download_content,
    _load_performance_dump,
    _payload,
    _request_json,
    _validate_endpoint,
    run_case,
)
from h3fast.exceptions import ValidationError

_SGLANG_COMMIT = "6eb941a34cb100b708a42ed1d26d2bdefafbd01e"


def test_run_case_records_hashes_without_prompt(tmp_path: Path, monkeypatch) -> None:
    responses = iter(
        (
            {"id": "job-1"},
            {"status": "processing"},
            {"status": "completed"},
        )
    )
    monotonic = iter((10.0, 11.0, 12.0, 13.0))
    monkeypatch.setattr(
        "h3fast.benchmarks.client._request_json",
        lambda _method, _url, _payload, _timeout: next(responses),
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.client._download_content",
        lambda _url, destination, _timeout: destination.write_bytes(b"video"),
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.client.time.monotonic", lambda: next(monotonic)
    )
    monkeypatch.setattr("h3fast.benchmarks.client.time.sleep", lambda _seconds: None)
    output = tmp_path / "results"

    result = run_case(
        Path("benchmarks/protocol.yaml"),
        case_id="smoke-001",
        endpoint="http://127.0.0.1:30010",
        output_dir=output,
    )

    value = result.to_dict()
    assert value["status"] == "completed"
    assert value["elapsed_seconds"] == 3.0
    assert "prompt" not in value["request"]
    assert len(value["prompt_sha256"]) == 64
    saved = json.loads(
        (output / "h3fast-phase1a-baseline-v1-smoke-001.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["artifact"]["sha256"] == result.artifact_sha256


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:30010",
        "http://example.com:30010",
        "http://user:secret@localhost:30010",
        "http://localhost:30010?token=secret",
    ],
)
def test_validate_endpoint_rejects_nonlocal_or_credentialed_urls(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        _validate_endpoint(endpoint)


def test_run_case_reports_server_failure(tmp_path: Path, monkeypatch) -> None:
    responses = iter(({"id": "job-2"}, {"status": "failed", "error": "OOM"}))
    monotonic = iter((1.0, 2.0))
    monkeypatch.setattr(
        "h3fast.benchmarks.client._request_json",
        lambda _method, _url, _payload, _timeout: next(responses),
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.client.time.monotonic", lambda: next(monotonic)
    )

    with pytest.raises(ValidationError, match="OOM"):
        run_case(
            Path("benchmarks/protocol.yaml"),
            case_id="smoke-001",
            endpoint="http://localhost:30010",
            output_dir=tmp_path,
        )


def test_http_helpers_parse_json_and_write_content(tmp_path: Path, monkeypatch) -> None:
    responses = iter((io.BytesIO(b'{"id":"job-3"}'), io.BytesIO(b"content")))
    monkeypatch.setattr(
        "h3fast.benchmarks.client.urllib.request.urlopen",
        lambda _request, **_kwargs: next(responses),
    )

    response = _request_json(
        "POST", "http://127.0.0.1:30010/v1/videos", {"value": 1}, 5
    )
    destination = tmp_path / "content.mp4"
    _download_content("http://127.0.0.1:30010/v1/videos/job-3/content", destination, 5)

    assert response == {"id": "job-3"}
    assert destination.read_bytes() == b"content"


def test_request_json_rejects_non_object_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3fast.benchmarks.client.urllib.request.urlopen",
        lambda _request, **_kwargs: io.BytesIO(b"[]"),
    )

    with pytest.raises(ValidationError, match="non-object"):
        _request_json("GET", "http://127.0.0.1:30010/v1/videos/job", None, 5)


def test_run_case_records_server_performance_dump(tmp_path: Path, monkeypatch) -> None:
    performance = tmp_path / "server" / "metrics.json"
    responses = iter(({"id": "job-4"}, {"status": "completed"}))

    def request(*_args, **_kwargs):
        response = next(responses)
        if response.get("status") == "completed":
            performance.parent.mkdir()
            performance.write_text(
                json.dumps(
                    {
                        "request_id": "job-4",
                        "commit_hash": _SGLANG_COMMIT,
                        "total_duration_ms": 1200,
                        "steps": [
                            {"name": "TextEncodingStage", "duration_ms": 200},
                            {"name": "DenoisingStage", "duration_ms": 900},
                        ],
                        "denoise_steps_ms": [
                            {"step": 0, "duration_ms": 450},
                            {"step": 1, "duration_ms": 450},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            response.update(
                {
                    "inference_time_s": 1.2,
                    "peak_memory_mb": 1234,
                    "size": "1344x768",
                    "seconds": "5.166667",
                }
            )
        return response

    monotonic = iter((1.0, 2.0, 3.0))
    monkeypatch.setattr("h3fast.benchmarks.client._request_json", request)
    monkeypatch.setattr(
        "h3fast.benchmarks.client._download_content",
        lambda _url, destination, _timeout: destination.write_bytes(b"video"),
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.client.time.monotonic", lambda: next(monotonic)
    )

    result = run_case(
        Path("benchmarks/protocol.yaml"),
        case_id="smoke-001",
        endpoint="http://127.0.0.1:30010",
        output_dir=tmp_path / "result",
        server_perf_dump_path="/outputs/h3fast-metrics/metrics.json",
        performance_dump_path=performance,
    )

    assert result.server["peak_memory_mib"] == 1234.0
    assert result.server["media_contract"] == {
        "size": "1344x768",
        "seconds": "5.166667",
    }
    assert result.server["performance"]["pipeline_total_seconds"] == 1.2
    assert result.server["performance"]["sglang_commit"] == _SGLANG_COMMIT


def test_performance_dump_rejects_mismatched_job(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "request_id": "another-job",
                "commit_hash": _SGLANG_COMMIT,
                "total_duration_ms": 1,
                "steps": [{"name": "stage", "duration_ms": 1}],
                "denoise_steps_ms": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="request id"):
        _load_performance_dump(path, "expected-job", _SGLANG_COMMIT)


def test_performance_dump_rejects_commit_mismatch_and_empty_denoise(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.json"
    value = {
        "request_id": "job",
        "commit_hash": "0" * 40,
        "total_duration_ms": 1,
        "steps": [{"name": "stage", "duration_ms": 1}],
        "denoise_steps_ms": [{"step": 0, "duration_ms": 1}],
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValidationError, match="SGLang commit"):
        _load_performance_dump(path, "job", _SGLANG_COMMIT)

    value["commit_hash"] = _SGLANG_COMMIT
    value["denoise_steps_ms"] = []
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValidationError, match="non-empty"):
        _load_performance_dump(path, "job", _SGLANG_COMMIT)


def test_payload_restricts_server_performance_path() -> None:
    case = {"prompt": "safe", "duration_seconds": 5}

    with pytest.raises(ValidationError, match="direct file"):
        _payload(case, server_perf_dump_path="/tmp/metrics.json")
    with pytest.raises(ValidationError, match="direct file"):
        _payload(
            case,
            server_perf_dump_path="/outputs/h3fast-metrics/nested/metrics.json",
        )
