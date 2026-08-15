"""Tests for the asynchronous local video benchmark client."""

import io
import json
from pathlib import Path

import pytest

from h3fast.benchmarks.client import (
    _download_content,
    _request_json,
    _validate_endpoint,
    run_case,
)
from h3fast.exceptions import ValidationError


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
