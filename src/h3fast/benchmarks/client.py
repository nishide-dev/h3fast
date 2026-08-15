"""Standard-library client for one pinned asynchronous video benchmark case."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from h3fast.benchmarks.protocol import validate_protocol
from h3fast.exceptions import ValidationError
from h3fast.manifest.checksums import sha256_file

if TYPE_CHECKING:
    from pathlib import Path

_TERMINAL_FAILURES = {"failed", "cancelled", "canceled", "expired"}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One end-to-end benchmark case result without prompt contents."""

    protocol_id: str
    case_id: str
    job_id: str
    started_at: str
    completed_at: str
    elapsed_seconds: float
    prompt_sha256: str
    request: dict[str, object]
    artifact_path: str
    artifact_size: int
    artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable result data."""
        return {
            "schema_version": "1.0",
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "job_id": self.job_id,
            "status": "completed",
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "prompt_sha256": self.prompt_sha256,
            "request": self.request,
            "artifact": {
                "path": self.artifact_path,
                "size": self.artifact_size,
                "sha256": self.artifact_sha256,
            },
        }


def _load_case(protocol_path: Path, case_id: str) -> tuple[str, dict[str, object]]:
    validate_protocol(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cases = protocol["cases"]
    for value in cases:
        if isinstance(value, dict) and value.get("id") == case_id:
            return str(protocol["protocol_id"]), value
    message = f"benchmark case is not defined: {case_id}"
    raise ValidationError(message)


def _validate_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
        message = "benchmark endpoint must be an HTTP loopback address"
        raise ValidationError(message)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        message = "benchmark endpoint cannot contain credentials, query, or fragment"
        raise ValidationError(message)
    return endpoint.rstrip("/")


def _request_json(
    method: str, url: str, payload: dict[str, object] | None, timeout: float
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        message = f"video API request failed: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict):
        message = "video API returned a non-object JSON response"
        raise ValidationError(message)
    return value


def _download_content(url: str, destination: Path, timeout: float) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, method="GET")  # noqa: S310
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,  # noqa: S310
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        message = f"video content download failed: {error}"
        raise ValidationError(message) from error


def _payload(case: dict[str, object]) -> tuple[dict[str, object], str]:
    prompt = case.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        message = "benchmark case prompt must be a non-empty string"
        raise ValidationError(message)
    duration = case.get("duration_seconds")
    payload: dict[str, object] = {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": prompt,
        "seconds": duration,
        "task": "t2va",
        "conditions": case.get("conditions", []),
        "target": {
            "short_edge": case.get("short_edge"),
            "aspect_ratio": case.get("aspect_ratio"),
            "duration_seconds": duration,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": case.get("sigma_points"),
        "flow_shift": case.get("flow_shift"),
        "audio_flow_shift": case.get("audio_flow_shift"),
        "seed": case.get("seed"),
        "quality": "lossless",
    }
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return payload, prompt_sha256


def run_case(
    protocol_path: Path,
    *,
    case_id: str,
    endpoint: str,
    output_dir: Path,
    poll_interval: float = 1.0,
    timeout: float = 7200.0,
) -> BenchmarkResult:
    """Submit, poll, and download one protocol case from a local SGLang server."""
    if poll_interval <= 0 or timeout <= 0:
        message = "poll interval and timeout must be positive"
        raise ValidationError(message)
    endpoint = _validate_endpoint(endpoint)
    protocol_id, case = _load_case(protocol_path, case_id)
    payload, prompt_sha256 = _payload(case)
    started_wall = datetime.now(UTC)
    started_monotonic = time.monotonic()
    response = _request_json("POST", f"{endpoint}/v1/videos", payload, timeout)
    job_id = response.get("id")
    if not isinstance(job_id, str) or not job_id:
        message = "video API submission did not return a job id"
        raise ValidationError(message)

    while True:
        elapsed = time.monotonic() - started_monotonic
        if elapsed >= timeout:
            message = f"video job timed out after {timeout:.1f} seconds"
            raise ValidationError(message)
        status_response = _request_json(
            "GET", f"{endpoint}/v1/videos/{job_id}", None, timeout
        )
        status = status_response.get("status")
        if status == "completed":
            break
        if status in _TERMINAL_FAILURES:
            detail = (
                status_response.get("error") or status_response.get("message") or status
            )
            message = f"video job {job_id} failed: {detail}"
            raise ValidationError(message)
        if not isinstance(status, str):
            message = f"video job {job_id} returned an invalid status"
            raise ValidationError(message)
        time.sleep(poll_interval)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{protocol_id}-{case_id}.mp4"
    _download_content(f"{endpoint}/v1/videos/{job_id}/content", artifact, timeout)
    completed = datetime.now(UTC)
    elapsed_seconds = time.monotonic() - started_monotonic
    request_metadata = dict(payload)
    request_metadata.pop("prompt")
    result = BenchmarkResult(
        protocol_id=protocol_id,
        case_id=case_id,
        job_id=job_id,
        started_at=started_wall.isoformat(),
        completed_at=completed.isoformat(),
        elapsed_seconds=elapsed_seconds,
        prompt_sha256=prompt_sha256,
        request=request_metadata,
        artifact_path=str(artifact.resolve()),
        artifact_size=artifact.stat().st_size,
        artifact_sha256=sha256_file(artifact),
    )
    result_path = output_dir / f"{protocol_id}-{case_id}.json"
    temporary = result_path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(result_path)
    return result
