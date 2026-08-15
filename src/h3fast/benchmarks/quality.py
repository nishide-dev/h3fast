"""Exact decoded-artifact quality references for placement-only experiments."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from h3fast.exceptions import ValidationError
from h3fast.manifest.checksums import sha256_file

QUALITY_METHOD_ID = "exact-decoded-artifact-v1"
_VIDEO_DECODE_FORMAT = "rgb24"
_AUDIO_DECODE_FORMAT = "pcm_s16le"
_HASH_PATTERN_LENGTH = 64
_NUMERIC_QUALITY_METRICS = (
    "artifact_size",
    "video_duration_seconds",
    "video_frames",
    "audio_duration_seconds",
    "audio_frames",
    "container_duration_seconds",
    "av_duration_drift_seconds",
)


@dataclass(frozen=True, slots=True)
class MediaObservation:
    """Normalized video, audio, and container metadata for one MP4."""

    video_codec: str
    video_pixel_format: str
    width: int
    height: int
    frame_rate: str
    video_duration_seconds: float
    video_frames: int
    audio_codec: str
    audio_sample_rate_hz: int
    audio_channels: int
    audio_duration_seconds: float
    audio_frames: int
    container_format: str
    container_duration_seconds: float

    @property
    def av_duration_drift_seconds(self) -> float:
        """Return the absolute audio/video duration difference."""
        return abs(self.audio_duration_seconds - self.video_duration_seconds)

    def to_dict(self) -> dict[str, object]:
        """Return stable machine-readable media metadata."""
        return {
            "video": {
                "codec": self.video_codec,
                "pixel_format": self.video_pixel_format,
                "width": self.width,
                "height": self.height,
                "frame_rate": self.frame_rate,
                "duration_seconds": self.video_duration_seconds,
                "frames": self.video_frames,
            },
            "audio": {
                "codec": self.audio_codec,
                "sample_rate_hz": self.audio_sample_rate_hz,
                "channels": self.audio_channels,
                "duration_seconds": self.audio_duration_seconds,
                "frames": self.audio_frames,
            },
            "container": {
                "format": self.container_format,
                "duration_seconds": self.container_duration_seconds,
            },
            "av_duration_drift_seconds": self.av_duration_drift_seconds,
        }


@dataclass(frozen=True, slots=True)
class ArtifactObservation:
    """Container and independently decoded content identity for one artifact."""

    artifact_sha256: str
    artifact_size: int
    video_decoded_sha256: str
    audio_decoded_sha256: str
    media: MediaObservation

    def to_dict(self) -> dict[str, object]:
        """Return stable machine-readable artifact metadata."""
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_size": self.artifact_size,
            "video_decoded_sha256": self.video_decoded_sha256,
            "audio_decoded_sha256": self.audio_decoded_sha256,
            "media": self.media.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MeasuredArtifact:
    """One measured suite run and its quality-relevant provenance."""

    label: str
    prompt_sha256: str
    request_sha256: str
    sglang_commit: str
    path: Path
    declared_sha256: str
    declared_size: int


@dataclass(frozen=True, slots=True)
class ProtocolIdentity:
    """Protocol fields that must not change in a quality comparison."""

    repository: str
    revision: str
    task_family: str
    task: str
    case_id: str
    prompt_sha256: str
    case_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return identity fields without the prompt contents."""
        return {
            "repository": self.repository,
            "revision": self.revision,
            "task_family": self.task_family,
            "task": self.task,
            "case_id": self.case_id,
            "prompt_sha256": self.prompt_sha256,
            "case_sha256": self.case_sha256,
        }


ArtifactInspector = Callable[[Path, str, str], ArtifactObservation]
VersionReader = Callable[[str], str]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"could not read {description} {path}: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict):
        message = f"{description} root must be an object"
        raise ValidationError(message)
    return value


def _required_string(mapping: dict[str, object], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        message = f"quality metadata field {field!r} must be a non-empty string"
        raise ValidationError(message)
    return value


def _required_int(mapping: dict[str, object], field: str, *, positive: bool) -> int:
    value = mapping.get(field)
    minimum = 1 if positive else 0
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        message = f"quality metadata field {field!r} must be an integer >= {minimum}"
        raise ValidationError(message)
    return value


def _required_float(mapping: dict[str, object], field: str) -> float:
    value = mapping.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        message = f"quality metadata field {field!r} must be finite and non-negative"
        raise ValidationError(message)
    return float(value)


def _validate_sha256(value: str, field: str) -> str:
    if len(value) != _HASH_PATTERN_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        message = f"quality metadata field {field!r} must be a lowercase SHA-256"
        raise ValidationError(message)
    return value


def _resolve_artifact_path(raw_path: str, suite_path: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = suite_path.parent / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        message = f"quality artifact is unavailable: {path}: {error}"
        raise ValidationError(message) from error
    if not resolved.is_file():
        message = f"quality artifact is not a file: {resolved}"
        raise ValidationError(message)
    return resolved


def _load_measured_artifacts(
    suite_path: Path,
) -> tuple[dict[str, object], list[MeasuredArtifact]]:
    suite = _load_object(suite_path, "benchmark suite")
    if suite.get("schema_version") != "1.0" or suite.get("status") != "completed":
        message = "quality reference requires a completed schema 1.0 benchmark suite"
        raise ValidationError(message)
    protocol_id = _required_string(suite, "protocol_id")
    case_id = _required_string(suite, "case_id")
    measured_count = _required_int(suite, "measured_runs", positive=True)
    runs = suite.get("runs")
    if not isinstance(runs, list):
        message = "benchmark suite runs must be an array"
        raise ValidationError(message)

    measured: list[MeasuredArtifact] = []
    for run in runs:
        if not isinstance(run, dict) or run.get("kind") != "measured":
            continue
        if run.get("status") != "completed":
            message = "quality reference cannot include failed measured runs"
            raise ValidationError(message)
        label = _required_string(run, "label")
        result = run.get("result")
        if not isinstance(result, dict):
            message = f"measured run {label!r} is missing its result"
            raise ValidationError(message)
        if result.get("protocol_id") != protocol_id or result.get("case_id") != case_id:
            message = f"measured run {label!r} does not match its suite identity"
            raise ValidationError(message)
        prompt_sha256 = _validate_sha256(
            _required_string(result, "prompt_sha256"), "prompt_sha256"
        )
        request = result.get("request")
        if not isinstance(request, dict):
            message = f"measured run {label!r} request metadata must be an object"
            raise ValidationError(message)
        artifact = result.get("artifact")
        server = result.get("server")
        if not isinstance(artifact, dict) or not isinstance(server, dict):
            message = f"measured run {label!r} is missing artifact or server metadata"
            raise ValidationError(message)
        performance = server.get("performance")
        if not isinstance(performance, dict):
            message = f"measured run {label!r} is missing performance provenance"
            raise ValidationError(message)
        path = _resolve_artifact_path(_required_string(artifact, "path"), suite_path)
        declared_sha256 = _validate_sha256(
            _required_string(artifact, "sha256"), "artifact.sha256"
        )
        declared_size = _required_int(artifact, "size", positive=True)
        measured.append(
            MeasuredArtifact(
                label=label,
                prompt_sha256=prompt_sha256,
                request_sha256=_canonical_sha256(request),
                sglang_commit=_required_string(performance, "sglang_commit"),
                path=path,
                declared_sha256=declared_sha256,
                declared_size=declared_size,
            )
        )

    if len(measured) != measured_count:
        message = "benchmark suite measured run count does not match its completed runs"
        raise ValidationError(message)
    if len({run.label for run in measured}) != len(measured):
        message = "benchmark suite measured run labels must be unique"
        raise ValidationError(message)
    return suite, measured


def _protocol_identity(protocol_path: Path, case_id: str) -> ProtocolIdentity:
    protocol = _load_object(protocol_path, "benchmark protocol")
    base_model = protocol.get("base_model")
    cases = protocol.get("cases")
    if not isinstance(base_model, dict) or not isinstance(cases, list):
        message = "quality protocol is missing base_model or cases"
        raise ValidationError(message)
    case: dict[str, object] | None = None
    for value in cases:
        if isinstance(value, dict) and value.get("id") == case_id:
            case = value
            break
    if case is None:
        message = f"quality protocol does not define case {case_id!r}"
        raise ValidationError(message)
    prompt = case.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        message = "quality protocol case prompt must be a non-empty string"
        raise ValidationError(message)
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    redacted_case = dict(case)
    redacted_case["prompt"] = {"sha256": prompt_sha256}
    return ProtocolIdentity(
        repository=_required_string(base_model, "repository"),
        revision=_required_string(base_model, "revision"),
        task_family=_required_string(base_model, "task_family"),
        task=_required_string(base_model, "task"),
        case_id=case_id,
        prompt_sha256=prompt_sha256,
        case_sha256=_canonical_sha256(redacted_case),
    )


def _resolve_executable(value: str) -> str:
    executable = shutil.which(value)
    if executable is None:
        message = f"required quality tool is unavailable: {value}"
        raise ValidationError(message)
    return executable


def tool_version(executable: str) -> str:
    """Return the first version line for a local media tool."""
    resolved = _resolve_executable(executable)
    try:
        result = subprocess.run(  # noqa: S603
            [resolved, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        message = f"could not query quality tool version: {error}"
        raise ValidationError(message) from error
    if result.returncode != 0 or not result.stdout.splitlines():
        message = f"quality tool version query failed: {resolved}"
        raise ValidationError(message)
    return result.stdout.splitlines()[0].strip()


def _probe_media(path: Path, ffprobe: str) -> MediaObservation:
    executable = _resolve_executable(ffprobe)
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,pix_fmt,width,height,r_frame_rate,"
            "duration,nb_frames,sample_rate,channels"
        ),
        "-show_entries",
        "format=format_name,duration,size",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        message = f"ffprobe failed for quality artifact: {error}"
        raise ValidationError(message) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown ffprobe error"
        message = f"ffprobe failed for quality artifact: {detail}"
        raise ValidationError(message)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        message = f"ffprobe returned invalid JSON: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict):
        message = "ffprobe result root must be an object"
        raise ValidationError(message)
    streams = value.get("streams")
    container = value.get("format")
    if not isinstance(streams, list) or not isinstance(container, dict):
        message = "ffprobe result is missing streams or format"
        raise ValidationError(message)
    videos = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    audios = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    if len(videos) != 1 or len(audios) != 1:
        message = "quality artifact must contain exactly one video and one audio stream"
        raise ValidationError(message)
    video = videos[0]
    audio = audios[0]

    def integer(mapping: dict[str, object], field: str) -> int:
        raw = mapping.get(field)
        try:
            value_int = int(raw) if isinstance(raw, (str, int)) else -1
        except ValueError:
            value_int = -1
        if value_int <= 0:
            message = f"ffprobe field {field!r} must be a positive integer"
            raise ValidationError(message)
        return value_int

    def number(mapping: dict[str, object], field: str) -> float:
        raw = mapping.get(field)
        try:
            value_float = float(raw) if isinstance(raw, (str, int, float)) else -1.0
        except ValueError:
            value_float = -1.0
        if not math.isfinite(value_float) or value_float < 0:
            message = f"ffprobe field {field!r} must be finite and non-negative"
            raise ValidationError(message)
        return value_float

    return MediaObservation(
        video_codec=_required_string(video, "codec_name"),
        video_pixel_format=_required_string(video, "pix_fmt"),
        width=integer(video, "width"),
        height=integer(video, "height"),
        frame_rate=_required_string(video, "r_frame_rate"),
        video_duration_seconds=number(video, "duration"),
        video_frames=integer(video, "nb_frames"),
        audio_codec=_required_string(audio, "codec_name"),
        audio_sample_rate_hz=integer(audio, "sample_rate"),
        audio_channels=integer(audio, "channels"),
        audio_duration_seconds=number(audio, "duration"),
        audio_frames=integer(audio, "nb_frames"),
        container_format=_required_string(container, "format_name"),
        container_duration_seconds=number(container, "duration"),
    )


def _decoded_stream_sha256(
    path: Path,
    ffmpeg: str,
    *,
    stream: str,
    media: MediaObservation,
) -> str:
    executable = _resolve_executable(ffmpeg)
    if stream == "video":
        output_arguments = [
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            _VIDEO_DECODE_FORMAT,
        ]
    elif stream == "audio":
        output_arguments = [
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-f",
            "s16le",
            "-acodec",
            _AUDIO_DECODE_FORMAT,
            "-ar",
            str(media.audio_sample_rate_hz),
            "-ac",
            str(media.audio_channels),
        ]
    else:
        message = f"unsupported decoded quality stream: {stream}"
        raise ValidationError(message)
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        *output_arguments,
        "pipe:1",
    ]
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.PIPE,
                stderr=errors,
            )
        except OSError as error:
            message = f"could not start ffmpeg {stream} decode: {error}"
            raise ValidationError(message) from error
        assert process.stdout is not None
        try:
            while chunk := process.stdout.read(1024 * 1024):
                digest.update(chunk)
            return_code = process.wait(timeout=600)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            message = f"ffmpeg {stream} decode timed out"
            raise ValidationError(message) from error
        if return_code != 0:
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace").strip()
            message = f"ffmpeg {stream} decode failed: {detail or 'unknown error'}"
            raise ValidationError(message)
    return digest.hexdigest()


def inspect_artifact(path: Path, ffmpeg: str, ffprobe: str) -> ArtifactObservation:
    """Inspect container metadata and hash independently decoded streams."""
    media = _probe_media(path, ffprobe)
    return ArtifactObservation(
        artifact_sha256=sha256_file(path),
        artifact_size=path.stat().st_size,
        video_decoded_sha256=_decoded_stream_sha256(
            path, ffmpeg, stream="video", media=media
        ),
        audio_decoded_sha256=_decoded_stream_sha256(
            path, ffmpeg, stream="audio", media=media
        ),
        media=media,
    )


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _statistics(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        message = "quality statistics require finite non-negative values"
        raise ValidationError(message)
    return {
        "min": min(values),
        "p5": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _observation_statistics(
    observations: list[ArtifactObservation],
) -> dict[str, dict[str, float]]:
    return {
        "artifact_size": _statistics(
            [float(observation.artifact_size) for observation in observations]
        ),
        "video_duration_seconds": _statistics(
            [observation.media.video_duration_seconds for observation in observations]
        ),
        "video_frames": _statistics(
            [float(observation.media.video_frames) for observation in observations]
        ),
        "audio_duration_seconds": _statistics(
            [observation.media.audio_duration_seconds for observation in observations]
        ),
        "audio_frames": _statistics(
            [float(observation.media.audio_frames) for observation in observations]
        ),
        "container_duration_seconds": _statistics(
            [
                observation.media.container_duration_seconds
                for observation in observations
            ]
        ),
        "av_duration_drift_seconds": _statistics(
            [
                observation.media.av_duration_drift_seconds
                for observation in observations
            ]
        ),
    }


def _observation_numeric_values(
    observation: ArtifactObservation,
) -> dict[str, float]:
    return {
        "artifact_size": float(observation.artifact_size),
        "video_duration_seconds": observation.media.video_duration_seconds,
        "video_frames": float(observation.media.video_frames),
        "audio_duration_seconds": observation.media.audio_duration_seconds,
        "audio_frames": float(observation.media.audio_frames),
        "container_duration_seconds": observation.media.container_duration_seconds,
        "av_duration_drift_seconds": observation.media.av_duration_drift_seconds,
    }


def _baseline_envelopes(value: object) -> dict[str, tuple[float, float]]:
    if not isinstance(value, dict):
        message = "quality reference baseline_statistics must be an object"
        raise ValidationError(message)
    envelopes: dict[str, tuple[float, float]] = {}
    for metric in _NUMERIC_QUALITY_METRICS:
        statistics = value.get(metric)
        if not isinstance(statistics, dict):
            message = f"quality reference statistics are missing {metric!r}"
            raise ValidationError(message)
        ordered = [
            _required_float(statistics, field)
            for field in ("min", "p5", "p50", "p95", "max")
        ]
        if ordered != sorted(ordered):
            message = f"quality reference statistics are not ordered for {metric!r}"
            raise ValidationError(message)
        envelopes[metric] = (ordered[0], ordered[-1])
    return envelopes


def _envelope_check(
    name: str, envelope: tuple[float, float], actual: float
) -> dict[str, object]:
    minimum, maximum = envelope
    return {
        "name": f"{name}_baseline_envelope",
        "passed": minimum <= actual <= maximum,
        "expected": {"min": minimum, "max": maximum},
        "actual": actual,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_quality_reference(
    suite_path: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    reference_id: str,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    artifact_inspector: ArtifactInspector = inspect_artifact,
    version_reader: VersionReader = tool_version,
) -> dict[str, object]:
    """Build a redacted exact reference from stable measured baseline runs."""
    if not reference_id or any(character.isspace() for character in reference_id):
        message = "quality reference id must be non-empty and contain no whitespace"
        raise ValidationError(message)
    suite, measured = _load_measured_artifacts(suite_path)
    if len(measured) < 3:
        message = "exact quality reference requires at least three measured runs"
        raise ValidationError(message)
    protocol_id = _required_string(suite, "protocol_id")
    case_id = _required_string(suite, "case_id")
    protocol = _load_object(protocol_path, "benchmark protocol")
    if protocol.get("protocol_id") != protocol_id:
        message = "quality protocol id does not match the baseline suite"
        raise ValidationError(message)
    identity = _protocol_identity(protocol_path, case_id)

    observations: list[ArtifactObservation] = []
    for run in measured:
        if sha256_file(run.path) != run.declared_sha256:
            message = f"measured artifact hash does not match run {run.label!r}"
            raise ValidationError(message)
        if run.path.stat().st_size != run.declared_size:
            message = f"measured artifact size does not match run {run.label!r}"
            raise ValidationError(message)
        observation = artifact_inspector(run.path, ffmpeg, ffprobe)
        if (
            observation.artifact_sha256 != run.declared_sha256
            or observation.artifact_size != run.declared_size
        ):
            message = f"artifact inspector disagrees with run {run.label!r}"
            raise ValidationError(message)
        observations.append(observation)

    provenance = {
        (run.prompt_sha256, run.request_sha256, run.sglang_commit) for run in measured
    }
    if len(provenance) != 1:
        message = "baseline measured runs have inconsistent quality provenance"
        raise ValidationError(message)
    if (
        len({_canonical_sha256(observation.to_dict()) for observation in observations})
        != 1
    ):
        message = "exact quality reference requires bitwise-stable decoded artifacts"
        raise ValidationError(message)
    prompt_sha256, request_sha256, sglang_commit = provenance.pop()
    if prompt_sha256 != identity.prompt_sha256:
        message = "baseline prompt digest does not match the quality protocol"
        raise ValidationError(message)
    expected = observations[0]
    value: dict[str, object] = {
        "schema_version": "1.0",
        "reference_id": reference_id,
        "method": {
            "id": QUALITY_METHOD_ID,
            "profile": "exact",
            "video_decode_format": _VIDEO_DECODE_FORMAT,
            "audio_decode_format": _AUDIO_DECODE_FORMAT,
            "ffmpeg_version": version_reader(ffmpeg),
            "ffprobe_version": version_reader(ffprobe),
        },
        "source": {
            "protocol_id": protocol_id,
            "protocol_identity": identity.to_dict(),
            "suite_completed_at": _required_string(suite, "completed_at"),
            "measured_runs": len(measured),
            "prompt_sha256": prompt_sha256,
            "request_sha256": request_sha256,
            "sglang_commit": sglang_commit,
        },
        "expected": expected.to_dict(),
        "baseline_statistics": _observation_statistics(observations),
        "limitations": [
            "single-case placement-only regression gate",
            "not evidence of general perceptual quality equivalence",
            "not a lossless or support-tier claim",
        ],
    }
    _write_json(output_path, value)
    return value


def _media_from_dict(value: object) -> MediaObservation:
    if not isinstance(value, dict):
        message = "quality reference expected.media must be an object"
        raise ValidationError(message)
    video = value.get("video")
    audio = value.get("audio")
    container = value.get("container")
    if (
        not isinstance(video, dict)
        or not isinstance(audio, dict)
        or not isinstance(container, dict)
    ):
        message = "quality reference media streams must be objects"
        raise ValidationError(message)
    return MediaObservation(
        video_codec=_required_string(video, "codec"),
        video_pixel_format=_required_string(video, "pixel_format"),
        width=_required_int(video, "width", positive=True),
        height=_required_int(video, "height", positive=True),
        frame_rate=_required_string(video, "frame_rate"),
        video_duration_seconds=_required_float(video, "duration_seconds"),
        video_frames=_required_int(video, "frames", positive=True),
        audio_codec=_required_string(audio, "codec"),
        audio_sample_rate_hz=_required_int(audio, "sample_rate_hz", positive=True),
        audio_channels=_required_int(audio, "channels", positive=True),
        audio_duration_seconds=_required_float(audio, "duration_seconds"),
        audio_frames=_required_int(audio, "frames", positive=True),
        container_format=_required_string(container, "format"),
        container_duration_seconds=_required_float(container, "duration_seconds"),
    )


def _observation_from_reference(value: object) -> ArtifactObservation:
    if not isinstance(value, dict):
        message = "quality reference expected must be an object"
        raise ValidationError(message)
    return ArtifactObservation(
        artifact_sha256=_validate_sha256(
            _required_string(value, "artifact_sha256"), "artifact_sha256"
        ),
        artifact_size=_required_int(value, "artifact_size", positive=True),
        video_decoded_sha256=_validate_sha256(
            _required_string(value, "video_decoded_sha256"),
            "video_decoded_sha256",
        ),
        audio_decoded_sha256=_validate_sha256(
            _required_string(value, "audio_decoded_sha256"),
            "audio_decoded_sha256",
        ),
        media=_media_from_dict(value.get("media")),
    )


def _protocol_identity_from_reference(value: object) -> ProtocolIdentity:
    if not isinstance(value, dict):
        message = "quality reference protocol_identity must be an object"
        raise ValidationError(message)
    return ProtocolIdentity(
        repository=_required_string(value, "repository"),
        revision=_required_string(value, "revision"),
        task_family=_required_string(value, "task_family"),
        task=_required_string(value, "task"),
        case_id=_required_string(value, "case_id"),
        prompt_sha256=_validate_sha256(
            _required_string(value, "prompt_sha256"), "prompt_sha256"
        ),
        case_sha256=_validate_sha256(
            _required_string(value, "case_sha256"), "case_sha256"
        ),
    )


def _check(name: str, expected: object, actual: object) -> dict[str, object]:
    return {
        "name": name,
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def check_quality(
    reference_path: Path,
    suite_path: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    artifact_inspector: ArtifactInspector = inspect_artifact,
    version_reader: VersionReader = tool_version,
) -> dict[str, object]:
    """Check a candidate measured suite against one exact reference."""
    reference = _load_object(reference_path, "quality reference")
    if reference.get("schema_version") != "1.0":
        message = "unsupported quality reference schema_version"
        raise ValidationError(message)
    method = reference.get("method")
    source = reference.get("source")
    if not isinstance(method, dict) or not isinstance(source, dict):
        message = "quality reference is missing method or source"
        raise ValidationError(message)
    if method.get("id") != QUALITY_METHOD_ID or method.get("profile") != "exact":
        message = "unsupported quality reference method"
        raise ValidationError(message)
    if (
        method.get("video_decode_format") != _VIDEO_DECODE_FORMAT
        or method.get("audio_decode_format") != _AUDIO_DECODE_FORMAT
    ):
        message = "unsupported quality reference decode format"
        raise ValidationError(message)
    expected = _observation_from_reference(reference.get("expected"))
    envelopes = _baseline_envelopes(reference.get("baseline_statistics"))
    expected_identity = _protocol_identity_from_reference(
        source.get("protocol_identity")
    )

    suite, measured = _load_measured_artifacts(suite_path)
    case_id = _required_string(suite, "case_id")
    protocol = _load_object(protocol_path, "benchmark protocol")
    if protocol.get("protocol_id") != suite.get("protocol_id"):
        message = "candidate protocol id does not match the benchmark suite"
        raise ValidationError(message)
    candidate_identity = _protocol_identity(protocol_path, case_id)
    if candidate_identity != expected_identity:
        message = "candidate protocol identity does not match the quality reference"
        raise ValidationError(message)
    expected_runs = _required_int(source, "measured_runs", positive=True)
    if len(measured) != expected_runs:
        message = "candidate measured run count does not match the quality reference"
        raise ValidationError(message)
    expected_prompt = _validate_sha256(
        _required_string(source, "prompt_sha256"), "prompt_sha256"
    )
    expected_request = _validate_sha256(
        _required_string(source, "request_sha256"), "request_sha256"
    )
    expected_sglang = _required_string(source, "sglang_commit")
    candidate_ffmpeg_version = version_reader(ffmpeg)
    candidate_ffprobe_version = version_reader(ffprobe)
    environment_checks = [
        _check(
            "ffmpeg_version",
            _required_string(method, "ffmpeg_version"),
            candidate_ffmpeg_version,
        ),
        _check(
            "ffprobe_version",
            _required_string(method, "ffprobe_version"),
            candidate_ffprobe_version,
        ),
    ]

    observations: list[ArtifactObservation] = []
    run_reports: list[dict[str, object]] = []
    failed_counts: list[int] = []
    for run in measured:
        observation = artifact_inspector(run.path, ffmpeg, ffprobe)
        observations.append(observation)
        checks = [
            _check(
                "declared_artifact_sha256",
                run.declared_sha256,
                observation.artifact_sha256,
            ),
            _check(
                "declared_artifact_size", run.declared_size, observation.artifact_size
            ),
            _check("prompt_sha256", expected_prompt, run.prompt_sha256),
            _check("request_sha256", expected_request, run.request_sha256),
            _check("sglang_commit", expected_sglang, run.sglang_commit),
            _check(
                "artifact_sha256", expected.artifact_sha256, observation.artifact_sha256
            ),
            _check("artifact_size", expected.artifact_size, observation.artifact_size),
            _check(
                "video_decoded_sha256",
                expected.video_decoded_sha256,
                observation.video_decoded_sha256,
            ),
            _check(
                "audio_decoded_sha256",
                expected.audio_decoded_sha256,
                observation.audio_decoded_sha256,
            ),
            _check(
                "video_media",
                expected.media.to_dict()["video"],
                observation.media.to_dict()["video"],
            ),
            _check(
                "audio_media",
                expected.media.to_dict()["audio"],
                observation.media.to_dict()["audio"],
            ),
            _check(
                "container_media",
                expected.media.to_dict()["container"],
                observation.media.to_dict()["container"],
            ),
            _check(
                "av_duration_drift_seconds",
                expected.media.av_duration_drift_seconds,
                observation.media.av_duration_drift_seconds,
            ),
        ]
        checks.extend(
            _envelope_check(metric, envelopes[metric], actual)
            for metric, actual in _observation_numeric_values(observation).items()
        )
        run_reports.append(
            {
                "label": run.label,
                "passed": all(bool(check["passed"]) for check in checks),
                "checks": checks,
            }
        )
        failed_counts.append(sum(not bool(check["passed"]) for check in checks))
    worst_index = max(range(len(run_reports)), key=failed_counts.__getitem__)
    environment_failures = sum(
        not bool(check["passed"]) for check in environment_checks
    )
    passed = environment_failures == 0 and all(
        bool(run_report["passed"]) for run_report in run_reports
    )
    if environment_failures > failed_counts[worst_index]:
        worst_case: dict[str, object] = {
            "scope": "environment",
            "failed_checks": environment_failures,
        }
    else:
        worst_case = {
            "scope": "run",
            "label": run_reports[worst_index]["label"],
            "failed_checks": failed_counts[worst_index],
        }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "reference_id": _required_string(reference, "reference_id"),
        "method_id": QUALITY_METHOD_ID,
        "status": "passed" if passed else "failed",
        "recorded_at": datetime.now(UTC).isoformat(),
        "candidate": {
            "protocol_id": _required_string(suite, "protocol_id"),
            "protocol_identity": candidate_identity.to_dict(),
            "case_id": case_id,
            "measured_runs": len(measured),
            "ffmpeg_version": candidate_ffmpeg_version,
            "ffprobe_version": candidate_ffprobe_version,
        },
        "environment_checks": environment_checks,
        "runs": run_reports,
        "candidate_statistics": _observation_statistics(observations),
        "worst_case": worst_case,
        "limitations": reference.get("limitations", []),
    }
    _write_json(output_path, report)
    return report
