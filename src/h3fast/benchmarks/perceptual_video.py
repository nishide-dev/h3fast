"""LPIPS perceptual-video metric adapter with offline pinned weights.

The heavy dependencies (torch, torchvision, lpips) are imported lazily so
the package keeps a CPU-safe, dependency-free import path.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import subprocess
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from h3fast.benchmarks.quality import _resolve_executable, tool_version
from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

PERCEPTUAL_VIDEO_METHOD_ID = "lpips-alex-0.1.4-v1"
ALEXNET_BACKBONE_FILENAME = "alexnet-owt-7be5be79.pth"
ALEXNET_BACKBONE_SHA256 = (
    "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
)
_DECODE_TIMEOUT_SECONDS = 600


@dataclass(frozen=True, slots=True)
class PerceptualVideoReport:
    """Aggregate LPIPS distances between two decoded videos."""

    method_id: str
    frame_count: int
    width: int
    height: int
    mean_lpips: float
    max_lpips: float
    backbone_sha256: str
    torch_num_threads: int
    lpips_version: str
    torch_version: str
    ffmpeg_version: str

    def to_dict(self) -> dict[str, object]:
        """Return score metadata without any local paths."""
        return {
            "schema_version": "1.0",
            "method_id": self.method_id,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "mean_lpips": self.mean_lpips,
            "max_lpips": self.max_lpips,
            "backbone_sha256": self.backbone_sha256,
            "torch_num_threads": self.torch_num_threads,
            "lpips_version": self.lpips_version,
            "torch_version": self.torch_version,
            "ffmpeg_version": self.ffmpeg_version,
        }


def _verify_backbone(backbone_dir: Path, expected_sha256: str) -> Path:
    checkpoint = backbone_dir / "checkpoints" / ALEXNET_BACKBONE_FILENAME
    if not checkpoint.is_file():
        message = "pinned LPIPS backbone checkpoint is missing"
        raise ValidationError(message)
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as error:
        message = f"pinned LPIPS backbone checkpoint could not be read: {error}"
        raise ValidationError(message) from error
    if digest.hexdigest() != expected_sha256:
        message = "pinned LPIPS backbone checkpoint digest does not match"
        raise ValidationError(message)
    return checkpoint


def _import_scoring_dependencies() -> None:
    try:
        import lpips  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        message = (
            "perceptual-video scoring requires the quality-metrics dependency "
            f"group (torch, torchvision, lpips): {error}"
        )
        raise ValidationError(message) from error


def _load_lpips_model(backbone_dir: Path, expected_sha256: str):
    checkpoint = _verify_backbone(backbone_dir, expected_sha256)
    import torch
    import torch.hub

    checkpoint_dir = checkpoint.parent
    files_before = set(checkpoint_dir.iterdir())
    previous_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(backbone_dir))
    try:
        import lpips

        with warnings.catch_warnings():
            # The pinned lpips 0.1.4 still uses torchvision's legacy
            # `pretrained` argument; the mapped weights stay the pinned ones.
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            model = lpips.LPIPS(net="alex", verbose=False)
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        KeyError,
        pickle.UnpicklingError,
        zipfile.BadZipFile,
    ) as error:
        message = f"pinned LPIPS model could not be constructed offline: {error}"
        raise ValidationError(message) from error
    finally:
        torch.hub.set_dir(previous_hub_dir)
    if set(checkpoint_dir.iterdir()) != files_before:
        message = (
            "LPIPS model construction unexpectedly added checkpoint files;"
            " the verified backbone was not the one loaded"
        )
        raise ValidationError(message)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@dataclass(frozen=True, slots=True)
class _VideoContract:
    width: int
    height: int
    frame_rate: str


def _probe_video(path: Path, ffprobe: str) -> _VideoContract:
    executable = _resolve_executable(ffprobe)
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
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
        message = f"ffprobe failed for perceptual-video input: {error}"
        raise ValidationError(message) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown ffprobe error"
        message = f"ffprobe failed for perceptual-video input: {detail}"
        raise ValidationError(message)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        message = f"ffprobe returned invalid JSON: {error}"
        raise ValidationError(message) from error
    streams = value.get("streams") if isinstance(value, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        message = "perceptual-video input has no video stream"
        raise ValidationError(message)
    stream = streams[0]
    width = stream.get("width")
    height = stream.get("height")
    frame_rate = stream.get("r_frame_rate")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or not isinstance(frame_rate, str)
    ):
        message = "perceptual-video input has an invalid video stream contract"
        raise ValidationError(message)
    return _VideoContract(width=width, height=height, frame_rate=frame_rate)


def _require_matching_contract(
    baseline: _VideoContract, candidate: _VideoContract
) -> None:
    if (baseline.width, baseline.height) != (candidate.width, candidate.height):
        message = (
            "baseline and candidate videos have mismatched resolution: "
            f"{baseline.width}x{baseline.height} != {candidate.width}x{candidate.height}"
        )
        raise ValidationError(message)
    if baseline.frame_rate != candidate.frame_rate:
        message = (
            "baseline and candidate videos have mismatched frame rate: "
            f"{baseline.frame_rate} != {candidate.frame_rate}"
        )
        raise ValidationError(message)


def _decoded_frames(
    path: Path, ffmpeg: str, *, width: int, height: int
) -> Generator[bytes, None, None]:
    executable = _resolve_executable(ffmpeg)
    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    frame_bytes = width * height * 3
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.PIPE,
                stderr=errors,
            )
        except OSError as error:
            message = f"could not start ffmpeg video decode: {error}"
            raise ValidationError(message) from error
        assert process.stdout is not None
        try:
            while True:
                frame = process.stdout.read(frame_bytes)
                if not frame:
                    break
                if len(frame) != frame_bytes:
                    process.kill()
                    process.wait()
                    errors.seek(0)
                    detail = errors.read().decode("utf-8", errors="replace").strip()
                    message = "ffmpeg video decode returned a truncated frame" + (
                        f": {detail}" if detail else ""
                    )
                    raise ValidationError(message)
                yield frame
            return_code = process.wait(timeout=_DECODE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            message = "ffmpeg video decode timed out"
            raise ValidationError(message) from error
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        if return_code != 0:
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace").strip()
            message = f"ffmpeg video decode failed: {detail or 'unknown error'}"
            raise ValidationError(message)


def _frame_tensor(frame: bytes, *, width: int, height: int):
    import torch

    tensor = torch.frombuffer(bytearray(frame), dtype=torch.uint8)
    tensor = tensor.reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(torch.float32) / 127.5 - 1.0


def score_perceptual_video(
    baseline_path: Path,
    candidate_path: Path,
    *,
    backbone_dir: Path,
    expected_backbone_sha256: str = ALEXNET_BACKBONE_SHA256,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> PerceptualVideoReport:
    """Score frame-aligned LPIPS between a baseline and a candidate video."""
    _import_scoring_dependencies()
    ffmpeg_version = tool_version(ffmpeg)
    baseline_media = _probe_video(baseline_path, ffprobe)
    candidate_media = _probe_video(candidate_path, ffprobe)
    _require_matching_contract(baseline_media, candidate_media)
    width = baseline_media.width
    height = baseline_media.height

    model = _load_lpips_model(backbone_dir, expected_backbone_sha256)
    import lpips
    import torch

    frame_count = 0
    total = 0.0
    worst = 0.0
    baseline_frames = _decoded_frames(baseline_path, ffmpeg, width=width, height=height)
    candidate_frames = _decoded_frames(
        candidate_path, ffmpeg, width=width, height=height
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            while True:
                baseline_frame = next(baseline_frames, None)
                candidate_frame = next(candidate_frames, None)
                if baseline_frame is None and candidate_frame is None:
                    break
                if baseline_frame is None or candidate_frame is None:
                    remaining = (
                        candidate_frames if baseline_frame is None else baseline_frames
                    )
                    # Drain the longer stream so its own ffmpeg exit-code
                    # check runs and a decode failure is reported as such.
                    for _ in remaining:
                        pass
                    message = (
                        "baseline and candidate videos have mismatched frame count"
                    )
                    raise ValidationError(message)
                try:
                    distance = model(
                        _frame_tensor(baseline_frame, width=width, height=height),
                        _frame_tensor(candidate_frame, width=width, height=height),
                    )
                except (RuntimeError, ValueError) as error:
                    message = f"LPIPS forward pass failed: {error}"
                    raise ValidationError(message) from error
                value = float(distance.reshape(()).item())
                if not math.isfinite(value):
                    message = (
                        f"LPIPS produced a non-finite distance at frame {frame_count}"
                    )
                    raise ValidationError(message)
                frame_count += 1
                total += value
                worst = max(worst, value)
    finally:
        torch.set_num_threads(previous_threads)
        baseline_frames.close()
        candidate_frames.close()
    if frame_count == 0:
        message = "baseline and candidate videos contain no decodable frames"
        raise ValidationError(message)

    return PerceptualVideoReport(
        method_id=PERCEPTUAL_VIDEO_METHOD_ID,
        frame_count=frame_count,
        width=width,
        height=height,
        mean_lpips=total / frame_count,
        max_lpips=worst,
        backbone_sha256=expected_backbone_sha256,
        torch_num_threads=1,
        lpips_version=str(getattr(lpips, "__version__", "0.1.4")),
        torch_version=str(torch.__version__),
        ffmpeg_version=ffmpeg_version,
    )
