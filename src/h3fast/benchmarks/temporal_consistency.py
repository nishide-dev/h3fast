"""Adjacent-frame LPIPS trajectory temporal-consistency adapter.

Contract ``adjacent-frame-lpips-trajectory-v1``: for each video the
trajectory is the sequence of LPIPS distances between adjacent decoded
frames. The score compares index-aligned trajectory steps between the
baseline and the candidate as absolute differences; scene cuts are not
excluded, so a cut preserved by the candidate cancels out while a moved
or removed cut surfaces as a delta. Inputs require at least two frames
and the same alignment contract as perceptual-video (identical
resolution, frame rate, and frame count; no temporal resampling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from h3fast.benchmarks.perceptual_video import (
    ALEXNET_BACKBONE_SHA256,
    _decoded_frames,
    _frame_tensor,
    _import_scoring_dependencies,
    _load_lpips_model,
    _probe_video,
    _require_matching_contract,
)
from h3fast.benchmarks.quality import tool_version
from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

TEMPORAL_CONSISTENCY_METHOD_ID = "adjacent-frame-lpips-trajectory-v1"


@dataclass(frozen=True, slots=True)
class TemporalConsistencyReport:
    """Aggregate adjacent-frame LPIPS trajectory comparison."""

    method_id: str
    frame_count: int
    step_count: int
    width: int
    height: int
    mean_abs_trajectory_delta: float
    max_abs_trajectory_delta: float
    baseline_mean_step_lpips: float
    candidate_mean_step_lpips: float
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
            "step_count": self.step_count,
            "width": self.width,
            "height": self.height,
            "mean_abs_trajectory_delta": self.mean_abs_trajectory_delta,
            "max_abs_trajectory_delta": self.max_abs_trajectory_delta,
            "baseline_mean_step_lpips": self.baseline_mean_step_lpips,
            "candidate_mean_step_lpips": self.candidate_mean_step_lpips,
            "backbone_sha256": self.backbone_sha256,
            "torch_num_threads": self.torch_num_threads,
            "lpips_version": self.lpips_version,
            "torch_version": self.torch_version,
            "ffmpeg_version": self.ffmpeg_version,
        }


def _finite_step(model, previous, current, step: int) -> float:  # noqa: ANN001
    try:
        distance = model(previous, current)
    except (RuntimeError, ValueError) as error:
        message = f"LPIPS forward pass failed: {error}"
        raise ValidationError(message) from error
    value = float(distance.reshape(()).item())
    if not math.isfinite(value):
        message = f"LPIPS produced a non-finite distance at step {step}"
        raise ValidationError(message)
    return value


def score_temporal_consistency(
    baseline_path: Path,
    candidate_path: Path,
    *,
    backbone_dir: Path,
    expected_backbone_sha256: str = ALEXNET_BACKBONE_SHA256,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> TemporalConsistencyReport:
    """Compare adjacent-frame LPIPS trajectories of two aligned videos."""
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
    step_count = 0
    delta_total = 0.0
    delta_worst = 0.0
    baseline_total = 0.0
    candidate_total = 0.0
    baseline_frames = _decoded_frames(baseline_path, ffmpeg, width=width, height=height)
    candidate_frames = _decoded_frames(
        candidate_path, ffmpeg, width=width, height=height
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        previous_baseline = None
        previous_candidate = None
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
                frame_count += 1
                baseline_tensor = _frame_tensor(
                    baseline_frame, width=width, height=height
                )
                candidate_tensor = _frame_tensor(
                    candidate_frame, width=width, height=height
                )
                if previous_baseline is not None and previous_candidate is not None:
                    baseline_step = _finite_step(
                        model, previous_baseline, baseline_tensor, step_count
                    )
                    candidate_step = _finite_step(
                        model, previous_candidate, candidate_tensor, step_count
                    )
                    step_count += 1
                    baseline_total += baseline_step
                    candidate_total += candidate_step
                    delta = abs(candidate_step - baseline_step)
                    delta_total += delta
                    delta_worst = max(delta_worst, delta)
                previous_baseline = baseline_tensor
                previous_candidate = candidate_tensor
    finally:
        torch.set_num_threads(previous_threads)
        baseline_frames.close()
        candidate_frames.close()
    if frame_count < 2:
        message = "temporal-consistency inputs must contain at least two frames each"
        raise ValidationError(message)

    return TemporalConsistencyReport(
        method_id=TEMPORAL_CONSISTENCY_METHOD_ID,
        frame_count=frame_count,
        step_count=step_count,
        width=width,
        height=height,
        mean_abs_trajectory_delta=delta_total / step_count,
        max_abs_trajectory_delta=delta_worst,
        baseline_mean_step_lpips=baseline_total / step_count,
        candidate_mean_step_lpips=candidate_total / step_count,
        backbone_sha256=expected_backbone_sha256,
        torch_num_threads=1,
        lpips_version=str(getattr(lpips, "__version__", "0.1.4")),
        torch_version=str(torch.__version__),
        ffmpeg_version=ffmpeg_version,
    )
