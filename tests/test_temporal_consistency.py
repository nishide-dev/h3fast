"""Tests for the adjacent-frame LPIPS temporal-consistency adapter."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from h3fast.exceptions import ValidationError

pytest.importorskip("torch")
pytest.importorskip("lpips")

from h3fast.benchmarks import score_temporal_consistency
from h3fast.benchmarks.perceptual_video import ALEXNET_BACKBONE_FILENAME

_WIDTH = 64
_HEIGHT = 64
_FRAMES = 8
_RATE = 8
_FRAME_BYTES = _WIDTH * _HEIGHT * 3


def _static_frames(count: int = _FRAMES, *, flicker: int = 0) -> bytes:
    import numpy as np

    generator = np.random.default_rng(20260816)
    frame = generator.integers(0, 256, size=(1, _HEIGHT, _WIDTH, 3), dtype=np.int16)
    frames = np.repeat(frame, count, axis=0)
    if flicker:
        noise = np.random.default_rng(7).integers(
            -1, 2, size=frames.shape, dtype=np.int16
        )
        signs = np.array([1 if i % 2 else -1 for i in range(count)]).reshape(
            count, 1, 1, 1
        )
        frames = frames + noise * signs * flicker
    return np.clip(frames, 0, 255).astype(np.uint8).tobytes()


def _cut_frames(*, preserve_cut: bool) -> bytes:
    import numpy as np

    first = np.random.default_rng(1).integers(
        0, 256, size=(1, _HEIGHT, _WIDTH, 3), dtype=np.uint8
    )
    second = np.random.default_rng(2).integers(
        0, 256, size=(1, _HEIGHT, _WIDTH, 3), dtype=np.uint8
    )
    half = _FRAMES // 2
    if preserve_cut:
        frames = np.concatenate(
            [np.repeat(first, half, axis=0), np.repeat(second, half, axis=0)]
        )
    else:
        frames = np.repeat(first, _FRAMES, axis=0)
    return frames.tobytes()


def _encode(path: Path, data: bytes, *, rate: int = _RATE) -> Path:
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{_WIDTH}x{_HEIGHT}",
            "-r",
            str(rate),
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(path),
        ],
        check=True,
        input=data,
    )
    return path


@pytest.fixture(scope="session")
def backbone_dir(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    import torch
    from torchvision.models import alexnet

    root = tmp_path_factory.mktemp("hub-temporal")
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir()
    torch.manual_seed(0)
    weights = alexnet(weights=None).state_dict()
    checkpoint = checkpoint_dir / ALEXNET_BACKBONE_FILENAME
    torch.save(weights, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return root, digest


def _score(baseline: Path, candidate: Path, backbone: tuple[Path, str]):
    root, digest = backbone
    return score_temporal_consistency(
        baseline,
        candidate,
        backbone_dir=root,
        expected_backbone_sha256=digest,
    )


def test_identical_videos_have_zero_trajectory_delta(
    tmp_path: Path, backbone_dir
) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _cut_frames(preserve_cut=True))
    candidate = _encode(tmp_path / "candidate.mkv", _cut_frames(preserve_cut=True))

    report = _score(baseline, candidate, backbone_dir)

    assert report.frame_count == _FRAMES
    assert report.step_count == _FRAMES - 1
    assert report.mean_abs_trajectory_delta == 0.0
    assert report.max_abs_trajectory_delta == 0.0
    assert report.baseline_mean_step_lpips == report.candidate_mean_step_lpips
    payload = report.to_dict()
    assert payload["method_id"] == "adjacent-frame-lpips-trajectory-v1"
    assert str(tmp_path) not in json.dumps(payload)


def test_static_scenes_score_zero_and_flicker_scores_higher(
    tmp_path: Path, backbone_dir
) -> None:
    static_a = _encode(tmp_path / "static-a.mkv", _static_frames())
    static_b = _encode(tmp_path / "static-b.mkv", _static_frames())
    flicker = _encode(tmp_path / "flicker.mkv", _static_frames(flicker=24))

    static_report = _score(static_a, static_b, backbone_dir)
    flicker_report = _score(static_a, flicker, backbone_dir)

    assert static_report.baseline_mean_step_lpips == 0.0
    assert static_report.mean_abs_trajectory_delta == 0.0
    assert flicker_report.mean_abs_trajectory_delta > 0.0
    assert flicker_report.candidate_mean_step_lpips > 0.0


def test_removed_cut_scores_higher_than_preserved_cut(
    tmp_path: Path, backbone_dir
) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _cut_frames(preserve_cut=True))
    preserved = _encode(tmp_path / "preserved.mkv", _cut_frames(preserve_cut=True))
    removed = _encode(tmp_path / "removed.mkv", _cut_frames(preserve_cut=False))

    preserved_report = _score(baseline, preserved, backbone_dir)
    removed_report = _score(baseline, removed, backbone_dir)

    assert preserved_report.mean_abs_trajectory_delta == 0.0
    assert removed_report.mean_abs_trajectory_delta > 0.0
    assert (
        removed_report.max_abs_trajectory_delta
        > preserved_report.max_abs_trajectory_delta
    )


def test_scores_are_deterministic(tmp_path: Path, backbone_dir) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _cut_frames(preserve_cut=True))
    candidate = _encode(tmp_path / "candidate.mkv", _static_frames(flicker=8))

    first = _score(baseline, candidate, backbone_dir)
    second = _score(baseline, candidate, backbone_dir)

    assert first.mean_abs_trajectory_delta == second.mean_abs_trajectory_delta
    assert first.max_abs_trajectory_delta == second.max_abs_trajectory_delta


def test_rejects_frame_count_mismatch_and_single_frame(
    tmp_path: Path, backbone_dir
) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _static_frames())
    short = _encode(tmp_path / "short.mkv", _static_frames(count=_FRAMES - 1))

    with pytest.raises(ValidationError, match="frame count"):
        _score(baseline, short, backbone_dir)

    single_a = _encode(tmp_path / "single-a.mkv", _static_frames(count=1))
    single_b = _encode(tmp_path / "single-b.mkv", _static_frames(count=1))
    with pytest.raises(ValidationError, match="at least two frames"):
        _score(single_a, single_b, backbone_dir)


def test_decode_failure_reports_decode_error(tmp_path: Path, backbone_dir) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _static_frames())
    candidate = _encode(tmp_path / "candidate.mkv", _static_frames())
    fake_ffmpeg = tmp_path / "fake-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-version" ]; then echo "fake-ffmpeg version 0"; exit 0; fi\n'
        f"head -c {_FRAME_BYTES * 3} /dev/zero\n"
        'echo "synthetic decode failure" >&2\n'
        "exit 1\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o700)
    root, digest = backbone_dir

    with pytest.raises(ValidationError, match="decode failed"):
        score_temporal_consistency(
            baseline,
            candidate,
            backbone_dir=root,
            expected_backbone_sha256=digest,
            ffmpeg=str(fake_ffmpeg),
        )


def test_non_finite_step_distance_fails_closed(
    tmp_path: Path, backbone_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    baseline = _encode(tmp_path / "baseline.mkv", _static_frames())
    candidate = _encode(tmp_path / "candidate.mkv", _static_frames())

    monkeypatch.setattr(
        "h3fast.benchmarks.temporal_consistency._load_lpips_model",
        lambda *_args, **_kwargs: lambda *_inputs: torch.tensor(float("nan")),
    )
    root, digest = backbone_dir
    with pytest.raises(ValidationError, match="non-finite"):
        score_temporal_consistency(
            baseline,
            candidate,
            backbone_dir=root,
            expected_backbone_sha256=digest,
        )


def test_cli_scores_temporal_consistency(tmp_path: Path, backbone_dir, capsys) -> None:
    from h3fast.cli import main

    root, digest = backbone_dir
    baseline = _encode(tmp_path / "baseline.mkv", _cut_frames(preserve_cut=True))
    candidate = _encode(tmp_path / "candidate.mkv", _cut_frames(preserve_cut=False))

    status = main(
        [
            "benchmark",
            "score-temporal-consistency",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--backbone-dir",
            str(root),
            "--expected-backbone-sha256",
            digest,
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert status == 0
    assert output["method_id"] == "adjacent-frame-lpips-trajectory-v1"
    assert output["step_count"] == _FRAMES - 1
    assert output["mean_abs_trajectory_delta"] > 0.0
    assert str(tmp_path) not in json.dumps(output)
