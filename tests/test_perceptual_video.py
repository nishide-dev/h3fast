"""Tests for the LPIPS perceptual-video metric adapter."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from h3fast.exceptions import ValidationError

pytest.importorskip("torch")
pytest.importorskip("lpips")

from h3fast.benchmarks import score_perceptual_video
from h3fast.benchmarks.perceptual_video import (
    ALEXNET_BACKBONE_FILENAME,
)

_WIDTH = 64
_HEIGHT = 64
_FRAMES = 8
_RATE = 8


def _frames(noise_scale: int) -> bytes:
    import numpy as np

    generator = np.random.default_rng(20260816)
    base = generator.integers(
        0, 256, size=(_FRAMES, _HEIGHT, _WIDTH, 3), dtype=np.int16
    )
    noise = np.random.default_rng(42).integers(-1, 2, size=base.shape, dtype=np.int16)
    frames = np.clip(base + noise * noise_scale, 0, 255)
    return frames.astype(np.uint8).tobytes()


def _encode(
    path: Path,
    data: bytes,
    *,
    width: int = _WIDTH,
    height: int = _HEIGHT,
    rate: int = _RATE,
) -> Path:
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
            f"{width}x{height}",
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

    root = tmp_path_factory.mktemp("hub")
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir()
    torch.manual_seed(0)
    weights = alexnet(weights=None).state_dict()
    checkpoint = checkpoint_dir / ALEXNET_BACKBONE_FILENAME
    torch.save(weights, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return root, digest


def _score(
    baseline: Path,
    candidate: Path,
    backbone: tuple[Path, str],
):
    root, digest = backbone
    return score_perceptual_video(
        baseline,
        candidate,
        backbone_dir=root,
        expected_backbone_sha256=digest,
    )


def test_identical_videos_score_zero(tmp_path: Path, backbone_dir) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))

    report = _score(baseline, candidate, backbone_dir)

    assert report.frame_count == _FRAMES
    assert report.width == _WIDTH
    assert report.height == _HEIGHT
    assert report.mean_lpips == 0.0
    assert report.max_lpips == 0.0
    payload = report.to_dict()
    assert payload["method_id"] == "lpips-alex-0.1.4-v1"
    assert str(tmp_path) not in json.dumps(payload)


def test_larger_perturbation_scores_higher(tmp_path: Path, backbone_dir) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    small = _encode(tmp_path / "small.mkv", _frames(8))
    large = _encode(tmp_path / "large.mkv", _frames(40))

    small_report = _score(baseline, small, backbone_dir)
    large_report = _score(baseline, large, backbone_dir)

    assert 0.0 < small_report.mean_lpips < large_report.mean_lpips
    assert small_report.max_lpips <= large_report.max_lpips


def test_scores_are_deterministic(tmp_path: Path, backbone_dir) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(8))

    first = _score(baseline, candidate, backbone_dir)
    second = _score(baseline, candidate, backbone_dir)

    assert first.mean_lpips == second.mean_lpips
    assert first.max_lpips == second.max_lpips


def test_rejects_mismatched_media_contract(tmp_path: Path, backbone_dir) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))

    short = _encode(
        tmp_path / "short.mkv",
        _frames(0)[: (_FRAMES - 1) * _HEIGHT * _WIDTH * 3],
    )
    with pytest.raises(ValidationError, match="frame count"):
        _score(baseline, short, backbone_dir)

    import numpy as np

    small_frames = (
        np.frombuffer(_frames(0), dtype=np.uint8)
        .reshape(_FRAMES, _HEIGHT, _WIDTH, 3)[:, : _HEIGHT // 2, : _WIDTH // 2, :]
        .copy()
        .tobytes()
    )
    resized = _encode(
        tmp_path / "resized.mkv",
        small_frames,
        width=_WIDTH // 2,
        height=_HEIGHT // 2,
    )
    with pytest.raises(ValidationError, match="resolution"):
        _score(baseline, resized, backbone_dir)

    slow = _encode(tmp_path / "slow.mkv", _frames(0), rate=_RATE // 2)
    with pytest.raises(ValidationError, match="frame rate"):
        _score(baseline, slow, backbone_dir)


def test_rejects_missing_or_tampered_backbone(tmp_path: Path, backbone_dir) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))

    with pytest.raises(ValidationError, match="backbone"):
        score_perceptual_video(
            baseline,
            candidate,
            backbone_dir=tmp_path / "missing-hub",
            expected_backbone_sha256="0" * 64,
        )

    root, _digest = backbone_dir
    with pytest.raises(ValidationError, match="digest"):
        score_perceptual_video(
            baseline,
            candidate,
            backbone_dir=root,
            expected_backbone_sha256="0" * 64,
        )


def test_cli_scores_perceptual_video(tmp_path: Path, backbone_dir, capsys) -> None:
    from h3fast.cli import main

    root, digest = backbone_dir
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(8))

    status = main(
        [
            "benchmark",
            "score-perceptual-video",
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
    assert output["method_id"] == "lpips-alex-0.1.4-v1"
    assert output["frame_count"] == _FRAMES
    assert output["mean_lpips"] > 0.0
    assert str(tmp_path) not in json.dumps(output)


def test_package_import_stays_torch_free() -> None:
    code = (
        "import sys; import h3fast.benchmarks; "
        "assert 'torch' not in sys.modules; assert 'lpips' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
