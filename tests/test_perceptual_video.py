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


def test_pinned_backbone_constants_are_consistent() -> None:
    from h3fast.benchmarks.perceptual_video import ALEXNET_BACKBONE_SHA256

    assert (
        ALEXNET_BACKBONE_SHA256
        == "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
    )
    assert f"alexnet-owt-{ALEXNET_BACKBONE_SHA256[:8]}.pth" == ALEXNET_BACKBONE_FILENAME


def test_cli_uses_pinned_digest_by_default(
    tmp_path: Path, backbone_dir, capsys
) -> None:
    from h3fast.cli import main

    root, _digest = backbone_dir
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))

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
        ]
    )

    assert status == 2
    assert "digest" in capsys.readouterr().err


def test_missing_scoring_dependencies_fail_closed(
    tmp_path: Path, backbone_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))

    monkeypatch.setitem(sys.modules, "torch", None)
    root, digest = backbone_dir
    with pytest.raises(ValidationError, match="quality-metrics"):
        score_perceptual_video(
            baseline,
            candidate,
            backbone_dir=root,
            expected_backbone_sha256=digest,
        )


def test_corrupt_checkpoint_fails_closed_and_restores_hub_dir(
    tmp_path: Path,
) -> None:
    import torch.hub

    root = tmp_path / "hub"
    (root / "checkpoints").mkdir(parents=True)
    checkpoint = root / "checkpoints" / ALEXNET_BACKBONE_FILENAME
    checkpoint.write_bytes(b"not a checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))
    hub_dir_before = torch.hub.get_dir()

    with pytest.raises(ValidationError, match="could not be constructed"):
        score_perceptual_video(
            baseline,
            candidate,
            backbone_dir=root,
            expected_backbone_sha256=digest,
        )
    assert torch.hub.get_dir() == hub_dir_before


def test_construction_must_not_add_checkpoint_files(
    tmp_path: Path, backbone_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lpips

    root, digest = backbone_dir
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))
    real_lpips = lpips.LPIPS

    def downloading_lpips(*args: object, **kwargs: object):
        (root / "checkpoints" / "downloaded.pth").write_bytes(b"downloaded")
        return real_lpips(*args, **kwargs)

    monkeypatch.setattr(lpips, "LPIPS", downloading_lpips)
    try:
        with pytest.raises(ValidationError, match="unexpected"):
            score_perceptual_video(
                baseline,
                candidate,
                backbone_dir=root,
                expected_backbone_sha256=digest,
            )
    finally:
        (root / "checkpoints" / "downloaded.pth").unlink(missing_ok=True)


def test_rejects_non_video_and_missing_inputs(tmp_path: Path, backbone_dir) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))

    with pytest.raises(ValidationError, match="ffprobe failed"):
        _score(baseline, tmp_path / "missing.mkv", backbone_dir)

    audio_only = tmp_path / "audio.mkv"
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:a",
            "flac",
            str(audio_only),
        ],
        check=True,
    )
    with pytest.raises(ValidationError, match="no video stream"):
        _score(baseline, audio_only, backbone_dir)


def test_decode_failure_on_longer_stream_reports_decode_error(
    tmp_path: Path, backbone_dir
) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))
    frame_bytes = _WIDTH * _HEIGHT * 3
    fake_ffmpeg = tmp_path / "fake-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/bin/sh\n"
        'input=""\n'
        'previous=""\n'
        'for argument in "$@"; do\n'
        '  if [ "$previous" = "-i" ]; then input="$argument"; fi\n'
        '  previous="$argument"\n'
        "done\n"
        'case "$input" in\n'
        f"  *baseline*) head -c {frame_bytes * 5} /dev/zero;"
        ' echo "synthetic baseline decode failure" >&2; exit 1;;\n'
        f"  *) head -c {frame_bytes * 3} /dev/zero; exit 0;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o700)
    root, digest = backbone_dir

    with pytest.raises(ValidationError, match="decode failed"):
        score_perceptual_video(
            baseline,
            candidate,
            backbone_dir=root,
            expected_backbone_sha256=digest,
            ffmpeg=str(fake_ffmpeg),
        )


def test_non_finite_distance_fails_closed(
    tmp_path: Path, backbone_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))

    monkeypatch.setattr(
        "h3fast.benchmarks.perceptual_video._load_lpips_model",
        lambda *_args, **_kwargs: lambda *_inputs: torch.tensor(float("nan")),
    )
    root, digest = backbone_dir
    with pytest.raises(ValidationError, match="non-finite"):
        score_perceptual_video(
            baseline,
            candidate,
            backbone_dir=root,
            expected_backbone_sha256=digest,
        )


def test_zero_frames_fail_closed(
    tmp_path: Path, backbone_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _encode(tmp_path / "baseline.mkv", _frames(0))
    candidate = _encode(tmp_path / "candidate.mkv", _frames(0))

    monkeypatch.setattr(
        "h3fast.benchmarks.perceptual_video._decoded_frames",
        lambda *_args, **_kwargs: (frame for frame in ()),
    )
    root, digest = backbone_dir
    with pytest.raises(ValidationError, match="no decodable frames"):
        score_perceptual_video(
            baseline,
            candidate,
            backbone_dir=root,
            expected_backbone_sha256=digest,
        )


def test_package_import_stays_torch_free() -> None:
    code = (
        "import sys; import h3fast.benchmarks; "
        "assert 'torch' not in sys.modules; assert 'lpips' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
