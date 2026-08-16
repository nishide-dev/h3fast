"""Tests for the SigLIP2 prompt-adherence metric adapter."""

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import pytest

from h3fast.exceptions import ValidationError

pytest.importorskip("torch")
pytest.importorskip("transformers")

from h3fast.benchmarks import score_prompt_adherence
from h3fast.benchmarks.prompt_adherence import (
    PROMPT_ADHERENCE_METHOD_ID,
    SIGLIP2_MODEL_FILE_SHA256S,
)

_WIDTH = 64
_HEIGHT = 64
_FRAMES = 8
_RATE = 8
_PROMPT = "a cat video of red motion"


def _frames(count: int = _FRAMES, *, seed: int = 20260816) -> bytes:
    import numpy as np

    generator = np.random.default_rng(seed)
    return generator.integers(
        0, 256, size=(count, _HEIGHT, _WIDTH, 3), dtype=np.uint8
    ).tobytes()


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
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Siglip2Config,
        Siglip2ImageProcessor,
        Siglip2Model,
        Siglip2Processor,
    )

    root = tmp_path_factory.mktemp("siglip2-tiny")
    config = Siglip2Config(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 64,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "projection_size": 32,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 16,
            "num_patches": 256,
        },
    )
    torch.manual_seed(0)
    model = Siglip2Model(config).eval()
    vocabulary = {
        word: index
        for index, word in enumerate(
            [
                "<unk>",
                "<pad>",
                "a",
                "cat",
                "dog",
                "red",
                "blue",
                "video",
                "of",
                "motion",
            ]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>"
    )
    processor = Siglip2Processor(
        image_processor=Siglip2ImageProcessor(), tokenizer=fast_tokenizer
    )
    model.save_pretrained(root)
    processor.save_pretrained(root)
    manifest = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(root.iterdir())
        if item.is_file()
    }
    return root, manifest


def _prompt_file(tmp_path: Path, text: str = _PROMPT) -> tuple[Path, str]:
    path = tmp_path / "prompt.txt"
    data = text.encode("utf-8")
    path.write_bytes(data)
    path.chmod(0o600)
    return path, hashlib.sha256(data).hexdigest()


def _score(video: Path, tmp_path: Path, model: tuple[Path, dict[str, str]], **kwargs):
    root, manifest = model
    prompt_path, prompt_sha = _prompt_file(tmp_path, kwargs.pop("prompt", _PROMPT))
    return score_prompt_adherence(
        video,
        prompt_file=prompt_path,
        expected_prompt_sha256=kwargs.pop("expected_prompt_sha256", prompt_sha),
        model_dir=root,
        expected_file_sha256s=kwargs.pop("expected_file_sha256s", manifest),
        **kwargs,
    )


def test_scores_are_deterministic_and_bounded(tmp_path: Path, model_dir) -> None:
    video = _encode(tmp_path / "video.mkv", _frames())

    first = _score(video, tmp_path, model_dir)
    second = _score(video, tmp_path, model_dir)

    assert first.method_id == PROMPT_ADHERENCE_METHOD_ID
    assert first.frame_count == _FRAMES
    assert first.sampled_frame_count == _FRAMES
    assert -1.0 <= first.min_similarity <= first.mean_similarity <= 1.0
    assert first.mean_similarity == second.mean_similarity
    assert first.min_similarity == second.min_similarity
    payload = first.to_dict()
    assert str(tmp_path) not in json.dumps(payload)
    assert _PROMPT not in json.dumps(payload)


def test_frame_sampling_caps_at_fixed_limit(tmp_path: Path, model_dir) -> None:
    video = _encode(tmp_path / "long.mkv", _frames(count=40))

    report = _score(video, tmp_path, model_dir)

    assert report.frame_count == 40
    assert report.sampled_frame_count == 16


def test_different_prompts_change_the_score(tmp_path: Path, model_dir) -> None:
    video = _encode(tmp_path / "video.mkv", _frames())

    first = _score(video, tmp_path, model_dir, prompt="a cat video")
    second = _score(video, tmp_path, model_dir, prompt="blue dog motion")

    assert first.mean_similarity != second.mean_similarity


def test_rejects_prompt_digest_mismatch_and_exposed_file(
    tmp_path: Path, model_dir
) -> None:
    video = _encode(tmp_path / "video.mkv", _frames())
    root, manifest = model_dir

    prompt_path, _sha = _prompt_file(tmp_path)
    with pytest.raises(ValidationError, match="prompt digest"):
        score_prompt_adherence(
            video,
            prompt_file=prompt_path,
            expected_prompt_sha256="0" * 64,
            model_dir=root,
            expected_file_sha256s=manifest,
        )

    prompt_path.chmod(0o644)
    with pytest.raises(ValidationError, match="group or other"):
        score_prompt_adherence(
            video,
            prompt_file=prompt_path,
            expected_prompt_sha256=hashlib.sha256(_PROMPT.encode()).hexdigest(),
            model_dir=root,
            expected_file_sha256s=manifest,
        )


def test_rejects_missing_tampered_or_unlisted_model_files(
    tmp_path: Path, model_dir
) -> None:
    video = _encode(tmp_path / "video.mkv", _frames())
    root, manifest = model_dir
    prompt_path, prompt_sha = _prompt_file(tmp_path)

    missing = dict(manifest)
    missing["not-there.json"] = "0" * 64
    with pytest.raises(ValidationError, match="missing"):
        score_prompt_adherence(
            video,
            prompt_file=prompt_path,
            expected_prompt_sha256=prompt_sha,
            model_dir=root,
            expected_file_sha256s=missing,
        )

    tampered = dict(manifest)
    tampered["config.json"] = "0" * 64
    with pytest.raises(ValidationError, match="digest"):
        score_prompt_adherence(
            video,
            prompt_file=prompt_path,
            expected_prompt_sha256=prompt_sha,
            model_dir=root,
            expected_file_sha256s=tampered,
        )

    stray = root / "extra.safetensors"
    stray.write_bytes(b"stray weights")
    try:
        with pytest.raises(ValidationError, match="unexpected"):
            score_prompt_adherence(
                video,
                prompt_file=prompt_path,
                expected_prompt_sha256=prompt_sha,
                model_dir=root,
                expected_file_sha256s=manifest,
            )
    finally:
        stray.unlink()


def test_non_finite_similarity_fails_closed(
    tmp_path: Path, model_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _encode(tmp_path / "video.mkv", _frames())

    monkeypatch.setattr(
        "h3fast.benchmarks.prompt_adherence._similarity_values",
        lambda *_args, **_kwargs: [float("nan")] * _FRAMES,
    )
    with pytest.raises(ValidationError, match="non-finite"):
        _score(video, tmp_path, model_dir)


def test_zero_frames_fail_closed(
    tmp_path: Path, model_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _encode(tmp_path / "video.mkv", _frames())

    monkeypatch.setattr(
        "h3fast.benchmarks.prompt_adherence._decoded_frames",
        lambda *_args, **_kwargs: (frame for frame in ()),
    )
    with pytest.raises(ValidationError, match="no decodable frames"):
        _score(video, tmp_path, model_dir)


def test_sample_indices_are_pinned_exactly() -> None:
    from h3fast.benchmarks.prompt_adherence import _sample_indices

    assert _sample_indices(1, 16) == [0]
    assert _sample_indices(15, 16) == list(range(15))
    assert _sample_indices(16, 16) == list(range(16))
    assert _sample_indices(17, 16) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]
    forty = _sample_indices(40, 16)
    assert len(forty) == 16
    assert forty[0] == 0
    assert forty[-1] == 39
    assert all(a < b for a, b in itertools.pairwise(forty))


def test_exact_aggregation_of_known_similarities(
    tmp_path: Path, model_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _encode(tmp_path / "video.mkv", _frames())
    values = [0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.2, 0.6]

    monkeypatch.setattr(
        "h3fast.benchmarks.prompt_adherence._similarity_values",
        lambda *_args, **_kwargs: list(values),
    )
    report = _score(video, tmp_path, model_dir)

    assert report.mean_similarity == pytest.approx(sum(values) / len(values))
    assert report.min_similarity == 0.1


def test_rejects_unexpected_architecture(tmp_path: Path) -> None:
    import hashlib as _hashlib

    from transformers import BertConfig, BertModel

    root = tmp_path / "bert-tiny"
    root.mkdir()
    model = BertModel(
        BertConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            vocab_size=64,
        )
    )
    model.save_pretrained(root)
    manifest = {
        item.name: _hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(root.iterdir())
        if item.is_file()
    }
    video = _encode(tmp_path / "video.mkv", _frames())
    prompt_path, prompt_sha = _prompt_file(tmp_path)

    with pytest.raises(ValidationError, match="unexpected architecture"):
        score_prompt_adherence(
            video,
            prompt_file=prompt_path,
            expected_prompt_sha256=prompt_sha,
            model_dir=root,
            expected_file_sha256s=manifest,
        )


def test_corrupt_weights_with_matching_digest_fail_closed(
    tmp_path: Path, model_dir
) -> None:
    import shutil

    root, manifest = model_dir
    corrupt_root = tmp_path / "corrupt"
    shutil.copytree(root, corrupt_root)
    weights = corrupt_root / "model.safetensors"
    weights.write_bytes(b"not a safetensors file")
    corrupt_manifest = dict(manifest)
    corrupt_manifest["model.safetensors"] = hashlib.sha256(
        weights.read_bytes()
    ).hexdigest()
    video = _encode(tmp_path / "video.mkv", _frames())
    prompt_path, prompt_sha = _prompt_file(tmp_path)

    with pytest.raises(ValidationError, match="could not be constructed"):
        score_prompt_adherence(
            video,
            prompt_file=prompt_path,
            expected_prompt_sha256=prompt_sha,
            model_dir=corrupt_root,
            expected_file_sha256s=corrupt_manifest,
        )


def test_rejects_subdirectory_and_manifest_without_weights(
    tmp_path: Path, model_dir
) -> None:
    root, manifest = model_dir
    video = _encode(tmp_path / "video.mkv", _frames())
    prompt_path, prompt_sha = _prompt_file(tmp_path)

    nested = root / "nested"
    nested.mkdir()
    stray = nested / "evil.safetensors"
    stray.write_bytes(b"stray")
    try:
        with pytest.raises(ValidationError, match="unexpected files"):
            score_prompt_adherence(
                video,
                prompt_file=prompt_path,
                expected_prompt_sha256=prompt_sha,
                model_dir=root,
                expected_file_sha256s=manifest,
            )
    finally:
        stray.unlink()
        nested.rmdir()

    incomplete = {
        name: digest for name, digest in manifest.items() if name != "model.safetensors"
    }
    with pytest.raises(ValidationError, match=r"must include model\.safetensors"):
        score_prompt_adherence(
            video,
            prompt_file=prompt_path,
            expected_prompt_sha256=prompt_sha,
            model_dir=root,
            expected_file_sha256s=incomplete,
        )


def test_prompt_file_content_contract_edges(tmp_path: Path, model_dir) -> None:
    video = _encode(tmp_path / "video.mkv", _frames())
    root, manifest = model_dir

    with pytest.raises(ValidationError, match="prompt file is missing"):
        score_prompt_adherence(
            video,
            prompt_file=tmp_path / "absent.txt",
            expected_prompt_sha256="0" * 64,
            model_dir=root,
            expected_file_sha256s=manifest,
        )

    binary = tmp_path / "binary.txt"
    payload = b"\xff\xfe invalid utf-8"
    binary.write_bytes(payload)
    binary.chmod(0o600)
    with pytest.raises(ValidationError, match="not valid UTF-8"):
        score_prompt_adherence(
            video,
            prompt_file=binary,
            expected_prompt_sha256=hashlib.sha256(payload).hexdigest(),
            model_dir=root,
            expected_file_sha256s=manifest,
        )

    blank = tmp_path / "blank.txt"
    blank.write_bytes(b"   \n")
    blank.chmod(0o600)
    with pytest.raises(ValidationError, match="non-empty prompt"):
        score_prompt_adherence(
            video,
            prompt_file=blank,
            expected_prompt_sha256=hashlib.sha256(b"   \n").hexdigest(),
            model_dir=root,
            expected_file_sha256s=manifest,
        )


def test_cli_rejects_malformed_manifest(tmp_path: Path, model_dir, capsys) -> None:
    from h3fast.cli import main

    root, _manifest = model_dir
    video = _encode(tmp_path / "video.mkv", _frames())
    prompt_path, prompt_sha = _prompt_file(tmp_path)

    def run_with_manifest(manifest_path: Path) -> int:
        return main(
            [
                "benchmark",
                "score-prompt-adherence",
                "--video",
                str(video),
                "--prompt-file",
                str(prompt_path),
                "--expected-prompt-sha256",
                prompt_sha,
                "--model-dir",
                str(root),
                "--model-manifest",
                str(manifest_path),
            ]
        )

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert run_with_manifest(malformed) == 2
    assert "could not be read" in capsys.readouterr().err

    wrong_type = tmp_path / "wrong-type.json"
    wrong_type.write_text('{"a": 1}', encoding="utf-8")
    assert run_with_manifest(wrong_type) == 2
    assert "must map file names" in capsys.readouterr().err

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    assert run_with_manifest(empty) == 2
    assert "must not be empty" in capsys.readouterr().err


def test_pinned_manifest_constants_are_wellformed() -> None:
    assert "model.safetensors" in SIGLIP2_MODEL_FILE_SHA256S
    assert "config.json" in SIGLIP2_MODEL_FILE_SHA256S
    for name, digest in SIGLIP2_MODEL_FILE_SHA256S.items():
        assert name
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)


def test_cli_scores_prompt_adherence(tmp_path: Path, model_dir, capsys) -> None:
    from h3fast.cli import main

    root, manifest = model_dir
    video = _encode(tmp_path / "video.mkv", _frames())
    prompt_path, prompt_sha = _prompt_file(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status = main(
        [
            "benchmark",
            "score-prompt-adherence",
            "--video",
            str(video),
            "--prompt-file",
            str(prompt_path),
            "--expected-prompt-sha256",
            prompt_sha,
            "--model-dir",
            str(root),
            "--model-manifest",
            str(manifest_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert status == 0
    assert output["method_id"] == PROMPT_ADHERENCE_METHOD_ID
    assert output["sampled_frame_count"] == _FRAMES
    assert str(tmp_path) not in json.dumps(output)
    assert _PROMPT not in json.dumps(output)


def test_cli_uses_pinned_manifest_by_default(tmp_path: Path, model_dir, capsys) -> None:
    from h3fast.cli import main

    root, _manifest = model_dir
    video = _encode(tmp_path / "video.mkv", _frames())
    prompt_path, prompt_sha = _prompt_file(tmp_path)

    status = main(
        [
            "benchmark",
            "score-prompt-adherence",
            "--video",
            str(video),
            "--prompt-file",
            str(prompt_path),
            "--expected-prompt-sha256",
            prompt_sha,
            "--model-dir",
            str(root),
        ]
    )

    assert status == 2
    error_output = capsys.readouterr().err
    assert "missing" in error_output or "digest" in error_output
