"""SigLIP2 prompt-adherence metric adapter with offline pinned files.

Contract ``siglip2-base-patch16-256-cosine-v1``: uniformly sample up to
16 decoded RGB frames, encode the prompt once and every sampled frame
with the pinned SigLIP2 snapshot, and aggregate the per-frame cosine
similarity between L2-normalized text and image features as mean and
minimum (higher is better). The prompt text is supplied as a private
local file whose exact bytes must match the formal case prompt digest.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import stat
import warnings
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from h3fast.benchmarks.perceptual_video import (
    _decoded_frames,
    _probe_video,
)
from h3fast.benchmarks.quality import tool_version
from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

PROMPT_ADHERENCE_METHOD_ID = "siglip2-base-patch16-256-cosine-v1"
SIGLIP2_MODEL_FILE_SHA256S: dict[str, str] = {
    "config.json": "7b5aedcb8893e31376e129c1ffd7a5392f1a806dbc793ce53eda220c2ec59edf",
    "model.safetensors": (
        "6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863"
    ),
    "preprocessor_config.json": (
        "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff"
    ),
    "special_tokens_map.json": (
        "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"
    ),
    "tokenizer.json": (
        "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"
    ),
    "tokenizer.model": (
        "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"
    ),
    "tokenizer_config.json": (
        "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8"
    ),
}
_FRAME_SAMPLE_LIMIT = 16
_TEXT_MAX_LENGTH = 64
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".gguf"})


@dataclass(frozen=True, slots=True)
class PromptAdherenceReport:
    """Aggregate SigLIP2 prompt-frame similarity for one video."""

    method_id: str
    frame_count: int
    sampled_frame_count: int
    width: int
    height: int
    mean_similarity: float
    min_similarity: float
    prompt_sha256: str
    model_weights_sha256: str
    torch_num_threads: int
    transformers_version: str
    torch_version: str
    ffmpeg_version: str

    def to_dict(self) -> dict[str, object]:
        """Return score metadata without prompt text or local paths."""
        return {
            "schema_version": "1.0",
            "method_id": self.method_id,
            "frame_count": self.frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "width": self.width,
            "height": self.height,
            "mean_similarity": self.mean_similarity,
            "min_similarity": self.min_similarity,
            "prompt_sha256": self.prompt_sha256,
            "model_weights_sha256": self.model_weights_sha256,
            "torch_num_threads": self.torch_num_threads,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "ffmpeg_version": self.ffmpeg_version,
        }


def _import_prompt_dependencies() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        message = (
            "prompt-adherence scoring requires the quality-metrics dependency "
            f"group (torch, transformers): {error}"
        )
        raise ValidationError(message) from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as error:
        message = f"pinned SigLIP2 file could not be read: {error}"
        raise ValidationError(message) from error
    return digest.hexdigest()


def _verify_model_dir(model_dir: Path, expected: Mapping[str, str]) -> None:
    if not expected:
        message = "SigLIP2 file manifest must not be empty"
        raise ValidationError(message)
    for name, digest in expected.items():
        target = model_dir / name
        if not target.is_file():
            message = f"pinned SigLIP2 file is missing: {name}"
            raise ValidationError(message)
        if _file_sha256(target) != digest:
            message = f"pinned SigLIP2 file digest does not match: {name}"
            raise ValidationError(message)
    try:
        entries = list(model_dir.iterdir())
    except OSError as error:
        message = f"SigLIP2 model directory could not be listed: {error}"
        raise ValidationError(message) from error
    for entry in entries:
        if entry.suffix.lower() in _WEIGHT_SUFFIXES and entry.name not in expected:
            message = (
                f"SigLIP2 model directory contains unexpected weights: {entry.name}"
            )
            raise ValidationError(message)


def _load_siglip2(model_dir: Path, expected: Mapping[str, str]):
    _verify_model_dir(model_dir, expected)
    import transformers
    from transformers import AutoModel, AutoProcessor, Siglip2Model, SiglipModel

    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # The pinned FixRes SigLIP2 snapshot declares model_type
            # "siglip"; the digest manifest fixes the loaded identity.
            model = AutoModel.from_pretrained(model_dir, local_files_only=True)
            processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        EOFError,
        pickle.UnpicklingError,
        zipfile.BadZipFile,
    ) as error:
        message = f"pinned SigLIP2 model could not be constructed offline: {error}"
        raise ValidationError(message) from error
    if not isinstance(model, (SiglipModel, Siglip2Model)):
        message = (
            "pinned SigLIP2 directory resolved to an unexpected architecture: "
            f"{type(model).__name__}"
        )
        raise ValidationError(message)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def _load_private_prompt(prompt_file: Path, expected_sha256: str) -> str:
    try:
        mode = stat.S_IMODE(prompt_file.stat().st_mode)
        raw = prompt_file.read_bytes()
    except FileNotFoundError as error:
        message = "prompt file is missing"
        raise ValidationError(message) from error
    except OSError as error:
        message = f"prompt file could not be read: {error}"
        raise ValidationError(message) from error
    if mode & 0o077:
        message = "prompt file must not be accessible by group or other users"
        raise ValidationError(message)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        message = "prompt digest does not match the expected case prompt"
        raise ValidationError(message)
    try:
        prompt = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        message = "prompt file is not valid UTF-8"
        raise ValidationError(message) from error
    if not prompt.strip():
        message = "prompt file must contain a non-empty prompt"
        raise ValidationError(message)
    return prompt


def _sample_indices(frame_count: int, limit: int) -> list[int]:
    if frame_count <= limit:
        return list(range(frame_count))
    step = (frame_count - 1) / (limit - 1)
    return sorted({round(index * step) for index in range(limit)})


def _feature_tensor(output):  # noqa: ANN001
    import torch

    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


def _similarity_values(
    model,  # noqa: ANN001
    processor,  # noqa: ANN001
    prompt: str,
    frames: Sequence[object],
) -> list[float]:
    import torch

    inputs = processor(
        images=frames,
        text=[prompt],
        padding="max_length",
        max_length=_TEXT_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    image_keys = ("pixel_values", "pixel_attention_mask", "spatial_shapes")
    try:
        with torch.no_grad():
            text_features = _feature_tensor(
                model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
            )
            image_features = _feature_tensor(
                model.get_image_features(
                    **{key: inputs[key] for key in image_keys if key in inputs}
                )
            )
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            similarities = image_features @ text_features.reshape(-1)
    except (RuntimeError, ValueError) as error:
        message = f"SigLIP2 forward pass failed: {error}"
        raise ValidationError(message) from error
    return [float(value) for value in similarities.reshape(-1).tolist()]


def score_prompt_adherence(
    video_path: Path,
    *,
    prompt_file: Path,
    expected_prompt_sha256: str,
    model_dir: Path,
    expected_file_sha256s: Mapping[str, str] = SIGLIP2_MODEL_FILE_SHA256S,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> PromptAdherenceReport:
    """Score prompt adherence of one video against a digest-bound prompt."""
    _import_prompt_dependencies()
    ffmpeg_version = tool_version(ffmpeg)
    prompt = _load_private_prompt(prompt_file, expected_prompt_sha256)
    media = _probe_video(video_path, ffprobe)
    model, processor = _load_siglip2(model_dir, expected_file_sha256s)
    import numpy as np
    import torch
    import transformers

    frames: list[bytes] = list(
        _decoded_frames(video_path, ffmpeg, width=media.width, height=media.height)
    )
    frame_count = len(frames)
    if frame_count == 0:
        message = "prompt-adherence input contains no decodable frames"
        raise ValidationError(message)
    indices = _sample_indices(frame_count, _FRAME_SAMPLE_LIMIT)
    sampled = [
        np.frombuffer(frames[index], dtype=np.uint8)
        .reshape(media.height, media.width, 3)
        .copy()
        for index in indices
    ]

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        values = _similarity_values(model, processor, prompt, sampled)
    finally:
        torch.set_num_threads(previous_threads)
    if len(values) != len(indices):
        message = "SigLIP2 returned an unexpected number of similarities"
        raise ValidationError(message)
    for value in values:
        if not math.isfinite(value):
            message = "SigLIP2 produced a non-finite similarity"
            raise ValidationError(message)

    return PromptAdherenceReport(
        method_id=PROMPT_ADHERENCE_METHOD_ID,
        frame_count=frame_count,
        sampled_frame_count=len(indices),
        width=media.width,
        height=media.height,
        mean_similarity=sum(values) / len(values),
        min_similarity=min(values),
        prompt_sha256=expected_prompt_sha256,
        model_weights_sha256=expected_file_sha256s.get("model.safetensors", ""),
        torch_num_threads=1,
        transformers_version=str(transformers.__version__),
        torch_version=str(torch.__version__),
        ffmpeg_version=ffmpeg_version,
    )
