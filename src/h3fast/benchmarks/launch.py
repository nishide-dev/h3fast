"""Pinned Singularity launch planning for the H3 reference backend."""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from h3fast.backends.sglang import REFERENCE_SGLANG_COMMIT
from h3fast.exceptions import ValidationError
from h3fast.manifest.checksums import sha256_file

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE_RUNTIME_IMAGE = (
    "lmsysorg/sglang@"
    "sha256:29f0f645122be1799a594c15907d81da326dbbe6ccd6395710a07a4292125a5f"
)


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """An inspectable, shell-independent SGLang launch command."""

    argv: tuple[str, ...]
    selected_gpus: tuple[int, ...]
    sglang_revision: str
    base_image: str
    ffprobe_adapter_sha256: str
    runtime_settings: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable launch metadata."""
        return {
            "argv": list(self.argv),
            "shell_command": shlex.join(self.argv),
            "selected_gpus": list(self.selected_gpus),
            "sglang_revision": self.sglang_revision,
            "base_image": self.base_image,
            "ffprobe_adapter_sha256": self.ffprobe_adapter_sha256,
            "runtime_settings": self.runtime_settings,
        }


def build_singularity_launch(
    *,
    snapshot_path: Path,
    runtime_image: Path,
    sglang_source: Path,
    ffprobe_adapter: Path,
    output_path: Path,
    selected_gpus: tuple[int, ...],
    dit_layerwise_resident_layers: int,
    port: int = 30010,
) -> LaunchPlan:
    """Build the pinned two-GPU reference launch command."""
    executable = shutil.which("singularity")
    if executable is None:
        message = "singularity is required for the pinned benchmark runtime"
        raise ValidationError(message)
    for name, path, kind in (
        ("snapshot", snapshot_path, "directory"),
        ("runtime image", runtime_image, "file"),
        ("SGLang source", sglang_source, "directory"),
        ("ffprobe adapter", ffprobe_adapter, "file"),
    ):
        valid = path.is_dir() if kind == "directory" else path.is_file()
        if not valid:
            message = f"{name} {kind} is missing: {path}"
            raise ValidationError(message)
    if not ffprobe_adapter.stat().st_mode & 0o111:
        message = f"ffprobe adapter is not executable: {ffprobe_adapter}"
        raise ValidationError(message)
    if len(selected_gpus) != 2 or len(set(selected_gpus)) != 2:
        message = "the pinned launch profile requires two distinct GPUs"
        raise ValidationError(message)
    if not (1 <= port <= 65535):
        message = "port must be between 1 and 65535"
        raise ValidationError(message)
    if (
        not isinstance(dit_layerwise_resident_layers, int)
        or isinstance(dit_layerwise_resident_layers, bool)
        or not 1 <= dit_layerwise_resident_layers <= 50
    ):
        message = "DiT layerwise resident layers must be between 1 and 50"
        raise ValidationError(message)

    output_path.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_path.resolve()
    image = runtime_image.resolve()
    source = sglang_source.resolve()
    media_probe = ffprobe_adapter.resolve()
    output = output_path.resolve()
    visible_devices = ",".join(str(index) for index in selected_gpus)
    argv = (
        executable,
        "exec",
        "--nv",
        "--cleanenv",
        "--env",
        f"CUDA_VISIBLE_DEVICES={visible_devices}",
        "--env",
        "PYTHONPATH=/opt/h3fast/sglang/python",
        "--env",
        f"SGLANG_GIT_COMMIT={REFERENCE_SGLANG_COMMIT}",
        "--env",
        "SGLANG_USE_RUNAI_MODEL_STREAMER=false",
        "--bind",
        f"{snapshot}:/models/MiniMax-H3:ro",
        "--bind",
        f"{source}:/opt/h3fast/sglang:ro",
        "--bind",
        f"{media_probe}:/usr/local/bin/ffprobe:ro",
        "--bind",
        f"{output}:/outputs",
        "--pwd",
        "/outputs",
        str(image),
        "sglang",
        "serve",
        "--model-path",
        "/models/MiniMax-H3",
        "--model-variant",
        "fl2va",
        "--num-gpus",
        "2",
        "--tp-size",
        "2",
        "--ulysses-degree",
        "1",
        "--performance-mode",
        "memory",
        "--layerwise-offload-components",
        "dit,text_encoder,vae",
        "--dit-offload-prefetch-size",
        "1",
        "--dit-layerwise-resident-layers",
        str(dit_layerwise_resident_layers),
        "--enable-torch-compile",
        "false",
        "--port",
        str(port),
    )
    return LaunchPlan(
        argv=argv,
        selected_gpus=selected_gpus,
        sglang_revision=REFERENCE_SGLANG_COMMIT,
        base_image=REFERENCE_RUNTIME_IMAGE,
        ffprobe_adapter_sha256=sha256_file(media_probe),
        runtime_settings={
            "dit_layerwise_resident_layers": dit_layerwise_resident_layers,
        },
    )
