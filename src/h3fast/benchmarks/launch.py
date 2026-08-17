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
    runtime_settings: dict[str, object]

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
    master_port: int | None = None,
    attention_backend: str = "auto",
    sage_attention_path: Path | None = None,
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
    if master_port is not None and (
        not (1 <= master_port <= 65535) or master_port == port
    ):
        message = "master port must be between 1 and 65535 and differ from port"
        raise ValidationError(message)
    if attention_backend not in {"auto", "fa", "sage_attn"}:
        message = f"unsupported attention backend: {attention_backend}"
        raise ValidationError(message)
    if attention_backend == "sage_attn":
        if sage_attention_path is None:
            message = "attention backend sage_attn requires a SageAttention path"
            raise ValidationError(message)
        if not (sage_attention_path / "sageattention").is_dir():
            message = (
                "attention backend sage_attn requires a built SageAttention "
                f"package at {sage_attention_path}"
            )
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
        (
            "PYTHONPATH=/opt/h3fast/sglang/python"
            if sage_attention_path is None
            else "PYTHONPATH=/opt/h3fast/sage:/opt/h3fast/sglang/python"
        ),
        "--env",
        f"SGLANG_GIT_COMMIT={REFERENCE_SGLANG_COMMIT}",
        "--env",
        "SGLANG_USE_RUNAI_MODEL_STREAMER=false",
        "--bind",
        f"{snapshot}:/models/MiniMax-H3:ro",
        "--bind",
        f"{source}:/opt/h3fast/sglang:ro",
        *(
            ()
            if sage_attention_path is None
            else ("--bind", f"{sage_attention_path.resolve()}:/opt/h3fast/sage:ro")
        ),
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
        # A distinct rendezvous port allows two guarded servers on one
        # host; it does not change the compute graph or schedule.
        *(() if master_port is None else ("--master-port", str(master_port))),
        # Unlike placement settings, the attention backend changes numerics.
        #
        # H3 resolves its attention backend lazily on the first forward, by
        # which time SGLang's component override context has closed. A
        # component-scoped `transformer=` override is therefore accepted,
        # logged, and then silently lost, leaving the DiT on platform
        # auto-selection (measured: output stayed bit-identical to the
        # FlashAttention baseline). Request the backend globally so it
        # survives in server_args, and scope the text encoder down instead
        # because its attention layer rejects sage_attn.
        *(
            ()
            if attention_backend == "auto"
            else (
                "--attention-backend",
                attention_backend,
                "--component-attention-backends",
                "text_encoder=torch_sdpa",
            )
        ),
    )
    return LaunchPlan(
        argv=argv,
        selected_gpus=selected_gpus,
        sglang_revision=REFERENCE_SGLANG_COMMIT,
        base_image=REFERENCE_RUNTIME_IMAGE,
        ffprobe_adapter_sha256=sha256_file(media_probe),
        runtime_settings={
            "dit_layerwise_resident_layers": dit_layerwise_resident_layers,
            "attention_backend": attention_backend,
        },
    )
