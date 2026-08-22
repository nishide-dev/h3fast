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


REFERENCE_ASSETS_MOUNT = "/reference-assets"
LORA_MOUNT = "/opt/h3fast/lora"
TEXT_ENCODER_MOUNT = "/opt/h3fast/text-encoder"

# Only methods that quantize online from the BF16 snapshot on the tested
# hardware. Other SGLang methods need a pre-quantized checkpoint or
# another vendor (ROCm MI350+, Ascend NPU), so they fail closed here.
SUPPORTED_QUANTIZATION = ("fp8",)

# MiniMax H3 DiT attention head count; the sequence-parallel degree must
# divide it evenly.
H3_ATTENTION_HEADS = 56

# Components the pinned profile offloads layer by layer. The DiT must stay
# in the set because dit_layerwise_resident_layers only means something
# for an offloaded DiT.
OFFLOADABLE_COMPONENTS = ("dit", "text_encoder", "vae")
# Keeping the video VAE resident cut decode by 65% with bit-identical
# artifacts (experiment 0020), so it is no longer offloaded by default.
DEFAULT_OFFLOAD_COMPONENTS = ("dit", "text_encoder")

# A served partition only answers its own task families. Requesting a family
# outside the loaded variant fails in the scheduler after the model is
# resident, so the variant is selected up front and checked against the
# snapshot rather than discovered at generation time.
MODEL_VARIANT_TASKS: dict[str, tuple[str, ...]] = {
    "fl2va": ("t2va", "fl2va"),
    "ref2va": ("t2va", "ref2va"),
}
_VARIANT_SNAPSHOT_DIRS = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}


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
    reference_assets_path: Path | None = None,
    model_variant: str = "fl2va",
    lora: dict[str, object] | None = None,
    lora_path: Path | None = None,
    quantization: str | None = None,
    synchronized_stage_profiling: bool = False,
    ulysses_degree: int = 1,
    text_encoder_path: Path | None = None,
    layerwise_offload_components: tuple[str, ...] = DEFAULT_OFFLOAD_COMPONENTS,
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
    # TP2 is the pinned profile; TP1 is supported upstream (the H3 DiT has a
    # tp_size == 1 path) and removes the AllReduce that dominates the TP2
    # denoise profile. No other topology has been validated here.
    if len(selected_gpus) not in (1, 2) or len(set(selected_gpus)) != len(
        selected_gpus
    ):
        message = "the launch profile requires one or two distinct GPUs"
        raise ValidationError(message)
    if not (1 <= port <= 65535):
        message = "port must be between 1 and 65535"
        raise ValidationError(message)
    if master_port is not None and (
        not (1 <= master_port <= 65535) or master_port == port
    ):
        message = "master port must be between 1 and 65535 and differ from port"
        raise ValidationError(message)
    if reference_assets_path is not None and not reference_assets_path.is_dir():
        message = f"reference asset directory is missing: {reference_assets_path}"
        raise ValidationError(message)
    if model_variant not in MODEL_VARIANT_TASKS:
        supported = ", ".join(sorted(MODEL_VARIANT_TASKS))
        message = f"unsupported model variant {model_variant!r}; expected {supported}"
        raise ValidationError(message)
    variant_dir = _VARIANT_SNAPSHOT_DIRS[model_variant]
    if not (snapshot_path / variant_dir).is_dir():
        message = (
            f"model variant {model_variant!r} requires {variant_dir} weights in the "
            f"snapshot: {snapshot_path / variant_dir}"
        )
        raise ValidationError(message)
    # World size = TP x Ulysses x Ring, and H3's 56 attention heads must
    # divide evenly across the sequence-parallel degree.
    world_size = len(selected_gpus)
    if (
        not isinstance(ulysses_degree, int)
        or isinstance(ulysses_degree, bool)
        or ulysses_degree < 1
        or world_size % ulysses_degree != 0
        or H3_ATTENTION_HEADS % ulysses_degree != 0
    ):
        message = (
            f"ulysses degree {ulysses_degree!r} must divide the world size "
            f"({world_size}) and H3's {H3_ATTENTION_HEADS} attention heads"
        )
        raise ValidationError(message)
    # A component override must be a loadable HF directory; SGLang resolves
    # the component by path and reads its config, so a bare directory or a
    # single weight file would fail only after the model is resident.
    if (
        text_encoder_path is not None
        and not (text_encoder_path / "config.json").is_file()
    ):
        message = (
            "text encoder override must be a directory containing config.json: "
            f"{text_encoder_path}"
        )
        raise ValidationError(message)
    offload = tuple(layerwise_offload_components)
    if (
        "dit" not in offload
        or len(set(offload)) != len(offload)
        or any(component not in OFFLOADABLE_COMPONENTS for component in offload)
    ):
        supported = ", ".join(OFFLOADABLE_COMPONENTS)
        message = (
            "layerwise offload components must be distinct values from "
            f"{supported} and must include dit; got {offload}"
        )
        raise ValidationError(message)
    if quantization is not None and quantization not in SUPPORTED_QUANTIZATION:
        supported = ", ".join(SUPPORTED_QUANTIZATION)
        message = (
            f"unsupported quantization {quantization!r}; this launch profile "
            f"supports online quantization methods: {supported}"
        )
        raise ValidationError(message)
    if (lora is None) != (lora_path is None):
        message = "lora settings and the lora directory must be provided together"
        raise ValidationError(message)
    if lora is not None and lora_path is not None:
        weight = lora_path / str(lora["weight_name"])
        if not weight.is_file():
            message = f"lora weight file is missing: {weight}"
            raise ValidationError(message)
        digest = sha256_file(weight)
        if digest != lora["weight_sha256"]:
            message = "lora weight digest does not match the pinned protocol identity"
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
        # Without this, queued denoise work leaks into the next blocking
        # stage and inflates the decoding stage by 2-3x, so stage times
        # cannot be used to choose an optimization target.
        *(
            ()
            if not synchronized_stage_profiling
            else ("--env", "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1")
        ),
        "--bind",
        f"{snapshot}:/models/MiniMax-H3:ro",
        "--bind",
        f"{source}:/opt/h3fast/sglang:ro",
        *(
            ()
            if sage_attention_path is None
            else ("--bind", f"{sage_attention_path.resolve()}:/opt/h3fast/sage:ro")
        ),
        # Reference-conditioned tasks read their inputs through file:// URIs
        # that the server resolves inside the container, so the asset tree
        # needs its own read-only mount at a fixed path.
        *(
            ()
            if reference_assets_path is None
            else (
                "--bind",
                f"{reference_assets_path.resolve()}:{REFERENCE_ASSETS_MOUNT}:ro",
            )
        ),
        *(
            ()
            if lora_path is None
            else ("--bind", f"{lora_path.resolve()}:{LORA_MOUNT}:ro")
        ),
        *(
            ()
            if text_encoder_path is None
            else (
                "--bind",
                f"{text_encoder_path.resolve()}:{TEXT_ENCODER_MOUNT}:ro",
            )
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
        model_variant,
        "--num-gpus",
        str(world_size),
        "--tp-size",
        str(world_size // ulysses_degree),
        "--ulysses-degree",
        str(ulysses_degree),
        "--performance-mode",
        "memory",
        "--layerwise-offload-components",
        ",".join(offload),
        "--dit-offload-prefetch-size",
        "1",
        "--dit-layerwise-resident-layers",
        str(dit_layerwise_resident_layers),
        "--enable-torch-compile",
        "false",
        *(() if quantization is None else ("--quantization", quantization)),
        *(
            ()
            if text_encoder_path is None
            else ("--text-encoder-path", TEXT_ENCODER_MOUNT)
        ),
        *(
            ()
            if lora is None
            else (
                "--lora-path",
                LORA_MOUNT,
                "--lora-weight-name",
                str(lora["weight_name"]),
                "--lora-nickname",
                str(lora["nickname"]),
                "--lora-scale",
                str(lora["scale"]),
                "--lora-merge-mode",
                str(lora["merge_mode"]),
            )
        ),
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
            "model_variant": model_variant,
            "lora": dict(lora) if lora is not None else None,
            "quantization": quantization,
            "synchronized_stage_profiling": synchronized_stage_profiling,
            "tensor_parallel_size": world_size // ulysses_degree,
            "ulysses_degree": ulysses_degree,
            "text_encoder_override": text_encoder_path is not None,
            "layerwise_offload_components": list(offload),
        },
    )
