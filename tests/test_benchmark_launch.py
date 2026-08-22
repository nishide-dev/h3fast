"""Tests for the pinned Singularity launch plan."""

from pathlib import Path

import pytest

from h3fast.backends.sglang import REFERENCE_SGLANG_COMMIT
from h3fast.benchmarks.launch import TEXT_ENCODER_MOUNT, build_singularity_launch
from h3fast.exceptions import ValidationError


def test_build_singularity_launch_requires_singularity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("h3fast.benchmarks.launch.shutil.which", lambda _name: None)

    with pytest.raises(ValidationError, match="singularity is required"):
        build_singularity_launch(
            snapshot_path=tmp_path,
            runtime_image=tmp_path / "runtime.sif",
            sglang_source=tmp_path,
            ffprobe_adapter=tmp_path / "ffprobe.py",
            output_path=tmp_path,
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=20,
        )


def test_build_singularity_launch_is_pinned(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    output = tmp_path / "server-output"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )

    plan = build_singularity_launch(
        snapshot_path=snapshot,
        runtime_image=image,
        sglang_source=source,
        ffprobe_adapter=adapter,
        output_path=output,
        selected_gpus=(1, 2),
        dit_layerwise_resident_layers=40,
    )

    assert plan.sglang_revision == REFERENCE_SGLANG_COMMIT
    assert "CUDA_VISIBLE_DEVICES=1,2" in plan.argv
    assert f"SGLANG_GIT_COMMIT={REFERENCE_SGLANG_COMMIT}" in plan.argv
    assert "SGLANG_USE_RUNAI_MODEL_STREAMER=false" in plan.argv
    assert "--enable-torch-compile" in plan.argv
    assert "false" in plan.argv
    resident_index = plan.argv.index("--dit-layerwise-resident-layers")
    assert plan.argv[resident_index + 1] == "40"
    assert plan.runtime_settings == {
        "dit_layerwise_resident_layers": 40,
        "attention_backend": "auto",
        "model_variant": "fl2va",
        "lora": None,
        "quantization": None,
        "synchronized_stage_profiling": False,
        "tensor_parallel_size": 2,
        "ulysses_degree": 1,
        "text_encoder_override": False,
        "layerwise_offload_components": ["dit", "text_encoder"],
    }
    variant_index = plan.argv.index("--model-variant")
    assert plan.argv[variant_index + 1] == "fl2va"
    assert any("/usr/local/bin/ffprobe:ro" in value for value in plan.argv)
    assert len(plan.ffprobe_adapter_sha256) == 64
    assert plan.to_dict()["shell_command"].startswith("/usr/bin/singularity exec")
    assert output.is_dir()
    assert "--master-port" not in plan.argv

    parallel = build_singularity_launch(
        snapshot_path=snapshot,
        runtime_image=image,
        sglang_source=source,
        ffprobe_adapter=adapter,
        output_path=output,
        selected_gpus=(1, 2),
        dit_layerwise_resident_layers=40,
        port=30011,
        master_port=35011,
    )
    master_index = parallel.argv.index("--master-port")
    assert parallel.argv[master_index + 1] == "35011"

    sage = tmp_path / "sage-site-packages"
    (sage / "sageattention").mkdir(parents=True)
    sage_plan = build_singularity_launch(
        snapshot_path=snapshot,
        runtime_image=image,
        sglang_source=source,
        ffprobe_adapter=adapter,
        output_path=output,
        selected_gpus=(1, 2),
        dit_layerwise_resident_layers=40,
        attention_backend="sage_attn",
        sage_attention_path=sage,
    )
    # H3 resolves its attention backend lazily on the first forward, after
    # the component override context has closed. A component-scoped
    # transformer override is therefore silently lost and the DiT falls back
    # to platform auto-selection. Request the backend globally and scope the
    # text encoder down instead, since it rejects sage_attn.
    backend_index = sage_plan.argv.index("--attention-backend")
    assert sage_plan.argv[backend_index + 1] == "sage_attn"
    component_index = sage_plan.argv.index("--component-attention-backends")
    assert sage_plan.argv[component_index + 1] == "text_encoder=torch_sdpa"
    assert "transformer=sage_attn" not in sage_plan.argv
    assert any(f"{sage.resolve()}:/opt/h3fast/sage:ro" in v for v in sage_plan.argv)
    assert any("PYTHONPATH=/opt/h3fast/sage:" in v for v in sage_plan.argv)
    assert sage_plan.runtime_settings["attention_backend"] == "sage_attn"

    assets = tmp_path / "reference-assets"
    assets.mkdir()
    (assets / "frame-first.png").write_bytes(b"png")
    asset_plan = build_singularity_launch(
        snapshot_path=snapshot,
        runtime_image=image,
        sglang_source=source,
        ffprobe_adapter=adapter,
        output_path=output,
        selected_gpus=(1, 2),
        dit_layerwise_resident_layers=40,
        reference_assets_path=assets,
    )
    assert any(
        f"{assets.resolve()}:/reference-assets:ro" in value for value in asset_plan.argv
    )
    assert "--bind" in asset_plan.argv

    with pytest.raises(ValidationError, match="reference asset directory"):
        build_singularity_launch(
            snapshot_path=snapshot,
            runtime_image=image,
            sglang_source=source,
            ffprobe_adapter=adapter,
            output_path=output,
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=40,
            reference_assets_path=tmp_path / "missing-assets",
        )

    with pytest.raises(ValidationError, match="sage_attn requires"):
        build_singularity_launch(
            snapshot_path=snapshot,
            runtime_image=image,
            sglang_source=source,
            ffprobe_adapter=adapter,
            output_path=output,
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=40,
            attention_backend="sage_attn",
        )

    with pytest.raises(ValidationError, match="master port"):
        build_singularity_launch(
            snapshot_path=snapshot,
            runtime_image=image,
            sglang_source=source,
            ffprobe_adapter=adapter,
            output_path=output,
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=40,
            master_port=70000,
        )


def test_launch_serves_the_requested_model_variant(tmp_path: Path, monkeypatch) -> None:
    """A partition only serves its own task families, so the variant is explicit."""
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    (snapshot / "Ref2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "selected_gpus": (1, 2),
        "dit_layerwise_resident_layers": 40,
    }

    plan = build_singularity_launch(**arguments, model_variant="ref2va")

    variant_index = plan.argv.index("--model-variant")
    assert plan.argv[variant_index + 1] == "ref2va"
    assert plan.runtime_settings["model_variant"] == "ref2va"

    with pytest.raises(ValidationError, match="unsupported model variant"):
        build_singularity_launch(**arguments, model_variant="t2va")


def test_launch_requires_the_variant_weights_in_the_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """Serving a variant whose weights are absent fails at generation time."""
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "selected_gpus": (1, 2),
        "dit_layerwise_resident_layers": 40,
    }

    assert build_singularity_launch(**arguments, model_variant="fl2va").argv

    with pytest.raises(ValidationError, match="Ref2VA"):
        build_singularity_launch(**arguments, model_variant="ref2va")


def test_launch_binds_a_digest_verified_lora(tmp_path: Path, monkeypatch) -> None:
    """A LoRA changes output bytes, so the launch pins and verifies it."""
    import hashlib

    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    weight = lora_dir / "turbo.safetensors"
    weight.write_bytes(b"lora-weights")
    digest = hashlib.sha256(b"lora-weights").hexdigest()
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    lora = {
        "nickname": "turbo",
        "weight_name": "turbo.safetensors",
        "weight_sha256": digest,
        "scale": 1.0,
        "merge_mode": "auto",
        "source": "example/repo@" + "4" * 40,
    }
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "selected_gpus": (1, 2),
        "dit_layerwise_resident_layers": 40,
    }

    plan = build_singularity_launch(**arguments, lora=lora, lora_path=lora_dir)

    lora_index = plan.argv.index("--lora-path")
    assert plan.argv[lora_index + 1] == "/opt/h3fast/lora"
    weight_index = plan.argv.index("--lora-weight-name")
    assert plan.argv[weight_index + 1] == "turbo.safetensors"
    nickname_index = plan.argv.index("--lora-nickname")
    assert plan.argv[nickname_index + 1] == "turbo"
    scale_index = plan.argv.index("--lora-scale")
    assert plan.argv[scale_index + 1] == "1.0"
    merge_index = plan.argv.index("--lora-merge-mode")
    assert plan.argv[merge_index + 1] == "auto"
    assert any(
        f"{lora_dir.resolve()}:/opt/h3fast/lora:ro" in value for value in plan.argv
    )
    assert plan.runtime_settings["lora"] == lora
    assert str(lora_dir) not in str(plan.runtime_settings)

    with pytest.raises(ValidationError, match="digest"):
        build_singularity_launch(
            **arguments,
            lora={**lora, "weight_sha256": "0" * 64},
            lora_path=lora_dir,
        )
    with pytest.raises(ValidationError, match="lora"):
        build_singularity_launch(
            **arguments,
            lora={**lora, "weight_name": "missing.safetensors"},
            lora_path=lora_dir,
        )
    with pytest.raises(ValidationError, match="lora"):
        build_singularity_launch(**arguments, lora=lora)
    with pytest.raises(ValidationError, match="lora"):
        build_singularity_launch(**arguments, lora_path=lora_dir)


def test_launch_passes_a_supported_quantization(tmp_path: Path, monkeypatch) -> None:
    """Quantization changes numerics, so only vetted methods are accepted."""
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "selected_gpus": (1, 2),
        "dit_layerwise_resident_layers": 40,
    }

    plan = build_singularity_launch(**arguments, quantization="fp8")

    index = plan.argv.index("--quantization")
    assert plan.argv[index + 1] == "fp8"
    assert plan.runtime_settings["quantization"] == "fp8"

    # Default keeps the pinned argv byte-identical.
    assert "--quantization" not in build_singularity_launch(**arguments).argv

    # Methods needing other vendors or pre-quantized checkpoints are rejected.
    for method in ("mxfp4", "mxfp8", "modelslim", "int4", ""):
        with pytest.raises(ValidationError, match="quantization"):
            build_singularity_launch(**arguments, quantization=method)


def test_launch_can_enable_synchronized_stage_profiling(
    tmp_path: Path, monkeypatch
) -> None:
    """Stage attribution needs the sync flag; without it decode inflates 2-3x."""
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "selected_gpus": (1, 2),
        "dit_layerwise_resident_layers": 40,
    }

    plan = build_singularity_launch(**arguments, synchronized_stage_profiling=True)

    assert "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1" in plan.argv
    assert plan.runtime_settings["synchronized_stage_profiling"] is True

    # The pinned default must stay byte-identical.
    default = build_singularity_launch(**arguments)
    assert not any("SYNC_STAGE_PROFILING" in value for value in default.argv)
    assert default.runtime_settings["synchronized_stage_profiling"] is False


def test_launch_supports_a_single_gpu_topology(tmp_path: Path, monkeypatch) -> None:
    """TP1 removes the AllReduce that dominates the TP2 denoise profile."""
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "dit_layerwise_resident_layers": 40,
    }

    plan = build_singularity_launch(**arguments, selected_gpus=(1,))

    assert "CUDA_VISIBLE_DEVICES=1" in plan.argv
    gpus_index = plan.argv.index("--num-gpus")
    assert plan.argv[gpus_index + 1] == "1"
    tp_index = plan.argv.index("--tp-size")
    assert plan.argv[tp_index + 1] == "1"
    assert plan.runtime_settings["tensor_parallel_size"] == 1

    # The pinned two-GPU profile keeps TP2.
    two = build_singularity_launch(**arguments, selected_gpus=(1, 2))
    assert two.argv[two.argv.index("--tp-size") + 1] == "2"
    assert two.runtime_settings["tensor_parallel_size"] == 2

    # Only 1 or 2 GPUs are validated topologies; anything else fails closed.
    for gpus in ((), (1, 1), (1, 2, 3)):
        with pytest.raises(ValidationError, match="GPU"):
            build_singularity_launch(**arguments, selected_gpus=gpus)


def test_launch_supports_ulysses_sequence_parallel(tmp_path: Path, monkeypatch) -> None:
    """World size = TP x Ulysses x Ring; H3's 56 heads must divide by Ulysses."""
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "selected_gpus": (1, 2),
        "dit_layerwise_resident_layers": 40,
    }

    plan = build_singularity_launch(**arguments, ulysses_degree=2)

    assert plan.argv[plan.argv.index("--ulysses-degree") + 1] == "2"
    assert plan.argv[plan.argv.index("--tp-size") + 1] == "1"
    assert plan.runtime_settings["ulysses_degree"] == 2
    assert plan.runtime_settings["tensor_parallel_size"] == 1

    # Default keeps the pinned TP2 topology.
    default = build_singularity_launch(**arguments)
    assert default.argv[default.argv.index("--ulysses-degree") + 1] == "1"
    assert default.argv[default.argv.index("--tp-size") + 1] == "2"
    assert default.runtime_settings["ulysses_degree"] == 1

    # World size must equal TP x Ulysses; degree 3 does not fit two GPUs.
    with pytest.raises(ValidationError, match="world size"):
        build_singularity_launch(**arguments, ulysses_degree=3)

    # H3 has 56 attention heads, so the degree must divide them evenly.
    single = {**arguments, "selected_gpus": (1,)}
    with pytest.raises(ValidationError, match="world size"):
        build_singularity_launch(**single, ulysses_degree=2)


def test_launch_can_override_the_text_encoder(tmp_path: Path, monkeypatch) -> None:
    """A pre-quantized text encoder cuts VRAM but changes numerics."""
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    (encoder / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "selected_gpus": (1, 2),
        "dit_layerwise_resident_layers": 40,
    }

    plan = build_singularity_launch(**arguments, text_encoder_path=encoder)

    index = plan.argv.index("--text-encoder-path")
    assert plan.argv[index + 1] == TEXT_ENCODER_MOUNT
    assert any(
        f"{encoder.resolve()}:{TEXT_ENCODER_MOUNT}:ro" in value for value in plan.argv
    )
    assert plan.runtime_settings["text_encoder_override"] is True

    default = build_singularity_launch(**arguments)
    assert "--text-encoder-path" not in default.argv
    assert default.runtime_settings["text_encoder_override"] is False

    # A directory without a config is not a loadable component.
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(ValidationError, match="text encoder"):
        build_singularity_launch(**arguments, text_encoder_path=bare)
    with pytest.raises(ValidationError, match="text encoder"):
        build_singularity_launch(**arguments, text_encoder_path=tmp_path / "missing")


def test_launch_can_keep_the_vae_resident(tmp_path: Path, monkeypatch) -> None:
    """Offload placement changes transfer scheduling, not the compute graph."""
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "server-output",
        "selected_gpus": (1, 2),
        "dit_layerwise_resident_layers": 40,
    }

    plan = build_singularity_launch(
        **arguments, layerwise_offload_components=("dit", "text_encoder")
    )

    index = plan.argv.index("--layerwise-offload-components")
    assert plan.argv[index + 1] == "dit,text_encoder"
    assert plan.runtime_settings["layerwise_offload_components"] == [
        "dit",
        "text_encoder",
    ]

    # The default keeps the VAE resident (experiment 0020: 65% faster decode,
    # bit-identical artifacts).
    default = build_singularity_launch(**arguments)
    default_index = default.argv.index("--layerwise-offload-components")
    assert default.argv[default_index + 1] == "dit,text_encoder"

    # The DiT must stay offloaded; its residency knob assumes it.
    for components in ((), ("vae",), ("dit", "dit"), ("dit", "unknown")):
        with pytest.raises(ValidationError, match="offload"):
            build_singularity_launch(
                **arguments, layerwise_offload_components=components
            )


def test_build_singularity_launch_rejects_missing_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )

    with pytest.raises(ValidationError, match="snapshot directory is missing"):
        build_singularity_launch(
            snapshot_path=tmp_path / "missing",
            runtime_image=tmp_path / "missing.sif",
            sglang_source=tmp_path / "source",
            ffprobe_adapter=tmp_path / "missing-probe.py",
            output_path=tmp_path / "output",
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=20,
        )


def test_build_singularity_launch_rejects_bad_gpu_count_and_port(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "source"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    (snapshot / "FL2VA").mkdir(parents=True)
    source.mkdir()
    image.write_bytes(b"image")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.launch.shutil.which", lambda _name: "/usr/bin/singularity"
    )
    arguments = {
        "snapshot_path": snapshot,
        "runtime_image": image,
        "sglang_source": source,
        "ffprobe_adapter": adapter,
        "output_path": tmp_path / "output",
    }

    with pytest.raises(ValidationError, match="one or two distinct GPUs"):
        build_singularity_launch(
            **arguments, selected_gpus=(1, 2, 3), dit_layerwise_resident_layers=20
        )
    with pytest.raises(ValidationError, match="port"):
        build_singularity_launch(
            **arguments,
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=20,
            port=0,
        )
    with pytest.raises(ValidationError, match="resident layers"):
        build_singularity_launch(
            **arguments,
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=51,
        )
    with pytest.raises(ValidationError, match="resident layers"):
        build_singularity_launch(
            **arguments,
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=True,
        )

    adapter.chmod(0o644)
    with pytest.raises(ValidationError, match="not executable"):
        build_singularity_launch(
            **arguments,
            selected_gpus=(1, 2),
            dit_layerwise_resident_layers=20,
        )
