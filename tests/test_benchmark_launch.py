"""Tests for the pinned Singularity launch plan."""

from pathlib import Path

import pytest

from h3fast.backends.sglang import REFERENCE_SGLANG_COMMIT
from h3fast.benchmarks.launch import build_singularity_launch
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
    snapshot.mkdir()
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
    assert plan.runtime_settings == {"dit_layerwise_resident_layers": 40}
    assert any("/usr/local/bin/ffprobe:ro" in value for value in plan.argv)
    assert len(plan.ffprobe_adapter_sha256) == 64
    assert plan.to_dict()["shell_command"].startswith("/usr/bin/singularity exec")
    assert output.is_dir()


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
    snapshot.mkdir()
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

    with pytest.raises(ValidationError, match="two distinct GPUs"):
        build_singularity_launch(
            **arguments, selected_gpus=(1,), dit_layerwise_resident_layers=20
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
