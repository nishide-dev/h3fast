"""Tests for fail-closed baseline environment checks."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from h3fast.backends.sglang import REFERENCE_SGLANG_COMMIT
from h3fast.benchmarks.preflight import (
    NvidiaDevice,
    _hf_snapshot_revision,
    _query_nvidia,
    _run,
    _validate_hf_snapshot_revision,
    run_preflight,
)
from h3fast.exceptions import ValidationError
from h3fast.manifest.snapshot import REQUIRED_COMPONENTS


def test_nvidia_command_allows_driver_busy_interval(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def run(command, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("h3fast.benchmarks.preflight.subprocess.run", run)

    _run(["nvidia-smi"])

    assert observed["timeout"] == 30.0


def _write_protocol(path: Path) -> None:
    value = {
        "schema_version": "1.0",
        "protocol_id": "test-baseline",
        "status": "draft",
        "unresolved": ["quality reference set"],
        "base_model": {
            "repository": "MiniMaxAI/MiniMax-H3",
            "revision": "a" * 40,
            "min_snapshot_bytes": 1,
        },
        "environment": {
            "accelerator": {
                "model": "NVIDIA RTX 6000 Ada Generation",
                "count": 2,
                "min_free_memory_mib": 45000,
            },
            "host": {"min_ram_gib": 384, "min_output_free_gib": 1},
            "software": {
                "sglang": f"git:{REFERENCE_SGLANG_COMMIT}",
                "sglang_runtime_sif_sha256": hashlib.sha256(b"runtime").hexdigest(),
                "ffprobe_adapter_sha256": hashlib.sha256(
                    b"#!/usr/bin/env python3\n"
                ).hexdigest(),
            },
        },
        "measurement": {},
        "cases": [{"id": "smoke"}],
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_snapshot(root: Path) -> None:
    root.mkdir()
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    variant = root / "FL2VA"
    variant.mkdir()
    (variant / "model_index.json").write_text("{}", encoding="utf-8")
    for component in REQUIRED_COMPONENTS:
        component_path = variant / component
        component_path.mkdir()
        (component_path / "config.json").write_text("{}", encoding="utf-8")


def _devices() -> tuple[NvidiaDevice, NvidiaDevice]:
    return (
        NvidiaDevice(
            1,
            "NVIDIA RTX 6000 Ada Generation",
            49140,
            48000,
            "555.58.02",
            "8.9",
            "gpu-1",
        ),
        NvidiaDevice(
            2,
            "NVIDIA RTX 6000 Ada Generation",
            49140,
            48000,
            "555.58.02",
            "8.9",
            "gpu-2",
        ),
    )


def test_preflight_passes_for_idle_pinned_environment(tmp_path, monkeypatch) -> None:
    protocol = tmp_path / "protocol.json"
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "sglang"
    image = tmp_path / "runtime.sif"
    adapter = tmp_path / "ffprobe.py"
    _write_protocol(protocol)
    _write_snapshot(snapshot)
    source.mkdir()
    image.write_bytes(b"runtime")
    adapter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter.chmod(0o755)
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._query_nvidia", lambda: (_devices(), {})
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._ram_total_bytes", lambda: 500 * 1024**3
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._source_revision",
        lambda _path: REFERENCE_SGLANG_COMMIT,
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._hf_snapshot_revision", lambda _path: "a" * 40
    )

    report = run_preflight(
        protocol,
        snapshot_path=snapshot,
        selected_gpus=(1, 2),
        output_path=tmp_path / "results" / "preflight.json",
        sglang_source=source,
        runtime_image=image,
        ffprobe_adapter=adapter,
    )

    assert report.ready is True
    assert report.to_dict()["selected_gpus"] == [1, 2]
    assert {check.name for check in report.checks} == {
        "gpu-selection",
        "gpu-capacity",
        "driver",
        "host-ram",
        "snapshot",
        "output-storage",
        "sglang-source",
        "runtime-image",
        "ffprobe-adapter",
    }

    image.write_bytes(b"changed runtime")
    adapter.write_text("#!/usr/bin/env python3\n# changed\n", encoding="utf-8")
    adapter.chmod(0o755)
    mismatch = run_preflight(
        protocol,
        snapshot_path=snapshot,
        selected_gpus=(1, 2),
        output_path=tmp_path / "results" / "preflight.json",
        sglang_source=source,
        runtime_image=image,
        ffprobe_adapter=adapter,
    )
    failures = {check.name for check in mismatch.checks if check.status == "fail"}
    assert {"runtime-image", "ffprobe-adapter"} <= failures


def test_preflight_accepts_optional_nvidia_model_prefix(tmp_path, monkeypatch) -> None:
    protocol = tmp_path / "protocol.json"
    snapshot = tmp_path / "snapshot"
    _write_protocol(protocol)
    value = json.loads(protocol.read_text(encoding="utf-8"))
    value["environment"]["accelerator"]["model"] = "RTX 6000 Ada Generation"
    protocol.write_text(json.dumps(value), encoding="utf-8")
    _write_snapshot(snapshot)
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._query_nvidia", lambda: (_devices(), {})
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._ram_total_bytes", lambda: 500 * 1024**3
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._hf_snapshot_revision", lambda _path: "a" * 40
    )

    report = run_preflight(
        protocol,
        snapshot_path=snapshot,
        selected_gpus=(1, 2),
        output_path=tmp_path,
    )

    gpu_check = next(check for check in report.checks if check.name == "gpu-capacity")
    assert gpu_check.status == "pass"


def test_preflight_rejects_occupied_or_mismatched_gpus(tmp_path, monkeypatch) -> None:
    protocol = tmp_path / "protocol.json"
    snapshot = tmp_path / "snapshot"
    _write_protocol(protocol)
    _write_snapshot(snapshot)
    occupied = {"gpu-1": [{"pid": 123, "process_name": "other", "used_memory_mib": 1}]}
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._query_nvidia", lambda: (_devices(), occupied)
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._ram_total_bytes", lambda: 100 * 1024**3
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._hf_snapshot_revision", lambda _path: "a" * 40
    )

    report = run_preflight(
        protocol,
        snapshot_path=snapshot,
        selected_gpus=(1,),
        output_path=tmp_path,
    )

    assert report.ready is False
    failures = {check.name for check in report.checks if check.status == "fail"}
    assert {"gpu-selection", "gpu-capacity", "host-ram"} <= failures


def test_query_nvidia_parses_devices_and_compute_apps(monkeypatch) -> None:
    results = iter(
        (
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="1, NVIDIA RTX 6000 Ada Generation, 49140, 48000, 555.58.02, 8.9, gpu-1\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="gpu-1, 42, python, 512\n",
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight.shutil.which", lambda _name: "/usr/bin/nvidia-smi"
    )
    monkeypatch.setattr(
        "h3fast.benchmarks.preflight._run", lambda _command: next(results)
    )

    devices, applications = _query_nvidia()

    assert devices[0].index == 1
    assert devices[0].to_dict()["compute_capability"] == "8.9"
    assert applications["gpu-1"][0]["pid"] == 42


def test_hf_snapshot_revision_requires_one_matching_revision(tmp_path: Path) -> None:
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "one.metadata").write_text(f"{'a' * 40}\netag\n", encoding="utf-8")

    assert _hf_snapshot_revision(tmp_path) == "a" * 40
    _validate_hf_snapshot_revision(tmp_path, "a" * 40)

    (metadata / "two.metadata").write_text(f"{'b' * 40}\netag\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="one verified"):
        _hf_snapshot_revision(tmp_path)

    (metadata / "two.metadata").write_text(f"{'a' * 40}\netag\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="expected"):
        _validate_hf_snapshot_revision(tmp_path, "b" * 40)
