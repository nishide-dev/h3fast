"""Fail-closed preflight checks for the pinned MiniMax H3 baseline."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from h3fast.backends.sglang import REFERENCE_SGLANG_COMMIT
from h3fast.benchmarks.protocol import validate_protocol
from h3fast.exceptions import ValidationError
from h3fast.manifest import inspect_snapshot
from h3fast.manifest.checksums import sha256_file

CheckStatus = Literal["pass", "fail"]


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One mandatory baseline preflight check."""

    name: str
    status: CheckStatus
    message: str
    details: object | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable check data."""
        value: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.details is not None:
            value["details"] = self.details
        return value


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Reproducible baseline environment readiness report."""

    protocol_id: str
    selected_gpus: tuple[int, ...]
    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        """Return whether every mandatory check passed."""
        return all(check.status == "pass" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data."""
        return {
            "schema_version": "1.0",
            "protocol_id": self.protocol_id,
            "ready": self.ready,
            "selected_gpus": list(self.selected_gpus),
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class NvidiaDevice:
    """One physical NVIDIA device reported by nvidia-smi."""

    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int
    driver_version: str
    compute_capability: str
    uuid: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable device data."""
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "memory_free_mib": self.memory_free_mib,
            "driver_version": self.driver_version,
            "compute_capability": self.compute_capability,
            "uuid": self.uuid,
        }


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _protocol(path: Path) -> dict[str, object]:
    validate_protocol(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):  # Defensive; validate_protocol already checks this.
        msg = "benchmark protocol root must be an object"
        raise ValidationError(msg)
    return value


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"benchmark protocol field {field!r} must be an object"
        raise ValidationError(msg)
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"benchmark protocol field {field!r} must be a positive integer"
        raise ValidationError(msg)
    return value


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _normalize_gpu_name(value: str) -> str:
    """Normalize the optional vendor prefix emitted by nvidia-smi versions."""
    name = " ".join(value.split())
    return name.removeprefix("NVIDIA ")


def _query_nvidia() -> tuple[
    tuple[NvidiaDevice, ...], dict[str, list[dict[str, object]]]
]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        msg = "nvidia-smi is required for the GPU baseline"
        raise ValidationError(msg)
    gpu_result = _run(
        [
            executable,
            "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_result.returncode != 0:
        detail = gpu_result.stderr.strip() or "unknown nvidia-smi failure"
        message = f"nvidia-smi GPU query failed: {detail}"
        raise ValidationError(message)

    devices: list[NvidiaDevice] = []
    for row in csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 7:
            message = "nvidia-smi returned an unexpected GPU row"
            raise ValidationError(message)
        devices.append(
            NvidiaDevice(
                index=int(row[0]),
                name=row[1].strip(),
                memory_total_mib=int(row[2]),
                memory_free_mib=int(row[3]),
                driver_version=row[4].strip(),
                compute_capability=row[5].strip(),
                uuid=row[6].strip(),
            )
        )

    app_result = _run(
        [
            executable,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if app_result.returncode != 0:
        detail = app_result.stderr.strip() or "unknown nvidia-smi failure"
        message = f"nvidia-smi process query failed: {detail}"
        raise ValidationError(message)
    applications: dict[str, list[dict[str, object]]] = {}
    for row in csv.reader(app_result.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 4:
            message = "nvidia-smi returned an unexpected process row"
            raise ValidationError(message)
        applications.setdefault(row[0].strip(), []).append(
            {
                "pid": int(row[1]),
                "process_name": row[2].strip(),
                "used_memory_mib": int(row[3]),
            }
        )
    return tuple(devices), applications


def _ram_total_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError) as error:
        message = f"could not determine host RAM: {error}"
        raise ValidationError(message) from error
    message = "could not determine host RAM"
    raise ValidationError(message)


def _existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            message = f"could not resolve an existing parent for {path}"
            raise ValidationError(message)
        candidate = candidate.parent
    return candidate


def _source_revision(path: Path) -> str:
    result = _run(["git", "-C", str(path), "rev-parse", "HEAD"])
    if result.returncode != 0:
        detail = result.stderr.strip() or "not a Git checkout"
        message = f"could not inspect SGLang source: {detail}"
        raise ValidationError(message)
    return result.stdout.strip()


def _hf_snapshot_revision(path: Path) -> str:
    metadata_root = path / ".cache" / "huggingface" / "download"
    revisions: set[str] = set()
    try:
        for metadata_path in metadata_root.rglob("*.metadata"):
            first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0]
            revisions.add(first_line)
    except (IndexError, OSError) as error:
        message = f"could not inspect Hugging Face snapshot metadata: {error}"
        raise ValidationError(message) from error
    if len(revisions) != 1:
        message = "snapshot must contain one verified Hugging Face revision"
        raise ValidationError(message)
    return revisions.pop()


def _validate_hf_snapshot_revision(path: Path, expected_revision: str) -> None:
    actual_revision = _hf_snapshot_revision(path)
    if actual_revision != expected_revision:
        message = f"snapshot metadata revision is {actual_revision}, expected {expected_revision}"
        raise ValidationError(message)


def run_preflight(
    protocol_path: Path,
    *,
    snapshot_path: Path,
    selected_gpus: tuple[int, ...],
    output_path: Path,
    sglang_source: Path | None = None,
    runtime_image: Path | None = None,
    ffprobe_adapter: Path | None = None,
) -> PreflightReport:
    """Check a local baseline environment without importing GPU frameworks."""
    protocol = _protocol(protocol_path)
    protocol_id = str(protocol["protocol_id"])
    environment = _mapping(protocol.get("environment"), "environment")
    accelerator = _mapping(environment.get("accelerator"), "environment.accelerator")
    host = _mapping(environment.get("host"), "environment.host")
    software = _mapping(environment.get("software"), "environment.software")
    base_model = _mapping(protocol.get("base_model"), "base_model")
    checks: list[PreflightCheck] = []

    expected_count = _positive_int(accelerator.get("count"), "accelerator.count")
    if len(selected_gpus) == expected_count and len(set(selected_gpus)) == len(
        selected_gpus
    ):
        checks.append(
            PreflightCheck("gpu-selection", "pass", f"selected {expected_count} GPUs")
        )
    else:
        checks.append(
            PreflightCheck(
                "gpu-selection",
                "fail",
                f"protocol requires {expected_count} distinct GPUs",
            )
        )

    try:
        devices, applications = _query_nvidia()
        by_index = {device.index: device for device in devices}
        chosen = [by_index[index] for index in selected_gpus]
        expected_model = str(accelerator.get("model"))
        min_free = _positive_int(
            accelerator.get("min_free_memory_mib"),
            "accelerator.min_free_memory_mib",
        )
        errors: list[str] = []
        for device in chosen:
            if _normalize_gpu_name(device.name) != _normalize_gpu_name(expected_model):
                errors.append(
                    f"GPU {device.index} is {device.name}, expected {expected_model}"
                )
            if device.memory_free_mib < min_free:
                errors.append(
                    f"GPU {device.index} has {device.memory_free_mib} MiB free, "
                    f"requires {min_free} MiB"
                )
            processes = applications.get(device.uuid, [])
            if processes:
                errors.append(f"GPU {device.index} has active compute processes")
        if errors:
            checks.append(PreflightCheck("gpu-capacity", "fail", "; ".join(errors)))
        else:
            checks.append(
                PreflightCheck(
                    "gpu-capacity",
                    "pass",
                    "selected GPUs match the protocol and have no compute processes",
                    [device.to_dict() for device in chosen],
                )
            )
        minimum_driver = "535.0"
        drivers = [device.driver_version for device in chosen]
        if all(
            _version_tuple(version) >= _version_tuple(minimum_driver)
            for version in drivers
        ):
            checks.append(
                PreflightCheck(
                    "driver",
                    "pass",
                    f"driver supports the pinned CUDA 12.9 image (minimum {minimum_driver})",
                    {"versions": drivers},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "driver",
                    "fail",
                    f"CUDA 12.9 image requires driver {minimum_driver} or newer",
                    {"versions": drivers},
                )
            )
    except (KeyError, OSError, subprocess.TimeoutExpired, ValidationError) as error:
        checks.append(PreflightCheck("gpu-capacity", "fail", str(error)))

    min_ram_gib = _positive_int(host.get("min_ram_gib"), "host.min_ram_gib")
    try:
        ram_bytes = _ram_total_bytes()
        ram_gib = ram_bytes / 1024**3
        status: CheckStatus = "pass" if ram_gib >= min_ram_gib else "fail"
        checks.append(
            PreflightCheck(
                "host-ram",
                status,
                f"host RAM is {ram_gib:.1f} GiB; requires {min_ram_gib} GiB",
            )
        )
    except ValidationError as error:
        checks.append(PreflightCheck("host-ram", "fail", str(error)))

    revision = str(base_model.get("revision"))
    try:
        snapshot = inspect_snapshot(
            snapshot_path,
            variant="fl2va",
            base_revision=revision,
            base_model=str(base_model.get("repository")),
        )
        _validate_hf_snapshot_revision(snapshot_path, revision)
        snapshot_bytes = sum(file.size for file in snapshot.files)
        min_snapshot_bytes = _positive_int(
            base_model.get("min_snapshot_bytes"), "base_model.min_snapshot_bytes"
        )
        status = "pass" if snapshot_bytes >= min_snapshot_bytes else "fail"
        checks.append(
            PreflightCheck(
                "snapshot",
                status,
                f"snapshot contains {snapshot_bytes} bytes; requires at least "
                f"{min_snapshot_bytes} bytes",
                {
                    "path": str(snapshot_path.resolve()),
                    "revision": revision,
                    "file_count": len(snapshot.files),
                },
            )
        )
    except ValidationError as error:
        checks.append(PreflightCheck("snapshot", "fail", str(error)))

    min_output_gib = _positive_int(
        host.get("min_output_free_gib"), "host.min_output_free_gib"
    )
    output_parent = _existing_parent(output_path)
    free_bytes = shutil.disk_usage(output_parent).free
    free_gib = free_bytes / 1024**3
    status = "pass" if free_gib >= min_output_gib else "fail"
    checks.append(
        PreflightCheck(
            "output-storage",
            status,
            f"output filesystem has {free_gib:.1f} GiB free; requires "
            f"{min_output_gib} GiB",
        )
    )

    if sglang_source is not None:
        try:
            actual_revision = _source_revision(sglang_source)
            status = "pass" if actual_revision == REFERENCE_SGLANG_COMMIT else "fail"
            checks.append(
                PreflightCheck(
                    "sglang-source",
                    status,
                    f"SGLang source revision is {actual_revision}",
                    {"expected_revision": REFERENCE_SGLANG_COMMIT},
                )
            )
        except (OSError, subprocess.TimeoutExpired, ValidationError) as error:
            checks.append(PreflightCheck("sglang-source", "fail", str(error)))

    if runtime_image is not None:
        if runtime_image.is_file() and runtime_image.stat().st_size > 0:
            actual_sha256 = sha256_file(runtime_image)
            expected_sha256 = str(software.get("sglang_runtime_sif_sha256"))
            status = "pass" if actual_sha256 == expected_sha256 else "fail"
            checks.append(
                PreflightCheck(
                    "runtime-image",
                    status,
                    "runtime image digest matches the protocol"
                    if status == "pass"
                    else "runtime image digest does not match the protocol",
                    {
                        "path": str(runtime_image.resolve()),
                        "size": runtime_image.stat().st_size,
                        "sha256": actual_sha256,
                        "expected_sha256": expected_sha256,
                    },
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "runtime-image",
                    "fail",
                    f"runtime image is missing: {runtime_image}",
                )
            )

    if ffprobe_adapter is not None:
        if (
            ffprobe_adapter.is_file()
            and ffprobe_adapter.stat().st_size > 0
            and ffprobe_adapter.stat().st_mode & 0o111
        ):
            actual_sha256 = sha256_file(ffprobe_adapter)
            expected_sha256 = str(software.get("ffprobe_adapter_sha256"))
            status = "pass" if actual_sha256 == expected_sha256 else "fail"
            checks.append(
                PreflightCheck(
                    "ffprobe-adapter",
                    status,
                    "ffprobe adapter digest matches the protocol"
                    if status == "pass"
                    else "ffprobe adapter digest does not match the protocol",
                    {
                        "path": str(ffprobe_adapter.resolve()),
                        "size": ffprobe_adapter.stat().st_size,
                        "sha256": actual_sha256,
                        "expected_sha256": expected_sha256,
                    },
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "ffprobe-adapter",
                    "fail",
                    f"ffprobe adapter is missing, empty, or not executable: "
                    f"{ffprobe_adapter}",
                )
            )

    return PreflightReport(
        protocol_id=protocol_id,
        selected_gpus=selected_gpus,
        checks=tuple(checks),
    )
