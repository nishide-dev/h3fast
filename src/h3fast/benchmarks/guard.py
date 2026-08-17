"""Guard a local benchmark server against foreign GPU compute processes."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from h3fast.benchmarks.client import _validate_endpoint
from h3fast.benchmarks.preflight import _query_nvidia
from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from h3fast.benchmarks.launch import LaunchPlan


@dataclass(frozen=True, slots=True)
class ForeignGpuProcess:
    """A compute process on a selected GPU outside the guarded process tree."""

    gpu_index: int
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable process data."""
        return {
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
            "pid": self.pid,
            "process_name": self.process_name,
            "used_memory_mib": self.used_memory_mib,
        }


def _parent_pid(pid: int, proc_root: Path) -> int | None:
    try:
        lines = (
            (proc_root / str(pid) / "status")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
    except OSError:
        return None
    for line in lines:
        if line.startswith("PPid:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _is_descendant(pid: int, root_pid: int, proc_root: Path = Path("/proc")) -> bool:
    current = pid
    visited: set[int] = set()
    while current > 0 and current not in visited:
        if current == root_pid:
            return True
        visited.add(current)
        parent = _parent_pid(current, proc_root)
        if parent is None:
            return False
        current = parent
    return False


def find_foreign_gpu_processes(
    selected_gpus: tuple[int, ...],
    *,
    allowed_root_pid: int,
    proc_root: Path = Path("/proc"),
) -> tuple[ForeignGpuProcess, ...]:
    """Return selected-GPU processes outside one launched server process tree."""
    devices, applications = _query_nvidia()
    selected = {
        device.index: device for device in devices if device.index in selected_gpus
    }
    if set(selected) != set(selected_gpus):
        missing = sorted(set(selected_gpus) - set(selected))
        message = f"selected GPUs disappeared from nvidia-smi: {missing}"
        raise ValidationError(message)

    foreign: list[ForeignGpuProcess] = []
    for index in selected_gpus:
        device = selected[index]
        for application in applications.get(device.uuid, []):
            pid = application.get("pid")
            name = application.get("process_name")
            used_memory = application.get("used_memory_mib")
            if (
                not isinstance(pid, int)
                or not isinstance(name, str)
                or not isinstance(used_memory, int)
            ):
                message = "nvidia-smi returned invalid compute process data"
                raise ValidationError(message)
            if not _is_descendant(pid, allowed_root_pid, proc_root):
                foreign.append(
                    ForeignGpuProcess(
                        gpu_index=index,
                        gpu_uuid=device.uuid,
                        pid=pid,
                        process_name=name,
                        used_memory_mib=used_memory,
                    )
                )
    return tuple(foreign)


def _query_pmon(
    selected_gpus: tuple[int, ...],
) -> dict[int, list[tuple[int, str, int]]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        message = "nvidia-smi is required for the GPU guard"
        raise ValidationError(message)
    result = subprocess.run(  # noqa: S603
        [
            executable,
            "pmon",
            "-i",
            ",".join(str(index) for index in selected_gpus),
            "-c",
            "1",
            "-s",
            "m",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown nvidia-smi pmon failure"
        message = f"nvidia-smi pmon failed: {detail}"
        raise ValidationError(message)
    applications = {index: [] for index in selected_gpus}
    for line in result.stdout.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            message = "nvidia-smi pmon returned an unexpected process row"
            raise ValidationError(message)
        gpu_raw, pid_raw, process_type, memory_raw, _ccpm, process_name = fields
        try:
            gpu_index = int(gpu_raw)
            pid = int(pid_raw)
            used_memory_mib = int(memory_raw)
        except ValueError as error:
            message = "nvidia-smi pmon returned invalid process data"
            raise ValidationError(message) from error
        if gpu_index not in applications:
            message = "nvidia-smi pmon returned an unselected GPU"
            raise ValidationError(message)
        if "C" not in process_type:
            continue
        applications[gpu_index].append((pid, process_name, used_memory_mib))
    return applications


def _query_gpu_uuids(selected_gpus: tuple[int, ...]) -> dict[int, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        message = "nvidia-smi is required for the GPU guard"
        raise ValidationError(message)
    result = subprocess.run(  # noqa: S603
        [
            executable,
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown nvidia-smi UUID query failure"
        message = f"nvidia-smi GPU UUID query failed: {detail}"
        raise ValidationError(message)
    selected = set(selected_gpus)
    gpu_uuids: dict[int, str] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            message = "nvidia-smi returned an unexpected GPU UUID row"
            raise ValidationError(message)
        try:
            gpu_index = int(fields[0])
        except ValueError as error:
            message = "nvidia-smi returned an invalid GPU index"
            raise ValidationError(message) from error
        if gpu_index in selected:
            gpu_uuids[gpu_index] = fields[1]
    if set(gpu_uuids) != selected:
        missing = sorted(selected - set(gpu_uuids))
        message = f"selected GPUs disappeared from nvidia-smi: {missing}"
        raise ValidationError(message)
    return gpu_uuids


def find_foreign_gpu_processes_pmon(
    selected_gpus: tuple[int, ...],
    *,
    allowed_root_pid: int,
    gpu_uuids: dict[int, str],
    proc_root: Path = Path("/proc"),
) -> tuple[ForeignGpuProcess, ...]:
    """Return foreign compute processes using the lightweight pmon interface."""
    if set(gpu_uuids) != set(selected_gpus):
        message = "GPU guard UUID map does not match the selected GPUs"
        raise ValidationError(message)
    applications = _query_pmon(selected_gpus)
    foreign: list[ForeignGpuProcess] = []
    for gpu_index, values in applications.items():
        for pid, process_name, used_memory_mib in values:
            if _is_descendant(pid, allowed_root_pid, proc_root):
                continue
            foreign.append(
                ForeignGpuProcess(
                    gpu_index=gpu_index,
                    gpu_uuid=gpu_uuids[gpu_index],
                    pid=pid,
                    process_name=process_name,
                    used_memory_mib=used_memory_mib,
                )
            )
    return tuple(foreign)


def _device_holder_pids(
    selected_gpus: tuple[int, ...], proc_root: Path = Path("/proc")
) -> dict[int, set[int]]:
    targets = {f"/dev/nvidia{index}": index for index in selected_gpus}
    holders = {index: set() for index in selected_gpus}
    try:
        processes = tuple(proc_root.iterdir())
    except OSError as error:
        message = f"could not scan GPU device holders: {error}"
        raise ValidationError(message) from error
    for process_path in processes:
        if not process_path.name.isdigit():
            continue
        try:
            descriptors = tuple((process_path / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = str(descriptor.readlink())
            except OSError:
                continue
            gpu_index = targets.get(target)
            if gpu_index is not None:
                holders[gpu_index].add(int(process_path.name))
    return holders


def find_foreign_gpu_device_holders(
    selected_gpus: tuple[int, ...],
    *,
    allowed_root_pid: int,
    baseline_holders: dict[int, set[int]],
    gpu_uuids: dict[int, str],
    proc_root: Path = Path("/proc"),
) -> tuple[ForeignGpuProcess, ...]:
    """Use GPU device holders when the driver process query is unavailable."""
    selected = set(selected_gpus)
    if set(baseline_holders) != selected or set(gpu_uuids) != selected:
        message = "GPU device holder guard does not match the selected GPUs"
        raise ValidationError(message)
    holders = _device_holder_pids(selected_gpus, proc_root)
    foreign: list[ForeignGpuProcess] = []
    for gpu_index, pids in holders.items():
        for pid in sorted(pids - baseline_holders[gpu_index]):
            if _is_descendant(pid, allowed_root_pid, proc_root):
                continue
            try:
                process_name = (
                    (proc_root / str(pid) / "comm")
                    .read_text(encoding="utf-8", errors="replace")
                    .strip()
                )
            except OSError:
                process_name = "unknown"
            foreign.append(
                ForeignGpuProcess(
                    gpu_index=gpu_index,
                    gpu_uuid=gpu_uuids[gpu_index],
                    pid=pid,
                    process_name=process_name or "unknown",
                    used_memory_mib=0,
                )
            )
    return tuple(foreign)


def _health_ready(endpoint: str, timeout: float) -> bool:
    request = urllib.request.Request(  # noqa: S310
        f"{endpoint.rstrip('/')}/health", method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status == 200
    except (OSError, urllib.error.HTTPError):
        return False


def _query_with_timeout_retry(
    query: Callable[[tuple[int, ...], int], tuple[ForeignGpuProcess, ...]],
    selected_gpus: tuple[int, ...],
    root_pid: int,
) -> tuple[ForeignGpuProcess, ...]:
    """Retry one transient nvidia-smi timeout, while failing other errors immediately."""
    try:
        return query(selected_gpus, root_pid)
    except subprocess.TimeoutExpired:
        return query(selected_gpus, root_pid)


def _signal_and_wait(
    process: subprocess.Popen[bytes], process_signal: signal.Signals, timeout: float
) -> bool:
    try:
        os.killpg(process.pid, process_signal)
    except OSError:
        return False
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            process.wait(timeout=remaining)
        except KeyboardInterrupt:
            continue
        except subprocess.TimeoutExpired:
            return False
        return True


def _stop_process(process: subprocess.Popen[bytes], timeout: float = 30.0) -> None:
    if process.poll() is not None:
        return
    if _signal_and_wait(process, signal.SIGINT, timeout):
        return
    if _signal_and_wait(process, signal.SIGTERM, 10.0):
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        return
    process.wait(timeout=10.0)


def _write_failure(
    path: Path,
    *,
    started_at: datetime,
    selected_gpus: tuple[int, ...],
    server_pid: int,
    error: str,
    foreign: tuple[ForeignGpuProcess, ...] = (),
) -> None:
    value = {
        "schema_version": "1.0",
        "status": "failed",
        "started_at": started_at.isoformat(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "selected_gpus": list(selected_gpus),
        "server_pid": server_pid,
        "error": error,
        "foreign_processes": [process.to_dict() for process in foreign],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_lifecycle(
    path: Path,
    *,
    started_at: datetime,
    ready_at: datetime,
    startup_seconds: float,
    selected_gpus: tuple[int, ...],
    server_pid: int,
    endpoint: str,
    runtime_settings: dict[str, object],
) -> None:
    value = {
        "schema_version": "1.0",
        "status": "ready",
        "started_at": started_at.isoformat(),
        "ready_at": ready_at.isoformat(),
        "startup_seconds": startup_seconds,
        "selected_gpus": list(selected_gpus),
        "server_pid": server_pid,
        "endpoint": endpoint,
        "runtime_settings": runtime_settings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def serve_guarded(
    plan: LaunchPlan,
    *,
    endpoint: str,
    report_path: Path,
    lifecycle_path: Path | None = None,
    startup_timeout: float = 3600.0,
    poll_interval: float = 2.0,
    foreign_process_query: Callable[
        [tuple[int, ...], int], tuple[ForeignGpuProcess, ...]
    ]
    | None = None,
) -> None:
    """Launch and continuously guard a server until interrupted or failed."""
    if startup_timeout <= 0 or poll_interval <= 0:
        message = "startup timeout and poll interval must be positive"
        raise ValidationError(message)
    endpoint = _validate_endpoint(endpoint)
    report_path.unlink(missing_ok=True)
    if lifecycle_path is not None:
        lifecycle_path.unlink(missing_ok=True)
    gpu_uuids: dict[int, str] = {}
    baseline_holders = (
        _device_holder_pids(plan.selected_gpus) if foreign_process_query is None else {}
    )
    pmon_enabled = True
    require_pmon = False

    def default_query(
        gpus: tuple[int, ...], root_pid: int
    ) -> tuple[ForeignGpuProcess, ...]:
        nonlocal pmon_enabled
        if not gpu_uuids:
            gpu_uuids.update(_query_gpu_uuids(gpus))
        if pmon_enabled or require_pmon:
            try:
                return find_foreign_gpu_processes_pmon(
                    gpus, allowed_root_pid=root_pid, gpu_uuids=gpu_uuids
                )
            except subprocess.TimeoutExpired:
                if require_pmon:
                    raise
                pmon_enabled = False
        # During model loading, NVIDIA's process query can stall while unrelated
        # processes merely enumerate every GPU device. Keep scanning holders as
        # candidates, but do not call them compute processes until pmon confirms
        # them. Readiness is withheld until that confirmation succeeds.
        find_foreign_gpu_device_holders(
            gpus,
            allowed_root_pid=root_pid,
            baseline_holders=baseline_holders,
            gpu_uuids=gpu_uuids,
        )
        return ()

    query = foreign_process_query or default_query
    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    try:
        process = subprocess.Popen(plan.argv, start_new_session=True)  # noqa: S603
    except OSError as error:
        message = f"could not launch guarded server: {error}"
        raise ValidationError(message) from error
    ready = False
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                message = f"guarded server exited with status {return_code}"
                _write_failure(
                    report_path,
                    started_at=started_at,
                    selected_gpus=plan.selected_gpus,
                    server_pid=process.pid,
                    error=message,
                )
                raise ValidationError(message)

            try:
                foreign = _query_with_timeout_retry(
                    query, plan.selected_gpus, process.pid
                )
            except (OSError, subprocess.TimeoutExpired, ValidationError) as error:
                message = f"GPU guard query failed: {error}"
                _write_failure(
                    report_path,
                    started_at=started_at,
                    selected_gpus=plan.selected_gpus,
                    server_pid=process.pid,
                    error=message,
                )
                raise ValidationError(message) from error
            if foreign:
                pids = ", ".join(str(value.pid) for value in foreign)
                message = f"foreign GPU compute processes detected: {pids}"
                _write_failure(
                    report_path,
                    started_at=started_at,
                    selected_gpus=plan.selected_gpus,
                    server_pid=process.pid,
                    error=message,
                    foreign=foreign,
                )
                raise ValidationError(message)

            if not ready and _health_ready(endpoint, min(poll_interval, 5.0)):
                if foreign_process_query is None and not pmon_enabled:
                    try:
                        foreign = find_foreign_gpu_processes_pmon(
                            plan.selected_gpus,
                            allowed_root_pid=process.pid,
                            gpu_uuids=gpu_uuids,
                        )
                    except subprocess.TimeoutExpired:
                        time.sleep(poll_interval)
                        continue
                    if foreign:
                        pids = ", ".join(str(value.pid) for value in foreign)
                        message = f"foreign GPU compute processes detected: {pids}"
                        _write_failure(
                            report_path,
                            started_at=started_at,
                            selected_gpus=plan.selected_gpus,
                            server_pid=process.pid,
                            error=message,
                            foreign=foreign,
                        )
                        raise ValidationError(message)
                    pmon_enabled = True
                ready = True
                require_pmon = foreign_process_query is None
                ready_at = datetime.now(UTC)
                startup_seconds = time.monotonic() - started_monotonic
                if lifecycle_path is not None:
                    _write_lifecycle(
                        lifecycle_path,
                        started_at=started_at,
                        ready_at=ready_at,
                        startup_seconds=startup_seconds,
                        selected_gpus=plan.selected_gpus,
                        server_pid=process.pid,
                        endpoint=endpoint,
                        runtime_settings=plan.runtime_settings,
                    )
                event = json.dumps(
                    {
                        "event": "server-ready",
                        "endpoint": endpoint,
                        "server_pid": process.pid,
                        "startup_seconds": startup_seconds,
                        "runtime_settings": plan.runtime_settings,
                    },
                    sort_keys=True,
                )
                sys.stdout.write(event + "\n")
                sys.stdout.flush()
            if not ready and time.monotonic() - started_monotonic >= startup_timeout:
                message = (
                    f"guarded server startup timed out after {startup_timeout:.1f}s"
                )
                _write_failure(
                    report_path,
                    started_at=started_at,
                    selected_gpus=plan.selected_gpus,
                    server_pid=process.pid,
                    error=message,
                )
                raise ValidationError(message)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        return
    finally:
        _stop_process(process)
