"""Tests for continuous selected-GPU process guarding."""

import json
import signal
import subprocess
from pathlib import Path
from typing import Self

import pytest

from h3fast.benchmarks.guard import (
    ForeignGpuProcess,
    _health_ready,
    _signal_and_wait,
    find_foreign_gpu_processes,
    serve_guarded,
)
from h3fast.benchmarks.launch import LaunchPlan
from h3fast.benchmarks.preflight import NvidiaDevice
from h3fast.exceptions import ValidationError


def _status(root: Path, pid: int, parent: int) -> None:
    path = root / str(pid)
    path.mkdir()
    (path / "status").write_text(
        f"Name:\tprocess\nPid:\t{pid}\nPPid:\t{parent}\n", encoding="utf-8"
    )


def _device() -> NvidiaDevice:
    return NvidiaDevice(1, "GPU", 49140, 48000, "555.58.02", "8.9", "gpu-1")


def _plan() -> LaunchPlan:
    return LaunchPlan(
        argv=("server",),
        selected_gpus=(1, 2),
        sglang_revision="revision",
        base_image="image",
        ffprobe_adapter_sha256="a" * 64,
    )


def test_find_foreign_gpu_processes_allows_only_descendants(
    tmp_path: Path, monkeypatch
) -> None:
    _status(tmp_path, 100, 1)
    _status(tmp_path, 200, 100)
    _status(tmp_path, 300, 1)
    applications = {
        "gpu-1": [
            {"pid": 200, "process_name": "server-worker", "used_memory_mib": 100},
            {"pid": 300, "process_name": "foreign", "used_memory_mib": 200},
        ]
    }
    monkeypatch.setattr(
        "h3fast.benchmarks.guard._query_nvidia",
        lambda: ((_device(),), applications),
    )

    result = find_foreign_gpu_processes((1,), allowed_root_pid=100, proc_root=tmp_path)

    assert len(result) == 1
    assert result[0].pid == 300
    assert result[0].to_dict()["gpu_index"] == 1


def test_find_foreign_gpu_processes_fails_for_missing_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3fast.benchmarks.guard._query_nvidia", lambda: ((_device(),), {})
    )

    with pytest.raises(ValidationError, match="disappeared"):
        find_foreign_gpu_processes((2,), allowed_root_pid=100)


def test_find_foreign_gpu_processes_rejects_invalid_application(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3fast.benchmarks.guard._query_nvidia",
        lambda: ((_device(),), {"gpu-1": [{"pid": "invalid"}]}),
    )

    with pytest.raises(ValidationError, match="invalid compute process"):
        find_foreign_gpu_processes((1,), allowed_root_pid=100)


class _Process:
    pid = 100

    def __init__(self, return_code=None) -> None:
        self.return_code = return_code

    def poll(self):
        return self.return_code

    def wait(self, timeout: float):
        self.return_code = 0
        return 0


def test_health_ready_handles_success_and_failure(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args) -> bool:
            return False

    monkeypatch.setattr(
        "h3fast.benchmarks.guard.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    assert _health_ready("http://127.0.0.1:30010", 1.0) is True

    def fail(*_args, **_kwargs):
        message = "not ready"
        raise OSError(message)

    monkeypatch.setattr("h3fast.benchmarks.guard.urllib.request.urlopen", fail)
    assert _health_ready("http://127.0.0.1:30010", 1.0) is False


def test_signal_and_wait_reports_success_and_timeout(monkeypatch) -> None:
    process = _Process()
    monkeypatch.setattr("h3fast.benchmarks.guard.os.killpg", lambda *_args: None)
    assert _signal_and_wait(process, signal.SIGINT, 1.0) is True

    def timeout(*, timeout: float):
        command = "server"
        raise subprocess.TimeoutExpired(command, timeout)

    process.return_code = None
    monkeypatch.setattr(process, "wait", timeout)
    assert _signal_and_wait(process, signal.SIGTERM, 1.0) is False


def test_signal_and_wait_finishes_cleanup_after_interrupt(monkeypatch) -> None:
    process = _Process()
    waits = iter((KeyboardInterrupt(), 0))
    monkeypatch.setattr("h3fast.benchmarks.guard.os.killpg", lambda *_args: None)

    def wait(**_kwargs):
        result = next(waits)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(process, "wait", wait)

    assert _signal_and_wait(process, signal.SIGINT, 1.0) is True


def test_serve_guarded_records_foreign_process_and_stops_server(
    tmp_path: Path, monkeypatch
) -> None:
    process = _Process()
    stopped: list[object] = []
    foreign = ForeignGpuProcess(1, "gpu-1", 300, "foreign", 200)
    monkeypatch.setattr(
        "h3fast.benchmarks.guard.subprocess.Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr("h3fast.benchmarks.guard._stop_process", stopped.append)
    report = tmp_path / "guard-failure.json"

    with pytest.raises(ValidationError, match="foreign GPU"):
        serve_guarded(
            _plan(),
            endpoint="http://127.0.0.1:30010",
            report_path=report,
            foreign_process_query=lambda _gpus, _root: (foreign,),
        )

    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["foreign_processes"][0]["pid"] == 300
    assert stopped == [process]


def test_serve_guarded_reports_ready_and_handles_interrupt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    process = _Process()
    stopped: list[object] = []
    monkeypatch.setattr(
        "h3fast.benchmarks.guard.subprocess.Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr("h3fast.benchmarks.guard._health_ready", lambda *_args: True)
    monkeypatch.setattr("h3fast.benchmarks.guard._stop_process", stopped.append)

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("h3fast.benchmarks.guard.time.sleep", interrupt)

    report = tmp_path / "unused.json"
    report.write_text("stale", encoding="utf-8")
    serve_guarded(
        _plan(),
        endpoint="http://127.0.0.1:30010",
        report_path=report,
        foreign_process_query=lambda _gpus, _root: (),
    )

    assert json.loads(capsys.readouterr().out)["event"] == "server-ready"
    assert stopped == [process]
    assert not report.exists()


def test_serve_guarded_records_unexpected_server_exit(
    tmp_path: Path, monkeypatch
) -> None:
    process = _Process(return_code=17)
    monkeypatch.setattr(
        "h3fast.benchmarks.guard.subprocess.Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr("h3fast.benchmarks.guard._stop_process", lambda _value: None)
    report = tmp_path / "server-failure.json"

    with pytest.raises(ValidationError, match="status 17"):
        serve_guarded(
            _plan(),
            endpoint="http://127.0.0.1:30010",
            report_path=report,
        )

    assert json.loads(report.read_text(encoding="utf-8"))["server_pid"] == 100


def test_serve_guarded_rejects_external_health_endpoint(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        serve_guarded(
            _plan(),
            endpoint="http://example.com:30010",
            report_path=tmp_path / "unused.json",
        )


def test_serve_guarded_records_gpu_query_failure(tmp_path: Path, monkeypatch) -> None:
    process = _Process()
    monkeypatch.setattr(
        "h3fast.benchmarks.guard.subprocess.Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr("h3fast.benchmarks.guard._stop_process", lambda _value: None)
    report = tmp_path / "guard-failure.json"

    def fail(_gpus: tuple[int, ...], _root: int):
        message = "nvidia-smi unavailable"
        raise ValidationError(message)

    with pytest.raises(ValidationError, match="guard query failed"):
        serve_guarded(
            _plan(),
            endpoint="http://127.0.0.1:30010",
            report_path=report,
            foreign_process_query=fail,
        )

    assert (
        "nvidia-smi unavailable"
        in json.loads(report.read_text(encoding="utf-8"))["error"]
    )


def test_serve_guarded_records_startup_timeout(tmp_path: Path, monkeypatch) -> None:
    process = _Process()
    monotonic = iter((0.0, 2.0))
    monkeypatch.setattr(
        "h3fast.benchmarks.guard.subprocess.Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr("h3fast.benchmarks.guard._health_ready", lambda *_args: False)
    monkeypatch.setattr("h3fast.benchmarks.guard._stop_process", lambda _value: None)
    monkeypatch.setattr(
        "h3fast.benchmarks.guard.time.monotonic", lambda: next(monotonic)
    )
    report = tmp_path / "timeout.json"

    with pytest.raises(ValidationError, match="startup timed out"):
        serve_guarded(
            _plan(),
            endpoint="http://127.0.0.1:30010",
            report_path=report,
            startup_timeout=1.0,
            foreign_process_query=lambda _gpus, _root: (),
        )

    assert "timed out" in json.loads(report.read_text(encoding="utf-8"))["error"]


def test_serve_guarded_rejects_bad_timing_and_launch_failure(
    tmp_path: Path, monkeypatch
) -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        serve_guarded(
            _plan(),
            endpoint="http://127.0.0.1:30010",
            report_path=tmp_path / "unused.json",
            poll_interval=0,
        )

    def fail_launch(*_args, **_kwargs):
        message = "permission denied"
        raise OSError(message)

    monkeypatch.setattr("h3fast.benchmarks.guard.subprocess.Popen", fail_launch)
    with pytest.raises(ValidationError, match="could not launch"):
        serve_guarded(
            _plan(),
            endpoint="http://127.0.0.1:30010",
            report_path=tmp_path / "unused.json",
        )
