"""Worker lifetime and cleanup against real Windows process jobs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows process jobs")

# sys._base_executable avoids the venv redirector, whose child has a different pid.


@pytest.mark.parametrize("opened", [False, True])
def test_failed_job_queries_report_unknown(monkeypatch, caplog, opened) -> None:
    import ctypes
    from unittest.mock import Mock

    from nscr_houdini_mcp import pool

    kernel = Mock()
    kernel.OpenProcess.return_value = 123 if opened else 0
    kernel.IsProcessInJob.return_value = 0
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: pool.ERROR_ACCESS_DENIED)
    assert pool._windows_in_job(42) is None
    assert "error 5" in caplog.text
    assert "server-bound" in caplog.text
    if opened:
        kernel.CloseHandle.assert_called_once_with(123)
    else:
        kernel.CloseHandle.assert_not_called()


@pytest.mark.parametrize("membership", [(None, False), (False, None), (None, None)])
def test_unknown_job_membership_keeps_the_worker_server_bound(tmp_path, monkeypatch, membership):
    from nscr_houdini_mcp import pool

    answers = iter(membership)
    monkeypatch.setattr(pool, "_windows_in_job", lambda pid: next(answers))
    worker = pool.spawn_detached(
        [sys._base_executable, "-c", "import time; time.sleep(60)"], log=tmp_path / "worker.log"
    )
    try:
        assert worker.server_bound
    finally:
        worker.kill()
        worker.wait(10)
        pool.reap_started()


@pytest.mark.parametrize("breakaway", [False, True])
def test_a_worker_breaks_away_or_stays_with_its_launchers_job(
    tmp_path: Path, breakaway: bool
) -> None:
    win32api = pytest.importorskip("win32api")
    win32con = pytest.importorskip("win32con")
    win32event = pytest.importorskip("win32event")
    win32job = pytest.importorskip("win32job")
    win32process = pytest.importorskip("win32process")

    script = tmp_path / "launcher.py"
    script.write_text(
        "import json, logging, os, sys\n"
        "from pathlib import Path\n"
        "from nscr_houdini_mcp import pool\n"
        "logging.basicConfig(level=logging.INFO)\n"
        "print(os.getpid(), flush=True)\n"
        "sys.stdin.readline()\n"
        "worker = pool.spawn_detached(\n"
        "    [sys._base_executable, '-c', "
        "'import os, time; print(os.environ[\"WORKER_VALUE\"], flush=True); time.sleep(120)'],\n"
        "    log=Path(sys.argv[1]), env=dict(os.environ, WORKER_VALUE='spaces and café'))\n"
        "print(json.dumps({'pid': worker.pid, 'server_bound': worker.server_bound}), flush=True)\n"
        "sys.stdin.readline()\n",
        encoding="utf-8",
    )
    job = win32job.CreateJobObject(None, "")
    limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    flags = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if breakaway:
        flags |= win32job.JOB_OBJECT_LIMIT_BREAKAWAY_OK
    limits["BasicLimitInformation"]["LimitFlags"] = flags
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
    log = tmp_path / "worker output.log"
    launcher = subprocess.Popen(
        [sys.executable, str(script), str(log)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    worker = None
    try:
        pid = int(launcher.stdout.readline())
        handle = win32api.OpenProcess(
            win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, pid
        )
        try:
            win32job.AssignProcessToJobObject(job, handle)
        finally:
            handle.Close()
        launcher.stdin.write("start\n")
        launcher.stdin.flush()
        line = launcher.stdout.readline()
        assert line, launcher.stderr.read()
        report = json.loads(line)
        worker = win32api.OpenProcess(
            win32con.SYNCHRONIZE | win32con.PROCESS_TERMINATE | win32con.PROCESS_QUERY_INFORMATION,
            False,
            report["pid"],
        )
        assert report["server_bound"] is not breakaway
        assert win32event.WaitForSingleObject(worker, 1000) == win32con.WAIT_TIMEOUT
        assert win32job.IsProcessInJob(worker, None) is not breakaway
        assert "spaces and café" in log.read_text(encoding="utf-8")
        _, messages = launcher.communicate("stop\n", timeout=30)
        assert launcher.returncode == 0
        assert messages.count("ends when this server ends") == int(not breakaway)
        assert ("WARNING:nscr_houdini_mcp.pool:" in messages) is not breakaway
        job.Close()
        expected = win32con.WAIT_TIMEOUT if breakaway else win32con.WAIT_OBJECT_0
        assert win32event.WaitForSingleObject(worker, 1000) == expected
    finally:
        try:
            if launcher.poll() is None:
                launcher.kill()
            launcher.wait(timeout=30)
        finally:
            job.Close()
            if worker is not None:
                try:
                    if win32event.WaitForSingleObject(worker, 0) == win32con.WAIT_TIMEOUT:
                        win32process.TerminateProcess(worker, 1)
                    assert win32event.WaitForSingleObject(worker, 10000) == win32con.WAIT_OBJECT_0
                finally:
                    worker.Close()
