"""Jobs through the real server, against a real worker.

A client starts the server over stdio, and the server starts a hython worker
through the pool. Each `run` below is a server process of its own, so every
check that follows a job across two of them is a client restart as it happens
in use: the job is found again by its id in the store, and nothing else.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker at a time (the pool cap
in this file's own config is one), a state folder of this file's own, the
pool's port range, and every worker stopped again whatever happened. The last
check ends its worker the way a crash does, which is why it is last.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

import support
from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge import registry


def hython_available() -> bool:
    try:
        pool.hython_path()
    except pool.HythonNotFound:
        return False
    return True


pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
]

PORT_RANGE = support.POOL_PORTS

SERVER_CODE = "from nscr_houdini_mcp.cli import main; raise SystemExit(main([]))"

READ_TIMEOUT_S = 300.0

# Code that runs until it is told to stop, and a little longer than any check
# waits if it never is.
LOOP = (
    "import time\n"
    "laps = 0\n"
    "while not mcp.cancelled() and laps < 3000:\n"
    "    laps += 1\n"
    "    if laps % 10 == 0:\n"
    "        mcp.progress(laps, 3000, 'looping')\n"
    "    time.sleep(0.02)\n"
    "result = laps\n"
)


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A state folder with a config of its own and one worker, stopped at the end."""
    root = tmp_path_factory.mktemp("jobs")
    home = root / "home"
    home.mkdir()
    scratch = root / "houdini-temp"
    scratch.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\ninline_wait_s = 1\n",
        encoding="utf-8",
    )
    made: dict[str, Any] = {"home": home, "scratch": scratch}
    try:
        [started] = run(made, ("hou_sessions", {"action": "start"}))
        made["worker"] = ok(started)["session"]
        yield made
    finally:
        left = support.stop_everything(home, pool.PoolConfig(home=home))
        assert left == [], f"workers were left running: {left}"
        with pool.open_store(home) as store:
            for worker in store.list_workers(active_only=False):
                assert worker.pid is None or not pool.worker_is_alive(worker), worker.alias
        assert registry.live_entries(home) == []


def server_params(place: dict[str, Any]) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", SERVER_CODE],
        env={
            "NSCR_MCP_HOME": str(place["home"]),
            "HOUDINI_TEMP_DIR": str(place["scratch"]),
            "PYTHONIOENCODING": "utf-8",
        },
    )


async def _run(place: dict[str, Any], calls: list[tuple[str, dict]]) -> list[Any]:
    async with Client(
        server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        return [await connected.call_tool(name, arguments) for name, arguments in calls]


def run(place: dict[str, Any], *calls: tuple[str, dict]) -> list[Any]:
    """Every call through one server process of its own."""
    return asyncio.run(_run(place, list(calls)))


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def settled(place: dict[str, Any], job_id: str, *, rounds: int = 6) -> dict[str, Any]:
    """Wait on a job until it ends, one held status after another."""
    body: dict[str, Any] = {}
    for _ in range(rounds):
        [held] = run(place, ("hou_jobs", {"job_id": job_id, "wait_s": 10}))
        body = ok(held)
        if body["state"] not in ("queued", "running"):
            return body
    return body


def test_slow_code_becomes_a_job_that_a_held_status_sees_end(place: dict[str, Any]) -> None:
    code = "import time\ntime.sleep(5)\nresult = {'slept': 5}"
    [started] = run(place, ("hou_python", {"code": code}))
    handle = ok(started)
    assert handle["state"] in ("queued", "running")
    assert handle["kind"] == "python"
    assert "result" not in handle
    job_id = handle["job_id"]
    # Another server process, as a client restart makes.
    [held] = run(place, ("hou_jobs", {"job_id": job_id, "wait_s": 10}))
    body = ok(held)
    if body["state"] == "running":
        body = settled(place, job_id)
    assert body["state"] == "done"
    assert body["outputs"]["result"] == {"slept": 5}
    assert body["operation_id"] == handle["operation_id"]
    assert body["elapsed_s"] >= 4.5
    # The worker's scene has no file, so the readable copy is in its scratch folder.
    copy = Path(body["export_path"])
    assert copy.is_file()
    assert copy.is_relative_to(place["scratch"])
    [listed] = run(place, ("hou_jobs", {"action": "list", "limit": 5}))
    assert job_id in [row["job_id"] for row in ok(listed)["jobs"]]


def test_cancel_stops_a_loop_that_looks_at_the_flag(place: dict[str, Any]) -> None:
    [started] = run(place, ("hou_python", {"code": LOOP, "background": True}))
    job_id = ok(started)["job_id"]
    [cancelled] = run(place, ("hou_jobs", {"job_id": job_id, "action": "cancel"}))
    asked = ok(cancelled)
    assert asked["cancel"]["requested"] is True
    assert asked["cancel"]["reached_session"] is True
    body = settled(place, job_id)
    assert body["state"] == "cancelled"
    assert body["cancel_requested"] is True
    assert body["outputs"]["result"] < 3000
    [after] = run(place, ("hou_ping", {"wait_s": 0}))
    assert ok(after)["health"]["busy"] is False


def test_a_second_server_follows_a_job_by_id(place: dict[str, Any]) -> None:
    code = "import time\ntime.sleep(3)\nresult = 'followed'"
    [started] = run(place, ("hou_python", {"code": code, "background": True}))
    job_id = ok(started)["job_id"]
    body = settled(place, job_id)
    assert body["state"] == "done"
    assert body["outputs"]["result"] == "followed"


def test_a_worker_reports_unsaved_changes_from_the_bridges_own_mark(
    place: dict[str, Any], tmp_path: Path
) -> None:
    hip = tmp_path / "marked.hip"
    changed, after_change, saved_by_code, after_code, saved, after_save = run(
        place,
        ("hou_python", {"code": "hou.node('/obj').createNode('null')"}),
        ("hou_scene", {}),
        ("hou_python", {"code": f"hou.hipFile.save({str(hip)!r})"}),
        ("hou_scene", {}),
        ("hou_scene", {"action": "save"}),
        ("hou_scene", {}),
    )
    ok(changed)
    assert (ok(after_change)["unsaved"], ok(after_change)["unsaved_source"]) == (True, "bridge")
    ok(saved_by_code)
    # The code saved and could have changed things after, so nobody knows.
    assert ok(after_code)["unsaved"] is None
    ok(saved)
    assert (ok(after_save)["unsaved"], ok(after_save)["unsaved_source"]) == (False, "bridge")
    assert hip.is_file()


def test_a_killed_worker_leaves_its_job_lost(place: dict[str, Any]) -> None:
    [started, info] = run(
        place,
        ("hou_python", {"code": LOOP, "background": True}),
        ("hou_sessions", {"action": "info", "session": place["worker"]["session_id"]}),
    )
    job_id = ok(started)["job_id"]
    pid = ok(info)["session"]["pid"]
    assert isinstance(pid, int) and pid != os.getpid()
    with pool.open_store(place["home"]) as store:
        [worker] = [w for w in store.list_workers() if w.pid == pid]
        assert pool.worker_is_alive(worker)
    os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    support.wait_until(lambda: not pool.worker_is_alive(worker), timeout_s=30.0)
    [status] = run(place, ("hou_jobs", {"job_id": job_id}))
    body = ok(status)
    assert body["state"] == "lost"
    assert body["error"]["code"] == "SESSION_ENDED"
    assert body["ended_at"] is not None
