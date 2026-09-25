"""`hou_python` through the real server, against a real worker.

A client starts the server over stdio the way any client does, and the server
starts a hython worker through the pool and runs code in it. Nothing is stood
in for. A reply is lost the way it is lost in use: the client gives up and its
server goes away while the code still runs, and a second server, with a
default namespace of its own, sends the same call again.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker at a time (the pool cap in
this file's own config is one), a state folder of this file's own, the pool's
port range, and every worker stopped again whatever happened, with a check
that nothing is left. Every scene file read or written is inside the test's
own temporary folder.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

import support
from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge import client, registry


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

# A worker start is a cold Houdini, so the client waits longer than it would
# for an ordinary call.
READ_TIMEOUT_S = 300.0

# How long the caller waits for a held back answer before it counts it lost.
GIVE_UP_S = 2.0


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A state folder with a config of its own and one worker, stopped at the end."""
    root = tmp_path_factory.mktemp("python")
    home = root / "home"
    home.mkdir()
    scratch = root / "houdini-temp"
    scratch.mkdir()
    scenes = root / "scenes"
    scenes.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    made: dict[str, Any] = {"home": home, "scratch": scratch, "scenes": scenes}
    try:
        [started] = run(made, ("hou_sessions", {"action": "start"}))
        made["worker"] = ok(started)["session"]
        if made["worker"].get("lifetime") == "server":
            pytest.skip(
                "Windows refused worker breakaway; this test needs it to outlive its server"
            )
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
            # Carried on to the worker, so it will hold an answer back when
            # a call asks it to.
            "NSCR_MCP_SELFCHECK": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )


async def _run(place: dict[str, Any], calls: list[tuple[str, dict]]) -> list[Any]:
    async with Client(
        server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        results = [await connected.call_tool(name, arguments) for name, arguments in calls]
        await support.stop_server_bound_workers(connected, results)
        return results


def run(place: dict[str, Any], *calls: tuple[str, dict]) -> list[Any]:
    """Every call through one server process, so its default namespace holds."""
    return asyncio.run(_run(place, list(calls)))


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def refused(result: Any) -> dict[str, Any]:
    assert result.is_error is True
    return result.structured_content["error"]


def children(place: dict[str, Any]) -> int:
    [counted] = run(place, ("hou_python", {"code": "result = len(hou.node('/obj').children())"}))
    return ok(counted)["result"]


def test_result_namespaces_and_reset(place: dict[str, Any]) -> None:
    counted, first, second, cleared, own = run(
        place,
        ("hou_python", {"code": "result = len(hou.node('/obj').children())"}),
        ("hou_python", {"code": "kept = 41"}),
        ("hou_python", {"code": "result = kept + 1"}),
        ("hou_python", {"code": "result = 'kept' in globals()", "reset": True}),
        ("hou_python", {"code": "result = sorted(k for k in globals() if k != '__builtins__')"}),
    )
    assert ok(counted)["result"] == 0
    assert ok(counted)["namespace"].startswith("c_")
    assert ok(first)["result"] is None
    assert ok(second)["result"] == 42
    assert ok(cleared)["result"] is False
    assert ok(own)["result"] == ["hou", "mcp"]
    assert ok(counted)["scene_epoch"] == ok(own)["scene_epoch"]


def test_an_exception_comes_back_with_its_traceback_tail(place: dict[str, Any]) -> None:
    code = "def inner():\n    return hou.node('/obj').children()[99]\ninner()\n"
    [result] = run(place, ("hou_python", {"code": code}))
    assert result.is_error is True
    error = result.structured_content["error"]
    assert error["type"] == "IndexError"
    assert "line 2, in inner" in error["traceback_tail"]
    assert "children()[99]" in error["traceback_tail"]
    assert len(error["traceback_tail"].splitlines()) <= 20
    assert "IndexError" in result.content[0].text


def test_printing_past_max_chars_spills_the_whole(place: dict[str, Any]) -> None:
    code = "for n in range(2000):\n    print('row', n)"
    [result] = run(place, ("hou_python", {"code": code, "max_chars": 500}))
    body = ok(result)
    assert len(body["stdout_tail"]) == 500
    assert body["stdout_tail"].endswith("row 1999\n")
    printed = "".join(f"row {n}\n" for n in range(2000))
    assert body["elided_chars"] == len(printed) - 500
    spill = Path(body["spill_path"])
    assert spill.is_relative_to(place["home"])
    assert json.loads(spill.read_text(encoding="utf-8"))["stdout"] == printed


def test_a_lost_reply_sent_again_by_another_server_makes_one_node_not_two(
    place: dict[str, Any],
) -> None:
    before = children(place)
    operation_id = client.new_operation_id()
    code = "import time\ntime.sleep(4)\nresult = hou.node('/obj').createNode('geo').path()"
    call = {"code": code, "operation_id": operation_id}
    # The first server sends the call, its client gives up, and the server
    # goes away with it. The code is still running in the worker.
    assert asyncio.run(_abandon(place, ("hou_python", call), give_up_s=GIVE_UP_S)) is None
    # A second server has a default namespace of its own; the call names none.
    # Sent while the code still runs, it is told so at once, with the job.
    [retried] = run(place, ("hou_python", {**call, "wait_s": 30}))
    body = ok(retried)
    assert body["trace"]["operation_id"] == operation_id
    if "result" not in body:
        assert body["job_id"] == f"job-{operation_id}"
        followed(place, body["job_id"])
        [retried] = run(place, ("hou_python", call))
        body = ok(retried)
    [own] = run(place, ("hou_python", {"code": "result = 1"}))
    assert body["result"].startswith("/obj/geo")
    assert body["namespace"].startswith("c_")
    assert body["namespace"] != ok(own)["namespace"]
    assert children(place) == before + 1


def followed(place: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Wait on a job with held statuses until it ends."""
    body: dict[str, Any] = {}
    for _ in range(6):
        [held] = run(place, ("hou_jobs", {"job_id": job_id, "wait_s": 10}))
        body = ok(held)
        if body["state"] not in ("queued", "running"):
            break
    return body


async def _abandon(place: dict[str, Any], call: tuple[str, dict], *, give_up_s: float) -> Any:
    """Send one call and give up on it, taking the server down on the way out."""
    async with Client(
        server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        try:
            return await asyncio.wait_for(connected.call_tool(*call), give_up_s)
        except TimeoutError:
            return None


def test_an_epoch_from_before_an_open_is_refused_and_outputs_follow_the_scene(
    place: dict[str, Any],
) -> None:
    hip = place["scenes"] / "python_scene.hip"
    saved, info = run(
        place,
        ("hou_python", {"code": f"hou.hipFile.save({str(hip)!r})"}),
        ("hou_scene", {}),
    )
    before = ok(saved)["scene_epoch"]
    assert Path(ok(info)["hip_path"]).resolve() == hip.resolve()
    opened, stale, current = run(
        place,
        ("hou_scene", {"action": "open", "path": str(hip)}),
        ("hou_python", {"code": "result = 1", "scene_epoch": before}),
        ("hou_python", {"code": "result = mcp.output_path('cache', 'test')"}),
    )
    assert ok(opened)["scene_epoch"] == before + 1
    error = refused(stale)
    assert error["code"] == "SCENE_REPLACED"
    path = Path(ok(current)["result"])
    assert path.name == "test_v001.$F4.bgeo.sc"
    folder = (place["scenes"] / "geo" / "test" / "v001").resolve()
    assert path.parent.resolve() == folder
    assert folder.is_dir()


def test_code_past_its_timeout_answers_and_a_retry_follows_its_job(
    place: dict[str, Any],
) -> None:
    code = "import time\ntime.sleep(5)\nruns = globals().get('runs', 0) + 1\nresult = runs"
    call = {"code": code, "namespace": "slow", "operation_id": client.new_operation_id()}
    operation_id = call["operation_id"]
    slow, again = run(
        place,
        ("hou_python", {**call, "timeout_s": 1, "background": False}),
        # Sent while the code still runs: it is told so at once, with the job
        # to follow, rather than queueing behind itself.
        ("hou_python", {**call, "wait_s": 20}),
    )
    error = refused(slow)
    assert error["code"] == "TIMEOUT"
    assert error["details"]["still_running"] is True
    assert error["details"]["operation_id"] == operation_id
    assert ok(again)["job_id"] == error["details"]["job_id"]
    assert ok(again)["state"] == "running"
    ended = followed(place, error["details"]["job_id"])
    assert ended["state"] == "done"
    assert ended["outputs"]["result"] == 1
    fetched, counted, after = run(
        place,
        ("hou_python", call),
        ("hou_python", {"code": "result = runs", "namespace": "slow"}),
        ("hou_ping", {"wait_s": 0}),
    )
    assert ok(fetched)["result"] == 1
    assert ok(counted)["result"] == 1
    assert ok(after)["call"]["ok"] is True
    assert ok(after)["health"]["busy"] is False


def test_surrogates_and_code_that_will_not_compile_come_back_as_data(
    place: dict[str, Any],
) -> None:
    printed, deep = run(
        place,
        ("hou_python", {"code": "print('a\\udcff')\nresult = ['b\\ud800']"}),
        ("hou_python", {"code": "x = " + "-" * 100_000 + "1"}),
    )
    body = ok(printed)
    assert body["stdout_tail"] == "a\\udcff\n"
    assert body["result"] == ["b\\ud800"]
    assert body["lossy"] is True
    assert deep.is_error is True
    assert deep.structured_content["error"]["type"] in ("RecursionError", "MemoryError")


def test_a_log_handler_from_one_call_writes_into_the_next(place: dict[str, Any]) -> None:
    make = (
        "import logging, io, sys\n"
        "log = logging.getLogger('nscr_python_check')\n"
        "log.propagate = False\n"
        "log.addHandler(logging.StreamHandler())\n"
        "log.warning('one')\n"
        "sys.stdout = io.StringIO()"
    )
    first, second = run(
        place,
        ("hou_python", {"code": make, "namespace": "logs"}),
        ("hou_python", {"code": "log.warning('two')\nprint('three')", "namespace": "logs"}),
    )
    assert ok(first)["stdout_tail"] == "one\n"
    assert ok(second)["stdout_tail"] == "two\nthree\n"
