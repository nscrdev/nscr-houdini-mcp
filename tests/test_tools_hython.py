"""`hou_sessions` and `hou_scene` through the real server, against a real worker.

A client starts the server over stdio the way any client does, and the server
starts, lists, reads, saves, opens and stops a hython worker through the pool.
Nothing is stood in for.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker at a time (the pool cap in
this file's own config is one), a state folder of this file's own, the pool's
port range, and every worker stopped again whatever happened, with a check
that nothing is left. Every scene file read or written is inside the test's
own temporary folder.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import threading
import time
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

# How long the self check keeps the worker busy while the list is taken.
BUSY_S = 8.0


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A state folder with a config of its own, cleared of workers at the end."""
    root = tmp_path_factory.mktemp("tools")
    home = root / "home"
    home.mkdir()
    scratch = root / "houdini-temp"
    scratch.mkdir()
    scenes = root / "scenes"
    scenes.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    made = {"home": home, "scratch": scratch, "scenes": scenes}
    with support.persistent_client(server_params(made), timeout_s=READ_TIMEOUT_S) as send:
        made["send"] = send
        try:
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
            # Carried on to the workers the server starts, so the check can
            # keep one busy on purpose.
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
    if place.get("send") is not None:
        return place["send"](*calls)
    return asyncio.run(_run(place, list(calls)))


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def code(result: Any) -> str:
    assert result.is_error is True
    return result.structured_content["error"]["code"]


def rows(result: Any) -> dict[str, dict[str, Any]]:
    return {row["session_id"]: row for row in ok(result)["sessions"]}


def selfcheck(home: Path, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Drive the worker's self check directly, the way the bridge checks do."""
    session = client.Session.open(home, session_id)
    answer = client.call(
        session,
        "bridge.selfcheck",
        arguments=arguments,
        operation_id=client.new_operation_id(),
        wait_s=30.0,
        timeout_s=120.0,
    )
    assert answer.payload["ok"] is True, answer.payload
    return answer.payload


def test_sessions_and_scene_files_through_a_real_worker(place: dict[str, Any]) -> None:
    home, scratch, scenes = place["home"], place["scratch"], place["scenes"]

    # Start one worker; the pool has room for no second one.
    empty, started, full = run(
        place,
        ("hou_sessions", {}),
        ("hou_sessions", {"action": "start"}),
        ("hou_sessions", {"action": "start"}),
    )
    assert ok(empty)["sessions"] == []
    worker = ok(started)["session"]
    session_id = worker["session_id"]
    assert worker["kind"] == "hython"
    assert worker["state"] == "live"
    assert ok(started)["trace"]["operation_id"]
    assert code(full) == "POOL_FULL"
    assert full.content[0].text.startswith("POOL_FULL: ")

    # It lists as live and answers for itself.
    listing, info, scene_info, untitled_save = run(
        place,
        ("hou_sessions", {}),
        ("hou_sessions", {"action": "info", "session": worker["alias"]}),
        ("hou_scene", {}),
        ("hou_scene", {"action": "save"}),
    )
    row = rows(listing)[session_id]
    assert row["state"] == "live"
    assert row["lease_age_s"] >= 0
    assert "22." in row["capabilities"]
    assert ok(info)["session"]["health"]["busy"] is False
    assert ok(info)["session"]["state"] == "live"
    assert ok(scene_info)["untitled"] is True
    assert code(untitled_save) == "SCENE_UNTITLED"
    assert "save_increment" in untitled_save.content[0].text

    # A busy cook shows as busy in the list, and the list still answers.
    busy: dict[str, Any] = {}
    holder = threading.Thread(
        target=lambda: busy.update(selfcheck(home, session_id, {"sleep_s": BUSY_S})),
        daemon=True,
    )
    holder.start()
    support.wait_until(lambda: _busy(home, session_id), timeout_s=30.0)
    [during] = run(place, ("hou_sessions", {}))
    holder.join(BUSY_S + 60.0)
    assert rows(during)[session_id]["state"] == "busy"
    assert rows(during)[session_id]["current_op"] == "bridge.selfcheck"
    assert busy["data"]["slept_s"] >= BUSY_S - 0.5

    # An untitled scene saves to scratch, twice, as v001 then v002.
    selfcheck(home, session_id, {"creates": 1})
    first, second = run(
        place,
        ("hou_scene", {"action": "save_increment"}),
        ("hou_scene", {"action": "save_increment"}),
    )
    one, two = ok(first), ok(second)
    folder = scratch / "nscr-houdini-mcp" / session_id
    assert Path(one["hip_path"]) == folder / "untitled_v001.hip"
    assert Path(two["hip_path"]) == folder / "untitled_v002.hip"
    assert one["version"] == 1 and two["version"] == 2
    assert one["unsaved_hip"] is True
    kept = Path(one["hip_path"]).read_bytes()
    assert kept and Path(two["hip_path"]).is_file()
    # Written under a private name and published by a link: nothing else is left.
    assert list(folder.glob("*.part*")) == []
    assert list(folder.glob("*.claim")) == []

    # Never over a file that is there: one already sits where v003 would go.
    decoy = folder / "untitled_v003.hip"
    decoy.write_bytes(b"not a scene, and not to be written over")
    third, in_place = run(
        place,
        ("hou_scene", {"action": "save_increment"}),
        ("hou_scene", {"action": "save"}),
    )
    assert Path(ok(third)["hip_path"]) == folder / "untitled_v004.hip"
    assert decoy.read_bytes() == b"not a scene, and not to be written over"
    assert Path(one["hip_path"]).read_bytes() == kept
    assert ok(in_place)["hip_path"] == ok(third)["hip_path"]

    # A scene with a node type this build does not have opens, and says so.
    broken = scenes / "broken.hip"
    data = Path(one["hip_path"]).read_bytes()
    assert b"type = geo\n" in data
    broken.write_bytes(data.replace(b"type = geo\n", b"type = gez\n"))
    before = ok(in_place)["trace"]["scene_epoch"]
    opened, reopened, missing, after = run(
        place,
        ("hou_scene", {"action": "open", "path": str(broken)}),
        ("hou_scene", {"action": "open", "path": one["hip_path"]}),
        ("hou_scene", {"action": "open", "path": str(scenes / "not_there.hip")}),
        ("hou_scene", {"detail": "full"}),
    )
    report = ok(opened)["dependencies"]
    assert {"type": "gez", "parent": "/obj"} in report["unresolved_types"]
    assert any("gez" in line for line in report["load_warnings"])
    assert ok(opened)["scene_epoch"] == before + 1
    assert ok(reopened)["hip_path"] == one["hip_path"]
    assert ok(reopened)["dependencies"]["unresolved_types"] == []
    assert ok(reopened)["scene_epoch"] == before + 2
    assert code(missing) == "FILE_NOT_FOUND"
    assert ok(after)["hip_path"] == one["hip_path"]
    assert ok(after)["version"] == 1
    assert ok(after)["dependencies"]["missing_files"] == []

    # Stop ends the worker, and the list says it has gone.
    stopped, listed = run(
        place,
        ("hou_sessions", {"action": "stop", "session": session_id}),
        ("hou_sessions", {}),
    )
    assert ok(stopped)["stopped"]["ended"] is True
    assert rows(listed)[session_id]["state"] == "gone"
    with pool.open_store(home) as store:
        workers = store.list_workers(active_only=False)
    record = next(w for w in workers if w.session_id == session_id)
    assert not pool.worker_is_alive(record)
    assert record.state == "stopped"

    # A worker that is killed shows as crashed on the next list.
    [again] = run(place, ("hou_sessions", {"action": "start"}))
    fresh = ok(again)["session"]
    assert fresh["session_id"] != session_id
    with pool.open_store(home) as store:
        record = next(w for w in store.list_workers() if w.session_id == fresh["session_id"])
    assert pool.kill_process(int(record.pid), record.pid_start)
    support.wait_until(lambda: not pool.worker_is_alive(record), timeout_s=30.0)
    [after_kill] = run(place, ("hou_sessions", {}))
    assert rows(after_kill)[fresh["session_id"]]["state"] == "crashed"
    shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job shutdown")
def test_the_store_reopens_after_a_server_bound_worker_is_killed(tmp_path: Path) -> None:
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    home.mkdir()
    scratch.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )

    async def start_and_leave() -> dict[str, Any]:
        async with Client(
            server_params({"home": home, "scratch": scratch}),
            mode="auto",
            read_timeout_seconds=READ_TIMEOUT_S,
        ) as connected:
            return ok(await connected.call_tool("hou_sessions", {"action": "start"}))["session"]

    worker = asyncio.run(start_and_leave())
    try:
        if worker.get("lifetime") != "server":
            pytest.skip("this client permits worker breakaway")
        support.wait_until(lambda: not pool.process_is_alive(worker["pid"]), timeout_s=30)
        with pool.open_store(home) as store:
            assert store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            store.reclaim_workers()
            record = store.reserve_worker(cap=1, token="after-hard-shutdown")
            store.release_worker(record.token)
    finally:
        if pool.process_is_alive(worker["pid"]):
            assert support.stop_everything(home, pool.PoolConfig(home=home)) == []


def _busy(home: Path, session_id: str) -> bool:
    try:
        session = client.Session.open(home, session_id)
        answer = client.health(session, timeout_s=2.0)
    except (client.BridgeUnreachable, client.SessionGone, client.SessionDead):
        return False
    return bool((answer.payload or {}).get("data", {}).get("busy"))


# How long the cook in the ping check holds the worker.
COOK_S = 6.0

COOK_CODE = f"""
geo = hou.node('/obj').createNode('geo', 'held')
sop = geo.createNode('python', 'slow')
sop.parm('python').set('import time\\ntime.sleep({COOK_S})')
sop.cook(force=True)
result = sop.path()
"""


def test_hou_ping_answers_from_health_while_a_cook_holds_the_worker(
    place: dict[str, Any],
) -> None:
    async def check() -> dict[str, Any]:
        seen: dict[str, Any] = {}
        async with Client(
            server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
        ) as connected:
            started = ok(await connected.call_tool("hou_sessions", {"action": "start"}))
            session_id = started["session"]["session_id"]
            try:
                cook = asyncio.ensure_future(
                    connected.call_tool(
                        "hou_python", {"session": session_id, "code": COOK_CODE, "timeout_s": 60}
                    )
                )
                await asyncio.to_thread(
                    support.wait_until, lambda: _busy(place["home"], session_id), timeout_s=30.0
                )
                began = time.monotonic()
                seen["during"] = ok(await connected.call_tool("hou_ping", {"session": session_id}))
                seen["during_s"] = time.monotonic() - began
                seen["cooked"] = ok(await cook)
                seen["after"] = ok(await connected.call_tool("hou_ping", {"session": session_id}))
            finally:
                await connected.call_tool(
                    "hou_sessions", {"action": "stop", "session": session_id, "force": True}
                )
        return seen

    seen = asyncio.run(check())
    during = seen["during"]
    # Health answers from memory, so the ping does not wait on the cook.
    assert seen["during_s"] < 0.5, seen["during_s"]
    assert during["call"]["ok"] is False
    assert during["call"]["code"] == "SESSION_BUSY"
    assert during["call"]["skipped"] is True
    health = during["health"]
    assert health["busy"] is True
    assert health["current_op"] == "python.run"
    assert health["busy_cause"] == "running python.run"
    assert 0 <= health["busy_for_s"] < COOK_S + 30.0
    assert health["busy_since"] <= time.time()
    assert seen["cooked"]["result"].endswith("/held/slow")
    # Once the cook is over the ping goes through the main thread path again.
    assert seen["after"]["call"]["ok"] is True
    assert seen["after"]["health"]["busy"] is False
