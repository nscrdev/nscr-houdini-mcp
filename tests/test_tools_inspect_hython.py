"""`hou_inspect` through the real server, against a real worker.

A small network is built by a short hython script and saved into the test's
own temporary folder, then a worker the server starts opens it and every read
is made the way a client makes it. The checks that matter most here are the
ones a stand in cannot answer: that a read without `evaluate` cooks nothing at
all, and that a read with it cooks and the marks go.

Skipped, not failed, when there is no Houdini on this machine. The same house
rules as the other checks that start a Houdini: one at a time (the build
script ends before the worker starts, and the pool cap is one), a state folder
of this file's own, the pool's port range, and every worker stopped again
whatever happened, with a check that nothing is left.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
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
BUILD_TIMEOUT_S = 300.0

# How many plain nodes the network carries, so a tree has pages to turn.
FILLER = 24

BUILD = """
import sys
import hou

geo = hou.node("/obj").createNode("geo", "geo1")
box = geo.createNode("box", "box1")
box.parm("sizex").set(2)
move = geo.createNode("xform", "transform1")
move.setInput(0, box)
move.parm("tx").setExpression("$F*2")
move.parm("ty").setExpression('npoints("../box1")')
out = geo.createNode("null", "OUT")
out.setInput(0, move)
out.setDisplayFlag(True)
out.setRenderFlag(True)
wrangle = geo.createNode("attribwrangle", "attribwrangle1")
wrangle.setInput(0, box)
wrangle.parm("snippet").set("@P.y += 1;")
for index in range({filler}):
    geo.createNode("null", "n%02d" % index)
note = geo.createStickyNote()
note.setText("the base shape")
hou.hipFile.save(sys.argv[1])
"""


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    """A state folder with a config of its own, cleared of workers at the end."""
    root = tmp_path_factory.mktemp("inspect")
    home = root / "home"
    home.mkdir()
    scratch = root / "houdini-temp"
    scratch.mkdir()
    scenes = root / "scenes"
    scenes.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    try:
        yield {"home": home, "scratch": scratch, "scenes": scenes, "root": root}
    finally:
        left = support.stop_everything(home, pool.PoolConfig(home=home))
        assert left == [], f"workers were left running: {left}"
        with pool.open_store(home) as store:
            for worker in store.list_workers(active_only=False):
                assert worker.pid is None or not pool.worker_is_alive(worker), worker.alias
        assert registry.live_entries(home) == []


def build_scene(place: dict[str, Path]) -> Path:
    """Build the network in a hython of its own, which has ended when this returns."""
    script = place["root"] / "build_scene.py"
    script.write_text(BUILD.format(filler=FILLER), encoding="utf-8")
    hip = place["scenes"] / "inspect.hip"
    done = subprocess.run(
        [str(pool.hython_path()), str(script), str(hip)],
        cwd=place["scenes"],
        env={**os.environ, "HOUDINI_TEMP_DIR": str(place["scratch"])},
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_S,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert hip.is_file()
    return hip


def server_params(place: dict[str, Path]) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", SERVER_CODE],
        env={
            "NSCR_MCP_HOME": str(place["home"]),
            "HOUDINI_TEMP_DIR": str(place["scratch"]),
            "PYTHONIOENCODING": "utf-8",
        },
    )


async def _run(place: dict[str, Path], calls: list[tuple[str, dict]]) -> list[Any]:
    async with Client(
        server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        return [await connected.call_tool(name, arguments) for name, arguments in calls]


def run(place: dict[str, Path], *calls: tuple[str, dict]) -> list[Any]:
    return asyncio.run(_run(place, list(calls)))


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def counts(result: Any) -> dict[str, int]:
    return {entry["path"]: entry["cooks"] for entry in ok(result)["nodes"]}


WATCHED = ["/obj/geo1/box1", "/obj/geo1/transform1", "/obj/geo1/OUT", "/obj/geo1/attribwrangle1"]
COUNT = ("hou_inspect", {"mode": "node", "paths": WATCHED, "include": ["cook_time"]})


def test_reads_against_a_real_worker(place: dict[str, Path]) -> None:
    hip = build_scene(place)

    started, opened, before = run(
        place,
        ("hou_sessions", {"action": "start"}),
        ("hou_scene", {"action": "open", "path": str(hip)}),
        COUNT,
    )
    session_id = ok(started)["session"]["session_id"]
    assert ok(opened)["hip_path"] == str(hip)

    # Nothing has cooked since the load, and no read without evaluate cooks.
    summary, tree, full, parms, after = run(
        place,
        ("hou_inspect", {"mode": "node", "paths": WATCHED}),
        ("hou_inspect", {"path": "/obj/geo1", "detail": "standard"}),
        ("hou_inspect", {"mode": "node", "paths": WATCHED, "detail": "full"}),
        ("hou_inspect", {"mode": "parms", "path": "/obj/geo1/transform1", "parm_filter": "all"}),
        COUNT,
    )
    assert counts(after) == counts(before)
    assert set(counts(before).values()) == {0}
    entries = {entry["path"]: entry for entry in ok(summary)["nodes"]}
    assert entries["/obj/geo1/OUT"]["not_cooked"] is True
    assert entries["/obj/geo1/OUT"]["in"] == ["/obj/geo1/transform1"]
    assert "display" in entries["/obj/geo1/OUT"]["flags"]
    rows = {row["path"]: row for row in ok(tree)["rows"]}
    assert rows["/obj/geo1/box1"]["auto"] is True
    assert rows["/obj/geo1/transform1"]["in"] == ["/obj/geo1/box1"]
    assert ok(tree)["notes"][0]["text"] == "the base shape"
    wrangle = next(e for e in ok(full)["nodes"] if e["path"] == "/obj/geo1/attribwrangle1")
    snippet = next(row for row in wrangle["parms"] if row["n"] == "snippet")
    assert snippet == {"n": "snippet", "code": "vex", "v": "@P.y += 1;"}
    translate = next(row for row in ok(parms)["nodes"][0]["parms"] if row["n"] == "t")
    assert translate["expr"] == {"tx": "$F*2", "ty": 'npoints("../box1")'}
    assert translate["lang"] == "hscript"
    assert translate["not_cooked"] is True
    assert "v" not in translate

    # A tree pages by path, and the pages put together are the whole of it.
    whole = ok(run(place, ("hou_inspect", {"path": "/obj/geo1", "limit": 500}))[0])
    names = [row["path"] for row in whole["rows"]]
    assert names == sorted(names, key=lambda path: path.split("/"))
    assert len(names) == FILLER + 4
    seen: list[str] = []
    page = None
    for _ in range(10):
        arguments = {"path": "/obj/geo1", "limit": 10, **({"page": page} if page else {})}
        [result] = run(place, ("hou_inspect", arguments))
        body = ok(result)
        assert "scene_changed" not in body
        seen.extend(row["path"] for row in body["rows"])
        page = body.get("next_page")
        if not page:
            break
    assert seen == names

    # Found by a glob on the name.
    [found] = run(place, ("hou_inspect", {"mode": "find", "pattern": "n1*"}))
    assert [row["path"] for row in ok(found)["rows"]] == [
        f"/obj/geo1/n{index}" for index in range(10, 20)
    ]

    # With evaluate the read cooks, the marks go and the value is there.
    evaluated, cooked = run(
        place,
        (
            "hou_inspect",
            {"mode": "node", "path": "/obj/geo1/transform1", "detail": "full", "evaluate": True},
        ),
        COUNT,
    )
    [entry] = ok(evaluated)["nodes"]
    assert "not_cooked" not in entry and "stale" not in entry
    translate = next(row for row in entry["parms"] if row["n"] == "t")
    assert "not_cooked" not in translate
    assert translate["v"][1] == 8
    assert translate["v"][0] == pytest.approx(2.0)
    now = counts(cooked)
    assert now["/obj/geo1/transform1"] >= 1
    assert now["/obj/geo1/box1"] >= 1
    assert now["/obj/geo1/OUT"] == 0

    [stopped] = run(place, ("hou_sessions", {"action": "stop", "session": session_id}))
    assert ok(stopped)["stopped"]["ended"] is True
