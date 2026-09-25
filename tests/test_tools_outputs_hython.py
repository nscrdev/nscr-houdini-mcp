"""`hou_outputs` and the parameter ruling against a real worker.

A client starts the server over stdio the way any client does, and the server
starts a hython worker through the pool. A file cache node in a saved scene
gets its cache path. A render node's picture is frozen to a run's own path
while the code runs, gets its `$HIP` line back when the call ends, and the
scene saved afterwards is read back to show the run's path is not in it. The
lint then finds one picture set to an absolute path by hand.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker (the pool cap in this
file's own config is one), a state folder of this file's own, the pool's port
range, and every worker stopped again whatever happened, with a check that
nothing is left. Every scene file read or written is inside the test's own
temporary folder.
"""

from __future__ import annotations

import asyncio
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

# A worker start is a cold Houdini, so the client waits longer than it would
# for an ordinary call.
READ_TIMEOUT_S = 300.0


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A state folder with a config of its own and one worker, stopped at the end."""
    root = tmp_path_factory.mktemp("outputs")
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
    with support.persistent_client(server_params(made), timeout_s=READ_TIMEOUT_S) as send:
        made["send"] = send
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


def fresh_scene(place: dict[str, Any], name: str, build: str) -> Path:
    """A new scene with some nodes in it, saved in the test's own folder."""
    hip = place["scenes"] / name
    code = (
        "hou.hipFile.clear(suppress_save_prompt=True)\n"
        f"{build}\n"
        f"hou.hipFile.save({hip.as_posix()!r})\n"
        "result = hou.hipFile.path()"
    )
    [saved] = run(place, ("hou_python", {"code": code}))
    assert Path(ok(saved)["result"]).resolve() == hip.resolve()
    return hip


def test_a_cache_path_for_a_file_cache_node_in_a_saved_scene(place: dict[str, Any]) -> None:
    fresh_scene(
        place,
        "cache_v001.hip",
        "hou.node('/obj').createNode('geo', 'geo1').createNode('filecache', 'sim')",
    )
    [resolved] = run(
        place, ("hou_outputs", {"action": "resolve", "kind": "cache", "node": "/obj/geo1/sim"})
    )
    body = ok(resolved)
    assert body["parm_string"] == "$HIP/geo/${OS}/v001/${OS}_v001.$F4.bgeo.sc"
    expanded = Path(body["expanded_path"])
    assert expanded.name == "sim_v001.$F4.bgeo.sc"
    assert expanded.parent.resolve() == (place["scenes"] / "geo" / "sim" / "v001").resolve()
    assert Path(body["folder"]).is_dir()
    assert Path(body["sidecar"]).is_file()
    assert body["version"] == 1
    [listed] = run(place, ("hou_outputs", {"action": "list", "filter": {"kind": "cache"}}))
    [row] = ok(listed)["runs"]
    assert row["run_id"] == body["run_id"]
    assert row["node"] == "/obj/geo1/sim"
    assert row["on_disk"] is False


FREEZE = """
rop = hou.node('/out/beauty')
path = mcp.output_path('render', 'beauty')
mcp.freeze_parm(rop.parm('picture'), path)
result = {'during': rop.parm('picture').unexpandedString(), 'path': path}
"""

DEFAULT_PICTURE = "$HIP/render/$HIPNAME.$OS.$F4.exr"
READ_PICTURE = "result = hou.node('/out/beauty').parm('picture').unexpandedString()"


def test_a_frozen_picture_is_given_back_and_the_saved_scene_holds_no_machine_path(
    place: dict[str, Any],
) -> None:
    hip = fresh_scene(place, "frozen_v001.hip", "hou.node('/out').createNode('karma', 'beauty')")
    froze, after, saved = run(
        place,
        ("hou_python", {"code": FREEZE}),
        ("hou_python", {"code": READ_PICTURE}),
        ("hou_scene", {"action": "save"}),
    )
    body = ok(froze)
    frozen = body["result"]["path"]
    # While the code ran, the node held the run's own absolute path.
    assert body["result"]["during"] == frozen
    assert Path(frozen).is_absolute()
    [restored] = body["restored_parms"]
    assert restored["restored"] is True
    # Afterwards it holds what it held before, the line a person set.
    assert ok(after)["result"] == DEFAULT_PICTURE
    ok(saved)
    written = hip.read_bytes()
    assert frozen.encode("utf-8") not in written
    assert DEFAULT_PICTURE.encode("utf-8") in written
    opened, reread, linted = run(
        place,
        ("hou_scene", {"action": "open", "path": str(hip)}),
        ("hou_python", {"code": READ_PICTURE}),
        ("hou_outputs", {"action": "lint", "node": "/out"}),
    )
    assert "restored_parms" not in ok(opened)
    assert ok(reread)["result"] == DEFAULT_PICTURE
    problems = {row["problem"] for row in ok(linted)["rows"] if row["node"] == "/out/beauty"}
    assert "frozen_after_run" not in problems
    assert "absolute_path" not in problems


def test_a_save_during_the_run_never_writes_the_run_path(place: dict[str, Any]) -> None:
    hip = fresh_scene(
        place, "saved_during_v001.hip", "hou.node('/out').createNode('karma', 'beauty')"
    )
    code = (
        FREEZE
        + "hou.hipFile.save()\n"
        + "result['after_save'] = rop.parm('picture').unexpandedString()\n"
    )
    [froze] = run(place, ("hou_python", {"code": code}))
    body = ok(froze)
    frozen = body["result"]["path"]
    # The run went on with its own path after the save.
    assert body["result"]["after_save"] == frozen
    written = hip.read_bytes()
    assert frozen.encode("utf-8") not in written
    assert DEFAULT_PICTURE.encode("utf-8") in written
    assert body["restored_parms"][0]["restored"] is True


def test_a_node_renamed_during_the_run_gets_its_value_back(place: dict[str, Any]) -> None:
    fresh_scene(place, "renamed_v001.hip", "hou.node('/out').createNode('karma', 'beauty')")
    code = FREEZE + "rop.setName('renamed')\n"
    froze, after = run(
        place,
        ("hou_python", {"code": code}),
        (
            "hou_python",
            {"code": "result = hou.node('/out/renamed').parm('picture').unexpandedString()"},
        ),
    )
    [restored] = ok(froze)["restored_parms"]
    assert restored["restored"] is True
    assert restored["node"] == "/out/renamed"
    assert ok(after)["result"] == DEFAULT_PICTURE


def test_an_expression_link_comes_back_as_an_expression(place: dict[str, Any]) -> None:
    build = (
        "out = hou.node('/out')\n"
        "out.createNode('karma', 'source')\n"
        "rop = out.createNode('karma', 'beauty')\n"
        "rop.parm('picture').setExpression('chs(\"../source/picture\")', hou.exprLanguage.Hscript)"
    )
    hip = fresh_scene(place, "linked_v001.hip", build)
    check = (
        "p = hou.node('/out/beauty').parm('picture')\n"
        "result = [p.expression(), str(p.expressionLanguage())]"
    )
    # A save while the run holds the path, and one after it is given back.
    froze, after, saved = run(
        place,
        ("hou_python", {"code": FREEZE + "hou.hipFile.save()\n"}),
        ("hou_python", {"code": check}),
        ("hou_scene", {"action": "save"}),
    )
    body = ok(froze)
    frozen = body["result"]["path"]
    assert body["result"]["during"] == frozen
    assert body["restored_parms"][0]["owed"] == {
        "expression": 'chs("../source/picture")',
        "language": "hscript",
    }
    assert ok(after)["result"] == ['chs("../source/picture")', "exprLanguage.Hscript"]
    ok(saved)
    # Houdini keeps the value an expression was set over and saves it with
    # the channel, so the path must not be what it was set over.
    assert frozen.encode("utf-8") not in hip.read_bytes()


def test_lint_finds_the_one_picture_set_to_an_absolute_path(place: dict[str, Any]) -> None:
    folder = place["scenes"].as_posix()
    build = (
        "out = hou.node('/out')\n"
        "out.createNode('karma', 'managed').parm('picture').set("
        "'$HIP/render/managed_v001.$F4.exr')\n"
        "out.createNode('karma', 'by_hand').parm('picture').set("
        f"'{folder}/render/by_hand_v001.$F4.exr')\n"
        "out.createNode('alembic', 'abc')\n"
        "out.createNode('comp', 'comp1')\n"
        "hou.node('/stage').createNode('usdrender_rop', 'husk')\n"
        "hou.node('/obj').createNode('geo', 'geo1').createNode('filecache', 'sim')"
    )
    fresh_scene(place, "lint_v001.hip", build)
    [linted] = run(place, ("hou_outputs", {"action": "lint"}))
    rows = ok(linted)["rows"]
    absolute = [row for row in rows if row["problem"] == "absolute_path"]
    assert [(row["node"], row["parm"]) for row in absolute] == [("/out/by_hand", "picture")]
    assert absolute[0]["raw"] == f"{folder}/render/by_hand_v001.$F4.exr"
    parms = {(row["node"], row["parm"]) for row in rows}
    # The parameter names come from each node type: the cache writes `file`,
    # an Alembic node an unmarked `filename`.
    assert ("/obj/geo1/sim", "file") in parms
    assert ("/out/abc", "filename") in parms
    # The deep output is off with its toggle, which a fresh node only says
    # once asked to work out its parameters; a renderer's logs, a render it
    # reads back and an empty unmarked image are not outputs.
    assert {parm for node, parm in parms if node == "/out/managed"} == {"picture"}
    assert {row["problem"] for row in rows if row["node"] == "/out/managed"} == {"missing_on_disk"}
    assert not {parm for node, parm in parms if node == "/stage/husk"} & {
        "husk_stdout",
        "husk_stderr",
        "renderexisting",
        "outputimage",
    }
    # A comp node's five spare outputs are empty and unused, not missing.
    assert not {parm for node, parm in parms if node == "/out/comp1" and parm.startswith("copaux")}
    assert ok(linted)["parms_checked"] >= 3
