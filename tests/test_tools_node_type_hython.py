"""`hou_node_type` through the real server, against a real worker.

A client starts the server over stdio, the server starts a hython worker, and
every lookup is made the way a client makes it, against the types this build
of Houdini really has: a wrangle's inputs and its code parameter, an object
null, a keyword search, a misspelled name, and a namespaced type where the
build ships one. The worker's scene stays empty throughout, which is the check
that a lookup makes no node.

Skipped, not failed, when there is no Houdini on this machine. The same house
rules as the other checks that start a Houdini: one worker at a time (the pool
cap in this file's own config is one), a state folder of this file's own, the
pool's port range, no scene file opened or written, and every worker stopped
again whatever happened, with a check that nothing is left.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
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

# A namespaced type a build may ship, and the context it is in.
NAMESPACED = ("sop", "kinefx::rigattribwrangle")


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A state folder with a config of its own and one worker, stopped at the end."""
    root = tmp_path_factory.mktemp("node_type")
    home = root / "home"
    home.mkdir()
    scratch = root / "houdini-temp"
    scratch.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    found: dict[str, Any] = {"home": home, "scratch": scratch}
    try:
        [started] = run(found, ("hou_sessions", {"action": "start"}))
        found["session"] = ok(started)["session"]["session_id"]
        yield found
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
    return asyncio.run(_run(place, list(calls)))


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def refused(result: Any) -> dict[str, Any]:
    assert result.is_error is True, result.content[0].text
    return result.structured_content["error"]


def test_types_as_the_build_defines_them(place: dict[str, Any]) -> None:
    wrangle, full, null, search, misspelled, elsewhere, tree = run(
        place,
        ("hou_node_type", {"context": "sop", "type": "attribwrangle", "detail": "standard"}),
        (
            "hou_node_type",
            {"context": "sop", "type": "attribwrangle", "detail": "full", "parm_filter": "class"},
        ),
        ("hou_node_type", {"context": "obj", "type": "null", "detail": "standard"}),
        ("hou_node_type", {"context": "sop", "query": "wrangle"}),
        ("hou_node_type", {"context": "sop", "type": "atribwrangle"}),
        ("hou_node_type", {"context": "obj", "type": "attribwrangle"}),
        ("hou_inspect", {"path": "/obj"}),
    )

    body = ok(wrangle)
    assert body["type"] == "attribwrangle"
    assert body["category"] == "Sop"
    assert body["max_inputs"] == 4
    assert len(body["inputs"]) == 4
    assert all(item["label"] for item in body["inputs"])
    assert body["inputs"][0]["optional"] is True
    snippet = next(row for row in body["parms"] if row["name"] == "snippet")
    assert snippet["type"] == "String"
    assert snippet["code"] == "vex"
    assert body["trace"]["session_id"] == place["session"]

    [run_over] = ok(full)["parms"]
    assert run_over["name"] == "class"
    assert run_over["default"] == "point"
    assert [item["token"] for item in run_over["menu"]][:4] == [
        "detail",
        "primitive",
        "point",
        "vertex",
    ]
    assert ok(full)["help_summary"]
    assert ok(full)["help_path"] == "/nodes/sop/attribwrangle"

    null_body = ok(null)
    assert null_body["category"] == "Object"
    translate = next(row for row in null_body["parms"] if row["name"] == "t")
    assert translate["size"] == 3
    assert translate["default"] == [0.0, 0.0, 0.0]

    found = [row["type"] for row in ok(search)["rows"]]
    assert "attribwrangle" in found
    assert all("wrangle" in name for name in found[:3])

    error = refused(misspelled)
    assert error["code"] == "TYPE_NOT_FOUND"
    assert error["details"]["did_you_mean"][0] == "attribwrangle"
    assert len(error["details"]["did_you_mean"]) <= 5

    wrong = refused(elsewhere)
    assert wrong["code"] == "TYPE_NOT_FOUND"
    assert {"category": "Sop", "context": "sop"} in wrong["details"]["found_in"]

    # Nothing was made to find any of that out.
    assert ok(tree)["rows"] == []


def test_the_parameters_page_against_a_real_type(place: dict[str, Any]) -> None:
    arguments = {"context": "sop", "type": "attribwrangle", "detail": "full"}
    [whole] = run(place, ("hou_node_type", {**arguments, "limit": 2000}))
    rows = ok(whole)["parms"]
    seen: list[dict[str, Any]] = []
    page = None
    calls = 0
    while calls < 50:
        calls += 1
        extra = {"page": page} if page else {}
        [result] = run(place, ("hou_node_type", {**arguments, "limit": 7, **extra}))
        body = ok(result)
        assert "changed" not in body
        seen.extend(body["parms"])
        page = body.get("next_page")
        if not page:
            break
    assert seen == rows


def test_a_namespaced_type_where_the_build_ships_one(place: dict[str, Any]) -> None:
    context, name = NAMESPACED
    [result] = run(
        place, ("hou_node_type", {"context": context, "type": name, "detail": "standard"})
    )
    if result.is_error and result.structured_content["error"]["code"] == "TYPE_NOT_FOUND":
        pytest.skip(f"this build has no {name}")
    body = ok(result)
    assert body["type"] == name
    assert body["namespace"] == name.split("::")[0]
    assert body["parms"]


def test_a_menu_whose_items_toggle_is_read_without_asking_for_its_token(
    place: dict[str, Any],
) -> None:
    # Asking this build for the default token of such a menu ends the process.
    [result] = run(
        place,
        ("hou_node_type", {"context": "obj", "type": "blend", "detail": "full", "limit": 2000}),
    )
    rows = {row["name"]: row for row in ok(result)["parms"]}
    toggles = [row for row in rows.values() if row.get("menu_toggles")]
    assert toggles
    assert all(isinstance(row["default"], int) for row in toggles)
    [alive] = run(place, ("hou_ping", {}))
    assert ok(alive)["call"]["ok"] is True


# A namespaced name whose newest version is another type, and one whose
# namespace order starts with a type of another namespace.
MADE_AS = ("kinefx::ragdollsolver", "apex::invokegraph")

MAKE = """
geo = hou.node("/obj").createNode("geo", "made_as_probe")
result = {}
for name in %r:
    try:
        result[name] = geo.createNode(name).type().name()
    except hou.OperationFailed:
        result[name] = None
geo.destroy()
"""


def test_a_card_is_the_type_making_a_node_of_that_name_gives(place: dict[str, Any]) -> None:
    made, *cards = run(
        place,
        ("hou_python", {"code": MAKE % (MADE_AS,)}),
        *(("hou_node_type", {"context": "sop", "type": name}) for name in MADE_AS),
    )
    created = ok(made)["result"]
    checked = 0
    for name, card in zip(MADE_AS, cards, strict=True):
        if created[name] is None:
            continue
        assert ok(card)["type"] == created[name], name
        checked += 1
    if not checked:
        pytest.skip("this build has neither type")
    [tree] = run(place, ("hou_inspect", {"path": "/obj"}))
    assert ok(tree)["rows"] == []
