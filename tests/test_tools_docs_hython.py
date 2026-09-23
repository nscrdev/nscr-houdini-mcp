"""`hou_docs` through the real server, against a real worker and a real install.

Each page is read twice: first from the help server of a live worker, then,
once that worker is stopped, from the install's own help folder, which the
config names through its hython. A search runs both ways too. Then the time a
page read from the folder takes is checked with the index already built.

Skipped, not failed, when there is no Houdini on this machine. The same house
rules as the other checks that start a Houdini: one at a time (the pool cap is
one), a state folder of this file's own, the pool's port range, and every
worker stopped again whatever happened, with a check that nothing is left.
"""

from __future__ import annotations

import asyncio
import sys
import time
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

# What a page read from the folder may take, once the index is there.
FOLDER_READ_S = 0.1

WRANGLE = ("hou_docs", {"mode": "page", "path": "nodes/sop/attribwrangle"})
NOISE = ("hou_docs", {"mode": "vex", "function": "noise"})
SEARCH = ("hou_docs", {"mode": "search", "query": "wrangle", "limit": 20})


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    """A state folder with a config of its own, cleared of workers at the end."""
    root = tmp_path_factory.mktemp("docs")
    home = root / "home"
    home.mkdir()
    scratch = root / "houdini-temp"
    scratch.mkdir()
    hython = pool.hython_path()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\nhython = '{hython}'\n",
        encoding="utf-8",
    )
    try:
        yield {"home": home, "scratch": scratch, "root": root}
    finally:
        left = support.stop_everything(home, pool.PoolConfig(home=home))
        assert left == [], f"workers were left running: {left}"
        with pool.open_store(home) as store:
            for worker in store.list_workers(active_only=False):
                assert worker.pid is None or not pool.worker_is_alive(worker), worker.alias
        assert registry.live_entries(home) == []


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


async def _run(place: dict[str, Path], calls: list[tuple[str, dict]]) -> list[tuple[Any, float]]:
    """Each call's result and how long its answer took, as the client saw it."""
    answered = []
    async with Client(
        server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        for name, arguments in calls:
            started = time.perf_counter()
            result = await connected.call_tool(name, arguments)
            answered.append((result, time.perf_counter() - started))
    return answered


def run(place: dict[str, Path], *calls: tuple[str, dict]) -> list[tuple[Any, float]]:
    return asyncio.run(_run(place, list(calls)))


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def check_wrangle(body: dict[str, Any]) -> None:
    assert body["title"] == "Attribute Wrangle"
    assert body["path"] == "nodes/sop/attribwrangle"
    text = body["text"]
    assert text.startswith("Runs a VEX snippet to modify attribute values.")
    for name in ("Group:", "Run Over:", "Attributes to Create:", "Autobind by Name:"):
        assert name in text, name
    assert "On this page" not in text
    assert "#type" not in text and ":include" not in text


def check_noise(body: dict[str, Any]) -> None:
    assert body["title"] == "noise"
    assert body["path"] == "vex/functions/noise"
    assert "float noise(vector pos)" in body["text"]
    assert "Perlin" in body["text"]


def test_docs_from_a_live_help_server_then_from_the_install(place: dict[str, Path]) -> None:
    started, *reads = run(
        place,
        ("hou_sessions", {"action": "start"}),
        WRANGLE,
        NOISE,
        SEARCH,
        WRANGLE,
    )
    session_id = ok(started[0])["session"]["session_id"]
    wrangle, noise, found, again = (ok(result) for result, _ in reads)
    assert wrangle["source"] == "help_server", wrangle.get("note")
    check_wrangle(wrangle)
    assert noise["source"] == "help_server", noise.get("note")
    check_noise(noise)
    assert again["source"] == "help_server"
    assert wrangle["trace"]["session_id"] == session_id
    hits = {row["path"]: row for row in found["results"]}
    assert "nodes/sop/attribwrangle" in hits
    assert "help_server" in {row["source"] for row in found["results"]}
    assert "index of" in found["results"][0]["note"]

    [stopped] = run(place, ("hou_sessions", {"action": "stop", "session": session_id}))
    assert ok(stopped[0])["stopped"]["ended"] is True

    # The session is gone: the same reads come from the install's help folder.
    answered = run(place, SEARCH, WRANGLE, WRANGLE, NOISE)
    (found, _), (wrangle, cold_s), (_, warm_s), (noise, _) = answered
    found, wrangle, noise = ok(found), ok(wrangle), ok(noise)
    assert wrangle["source"] == "corpus"
    assert wrangle["trace"]["session_id"] is None
    check_wrangle(wrangle)
    assert noise["source"] == "corpus"
    check_noise(noise)
    assert {row["source"] for row in found["results"]} == {"corpus"}
    paths = [row["path"] for row in found["results"]]
    assert "nodes/sop/attribwrangle" in paths
    assert paths.index("nodes/sop/attribwrangle") < 20
    assert found["results"][0]["title"].lower() == "wrangle"
    assert "read from the cache" in found["results"][0]["note"]

    # With the index built, a page read from the folder answers quickly, the
    # first time and from the cache after.
    print(f"folder page read: {cold_s * 1000:.1f} ms, then {warm_s * 1000:.1f} ms from the cache")
    assert cold_s < FOLDER_READ_S
    assert warm_s < FOLDER_READ_S
