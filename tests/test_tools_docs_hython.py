"""`hou_docs` through the real server, against a real worker and a real install.

The worker's build has its help folder on this machine, so every read comes
from the folder, with the worker live, with the worker busy running code, and
with the worker gone. The worker's help server is read directly for the same
pages, the way the tool reads it for a page the folder does not have. Then
every node page of the real folder is rendered, and the slowest are checked.

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
from nscr_houdini_mcp import helpdocs, helptext, pool
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

# What a page read from the folder may take.
FOLDER_READ_S = 0.1

# How long the worker is kept busy running code while pages are read.
BUSY_S = 6.0
LOOP = f"""
import time
end = time.time() + {BUSY_S}
while time.time() < end:
    pass
result = "done"
"""

WRANGLE = ("hou_docs", {"mode": "page", "path": "nodes/sop/attribwrangle"})
NOISE = ("hou_docs", {"mode": "vex", "function": "noise"})
SEARCH = ("hou_docs", {"mode": "search", "query": "wrangle", "limit": 20})


def real_corpus() -> helpdocs.Corpus:
    hfs = helpdocs.hfs_of_hython(pool.hython_path())
    return helpdocs.Corpus(hfs, helpdocs.build_of(hfs) or "")


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


def client(place: dict[str, Path]) -> Client:
    return Client(server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S)


async def _timed(connected: Any, name: str, arguments: dict) -> tuple[Any, float]:
    started = time.perf_counter()
    result = await connected.call_tool(name, arguments)
    return result, time.perf_counter() - started


async def _run(place: dict[str, Path], calls: list[tuple[str, dict]]) -> list[tuple[Any, float]]:
    """Each call's result and how long its answer took, as the client saw it."""
    async with client(place) as connected:
        return [await _timed(connected, name, arguments) for name, arguments in calls]


def run(place: dict[str, Path], *calls: tuple[str, dict]) -> list[tuple[Any, float]]:
    return asyncio.run(_run(place, list(calls)))


async def _while_busy(
    place: dict[str, Path], calls: list[tuple[str, dict]]
) -> tuple[Any, list[tuple[Any, float]]]:
    """The calls, made while the worker runs a loop of Python, and the loop's answer."""
    async with client(place) as connected:
        loop = asyncio.create_task(connected.call_tool("hou_python", {"code": LOOP}))
        # Long enough for the loop to hold the session.
        await asyncio.sleep(1.5)
        answered = [await _timed(connected, name, arguments) for name, arguments in calls]
        assert not loop.done(), "the loop ended before the reads were made"
        return await loop, answered


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


def test_docs_with_a_live_worker_a_busy_one_and_none(
    place: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    build = real_corpus().build
    started, *reads, asked = run(
        place,
        ("hou_sessions", {"action": "start"}),
        WRANGLE,
        NOISE,
        SEARCH,
        ("hou_python", {"code": "result = hou.helpServerUrl()"}),
    )
    session_id = ok(started[0])["session"]["session_id"]
    wrangle, noise, found = (ok(result) for result, _ in reads)
    # The worker's build has its folder here, so the folder answers.
    assert wrangle["source"] == "corpus"
    assert wrangle["build"] == build
    assert wrangle["trace"]["session_id"] == session_id
    check_wrangle(wrangle)
    assert noise["source"] == "corpus"
    check_noise(noise)
    paths = [row["path"] for row in found["results"]]
    assert "nodes/sop/attribwrangle" in paths[:3]
    assert found["results"][0]["title"] == "Wrangle"

    # The worker's help server, read the way the tool reads it for a page the
    # folder does not have. Content is what is checked here, not speed: a
    # help server's first page can take a while.
    url = helpdocs.usable_url(ok(asked[0])["result"])
    assert url is not None
    monkeypatch.setattr(helpdocs, "HELP_TIMEOUT_S", 20.0)
    title, text = helptext.html_to_text(helpdocs.fetch(url, "nodes/sop/attribwrangle"))
    assert title == "Attribute Wrangle"
    assert "Group:" in text and "Run Over:" in text
    title, text = helptext.html_to_text(helpdocs.fetch(url, "vex/functions/noise"))
    assert title == "noise"
    assert "noise(vector pos)" in text
    hits = helptext.search_hits(
        helpdocs.fetch(url, "_search", query={"q": "wrangle"}), helpdocs.tidy_path
    )
    assert "nodes/sop/attribwrangle" in [hit["path"] for hit in hits]
    monkeypatch.undo()

    # While the worker runs code, pages still come at once, from the folder,
    # a page read before and pages never read.
    looped, answered = asyncio.run(
        _while_busy(
            place,
            [
                WRANGLE,
                ("hou_docs", {"mode": "page", "path": "nodes/sop/copytopoints"}),
                ("hou_docs", {"mode": "vex", "function": "pnoise"}),
            ],
        )
    )
    assert ok(looped)["result"] == "done"
    for result, took in answered:
        body = ok(result)
        assert body["source"] == "corpus"
        print(f"read while busy: {body['path']} in {took * 1000:.1f} ms")
        assert took < FOLDER_READ_S, (body["path"], took)
    copy = ok(answered[1][0])
    assert copy["version"] == "2.0"

    [stopped] = run(place, ("hou_sessions", {"action": "stop", "session": session_id}))
    assert ok(stopped[0])["stopped"]["ended"] is True

    # The session is gone: the configured install's folder answers.
    answered = run(
        place,
        SEARCH,
        ("hou_docs", {"mode": "page", "path": "nodes/sop/box"}),
        WRANGLE,
        NOISE,
        ("hou_docs", {"mode": "page", "path": "nodes/sop/copytopoints::1.0"}),
        ("hou_docs", {"mode": "search", "query": "copy to points"}),
    )
    (found, _), (box, cold_s), (wrangle, warm_s), (noise, _), (older, _), (copies, _) = answered
    found, box, wrangle, noise = ok(found), ok(box), ok(wrangle), ok(noise)
    # The version before the current one lives in the file with no number.
    assert ok(older)["title"] == "Copy to Points"
    assert ok(older)["version"] == "1.0"
    versions = {row["path"]: row.get("version") for row in ok(copies)["results"]}
    assert versions["nodes/sop/copytopoints"] == "2.0"
    assert versions["nodes/sop/copytopoints-"] == "older"
    assert wrangle["source"] == "corpus"
    assert wrangle["trace"]["session_id"] is None
    check_wrangle(wrangle)
    check_noise(noise)
    assert box["title"] == "Box"
    assert {row["source"] for row in found["results"]} == {"corpus"}
    assert "read from the cache" in found["results"][0]["note"]
    print(f"folder page read: {cold_s * 1000:.1f} ms, then {warm_s * 1000:.1f} ms from the cache")
    assert cold_s < FOLDER_READ_S
    assert warm_s < FOLDER_READ_S


def test_every_node_page_renders_quickly() -> None:
    corpus = real_corpus()
    if not corpus.exists():
        pytest.skip("this install has no help folder")
    index = helpdocs.build_index(corpus)
    times: list[tuple[float, str]] = []
    for path, *_ in index.pages:
        if not path.startswith("nodes/"):
            continue
        started = time.perf_counter()
        source = corpus.read(path)
        assert source is not None, path
        helptext.markup_to_text(source, read=corpus.read, where=path)
        times.append((time.perf_counter() - started, path))
    times.sort()
    p95 = times[int(len(times) * 0.95)][0]
    slowest = times[-1]
    print(
        f"{len(times)} node pages: p95 {p95 * 1000:.1f} ms,"
        f" slowest {slowest[1]} {slowest[0] * 1000:.1f} ms"
    )
    assert len(times) > 1000
    assert p95 < 0.1
    # The page with the most parts pulled in from other pages.
    started = time.perf_counter()
    source = corpus.read("nodes/sop/rbdmaterialfracture")
    assert source is not None
    page = helptext.markup_to_text(source, read=corpus.read, where="nodes/sop/rbdmaterialfracture")
    assert time.perf_counter() - started < 0.1
    assert page.title == "RBD Material Fracture"
