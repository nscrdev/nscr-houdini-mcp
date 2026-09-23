"""The scripted pass in `sequence.py`, against the real server and a real worker.

Three runs, each with a server process of its own over stdio and a worker the
pass starts and stops itself:

- `Client` in `auto` mode, which settles on the current revision, 2026-07-28.
- `Client` in `legacy` mode, which forces the initialize handshake and settles
  on 2025-11-25.
- `ClientSession` over `stdio_client`, the SDK's lower, generic client class,
  with the initialize handshake a hand written client makes.

A last check puts the three records side by side: every step must have the
same shape in each, keys and kinds of block, whatever revision carried it.
Values are not compared, since paths, ids and times differ every run.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker at a time (the pool cap in
each run's config is one), a state folder of each run's own, the pool's port
range, and every worker stopped again whatever happened, with a check that
nothing is left. Every scene file and output the pass writes is inside the
test's own temporary folder, because the worker's `$HOUDINI_TEMP_DIR` is.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from mcp.client.client import Client
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

import sequence
import support
from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge import registry
from nscr_houdini_mcp.tools.registry import TOOLS


def hython_available() -> bool:
    try:
        pool.hython_path()
    except pool.HythonNotFound:
        return False
    return True


# A cold worker start and a whole pass take longer than the suite's own per
# test limit on a slow runner (the client here waits up to 300 seconds a call),
# so every test in this file carries its own deadline and gets to its own
# cleanup rather than being cut off.
pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
    pytest.mark.timeout(900),
]

PORT_RANGE = support.POOL_PORTS

SERVER_CODE = "from nscr_houdini_mcp.cli import main; raise SystemExit(main([]))"

# A worker start is a cold Houdini, so the client waits longer than it would
# for an ordinary call.
READ_TIMEOUT_S = 300.0

# The records of the runs so far, by run name, for the side by side check.
RECORDS: dict[str, sequence.Record] = {}


@pytest.fixture
def place(tmp_path: Path) -> Iterator[dict[str, Path]]:
    """A state folder and a scratch folder of this run's own, emptied of workers after."""
    home = tmp_path / "home"
    home.mkdir()
    scratch = tmp_path / "houdini-temp"
    scratch.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\ninline_wait_s = 1\n",
        encoding="utf-8",
    )
    try:
        yield {"home": home, "scratch": scratch}
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
            # Carried on to the worker the server starts, so its untitled
            # scene, its saves and its outputs all land in this test's folder.
            "HOUDINI_TEMP_DIR": str(place["scratch"]),
            "PYTHONIOENCODING": "utf-8",
        },
    )


async def through_client(place: dict[str, Path], mode: str) -> sequence.Record:
    async with Client(
        server_params(place), mode=mode, read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        record = sequence.Record(protocol=connected.protocol_version)
        return await sequence.run_sequence(connected, record=record)


async def through_session(place: dict[str, Path]) -> sequence.Record:
    async with (
        stdio_client(server_params(place)) as (read, write),
        ClientSession(read, write, read_timeout_seconds=READ_TIMEOUT_S) as connected,
    ):
        await connected.initialize()
        record = sequence.Record(protocol=connected.protocol_version)
        return await sequence.run_sequence(connected, record=record)


RUNS: dict[str, tuple[str, Callable[[dict[str, Path]], Awaitable[sequence.Record]]]] = {
    "client-auto": ("2026-07-28", lambda place: through_client(place, "auto")),
    "client-legacy": ("2025-11-25", lambda place: through_client(place, "legacy")),
    "session-initialize": ("2025-11-25", through_session),
}


def report(name: str, record: sequence.Record) -> None:
    """Print the per step record, for a run with its output shown."""
    print(f"\n{name} ({record.protocol})")
    for row in record.rows():
        print("  " + json.dumps(row))


def run_named(place: dict[str, Path], name: str) -> sequence.Record:
    revision, start = RUNS[name]
    record = asyncio.run(start(place))
    assert record.protocol == revision
    assert record.tools == [spec.name for spec in TOOLS]
    RECORDS[name] = record
    report(name, record)
    # Everything the pass wrote is under this test's own folders.
    hips = sorted(place["scratch"].rglob("*.hip"))
    assert [hip.name for hip in hips] == ["untitled_v001.hip"]
    assert list(place["scratch"].rglob("*.claim")) == []
    compares = list(place["scratch"].rglob("result.json"))
    assert len(compares) == 1
    return record


def test_the_pass_under_the_current_revision(place: dict[str, Path]) -> None:
    run_named(place, "client-auto")


def test_the_pass_under_the_handshake_revision(place: dict[str, Path]) -> None:
    run_named(place, "client-legacy")


def test_the_pass_through_the_generic_client_session(place: dict[str, Path]) -> None:
    run_named(place, "session-initialize")


def test_every_step_has_the_same_shape_under_each_revision_and_client() -> None:
    missing = sorted(set(RUNS) - set(RECORDS))
    if missing:
        pytest.skip(f"the runs {missing} did not finish in this session")
    first = next(iter(RUNS))
    expected: list[dict] = []
    stamps: dict[str, list[str]] = {}
    for name in RUNS:
        shapes = RECORDS[name].shapes()
        stamps[name] = sorted({key for row in shapes for key in row.pop("meta")})
        if name == first:
            expected = shapes
            continue
        assert [row["step"] for row in shapes] == [row["step"] for row in expected], name
        for mine, theirs in zip(shapes, expected, strict=True):
            assert mine == theirs, f"{name} differs from {first} at {mine['step']}"
    print("\nprotocol additions by run: " + json.dumps(stamps))
    # The one thing the revision changes: the current one stamps every result
    # with the server's own details in its metadata, the handshake one does not.
    for name, (revision, _) in RUNS.items():
        if revision == "2025-11-25":
            assert stamps[name] == [], name
        else:
            assert stamps[name], name
            assert all("serverinfo" in key.lower() for key in stamps[name]), name
