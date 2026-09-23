"""Two clients hammering one session, with and without the pacing.

Two server processes, each over stdio with a client of its own, send calls to
one real worker as fast as their answers come back. The worker stands in for a
session with a user interface: each server is started with `pace_workers`, a
constructor argument only tests pass and no config can set, so the pacing
applies to it.

Paced, every call still succeeds within its `wait_s`, each client stays
within its rate cap and its pause, counted from the admission times the server
puts in the trace, and the calls that waited say so in `throttled_ms`. With
the pacing turned off the same two clients go faster and nothing waits, which
shows the cap is what held them. The numbers are printed for a run with its output shown.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker, in a state folder of this
file's own, on the pool's ports, stopped again whatever happened, with a check
that nothing is left. Nothing here opens or saves a scene.
"""

from __future__ import annotations

import asyncio
import json
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
from nscr_houdini_mcp.store import WorkerRecord, process_is_alive


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

# The server as the entry point runs it, with the one test only argument.
SERVER_CODE = "from nscr_houdini_mcp import server; server.run(pace_workers=True)"

READ_TIMEOUT_S = 120.0

# How many calls each client sends.
CALLS = 30

PAUSE_MS = 50
PER_S = 10


@pytest.fixture(scope="module")
def worker(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, WorkerRecord]]:
    """One warm worker in a state folder of its own, stopped at the end."""
    home = tmp_path_factory.mktemp("storm") / "home"
    home.mkdir()
    config = pool.PoolConfig(
        home=home, cap=1, max_idle_s=600.0, port_range=PORT_RANGE, start_timeout_s=240.0
    )
    with pool.open_store(home) as store:
        record = pool.start_worker(config, store)
    try:
        yield home, record
    finally:
        left = support.stop_everything(home, config)
        assert left == [], f"workers were left running: {left}"
        assert not pool.worker_is_alive(record)
        assert record.pid is None or not process_is_alive(record.pid)
        assert registry.live_entries(home) == []


def settings(home: Path, name: str, *, pause_ms: int, per_s: int) -> Path:
    """A config for one run's servers, pacing the worker as if it had an interface."""
    path = home.parent / f"{name}.toml"
    path.write_text(
        f"gui_min_pause_ms = {pause_ms}\ngui_max_calls_per_s = {per_s}\n",
        encoding="utf-8",
    )
    return path


def server_params(home: Path, config: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", SERVER_CODE],
        env={
            "NSCR_MCP_HOME": str(home),
            "NSCR_MCP_CONFIG": str(config),
            "PYTHONIOENCODING": "utf-8",
        },
    )


async def hammer(home: Path, config: Path, alias: str, ready: asyncio.Barrier) -> dict[str, Any]:
    """One client: connect, wait for the other, then send every call back to back."""
    async with Client(
        server_params(home, config), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        await connected.list_tools()
        await ready.wait()
        sent: list[float] = []
        waited: list[int] = []
        failed: list[str] = []
        started = time.time()
        admitted: list[float] = []
        for _ in range(CALLS):
            sent.append(time.time())
            # A wait long enough for any turn, so none is refused as too far off.
            result = await connected.call_tool("hou_ping", {"session": alias, "wait_s": 10})
            if result.is_error:
                failed.append(result.content[0].text[:200])
                continue
            trace = result.structured_content["trace"]
            waited.append(int(trace.get("throttled_ms") or 0))
            # The server's own moment of letting the call through, on the wall
            # clock both ends share. Unpaced calls have none: they went when sent.
            admitted.append(float(trace.get("admitted_at") or sent[-1]))
        ended = time.time()
    return {
        "sent": sent,
        "admitted": admitted,
        "waited": waited,
        "failed": failed,
        "started": started,
        "ended": ended,
    }


async def two_clients(home: Path, config: Path, alias: str) -> list[dict[str, Any]]:
    ready = asyncio.Barrier(2)
    return list(
        await asyncio.gather(hammer(home, config, alias, ready), hammer(home, config, alias, ready))
    )


# How far a moment may sit from the edge of a second and still be taken as on
# the other side: the wall clock read in two places, a little apart.
EDGE_S = 0.02


def busiest_second(times: list[float]) -> int:
    """The most of these moments inside any one second, less the edge."""
    ordered = sorted(times)
    most = 0
    for index, start in enumerate(ordered):
        inside = sum(1 for other in ordered[index:] if other < start + 1.0 - EDGE_S)
        most = max(most, inside)
    return most


def smallest_gap(times: list[float]) -> float:
    ordered = sorted(times)
    return min(later - earlier for earlier, later in zip(ordered, ordered[1:], strict=False))


def summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    clients = []
    for run in runs:
        span = run["ended"] - run["started"]
        paced = [ms for ms in run["waited"] if ms > 0]
        clients.append(
            {
                "calls": len(run["sent"]),
                "failed": len(run["failed"]),
                "seconds": round(span, 3),
                "calls_per_s": round(len(run["sent"]) / span, 2),
                "busiest_second": busiest_second(run["admitted"]),
                "smallest_gap_ms": round(smallest_gap(run["admitted"]) * 1000.0, 1),
                "throttled_calls": len(paced),
                "throttled_ms_mean": round(sum(paced) / len(paced), 1) if paced else 0,
                "throttled_ms_max": max(paced, default=0),
            }
        )
    started = min(run["started"] for run in runs)
    ended = max(run["ended"] for run in runs)
    every = [moment for run in runs for moment in run["admitted"]]
    return {
        "clients": clients,
        "together_calls_per_s": round(len(every) / (ended - started), 2),
        "together_busiest_second": busiest_second(every),
    }


# A worker start and two runs of calls take longer than the suite's own
# per test limit on a slow runner, so this test carries its own deadline and
# gets to its own cleanup rather than being cut off.
@pytest.mark.timeout(900)
def test_two_clients_on_one_session_are_each_held_to_the_cap_and_never_refused(
    worker: tuple[Path, WorkerRecord],
) -> None:
    home, record = worker
    paced_config = settings(home, "paced", pause_ms=PAUSE_MS, per_s=PER_S)
    open_config = settings(home, "open", pause_ms=0, per_s=0)

    paced = summary(asyncio.run(two_clients(home, paced_config, record.alias)))
    unpaced = summary(asyncio.run(two_clients(home, open_config, record.alias)))
    print("\npaced: " + json.dumps(paced))
    print("unpaced: " + json.dumps(unpaced))

    for client in paced["clients"]:
        # Past the cap a call waits, inside its wait_s; none was refused.
        assert client["failed"] == 0
        assert client["calls"] == CALLS
        # Counted from the server's own admission times.
        assert client["busiest_second"] <= PER_S
        # One call out at a time, and the pause after each before the next.
        assert client["smallest_gap_ms"] >= PAUSE_MS - EDGE_S * 1000.0
        # Nearly every call after the first found the last one too recent.
        assert client["throttled_calls"] >= (CALLS - 1) * 0.8
        assert 0 < client["throttled_ms_max"] <= 1000
    # Two clients together get no more than their two allowances.
    assert paced["together_busiest_second"] <= 2 * PER_S

    for client in unpaced["clients"]:
        assert client["failed"] == 0
        assert client["throttled_calls"] == 0
    # The cap is what held the paced run back.
    assert unpaced["together_busiest_second"] > paced["together_busiest_second"]
    assert unpaced["together_calls_per_s"] > paced["together_calls_per_s"]
