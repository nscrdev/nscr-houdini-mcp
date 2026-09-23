"""Shared machinery for the tests that need real processes.

Three things live here, because more than one test file needs each of them and
a second copy would drift from the first.

- A Houdini of our own: a session started in a state folder nobody else uses,
  and stopped again whatever the test did. Never the Houdini a person is
  working in.
- Children: several processes started at the same moment on one store or one
  pool, with a barrier so they really do arrive together, timeouts on every
  wait, and whatever is still running ended by the runner.
- Ways to break things on purpose: an answer thrown away after the work ran, a
  session ended outright, and a caller that goes away in the middle of its own
  call. Each one is a failure the bridge must survive, so each one has to be
  producible on demand rather than waited for.

Ports. Every file that starts a session uses a range of its own, so a bridge
somebody started by hand keeps the port it has and two test files never fight
over one number. The ranges in use are gathered here.

Children are started with the spawn method, which is the only one on every
supported system, so a child imports this module by name and calls a function
it was given at module level. Anything a child runs has to be importable that
way, which is why the targets are plain module level functions.

Nothing here imports pytest, so the acceptance script can use the same helpers
without the test runner.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_module
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.bridge.launcher import HythonBridge, hython_available

__all__ = [
    "APP_PORTS",
    "BRIDGE_PORTS",
    "FAILURE_PORTS",
    "INSTALL_PORTS",
    "POOL_PORTS",
    "call_and_die",
    "client_that_dies_mid_call",
    "hython_available",
    "hython_session",
    "kill_bridge",
    "live_sessions",
    "run_children",
    "stop_everything",
    "wait_until",
]

# Port ranges, one per file that starts a session. The default range a person
# gets, 18100 to 18199, is not in this table and is never used by a test.
#
# A run can move all of them at once by setting the base variable to the first
# port it may use, which is how two runs on one machine keep out of each
# other's way. Every range then sits inside a hundred ports from that base.
PORT_BASE_ENV_VAR = "NSCR_MCP_TEST_PORT_BASE"


def _ports(default: tuple[int, int], offset: int, width: int) -> tuple[int, int]:
    base = os.environ.get(PORT_BASE_ENV_VAR, "").strip()
    if not base:
        return default
    start = int(base) + offset
    return (start, start + width - 1)


APP_PORTS = _ports((18200, 18249), 0, 15)
BRIDGE_PORTS = _ports((18300, 18349), 15, 15)
POOL_PORTS = _ports((18360, 18399), 30, 10)
FAILURE_PORTS = _ports((18410, 18429), 40, 10)
INSTALL_PORTS = _ports((18430, 18449), 45, 5)

BARRIER_TIMEOUT_S = 60.0
RESULT_TIMEOUT_S = 300.0
JOIN_TIMEOUT_S = 60.0

# How long a test waits for something it expects to happen shortly.
READY_TIMEOUT_S = 120.0

# How long a session that is being torn down is given to go.
STOP_TIMEOUT_S = 60.0


def wait_until(ready: Callable[[], Any], *, timeout_s: float = READY_TIMEOUT_S) -> None:
    """Wait for something to become true, or say how long it did not."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.05)
    raise AssertionError(f"waited {timeout_s:g} seconds and it did not happen")


# Section: a Houdini of our own


@contextmanager
def hython_session(
    home: Path,
    *,
    port_range: tuple[int, int],
    alias: str | None = None,
    extra_args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> Iterator[HythonBridge]:
    """One hython session in a state folder of its own, stopped afterwards.

    The state folder is passed to the child as its own home, so nothing this
    session writes lands where a person's sessions live. The stop runs whatever
    happened inside, including a failure, so no test can leave a Houdini
    behind.
    """
    Path(home).mkdir(parents=True, exist_ok=True)
    started = HythonBridge(
        home=Path(home), port_range=port_range, alias=alias, extra_args=list(extra_args), env=env
    )
    started.start()
    try:
        yield started
    finally:
        started.stop(timeout_s=STOP_TIMEOUT_S)


def live_sessions(home: Path) -> list[str]:
    """The names of the sessions this state folder still has a file for."""
    entries = registry.list_entries(Path(home))
    return [str(entry.get("alias") or entry.get("session_id")) for entry in entries]


def stop_everything(home: Path, config: pool.PoolConfig) -> list[str]:
    """Stop every worker this state folder knows. Returns the ones that stayed."""
    if not pool.store_path(home).exists():
        return []
    left: list[str] = []
    with pool.open_store(home) as store:
        for record in store.list_workers():
            if record.pid is None:
                store.release_worker(record.token)
                continue
            stopped = pool.stop_worker(config, store, record.token)
            if not stopped.ended:
                left.append(record.alias)
    return left


# Section: several processes at once


def run_children(
    target: Callable[..., None],
    count: int,
    args: tuple[Any, ...] = (),
    *,
    barrier: bool = False,
    result_timeout_s: float = RESULT_TIMEOUT_S,
    join_timeout_s: float = JOIN_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Start `count` spawned children and collect what each one reports.

    Every child is called as `target(*args, index, barrier, results)`, where
    the barrier is `None` when none was asked for. A child that reports an
    error fails the run, and anything still alive at the end is ended here, so
    a child that hangs cannot hold up the suite.
    """
    context = mp.get_context("spawn")
    results = context.Queue()
    gate = context.Barrier(count) if barrier else None
    children = [
        context.Process(target=target, args=(*args, index, gate, results), daemon=True)
        for index in range(count)
    ]
    collected: list[dict[str, Any]] = []
    try:
        for child in children:
            child.start()
        for _ in children:
            collected.append(results.get(timeout=result_timeout_s))
        for child in children:
            child.join(join_timeout_s)
    except queue_module.Empty:
        raise AssertionError(f"only {len(collected)} of {count} children reported back") from None
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(join_timeout_s)
    failures = [report["error"] for report in collected if report.get("error")]
    assert failures == [], "a child failed:\n" + "\n".join(failures)
    collected.sort(key=lambda report: report["index"])
    return collected


# Section: breaking things on purpose


def kill_bridge(bridge: HythonBridge) -> int | None:
    """End a session outright, the way a crash ends one.

    No word on the input and no chance to tidy up: the process is ended by the
    system, so its session file and its store row are left exactly as a crash
    would leave them.
    """
    process = bridge.process
    if process is None:
        return None
    process.kill()
    return process.wait(timeout=STOP_TIMEOUT_S)


def call_and_die(
    home: str,
    handle: str,
    tool: str,
    arguments: Mapping[str, Any],
    sent_marker: str,
    index: int,
    barrier: Any,
    results: Any,
) -> None:
    """A caller that sends one call and is ended before the answer arrives.

    It writes the marker file just before it sends, so whoever started it knows
    when to end it, and reports nothing back: the point is that this process
    never gets to say anything.
    """
    session = client.Session.open(Path(home), handle)
    operation_id = client.new_operation_id()
    Path(sent_marker).write_text(operation_id, encoding="utf-8")
    answer = client.call(
        session,
        tool,
        arguments=dict(arguments),
        operation_id=operation_id,
        wait_s=30.0,
        timeout_s=120.0,
    )
    results.put({"index": index, "error": None, "status": answer.status})


@contextmanager
def client_that_dies_mid_call(
    home: Path,
    handle: str,
    *,
    tool: str,
    arguments: Mapping[str, Any],
    marker: Path,
    ready: Callable[[], Any],
) -> Iterator[Any]:
    """Start a caller, wait until its call is really running, then end it.

    `ready` is what says the call has reached the session, which is asked of
    the session itself rather than of the caller: the caller is about to be
    ended and cannot be trusted to report anything.
    """
    context = mp.get_context("spawn")
    results = context.Queue()
    child = context.Process(
        target=call_and_die,
        args=(str(home), handle, tool, dict(arguments), str(marker), 0, None, results),
        daemon=True,
    )
    child.start()
    try:
        wait_until(lambda: marker.is_file())
        wait_until(ready)
        child.kill()
        child.join(JOIN_TIMEOUT_S)
        assert not child.is_alive(), "the caller would not go"
        yield child
    finally:
        if child.is_alive():
            child.kill()
            child.join(JOIN_TIMEOUT_S)
