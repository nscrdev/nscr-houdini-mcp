"""The worker pool against a real Houdini.

Skipped, not failed, when there is no Houdini on this machine, which is the
case on the build machines. Two things are checked here that a stand in cannot
show: that a worker survives the process that started it, and that eight
processes asking for the last slot at once agree on one winner.

House rules. Never more than one real worker at a time, well under the cap.
Every worker started here is stopped here, and the module refuses to end while
one is still running. The ports are a range of this file's own, so a session
somebody started by hand keeps the port it has. Nothing here opens, saves or
looks at a scene file.

Children are started with the spawn method, which is the only one on every
supported system, so a child imports this module by name and calls the
function it was given. The runner that starts them is shared with the other
files that need a crowd, in `support`.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import support
from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.store import PoolFull, process_is_alive, process_start_stamp


def hython_available() -> bool:
    """What the pool itself finds, which is what these checks start."""
    try:
        pool.hython_path()
    except pool.HythonNotFound:
        return False
    return True


pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
]

# A range of this file's own, away from the one a bridge started by hand uses.
PORT_RANGE = support.POOL_PORTS

# Two of three slots are held before the children start, so there is exactly
# one left for all of them to fight over, and at most one real worker.
POOL_CAP = 3
RACERS = 8

BARRIER_TIMEOUT_S = support.BARRIER_TIMEOUT_S

# Long enough that a worker started for one check is never taken for idle
# while the check runs.
WARM_IDLE_S = 600.0

# The short lease, for the check that a worker nobody wants ends itself.
SHORT_IDLE_S = 5.0
IDLE_EXIT_TIMEOUT_S = 120.0


def config_for(home: Path, *, max_idle_s: float = WARM_IDLE_S) -> pool.PoolConfig:
    return pool.PoolConfig(
        home=home,
        cap=POOL_CAP,
        max_idle_s=max_idle_s,
        port_range=PORT_RANGE,
        start_timeout_s=240.0,
    )


# Section: what the children run


def ask_for_a_worker(home: str, hython: str, index: int, barrier, results) -> None:
    """One racer: ask for the last slot, and say what came of it."""
    report: dict[str, object] = {"index": index, "pid": os.getpid(), "error": None}
    try:
        config = config_for(Path(home))
        with pool.open_store(home) as store:
            barrier.wait(BARRIER_TIMEOUT_S)
            try:
                record = pool.start_worker(config, store, hython=hython)
                report["alias"] = record.alias
                report["session_id"] = record.session_id
            except PoolFull:
                report["alias"] = None
                report["full"] = True
    except BaseException as error:  # reported, so a failure reads as a message
        report["error"] = f"{type(error).__name__}: {error}"
    results.put(report)


def start_and_leave(home: str, hython: str, max_idle_s: float, index, barrier, results) -> None:
    """Start a worker and go away, which is what a client exit looks like."""
    report: dict[str, object] = {"index": index, "pid": os.getpid(), "error": None}
    try:
        config = config_for(Path(home), max_idle_s=max_idle_s)
        with pool.open_store(home) as store:
            record = pool.start_worker(config, store, hython=hython)
        report["alias"] = record.alias
        report["session_id"] = record.session_id
        report["token"] = record.token
        report["worker_pid"] = record.pid
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    results.put(report)


def run_children(target, count: int, args: tuple, *, barrier: bool) -> list[dict]:
    """Start `count` spawned children and collect their reports."""
    return support.run_children(target, count, args, barrier=barrier)


# Section: the home every check gets, and the promise to leave nothing running


@pytest.fixture
def home(tmp_path: Path) -> Iterator[Path]:
    folder = tmp_path / "home"
    folder.mkdir()
    yield folder
    left = support.stop_everything(folder, config_for(folder))
    assert left == [], f"workers were left running: {left}"


def health_of(home: Path, session_id: str):
    entry = registry.find_entry(home, session_id, remove_stale=False)
    assert entry is not None, f"no session file for {session_id}"
    return client.health(client.Session.from_entry(entry))


# Section: acceptance check 1, concurrent starts


def test_eight_processes_asking_at_once_agree_on_one_winner(home: Path) -> None:
    hython = pool.hython_path()
    with pool.open_store(home) as store:
        for slot in range(POOL_CAP - 1):
            store.reserve_worker(cap=POOL_CAP, token=f"held-{slot}")
    reports = run_children(ask_for_a_worker, RACERS, (str(home), str(hython)), barrier=True)

    assert len({report["pid"] for report in reports}) == RACERS
    winners = [report for report in reports if report.get("alias")]
    assert len(winners) == 1
    assert [report.get("full") for report in reports if not report.get("alias")] == [True] * (
        RACERS - 1
    )
    # The winner is a Houdini that is really there, and it outlived the child
    # process that asked for it.
    answer = health_of(home, str(winners[0]["session_id"]))
    assert answer.payload["data"]["status"] == "ok"
    with pool.open_store(home) as store:
        running = [record for record in store.list_workers() if record.pid is not None]
        assert len(running) == 1
        assert pool.worker_is_alive(running[0])


def failing_spawn(command, *, log: Path, env) -> pool.Launched:
    """A hython that is really started and comes up with no bridge in it."""
    return pool.spawn_detached([sys.executable, "-c", "raise SystemExit(3)"], log=log, env=env)


def test_a_failed_start_gives_its_slot_back(home: Path) -> None:
    config = config_for(home)
    with pool.open_store(home) as store:
        for slot in range(POOL_CAP - 1):
            store.reserve_worker(cap=POOL_CAP, token=f"held-{slot}")
        with pytest.raises(pool.WorkerStartFailed):
            pool.start_worker(config, store, hython=pool.hython_path(), spawn=failing_spawn)
        # The slot is the pool's again, and nothing is holding it.
        assert len(store.list_workers()) == POOL_CAP - 1
        record = store.reserve_worker(cap=POOL_CAP, token="after")
        assert record.alias
        store.release_worker("after")
    assert pool.log_path(home, "w3").is_file()


# Section: acceptance check 6, a worker outliving the process that started it


def test_a_worker_outlives_its_launcher_and_a_new_server_sees_it(home: Path) -> None:
    hython = pool.hython_path()
    [report] = run_children(
        start_and_leave, 1, (str(home), str(hython), WARM_IDLE_S), barrier=False
    )
    alias = str(report["alias"])
    session_id = str(report["session_id"])

    # The process that started it has gone. The worker has not.
    assert not process_is_alive(int(report["pid"]))
    assert health_of(home, session_id).payload["data"]["status"] == "ok"

    # A server that was not there when the worker started sees it as it is.
    with pool.open_store(home) as store:
        rows = pool.list_workers(store)
    assert [row["alias"] for row in rows] == [alias]
    assert rows[0]["state"] == "running"
    assert rows[0]["session_id"] == session_id
    assert rows[0]["capabilities"] != "-"

    # And that new server can drive it.
    with pool.open_store(home) as store:
        assert pool.reserve(store, alias, job_id="job-1").job_id == "job-1"
        assert pool.release(store, alias).job_id is None
        stopped = pool.stop_worker(config_for(home), store, alias)
    assert stopped.ended
    assert not stopped.killed


def test_a_worker_nobody_wants_ends_itself(home: Path) -> None:
    hython = pool.hython_path()
    [report] = run_children(
        start_and_leave, 1, (str(home), str(hython), SHORT_IDLE_S), barrier=False
    )
    token = str(report["token"])
    worker_pid = int(report["worker_pid"])

    deadline = time.monotonic() + IDLE_EXIT_TIMEOUT_S
    while time.monotonic() < deadline:
        with pool.open_store(home) as store:
            record = store.get_worker(token)
        if record is not None and record.state == "stopped":
            break
        time.sleep(1.0)
    assert record is not None and record.state == "stopped", "the idle worker held its slot"

    gone = time.monotonic() + 30.0
    while process_is_alive(worker_pid) and time.monotonic() < gone:
        time.sleep(0.5)
    assert not process_is_alive(worker_pid), "the idle worker is still running"
    # Its slot is free without anybody having tidied up after it.
    with pool.open_store(home) as store:
        assert store.list_workers() == []


def test_a_worker_that_will_not_go_is_ended(home: Path) -> None:
    """The last resort, with a process that is not listening for anything."""
    config = config_for(home)
    with pool.open_store(home) as store:
        record = store.reserve_worker(cap=POOL_CAP, token="stubborn")
        launched = pool.spawn_detached(
            [sys.executable, "-c", "import time; time.sleep(600)"],
            log=pool.log_path(home, record.alias),
            env=dict(os.environ),
        )
        store.set_worker_state(
            "stubborn",
            "running",
            pid=launched.pid,
            pid_start=process_start_stamp(launched.pid),
            owner_pid=launched.pid,
        )
        stopped = pool.stop_worker(config, store, "stubborn", grace_s=2.0, poll_s=0.25)
    assert stopped.killed
    assert stopped.ended
    assert stopped.record.state == "stopped"
