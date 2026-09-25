"""The worker pool, with a stand in for hython.

Nothing here starts a Houdini. The pool is handed a launcher that writes the
session file a real worker would write, so admission, the cap, a failed start,
reclaiming, leases, weights and the capability record can all be checked in
one process.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import pool
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.store import PoolFull, Store

CAPABILITIES = {
    "houdini_version": "22.0.368",
    "gui": False,
    "license": "Commercial",
    "renderers": ["husk", "karma"],
    "capture": ["opengl_rop"],
    "cancellation": True,
}


class FakeClock:
    """A clock that only moves when a test says so."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class FakeLauncher:
    """Stands in for hython. It writes what a worker writes and nothing else.

    `fails` makes the process exit before any bridge announces itself, which
    is the case that has to give its slot back.
    """

    def __init__(self, home: Path, *, fails: bool = False) -> None:
        self.home = home
        self.fails = fails
        self.commands: list[list[str]] = []
        self.logs: list[Path] = []
        self.envs: list[dict[str, str]] = []
        self.sessions: list[str] = []

    def spawn(self, command, *, log: Path, env) -> pool.Launched:
        self.commands.append(list(command))
        self.logs.append(log)
        self.envs.append(dict(env))
        pool.open_log(log)
        log.write_text("a worker said something\n", encoding="utf-8")
        if self.fails:
            return pool.Launched(pid=os.getpid(), poll=lambda: 1)
        session_id = f"session-{len(self.sessions) + 1}"
        self.sessions.append(session_id)
        write_entry(self.home, session_id, _alias(command), port=18400 + len(self.sessions))
        return pool.Launched(pid=os.getpid())

    def probe(self, entry, **rest: Any) -> dict[str, Any]:
        return dict(CAPABILITIES)


def write_entry(
    home: Path, session_id: str, alias: str, *, port: int, started_at: float | None = None
) -> Path:
    """The session file a worker writes when it is ready.

    The pid is this process, so every liveness check is true while the test
    runs, which is what a live worker looks like.
    """
    return registry.write_entry(
        home,
        {
            "session_id": session_id,
            "alias": alias,
            "kind": "hython",
            "pid": os.getpid(),
            "pid_start": store_module.process_start_stamp(),
            "port": port,
            "token": "not-a-real-token",
            "started_at": time.time() if started_at is None else started_at,
        },
    )


def _alias(command) -> str:
    parts = list(command)
    return parts[parts.index("--alias") + 1]


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


@pytest.fixture
def config(home: Path) -> pool.PoolConfig:
    return pool.PoolConfig(home=home, cap=2, max_idle_s=60.0)


@pytest.fixture
def store(home: Path) -> Iterator[Store]:
    with pool.open_store(home) as opened:
        yield opened


@pytest.fixture
def hython(tmp_path: Path) -> Path:
    named = tmp_path / "hython"
    named.write_text("", encoding="utf-8")
    return named


def start(config: pool.PoolConfig, store: Store, launcher: FakeLauncher, hython: Path, **rest):
    return pool.start_worker(
        config,
        store,
        hython=hython,
        spawn=launcher.spawn,
        probe=launcher.probe,
        **rest,
    )


# Section: admission


def test_a_worker_comes_up_and_records_what_it_is(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    record = start(config, store, launcher, hython)
    assert record.alias == "w1"
    assert record.state == "running"
    assert record.session_id == "session-1"
    assert record.pid == os.getpid()
    assert record.pid_start is not None
    assert record.capabilities == CAPABILITIES
    assert record.weight == pool.WEIGHTS["light"]


def test_a_worker_bound_to_its_server_records_that_lifetime(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)

    def spawn(command, *, log, env):
        launched = launcher.spawn(command, log=log, env=env)
        return pool.Launched(launched.pid, server_bound=True)

    record = pool.start_worker(config, store, hython=hython, spawn=spawn, probe=launcher.probe)
    assert record.capabilities["lifetime"] == "server"


def test_the_worker_owns_its_own_slot_once_it_is_up(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    """The server that started it may go away without taking the worker."""
    record = start(config, store, FakeLauncher(config.home), hython)
    assert record.owner_pid == record.pid


def test_the_cap_is_what_stops_a_fourth_worker(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    for _ in range(config.cap):
        start(config, store, launcher, hython)
    with pytest.raises(PoolFull):
        start(config, store, launcher, hython)
    assert [record.alias for record in store.list_workers()] == ["w1", "w2"]


def test_a_failed_start_gives_its_slot_back(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home, fails=True)
    with pytest.raises(pool.WorkerStartFailed):
        start(config, store, launcher, hython)
    assert store.list_workers() == []
    assert [record.state for record in store.list_workers(active_only=False)] == ["failed"]
    # And the slot really is free again.
    assert start(config, store, FakeLauncher(config.home), hython).alias == "w1"


def test_a_start_that_is_given_up_on_does_not_keep_the_slot(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    def refuse(entry, **rest: Any) -> dict[str, Any]:
        raise KeyboardInterrupt

    launcher = FakeLauncher(config.home)
    with pytest.raises(KeyboardInterrupt):
        pool.start_worker(config, store, hython=hython, spawn=ends(launcher.spawn), probe=refuse)
    assert store.list_workers() == []


def test_a_slot_held_by_a_process_that_is_gone_is_reclaimed_before_counting(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    """A crash must not shrink the pool for good."""
    store.reserve_worker(cap=config.cap, token="held")
    store.reserve_worker(cap=config.cap, token="crashed", owner_pid=_pid_that_is_gone())
    # The pool is full by the cap until the slot of the process that is gone
    # is counted for what it is.
    record = start(config, store, FakeLauncher(config.home), hython)
    assert record.alias == "w2"
    assert store.get_worker("crashed").state == "failed"


def test_a_worker_whose_process_has_gone_stops_holding_its_slot(
    config: pool.PoolConfig, store: Store
) -> None:
    store.reserve_worker(cap=config.cap, token="dead")
    store.set_worker_state("dead", "running", pid=_pid_that_is_gone(), pid_start="whenever")
    assert store.reclaim_workers() == ["dead"]


def _pid_that_is_gone() -> int:
    """A pid number nothing is using. High numbers wrap, so this is checked."""
    for number in range(999_000, 999_500):
        if not store_module.process_is_alive(number):
            return number
    raise AssertionError("every pid tried was in use")


# Section: weights


def test_a_heavy_worker_is_refused_when_the_budget_is_spent(
    home: Path, store: Store, hython: Path
) -> None:
    config = pool.PoolConfig(home=home, cap=4, weight_budget=4.0)
    launcher = FakeLauncher(home)
    start(config, store, launcher, hython, weight="light")
    start(config, store, launcher, hython, weight="light")
    with pytest.raises(PoolFull) as raised:
        start(config, store, launcher, hython, weight="heavy")
    assert "budget" in str(raised.value)
    # A light one still fits, so it is the weight that was refused, not a slot.
    assert start(config, store, launcher, hython, weight="light").alias == "w3"


def test_one_heavy_worker_fills_a_default_pool(home: Path, store: Store, hython: Path) -> None:
    config = pool.PoolConfig(home=home)
    launcher = FakeLauncher(home)
    start(config, store, launcher, hython, weight="heavy")
    with pytest.raises(PoolFull):
        start(config, store, launcher, hython, weight="heavy")


def test_an_unknown_weight_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError):
        pool.weight_of("enormous")


# Section: taking a worker and handing it back


def test_a_worker_is_held_by_its_job_and_handed_back_after_it(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    start(config, store, FakeLauncher(config.home), hython)
    taken = pool.reserve(store, "w1", job_id="job-1")
    assert (taken.state, taken.job_id) == ("leased", "job-1")
    back = pool.release(store, "w1")
    assert (back.state, back.job_id) == ("running", None)


def test_a_worker_on_a_job_is_not_offered_to_another(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    start(config, store, FakeLauncher(config.home), hython)
    pool.reserve(store, "w1", job_id="job-1")
    with pytest.raises(pool.WorkerBusy):
        pool.reserve(store, "w1", job_id="job-2")


def test_two_servers_that_both_picked_one_worker_end_with_one_owner(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    """Both look before either writes, which is how a lease used to be lost."""
    start(config, store, FakeLauncher(config.home), hython)
    with pool.open_store(config.home) as other:
        assert pool.find_worker(store, "w1").job_id is None
        assert pool.find_worker(other, "w1").job_id is None
        pool.reserve(store, "w1", job_id="job-1")
        with pytest.raises(pool.WorkerBusy):
            pool.reserve(other, "w1", job_id="job-2")
        assert pool.find_worker(other, "w1").job_id == "job-1"


def test_a_worker_started_for_a_job_says_who_took_it(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    record = start(config, store, FakeLauncher(config.home), hython, job_id="job-1")
    assert (record.state, record.job_id) == ("leased", "job-1")
    assert record.lessee_pid == os.getpid()


def test_a_job_held_by_a_server_that_died_does_not_hold_the_worker(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    record = start(config, store, FakeLauncher(config.home), hython)
    store.lease_worker(record.token, job_id="job-1", lessee_pid=_pid_that_is_gone())
    rows = pool.list_workers(store)
    assert (rows[0]["state"], rows[0]["job"]) == ("running", "-")
    # And it is a warm worker again, free for the next job.
    assert pool.reserve(store, "w1", job_id="job-2").job_id == "job-2"


def test_a_name_nothing_answers_to_says_so(store: Store) -> None:
    with pytest.raises(pool.UnknownWorker):
        pool.reserve(store, "w9", job_id="job-1")


# Section: the idle lease


def lease_store(home: Path, clock: FakeClock) -> Store:
    return pool.open_store(home, clock=clock)


def test_an_idle_worker_ends_itself_when_nobody_has_wanted_it(home: Path, hython: Path) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        clock.tick(59.0)
        stop = threading.Event()
        assert _watch_once(store, record.token, clock, stop) is None
        clock.tick(2.0)
        assert _watch_once(store, record.token, clock, stop) == "idle"
        # The slot stays taken until the process is about to exit.
        assert store.get_worker(record.token).state == "stopping"
        pool.release_on_exit(home, record.token)
        assert store.get_worker(record.token).state == "stopped"


@pytest.mark.parametrize("state", ["reserved", "starting"])
def test_startup_does_not_spend_the_workers_idle_lease(home: Path, state: str) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        record = store.reserve_worker(cap=1, token="cold", start_budget_s=180)
        store.set_worker_state(record.token, state)
        clock.tick(90)
        assert pool._lease_pass(store, record.token, max_idle_s=5, clock=clock) is None
        store.set_worker_state(record.token, "running")
        clock.tick(4)
        assert pool._lease_pass(store, record.token, max_idle_s=5, clock=clock) is None
        clock.tick(2)
        assert pool._lease_pass(store, record.token, max_idle_s=5, clock=clock) == "idle"


def test_an_unfinished_start_does_not_keep_a_worker_forever(home: Path) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        record = store.reserve_worker(cap=1, token="unfinished", start_budget_s=180)
        store.set_worker_state(record.token, "starting")
        clock.tick(181)
        assert pool._lease_pass(store, record.token, max_idle_s=5, clock=clock) == "idle"


def test_a_worker_on_a_job_is_never_idle(home: Path, hython: Path) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        pool.reserve(store, "w1", job_id="job-1")
        clock.tick(10_000.0)
        assert _watch_once(store, record.token, clock, threading.Event()) is None


def test_a_worker_running_a_call_is_never_idle_however_long_it_runs(
    home: Path, hython: Path
) -> None:
    clock = FakeClock()
    running = [True]
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        clock.tick(31 * 60.0)
        stop = threading.Event()
        stop.set()
        busy = pool.watch_lease(
            store,
            record.token,
            max_idle_s=60.0,
            stop=stop,
            interval_s=0.0,
            clock=clock,
            busy=lambda: running[0],
        )
        assert busy == "stopped"
        assert store.get_worker(record.token).state == "running"
        running[0] = False
        assert _watch_once(store, record.token, clock, threading.Event()) == "idle"


def test_a_job_on_the_worker_row_keeps_it_and_its_end_starts_the_idle_wait(
    home: Path, hython: Path
) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        assert store.hold_worker_for_job(record.session_id, "job-a") is True
        clock.tick(31 * 60.0)
        assert _watch_once(store, record.token, clock, threading.Event()) is None
        assert store.renew_worker_of_session(record.session_id) is True
        assert store.get_worker(record.token).leased_at == clock.now
        clock.tick(30.0)
        assert store.free_worker_of_job(record.session_id, "job-a") is True
        assert store.get_worker(record.token).job_id is None
        clock.tick(59.0)
        assert _watch_once(store, record.token, clock, threading.Event()) is None
        clock.tick(2.0)
        assert _watch_once(store, record.token, clock, threading.Event()) == "idle"


def test_a_job_a_server_took_the_worker_for_is_not_replaced(home: Path, hython: Path) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        pool.reserve(store, "w1", job_id="job-server")
        assert store.hold_worker_for_job(record.session_id, "job-a") is False
        assert store.free_worker_of_job(record.session_id, "job-a") is False
        assert store.get_worker(record.token).job_id == "job-server"
        assert store.hold_worker_for_job("no-such-session", "job-a") is False


def test_touching_the_lease_keeps_a_worker_alive(home: Path, hython: Path) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        clock.tick(59.0)
        pool.touch(store, "w1")
        clock.tick(59.0)
        assert _watch_once(store, record.token, clock, threading.Event()) is None


def test_a_worker_asked_to_stop_reads_that_from_its_own_row(home: Path, hython: Path) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        store.set_worker_state(record.token, "stopping")
        assert _watch_once(store, record.token, clock, threading.Event()) == "asked"
        # Still stopping, and still counted, while the process is there.
        assert store.get_worker(record.token).state == "stopping"
        assert _watch_once(store, record.token, clock, threading.Event()) == "asked"
        assert [w.token for w in store.list_workers()] == [record.token]
        pool.release_on_exit(home, record.token)
        assert store.get_worker(record.token).state == "stopped"


def test_a_stopping_worker_is_never_taken_for_a_job(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    record = start(config, store, FakeLauncher(config.home), hython)
    store.set_worker_state(record.token, "stopping")
    with pytest.raises(store_module.WorkerTaken):
        store.lease_worker(record.token, job_id="job-1")
    with pytest.raises(pool.UnknownWorker):
        pool.reserve(store, record.alias, job_id="job-1")
    assert pool.find_worker(store, record.alias, include_stopping=True).token == record.token


def test_a_row_that_has_gone_ends_the_watch(home: Path, hython: Path) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        store.release_worker(record.token)
        assert _watch_once(store, record.token, clock, threading.Event()) == "gone"


class StoreThatTrips:
    """A store that fails a read once, then behaves."""

    def __init__(self, store: Store, failures: int = 1) -> None:
        self._store = store
        self.left = failures

    def get_worker(self, token: str):
        if self.left > 0:
            self.left -= 1
            raise store_module.StoreBusy("the store is locked by another process")
        return self._store.get_worker(token)

    def __getattr__(self, name: str):
        return getattr(self._store, name)


def test_one_store_hiccup_does_not_stop_a_worker_watching_its_lease(
    home: Path, hython: Path
) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        tripping = StoreThatTrips(store)
        lines: list[str] = []
        stop = threading.Event()
        # The pass that failed is given up on, not the watch. The next tick
        # reads the row and finds the worker has been asked to stop.
        store.set_worker_state(record.token, "stopping")
        reason = pool.watch_lease(
            tripping,
            record.token,
            max_idle_s=60.0,
            stop=stop,
            interval_s=0.0,
            clock=clock,
            log=lines.append,
        )
    assert reason == "asked"
    assert tripping.left == 0
    assert any("could not be read" in line for line in lines)


def test_a_store_that_stays_unreadable_ends_the_watch_with_a_reason(
    home: Path, hython: Path
) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        broken = StoreThatTrips(store, failures=1000)
        lines: list[str] = []

        class Waiting:
            """Stands in for the stop flag, and moves the clock while waiting."""

            def wait(self, _seconds: float) -> bool:
                clock.tick(120.0)
                return False

        reason = pool.watch_lease(
            broken,
            record.token,
            max_idle_s=60.0,
            stop=Waiting(),
            interval_s=0.0,
            clock=clock,
            trouble_s=600.0,
            log=lines.append,
        )
    assert reason == "unreadable"
    assert any("600" in line for line in lines)


def _watch_once(store: Store, token: str, clock: FakeClock, stop: threading.Event) -> str | None:
    """One pass of the lease watch. `None` means it would have gone on waiting."""
    stop.set()
    reason = pool.watch_lease(store, token, max_idle_s=60.0, stop=stop, interval_s=0.0, clock=clock)
    stop.clear()
    return None if reason == "stopped" else reason


# Section: what the worker is started with


def test_the_worker_is_told_where_to_find_the_package_and_the_state(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    start(config, store, launcher, hython)
    env = launcher.envs[0]
    assert str(Path(pool.__file__).resolve().parents[1]) in env["PYTHONPATH"]
    assert env[store_module.HOME_ENV_VAR] == str(config.home)
    assert "HOUDINI_MAXTHREADS" not in env


def test_a_thread_cap_reaches_the_worker(home: Path, store: Store, hython: Path) -> None:
    config = pool.PoolConfig(home=home, max_threads=4)
    launcher = FakeLauncher(home)
    start(config, store, launcher, hython)
    assert launcher.envs[0]["HOUDINI_MAXTHREADS"] == "4"


def test_the_command_says_what_is_not_a_secret(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    start(config, store, launcher, hython)
    command = launcher.commands[0]
    assert command[0] == str(hython)
    assert command[1:3] == ["-m", pool.WORKER_MODULE]
    assert command[command.index("--max-idle-s") + 1] == str(config.max_idle_s)
    assert command[command.index("--alias") + 1] == "w1"


def test_the_token_never_appears_on_the_command_line(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    """Every account on the machine can read a command line."""
    launcher = FakeLauncher(config.home)
    record = start(config, store, launcher, hython)
    assert record.token not in launcher.commands[0]
    assert not any(record.token in part for part in launcher.commands[0])
    assert launcher.envs[0][pool.TOKEN_ENV_VAR] == record.token


def test_each_worker_writes_its_own_log_under_the_state_folder(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    start(config, store, launcher, hython)
    assert launcher.logs[0] == config.home / "logs" / "worker-w1.log"
    assert launcher.logs[0].is_file()


def test_a_session_file_left_by_a_crash_is_not_taken_for_the_new_worker(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    """It can name the same process and the same name, and its port is free."""
    write_entry(config.home, "session-old", "w1", port=18499, started_at=time.time() - 3600.0)
    launcher = FakeLauncher(config.home)
    record = start(config, store, launcher, hython, timeout_s=5.0)
    assert record.session_id == "session-1"


def test_a_worker_that_never_writes_its_file_is_a_failed_start(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    def say_nothing(command, *, log: Path, env) -> pool.Launched:
        pool.open_log(log)
        return pool.Launched(pid=os.getpid())

    with pytest.raises(pool.WorkerStartFailed):
        pool.start_worker(
            config,
            store,
            hython=hython,
            spawn=ends(say_nothing),
            probe=lambda entry: {},
            timeout_s=0.5,
        )
    assert store.list_workers() == []


def ends(spawn: Any) -> Any:
    """A stand in whose process ends when its handle is told to, as a real one does."""

    def spawn_with_a_handle(command, *, log: Path, env) -> pool.Launched:
        launched = spawn(command, log=log, env=env)
        exit_code: list[int] = []
        return pool.Launched(
            pid=launched.pid,
            poll=lambda: exit_code[0] if exit_code else None,
            kill=lambda: exit_code.append(-9),
        )

    return spawn_with_a_handle


def test_a_start_that_fails_after_the_spawn_ends_the_process(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    """A real process stands in for hython, so there is something to end."""
    started: list[pool.Launched] = []

    def spawn(command, *, log: Path, env) -> pool.Launched:
        launched = pool.spawn_detached(
            [sys.executable, "-c", "import time; time.sleep(120)"], log=log
        )
        started.append(launched)
        registry.write_entry(
            config.home,
            {
                "session_id": "session-spawned",
                "alias": _alias(command),
                "kind": "hython",
                "pid": launched.pid,
                "pid_start": store_module.process_start_stamp(launched.pid),
                "port": 18490,
                "token": "not-a-real-token",
                "started_at": time.time(),
            },
        )
        return launched

    def refuse(entry, **rest: Any) -> dict[str, Any]:
        raise RuntimeError("the store went away")

    try:
        with pytest.raises(RuntimeError) as caught:
            pool.start_worker(config, store, hython=hython, spawn=spawn, probe=refuse)
        [launched] = started
        assert caught.value.spawned_pid == launched.pid
        assert caught.value.spawned_ended is True
        assert launched.poll() is not None
        assert not registry.entry_path(config.home, "session-spawned").exists()
        assert store.list_workers() == []
    finally:
        for launched in started:
            if launched.poll() is None:
                pool.kill_process(launched.pid, store_module.process_start_stamp(launched.pid))
        pool.reap_started()


def test_a_process_that_cannot_be_shown_to_be_the_one_started_is_left_and_said_so(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    """The stand in names this process, which is never ended from here."""

    def refuse(entry, **rest: Any) -> dict[str, Any]:
        raise RuntimeError("probe failed")

    with pytest.raises(RuntimeError) as caught:
        pool.start_worker(
            config, store, hython=hython, spawn=FakeLauncher(config.home).spawn, probe=refuse
        )
    assert caught.value.spawned_pid == os.getpid()
    assert caught.value.spawned_ended is False
    # The process is still there, so its slot is still taken, by a row that
    # names it for the reaper.
    [kept] = store.list_workers()
    assert kept.state == "stopping"
    assert kept.pid == os.getpid()


def test_a_worker_that_did_not_end_keeps_its_slot(config: pool.PoolConfig, store: Store) -> None:
    """With no start stamp the process cannot be shown to be the worker, so it
    is neither ended nor counted as gone, and its slot stays taken."""
    single = pool.PoolConfig(home=config.home, cap=1)
    store.reserve_worker(cap=1, token="kept")
    store.set_worker_state("kept", "running", session_id="session-kept", pid=os.getpid())
    stopped = pool.stop_worker(single, store, "kept", grace_s=0.1, poll_s=0.05)
    assert stopped.ended is False
    assert stopped.killed is False
    assert stopped.note
    assert store.get_worker("kept").state == "stopping"
    with pytest.raises(PoolFull):
        store.reserve_worker(cap=1, token="another")


def test_a_probe_that_could_not_be_sent_keeps_the_worker(
    config: pool.PoolConfig, store: Store, hython: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that is up and did not answer one read is still a worker."""

    def refuse(*args: Any, **rest: Any):
        raise client.BridgeUnreachable("nothing answered on that port")

    monkeypatch.setattr(pool.client, "call", refuse)
    record = pool.start_worker(config, store, hython=hython, spawn=FakeLauncher(config.home).spawn)
    assert record.state == "running"
    assert "did not answer" in record.capabilities["probe"]


# Section: the environment a worker is given


def test_a_light_worker_is_left_at_houdini_s_own_thread_default(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    start(config, store, launcher, hython, weight="light")
    assert pool.THREADS_ENV_VAR not in launcher.envs[0]


def test_an_inherited_thread_cap_does_not_reach_a_light_worker(config: pool.PoolConfig) -> None:
    given = pool.worker_env(config, weight=1.0, base={pool.THREADS_ENV_VAR: "2"})
    assert pool.THREADS_ENV_VAR not in given


def test_a_worker_draws_offscreen_unless_its_shell_says_otherwise(
    config: pool.PoolConfig,
) -> None:
    assert pool.worker_env(config, base={})[pool.QT_PLATFORM_ENV_VAR] == "offscreen"
    kept = pool.worker_env(config, base={pool.QT_PLATFORM_ENV_VAR: "xcb"})
    assert kept[pool.QT_PLATFORM_ENV_VAR] == "xcb"


def test_a_heavy_worker_is_given_the_machine(home: Path, store: Store, hython: Path) -> None:
    config = pool.PoolConfig(home=home)
    launcher = FakeLauncher(home)
    start(config, store, launcher, hython, weight="heavy")
    assert launcher.envs[0][pool.THREADS_ENV_VAR] == str(os.cpu_count() or 1)


def test_a_configured_thread_cap_beats_the_weight(home: Path) -> None:
    config = pool.PoolConfig(home=home, max_threads=4)
    assert pool.thread_cap(config, pool.WEIGHTS["heavy"]) == 4


# Section: ending a worker that will not end itself


def test_a_pid_that_cannot_be_shown_to_be_this_worker_is_not_killed(
    config: pool.PoolConfig, store: Store
) -> None:
    """Numbers are handed out again, so the wrong process is never ended."""
    store.reserve_worker(cap=config.cap, token="t1")
    # A row with no start stamp: the pid is running, and nothing says it is
    # still the process that was started under it.
    store.set_worker_state("t1", "running", pid=os.getpid())
    stopped = pool.stop_worker(config, store, "t1", grace_s=0.0, poll_s=0.0)
    assert stopped.killed is False
    assert stopped.ended is False
    assert "could not be shown" in stopped.note
    assert pool.kill_process(os.getpid(), None) is False
    assert pool.kill_process(os.getpid(), "not-when-this-started") is False


def test_the_session_file_goes_when_the_process_has(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    record = start(config, store, launcher, hython)
    # It went by itself, cleanly or not, and left its file behind.
    store.set_worker_state(record.token, "running", pid=_pid_that_is_gone(), pid_start="whenever")
    stopped = pool.stop_worker(config, store, record.alias, grace_s=0.0, poll_s=0.0)
    assert stopped.ended
    assert stopped.killed is False
    assert registry.find_entry(config.home, record.session_id, remove_stale=False) is None


# Section: the log a worker writes


def test_a_log_is_private_and_rolled_over_when_it_grows(home: Path) -> None:
    path = pool.log_path(home, "w1")
    pool.open_log(path)
    path.write_text("x" * 100, encoding="utf-8")
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    pool.open_log(path, max_bytes=50, keep=2)
    rolled = path.with_suffix(".1.log")
    assert rolled.read_text(encoding="utf-8") == "x" * 100
    assert path.read_text(encoding="utf-8") == ""
    # The oldest is dropped rather than kept for ever.
    path.write_text("y" * 100, encoding="utf-8")
    pool.open_log(path, max_bytes=50, keep=2)
    assert path.with_suffix(".2.log").read_text(encoding="utf-8") == "x" * 100
    assert rolled.read_text(encoding="utf-8") == "y" * 100
    path.write_text("z" * 100, encoding="utf-8")
    pool.open_log(path, max_bytes=50, keep=2)
    assert not path.with_suffix(".3.log").exists()


# Section: hython lookup


def test_a_configured_hython_is_used_as_given(hython: Path) -> None:
    assert pool.hython_path(hython) == hython


def test_a_configured_hython_that_is_not_there_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(pool.HythonNotFound):
        pool.hython_path(tmp_path / "missing")


def test_the_installs_on_the_machine_are_what_is_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(pool.HYTHON_ENV_VAR, raising=False)
    hfs = tmp_path / "Houdini22.0.1" / "hfs"
    binary = hfs / "bin" / pool._executable("hython")
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")
    found = [install_module_stub(hfs)]
    monkeypatch.setattr(pool.install_module, "find_installs", lambda *a, **k: found)
    assert pool.hython_path() == binary


def test_nowhere_to_look_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pool.HYTHON_ENV_VAR, raising=False)
    monkeypatch.setattr(pool.install_module, "find_installs", lambda *a, **k: [])
    with pytest.raises(pool.HythonNotFound):
        pool.hython_path()


def install_module_stub(hfs: Path):
    return pool.install_module.HoudiniInstall(version="22.0.1", root=hfs.parent, hfs=hfs)


# Section: what a person is shown


def test_a_listing_names_what_a_person_needs_to_pick_a_worker(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    start(config, store, FakeLauncher(config.home), hython)
    pool.reserve(store, "w1", job_id="job-1")
    row = pool.list_workers(store)[0]
    assert row["alias"] == "w1"
    assert row["session_id"] == "session-1"
    assert row["pid"] == os.getpid()
    assert row["state"] == "leased"
    assert row["job"] == "job-1"
    assert row["lease_age_s"] >= 0.0
    assert "22.0.368" in row["capabilities"]
    assert "karma" in row["capabilities"]


def test_a_worker_nothing_was_read_about_still_lists(store: Store) -> None:
    store.reserve_worker(cap=2, token="bare")
    row = pool.list_workers(store)[0]
    assert row["capabilities"] == "-"
    assert row["pid"] == "-"


def test_a_worker_keeps_an_autostart_package_out(config: pool.PoolConfig) -> None:
    given = pool.worker_env(config, base={"NSCR_MCP_AUTOSTART": "1"})
    assert given[install_module.NO_AUTOSTART_ENV_VAR] == "1"
