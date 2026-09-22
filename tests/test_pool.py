"""The worker pool, with a stand in for hython.

Nothing here starts a Houdini. The pool is handed a launcher that writes the
session file a real worker would write, so admission, the cap, a failed start,
reclaiming, leases, weights and the capability record can all be checked in
one process.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import pool
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import registry
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
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("a worker said something\n", encoding="utf-8")
        if self.fails:
            return pool.Launched(pid=os.getpid(), poll=lambda: 1)
        session_id = f"session-{len(self.sessions) + 1}"
        self.sessions.append(session_id)
        # The pid is this process, so every liveness check is true while the
        # test runs, which is what a live worker looks like.
        registry.write_entry(
            self.home,
            {
                "session_id": session_id,
                "alias": _alias(command),
                "kind": "hython",
                "pid": os.getpid(),
                "port": 18400 + len(self.sessions),
                "token": "not-a-real-token",
            },
        )
        return pool.Launched(pid=os.getpid())

    def probe(self, entry, **rest: Any) -> dict[str, Any]:
        return dict(CAPABILITIES)


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
        pool.start_worker(config, store, hython=hython, spawn=launcher.spawn, probe=refuse)
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
        assert store.get_worker(record.token).state == "stopped"


def test_a_worker_on_a_job_is_never_idle(home: Path, hython: Path) -> None:
    clock = FakeClock()
    with lease_store(home, clock) as store:
        config = pool.PoolConfig(home=home, max_idle_s=60.0)
        record = start(config, store, FakeLauncher(home), hython)
        pool.reserve(store, "w1", job_id="job-1")
        clock.tick(10_000.0)
        assert _watch_once(store, record.token, clock, threading.Event()) is None


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
        assert store.get_worker(record.token).state == "stopped"


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


def test_the_command_carries_the_token_and_the_idle_limit(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    record = start(config, store, launcher, hython)
    command = launcher.commands[0]
    assert command[0] == str(hython)
    assert command[1:3] == ["-m", pool.WORKER_MODULE]
    assert command[command.index("--worker-token") + 1] == record.token
    assert command[command.index("--max-idle-s") + 1] == str(config.max_idle_s)
    assert command[command.index("--alias") + 1] == "w1"


def test_each_worker_writes_its_own_log_under_the_state_folder(
    config: pool.PoolConfig, store: Store, hython: Path
) -> None:
    launcher = FakeLauncher(config.home)
    start(config, store, launcher, hython)
    assert launcher.logs[0] == config.home / "logs" / "worker-w1.log"
    assert launcher.logs[0].is_file()


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
