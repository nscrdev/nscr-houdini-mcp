from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.store import (
    CLEAR,
    AliasInUse,
    DuplicateRecord,
    JobIdTaken,
    JobMoveRefused,
    OperationMismatch,
    PoolFull,
    SceneReplaced,
    SchemaTooNew,
    Store,
    StoreBusy,
    StoreError,
    UndigestableArgument,
    UnknownRecord,
    WorkerTaken,
    default_home,
    default_store_path,
    digest_arguments,
    process_is_alive,
    shared_location_warning,
    write_export,
)
from nscr_houdini_mcp.tools.sessions import ended_state

DEAD_PID = 2**22 - 1  # Above every system's pid range, so never a live process.
LIVE_PID = os.getpid()  # A pid that is certainly running, on every system.


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def step(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "coord.sqlite") as opened:
        yield opened


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def timed_store(tmp_path: Path, clock: FakeClock) -> Store:
    with Store(tmp_path / "coord.sqlite", clock=clock) as opened:
        yield opened


# -- location and schema --------------------------------------------------


def test_home_follows_the_environment_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(store_module.HOME_ENV_VAR, str(tmp_path / "elsewhere"))
    assert default_home() == tmp_path / "elsewhere"
    assert default_store_path() == tmp_path / "elsewhere" / "coord.sqlite"


def test_home_without_an_override_is_a_per_user_folder(monkeypatch) -> None:
    monkeypatch.delenv(store_module.HOME_ENV_VAR, raising=False)
    home = default_home()
    assert home.name == store_module.APP_DIR_NAME
    assert home.is_absolute()


def test_open_creates_the_file_in_wal_mode_at_the_current_schema(tmp_path) -> None:
    path = tmp_path / "nested" / "coord.sqlite"
    with Store(path) as opened:
        assert path.is_file()
        assert opened.schema_version() == store_module.SCHEMA_VERSION
        mode = opened._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


def test_opening_an_existing_file_leaves_records_alone(tmp_path) -> None:
    path = tmp_path / "coord.sqlite"
    with Store(path) as first:
        first.register_session("s1", kind="gui", pid=1, alias="scene-1")
    with Store(path) as second:
        assert second.schema_version() == store_module.SCHEMA_VERSION
        assert second.resolve_session("scene-1").session_id == "s1"


def test_a_file_from_an_older_schema_is_migrated_forward(tmp_path) -> None:
    path = tmp_path / "coord.sqlite"
    sqlite3.connect(str(path)).close()
    with Store(path) as opened:
        assert opened.schema_version() == store_module.SCHEMA_VERSION
        assert opened.list_sessions() == []


def test_a_file_from_a_newer_build_is_refused(tmp_path) -> None:
    path = tmp_path / "coord.sqlite"
    with Store(path):
        pass
    raw = sqlite3.connect(str(path))
    raw.execute(f"PRAGMA user_version={store_module.SCHEMA_VERSION + 1}")
    raw.close()
    with pytest.raises(SchemaTooNew):
        Store(path)


def test_a_synced_or_network_path_is_worth_a_line_not_a_failure(tmp_path) -> None:
    assert shared_location_warning(tmp_path / "coord.sqlite") is None
    assert "network" in shared_location_warning("\\\\server\\share\\coord.sqlite")
    assert "synced" in shared_location_warning(Path.home() / "Dropbox" / "coord.sqlite")
    with Store(tmp_path / "coord.sqlite") as opened:
        assert opened.location_warning is None


def test_transactions_do_not_nest(store: Store) -> None:
    with store._txn(write=True):
        with pytest.raises(StoreError):
            with store._txn(write=True):
                pass


def test_a_blocked_start_leaves_the_handle_usable(tmp_path) -> None:
    """A transaction that never began must not look like one that is open."""
    path = tmp_path / "coord.sqlite"
    with Store(path) as holder, Store(path, busy_timeout_s=0.05) as waiter:
        waiter.register_session("s1", kind="gui", pid=os.getpid(), alias="scene-1")
        with holder._txn(write=True):
            with pytest.raises(StoreBusy):
                waiter.register_session("s2", kind="gui", pid=os.getpid(), alias="scene-2")
        # The holder has committed, so the waiter carries on as normal.
        assert (
            waiter.register_session("s2", kind="gui", pid=os.getpid(), alias="scene-2").alias
            == "scene-2"
        )
        assert [s.session_id for s in waiter.list_sessions()] == ["s1", "s2"]


def test_a_duplicate_id_comes_back_as_a_store_error(store: Store) -> None:
    store.register_session("s1", kind="gui", pid=1, alias="scene-1")
    store.end_session("s1")
    with pytest.raises(DuplicateRecord):
        store.register_session("s1", kind="gui", pid=1, alias="scene-2")


# -- process liveness -----------------------------------------------------


def test_liveness_knows_this_process_from_a_pid_that_is_not_running() -> None:
    assert process_is_alive(os.getpid()) is True
    assert process_is_alive(DEAD_PID) is False
    assert process_is_alive(None) is False
    assert process_is_alive(0) is False


def test_liveness_sees_a_child_exit() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    assert process_is_alive(child.pid) is False


needs_exit_watch = pytest.mark.skipif(
    sys.platform != "darwin", reason="only this system reads start stamps from a program"
)


@needs_exit_watch
def test_a_start_stamp_is_read_once_while_its_process_runs(monkeypatch) -> None:
    known = store_module._KnownStarts()
    reads: list[int] = []
    real = store_module._ps_start

    def counted(pid: int) -> str | None:
        reads.append(pid)
        return real(pid)

    monkeypatch.setattr(store_module, "_ps_start", counted)
    first = known.stamp(os.getpid())
    assert first and first == real(os.getpid())
    assert known.stamp(os.getpid()) == first
    assert reads == [os.getpid()]


@needs_exit_watch
def test_a_kept_start_stamp_is_dropped_when_its_process_exits(monkeypatch) -> None:
    known = store_module._KnownStarts()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        stamp = known.stamp(child.pid)
        assert stamp
        # A stamp read while the process ran is not trusted past its exit, even
        # when the listing would now name some other process under the pid.
        monkeypatch.setattr(store_module, "_ps_start", lambda pid: "someone else")
        assert known.stamp(child.pid) == stamp
    finally:
        child.kill()
        child.wait(timeout=30)
    # The kernel says no process has the pid now, so there is no stamp, and
    # the listing is not asked.
    assert known.stamp(child.pid) is None
    assert child.pid not in known._kept


class FakeWatch:
    """A watch on one process's exit, fired by the test."""

    def __init__(self) -> None:
        self.exited = False
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_kept_start_stamps_drop_ended_processes_first_then_the_oldest(monkeypatch) -> None:
    watches: dict[int, FakeWatch] = {}

    def watch(pid: int) -> FakeWatch:
        watches[pid] = FakeWatch()
        return watches[pid]

    monkeypatch.setattr(store_module, "_watch_exit", watch)
    monkeypatch.setattr(store_module, "_has_exited", lambda made: made.exited)
    monkeypatch.setattr(store_module, "_ps_start", lambda pid: f"stamp {pid}")
    known = store_module._KnownStarts()
    known.LIMIT = 4
    for pid in (1, 2, 3, 4):
        assert known.stamp(pid) == f"stamp {pid}"
    # Two of them end. The next new process takes their room, and the live
    # ones are kept, the oldest included.
    watches[2].exited = True
    watches[3].exited = True
    known.stamp(5)
    assert list(known._kept) == [1, 4, 5]
    assert watches[2].closed and watches[3].closed
    assert not watches[1].closed
    # Full with live processes: the oldest goes, not the newest.
    known.stamp(6)
    known.stamp(7)
    assert list(known._kept) == [4, 5, 6, 7]
    assert watches[1].closed
    # A long run of short lived processes takes one room, not every room:
    # the first of them pushes out the oldest, and each ended one makes way.
    for pid in range(100, 200):
        known.stamp(pid)
        watches[pid].exited = True
    assert [pid for pid in known._kept if pid < 100] == [5, 6, 7]


def watching(monkeypatch, listing, watches: dict[int, list[FakeWatch]]) -> None:
    """Stand in watches, kept in `watches`, and a listing the test writes."""

    def watch(pid: int) -> FakeWatch:
        made = FakeWatch()
        watches.setdefault(pid, []).append(made)
        return made

    monkeypatch.setattr(store_module, "_watch_exit", watch)
    monkeypatch.setattr(store_module, "_has_exited", lambda made: made.exited)
    monkeypatch.setattr(store_module, "_ps_start", listing)


def test_a_process_that_ends_while_it_is_read_gives_no_stamp(monkeypatch) -> None:
    watches: dict[int, list[FakeWatch]] = {}

    def listing(pid: int) -> str:
        # The watched process exits while the listing is being read.
        watches[pid][-1].exited = True
        return "the old stamp"

    watching(monkeypatch, listing, watches)
    known = store_module._KnownStarts()
    assert known.stamp(7) is None
    assert 7 not in known._kept
    assert all(made.closed for made in watches[7])
    assert len(watches[7]) == store_module._KnownStarts.TRIES


def test_a_pid_taken_again_while_it_is_read_is_read_under_a_new_watch(monkeypatch) -> None:
    said = iter(["the old stamp", "the new stamp"])
    watches: dict[int, list[FakeWatch]] = {}

    def listing(pid: int) -> str:
        if len(watches[pid]) == 1:
            watches[pid][-1].exited = True
        return next(said)

    watching(monkeypatch, listing, watches)
    known = store_module._KnownStarts()
    assert known.stamp(7) == "the new stamp"
    assert known._kept[7][0] == "the new stamp"


@needs_exit_watch
def test_a_pid_with_no_process_gives_no_stamp() -> None:
    assert store_module._KnownStarts().stamp(DEAD_PID) is None


def test_one_read_per_pid_is_under_way_and_the_rest_wait_for_it(monkeypatch) -> None:
    reads: list[int] = []

    def listing(pid: int) -> str:
        reads.append(pid)
        time.sleep(0.05)
        return f"stamp {pid}"

    watching(monkeypatch, listing, {})
    known = store_module._KnownStarts()
    answers: list[str | None] = []
    threads = [threading.Thread(target=lambda: answers.append(known.stamp(7))) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert answers == ["stamp 7"] * 20
    assert reads == [7]


def test_only_a_few_watches_are_open_for_reads_at_once(monkeypatch) -> None:
    lock = threading.Lock()
    open_now = [0, 0]  # now, most

    class Counted(FakeWatch):
        def __init__(self) -> None:
            super().__init__()
            with lock:
                open_now[0] += 1
                open_now[1] = max(open_now[1], open_now[0])

    def listing(pid: int) -> str:
        time.sleep(0.05)
        return f"stamp {pid}"

    monkeypatch.setattr(store_module, "_watch_exit", lambda pid: Counted())
    monkeypatch.setattr(store_module, "_has_exited", lambda made: made.exited)
    monkeypatch.setattr(store_module, "_ps_start", listing)
    known = store_module._KnownStarts()
    real_keep = known._keep

    def keep(pid: int, stamp: str, watch: Any) -> None:
        # A kept watch is no longer one in flight.
        with lock:
            open_now[0] -= 1
        real_keep(pid, stamp, watch)

    known._keep = keep  # type: ignore[method-assign]
    answers: dict[int, str | None] = {}
    threads = [
        threading.Thread(target=lambda pid=pid: answers.update({pid: known.stamp(pid)}))
        for pid in range(40)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    # Every pid has its stamp, whether it was read under a watch or without.
    assert answers == {pid: f"stamp {pid}" for pid in range(40)}
    assert open_now[1] <= store_module._KnownStarts.READING


def test_same_process_tells_a_reused_pid_apart_after_the_first_look() -> None:
    stamp = store_module.process_start_stamp()
    if not stamp:
        pytest.skip("this system gives no start stamp")
    assert store_module.same_process(os.getpid(), stamp) is True
    assert store_module.same_process(os.getpid(), stamp) is True
    assert store_module.same_process(os.getpid(), "a different stamp") is False


# -- sessions -------------------------------------------------------------


def test_register_and_resolve_by_id_or_alias(store: Store) -> None:
    record = store.register_session(
        "s1", kind="gui", pid=42, port=9000, alias="shot-1", hip_path="/scenes/shot.hip"
    )
    assert record.alias == "shot-1"
    assert record.scene_epoch == 0
    assert store.resolve_session("s1") == record
    assert store.resolve_session("shot-1") == record
    assert store.resolve_session("nothing") is None


def test_capabilities_survive_a_round_trip(store: Store) -> None:
    store.register_session(
        "s1", kind="hython", pid=1, alias="w1", capabilities={"gui": False, "routes": ["rop"]}
    )
    assert store.get_session("s1").capabilities == {"gui": False, "routes": ["rop"]}


def test_an_alias_cannot_be_taken_twice_while_it_is_live(store: Store) -> None:
    store.register_session("s1", kind="gui", pid=LIVE_PID, alias="shot-1")
    with pytest.raises(AliasInUse):
        store.register_session("s2", kind="gui", pid=LIVE_PID, alias="shot-1")


def test_an_alias_template_takes_the_lowest_free_name(store: Store) -> None:
    first = store.register_session("s1", kind="gui", pid=LIVE_PID, alias_template="shot-{n}")
    second = store.register_session("s2", kind="gui", pid=LIVE_PID, alias_template="shot-{n}")
    assert [first.alias, second.alias] == ["shot-1", "shot-2"]
    store.end_session("s1")
    third = store.register_session("s3", kind="gui", pid=LIVE_PID, alias_template="shot-{n}")
    assert third.alias == "shot-1"


def test_a_renamed_session_takes_the_scene_file_and_keeps_its_old_name(store: Store) -> None:
    store.register_session(
        "s1", kind="gui", pid=LIVE_PID, alias_template="untitled-{n}", hip_path="/u/untitled.hip"
    )
    store.register_session("s2", kind="gui", pid=LIVE_PID, alias="shot-1")

    renamed = store.rename_session("s1", alias_template="shot-{n}", hip_path="/s/shot.hip")

    assert renamed.alias == "shot-2"
    assert renamed.previous_alias == "untitled-1"
    assert renamed.hip_path == "/s/shot.hip"
    assert store.resolve_session("shot-2").session_id == "s1"
    # The old name still finds it, because nobody else can have it meanwhile.
    assert store.resolve_session("untitled-1").session_id == "s1"
    # Its own names do not count against it.
    again = store.rename_session("s1", alias_template="shot-{n}")
    assert (again.alias, again.previous_alias) == ("shot-2", "untitled-1")


def test_a_renamed_session_holds_its_old_name_until_it_ends(store: Store) -> None:
    """A caller that read the old name must never reach another Houdini by it."""
    store.register_session("s1", kind="gui", pid=LIVE_PID, alias_template="untitled-{n}")
    store.rename_session("s1", alias_template="shot-{n}")

    second = store.register_session("s2", kind="gui", pid=LIVE_PID, alias_template="untitled-{n}")
    assert second.alias == "untitled-2"
    with pytest.raises(AliasInUse):
        store.register_session("s3", kind="gui", pid=LIVE_PID, alias="untitled-1")
    assert store.rename_session("s2", alias_template="untitled-{n}").alias == "untitled-2"

    store.end_session("s1")
    later = store.register_session("s4", kind="gui", pid=LIVE_PID, alias_template="untitled-{n}")
    assert later.alias == "untitled-1"


def test_a_session_that_is_gone_or_unknown_cannot_be_renamed(store: Store) -> None:
    store.register_session("s1", kind="gui", pid=LIVE_PID, alias="shot-1")
    store.end_session("s1")
    for session_id in ("s1", "nope"):
        with pytest.raises(UnknownRecord):
            store.rename_session(session_id, alias_template="other-{n}")


def test_a_name_held_by_a_session_that_crashed_is_free_again(store: Store) -> None:
    """A crash cannot end its own row, so the next start ends it instead."""
    store.register_session("s1", kind="hython", pid=DEAD_PID, alias="w1")

    restarted = store.register_session("s2", kind="hython", pid=LIVE_PID, alias="w1")

    assert restarted.alias == "w1"
    assert store.get_session("s1").state == "gone"
    assert store.resolve_session("w1").session_id == "s2"


def test_a_name_is_not_taken_from_a_session_whose_process_is_still_there(store: Store) -> None:
    """A pid and the moment it started is the identity, not the pid alone."""
    stamp = store_module.process_start_stamp(LIVE_PID)
    store.register_session("s1", kind="hython", pid=LIVE_PID, pid_start=stamp, alias="w1")

    with pytest.raises(AliasInUse):
        store.register_session("s2", kind="hython", pid=LIVE_PID, alias="w1")
    assert store.get_session("s1").pid_start == stamp


def test_a_pid_that_belongs_to_another_process_now_frees_the_name(store: Store) -> None:
    """The number is alive, but it is not the process that took the name."""
    store.register_session(
        "s1", kind="hython", pid=LIVE_PID, pid_start="a moment that has passed", alias="w1"
    )

    restarted = store.register_session("s2", kind="hython", pid=LIVE_PID, alias="w1")

    assert restarted.session_id == "s2"
    assert store.get_session("s1").state == "gone"


def test_a_session_from_an_older_file_keeps_its_name_while_its_pid_is_alive(tmp_path) -> None:
    """A row written before the stamp existed still answers the question."""
    path = tmp_path / "coord.sqlite"
    with Store(path) as before:
        before.register_session("s1", kind="hython", pid=LIVE_PID, alias="w1")
    with Store(path) as after:
        assert after.schema_version() == store_module.SCHEMA_VERSION
        assert after.get_session("s1").pid_start is None
        with pytest.raises(AliasInUse):
            after.register_session("s2", kind="hython", pid=LIVE_PID, alias="w1")


def test_a_file_from_before_sessions_kept_how_they_ended_reads_right(tmp_path) -> None:
    """A schema 6 file with rows in it, opened by a build at schema 7."""
    path = tmp_path / "coord.sqlite"
    raw = sqlite3.connect(str(path))
    for statements in store_module.MIGRATIONS[:6]:
        for statement in statements:
            raw.execute(statement)
    raw.execute("PRAGMA user_version=6")
    insert = (
        "INSERT INTO sessions (session_id, alias, kind, pid, pid_start, port, state,"
        " scene_epoch, hip_path, capabilities, started_at, heartbeat_at)"
        " VALUES (?, ?, 'hython', ?, 'then', 18000, ?, 0, NULL, NULL, 1.0, 1.0)"
    )
    raw.execute(insert, ("s-old", "w1", DEAD_PID, "gone"))
    raw.execute(insert, ("s-dead", "w2", DEAD_PID, "live"))
    raw.commit()
    raw.close()

    with Store(path) as after:
        assert after.schema_version() == store_module.SCHEMA_VERSION
        assert after.get_session("s-old").ended_as is None
        assert after.get_session("s-dead").ended_as is None
        assert after.reclaim_sessions() == ["s-dead"]
        assert after.get_session("s-dead").ended_as == "crashed"
        assert after.get_session("s-old").ended_as is None
        # A row that ended before the store kept how lists as gone.
        assert ended_state(after.get_session("s-old")) == "gone"
        assert ended_state(after.get_session("s-dead")) == "crashed"


def test_sessions_whose_process_is_gone_can_be_tidied_up_on_their_own(store: Store) -> None:
    store.register_session("s1", kind="hython", pid=LIVE_PID, alias="w1")
    store.register_session("s2", kind="hython", pid=DEAD_PID, alias="w2")

    assert store.reclaim_sessions() == ["s2"]
    assert [record.session_id for record in store.list_sessions()] == ["s1"]
    assert store.reclaim_sessions() == []


def test_a_session_keeps_how_it_ended(store: Store) -> None:
    store.register_session("s1", kind="hython", pid=LIVE_PID, alias="w1")
    store.register_session("s2", kind="hython", pid=DEAD_PID, alias="w2")
    assert store.get_session("s1").ended_as is None

    store.end_session("s1")
    store.reclaim_sessions()
    assert store.get_session("s1").ended_as == "gone"
    assert store.get_session("s2").ended_as == "crashed"
    assert store.get_session("s2").state == "gone"

    # A stop that had to end the process says it was on purpose, whoever
    # found the process missing first.
    store.end_session("s2", how="gone")
    assert store.get_session("s2").ended_as == "gone"
    with pytest.raises(ValueError):
        store.end_session("s1", how="vanished")


def test_register_wants_exactly_one_of_alias_or_template(store: Store) -> None:
    with pytest.raises(ValueError):
        store.register_session("s1", kind="gui", pid=1)
    with pytest.raises(ValueError):
        store.register_session("s1", kind="gui", pid=1, alias="a", alias_template="a-{n}")


def test_unknown_session_kind_and_state_are_refused(store: Store) -> None:
    with pytest.raises(ValueError):
        store.register_session("s1", kind="mystery", pid=1, alias="a")
    with pytest.raises(ValueError):
        store.register_session("s2", kind="gui", pid=1, alias="b", state="mystery")


def test_a_session_id_stays_readable_after_the_process_is_gone(store: Store) -> None:
    store.register_session("s1", kind="gui", pid=1, alias="shot-1")
    store.end_session("s1")
    assert store.get_session("s1").state == "gone"
    assert store.resolve_session("shot-1") is None
    assert store.list_sessions() == []
    assert [s.session_id for s in store.list_sessions(include_gone=True)] == ["s1"]


def test_heartbeat_and_state_move_together(store: Store) -> None:
    started = store.register_session("s1", kind="gui", pid=1, alias="shot-1")
    beat = store.touch_session("s1", state="busy")
    record = store.get_session("s1")
    assert record.state == "busy"
    assert record.heartbeat_at == beat
    assert beat >= started.heartbeat_at


def test_a_heartbeat_carries_what_the_session_found_on_its_own_port(store: Store) -> None:
    """A beating heart says the process is running and nothing more."""
    store.register_session("s1", kind="gui", pid=1, alias="shot-1")
    record = store.get_session("s1")
    assert record.transport_ok is None
    assert record.transport_checked_at is None

    store.touch_session("s1", transport_ok=True)
    record = store.get_session("s1")
    assert record.transport_ok is True
    assert record.transport_checked_at is not None
    assert record.state == "live"

    store.touch_session("s1", state="unresponsive", transport_ok=False)
    record = store.get_session("s1")
    assert record.transport_ok is False
    assert record.state == "unresponsive"

    # A beat that says nothing about the port leaves what was there.
    store.touch_session("s1")
    assert store.get_session("s1").transport_ok is False


def test_scene_epoch_counts_up_and_records_the_new_hip(store: Store) -> None:
    store.register_session("s1", kind="gui", pid=1, alias="shot-1")
    assert store.bump_scene_epoch("s1") == 1
    assert store.bump_scene_epoch("s1", hip_path="/scenes/other.hip") == 2
    assert store.get_session("s1").hip_path == "/scenes/other.hip"


def test_a_session_can_write_the_epoch_it_says_it_is_on(store: Store) -> None:
    """The session that owns the scene owns the count."""
    store.register_session("s1", kind="gui", pid=LIVE_PID, alias="shot-1")

    assert store.set_scene_epoch("s1", 4, hip_path="/scenes/other.hip") == 4
    record = store.get_session("s1")
    assert record.scene_epoch == 4
    assert record.hip_path == "/scenes/other.hip"
    with pytest.raises(UnknownRecord):
        store.set_scene_epoch("nope", 1)


def test_session_helpers_report_an_unknown_id(store: Store) -> None:
    for call in (
        lambda: store.touch_session("nope"),
        lambda: store.bump_scene_epoch("nope"),
        lambda: store.end_session("nope"),
    ):
        with pytest.raises(UnknownRecord):
            call()


# -- workers --------------------------------------------------------------


def test_reservations_fill_the_pool_and_then_refuse(store: Store) -> None:
    first = store.reserve_worker(cap=2, token="t1")
    second = store.reserve_worker(cap=2, token="t2")
    assert [first.alias, second.alias] == ["w1", "w2"]
    assert first.state == "reserved"
    assert first.owner_pid == os.getpid()
    with pytest.raises(PoolFull):
        store.reserve_worker(cap=2, token="t3")


def test_a_starting_worker_still_holds_its_slot(store: Store) -> None:
    store.reserve_worker(cap=1, token="t1")
    store.set_worker_state("t1", "starting")
    with pytest.raises(PoolFull):
        store.reserve_worker(cap=1, token="t2")


def test_a_failed_start_gives_the_slot_back(store: Store) -> None:
    store.reserve_worker(cap=1, token="t1")
    store.release_worker("t1", state="failed")
    replacement = store.reserve_worker(cap=1, token="t2")
    assert replacement.alias == "w1"
    assert store.get_worker("t1").state == "failed"


def test_a_slot_held_by_a_process_that_died_is_reclaimed(store: Store) -> None:
    store.reserve_worker(cap=1, token="crashed", owner_pid=DEAD_PID)
    store.set_worker_state("crashed", "running", session_id="s1")
    assert store.reclaim_workers() == ["crashed"]
    assert store.get_worker("crashed").state == "failed"
    assert store.reserve_worker(cap=1, token="fresh").alias == "w1"


def test_reserving_reclaims_before_it_counts(store: Store) -> None:
    store.reserve_worker(cap=1, token="crashed", owner_pid=DEAD_PID)
    assert store.reserve_worker(cap=1, token="fresh").alias == "w1"
    assert store.get_worker("crashed").state == "failed"


def test_a_reservation_that_never_starts_runs_out_of_time(timed_store, clock) -> None:
    timed_store.reserve_worker(cap=1, token="stuck", start_budget_s=30.0)
    clock.step(29.0)
    with pytest.raises(PoolFull):
        timed_store.reserve_worker(cap=1, token="waiting")
    clock.step(2.0)
    assert timed_store.reserve_worker(cap=1, token="waiting").alias == "w1"
    assert timed_store.get_worker("stuck").state == "failed"


def test_a_running_worker_is_not_reclaimed_for_taking_its_time(timed_store, clock) -> None:
    timed_store.reserve_worker(cap=1, token="t1", start_budget_s=30.0)
    timed_store.set_worker_state("t1", "running", job_id="j1")
    clock.step(10_000.0)
    assert timed_store.reclaim_workers() == []
    assert timed_store.get_worker("t1").state == "running"


def test_worker_state_moves_carry_the_session_and_job(store: Store) -> None:
    store.reserve_worker(cap=2, token="t1")
    record = store.set_worker_state("t1", "running", session_id="s9", job_id="j1")
    assert (record.state, record.session_id, record.job_id) == ("running", "s9", "j1")
    kept = store.set_worker_state("t1", "leased")
    assert (kept.session_id, kept.job_id) == ("s9", "j1")


def test_a_finished_job_can_be_cleared_from_its_worker(store: Store) -> None:
    store.reserve_worker(cap=2, token="t1", job_id="j1")
    freed = store.set_worker_state("t1", "leased", job_id=CLEAR)
    assert freed.job_id is None
    assert freed.session_id is None
    store.set_worker_state("t1", "leased", session_id="s1")
    assert store.set_worker_state("t1", "leased", session_id=CLEAR).session_id is None


def test_releasing_a_worker_clears_the_job_it_held(store: Store) -> None:
    store.reserve_worker(cap=2, token="t1", job_id="j1")
    assert store.release_worker("t1").job_id is None


def test_a_worker_records_the_process_it_is_and_what_it_can_do(store: Store) -> None:
    store.reserve_worker(cap=2, token="t1")
    record = store.set_worker_state(
        "t1",
        "running",
        pid=4321,
        pid_start="whenever",
        capabilities={"renderers": ["husk"]},
    )
    assert (record.pid, record.pid_start) == (4321, "whenever")
    assert record.capabilities == {"renderers": ["husk"]}
    # What is not named again keeps what it held.
    assert store.set_worker_state("t1", "leased").capabilities == {"renderers": ["husk"]}


def test_a_worker_whose_own_process_has_gone_is_reclaimed(store: Store) -> None:
    """Its slot comes back although whoever started it is still running."""
    store.reserve_worker(cap=1, token="t1")
    store.set_worker_state("t1", "running", pid=DEAD_PID, pid_start="whenever")
    assert store.reclaim_workers() == ["t1"]
    assert store.reserve_worker(cap=1, token="t2").alias == "w1"


def test_weights_are_refused_past_the_budget_although_a_slot_is_free(store: Store) -> None:
    store.reserve_worker(cap=4, token="t1", weight=2.0, weight_budget=3.0)
    with pytest.raises(PoolFull):
        store.reserve_worker(cap=4, token="t2", weight=2.0, weight_budget=3.0)
    # The budget is the caller's, and a lighter job still fits.
    assert store.reserve_worker(cap=4, token="t3", weight=1.0, weight_budget=3.0).weight == 1.0
    with pytest.raises(ValueError):
        store.reserve_worker(cap=4, token="t4", weight=0.0)


def test_a_released_weight_is_not_counted_any_more(store: Store) -> None:
    store.reserve_worker(cap=4, token="t1", weight=3.0, weight_budget=3.0)
    store.release_worker("t1")
    assert store.reserve_worker(cap=4, token="t2", weight=3.0, weight_budget=3.0).weight == 3.0


def test_only_one_of_two_servers_takes_the_same_warm_worker(tmp_path: Path) -> None:
    """Both look, both decide, and the write is what settles it."""
    path = tmp_path / "coord.sqlite"
    with Store(path) as one, Store(path) as two:
        one.reserve_worker(cap=2, token="t1")
        one.set_worker_state("t1", "running")
        # Both read a worker with no job on it, which is the interleaving.
        assert one.get_worker("t1").job_id is None
        assert two.get_worker("t1").job_id is None
        assert one.lease_worker("t1", job_id="job-1").job_id == "job-1"
        with pytest.raises(WorkerTaken):
            two.lease_worker("t1", job_id="job-2")
        assert two.get_worker("t1").job_id == "job-1"
        # The same job asking again is the same claim, not a second one.
        assert one.lease_worker("t1", job_id="job-1").job_id == "job-1"


def test_taking_a_worker_records_which_process_took_it(store: Store) -> None:
    store.reserve_worker(cap=2, token="t1")
    record = store.lease_worker("t1", job_id="job-1")
    assert record.lessee_pid == os.getpid()
    assert record.lessee_start is not None
    # Handing it back takes the holder off with the job.
    freed = store.set_worker_state("t1", "running", job_id=CLEAR)
    assert (freed.lessee_pid, freed.lessee_start) == (None, None)


def test_a_lease_held_by_a_server_that_died_is_handed_back(store: Store) -> None:
    store.reserve_worker(cap=2, token="t1")
    store.set_worker_state("t1", "running", pid=os.getpid(), owner_pid=os.getpid())
    store.lease_worker("t1", job_id="job-1", lessee_pid=DEAD_PID, lessee_start="whenever")
    # The worker is fine, so its slot stays. The claim on it does not.
    assert store.reclaim_workers() == []
    record = store.get_worker("t1")
    assert (record.state, record.job_id, record.lessee_pid) == ("running", None, None)


def test_a_lease_whose_holder_cannot_be_asked_about_is_left_alone(store: Store) -> None:
    store.reserve_worker(cap=2, token="t1")
    store.set_worker_state("t1", "running", pid=os.getpid(), owner_pid=os.getpid())
    store.lease_worker("t1", job_id="job-1", lessee_pid=os.getpid())
    store.reclaim_workers()
    assert store.get_worker("t1").job_id == "job-1"


def test_leasing_needs_a_worker_that_is_there(store: Store) -> None:
    with pytest.raises(UnknownRecord):
        store.lease_worker("borrowed", job_id="job-1")
    store.reserve_worker(cap=2, token="t1")
    store.release_worker("t1")
    with pytest.raises(WorkerTaken):
        store.lease_worker("t1", job_id="job-1")


def test_worker_moves_need_a_known_token_and_a_known_state(store: Store) -> None:
    store.reserve_worker(cap=1, token="t1")
    with pytest.raises(UnknownRecord):
        store.set_worker_state("borrowed", "running")
    with pytest.raises(ValueError):
        store.set_worker_state("t1", "mystery")
    with pytest.raises(ValueError):
        store.release_worker("t1", state="running")


def test_listing_shows_active_reservations_only_by_default(store: Store) -> None:
    store.reserve_worker(cap=3, token="t1")
    store.reserve_worker(cap=3, token="t2")
    store.release_worker("t2")
    assert [w.token for w in store.list_workers()] == ["t1"]
    assert {w.token for w in store.list_workers(active_only=False)} == {"t1", "t2"}


def test_an_idle_lease_expires_but_a_worker_on_a_job_never_does(timed_store, clock) -> None:
    timed_store.reserve_worker(cap=3, token="idle")
    timed_store.reserve_worker(cap=3, token="busy", job_id="j1")
    clock.step(1_800.0)
    assert [w.token for w in timed_store.idle_workers(max_idle_s=600.0)] == ["idle"]
    timed_store.touch_worker_lease("idle")
    assert timed_store.idle_workers(max_idle_s=600.0) == []
    with pytest.raises(UnknownRecord):
        timed_store.touch_worker_lease("gone")


def test_a_clock_that_steps_backwards_cannot_rewind_a_lease(timed_store, clock) -> None:
    timed_store.reserve_worker(cap=3, token="idle")
    clock.step(-3_600.0)
    assert timed_store.idle_workers(max_idle_s=0.0) == [timed_store.get_worker("idle")]
    assert timed_store.idle_workers(max_idle_s=1.0) == []


def test_a_worker_whose_owner_is_gone_is_not_called_idle(store: Store) -> None:
    store.reserve_worker(cap=3, token="crashed", owner_pid=DEAD_PID)
    assert store.idle_workers(max_idle_s=0.0) == []
    assert store.reclaim_workers() == ["crashed"]


def test_a_cap_below_one_is_a_mistake(store: Store) -> None:
    with pytest.raises(ValueError):
        store.reserve_worker(cap=0, token="t1")


# -- argument digests -----------------------------------------------------


def test_argument_digests_ignore_key_order_and_notice_a_change() -> None:
    assert digest_arguments({"a": 1, "b": [2, 3]}) == digest_arguments({"b": [2, 3], "a": 1})
    assert digest_arguments({"a": 1}) != digest_arguments({"a": 2})


def test_numbers_that_cross_a_json_transport_digest_the_same() -> None:
    assert digest_arguments({"sx": 1}) == digest_arguments({"sx": 1.0})
    assert digest_arguments([1, 2]) == digest_arguments([1.0, 2.0])
    assert digest_arguments({"sx": 1}) != digest_arguments({"sx": 1.5})
    assert digest_arguments({"on": True}) != digest_arguments({"on": 1})


def test_collections_and_paths_have_one_stable_form() -> None:
    assert digest_arguments({"g": {"a", "b"}}) == digest_arguments({"g": {"b", "a"}})
    assert digest_arguments({"g": ("a", "b")}) == digest_arguments({"g": ["a", "b"]})
    assert digest_arguments({"p": Path("a/b")}) == digest_arguments({"p": "a/b"})


def test_a_value_with_no_stable_text_form_is_refused() -> None:
    with pytest.raises(UndigestableArgument):
        digest_arguments({"node": object()})
    with pytest.raises(UndigestableArgument):
        digest_arguments({"n": float("nan")})
    with pytest.raises(UndigestableArgument):
        digest_arguments({1: "a"})


def test_digests_match_across_processes_with_different_hash_seeds() -> None:
    """Set iteration order changes with the seed, the digest must not."""
    code = (
        "import sys; sys.path.insert(0, 'src');"
        " from nscr_houdini_mcp.store import digest_arguments;"
        " print(digest_arguments({'g': {'a', 'b', 'c'}, 'f': frozenset({1, 2}), 'sx': 1.0}))"
    )
    digests = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        digests.add(done.stdout.strip())
    assert len(digests) == 1


# -- operation receipts ---------------------------------------------------


def test_a_retry_with_the_same_digest_gets_the_stored_outcome(store: Store) -> None:
    digest = digest_arguments({"node": "/obj/box", "parm": "sx"})
    claim = store.begin_operation("op1", digest, session_id="s1", scene_epoch=3)
    assert claim.claimed is True
    assert claim.outcome_unknown is False
    assert claim.record.state == "running"

    store.finish_operation("op1", outcome={"created": ["/obj/box"]})
    again = store.begin_operation("op1", digest, session_id="s1", scene_epoch=3)
    assert again.claimed is False
    assert again.outcome_unknown is False
    assert again.record.state == "done"
    assert again.record.outcome == {"created": ["/obj/box"]}


def test_the_same_id_with_other_arguments_is_a_mismatch(store: Store) -> None:
    store.begin_operation("op1", digest_arguments({"parm": "sx"}))
    with pytest.raises(OperationMismatch):
        store.begin_operation("op1", digest_arguments({"parm": "sy"}))


def test_the_same_id_from_another_session_is_a_mismatch(store: Store) -> None:
    digest = digest_arguments({"parm": "sx"})
    store.begin_operation("op1", digest, session_id="s1")
    with pytest.raises(OperationMismatch):
        store.begin_operation("op1", digest, session_id="s2")


def test_a_retry_against_a_replaced_scene_says_so(store: Store) -> None:
    digest = digest_arguments({"parm": "sx"})
    store.begin_operation("op1", digest, session_id="s1", scene_epoch=3)
    store.finish_operation("op1", outcome={"created": ["/obj/box"]})
    with pytest.raises(SceneReplaced) as raised:
        store.begin_operation("op1", digest, session_id="s1", scene_epoch=4)
    assert raised.value.recorded_epoch == 3
    assert raised.value.current_epoch == 4


def test_a_receipt_left_behind_by_a_dead_process_is_taken_over(store: Store) -> None:
    digest = digest_arguments({"parm": "sx"})
    store.begin_operation("op1", digest, owner_pid=DEAD_PID)
    claim = store.begin_operation("op1", digest)
    assert claim.claimed is True
    assert claim.outcome_unknown is True
    assert claim.record.owner_pid == os.getpid()


def test_a_receipt_somebody_else_is_working_on_is_not_taken_over(timed_store, clock) -> None:
    digest = digest_arguments({"parm": "sx"})
    timed_store.begin_operation("op1", digest, lease_s=300.0)
    claim = timed_store.begin_operation("op1", digest, lease_s=300.0)
    assert claim.claimed is False
    assert claim.outcome_unknown is True

    clock.step(301.0)
    expired = timed_store.begin_operation("op1", digest, lease_s=300.0)
    assert expired.claimed is True
    assert expired.outcome_unknown is True


def test_a_lease_can_be_renewed_while_the_work_runs(timed_store, clock) -> None:
    digest = digest_arguments({"parm": "sx"})
    timed_store.begin_operation("op1", digest, lease_s=300.0)
    clock.step(290.0)
    timed_store.touch_operation("op1")
    clock.step(100.0)
    assert timed_store.begin_operation("op1", digest, lease_s=300.0).claimed is False
    with pytest.raises(UnknownRecord):
        timed_store.touch_operation("missing")


def test_a_failed_operation_keeps_its_error(store: Store) -> None:
    digest = digest_arguments({"parm": "sx"})
    store.begin_operation("op1", digest)
    store.finish_operation("op1", state="failed", error={"code": "PARM_NOT_FOUND"})
    stored = store.begin_operation("op1", digest)
    assert stored.claimed is False
    assert (stored.record.state, stored.record.error) == ("failed", {"code": "PARM_NOT_FOUND"})


def test_an_operation_can_point_at_a_job(store: Store) -> None:
    store.begin_operation("op1", digest_arguments({}))
    record = store.finish_operation("op1", outcome={"state": "running"}, job_id="j1")
    assert record.job_id == "j1"
    assert store.get_operation("op1").job_id == "j1"
    assert store.get_operation("missing") is None


def test_finishing_an_unknown_operation_is_refused(store: Store) -> None:
    with pytest.raises(UnknownRecord):
        store.finish_operation("op1")
    store.begin_operation("op2", "d")
    with pytest.raises(ValueError):
        store.finish_operation("op2", state="mystery")


def test_old_receipts_are_pruned(store: Store) -> None:
    store.begin_operation("op1", "d")
    assert store.prune_operations(max_age_s=600) == 0
    assert store.prune_operations(max_age_s=-1) == 1
    assert store.get_operation("op1") is None


# -- jobs -----------------------------------------------------------------


def test_a_job_records_the_scene_it_consumes(store: Store) -> None:
    scene = {"snapshot": "$HIP/.agent/jobs/j1/shot.hip", "hash": "abc", "scene_epoch": 2}
    record = store.create_job("j1", kind="render", session_id="s1", weight="heavy", scene=scene)
    assert (record.state, record.weight, record.scene) == ("queued", "heavy", scene)
    assert record.finished_at is None
    assert record.cancel_requested is False


def test_job_updates_keep_the_fields_they_are_not_given(store: Store) -> None:
    store.create_job("j1", kind="render")
    store.update_job("j1", state="running", progress={"frames_done": 1, "frames": 10})
    record = store.update_job("j1", outputs=["/renders/a.exr"])
    assert record.state == "running"
    assert record.progress == {"frames_done": 1, "frames": 10}
    assert record.outputs == ["/renders/a.exr"]
    assert record.finished_at is None


def test_a_final_state_stamps_the_finish_time(store: Store) -> None:
    store.create_job("j1", kind="cook")
    record = store.update_job("j1", state="done")
    assert record.finished_at is not None
    assert record.finished_at >= record.created_at


def test_a_cancel_request_is_a_flag_the_runner_reads(store: Store) -> None:
    store.create_job("j1", kind="render")
    assert store.request_job_cancel("j1").cancel_requested is True
    assert store.get_job("j1").cancel_requested is True
    with pytest.raises(UnknownRecord):
        store.request_job_cancel("missing")


def test_a_job_whose_runner_went_quiet_or_died_is_a_candidate_for_lost(timed_store, clock) -> None:
    timed_store.create_job("quiet", kind="render", state="running", worker_pid=os.getpid())
    timed_store.create_job("crashed", kind="render", state="running", worker_pid=DEAD_PID)
    timed_store.create_job("finished", kind="render", worker_pid=DEAD_PID)
    timed_store.update_job("finished", state="done")

    assert [job.job_id for job in timed_store.stale_jobs(max_silence_s=60.0)] == ["crashed"]
    clock.step(120.0)
    assert {job.job_id for job in timed_store.stale_jobs(max_silence_s=60.0)} == {
        "quiet",
        "crashed",
    }
    timed_store.touch_job("quiet")
    assert [job.job_id for job in timed_store.stale_jobs(max_silence_s=60.0)] == ["crashed"]
    with pytest.raises(UnknownRecord):
        timed_store.touch_job("missing")


def test_jobs_are_listed_newest_first_and_can_be_filtered(store: Store) -> None:
    store.create_job("j1", kind="render", session_id="s1")
    store.create_job("j2", kind="render", session_id="s2", state="running")
    assert [j.job_id for j in store.list_jobs()] == ["j2", "j1"]
    assert [j.job_id for j in store.list_jobs(session_id="s1")] == ["j1"]
    assert [j.job_id for j in store.list_jobs(states=["running"])] == ["j2"]
    assert [j.job_id for j in store.list_jobs(limit=1)] == ["j2"]


def test_unknown_jobs_and_states_are_refused(store: Store) -> None:
    with pytest.raises(UnknownRecord):
        store.update_job("j1", state="done")
    with pytest.raises(ValueError):
        store.create_job("j2", kind="render", state="mystery")
    store.create_job("j3", kind="render")
    with pytest.raises(ValueError):
        store.update_job("j3", state="mystery")


def test_a_job_keeps_its_operation_what_it_runs_and_when_it_began(store: Store) -> None:
    queued = store.create_job(
        "j1", kind="python", operation_id="op-1", spec={"namespace": "shared"}
    )
    assert (queued.operation_id, queued.spec, queued.started_at) == (
        "op-1",
        {"namespace": "shared"},
        None,
    )
    running = store.update_job("j1", state="running", scene={"hip_path": "/p/a.hip"})
    assert running.started_at is not None
    assert running.scene == {"hip_path": "/p/a.hip"}
    again = store.update_job("j1", state="running")
    assert again.started_at == running.started_at


def test_a_job_id_is_taken_again_only_once_its_job_ended_long_ago(timed_store, clock) -> None:
    timed_store.create_job("j1", kind="python", state="running")
    with pytest.raises(JobIdTaken):
        timed_store.create_job("j1", kind="python", replace_after_s=60.0)
    timed_store.update_job("j1", state="done")
    clock.step(30.0)
    with pytest.raises(JobIdTaken):
        timed_store.create_job("j1", kind="python", replace_after_s=60.0)
    with pytest.raises(DuplicateRecord):
        timed_store.create_job("j1", kind="python")
    clock.step(31.0)
    fresh = timed_store.create_job("j1", kind="python", replace_after_s=60.0)
    assert (fresh.state, fresh.finished_at) == ("queued", None)


def test_a_job_moves_only_the_way_the_rules_allow(store: Store) -> None:
    store.create_job("j1", kind="python")
    store.update_job("j1", state="running")
    with pytest.raises(JobMoveRefused):
        store.update_job("j1", state="queued")
    record = store.update_job("j1", state="failed", error={"code": "X"})
    assert record.state == "failed"
    with pytest.raises(JobMoveRefused):
        store.update_job("j1", state="done")
    with pytest.raises(JobMoveRefused):
        store.update_job("j1", state="done", late=True)


def test_an_ended_job_takes_no_more_progress_and_keeps_when_it_ended(timed_store, clock) -> None:
    timed_store.create_job("j1", kind="python", state="running")
    ended = timed_store.update_job("j1", state="done", progress={"done": 1})
    clock.step(10.0)
    after = timed_store.update_job("j1", progress={"done": 9}, outputs={"late": True})
    assert (after.progress, after.outputs, after.finished_at) == (
        {"done": 1},
        None,
        ended.finished_at,
    )
    timed_store.touch_job("j1")
    assert timed_store.get_job("j1").heartbeat_at == ended.heartbeat_at
    assert timed_store.beat_job("j1", progress={"done": 5}).progress == {"done": 1}
    assert timed_store.get_job("j1").updated_at == ended.updated_at


def test_a_late_finish_over_lost_takes_the_real_ending_and_clears_the_loss(
    timed_store, clock
) -> None:
    timed_store.create_job("j1", kind="python", state="running")
    [lost] = timed_store.lose_jobs(["j1"], error=store_module.SESSION_ENDED_ERROR)
    assert lost.error == store_module.SESSION_ENDED_ERROR
    with pytest.raises(JobMoveRefused):
        timed_store.update_job("j1", state="done")
    clock.step(5.0)
    late = timed_store.update_job("j1", state="done", outputs={"answer": 1}, late=True)
    assert (late.state, late.error, late.outputs) == ("done", None, {"answer": 1})
    assert late.finished_at == lost.finished_at + 5.0
    failed = timed_store.create_job("j2", kind="python", state="running")
    timed_store.lose_jobs([failed.job_id])
    record = timed_store.update_job("j2", state="failed", error={"type": "E"}, late=True)
    assert record.error == {"type": "E"}


def test_starting_a_job_moves_only_a_queued_one(store: Store) -> None:
    store.create_job("j1", kind="python")
    started = store.start_job("j1", scene={"hip_path": "/p/a.hip"})
    assert (started.state, started.scene) == ("running", {"hip_path": "/p/a.hip"})
    assert started.started_at is not None
    assert store.start_job("j1") is None
    store.lose_jobs(["j1"])
    assert store.start_job("j1") is None
    assert store.get_job("j1").state == "lost"
    assert store.start_job("missing") is None


def test_a_heartbeat_mends_a_missing_or_queued_row(store: Store) -> None:
    store.create_job("j1", kind="python")
    beaten = store.beat_job("j1", progress={"done": 2})
    assert (beaten.state, beaten.progress) == ("running", {"done": 2})
    assert store.beat_job("j1").progress == {"done": 2}
    assert store.beat_job("gone") is None
    repair = {"kind": "python", "session_id": "s1", "operation_id": "op-9", "spec": {"n": 1}}
    made = store.beat_job("job-op-9", progress={"done": 1}, worker_pid=LIVE_PID, repair=repair)
    assert (made.state, made.kind, made.session_id, made.operation_id) == (
        "running",
        "python",
        "s1",
        "op-9",
    )
    assert made.spec == {"n": 1}


def test_a_round_of_upkeep_is_taken_once_a_minute_per_store(timed_store, clock) -> None:
    assert timed_store.take_sweep("jobs", 60.0) is True
    assert timed_store.take_sweep("jobs", 60.0) is False
    clock.step(59.0)
    assert timed_store.take_sweep("jobs", 60.0) is False
    clock.step(2.0)
    assert timed_store.take_sweep("jobs", 60.0) is True
    clock.step(-3600.0)
    assert timed_store.take_sweep("jobs", 60.0) is True


def test_unfinished_jobs_are_lost_with_their_session(store: Store) -> None:
    store.register_session("s1", kind="hython", pid=LIVE_PID, alias="w1")
    store.register_session("s2", kind="hython", pid=DEAD_PID, alias="w2")
    store.create_job("a", kind="python", session_id="s1", state="running")
    store.create_job("b", kind="python", session_id="s2", state="running")
    store.update_job("b", progress={"done": 3}, outputs={"kept": True})
    store.create_job("c", kind="python", session_id="s2", state="queued")
    store.create_job("d", kind="python", session_id="s2", state="done")
    assert store.reclaim_sessions() == ["s2"]
    lost = store.get_job("b")
    assert (lost.state, lost.progress, lost.outputs) == ("lost", {"done": 3}, {"kept": True})
    assert lost.error == store_module.SESSION_ENDED_ERROR
    assert lost.finished_at is not None
    assert store.get_job("c").state == "lost"
    assert store.get_job("d").state == "done"
    assert store.get_job("a").state == "running"
    store.end_session("s1")
    assert store.get_job("a").state == "lost"


def test_losing_jobs_leaves_the_ones_that_finished(store: Store) -> None:
    store.create_job("a", kind="python", state="running")
    store.create_job("b", kind="python", state="running")
    store.update_job("b", state="done")
    marked = store.lose_jobs(["a", "b", "missing"], error={"code": "X"})
    assert [record.job_id for record in marked] == ["a"]
    assert store.get_job("a").error == {"code": "X"}
    assert store.get_job("b").state == "done"
    assert store.lose_jobs([]) == []


def test_a_job_that_never_ran_can_be_taken_off(store: Store) -> None:
    store.create_job("j1", kind="python")
    assert store.drop_job("j1") is True
    assert store.drop_job("j1") is False
    assert store.get_job("j1") is None


def test_jobs_page_from_the_last_row_a_caller_has(timed_store, clock) -> None:
    for name in ("a", "b", "c", "d"):
        timed_store.create_job(name, kind="python", session_id="s1" if name < "c" else "s2")
        if name != "b":
            clock.step(1.0)
    first = timed_store.list_jobs(limit=2)
    assert [job.job_id for job in first] == ["d", "c"]
    after = (first[-1].created_at, first[-1].seq)
    assert [job.job_id for job in timed_store.list_jobs(limit=2, before=after)] == ["b", "a"]
    # Two rows made at the same moment still come one after the other.
    tied = timed_store.list_jobs(limit=1, before=after)
    rest = timed_store.list_jobs(before=(tied[0].created_at, tied[0].seq))
    assert [job.job_id for job in rest] == ["a"]
    assert [j.job_id for j in timed_store.list_jobs(session_ids=["s2"])] == ["d", "c"]
    assert timed_store.list_jobs(session_ids=[]) == []


def test_a_file_from_before_jobs_named_their_operation_reads_right(tmp_path) -> None:
    """A schema 7 file with a job in it, opened by a build that knows more."""
    path = tmp_path / "coord.sqlite"
    raw = sqlite3.connect(str(path))
    for statements in store_module.MIGRATIONS[:7]:
        for statement in statements:
            raw.execute(statement)
    raw.execute("PRAGMA user_version=7")
    raw.execute(
        "INSERT INTO jobs (job_id, session_id, kind, state, weight, created_at, updated_at)"
        " VALUES ('old', NULL, 'render', 'done', 'light', 1.0, 1.0)"
    )
    raw.commit()
    raw.close()
    with Store(path) as after:
        assert after.schema_version() == store_module.SCHEMA_VERSION
        old = after.get_job("old")
        assert (old.operation_id, old.spec, old.started_at) == (None, None, None)


def test_old_jobs_are_pruned(store: Store) -> None:
    store.create_job("j1", kind="render")
    store.update_job("j1", state="done")
    assert store.prune_jobs(max_age_s=-1) == 1
    assert store.get_job("j1") is None


def test_pruning_keeps_jobs_that_are_still_going_by_when_they_ended(timed_store, clock) -> None:
    timed_store.create_job("running", kind="python", state="running")
    timed_store.create_job("queued", kind="python")
    timed_store.create_job("ended", kind="python", state="running")
    timed_store.update_job("ended", state="done")
    clock.step(10.0)
    timed_store.touch_job("running")
    # A heartbeat is not an ending, and an old start is not either.
    clock.step(8 * 24 * 3600.0)
    timed_store.update_job("running", progress={"done": 1})
    gone = timed_store.prune_final_jobs(7 * 24 * 3600.0)
    assert [record.job_id for record in gone] == ["ended"]
    assert timed_store.get_job("running").state == "running"
    assert timed_store.get_job("queued").state == "queued"


# -- versions and runs ----------------------------------------------------


def test_versions_count_up_per_kind_name_and_hip_family(store: Store) -> None:
    assert store.allocate_version(kind="render", name="beauty", hip_family="shot") == 1
    assert store.allocate_version(kind="render", name="beauty", hip_family="shot") == 2
    assert store.allocate_version(kind="cache", name="beauty", hip_family="shot") == 1
    assert store.allocate_version(kind="render", name="smoke", hip_family="shot") == 1
    assert store.allocate_version(kind="render", name="beauty", hip_family="other") == 1
    assert store.latest_version(kind="render", name="beauty", hip_family="shot") == 2
    assert store.latest_version(kind="render", name="nothing", hip_family="shot") == 0


def test_a_version_can_be_pointed_at_its_run_afterwards(store: Store) -> None:
    version = store.allocate_version(kind="render", name="beauty", hip_family="shot")
    store.create_run("r1", kind="render", paths={}, name="beauty", version=version)
    store.attach_version_run(
        kind="render", name="beauty", hip_family="shot", version=version, run_id="r1"
    )
    row = store._read_one("SELECT run_id FROM versions WHERE version = ?", (version,))
    assert row["run_id"] == "r1"
    with pytest.raises(UnknownRecord):
        store.attach_version_run(
            kind="render", name="beauty", hip_family="shot", version=99, run_id="r1"
        )


def test_a_run_freezes_its_expanded_paths(store: Store) -> None:
    paths = {"output": "/scenes/renders/20260921_beauty/v001/beauty_v001.$F4.exr"}
    record = store.create_run(
        "r1",
        kind="render",
        paths=paths,
        name="beauty",
        hip_family="shot",
        version=1,
        session_id="s1",
        source_node="/out/karma1",
        job_id="j1",
        scene={"frames": [1, 10]},
    )
    assert record.paths == paths
    assert store.get_run("r1") == record
    assert store.get_run("missing") is None
    store.create_run("r2", kind="cache", paths={})
    assert [r.run_id for r in store.list_runs()] == ["r2", "r1"]
    assert [r.run_id for r in store.list_runs(kind="render")] == ["r1"]


# -- readable exports -----------------------------------------------------


def test_a_run_exports_as_readable_json_beside_a_scene(store: Store, tmp_path: Path) -> None:
    store.create_run(
        "r1",
        kind="render",
        paths={"output": "$HIP/renders/20260921_beauty/v001/beauty_v001.$F4.exr"},
        name="beauty",
        version=1,
    )
    export = store.run_export("r1")
    assert export["run_id"] == "r1"
    assert export["paths"]["output"].startswith("$HIP/")
    assert export["created_utc"].endswith("+00:00")

    written = write_export(export, tmp_path / "v001" / "_run.json")
    assert json.loads(written.read_text(encoding="utf-8")) == export


def test_a_job_exports_with_readable_times(store: Store, tmp_path: Path) -> None:
    store.create_job("j1", kind="render", scene={"hash": "abc"})
    store.update_job("j1", state="done", outputs=["/renders/a.exr"])
    export = store.job_export("j1")
    assert export["state"] == "done"
    assert export["scene"] == {"hash": "abc"}
    assert export["finished_utc"] is not None
    assert "started_utc" in export
    assert "seq" not in export
    write_export(export, tmp_path / ".agent" / "jobs" / "j1" / "job.json")
    assert (tmp_path / ".agent" / "jobs" / "j1" / "job.json").is_file()


def test_exporting_a_record_that_is_not_there_is_refused(store: Store) -> None:
    with pytest.raises(UnknownRecord):
        store.run_export("r1")
    with pytest.raises(UnknownRecord):
        store.job_export("j1")


# -- runs a scene has made ------------------------------------------------


def _run(store: Store, run_id: str, *, family: str = "shot", **rest) -> None:
    store.create_run(run_id, kind=rest.pop("kind", "render"), hip_family=family, paths={}, **rest)


def test_a_scene_familys_runs_come_newest_first_a_batch_at_a_time(
    timed_store: Store, clock: FakeClock
) -> None:
    for index in range(5):
        _run(timed_store, f"r{index}", name="beauty" if index % 2 else "sim")
        clock.step(1)
    _run(timed_store, "other", family="elsewhere")
    first = timed_store.find_runs(hip_family="shot", limit=2)
    assert [record.run_id for record in first] == ["r4", "r3"]
    last = first[-1]
    rest = timed_store.find_runs(hip_family="shot", before=(last.created_at, last.seq), limit=10)
    assert [record.run_id for record in rest] == ["r2", "r1", "r0"]


def test_runs_made_in_the_same_instant_keep_the_order_they_were_made_in(
    timed_store: Store,
) -> None:
    for run_id in ("b", "c", "a"):
        _run(timed_store, run_id)
    assert [record.run_id for record in timed_store.find_runs(hip_family="shot")] == [
        "a",
        "c",
        "b",
    ]
    [newest] = timed_store.find_runs(hip_family="shot", limit=1)
    after = timed_store.find_runs(hip_family="shot", before=(newest.created_at, newest.seq))
    assert [record.run_id for record in after] == ["c", "b"]
    assert "seq" not in timed_store.run_export("a")


def test_runs_found_by_kind_name_glob_time_and_session(
    timed_store: Store, clock: FakeClock
) -> None:
    _run(timed_store, "a", kind="cache", name="sim_fluid", session_id="s1")
    clock.step(10)
    _run(timed_store, "b", kind="render", name="beauty", session_id="s2")
    later = clock.now
    clock.step(10)
    _run(timed_store, "c", kind="render", name="sim_smoke", session_id="s1")
    found = timed_store.find_runs
    assert [r.run_id for r in found(hip_family="shot", kind="render")] == ["c", "b"]
    assert [r.run_id for r in found(hip_family="shot", name_glob="sim_*")] == ["c", "a"]
    assert [r.run_id for r in found(hip_family="shot", since=later)] == ["c", "b"]
    assert [r.run_id for r in found(hip_family="shot", session_id="s1")] == ["c", "a"]


# -- frozen output parameters ---------------------------------------------


def _freeze(store: Store, session_id: str = "s1", **rest):
    options = {
        "node_path": "/out/karma1",
        "parm_name": "picture",
        "template": "$HIP/renders/20260921_beauty/v001/beauty_v001.$F4.exr",
        "frozen": "/shots/renders/20260921_beauty/v001/beauty_v001.$F4.exr",
        "run_id": "run-1",
        "hip_key": "/shots/shot.hip",
    }
    options.update(rest)
    return store.freeze_parm(session_id=session_id, **options)


def test_a_frozen_parm_is_recorded_until_it_is_thawed(timed_store: Store, clock: FakeClock) -> None:
    made = _freeze(timed_store)
    assert made.template.startswith("$HIP/")
    assert made.created_at == clock.now
    assert timed_store.get_frozen_parm("s1", "/out/karma1", "picture") == made
    assert timed_store.list_frozen_parms(session_id="s1") == [made]
    assert timed_store.list_frozen_parms(hip_key="/shots/shot.hip") == [made]
    assert timed_store.list_frozen_parms(hip_key="/elsewhere.hip") == []
    assert timed_store.thaw_parm("s1", "/out/karma1", "picture") is True
    assert timed_store.thaw_parm("s1", "/out/karma1", "picture") is False
    assert timed_store.get_frozen_parm("s1", "/out/karma1", "picture") is None


def test_freezing_the_same_parm_again_still_owes_the_value_from_before_the_first(
    store: Store,
) -> None:
    first = _freeze(
        store, node_sid=7, original_expression='chs("../a/picture")', original_language="hscript"
    )
    assert first.node_sid == 7
    again = _freeze(
        store,
        run_id="run-2",
        template="$HIP/renders/x/v002/x_v002.$F4.exr",
        node_sid=7,
        original=first.frozen,
    )
    assert again.run_id == "run-2"
    assert again.template.endswith("x_v002.$F4.exr")
    # The first run's path was never the parameter's own.
    assert again.original is None
    assert again.original_expression == 'chs("../a/picture")'
    assert again.original_language == "hscript"
    assert [row.run_id for row in store.list_frozen_parms()] == ["run-2"]


def test_a_frozen_parm_table_from_an_earlier_shape_is_made_again(tmp_path: Path) -> None:
    path = tmp_path / "coord.sqlite"
    with Store(path):
        pass
    raw = sqlite3.connect(str(path))
    raw.execute("DROP TABLE frozen_parms")
    raw.execute(
        "CREATE TABLE frozen_parms (session_id TEXT, node_path TEXT, parm_name TEXT,"
        " template TEXT, frozen TEXT, run_id TEXT, hip_key TEXT, created_at REAL)"
    )
    raw.execute(
        "INSERT INTO frozen_parms VALUES ('s1', '/out/a', 'picture', 't', 'f', 'r', 'h', 1)"
    )
    raw.commit()
    raw.close()
    with Store(path) as opened:
        columns = {row[1] for row in opened._conn.execute("PRAGMA table_info(frozen_parms)")}
        assert store_module.FROZEN_PARM_COLUMNS <= columns
        assert opened.list_frozen_parms() == []
        _freeze(opened, token="a")
        assert opened.get_frozen_parm("s1", "/out/karma1", "picture").token == "a"


def test_a_parm_held_under_one_token_is_refused_to_another(store: Store) -> None:
    held = _freeze(store, token="a")
    assert held.state == store_module.FROZEN_PREPARED
    with pytest.raises(store_module.ParmHeld) as refused:
        _freeze(store, run_id="run-2", token="b")
    assert refused.value.run_id == "run-1"
    assert store.activate_frozen_parm("s1", "/out/karma1", "picture", token="b") is False
    assert store.activate_frozen_parm("s1", "/out/karma1", "picture", token="a") is True
    row = store.get_frozen_parm("s1", "/out/karma1", "picture")
    assert row.state == store_module.FROZEN_ACTIVE
    # The same token freezing it again starts over as prepared.
    assert _freeze(store, run_id="run-3", token="a").state == store_module.FROZEN_PREPARED


def test_a_record_goes_only_under_its_own_token_and_state(store: Store) -> None:
    _freeze(store, token="a")
    assert store.thaw_parm("s1", "/out/karma1", "picture", token="b") is False
    assert (
        store.thaw_parm("s1", "/out/karma1", "picture", token="a", state=store_module.FROZEN_ACTIVE)
        is False
    )
    assert (
        store.thaw_parm(
            "s1", "/out/karma1", "picture", token="a", state=store_module.FROZEN_PREPARED
        )
        is True
    )
    _freeze(store, token="c")
    assert store.thaw_parm("s1", "/out/karma1", "picture", any_token=True) is True


def test_only_a_session_that_is_over_leaves_orphans(store: Store) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        stamp = store_module.process_start_stamp(child.pid)
        store.register_session("s-live", kind="hython", pid=LIVE_PID, alias="w1")
        store.register_session(
            "s-killed", kind="hython", pid=child.pid, alias="w2", pid_start=stamp
        )
        store.register_session("s-ended", kind="hython", pid=LIVE_PID, alias="w3")
        store.end_session("s-ended")
        _freeze(store, "s-live")
        _freeze(store, "s-killed", node_path="/out/karma2")
        _freeze(store, "s-ended", node_path="/out/karma3", hip_key=None)
        _freeze(store, "s-never", node_path="/out/karma4")
        assert store.session_is_over("s-live") is False
        assert store.session_is_over("s-killed") is False
        child.kill()
        child.wait(timeout=30)
        assert store.session_is_over("s-killed") is True
        orphans = {row.node_path for row in store.orphan_frozen_parms()}
        assert orphans == {"/out/karma2", "/out/karma3", "/out/karma4"}
        in_scene = store.orphan_frozen_parms(hip_key="/shots/shot.hip")
        assert {row.node_path for row in in_scene} == {"/out/karma2", "/out/karma4"}
        # The one from a scene never saved can never be given back.
        assert store.forget_unrestorable_frozen_parms() == 1
        assert {row.node_path for row in store.list_frozen_parms()} == {
            "/out/karma1",
            "/out/karma2",
            "/out/karma4",
        }
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
