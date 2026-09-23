"""`hou_jobs` and the jobs `hou_python` runs as, through the server to a bridge.

Every call goes the whole way, as in the `hou_python` checks: the server, the
router, a real dispatcher with real receipts and a real job keeper, and the
stand in for `hou`. The store is a real one in the test's own folder. What a
real Houdini does with the same calls is in the integration checks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client

import support
from fake_hou import Scene
from nscr_houdini_mcp import jobs as job_rules
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import tools
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.jobs import JobKeeper, JobNotAccepted, _Job
from nscr_houdini_mcp.tools import jobs as jobs_tool
from test_server import talk, text_of
from test_tools_python import Clock, Through, ok, python, refused, through
from test_tools_sessions import DEAD_PID, Bench

LOOP = (
    "import time\n"
    "laps = 0\n"
    "while not mcp.cancelled() and laps < 2000:\n"
    "    laps += 1\n"
    "    time.sleep(0.01)\n"
    "result = laps\n"
)


@pytest.fixture(name="scene")
def made_scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture(name="module")
def made_module(scene: Scene) -> Any:
    made = scene.module()
    # Something a check can hold the code on, reached the way the code
    # reaches everything else.
    made.gate = threading.Event()
    return made


@pytest.fixture(name="bench")
def made_bench(tmp_path: Path, module: Any) -> Bench:
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1")
    made.sent = Through(module, home, tools.Namespaces(clock=Clock()))  # type: ignore[assignment]
    return made


@pytest.fixture(autouse=True)
def fresh_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every check sweeps as if this process had never swept before."""
    monkeypatch.setattr(jobs_tool, "SWEEP_EVERY_S", 0.0)


def jobs(bench: Bench, **arguments: Any) -> Any:
    _, [result] = talk(bench.serve(), ("hou_jobs", arguments))
    return result


def job(bench: Bench, job_id: str, **rest: Any) -> dict[str, Any]:
    return ok(jobs(bench, job_id=job_id, **rest))


def row(bench: Bench, job_id: str) -> store_module.JobRecord:
    with bench.store() as store:
        found = store.get_job(job_id)
    assert found is not None
    return found


def idle(bench: Bench) -> None:
    support.wait_until(lambda: not through(bench).dispatcher.state()["busy"], timeout_s=10.0)


def in_the_background(bench: Bench, code: str, **rest: Any) -> dict[str, Any]:
    started = ok(python(bench, code=code, background=True, **rest))
    assert started["state"] in ("queued", "running"), started
    return started


# Section: every hou_python call is a job


def test_quick_code_answers_inline_and_leaves_a_done_job(bench: Bench) -> None:
    body = ok(python(bench, code="result = 6 * 7"))
    assert body["result"] == 42
    assert body["state"] == "done"
    operation_id = body["trace"]["operation_id"]
    assert body["job_id"] == job_rules.job_id_for(operation_id)
    status = job(bench, body["job_id"])
    assert status["state"] == "done"
    assert status["kind"] == "python"
    assert status["session"] == "s-1"
    assert status["alias"] == "w1"
    assert status["operation_id"] == operation_id
    assert status["outputs"]["result"] == 42
    assert status["outputs"]["namespace"] == body["namespace"]
    assert status["error"] is None
    assert status["ended_at"] >= status["started_at"]
    assert status["elapsed_s"] >= 0
    assert status["cancel_requested"] is False
    assert status["trace"]["session_id"] == "s-1"


def test_code_that_raises_leaves_a_failed_job_with_the_error(bench: Bench) -> None:
    result = python(bench, code="raise ValueError('no')")
    assert result.is_error is True
    body = result.structured_content
    assert body["state"] == "failed"
    status = job(bench, body["job_id"])
    assert status["state"] == "failed"
    assert status["error"]["type"] == "ValueError"


def test_slow_code_under_auto_turns_into_a_job_at_the_inline_wait(
    bench: Bench, module: Any
) -> None:
    bench.config = replace(bench.config, inline_wait_s=1)
    began = time.monotonic()
    handle = ok(python(bench, code="hou.gate.wait(10)\nresult = 'late'", namespace="slow"))
    waited = time.monotonic() - began
    assert 0.9 <= waited < 5.0
    assert through(bench).calls[-1]["timeout_s"] == 1
    assert handle["state"] == "running"
    assert handle["kind"] == "python"
    assert handle["namespace"] == "slow"
    assert handle["session"] == "s-1"
    assert handle["started_at"] is not None
    assert handle["job_id"] == job_rules.job_id_for(handle["operation_id"])
    assert "result" not in handle
    # Never abandoned: the code is still running, and finishes.
    assert through(bench).dispatcher.state()["busy"] is True
    module.gate.set()
    finished = job(bench, handle["job_id"], wait_s=10)
    assert finished["state"] == "done"
    assert finished["outputs"]["result"] == "late"


def test_background_true_answers_once_the_session_has_taken_the_call(
    bench: Bench, module: Any
) -> None:
    handle = in_the_background(bench, "hou.gate.wait(10)\nresult = 1")
    assert through(bench).calls[-1]["timeout_s"] == 0
    assert row(bench, handle["job_id"]).state in ("queued", "running")
    module.gate.set()
    assert job(bench, handle["job_id"], wait_s=10)["state"] == "done"


def test_background_false_times_out_with_the_job_to_follow(bench: Bench, module: Any) -> None:
    result = python(bench, code="hou.gate.wait(10)", timeout_s=0.2, background=False)
    error = refused(result)
    assert error["code"] == "TIMEOUT"
    assert error["details"]["still_running"] is True
    job_id = error["details"]["job_id"]
    assert job(bench, job_id)["state"] == "running"
    module.gate.set()
    assert job(bench, job_id, wait_s=10)["state"] == "done"


# Section: waiting


class HeldClock:
    """The clock a held status reads, moved by its own pauses.

    Each pause moves the clock on and then runs whatever the check asked to
    happen at that moment, so the order of events is fixed and no check
    waits on real time.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        self.pauses = 0
        self.at: dict[int, Any] = {}

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.pauses += 1
        happen = self.at.get(self.pauses)
        if happen is not None:
            happen()


@pytest.fixture
def held_clock(monkeypatch: pytest.MonkeyPatch) -> HeldClock:
    made = HeldClock()
    monkeypatch.setattr(jobs_tool, "monotonic", made.monotonic)
    monkeypatch.setattr(jobs_tool, "sleep", made.sleep)
    return made


def a_running_job(bench: Bench, job_id: str) -> None:
    with bench.store() as store:
        store.create_job(
            job_id, kind="python", session_id="s-1", state="running", worker_pid=os.getpid()
        )


def test_a_wait_returns_at_its_deadline_when_nothing_changes(
    bench: Bench, held_clock: HeldClock
) -> None:
    a_running_job(bench, "job-still")
    held = job(bench, "job-still", wait_s=3)
    assert held["changed"] is False
    assert held["state"] == "running"
    # Three seconds of pauses a quarter of a second long, and not one more.
    assert held_clock.pauses == 12


def test_a_wait_returns_when_the_job_ends(bench: Bench, held_clock: HeldClock) -> None:
    a_running_job(bench, "job-ends")

    def end() -> None:
        with bench.store() as store:
            store.update_job("job-ends", state="done", outputs={"answer": {"result": 1}})

    held_clock.at[3] = end
    held = job(bench, "job-ends", wait_s=20)
    assert held["changed"] is True
    assert held["state"] == "done"
    assert held_clock.pauses == 3


def test_a_wait_returns_when_progress_moves(bench: Bench, held_clock: HeldClock) -> None:
    a_running_job(bench, "job-moves")

    def note() -> None:
        with bench.store() as store:
            store.beat_job("job-moves", progress={"done": 1, "total": 4, "message": "one"})

    held_clock.at[2] = note
    held = job(bench, "job-moves", wait_s=20)
    assert held["changed"] is True
    assert held["state"] == "running"
    assert held["progress"] == {"done": 1, "total": 4, "message": "one"}
    assert held_clock.pauses == 2


def test_a_wait_ends_when_the_session_is_found_gone(bench: Bench, held_clock: HeldClock) -> None:
    bench.session("s-2", "w2")
    with bench.store() as store:
        store.create_job("job-orphan", kind="python", session_id="s-2", state="running")

    def gone() -> None:
        # The process behind the session ends, as a crash ends it: nothing
        # writes to the store, so only a look at the session can tell.
        raw = sqlite3.connect(str(bench.store_path))
        raw.execute("UPDATE sessions SET pid_start = 'another' WHERE session_id = 's-2'")
        raw.commit()
        raw.close()

    held_clock.at[1] = gone
    held = job(bench, "job-orphan", wait_s=20)
    assert held["state"] == "lost"
    assert held["error"]["code"] == "SESSION_ENDED"
    assert held["changed"] is True
    # Found at the first look at the session, two seconds in.
    assert held_clock.pauses == 8


def test_a_progress_note_written_before_the_hold_began_is_no_change(
    bench: Bench, held_clock: HeldClock
) -> None:
    with bench.store() as store:
        store.create_job(
            "job-noted",
            kind="python",
            session_id="s-1",
            state="running",
            worker_pid=os.getpid(),
            progress={"done": 1, "total": 4, "message": "one"},
        )
    held = job(bench, "job-noted", wait_s=1)
    assert held["changed"] is False
    assert held["progress"] == {"done": 1, "total": 4, "message": "one"}


def test_waiting_on_a_job_that_has_ended_answers_at_once(bench: Bench) -> None:
    body = ok(python(bench, code="result = 1"))
    began = time.monotonic()
    held = job(bench, body["job_id"], wait_s=30)
    assert time.monotonic() - began < 5.0
    assert held["changed"] is False
    assert held["state"] == "done"


# Section: cancelling


def test_cancel_reaches_a_running_call_and_it_ends_cancelled(bench: Bench) -> None:
    handle = in_the_background(bench, LOOP)
    job_id = handle["job_id"]
    support.wait_until(lambda: row(bench, job_id).state == "running", timeout_s=5.0)
    asked = job(bench, job_id, action="cancel")
    assert asked["cancel"]["requested"] is True
    assert asked["cancel"]["reached_session"] is True
    assert asked["cancel"]["asked"] is True
    assert asked["cancel_requested"] is True
    idle(bench)
    ended = job(bench, job_id)
    assert ended["state"] == "cancelled"
    assert ended["outputs"]["result"] < 2000
    assert ended["cancel_requested"] is True


def test_a_cancel_the_session_never_hears_still_arrives_through_the_store(
    bench: Bench,
) -> None:
    handle = in_the_background(bench, LOOP)
    job_id = handle["job_id"]
    support.wait_until(lambda: row(bench, job_id).state == "running", timeout_s=5.0)
    # The session's port stops answering; the call keeps running.
    bench.reachable.discard("s-1")
    bench.health["s-1"] = RuntimeError("unreachable")
    asked = job(bench, job_id, action="cancel")
    assert asked["cancel"]["requested"] is True
    assert asked["cancel"]["reached_session"] is False
    support.wait_until(lambda: row(bench, job_id).state == "cancelled", timeout_s=10.0)


def test_code_that_ignores_a_cancel_ends_done_with_the_request_on_the_row(
    bench: Bench, module: Any
) -> None:
    handle = in_the_background(bench, "hou.gate.wait(10)\nresult = 'ignored'")
    job_id = handle["job_id"]
    support.wait_until(lambda: row(bench, job_id).state == "running", timeout_s=5.0)
    job(bench, job_id, action="cancel")
    module.gate.set()
    ended = job(bench, job_id, wait_s=10)
    assert ended["state"] == "done"
    assert ended["cancel_requested"] is True


def test_cancelling_a_job_that_has_ended_changes_nothing(bench: Bench) -> None:
    body = ok(python(bench, code="result = 1"))
    said = job(bench, body["job_id"], action="cancel")
    assert said["cancel"]["requested"] is False
    assert said["cancel_requested"] is False
    assert said["state"] == "done"


# Section: lost


def test_the_jobs_of_a_killed_session_are_lost_with_what_they_wrote(bench: Bench) -> None:
    bench.session("s-dead", "w2", pid=DEAD_PID)
    with bench.store() as store:
        store.create_job(
            "job-dead",
            kind="python",
            session_id="s-dead",
            state="running",
            scene={"session_id": "s-dead", "alias": "w2", "scene_epoch": 3},
            progress={"done": 5, "total": 10, "message": "half"},
            operation_id="op-dead",
        )
        store.update_job("job-dead", outputs={"written": 5})
        # Its caller was handed the job to follow, so it leaves a copy.
        store.promote_job("job-dead")
        store.create_job("job-quiet-dead", kind="python", session_id="s-dead", state="running")
    status = job(bench, "job-dead")
    assert status["state"] == "lost"
    assert status["progress"] == {"done": 5, "total": 10, "message": "half"}
    assert status["outputs"] == {"written": 5}
    assert status["error"]["code"] == "SESSION_ENDED"
    assert status["ended_at"] is not None
    assert status["scene_epoch"] == 3
    # No scene file, so the readable copy went to the scratch folder.
    assert Path(status["export_path"]).is_file()
    assert "/.agent/jobs/job-dead.json" in Path(status["export_path"]).as_posix()
    # One whose caller had its answer, or none, leaves no copy.
    quiet = job(bench, "job-quiet-dead")
    assert quiet["state"] == "lost"
    assert "export_path" not in quiet


def test_a_session_stopped_on_purpose_loses_its_unfinished_jobs(bench: Bench) -> None:
    bench.session("s-2", "w2")
    with bench.store() as store:
        store.create_job("job-a", kind="python", session_id="s-2", state="running")
        store.create_job("job-b", kind="python", session_id="s-2", state="queued")
        store.create_job("job-c", kind="python", session_id="s-2", state="running")
        store.update_job("job-c", state="done")
        store.end_session("s-2")
        states = {job_id: store.get_job(job_id).state for job_id in ("job-a", "job-b", "job-c")}
    assert states == {"job-a": "lost", "job-b": "lost", "job-c": "done"}


def test_a_job_nobody_has_heard_from_is_never_lost_while_its_session_is_there(
    bench: Bench,
) -> None:
    hour = 3600.0
    with bench.store(clock=lambda: time.time() - 5 * hour) as store:
        store.create_job(
            "job-quiet", kind="python", session_id="s-1", state="running", worker_pid=os.getpid()
        )
    status = job(bench, "job-quiet")
    assert status["state"] == "running"
    assert status["error"] is None
    assert status["silent_s"] >= 5 * hour - 60


# Section: keeping and listing


def test_jobs_are_pruned_after_seven_days(bench: Bench) -> None:
    week = job_rules.KEEP_S
    then = time.time() - week - 60.0
    with bench.store(clock=lambda: then) as store:
        store.create_job("job-old", kind="python", session_id="s-1", state="done")
    with bench.store(clock=lambda: time.time() - week + 3600.0) as store:
        store.create_job("job-kept", kind="python", session_id="s-1", state="done")
    error = refused(jobs(bench, job_id="job-old"))
    assert error["code"] == "JOB_UNKNOWN"
    assert "list" in error["hint"]
    assert job(bench, "job-kept")["state"] == "done"


def test_an_id_nobody_knows_is_refused_with_a_way_on(bench: Bench) -> None:
    error = refused(jobs(bench, job_id="job-nothing"))
    assert error["code"] == "JOB_UNKNOWN"
    assert error["details"]["kept_days"] == 7
    error = refused(jobs(bench, action="status"))
    assert error["code"] == "BAD_ARGUMENTS"
    error = refused(jobs(bench, job_id="job-nothing", max_chars=0))
    assert (error["code"], error["details"]["argument"]) == ("BAD_ARGUMENTS", "max_chars")


def test_list_pages_newest_first_and_filters(bench: Bench) -> None:
    bench.session("s-2", "w2")
    moment = [1000.0]
    for index in range(5):
        moment[0] += 1.0
        with bench.store(clock=lambda: moment[0] + time.time() - 1000.0) as store:
            store.create_job(
                f"job-{index}",
                kind="python",
                session_id="s-2" if index % 2 else "s-1",
                state="done" if index < 3 else "running",
            )
    first = ok(jobs(bench, action="list", limit=2))
    assert [item["job_id"] for item in first["jobs"]] == ["job-4", "job-3"]
    second = ok(jobs(bench, action="list", limit=2, page=first["next_page"]))
    assert [item["job_id"] for item in second["jobs"]] == ["job-2", "job-1"]
    third = ok(jobs(bench, action="list", limit=2, page=second["next_page"]))
    assert [item["job_id"] for item in third["jobs"]] == ["job-0"]
    assert third["next_page"] is None
    by_alias = ok(jobs(bench, action="list", session="w2"))
    assert [item["job_id"] for item in by_alias["jobs"]] == ["job-3", "job-1"]
    running = ok(jobs(bench, action="list", state="running"))
    assert {item["job_id"] for item in running["jobs"]} == {"job-3", "job-4"}
    other = refused(jobs(bench, action="list", limit=2, state="done", page=first["next_page"]))
    assert other["code"] == "BAD_CURSOR"
    garbled = refused(jobs(bench, action="list", page="not a token"))
    assert garbled["code"] == "BAD_CURSOR"


# Section: another server, and the readable copy


def test_a_second_server_follows_a_job_by_id(bench: Bench, module: Any) -> None:
    handle = in_the_background(bench, "hou.gate.wait(10)\nresult = 'across'")
    module.gate.set()
    idle(bench)
    # A new server object, as a client restart makes: nothing is carried over
    # but the store.
    _, [held] = talk(bench.serve(), ("hou_jobs", {"job_id": handle["job_id"], "wait_s": 20}))
    body = ok(held)
    assert body["state"] == "done"
    assert body["outputs"]["result"] == "across"


def followed_to_the_end(bench: Bench, module: Any) -> dict[str, Any]:
    """A job handed out to follow, let run to its end, with its copy written."""
    handle = in_the_background(bench, "hou.gate.wait(10)\nresult = 3")
    module.gate.set()
    idle(bench)
    support.wait_until(lambda: row(bench, handle["job_id"]).export_path, timeout_s=5.0)
    return job(bench, handle["job_id"])


def test_a_job_handed_out_to_follow_leaves_a_readable_copy_beside_the_scene(
    bench: Bench, scene: Scene, module: Any, tmp_path: Path
) -> None:
    folder = tmp_path / "shots"
    folder.mkdir()
    hip = folder / "shot_v002.hip"
    hip.write_bytes(b"scene")
    scene.hipFile.setName(str(hip))
    status = followed_to_the_end(bench, module)
    copy = folder / ".agent" / "jobs" / f"{status['job_id']}.json"
    assert Path(status["export_path"]) == copy
    written = json.loads(copy.read_text(encoding="utf-8"))
    assert written["state"] == "done"
    assert written["job_id"] == status["job_id"]
    assert written["scene"]["hip_path"] == str(hip)
    assert written["finished_utc"]
    assert written["export_path"] == str(copy) or written["export_path"] == copy.as_posix()


def test_an_answer_given_inline_leaves_no_copy(bench: Bench, scene: Scene, tmp_path: Path) -> None:
    folder = tmp_path / "shots"
    folder.mkdir()
    hip = folder / "shot.hip"
    hip.write_bytes(b"scene")
    scene.hipFile.setName(str(hip))
    body = ok(python(bench, code="result = 3"))
    assert "export_path" not in job(bench, body["job_id"])
    assert not (folder / ".agent").exists()


def test_an_untitled_scene_puts_the_copy_in_the_scratch_folder(
    bench: Bench, scene: Scene, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HOUDINI_TEMP_DIR", raising=False)
    scene.hipFile.setName("untitled.hip")
    copy = Path(followed_to_the_end(bench, module)["export_path"])
    assert copy.is_file()
    assert copy.is_relative_to(bench.home / "temp")
    assert copy.parent.as_posix().endswith("s-1/.agent/jobs")


def test_a_pruned_job_takes_its_readable_copy_with_it(
    bench: Bench, scene: Scene, module: Any, tmp_path: Path
) -> None:
    hip = tmp_path / "shot.hip"
    hip.write_bytes(b"scene")
    scene.hipFile.setName(str(hip))
    status = followed_to_the_end(bench, module)
    copy = Path(status["export_path"])
    assert copy.is_file()
    # A week and a minute later, the upkeep round takes the row and the copy.
    with bench.store(clock=lambda: time.time() + job_rules.KEEP_S + 60.0) as store:
        jobs_tool.sweep(store, bench.home)
        assert store.get_job(status["job_id"]) is None
    assert not copy.exists()


# Section: the keeper on its own


class Running:
    """What the keeper reads of a running call."""

    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        self.job_id: str | None = None
        self.cancel = threading.Event()
        self.ended = threading.Event()
        self.noted = threading.Event()
        self.progress: list[dict[str, Any]] = []
        self.cancel_seen = False

    def note(self, done: int) -> None:
        self.progress.append({"done": done, "total": 100, "message": None})
        self.noted.set()


class FakeTime:
    """A clock the keeper's waits move, instead of sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Ended:
    """The end of the work, reached once the fake clock gets to `at`."""

    def __init__(self, clock: FakeTime, at: float) -> None:
        self.clock = clock
        self.at = at

    def is_set(self) -> bool:
        return self.clock.now >= self.at

    def set(self) -> None:
        self.at = self.clock.now

    def wait(self, seconds: float) -> bool:
        self.clock.now += seconds
        return self.is_set()


class Flood:
    """Progress notes that never stop coming: fifty more on every look."""

    def __init__(self, running: Running) -> None:
        self.running = running

    def wait(self, seconds: float) -> bool:
        for _ in range(50):
            self.running.note(len(self.running.progress))
        return True

    def is_set(self) -> bool:
        return True

    def set(self) -> None:
        pass

    def clear(self) -> None:
        pass


class Quiet:
    """No progress notes at all: every look waits out the whole poll."""

    def __init__(self, clock: FakeTime) -> None:
        self.clock = clock

    def wait(self, seconds: float) -> bool:
        self.clock.now += seconds
        return False

    def is_set(self) -> bool:
        return False

    def clear(self) -> None:
        pass


class Counting(store_module.Store):
    beats: list[Any] = []

    def beat_job(self, job_id: str, **rest: Any) -> Any:
        Counting.beats.append(rest.get("progress"))
        return super().beat_job(job_id, **rest)


def watched(path: Path, job_id: str) -> _Job:
    """A running job's row and the keeper's own note of it, with no thread watching."""
    with store_module.Store(path) as store:
        store.create_job(job_id, kind="python", session_id="s-1", state="running")
    return _Job(job_id=job_id, kind="python", repair={"kind": "python", "scene": {}})


def test_a_flood_of_progress_is_written_once_a_second(tmp_path: Path) -> None:
    path = tmp_path / "coord.sqlite"
    clock = FakeTime()
    Counting.beats = []
    keeper = JobKeeper(lambda: Counting(path), session_id="s-1", clock=clock)
    running = Running("op-notes")
    job = watched(path, "job-op-notes")
    running.ended = Ended(clock, 10.0)  # type: ignore[assignment]
    running.noted = Flood(running)  # type: ignore[assignment]
    keeper._watch(running, job)
    # Ten fake seconds of notes, hundreds of them, written ten times.
    assert len(Counting.beats) in (9, 10), Counting.beats
    assert all(note is not None for note in Counting.beats)
    assert len(running.progress) >= 450
    with store_module.Store(path) as store:
        assert store.get_job(job.job_id).progress["done"] >= 400


def test_the_keeper_finds_a_cancel_in_the_store_within_two_seconds(tmp_path: Path) -> None:
    path = tmp_path / "coord.sqlite"
    clock = FakeTime()
    keeper = JobKeeper(lambda: store_module.Store(path), session_id="s-1", clock=clock)
    running = Running("op-flag")
    job = watched(path, "job-op-flag")
    with store_module.Store(path) as store:
        store.request_job_cancel(job.job_id)
    ended = Ended(clock, 60.0)
    running.ended = ended  # type: ignore[assignment]
    running.noted = Quiet(clock)  # type: ignore[assignment]
    seen: list[float] = []
    running.cancel = SeenAt(clock, seen, ended)  # type: ignore[assignment]
    keeper._watch(running, job)
    assert seen == [2.0]


class SeenAt:
    """A cancel flag that notes when it was set and ends the watch there."""

    def __init__(self, clock: FakeTime, seen: list[float], ended: Ended) -> None:
        self.clock = clock
        self.seen = seen
        self.ended = ended

    def is_set(self) -> bool:
        return bool(self.seen)

    def set(self) -> None:
        self.seen.append(self.clock.now)
        self.ended.set()


def test_a_locked_store_at_accept_refuses_the_call_before_the_code_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "coord.sqlite"
    opened: list[int] = []

    def locked() -> store_module.Store:
        opened.append(1)
        raise store_module.StoreBusy("the store is locked by another process")

    keeper = JobKeeper(locked, session_id="s-1", backoff_s=(0.0, 0.0))
    running = Running("op-locked")
    with pytest.raises(JobNotAccepted) as refused:
        keeper.accept(running, kind="python", spec=None, identity={})
    assert refused.value.code == "STORE_UNAVAILABLE"
    assert len(opened) == 3
    assert running.job_id is None
    # One busy moment and then the write lands: the job is accepted.
    tries = iter([store_module.StoreBusy("locked"), None])

    def once_locked() -> store_module.Store:
        trouble = next(tries, None)
        if trouble is not None:
            raise trouble
        return store_module.Store(path)

    keeper = JobKeeper(once_locked, session_id="s-1", backoff_s=(0.0,))
    running = Running("op-late")
    job_id = keeper.accept(running, kind="python", spec=None, identity={})
    running.ended.set()
    with store_module.Store(path) as store:
        assert store.get_job(job_id).state == "queued"


def test_a_locked_store_at_accept_answers_the_caller_and_runs_nothing(
    bench: Bench, scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    keeper = through(bench).dispatcher._jobs
    monkeypatch.setattr(keeper, "_backoff_s", (0.0,))

    def locked() -> store_module.Store:
        raise store_module.StoreBusy("the store is locked by another process")

    monkeypatch.setattr(keeper, "_open_store", locked)
    error = refused(python(bench, code="hou.node('/obj').createNode('geo')", operation_id="op-x"))
    assert error["code"] == "STORE_UNAVAILABLE"
    assert scene.node("/obj").children() == ()
    assert through(bench).dispatcher.state()["busy"] is False
    monkeypatch.undo()
    # The receipt went back with the refusal, so the same id runs now.
    body = ok(python(bench, code="hou.node('/obj').createNode('geo')", operation_id="op-x"))
    assert body["state"] == "done"
    assert len(scene.node("/obj").children()) == 1


def test_an_operation_id_whose_job_is_still_kept_is_refused(bench: Bench) -> None:
    body = ok(python(bench, code="result = 1", operation_id="op-kept"))
    with bench.store() as store:
        store.prune_operations(max_age_s=-1)
    error = refused(python(bench, code="result = 2", operation_id="op-kept"))
    assert error["code"] == "JOB_ID_TAKEN"
    assert "new operation_id" in error["hint"]
    assert job(bench, body["job_id"])["outputs"]["result"] == 1


def test_work_that_was_never_picked_up_leaves_no_job(tmp_path: Path) -> None:
    path = tmp_path / "coord.sqlite"
    keeper = JobKeeper(lambda: store_module.Store(path), session_id="s-1")
    running = Running("op-never")
    job_id = keeper.accept(running, kind="python", spec=None, identity={})
    with store_module.Store(path) as store:
        assert store.get_job(job_id).state == "queued"
    running.ended.set()
    keeper.drop(running)
    with store_module.Store(path) as store:
        assert store.get_job(job_id) is None


def test_the_hou_jobs_tool_is_listed_after_hou_python_and_small(bench: Bench) -> None:
    listed, _ = talk(bench.serve())
    names = [tool.name for tool in listed.tools]
    assert names[names.index("hou_python") + 1] == "hou_jobs"
    [tool] = [tool for tool in listed.tools if tool.name == "hou_jobs"]
    assert set(tool.input_schema["properties"]) == {
        "action",
        "job_id",
        "wait_s",
        "session",
        "state",
        "limit",
        "page",
        "max_chars",
    }
    assert tool.input_schema["properties"]["wait_s"]["maximum"] == 50
    assert "7 days" in tool.description
    assert text_of(jobs(bench, action="list")).startswith("{")


def test_a_held_status_tells_a_client_that_asked_that_it_is_still_waiting(
    bench: Bench, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs_tool, "HOLD_NOTE_S", 0.2)
    handle = in_the_background(bench, "hou.gate.wait(10)")
    support.wait_until(lambda: row(bench, handle["job_id"]).state == "running", timeout_s=5.0)
    heard: list[tuple[float, Any, Any]] = []

    async def hear(progress: float, total: float | None, message: str | None) -> None:
        heard.append((progress, total, message))

    async def held() -> Any:
        async with Client(bench.serve()) as connected:
            return await connected.call_tool(
                "hou_jobs", {"job_id": handle["job_id"], "wait_s": 1}, progress_callback=hear
            )

    body = ok(asyncio.run(held()))
    assert body["changed"] is False
    assert heard, "no progress note arrived"
    assert [note[0] for note in heard] == sorted(note[0] for note in heard)
    assert heard[0][1] == 1
    assert "running" in heard[0][2]
    # Without a callback the same wait answers the same, and nothing depends on it.
    assert job(bench, handle["job_id"], wait_s=0.5)["changed"] is False
    module.gate.set()
    idle(bench)


def test_a_stop_asked_of_the_session_directly_is_put_on_the_row(bench: Bench) -> None:
    handle = in_the_background(bench, LOOP)
    job_id = handle["job_id"]
    support.wait_until(lambda: row(bench, job_id).state == "running", timeout_s=5.0)
    reply = through(bench).dispatcher.dispatch(
        Envelope(tool="bridge.cancel", arguments={"operation_id": handle["operation_id"]})
    )
    assert reply.payload["data"]["asked"] is True
    support.wait_until(lambda: row(bench, job_id).state == "cancelled", timeout_s=10.0)
    assert row(bench, job_id).cancel_requested is True


def test_a_running_job_sits_on_its_worker_row_until_it_ends(bench: Bench, module: Any) -> None:
    bench.worker("s-1", "wk-1")
    handle = in_the_background(bench, "hou.gate.wait(10)")
    with bench.store() as store:
        assert store.get_worker("wk-1").job_id == handle["job_id"]
    module.gate.set()
    idle(bench)
    support.wait_until(lambda: row(bench, handle["job_id"]).state == "done", timeout_s=5.0)
    with bench.store() as store:
        assert store.get_worker("wk-1").job_id is None


def test_work_that_ends_after_its_job_was_found_lost_takes_its_real_ending(
    bench: Bench, module: Any
) -> None:
    handle = in_the_background(bench, "hou.gate.wait(10)\nresult = 'late'")
    job_id = handle["job_id"]
    support.wait_until(lambda: row(bench, job_id).state == "running", timeout_s=5.0)
    with bench.store() as store:
        store.lose_jobs([job_id], error=store_module.SESSION_ENDED_ERROR)
    assert row(bench, job_id).state == "lost"
    module.gate.set()
    idle(bench)
    ended = row(bench, job_id)
    assert (ended.state, ended.error) == ("done", None)


def test_the_receipt_and_the_job_end_in_one_step(bench: Bench) -> None:
    body = ok(python(bench, code="result = 5", operation_id="op-joint"))
    with bench.store() as store:
        receipt = store.get_operation("op-joint")
        record = store.get_job(body["job_id"])
    assert receipt.state == "done"
    assert record.state == "done"
    assert record.outputs["answer"]["result"] == 5
    assert receipt.updated_at == record.updated_at


def test_a_receipt_settled_without_its_job_is_reconciled_by_the_sweep(bench: Bench) -> None:
    """The session wrote the receipt and went before the job row: not lost, done."""
    bench.session("s-dead", "w2", pid=DEAD_PID)
    payload = {"ok": True, "data": {"result": 7, "stdout": "", "namespace": "n"}}
    with bench.store() as store:
        store.create_job(
            "job-op-crash",
            kind="python",
            session_id="s-dead",
            state="running",
            operation_id="op-crash",
            spec={"namespace": "n"},
        )
        store.begin_operation("op-crash", "digest", session_id="s-dead")
        store.finish_operation("op-crash", outcome=payload)
    status = job(bench, "job-op-crash")
    assert status["state"] == "done"
    assert status["error"] is None
    assert status["outputs"]["result"] == 7


def test_a_joint_write_that_does_not_land_leaves_the_receipt_for_the_sweep(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    keeper = through(bench).dispatcher._jobs
    monkeypatch.setattr(keeper, "_backoff_s", ())

    def busy(self: Any, job_id: str, **rest: Any) -> Any:
        raise store_module.StoreBusy("the store is locked by another process")

    monkeypatch.setattr(store_module.Store, "finish_job", busy)
    body = ok(python(bench, code="result = 11", operation_id="op-split"))
    assert body["result"] == 11
    with bench.store() as store:
        assert store.get_operation("op-split").state == "done"
        assert store.get_job(body["job_id"]).state == "running"
    monkeypatch.undo()
    status = job(bench, body["job_id"])
    assert status["state"] == "done"
    assert status["outputs"]["result"] == 11


def test_a_job_stopped_by_its_session_going_down_is_lost_not_cancelled(bench: Bench) -> None:
    handle = in_the_background(bench, LOOP)
    job_id = handle["job_id"]
    support.wait_until(lambda: row(bench, job_id).state == "running", timeout_s=5.0)
    through(bench).stopping.set()
    idle(bench)
    ended = row(bench, job_id)
    assert ended.state == "lost"
    assert ended.error == store_module.SESSION_ENDED_ERROR
    assert ended.cancel_requested is False
    assert ended.outputs["answer"]["result"] < 2000


def test_a_retry_of_a_job_still_running_is_answered_at_once_with_the_job(
    bench: Bench, module: Any
) -> None:
    bench.config = replace(bench.config, inline_wait_s=1)
    code = "hou.gate.wait(20)\nresult = 'once'"
    handle = ok(python(bench, code=code, operation_id="op-again"))
    assert handle["state"] == "running"
    # The same call again, willing to queue for twenty seconds: it is told
    # at once, and the code does not run twice.
    began = time.monotonic()
    again = ok(python(bench, code=code, operation_id="op-again", wait_s=20))
    assert time.monotonic() - began < 5.0
    assert through(bench).dispatcher.state()["busy"] is True
    assert again["job_id"] == handle["job_id"]
    assert again["state"] == "running"
    waiting = refused(python(bench, code=code, operation_id="op-again", background=False))
    assert waiting["code"] == "TIMEOUT"
    assert waiting["details"]["job_id"] == handle["job_id"]
    other = refused(python(bench, code="result = 2", operation_id="op-again", wait_s=0))
    assert other["code"] == "OPERATION_MISMATCH"
    module.gate.set()
    idle(bench)
    replayed = ok(python(bench, code=code, operation_id="op-again"))
    assert replayed["result"] == "once"
    assert replayed["state"] == "done"


def test_a_held_status_ends_on_time_while_another_process_holds_the_store(
    bench: Bench, module: Any
) -> None:
    handle = in_the_background(bench, "hou.gate.wait(20)")
    support.wait_until(lambda: row(bench, handle["job_id"]).state == "running", timeout_s=5.0)
    locker = sqlite3.connect(str(bench.store_path), timeout=0.1, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        began = time.monotonic()
        held = job(bench, handle["job_id"], wait_s=1.0)
        taken = time.monotonic() - began
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    assert taken < 2.5, taken
    assert held["state"] == "running"
    assert held["changed"] is False
    module.gate.set()
    idle(bench)


def test_a_cancel_waits_on_the_session_no_longer_than_a_health_check(
    bench: Bench, module: Any
) -> None:
    handle = in_the_background(bench, "hou.gate.wait(20)")
    job(bench, handle["job_id"], action="cancel")
    [sent] = [call for call in through(bench).calls if call["tool"] == "bridge.cancel"]
    assert sent["http_timeout_s"] == jobs_tool.CANCEL_SOCKET_S == 2.0
    module.gate.set()
    idle(bench)


def test_a_large_answer_spills_once_and_honours_max_chars(bench: Bench) -> None:
    body = ok(python(bench, code="result = 'x' * 50000"))
    first = job(bench, body["job_id"])
    spilled = first["outputs"]["spill_path"]
    assert Path(spilled).is_file()
    assert row(bench, body["job_id"]).spill_path == spilled
    again = job(bench, body["job_id"], max_chars=100)
    assert again["outputs"]["spill_path"] == spilled
    assert len(again["outputs"]["result"]) == 100
    assert again["outputs"]["elided_chars"] > 49_000
    files = list(Path(spilled).parent.glob("*.json"))
    third = job(bench, body["job_id"])
    assert third["outputs"]["spill_path"] == spilled
    assert list(Path(spilled).parent.glob("*.json")) == files


def _token(body: Any) -> str:
    text = json.dumps(body, separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def test_a_page_token_is_read_only_exactly_as_this_server_writes_it(bench: Bench) -> None:
    for index in range(3):
        with bench.store() as store:
            store.create_job(f"job-{index}", kind="python", session_id="s-1", state="done")
    first = ok(jobs(bench, action="list", limit=1))
    good = first["next_page"]
    body = json.loads(base64.urlsafe_b64decode(good + "=" * (-len(good) % 4)))
    assert set(body) == {"v", "c", "r", "q"}
    assert ok(jobs(bench, action="list", limit=1, page=good))["jobs"]
    bad = [
        _token({**body, "c": True}),
        _token({**body, "r": True}),
        _token({**body, "v": True}),
        _token({**body, "r": 0}),
        _token({**body, "r": -3}),
        _token({**body, "r": 2.0}),
        _token({**body, "c": "1.0"}),
        _token({**body, "extra": 1}),
        _token({key: value for key, value in body.items() if key != "q"}),
        _token([body]),
        base64.urlsafe_b64encode(json.dumps(body).replace(str(body["c"]), "Infinity").encode())
        .decode()
        .rstrip("="),
        base64.urlsafe_b64encode(json.dumps(body).replace(str(body["c"]), "1e999").encode())
        .decode()
        .rstrip("="),
        base64.urlsafe_b64encode(('{"v":1,"v":1,' + json.dumps(body)[1:]).encode())
        .decode()
        .rstrip("="),
        good + "=",
        good + "!",
        good.replace("-", "+").replace("_", "/") if ("-" in good or "_" in good) else good + "+",
        base64.urlsafe_b64encode(b"\xff\xfe").decode().rstrip("="),
        "",
    ]
    for token in bad:
        error = refused(jobs(bench, action="list", limit=1, page=token))
        assert error["code"] == "BAD_CURSOR", token
