#!/usr/bin/env python3
"""Run the Stage A acceptance checks that need no user interface.

Each check is a numbered step. A step passes or fails on its own evidence, and
what it says is a line a person can read without opening anything else: how
many processes asked, which one won, which error code came back. The run
records the system, the Houdini build and the date beside the results, because
a pass on one machine is a pass on one machine.

    scripts/acceptance_stage_a.py                 # every headless step
    scripts/acceptance_stage_a.py --only 7        # one step on its own
    scripts/acceptance_stage_a.py --out FOLDER    # where the report goes

The report lands outside the repository by default, under this user's state
folder, because it is a record of one machine on one day and not source.

Two of the eleven checks need a session a person can see, so they are recorded
as not run here rather than quietly dropped. Everything that needs Houdini is
skipped, not failed, when there is no hython on this machine.

House rules this script keeps to. Never more than three Houdinis at once, and
in practice one or two. Every session it starts is stopped again, including
after a failure. Every scene file it writes is inside its own working folder.
It never opens, saves or looks at a scene a person is working on.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# The package and the shared test machinery, so a checkout runs this without
# being installed first. The helpers are the ones the suite uses, so a check
# here and a test there start a session the same way.
for extra in (REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import support  # noqa: E402
from nscr_houdini_mcp import install as install_module  # noqa: E402
from nscr_houdini_mcp import outputs, pool  # noqa: E402
from nscr_houdini_mcp import store as store_module  # noqa: E402
from nscr_houdini_mcp.bridge import client, net, registry, security, signing  # noqa: E402
from nscr_houdini_mcp.bridge.launcher import HythonBridge, find_hython  # noqa: E402
from nscr_houdini_mcp.bridge.serving import CALL_PATH, HEALTH_PATH  # noqa: E402
from nscr_houdini_mcp.store import PoolFull, Store  # noqa: E402

REPORT_STEM = "stage_a_headless"
FORM_TYPE = "application/x-www-form-urlencoded"

# A range of this script's own, above every range the suite uses, so a run
# never takes a port from a session somebody started by hand.
PORTS = {
    1: (18430, 18439),
    2: (18440, 18443),
    3: (18444, 18447),
    4: (18448, 18451),
    5: (18452, 18459),
    6: (18460, 18469),
    7: (18470, 18473),
    9: (18474, 18479),
    12: (18480, 18489),
}

POOL_CAP = 3
RACERS = 8

# How long a session that is holding an answer keeps it, and how long a caller
# waits before it decides the answer is lost. The caller gives up first.
HOLD_S = 8.0
GIVE_UP_S = 2.0

# The lease for the worker that has to end itself with nobody watching.
SHORT_IDLE_S = 5.0
WARM_IDLE_S = 900.0
IDLE_EXIT_TIMEOUT_S = 180.0

HAMMER_SECONDS = 600.0
HAMMER_CLIENTS = 2

# How often the health poll asks during the hammer, and the longest an answer
# may take before the run says the session stopped answering.
HEALTH_EVERY_S = 0.5
HEALTH_LIMIT_S = 5.0

# The longest a hammer call asks to wait for its turn. The other caller can be
# holding the session, so it has to be a long wait, and the bridge refuses one
# longer than fifty seconds.
HAMMER_WAIT_S = 45.0

# How long a caller waits on the socket, how long the work it then asks to stop
# would take, and how many times one call is sent before the run says the
# session is not answering.
HAMMER_SOCKET_S = 60.0
HAMMER_SLEEP_S = 6.0
HAMMER_SENDS = 3

# A pause between rounds, so the run looks like two agents working rather than
# two loops sending as fast as the socket allows. A caller thinks between
# calls, and a session that is asked a hundred times a second from two
# processes is not the case this check is here to cover.
HAMMER_PAUSE_S = 0.25

SEEDED_TERM = "a-seeded-private-term"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# Section: what a run and a step are


@dataclass
class Result:
    """What one step did, in the words the report prints."""

    number: int
    title: str
    result: str
    evidence: str
    seconds: float = 0.0
    detail: str = ""


@dataclass
class Run:
    """Everything the steps share: where to work, and what was found."""

    work: Path
    report: Path
    hython: Path | None
    hammer_seconds: float = HAMMER_SECONDS
    started_at: datetime = field(default_factory=datetime.now)
    houdini_build: str = ""
    results: list[Result] = field(default_factory=list)

    def folder(self, name: str) -> Path:
        """A working folder of one step's own."""
        path = self.work / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def home(self, name: str) -> Path:
        """A state folder of one step's own, so no step sees another's."""
        return self.folder(f"{name}/home")

    def note_build(self, version: Any) -> None:
        """Record the build the first session that answered reported."""
        if version and not self.houdini_build:
            self.houdini_build = str(version)


@dataclass(frozen=True)
class Step:
    number: int
    title: str
    needs_houdini: bool
    run: Callable[[Run], str] | None = None
    note: str = ""


# Section: shared helpers


@contextmanager
def session(run: Run, name: str, *, number: int, **rest: Any) -> Iterator[HythonBridge]:
    """One hython session for one step, stopped again whatever happens."""
    with support.hython_session(run.home(name), port_range=PORTS[number], **rest) as bridge:
        run.note_build(bridge.entry.get("houdini_version") if bridge.entry else None)
        yield bridge


def scene_nodes(bridge: HythonBridge) -> int:
    answer = bridge.call("scene.info")
    assert answer.payload["ok"] is True, answer.payload
    return int(answer.payload["data"]["nodes"]["/obj"])


def pool_config(home: Path, *, number: int, max_idle_s: float = WARM_IDLE_S) -> pool.PoolConfig:
    return pool.PoolConfig(
        home=home,
        cap=POOL_CAP,
        max_idle_s=max_idle_s,
        port_range=PORTS[number],
        start_timeout_s=240.0,
        # The workers this script drives exist to be driven, so they carry the
        # tool that can be asked to take its time and to fail on purpose.
        selfcheck=True,
    )


def git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - git and our own paths
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
    )


def png_size(path: Path) -> tuple[int, int]:
    """The size a PNG says it is, read from its own header."""
    raw = path.read_bytes()[:24]
    if len(raw) < 24 or raw[:8] != PNG_MAGIC or raw[12:16] != b"IHDR":
        raise AssertionError(f"{path.name} is not a PNG")
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


# Section: the children other processes run


def ask_for_a_slot(home: str, hython: str, number: int, index: int, barrier: Any, results: Any):
    """One racer asking for the last worker slot at the same moment as the rest."""
    report: dict[str, Any] = {"index": index, "pid": os.getpid(), "error": None}
    try:
        config = pool_config(Path(home), number=number)
        with pool.open_store(home) as store:
            barrier.wait(support.BARRIER_TIMEOUT_S)
            try:
                record = pool.start_worker(config, store, hython=hython)
                report["alias"] = record.alias
                report["session_id"] = record.session_id
                report["token"] = record.token
            except PoolFull:
                report["alias"] = None
                report["full"] = True
    except BaseException as error:  # reported, so a failure reads as a message
        report["error"] = f"{type(error).__name__}: {error}"
    results.put(report)


def start_a_worker_and_leave(
    home: str, hython: str, number: int, idle_s: float, index: int, barrier: Any, results: Any
) -> None:
    """A client that starts a worker and then goes away, as an exit looks."""
    report: dict[str, Any] = {"index": index, "pid": os.getpid(), "error": None}
    try:
        config = pool_config(Path(home), number=number, max_idle_s=idle_s)
        with pool.open_store(home) as store:
            record = pool.start_worker(config, store, hython=hython)
        report["alias"] = record.alias
        report["session_id"] = record.session_id
        report["token"] = record.token
        report["worker_pid"] = record.pid
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    results.put(report)


def hammer_client(
    home: str, handle: str, seconds: float, index: int, barrier: Any, results: Any
) -> None:
    """One of the two callers that keep a session busy for the whole run.

    It mixes reads, mutations that carry an operation id, calls that refuse to
    wait, and calls it then asks to stop. Everything it sends is checked
    against what came back, and anything that does not match is counted and
    described rather than raised, so the run ends with a number instead of a
    stack.

    A reply that never arrives is counted and the call is sent again, which is
    what a client is meant to do: a read costs nothing twice, and a mutation
    carries the same operation id, so the second send is a receipt lookup
    rather than the work again. A call that has no answer after three sends is
    counted separately, because by then the session is not answering at all.
    """
    report: dict[str, Any] = {
        "index": index,
        "pid": os.getpid(),
        "error": None,
        "reads": 0,
        "mutations": 0,
        "created": 0,
        "skipped": 0,
        "cancelled": 0,
        "busy": 0,
        "lost": 0,
        "unanswered": 0,
        "wrong": [],
        "operation_ids": [],
    }
    try:
        session = client.Session.open(Path(home), handle)
        if barrier is not None:
            barrier.wait(support.BARRIER_TIMEOUT_S)
        deadline = time.monotonic() + seconds
        round_number = 0
        while time.monotonic() < deadline:
            round_number += 1
            _hammer_round(session, index, round_number, report)
            if report["unanswered"]:
                # The session has stopped answering. Carrying on for the rest
                # of the ten minutes would only count the same silence again.
                break
            time.sleep(HAMMER_PAUSE_S)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
    results.put(report)


def _send(report: dict[str, Any], what: str, work: Callable[[], Any]) -> Any | None:
    """Send one call, and send it again when no answer comes back."""
    for _ in range(HAMMER_SENDS):
        try:
            return work()
        except client.BridgeUnreachable:
            report["lost"] += 1
    report["unanswered"] += 1
    report["wrong"].append(f"{what} had no answer after {HAMMER_SENDS} sends")
    return None


def _hammer_round(session: Any, index: int, number: int, report: dict[str, Any]) -> None:
    """One pass of the mixture, with every answer checked against the ask."""
    echo = f"c{index}-{number}"
    answer = _send(
        report,
        f"ping {echo}",
        lambda: client.call(
            session,
            "bridge.ping",
            arguments={"echo": echo},
            wait_s=HAMMER_WAIT_S,
            http_timeout_s=HAMMER_SOCKET_S,
        ),
    )
    report["reads"] += 1
    if answer is not None and answer.payload.get("ok"):
        if answer.payload["data"].get("echo") != echo:
            report["wrong"].append(f"ping {echo} came back as {answer.payload['data'].get('echo')}")

    read = _send(
        report,
        "scene.info",
        lambda: client.call(
            session, "scene.info", wait_s=HAMMER_WAIT_S, http_timeout_s=HAMMER_SOCKET_S
        ),
    )
    report["reads"] += 1
    if read is not None and read.payload.get("ok"):
        if read.payload["data"]["session_id"] != session.session_id:
            report["wrong"].append("scene.info named another session")

    operation_id = client.new_operation_id()
    report["operation_ids"].append(operation_id)
    made = _send(
        report,
        operation_id,
        lambda: client.call(
            session,
            "bridge.selfcheck",
            arguments={"creates": 1},
            operation_id=operation_id,
            wait_s=HAMMER_WAIT_S,
            timeout_s=60.0,
            http_timeout_s=HAMMER_SOCKET_S,
        ),
    )
    report["mutations"] += 1
    if made is None:
        pass
    elif made.payload.get("ok"):
        created = made.payload["data"]["created"]
        # A replayed answer is the receipt of work that already counted.
        if not made.payload.get("replayed"):
            report["created"] += len(created)
        if len(created) != 1:
            report["wrong"].append(f"{operation_id} created {len(created)} nodes")
        if made.payload.get("operation_id") != operation_id:
            report["wrong"].append(f"{operation_id} came back under another id")
    else:
        report["wrong"].append(f"{operation_id} failed: {made.payload.get('error')}")

    if number % 3 == 0:
        refused = _send(
            report,
            "a call that would not wait",
            lambda: client.call(
                session, "bridge.ping", skip_if_busy=True, http_timeout_s=HAMMER_SOCKET_S
            ),
        )
        report["skipped"] += 1
        if refused is not None:
            code = (refused.payload.get("error") or {}).get("code")
            if code == "SESSION_BUSY":
                report["busy"] += 1
            elif not refused.payload.get("ok"):
                report["wrong"].append(f"a call that would not wait answered {code}")

    if number % 5 == 0:
        _hammer_cancel(session, report)


def _hammer_cancel(session: Any, report: dict[str, Any]) -> None:
    """Start work that takes a while, ask it to stop, and check that it did."""
    operation_id = client.new_operation_id()
    report["operation_ids"].append(operation_id)
    report["cancelled"] += 1
    holder: list[Any] = []
    sending = threading.Thread(
        target=lambda: holder.append(
            _send(
                report,
                operation_id,
                lambda: client.call(
                    session,
                    "bridge.selfcheck",
                    arguments={"sleep_s": HAMMER_SLEEP_S},
                    operation_id=operation_id,
                    wait_s=HAMMER_WAIT_S,
                    timeout_s=60.0,
                    http_timeout_s=HAMMER_SOCKET_S,
                ),
            )
        )
    )
    sending.start()
    try:
        asked = False
        deadline = time.monotonic() + HAMMER_SOCKET_S
        while time.monotonic() < deadline and not asked:
            try:
                state = client.health(session).payload["data"]
            except client.BridgeUnreachable:
                report["lost"] += 1
                continue
            if state.get("current_op_id") == operation_id:
                _send(
                    report,
                    f"the stop of {operation_id}",
                    lambda: client.call(
                        session,
                        "bridge.cancel",
                        arguments={"operation_id": operation_id},
                        http_timeout_s=HAMMER_SOCKET_S,
                    ),
                )
                asked = True
            else:
                time.sleep(0.25)
    finally:
        sending.join(HAMMER_SOCKET_S * 4)
    reply = holder[0] if holder else None
    if reply is None:
        return
    if reply.payload.get("ok"):
        if asked and not reply.payload.get("replayed"):
            if reply.payload["data"]["slept_s"] >= HAMMER_SLEEP_S:
                report["wrong"].append(f"{operation_id} was asked to stop and did not")
    else:
        code = (reply.payload.get("error") or {}).get("code")
        report["wrong"].append(f"a call that was asked to stop answered {code}")


# Section: check 1, concurrent starts


def check_concurrent_starts(run: Run) -> str:
    home = run.home("step1")
    hython = str(run.hython)
    with pool.open_store(home) as store:
        for slot in range(POOL_CAP - 1):
            store.reserve_worker(cap=POOL_CAP, token=f"held-{slot}")

    reports = support.run_children(ask_for_a_slot, RACERS, (str(home), hython, 1), barrier=True)
    winners = [report for report in reports if report.get("alias")]
    refused = [report for report in reports if report.get("full")]
    assert len({report["pid"] for report in reports}) == RACERS, "the racers shared a process"
    assert len(winners) == 1, f"{len(winners)} racers took the last slot"
    assert len(refused) == RACERS - 1, f"{len(refused)} were refused, not {RACERS - 1}"

    entry = registry.find_entry(home, str(winners[0]["session_id"]), remove_stale=False)
    assert entry is not None, "the winner left no session file"
    answer = client.health(client.Session.from_entry(entry))
    assert answer.payload["data"]["status"] == "ok", answer.payload
    run.note_build(entry.get("houdini_version"))

    # A start that fails has to hand its slot straight back.
    with pool.open_store(home) as store:
        pool.stop_worker(pool_config(home, number=1), store, str(winners[0]["token"]))
        before = len(store.list_workers())
        try:
            pool.start_worker(
                pool_config(home, number=1), store, hython=hython, spawn=_spawn_that_fails
            )
        except pool.WorkerStartFailed:
            pass
        else:
            raise AssertionError("a hython with no bridge in it counted as a start")
        after = len(store.list_workers())
    assert after == before, f"a failed start left {after - before} slots taken"

    left = support.stop_everything(home, pool_config(home, number=1))
    assert left == [], f"workers were left running: {left}"
    return (
        f"{RACERS} processes asked at once with one slot free: "
        f"1 started {winners[0]['alias']} and answered health ok, "
        f"{len(refused)} got POOL_FULL, and a failed start gave its slot back"
    )


def _spawn_that_fails(command: Sequence[str], *, log: Path, env: Any) -> pool.Launched:
    """A process that is really started and comes up with no bridge in it."""
    return pool.spawn_detached([sys.executable, "-c", "raise SystemExit(3)"], log=log, env=env)


# Section: check 2, a lost reply


def check_lost_reply(run: Run) -> str:
    with session(
        run, "step2", number=2, alias="acc-lost", extra_args=["--drop-reply-s", str(HOLD_S)]
    ) as bridge:
        before = scene_nodes(bridge)
        operation_id = client.new_operation_id()
        arguments = {"creates": 1, "drop_reply": True}

        began = time.monotonic()
        answer = bridge.call(
            "bridge.selfcheck",
            arguments=arguments,
            operation_id=operation_id,
            http_timeout_s=GIVE_UP_S,
        )
        waited = time.monotonic() - began
        assert waited >= GIVE_UP_S, "the first answer was not lost"
        assert answer.payload["ok"] is True, answer.payload
        assert answer.payload["replayed"] is True, "the retry did the work again"
        after = scene_nodes(bridge)
        assert after == before + 1, f"the scene holds {after - before} results, not one"

        again = bridge.call(
            "bridge.selfcheck", arguments=arguments, operation_id=operation_id, wait_s=10.0
        )
        assert again.payload["data"]["created"] == answer.payload["data"]["created"]
        assert scene_nodes(bridge) == after, "a third send changed the scene"

        wrong = bridge.call(
            "bridge.selfcheck", arguments={"creates": 2}, operation_id=operation_id, wait_s=10.0
        )
        code = wrong.payload["error"]["code"]
        assert code == "OPERATION_MISMATCH", code
        assert scene_nodes(bridge) == after, "the mismatched send changed the scene"
    return (
        "the answer was dropped after the work ran: the retry came back from the receipt "
        f"({waited:.1f} s), the scene holds one node not two, and the same id with other "
        "arguments got OPERATION_MISMATCH"
    )


# Section: check 3, a session that was killed and replaced


def check_bridge_restart(run: Run) -> str:
    home = run.home("step3")
    first = HythonBridge(home=home, port_range=PORTS[3], alias="acc-restart")
    first.start()
    old_id = first.session_id
    support.kill_bridge(first)
    with support.hython_session(home, port_range=PORTS[3], alias="acc-restart") as second:
        run.note_build(second.entry.get("houdini_version") if second.entry else None)
        assert second.session_id != old_id, "the replacement took the dead session's id"
        try:
            client.Session.open(home, old_id)
        except client.SessionDead as refused:
            details = refused.details()
        else:
            raise AssertionError("a call to the dead session was allowed through")
        assert details["alias"] == "acc-restart", details
        assert details["live_session_id"] == second.session_id, details
        # Nothing was sent, so the new session has run nothing at all.
        state = second.health().payload["data"]
        assert state["last_op"] is None, state["last_op"]
        assert client.Session.open(home, "acc-restart").session_id == second.session_id
    return (
        "a call carrying the killed session's id was refused with SESSION_DEAD and the new id, "
        "and the new process had still run nothing"
    )


# Section: check 4, a scene that was replaced


def check_scene_replacement(run: Run) -> str:
    with session(run, "step4", number=4, alias="acc-scene") as bridge:
        epoch = bridge.call("scene.info").payload["data"]["scene_epoch"]
        bridge.call("bridge.selfcheck", arguments={"creates": 1}, wait_s=10.0)

        cleared = bridge.call("bridge.selfcheck", arguments={"new_scene": True}, wait_s=10.0)
        assert cleared.payload["ok"] is True, cleared.payload
        assert cleared.payload["scene_epoch"] == epoch + 1

        refused = bridge.call("scene.info", scene_epoch=epoch)
        error = refused.payload["error"]
        assert error["code"] == "SCENE_REPLACED", error
        summary = refused.payload["scene"]
        assert summary["changed"] == "cleared", summary
        assert summary["nodes"]["/obj"] == 0, summary

        scratch = run.folder("step4/scene") / "acceptance.hip"
        bridge.call(
            "bridge.selfcheck",
            arguments={"save_hip": str(scratch)},
            wait_s=10.0,
            timeout_s=180.0,
        )
        bridge.call(
            "bridge.selfcheck",
            arguments={"load_hip": str(scratch)},
            wait_s=10.0,
            timeout_s=180.0,
        )
        stale = bridge.call("scene.info", scene_epoch=epoch + 1)
        assert stale.payload["error"]["code"] == "SCENE_REPLACED", stale.payload
        assert stale.payload["scene"]["changed"] == "loaded", stale.payload["scene"]
        assert bridge.call("scene.info", scene_epoch=epoch + 2).payload["ok"] is True
    return (
        f"a new scene moved the epoch {epoch} to {epoch + 1} and a load moved it to {epoch + 2}: "
        "both refused the old epoch with SCENE_REPLACED and a summary of the scene there is now"
    )


# Section: check 5, two sessions on one scene


def check_same_name_sessions(run: Run) -> str:
    home = run.home("step5")
    pattern = ["--alias-template", "acc-{n}"]
    low = (PORTS[5][0], PORTS[5][0] + 3)
    high = (PORTS[5][0] + 4, PORTS[5][1])
    with support.hython_session(home, port_range=low, extra_args=pattern) as first:
        with support.hython_session(home, port_range=high, extra_args=pattern) as second:
            run.note_build(first.entry.get("houdini_version") if first.entry else None)
            assert first.session_id != second.session_id, "two sessions took one id"
            aliases = [first.entry["alias"], second.entry["alias"]]
            assert aliases == ["acc-1", "acc-2"], aliases
            assert first.port != second.port, "two sessions took one port"
            for started in (first, second):
                data = started.health().payload["data"]
                assert data["status"] == "ok", data
                assert data["session_id"] == started.session_id
    assert registry.list_entries(home) == [], "a session file was left behind"
    return (
        f"two Houdinis on the same untitled scene took the names {aliases[0]} and {aliases[1]}, "
        "two different session ids and two different ports"
    )


# Section: check 6, a client that goes away


def check_client_exit(run: Run) -> str:
    home = run.home("step6")
    hython = str(run.hython)
    [report] = support.run_children(
        start_a_worker_and_leave, 1, (str(home), hython, 6, WARM_IDLE_S), barrier=False
    )
    alias = str(report["alias"])
    token = str(report["token"])
    session_id = str(report["session_id"])
    assert not store_module.process_is_alive(int(report["pid"])), "the client is still running"

    entry = registry.find_entry(home, session_id, remove_stale=False)
    assert entry is not None, "the warm worker left no session file"
    run.note_build(entry.get("houdini_version"))
    assert client.health(client.Session.from_entry(entry)).payload["data"]["status"] == "ok"

    # A server that was not there when the worker started sees it as it is.
    with pool.open_store(home) as store:
        rows = pool.list_workers(store)
        assert [row["alias"] for row in rows] == [alias], rows
        assert rows[0]["capabilities"] != "-", rows[0]
        pool.reserve(store, alias, job_id="acc-job")

    # A second client is ended in the middle of its own call, holding that job.
    marker = run.folder("step6") / "sent.txt"
    bridge_session = client.Session.from_entry(entry)

    def busy() -> bool:
        return client.health(bridge_session).payload["data"]["busy"] is True

    with support.client_that_dies_mid_call(
        home,
        session_id,
        tool="bridge.selfcheck",
        arguments={"creates": 1, "sleep_s": 12.0},
        marker=marker,
        ready=busy,
    ):
        operation_id = marker.read_text(encoding="utf-8").strip()
        support.wait_until(lambda: client.health(bridge_session).payload["data"]["busy"] is False)

    with pool.open_store(home) as store:
        record = store.get_operation(operation_id)
        held = store.get_worker(token)
        assert record is not None and record.state == "done", record
        assert held is not None and held.job_id == "acc-job", held
        pool.release(store, alias)
        left = support.stop_everything(home, pool_config(home, number=6))
    assert left == [], f"workers were left running: {left}"
    log = pool.log_path(home, alias)
    assert log.is_file() and log.stat().st_size > 0, "the worker log was not captured"

    # A worker nobody wants ends itself, with nothing watching it.
    [short] = support.run_children(
        start_a_worker_and_leave, 1, (str(home), hython, 6, SHORT_IDLE_S), barrier=False
    )
    idle_token = str(short["token"])
    worker_pid = int(short["worker_pid"])
    deadline = time.monotonic() + IDLE_EXIT_TIMEOUT_S
    state = None
    while time.monotonic() < deadline:
        with pool.open_store(home) as store:
            found = store.get_worker(idle_token)
        state = None if found is None else found.state
        if state == "stopped":
            break
        time.sleep(1.0)
    assert state == "stopped", f"the idle worker held its slot, state {state}"
    support.wait_until(lambda: not store_module.process_is_alive(worker_pid), timeout_s=60.0)
    return (
        f"the client that started {alias} exited and the worker stayed up and answered; "
        "a caller ended mid call left its work finished, its receipt done, its job still held "
        f"and {log.name} captured; a worker with a {SHORT_IDLE_S:g} s lease ended itself"
    )


# Section: check 7, two captures in one second


CAPTURE_SCRIPT = '''"""Write one flipbook frame to each path this is given.

Run by the acceptance script inside hython. It builds its own small scene, so
nothing on disk is read and nothing a person owns is touched.
"""

import json
import sys

import hou

paths = sys.argv[1:]

obj = hou.node("/obj")
geo = obj.createNode("geo", "acc_geo")
shape = geo.createNode("torus")
shape.setDisplayFlag(True)
shape.setRenderFlag(True)
camera = obj.createNode("cam", "acc_cam")
camera.parmTuple("t").set((0, 3, 12))
camera.parmTuple("r").set((-12, 0, 0))
light = obj.createNode("hlight", "acc_light")
light.parmTuple("t").set((3, 4, 3))

rop = hou.node("/out").createNode("flipbook", "acc_flipbook")
for name, value in (("trange", 0), ("camera", camera.path()), ("vobjects", "*"), ("tres", 1)):
    parm = rop.parm(name)
    if parm is not None:
        parm.set(value)
size = rop.parmTuple("res")
if size is not None:
    size.set((320, 180))

written = []
for path in paths:
    rop.parm("picture").set(path)
    rop.render(frame_range=(1, 1, 1), verbose=False)
    written.append(path)

print(json.dumps({"written": written}))
'''


def check_same_second_captures(run: Run) -> str:
    folder = run.folder("step7")
    scene = folder / "scene"
    scene.mkdir(exist_ok=True)
    hip = scene / "acceptance.hip"
    home = run.home("step7")

    with Store(pool.store_path(home)) as store:
        first = outputs.allocate(store, "capture", name="beauty", hip_path=hip)
        second = outputs.allocate(store, "capture", name="beauty", hip_path=hip)
    assert first.tokens["time"] == second.tokens["time"], "the two captures were not in one second"
    assert first.path != second.path, "two captures in one second took one path"
    assert first.run_id != second.run_id

    script = folder / "capture.py"
    script.write_text(CAPTURE_SCRIPT, encoding="utf-8")
    done = subprocess.run(  # noqa: S603 - hython and our own paths
        [str(run.hython), str(script), first.path, second.path],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, f"hython exited {done.returncode}:\n{done.stdout}\n{done.stderr}"

    sizes = []
    for plan in (first, second):
        made = Path(plan.path)
        assert made.is_file(), f"{made} was not written"
        assert made.stat().st_size > 0, f"{made} is empty"
        width, height = png_size(made)
        assert width > 0 and height > 0, f"{made} says it is {width}x{height}"
        sizes.append((made.name, made.stat().st_size, width, height))
    return (
        f"two captures named beauty at {first.tokens['time']} gave two files, "
        f"{sizes[0][0]} ({sizes[0][1]} bytes, {sizes[0][2]}x{sizes[0][3]}) and "
        f"{sizes[1][0]} ({sizes[1][1]} bytes, {sizes[1][2]}x{sizes[1][3]})"
    )


# Section: check 9, what the port gives away


def check_security(run: Run) -> str:
    with session(run, "step9", number=9, alias="acc-security") as bridge:
        proof = net.prove_loopback_only(bridge.port)
        assert proof.private is True, proof.as_dict()
        assert net.addresses_holding_port(bridge.port) == []
        outside = net.outward_addresses()
        for address in outside:
            assert not net.can_connect(address, bridge.port), f"{address} answered"

        for path in (HEALTH_PATH, CALL_PATH):
            unsigned = client.request(bridge.port, path, body=b"{}")
            assert unsigned.status == 401, f"{path} answered {unsigned.status} unsigned"
            assert unsigned.payload["error"]["code"] == "UNAUTHORIZED"

        impostor = bridge.session._replace(token=security.mint_token())
        wrong = client.post(impostor, HEALTH_PATH, {}, verify=False)
        assert wrong.status == 401, wrong.status

        for header in ("Origin", "Referer"):
            from_page = client.post(
                bridge.session,
                HEALTH_PATH,
                {},
                headers={header: "http://evil.example"},
                verify=False,
            )
            assert from_page.status == 403, f"{header} answered {from_page.status}"
            assert from_page.payload["error"]["code"] == "FORBIDDEN"
            reflected = [n for n in from_page.headers if n.startswith("access-control-allow")]
            assert reflected == [], reflected

        healthy = bridge.health()
        cors = [name for name in healthy.headers if name.startswith("access-control-allow")]
        assert cors == [], cors

        form = b"json=" + b"%5B" * 5000 + b"%5D" * 5000
        old_route = client.request(bridge.port, "/api", body=form, content_type=FORM_TYPE)
        assert old_route.status == 404, f"the built in route answered {old_route.status}"

        bomb = b"[" * 5000 + b"]" * 5000
        headers = signing.sign_request(
            bridge.session.token,
            method="POST",
            path=CALL_PATH,
            session_id=bridge.session_id,
            body=bomb,
        )
        answered = client.request(bridge.port, CALL_PATH, body=bomb, headers=headers)
        assert answered.status == 400, answered.status
        assert answered.payload["error"]["code"] == "BODY_REFUSED", answered.payload
        assert bridge.health().status == 200, "the bomb stopped the session answering"
        assert bridge.process is not None and bridge.process.poll() is None
        assert bridge.call("bridge.ping").payload["ok"] is True

        tested = ", ".join(outside) if outside else "no address outside loopback to test"
    return (
        f"the port is loopback only ({tested}), unsigned and wrongly signed requests get 401 on "
        "both paths, an Origin or Referer gets 403 with no allow header, /api is 404, and the "
        "nesting bomb was answered BODY_REFUSED with the session still serving"
    )


# Section: check 10, the leak guard


def check_leak_guard(run: Run) -> str:
    clone = run.folder("step10") / "clone"
    if clone.exists():
        shutil.rmtree(clone)
    subprocess.run(  # noqa: S603 - git and our own paths
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    git("config", "user.email", "acceptance@example.test", cwd=clone)
    git("config", "user.name", "acceptance", cwd=clone)
    subprocess.run(  # noqa: S603 - our own script
        [sys.executable, str(clone / "scripts" / "install_hooks.py")],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
    )
    terms = clone / ".context" / "leak-terms.txt"
    terms.parent.mkdir(parents=True, exist_ok=True)
    terms.write_text(f"{SEEDED_TERM}\n", encoding="utf-8")

    # A term in a staged file.
    staged = clone / "notes.txt"
    staged.write_text(f"a line that names {SEEDED_TERM} in passing\n", encoding="utf-8")
    git("add", "notes.txt", cwd=clone)
    blocked_file = git("commit", "-m", "Add a note", cwd=clone, check=False)
    assert blocked_file.returncode != 0, "a staged private term was committed"
    assert "leak guard" in blocked_file.stderr, blocked_file.stderr

    # A term in the commit message.
    staged.write_text("a line with nothing private in it\n", encoding="utf-8")
    git("add", "notes.txt", cwd=clone)
    blocked_message = git("commit", "-m", f"Add a note about {SEEDED_TERM}", cwd=clone, check=False)
    assert blocked_message.returncode != 0, "a private term in a message was committed"
    assert "leak guard" in blocked_message.stderr, blocked_message.stderr

    # And a clean change goes through, so the guard is not simply refusing all.
    allowed = git("commit", "-m", "Add a note", cwd=clone, check=False)
    assert allowed.returncode == 0, f"a clean commit was refused:\n{allowed.stderr}"

    lint = subprocess.run(  # noqa: S603 - our own script
        [sys.executable, str(clone / "scripts" / "lint_client_names.py")],
        cwd=clone,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lint.returncode == 0, f"the client name lint failed:\n{lint.stdout}\n{lint.stderr}"
    counted = lint.stdout.strip()
    shutil.rmtree(clone, ignore_errors=True)
    return (
        "in a throwaway clone: a seeded term in a staged file was blocked, the same term in a "
        f"commit message was blocked, a clean commit went through, and {counted}"
    )


# Section: the two client hammer


def check_hammer(run: Run) -> str:
    seconds = run.hammer_seconds
    home = run.home("step12")
    with support.hython_session(home, port_range=PORTS[12], alias="acc-hammer") as bridge:
        run.note_build(bridge.entry.get("houdini_version") if bridge.entry else None)
        before = scene_nodes(bridge)
        watch = _HealthWatch(bridge)
        watch.start()
        try:
            reports = support.run_children(
                hammer_client,
                HAMMER_CLIENTS,
                (str(home), bridge.session_id, seconds),
                barrier=True,
                result_timeout_s=seconds + 600.0,
                join_timeout_s=300.0,
            )
        finally:
            watch.stop()

        unanswered = sum(report["unanswered"] for report in reports)
        if unanswered or watch.failures:
            raise AssertionError(
                f"the session stopped answering: {unanswered} calls had no answer after "
                f"{HAMMER_SENDS} sends and {len(watch.failures)} health polls were lost. "
                + _fatal_note(run.home("step12"))
            )
        wrong = [line for report in reports for line in report["wrong"]]
        assert wrong == [], f"the callers saw wrong data: {wrong[:5]}"
        assert watch.slowest <= HEALTH_LIMIT_S, f"health took {watch.slowest:.2f} s"

        state = bridge.health().payload["data"]
        assert state["busy"] is False, state
        assert state["queued"] == 0, state
        created = sum(report["created"] for report in reports)
        after = scene_nodes(bridge)
        assert after == before + created, f"the scene holds {after - before} nodes, not {created}"

        operation_ids = [name for report in reports for name in report["operation_ids"]]
        with Store(pool.store_path(home)) as store:
            unsettled = [
                (name, None if record is None else record.state)
                for name in operation_ids
                for record in [store.get_operation(name)]
                if record is None or record.state not in ("done", "failed")
            ]
        assert unsettled == [], f"receipts left unsettled: {unsettled[:5]}"
        process = bridge.process
        assert process is not None and process.poll() is None, "the worker did not survive"
        reads = sum(report["reads"] for report in reports)
        mutations = sum(report["mutations"] for report in reports)
        skipped = sum(report["skipped"] for report in reports)
        busy = sum(report["busy"] for report in reports)
        cancelled = sum(report["cancelled"] for report in reports)
        lost = sum(report["lost"] for report in reports)

    assert registry.list_entries(home) == [], "the worker was left behind"
    return (
        f"{HAMMER_CLIENTS} callers for {seconds / 60:.0f} minutes against one worker: "
        f"{reads} reads, {mutations} mutations with operation ids, {skipped} calls that would "
        f"not wait ({busy} answered SESSION_BUSY), {cancelled} calls asked to stop, "
        f"{lost} answers lost on the socket and every one of them recovered by sending again, "
        f"{watch.polls} health polls with none lost and the slowest {watch.slowest * 1000:.0f} ms, "
        f"{len(operation_ids)} receipts all settled, {created} nodes for the work that ran, "
        "and the worker ended busy false, queued 0 and still answering"
    )


def _fatal_note(home: Path) -> str:
    """What the session's own output says, when it says the process died.

    A Houdini that ends in a fatal error writes it where the session was
    started from, and that line is the finding rather than the timeouts the
    callers saw afterwards.
    """
    log = Path(home) / "hython.log"
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if "Fatal Python error" in line:
            return f"The session reported: {line.strip()}"
    return ""


class _HealthWatch:
    """Ask a session whether it is alive, over and over, from outside it."""

    def __init__(self, bridge: HythonBridge) -> None:
        self._bridge = bridge
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="acceptance-health", daemon=True)
        self.polls = 0
        self.slowest = 0.0
        self.failures: list[str] = []

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(60.0)

    def _run(self) -> None:
        while not self._stop.wait(HEALTH_EVERY_S):
            began = time.monotonic()
            try:
                answer = self._bridge.health(timeout_s=HEALTH_LIMIT_S * 2)
                took = time.monotonic() - began
                if answer.status != 200:
                    self.failures.append(f"health answered {answer.status}")
            except Exception as error:  # noqa: BLE001 - a lost poll is the finding
                self.failures.append(f"{type(error).__name__}: {error}")
                continue
            self.polls += 1
            self.slowest = max(self.slowest, took)


# Section: the list of steps


STEPS: tuple[Step, ...] = (
    Step(1, "Concurrent starts", True, check_concurrent_starts),
    Step(2, "Lost reply", True, check_lost_reply),
    Step(3, "Bridge restart", True, check_bridge_restart),
    Step(4, "Scene replacement", True, check_scene_replacement),
    Step(5, "Same name sessions", True, check_same_name_sessions),
    Step(6, "Client exit", True, check_client_exit),
    Step(7, "Same second captures", True, check_same_second_captures),
    Step(8, "Busy during a long cook", False, None, "needs a session a person can see"),
    Step(9, "Security", True, check_security),
    Step(10, "Leak guard", False, check_leak_guard),
    Step(11, "One undo entry per call", False, None, "needs a session a person can see"),
    Step(12, "Two client hammer", True, check_hammer),
)


# Section: running them and writing it down


def run_step(step: Step, run: Run) -> Result:
    if step.run is None:
        return Result(step.number, step.title, "gui", f"gui, not run here: {step.note}")
    if step.needs_houdini and run.hython is None:
        return Result(step.number, step.title, "skipped", "no hython on this machine")
    began = time.monotonic()
    try:
        evidence = step.run(run)
    except KeyboardInterrupt:
        raise
    except BaseException as error:  # noqa: BLE001 - a failed step is a result
        took = time.monotonic() - began
        return Result(
            step.number,
            step.title,
            "fail",
            f"{type(error).__name__}: {error}".splitlines()[0][:400],
            took,
            traceback.format_exc(),
        )
    return Result(step.number, step.title, "pass", evidence, time.monotonic() - began)


def report_data(run: Run) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in run.results:
        counts[result.result] = counts.get(result.result, 0) + 1
    return {
        "stage": "A",
        "date": run.started_at.strftime("%Y-%m-%d"),
        "started_at": run.started_at.isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "houdini_build": run.houdini_build or "none started",
        "hython": str(run.hython) if run.hython else None,
        "hammer_seconds": run.hammer_seconds,
        "counts": counts,
        "steps": [
            {
                "number": result.number,
                "title": result.title,
                "result": result.result,
                "evidence": result.evidence,
                "seconds": round(result.seconds, 1),
                "detail": result.detail,
            }
            for result in run.results
        ],
    }


def write_report(run: Run) -> tuple[Path, Path]:
    data = report_data(run)
    run.report.mkdir(parents=True, exist_ok=True)
    as_json = run.report / f"{REPORT_STEM}.json"
    as_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage A acceptance, headless steps",
        "",
        f"- Date: {data['date']}",
        f"- System: {data['os']}",
        f"- Houdini: {data['houdini_build']}",
        f"- Python: {data['python']}",
        f"- Hammer: {data['hammer_seconds'] / 60:.0f} minutes",
        "",
        "| # | Check | Result | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for step in data["steps"]:
        evidence = step["evidence"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {step['number']} | {step['title']} | {step['result']} | {evidence} |")
    failed = [step for step in data["steps"] if step["result"] == "fail"]
    if failed:
        lines += ["", "## What failed", ""]
        for step in failed:
            lines += [f"### {step['number']} {step['title']}", "", "```", step["detail"], "```", ""]
    as_markdown = run.report / f"{REPORT_STEM}.md"
    as_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return as_json, as_markdown


def default_report_dir() -> Path:
    """Outside the repository, beside the rest of this user's state."""
    return store_module.default_home() / "reports" / f"{datetime.now():%Y-%m-%d}_stage_a_headless"


def find_hython_or_none() -> Path | None:
    try:
        return find_hython()
    except Exception:  # noqa: BLE001 - no Houdini is a skip, not a failure
        return None


def install_version() -> str:
    installs = install_module.find_installs()
    return installs[0].version if installs else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=None, help="folder for the report, outside the repo by default"
    )
    parser.add_argument("--only", type=int, action="append", help="run one step by number")
    parser.add_argument(
        "--work", type=Path, default=None, help="working folder, a temporary one by default"
    )
    parser.add_argument("--keep-work", action="store_true", help="leave the working folder behind")
    parser.add_argument(
        "--hammer-seconds",
        type=float,
        default=HAMMER_SECONDS,
        help="how long the two client hammer runs, for a shorter trial run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wanted = set(args.only or [step.number for step in STEPS])
    unknown = wanted - {step.number for step in STEPS}
    if unknown:
        print(f"no such step: {sorted(unknown)}", file=sys.stderr)
        return 2

    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="nscr-mcp-acceptance-"))
    work.mkdir(parents=True, exist_ok=True)
    run = Run(
        work=work,
        report=Path(args.out) if args.out else default_report_dir(),
        hython=find_hython_or_none(),
        hammer_seconds=args.hammer_seconds,
    )
    run.houdini_build = install_version()
    print(f"working in {work}")
    print(f"hython: {run.hython or 'none on this machine'}")

    for step in STEPS:
        if step.number not in wanted:
            continue
        print(f"step {step.number}: {step.title} ... ", end="", flush=True)
        result = run_step(step, run)
        run.results.append(result)
        print(f"{result.result} ({result.seconds:.0f} s)")
        print(f"    {result.evidence}")

    as_json, as_markdown = write_report(run)
    print(f"\nreport: {as_markdown}")
    print(f"        {as_json}")
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if any(result.result == "fail" for result in run.results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
