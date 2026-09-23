"""`hou_jobs`: follow, wait on, cancel and list long running work.

A job is the operation that runs it, and its row lives in the coordination
store, so any server process can answer for any job, including one started
since the call that began it. Three actions.

- `status`, the default: one job. With `wait_s` the call is held here until
  the job's state or progress changes or the wait ends, and says which with
  `changed`. The store is read a few times a second while it waits; nothing
  is sent to the session. A job that has already ended answers at once.
- `cancel`: writes the request on the row, which the session reads within two
  seconds and turns into `mcp.cancelled()`, and asks the session directly as
  well when it can be reached. The reply says whether the direct request got
  there. A job that ignores the request runs to the end and ends `done` with
  `cancel_requested` still set. In a session with a user interface the request
  is best effort: the code runs on Houdini's main thread and stops only where
  it looks.
- `list`: jobs newest first, filtered by session and state, a page at a time
  with `next_page`.

What `status` and `cancel` answer, for every kind of job: `job_id`, `state`,
`kind`, `session` and `alias`, `progress` as `done`, `total` and `message`,
`outputs`, `error`, `started_at`, `ended_at`, `elapsed_s`, `scene_epoch` and
`cancel_requested`, and `export_path` once the readable copy beside the scene
has been written. For a Python job that has ended, `outputs` is the answer the
call would have given, read back from the job row and fitted to `max_chars`.
What does not fit is spilled once per job, and later looks name that file.

A job is `lost` only when its session is known to have ended: its process
is gone, its row says it ended, or it was stopped. Silence alone never ends
a job, since a long cook can hold the interpreter and a machine can sleep;
a job that has gone quiet says for how long in `silent_s`. Every call brings
its own job up to date, and one round of upkeep a minute per store, taken by
whichever process claims it, settles and marks the rest and drops jobs that
ended more than seven days ago. A held status only reads while it waits,
and nothing it writes waits past the end of the hold.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from nscr_houdini_mcp import jobs as job_rules
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.router import DEAD_STATES
from nscr_houdini_mcp.store import JobRecord
from nscr_houdini_mcp.tools import python as python_tool
from nscr_houdini_mcp.tools.base import SESSION, Call, ToolSpec, inputs, outputs

ACTIONS = ("status", "cancel", "list")
STATES = ("queued", "running", "done", "failed", "cancelled", "lost")
FINAL = store_module.JOB_FINAL_STATES

# How often a held status reads the row, and how often it looks, reading
# only, at whether the job's session is still there.
POLL_S = 0.25
HOLD_CHECK_S = 2.0
# How often a held status tells a client that asked to hear that it is still
# waiting, which keeps a client that times out on silence waiting too.
HOLD_NOTE_S = 5.0

# Upkeep of the whole table: at most one round a minute per store, whichever
# process takes it.
SWEEP_NAME = "jobs"
SWEEP_EVERY_S = 60.0

# The longest any write here waits on a busy store, and the least.
BUSY_S = 10.0
MIN_BUSY_S = 0.05

# The session is asked to stop a job at once or not at all: a hung session
# is not waited on, since the request in the store reaches it anyway.
CANCEL_SOCKET_S = 2.0

# How far back a sweep looks for lost jobs whose readable copy is missing.
EXPORT_LOOKBACK_S = 3600.0
EXPORT_BATCH = 20

DEFAULT_LIMIT = 20
MAX_LIMIT = 200

# The page token: a version, the last row's creation time and place, and a
# fingerprint of the filters the list was read with.
TOKEN_VERSION = 1
MAX_TOKEN_CHARS = 400

# The clock and the pause a held status uses, kept here so a check can move them.
monotonic: Callable[[], float] = time.monotonic
sleep: Callable[[float], None] = time.sleep


def jobs(call: Call) -> Mapping[str, Any]:
    action = call.arguments.get("action") or "status"
    return ACTION_HANDLERS[action](call)


# Section: status and cancel


def status(call: Call) -> dict[str, Any]:
    job_id = wanted_id(call, "status")
    wait_s = float(call.arguments.get("wait_s") or 0.0)
    began = monotonic()
    # A write before or during a hold waits no longer than the hold would.
    busy_s = min(BUSY_S, wait_s) if wait_s > 0 else BUSY_S
    if not call.router.store_path.is_file():
        raise unknown(job_id)
    maintain(call, busy_s=busy_s)
    with capped(call, busy_s) as store:
        record = settle_one(call, found(store, job_id), busy_s=busy_s)
        changed = None
        if wait_s > 0:
            record, changed = hold(call, store, record, began + wait_s)
        row = job_row(record, call, full=True)
    note(call, record)
    if changed is not None:
        row["changed"] = changed
    return row


def cancel(call: Call) -> dict[str, Any]:
    job_id = wanted_id(call, "cancel")
    if not call.router.store_path.is_file():
        raise unknown(job_id)
    maintain(call, busy_s=BUSY_S)
    with capped(call, BUSY_S) as store:
        record = settle_one(call, found(store, job_id), busy_s=BUSY_S)
        if record.state not in FINAL:
            record = stored(lambda: store.request_job_cancel(job_id))
    if record.state in FINAL:
        row = job_row(record, call, full=True)
        row["cancel"] = {
            "requested": False,
            "reason": f"the job has already ended {record.state}",
        }
        note(call, record)
        return row
    reached = ask_session(call, record)
    with capped(call, BUSY_S) as store:
        record = found(store, job_id)
    row = job_row(record, call, full=True)
    note(call, record)
    row["cancel"] = {"requested": True, **reached}
    return row


def ask_session(call: Call, record: JobRecord) -> dict[str, Any]:
    """Ask the session running the job to stop it now, rather than at its next look."""
    if not record.session_id or not record.operation_id:
        return {"reached_session": False, "reason": "the job names no session call to stop"}
    try:
        target = call.router.resolve(record.session_id)
        reply = call.router.call(
            target,
            "bridge.cancel",
            {"operation_id": record.operation_id},
            wait_s=0,
            socket_s=CANCEL_SOCKET_S,
        )
    except CallError as error:
        return {
            "reached_session": False,
            "reason": error.code,
            "note": "the session reads the request from the store within two seconds",
        }
    data = reply.get("data") if isinstance(reply.get("data"), Mapping) else {}
    said: dict[str, Any] = {"reached_session": True, "asked": bool(data.get("asked"))}
    if data.get("reason"):
        said["reason"] = data["reason"]
    return said


def hold(
    call: Call, store: store_module.Store, record: JobRecord, deadline: float
) -> tuple[JobRecord, bool]:
    """Wait here until the job's state or progress changes, or the deadline.

    Nothing is written while it waits: the row is read a few times a second
    through a handle opened before the wait, and every couple of seconds the
    job's session is looked at, reading only. A session found gone ends the
    wait, and the job is settled then, with no write allowed to wait past
    the deadline.
    """
    if record.state in FINAL:
        return record, False
    mark = signature(record)
    job_id = record.job_id
    began = monotonic()
    next_check = began + HOLD_CHECK_S
    next_note = began + HOLD_NOTE_S
    while True:
        left = deadline - monotonic()
        if left <= 0:
            return record, False
        sleep(min(POLL_S, left))
        now = monotonic()
        if now >= next_note:
            call.progress(
                round(now - began, 1), round(deadline - began, 1), f"job {job_id} is {record.state}"
            )
            next_note = now + HOLD_NOTE_S
        current = stored(lambda: store.get_job(job_id))
        if current is None:
            raise unknown(job_id)
        record = current
        gone = False
        if now >= next_check:
            gone = stored(partial(session_gone, store, current))
            next_check = now + HOLD_CHECK_S
        if gone:
            record = settle_one(call, record, busy_s=max(0.0, deadline - monotonic()))
        if signature(record) != mark:
            return record, True
        if gone:
            return record, False


def signature(record: JobRecord) -> tuple[str, str]:
    return record.state, json.dumps(record.progress, sort_keys=True, default=str)


def note(call: Call, record: JobRecord) -> None:
    """Put the job's session in the trace, as every tool puts the session it reached."""
    scene = record.scene if isinstance(record.scene, dict) else {}
    call.trace.update(
        {
            "session_id": record.session_id,
            "alias": scene.get("alias"),
            "scene_epoch": scene.get("scene_epoch"),
        }
    )


# Section: list


def list_jobs(call: Call) -> dict[str, Any]:
    arguments = call.arguments
    limit = int(arguments.get("limit") or DEFAULT_LIMIT)
    state = arguments.get("state")
    handle = arguments.get("session")
    query = fingerprint(handle, state)
    before = read_token(arguments.get("page"), query)
    maintain(call, busy_s=BUSY_S)
    with call.router.store() as store:
        if store is None:
            return {"jobs": [], "next_page": None}
        session_ids = sessions_named(store, handle) if handle else None
        records = stored(
            lambda: store.list_jobs(
                session_ids=session_ids,
                states=[state] if state else None,
                limit=limit + 1,
                before=before,
            )
        )
    more = len(records) > limit
    records = records[:limit]
    now = time.time()
    rows = [list_row(record, now) for record in records]
    last = records[-1] if records else None
    token = make_token(last, query) if more and last is not None else None
    return {"jobs": rows, "next_page": token}


def sessions_named(store: store_module.Store, handle: str) -> list[str]:
    """Every session a list filter means: one id, or every session under an alias."""
    handle = handle.strip()
    records = stored(lambda: store.list_sessions(include_gone=True))
    by_id = [record.session_id for record in records if record.session_id == handle]
    if by_id:
        return by_id
    return [record.session_id for record in records if record.alias == handle]


def list_row(record: JobRecord, now: float) -> dict[str, Any]:
    """One job in a list: who and how it is, not what it made."""
    scene = record.scene if isinstance(record.scene, dict) else {}
    started = record.started_at or record.created_at
    row: dict[str, Any] = {
        "job_id": record.job_id,
        "state": record.state,
        "kind": record.kind,
        "session": record.session_id,
        "alias": scene.get("alias"),
        "started_at": started,
        "ended_at": record.finished_at,
        "elapsed_s": round(max(0.0, (record.finished_at or now) - started), 3),
    }
    progress = job_rules.progress_of(record.progress)
    if progress is not None:
        row["progress"] = progress
    if record.cancel_requested:
        row["cancel_requested"] = True
    if isinstance(record.error, Mapping):
        row["error"] = record.error.get("code") or record.error.get("type")
    return row


def fingerprint(handle: Any, state: Any) -> str:
    text = json.dumps({"session": handle, "state": state}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_token(last: JobRecord, query: str) -> str:
    body = {"v": TOKEN_VERSION, "c": last.created_at, "r": last.seq, "q": query}
    text = json.dumps(body, separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


TOKEN_KEYS = frozenset({"v", "c", "r", "q"})
_TOKEN_TEXT = re.compile(r"[A-Za-z0-9_-]+")


def read_token(page: Any, query: str) -> tuple[float, int] | None:
    """The row a list goes on after, from a page token this server wrote.

    Only a token exactly as `make_token` writes it is read: plain base64 in
    the URL alphabet, JSON with the four keys and no others, a version, a
    finite creation time, a place above zero and the filters' fingerprint.
    Anything else, a true for a number included, is `BAD_CURSOR`.
    """
    if page is None:
        return None
    unreadable = CallError(
        "BAD_CURSOR", "the page token could not be read", details={"argument": "page"}
    )
    if not isinstance(page, str) or len(page) > MAX_TOKEN_CHARS or not _TOKEN_TEXT.fullmatch(page):
        raise unreadable
    try:
        padded = page + "=" * (-len(page) % 4)
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != page:
            raise ValueError("not the way this server writes a token")
        body = json.loads(
            raw.decode("utf-8"),
            parse_constant=_no_constant,
            object_pairs_hook=_no_repeats,
        )
    except (binascii.Error, UnicodeError, ValueError):
        raise unreadable from None
    if not isinstance(body, dict) or set(body) != TOKEN_KEYS:
        raise unreadable
    version, created, place, fingerprint_ = body["v"], body["c"], body["r"], body["q"]
    if type(version) is not int or version != TOKEN_VERSION:
        raise unreadable
    if type(created) not in (int, float) or not math.isfinite(created):
        raise unreadable
    if type(place) is not int or place <= 0 or type(fingerprint_) is not str:
        raise unreadable
    if fingerprint_ != query:
        raise CallError(
            "BAD_CURSOR",
            "that page token belongs to a list with other filters; send the same session and state",
            details={"argument": "page"},
        )
    return float(created), place


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a number a token holds")


def _no_repeats(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("a key appears twice")
    return dict(pairs)


# Section: one job as a caller reads it


def job_row(record: JobRecord, call: Call, *, full: bool) -> dict[str, Any]:
    scene = record.scene if isinstance(record.scene, dict) else {}
    started = record.started_at or record.created_at
    now = time.time()
    row: dict[str, Any] = {
        "job_id": record.job_id,
        "state": record.state,
        "kind": record.kind,
        "session": record.session_id,
        "alias": scene.get("alias"),
        "progress": job_rules.progress_of(record.progress),
        "outputs": record.outputs,
        "error": record.error,
        "started_at": started,
        "ended_at": record.finished_at,
        "elapsed_s": round(max(0.0, (record.finished_at or now) - started), 3),
        "scene_epoch": scene.get("scene_epoch"),
        "cancel_requested": record.cancel_requested,
        "operation_id": record.operation_id,
    }
    if record.state not in FINAL and record.heartbeat_at is not None:
        # How long the session has said nothing about a job still going. It
        # is a sign to look, not an ending: silence never makes a job lost.
        row["silent_s"] = round(max(0.0, now - record.heartbeat_at), 1)
    if full and record.kind == "python" and record.state in FINAL:
        answer = python_answer(call, record)
        if answer is not None:
            row["outputs"] = answer
            row["error"] = answer.get("error", record.error)
    if record.state in FINAL:
        path = job_rules.export_path(record)
        if path is not None:
            row["export_path"] = path
    return row


def python_answer(call: Call, record: JobRecord) -> dict[str, Any] | None:
    """The answer a Python job's call gave, as the job row keeps it.

    The row holds the whole answer, written in the same step as the call's
    receipt, so this never needs the receipt, which goes sooner.
    """
    kept = record.outputs if isinstance(record.outputs, Mapping) else {}
    answer = kept.get("answer")
    if not isinstance(answer, Mapping):
        return None
    reply = {key: kept[key] for key in ("undo", "scene_epoch", "cut", "lossy") if key in kept}
    reply["data"] = answer
    spec = record.spec if isinstance(record.spec, dict) else {}
    budget = int(call.arguments.get("max_chars") or python_tool.DEFAULT_MAX_CHARS)
    said = python_tool.shape(
        call, reply, budget=budget, named=spec.get("namespace"), spilled=record.spill_path
    )
    written = said.get("spill_path")
    if written and written != record.spill_path:
        # The first look that spilled keeps its file; the next ones use it.
        try:
            with capped(call, BUSY_S) as store:
                store.note_job_paths(record.job_id, spill_path=written)
        except (store_module.StoreError, sqlite3.Error, CallError):
            pass
    said["operation_id"] = record.operation_id
    return said


def reconcile(store: store_module.Store, record: JobRecord) -> JobRecord | None:
    """Take a job's ending from its call's receipt, when the receipt has one.

    The session writes both in one step, so this is for a row an older
    session wrote, or one whose joint write did not land and whose receipt
    was written alone. A job found lost takes it as a late finish.
    """
    if not record.operation_id:
        return None
    receipt = store.get_operation(record.operation_id)
    if receipt is None or receipt.state not in ("done", "failed"):
        return None
    outcome = receipt.outcome
    if not isinstance(outcome, Mapping):
        return None
    state, outputs, error = job_rules.ending(record.kind, record.operation_id, outcome)
    try:
        return store.finish_job(record.job_id, state=state, outputs=outputs, error=error)
    except store_module.JobMoveRefused:
        return None


# Section: upkeep


def capped(call: Call, busy_s: float) -> store_module.Store:
    """A store handle whose writes wait on a busy store no longer than `busy_s`."""
    try:
        return store_module.Store(
            call.router.store_path, busy_timeout_s=max(MIN_BUSY_S, min(BUSY_S, busy_s))
        )
    except (store_module.StoreError, sqlite3.Error, OSError) as error:
        raise CallError(
            "STORE_UNAVAILABLE",
            "the coordination store could not be read",
            details={"exception": type(error).__name__},
        ) from None


def session_gone(store: store_module.Store, record: JobRecord) -> bool:
    """Whether the session running a job is known to have ended. Reads only.

    Silence is not an ending: a session whose process is there is running
    its job, however long it has said nothing, because a long cook can hold
    the interpreter and a machine can sleep.
    """
    if record.state in FINAL:
        return False
    if record.session_id:
        session = store.get_session(record.session_id)
        if session is None or session.state in DEAD_STATES:
            return True
        if store_module.same_process(session.pid, session.pid_start) is False:
            return True
    return record.worker_pid is not None and not store_module.process_is_alive(record.worker_pid)


def settle_one(call: Call, record: JobRecord, *, busy_s: float) -> JobRecord:
    """Bring one job up to date: its ending from its receipt, or lost with its session.

    Writes only when there is something to write, and waits on a busy store
    no longer than `busy_s`.
    """
    if record.state in FINAL:
        return record
    try:
        with capped(call, busy_s) as store:
            receipt = store.get_operation(record.operation_id) if record.operation_id else None
            settled = receipt is not None and receipt.state in ("done", "failed")
            if not settled and not session_gone(store, record):
                return record
            ended = reconcile(store, record)
            if ended is not None:
                return ended
            store.reclaim_sessions()
            store.lose_jobs([record.job_id], error=store_module.SESSION_ENDED_ERROR)
            return store.get_job(record.job_id) or record
    except (store_module.StoreError, sqlite3.Error, CallError):
        # Out of time for a busy store: the job is as it was read, and the
        # next look settles it.
        return record


def maintain(call: Call, *, busy_s: float) -> None:
    """One round of upkeep for the whole table, when this process gets the turn.

    The turn is a claim in the store, at most one a minute per store, so
    several servers on one store take turns rather than all sweeping. Upkeep
    that meets a busy store gives up rather than holding up the call.
    """
    try:
        with capped(call, busy_s) as store:
            if store.take_sweep(SWEEP_NAME, SWEEP_EVERY_S):
                sweep(store, call.router.home)
    except (store_module.StoreError, sqlite3.Error, CallError):
        return


def sweep(store: store_module.Store, home: Any) -> None:
    """Settle what has ended, mark what is lost, and prune what is old.

    A job whose call's receipt says it ended takes that ending. A job whose
    session is gone, whose process is gone, or which was running in a
    session that has ended, is `lost`. A job nobody has heard from in a while
    is left as it is, with how long it has been silent in its status. Rows
    that ended more than a week ago go.
    """
    for record in store.list_jobs(states=list(store_module.JOB_LIVE_STATES), limit=MAX_LIMIT):
        reconcile(store, record)
    # A session whose process has gone loses its jobs as it is marked.
    store.reclaim_sessions()
    for record in store.list_jobs(states=list(store_module.JOB_LIVE_STATES), limit=MAX_LIMIT):
        if session_gone(store, record):
            store.lose_jobs([record.job_id], error=store_module.SESSION_ENDED_ERROR)
    # A job lost with its session may still have ended before it went.
    for record in store.list_jobs(states=["lost"], limit=EXPORT_BATCH):
        reconcile(store, record)
    export_lost(store, home)
    for record in store.prune_final_jobs(job_rules.KEEP_S):
        job_rules.remove_export(record)


def export_lost(store: store_module.Store, home: Any) -> None:
    """Write the readable copy of recently lost jobs that have none yet.

    A job that ends on its own writes its copy from the session. One that is
    lost cannot, so whoever finds it lost writes it, for a job its caller was
    handed to follow. Best effort.
    """
    try:
        recent = store.list_jobs(states=["lost"], limit=EXPORT_BATCH)
    except (store_module.StoreError, sqlite3.Error):
        return
    now = time.time()
    for record in recent:
        if record.finished_at is not None and now - record.finished_at > EXPORT_LOOKBACK_S:
            continue
        if not record.promoted or job_rules.export_path(record) is not None:
            continue
        try:
            job_rules.export(store, record.job_id, home=home)
        except Exception:  # noqa: BLE001 - the row is the record; the copy is a courtesy
            continue


# Section: helpers


def wanted_id(call: Call, action: str) -> str:
    job_id = call.arguments.get("job_id")
    if not job_id:
        raise CallError(
            "BAD_ARGUMENTS",
            f"{action} needs job_id: the id a hou_python call or a list gave",
            details={"argument": "job_id"},
        )
    return str(job_id)


def found(store: store_module.Store, job_id: str) -> JobRecord:
    record = stored(lambda: store.get_job(job_id))
    if record is None:
        raise unknown(job_id)
    return record


def unknown(job_id: str) -> CallError:
    return CallError(
        "JOB_UNKNOWN",
        f"no job {job_id} is kept",
        details={"job_id": job_id, "kept_days": int(job_rules.KEEP_S // 86400)},
    )


def stored(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except (store_module.StoreError, sqlite3.Error) as error:
        raise CallError(
            "STORE_UNAVAILABLE",
            "the coordination store could not be read",
            details={"exception": type(error).__name__},
        ) from None


ACTION_HANDLERS: dict[str, Callable[[Call], dict[str, Any]]] = {
    "status": status,
    "cancel": cancel,
    "list": list_jobs,
}


def summary_line(data: Mapping[str, Any]) -> str:
    """What a client that reads only text is shown of a long result."""
    if "jobs" in data:
        more = "; more with next_page" if data.get("next_page") else ""
        return f"hou_jobs: {len(data['jobs'])} jobs{more}"
    return f"hou_jobs: {data.get('job_id')} is {data.get('state')}"


HOU_JOBS = ToolSpec(
    name="hou_jobs",
    description=(
        "Status, wait, cancel or list long running jobs. wait_s holds the call until the "
        "job changes state, so do not poll with sleeps. Job ids are kept for 7 days."
    ),
    input_schema=inputs(
        {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "job_id": {"type": "string"},
            "wait_s": {"type": "number", "minimum": 0, "maximum": 50},
            "session": SESSION,
            "state": {"type": "string", "enum": list(STATES)},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT},
            "page": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": python_tool.MAX_MAX_CHARS},
        }
    ),
    output_schema=outputs({}),
    handler=jobs,
    open_world=False,
    summary=summary_line,
)
