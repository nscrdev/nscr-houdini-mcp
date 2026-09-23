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
call would have given, read back from its receipt, fitted to the usual budget.

Every call sweeps first, at most once a second per process: a session whose
process is gone takes its unfinished jobs with it as `lost`, and so does a job
nobody has heard from for fifteen minutes. Rows older than seven days go once
an hour.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
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

# How often a held status reads the row, and how often it sweeps meanwhile.
POLL_S = 0.25
HOLD_SWEEP_S = 2.0
# How often a held status tells a client that asked to hear that it is still
# waiting, which keeps a client that times out on silence waiting too.
HOLD_NOTE_S = 5.0

# The sweeps, per process and per store: at most one a second, and pruning
# at most one an hour.
SWEEP_EVERY_S = 1.0
PRUNE_EVERY_S = 3600.0

# How far back a sweep looks for lost jobs whose readable copy is missing.
EXPORT_LOOKBACK_S = 3600.0
EXPORT_BATCH = 20

DEFAULT_LIMIT = 20
MAX_LIMIT = 200

# The page token: a version, the last row's creation time and place, and a
# fingerprint of the filters the list was read with.
TOKEN_VERSION = 1
MAX_TOKEN_CHARS = 400

SILENT_ERROR = {
    "code": "JOB_SILENT",
    "message": "the session running this job has said nothing for too long",
}

_swept: dict[str, float] = {}
_pruned: dict[str, float] = {}
_sweep_lock = threading.Lock()

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
    with call.router.store() as store:
        if store is None:
            raise unknown(job_id)
        sweep(store, call)
        record = found(store, job_id)
        changed = None
        if wait_s > 0:
            record, changed = hold(store, call, record, wait_s)
        row = job_row(record, call, store=store, full=True)
    note(call, record)
    if changed is not None:
        row["changed"] = changed
    return row


def cancel(call: Call) -> dict[str, Any]:
    job_id = wanted_id(call, "cancel")
    with call.router.store() as store:
        if store is None:
            raise unknown(job_id)
        sweep(store, call)
        record = found(store, job_id)
        if record.state in FINAL:
            row = job_row(record, call, store=store, full=True)
            row["cancel"] = {
                "requested": False,
                "reason": f"the job has already ended {record.state}",
            }
            note(call, record)
            return row
        record = stored(lambda: store.request_job_cancel(job_id))
    reached = ask_session(call, record)
    with call.router.store() as store:
        record = found(store, job_id) if store is not None else record
        row = job_row(record, call, store=store, full=True)
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
            target, "bridge.cancel", {"operation_id": record.operation_id}, wait_s=0
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
    store: store_module.Store, call: Call, record: JobRecord, wait_s: float
) -> tuple[JobRecord, bool]:
    """Wait here until the job's state or progress changes, or the wait ends."""
    if record.state in FINAL:
        return record, False
    mark = signature(record)
    job_id = record.job_id
    began = monotonic()
    deadline = began + wait_s
    next_sweep = began + HOLD_SWEEP_S
    next_note = began + HOLD_NOTE_S
    while True:
        left = deadline - monotonic()
        if left <= 0:
            return record, False
        sleep(min(POLL_S, left))
        now = monotonic()
        if now >= next_sweep:
            sweep(store, call, force=True)
            next_sweep = now + HOLD_SWEEP_S
        if now >= next_note:
            call.progress(round(now - began, 1), wait_s, f"job {record.job_id} is {record.state}")
            next_note = now + HOLD_NOTE_S
        current = stored(lambda: store.get_job(job_id))
        if current is None:
            raise unknown(job_id)
        record = current
        if signature(record) != mark:
            return record, True


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
    with call.router.store() as store:
        if store is None:
            return {"jobs": [], "next_page": None}
        sweep(store, call)
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


def read_token(page: Any, query: str) -> tuple[float, int] | None:
    if page is None:
        return None
    unreadable = CallError(
        "BAD_CURSOR", "the page token could not be read", details={"argument": "page"}
    )
    if not isinstance(page, str) or len(page) > MAX_TOKEN_CHARS:
        raise unreadable
    try:
        padded = page + "=" * (-len(page) % 4)
        body = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (binascii.Error, UnicodeError, ValueError):
        raise unreadable from None
    if (
        not isinstance(body, dict)
        or body.get("v") != TOKEN_VERSION
        or not isinstance(body.get("c"), (int, float))
        or not isinstance(body.get("r"), int)
        or isinstance(body.get("r"), bool)
    ):
        raise unreadable
    if body.get("q") != query:
        raise CallError(
            "BAD_CURSOR",
            "that page token belongs to a list with other filters; send the same session and state",
            details={"argument": "page"},
        )
    return float(body["c"]), int(body["r"])


# Section: one job as a caller reads it


def job_row(
    record: JobRecord, call: Call, *, store: store_module.Store | None, full: bool
) -> dict[str, Any]:
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
    if full and record.kind == "python" and record.state in FINAL and store is not None:
        answer = python_answer(call, record)
        if answer is not None:
            row["outputs"] = answer
            row["error"] = answer.get("error", record.error)
    if record.state in FINAL:
        path = job_rules.export_path(record, home=call.router.home)
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
    said = python_tool.shape(
        call, reply, budget=python_tool.DEFAULT_MAX_CHARS, named=spec.get("namespace")
    )
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


# Section: the sweep


def sweep(store: store_module.Store, call: Call, *, force: bool = False) -> None:
    """Mark what is lost, write the copies it leaves, and prune old rows.

    At most once a second per process and store, unless forced.
    """
    key = str(store.path)
    now = monotonic()
    with _sweep_lock:
        if not force and now - _swept.get(key, float("-inf")) < SWEEP_EVERY_S:
            return
        _swept[key] = now
        prune = now - _pruned.get(key, float("-inf")) >= PRUNE_EVERY_S
        if prune:
            _pruned[key] = now
    home = call.router.home

    def mark() -> list[JobRecord]:
        # A job whose call has ended by its receipt takes that ending before
        # anything could call it lost.
        for record in store.list_jobs(states=list(store_module.JOB_LIVE_STATES), limit=MAX_LIMIT):
            reconcile(store, record)
        # A session whose process has gone loses its jobs as it is marked.
        store.reclaim_sessions()
        lost: list[JobRecord] = []
        for record in store.stale_jobs(job_rules.SILENCE_S):
            gone = record.worker_pid is not None and not store_module.process_is_alive(
                record.worker_pid
            )
            error = store_module.SESSION_ENDED_ERROR if gone else SILENT_ERROR
            lost += store.lose_jobs([record.job_id], error=error)
        live = store.list_jobs(states=list(store_module.JOB_LIVE_STATES), limit=MAX_LIMIT)
        for record in live:
            if not record.session_id:
                continue
            session = store.get_session(record.session_id)
            if session is None or session.state in DEAD_STATES:
                lost += store.lose_jobs([record.job_id], error=store_module.SESSION_ENDED_ERROR)
        # A job lost with its session may still have ended before it went.
        for record in store.list_jobs(states=["lost"], limit=EXPORT_BATCH):
            reconcile(store, record)
        return lost

    stored(mark)
    export_lost(store, home)
    if prune:
        stored(lambda: store.prune_jobs(job_rules.KEEP_S))


def export_lost(store: store_module.Store, home: Any) -> None:
    """Write the readable copy of recently lost jobs that have none yet.

    A job that ends on its own writes its copy from the session. One that is
    lost cannot, so whoever finds it lost writes it. Best effort.
    """
    try:
        recent = store.list_jobs(states=["lost"], limit=EXPORT_BATCH)
    except (store_module.StoreError, sqlite3.Error):
        return
    now = time.time()
    for record in recent:
        if record.finished_at is not None and now - record.finished_at > EXPORT_LOOKBACK_S:
            continue
        if job_rules.export_path(record, home=home) is not None:
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
        }
    ),
    output_schema=outputs({}),
    handler=jobs,
    open_world=False,
    summary=summary_line,
)
