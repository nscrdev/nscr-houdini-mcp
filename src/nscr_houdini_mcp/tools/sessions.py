"""`hou_sessions`: which Houdini sessions there are, and starting and stopping workers.

Four actions.

- `list` reads every session the store knows, with the state it is in now.
  The store keeps `live` and `unresponsive`, and for an ended session how it
  ended: `crashed` when its process was found missing, `gone` when it ended
  its own row or was stopped. That is written the moment the row is marked,
  because the signs it could be read from later are cleared by the next
  reader. `busy` comes from the session's own health answer. A session that
  ended in the last hour is still listed, so a caller that just lost one sees
  what became of it. Listing is not a use: it renews no lease.
- `info` is one session in full, with its health, whatever state it is in.
  Like the list, it renews no lease.
- `start` starts a hython worker through the pool, under the config's cap
  and with the config's hython and ports.
- `stop` stops a worker the pool started. A Houdini with a user interface is
  somebody's working session and is never closed from here. A worker a job
  holds, or one running a call, is refused with `WORKER_BUSY` unless the call
  says `force`.

`start` and `stop` change what runs on the machine, so each takes a receipt
under its operation id in the store. The same id sent again after a lost reply
gets the first answer rather than a second worker. A start or stop that failed
and changed nothing drops its receipt, so the id can be used again. One that
may have left a worker running, or whose attempt stopped part way, closes it
as abandoned, and the id answers `OUTCOME_UNKNOWN` from then on.

This module never imports `hou`.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import pool
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import marshal
from nscr_houdini_mcp.bridge.app import LOG_DIR_NAME
from nscr_houdini_mcp.config import Config, ConfigError, resolve_hython
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.router import LIVE_STATES, Router, Target, choose, dead
from nscr_houdini_mcp.store import SessionRecord, WorkerRecord
from nscr_houdini_mcp.tools.base import (
    DETAIL,
    OPERATION_ID,
    SESSION,
    Call,
    ToolSpec,
    inputs,
    outputs,
)

ACTIONS = ("list", "info", "start", "stop")

# How long a worker asked to stop is given to end itself before it is ended.
STOP_GRACE_S = pool.DEFAULT_STOP_GRACE_S

# How long an ended session stays in the list, and how many of them at most.
RECENT_S = 3600.0
MAX_ENDED = 20


def sessions(call: Call) -> Mapping[str, Any]:
    action = call.arguments.get("action") or "list"
    return ACTION_HANDLERS[action](call)


def settings(call: Call) -> Config:
    """The server's config, or the defaults for a call made without one."""
    return call.config if call.config is not None else Config(path=Path("config.toml"))


# Section: list and info


def list_sessions(call: Call) -> dict[str, Any]:
    full = call.arguments.get("detail") == "full"
    router = call.router
    now = time.time()
    records, workers, active = read_rows(router)
    rows: list[dict[str, Any]] = []
    ended: list[dict[str, Any]] = []
    for record in records:
        worker = workers.get(record.session_id)
        if record.state == store_module.SESSION_GONE:
            if now - record.heartbeat_at > RECENT_S:
                continue
            state = ended_state(record)
            ended.append(session_row(record, state, worker, None, None, now=now, full=full))
            continue
        target, health, state = look(router, record)
        rows.append(session_row(record, state, worker, target, health, now=now, full=full))
    ended.sort(key=lambda row: row.get("ended_at") or 0.0, reverse=True)
    return {
        "sessions": rows + ended[:MAX_ENDED],
        "pool": pool_summary(active, settings(call)),
    }


def session_info(call: Call) -> dict[str, Any]:
    """One session in full, whatever state it is in.

    A look, not a use: it renews no lease. A session that has ended or whose
    port has gone quiet is described, not refused.
    """
    router = call.router
    records, workers, _active = read_rows(router)
    record = named(records, call.arguments.get("session"), router.default_session)
    call.trace.update(
        {
            "session_id": record.session_id,
            "alias": record.alias,
            "scene_epoch": record.scene_epoch,
        }
    )
    if record.state == store_module.SESSION_GONE:
        target, health, state = None, None, ended_state(record)
    else:
        target, health, state = look(router, record)
    if health and isinstance(health.get("scene_epoch"), int):
        call.trace["scene_epoch"] = health["scene_epoch"]
    worker = workers.get(record.session_id)
    row = session_row(record, state, worker, target, health, now=time.time(), full=True)
    return {"session": row}


def read_rows(
    router: Router,
) -> tuple[list[SessionRecord], dict[str, WorkerRecord], list[WorkerRecord]]:
    """Every session row, the worker row behind each session, and the workers
    that hold a slot, after the store has marked the processes that are gone."""
    # A worker this process started and that has ended stays on the process
    # table until its exit is read, and until then it looks alive.
    pool.reap_started()
    with router.store() as store:
        if store is None:
            return [], {}, []

        def read() -> Any:
            store.reclaim_sessions()
            store.reclaim_workers()
            return store.list_sessions(include_gone=True), store.list_workers(active_only=False)

        records, workers = stored(read)
    active = [w for w in workers if w.state in store_module.WORKER_ACTIVE_STATES]
    return records, newest_workers(workers), active


def newest_workers(workers: list[WorkerRecord]) -> dict[str, WorkerRecord]:
    """The worker row behind each session id."""
    found: dict[str, WorkerRecord] = {}
    for worker in sorted(workers, key=lambda w: w.reserved_at):
        if worker.session_id:
            found[worker.session_id] = worker
    return found


def ended_state(record: SessionRecord) -> str:
    """How an ended session ended, as the store wrote it when it was marked.

    `crashed` when its process was found missing, `gone` when it ended its own
    row or was stopped. A row written before the store kept this says `gone`.
    """
    return record.ended_as or store_module.SESSION_GONE


def look(router: Router, record: SessionRecord) -> tuple[Target | None, dict[str, Any] | None, str]:
    """Ask one open session how it is. Its answer decides live or busy."""
    if record.state not in LIVE_STATES:
        return None, None, record.state
    try:
        target = router.reach(record)
        health = router.health(target)
    except CallError as error:
        if error.code == "SESSION_DEAD":
            return None, None, after_dead(router, record)
        # Whatever else went wrong, the session did not answer for itself.
        return None, {"error": error.code}, "unresponsive"
    return target, health, "busy" if is_busy(health) else "live"


def after_dead(router: Router, record: SessionRecord) -> str:
    """What a session is when its file could not be opened.

    That happens to a session that has just ended, and also to one that has
    registered its row and not yet written its file. The row says which: an
    ended row says how it ended, an open one is not answering yet.
    """
    try:
        records = router.records(include_gone=True)
    except CallError:
        return "unresponsive"
    now = next((r for r in records if r.session_id == record.session_id), None)
    if now is not None and now.state == store_module.SESSION_GONE:
        return ended_state(now)
    return "unresponsive"


def is_busy(health: Mapping[str, Any]) -> bool:
    """Busy with a call, or with a main thread that has not run our code for
    longer than the bridge itself allows before it calls a session busy.

    The second is a long cook in a session with a user interface: no call is
    running, but none would be picked up either.
    """
    if health.get("busy"):
        return True
    return main_thread_away_s(health) is not None


def main_thread_away_s(health: Mapping[str, Any]) -> float | None:
    """How long the main thread has been away, when that is too long."""
    thread = health.get("main_thread")
    if not isinstance(thread, Mapping) or not thread.get("installed"):
        return None
    age = thread.get("pulse_age_s")
    if not isinstance(age, (int, float)):
        return None
    if thread.get("away") or age > marshal.DEFAULT_STALE_S:
        return float(age)
    return None


def session_row(
    record: SessionRecord,
    state: str,
    worker: WorkerRecord | None,
    target: Target | None,
    health: Mapping[str, Any] | None,
    *,
    now: float,
    full: bool,
) -> dict[str, Any]:
    """One session as a caller reads it. `full` adds everything else."""
    health = health or {}
    scene = health.get("scene") if isinstance(health.get("scene"), Mapping) else {}
    capabilities = record.capabilities or (worker.capabilities if worker else None)
    row: dict[str, Any] = {
        "session_id": record.session_id,
        "alias": record.alias,
        "kind": record.kind,
        "state": state,
        "hip_path": scene.get("hip_path") or record.hip_path,
        "scene_epoch": health.get("scene_epoch", record.scene_epoch),
        "capabilities": pool.capability_summary(capabilities),
    }
    if state == "busy":
        row["current_op"] = health.get("current_op")
        away = main_thread_away_s(health)
        if away is not None:
            row["main_thread_away_s"] = away
    if worker is not None:
        row["job"] = worker.job_id
        row["lease_age_s"] = round(max(0.0, now - worker.leased_at), 1)
    if state in ("gone", "crashed"):
        row["ended_at"] = record.heartbeat_at
    if not full:
        return row
    row.update(
        {
            "pid": record.pid,
            "port": record.port,
            "houdini_version": _version(target, capabilities),
            "started_at": record.started_at,
            "heartbeat_age_s": round(max(0.0, now - record.heartbeat_at), 3),
            "transport_ok": record.transport_ok,
            "alias_drift": health.get("alias_drift"),
            "top_level_nodes": scene.get("nodes"),
            "capabilities": capabilities,
            "health": _health(health),
        }
    )
    if worker is not None:
        row["worker"] = {"state": worker.state, "weight": worker.weight, "job": worker.job_id}
    return row


def _version(target: Target | None, capabilities: Any) -> str | None:
    if target is not None and target.houdini_version:
        return target.houdini_version
    if isinstance(capabilities, Mapping):
        return capabilities.get("houdini_version")
    return None


def _health(health: Mapping[str, Any]) -> dict[str, Any] | None:
    if not health:
        return None
    keys = (
        "error",
        "busy",
        "queued",
        "current_op",
        "current_op_elapsed_s",
        "heartbeat_age_s",
        "last_self_check_ok",
        "round_trip_ms",
    )
    return {key: health[key] for key in keys if key in health}


def pool_summary(active: list[WorkerRecord], config: Config) -> dict[str, Any]:
    starting = [w.alias for w in active if w.state in store_module.WORKER_STARTING_STATES]
    return {
        "running": len(active) - len(starting),
        "starting": starting,
        "cap": config.pool_cap,
        "weight_held": sum(w.weight for w in active),
        "budget": float(config.pool_cap),
    }


# Section: start and stop


def start_worker(call: Call) -> dict[str, Any]:
    config = settings(call)
    router = call.router
    weight = call.arguments.get("weight") or pool.DEFAULT_WEIGHT
    operation_id = call.operation_id()
    digest = store_module.digest_arguments({"action": "start", "weight": weight})
    with router.store(create=True) as store:
        replayed = begin(store, operation_id, digest)
        if replayed is not None:
            return replayed
        try:
            record = pool.start_worker(pool_config(router, config), store, weight=weight)
        except Exception as error:
            if getattr(error, "spawned_pid", None) and not getattr(error, "spawned_ended", False):
                # A worker may be running that nothing recorded, so the id is
                # closed rather than freed for a second start.
                abandon(store, operation_id)
            else:
                stored(lambda: store.drop_operation(operation_id))
            raise start_failure(error, store, config) from None
        try:
            session = store.get_session(record.session_id or "")
            result = {
                "session": started_row(record, session),
                "pool": pool_summary(store.list_workers(), config),
            }
            store.finish_operation(operation_id, outcome=result)
        except (store_module.StoreError, sqlite3.Error) as error:
            # The worker is up; only the answer could not be recorded.
            abandon(store, operation_id)
            raise unavailable(error) from None
    return result


def pool_config(router: Router, config: Config) -> pool.PoolConfig:
    return pool.PoolConfig(
        home=router.home,
        cap=config.pool_cap,
        hython=resolve_hython(config),
        port_range=config.worker_ports,
    )


def started_row(record: WorkerRecord, session: SessionRecord | None) -> dict[str, Any]:
    if session is None:
        return {"session_id": record.session_id, "alias": record.alias, "state": "live"}
    row = session_row(session, "live", record, None, None, now=time.time(), full=False)
    row["pid"] = record.pid
    row["weight"] = record.weight
    return row


def start_failure(error: Exception, store: Any, config: Config) -> Exception:
    """What a failed start comes back as, coded where the pool said why."""
    coded = coded_start_failure(error, store, config)
    spawned = getattr(error, "spawned_pid", None)
    if isinstance(coded, CallError) and spawned:
        coded.details["spawned_pid"] = spawned
        coded.details["spawned_ended"] = bool(getattr(error, "spawned_ended", False))
    return coded


def coded_start_failure(error: Exception, store: Any, config: Config) -> Exception:
    if isinstance(error, store_module.PoolFull):
        active = stored(store.list_workers)
        return CallError(
            "POOL_FULL",
            str(error),
            details={"pool": pool_summary(active, config)},
        )
    if isinstance(error, pool.HythonNotFound):
        return CallError(
            "HYTHON_NOT_FOUND",
            str(error),
            details={"config": ["hython", "houdini_build"], "env": pool.HYTHON_ENV_VAR},
        )
    if isinstance(error, pool.WorkerStartFailed):
        return CallError(
            "WORKER_START_FAILED",
            str(error),
            details={"logs": f"{LOG_DIR_NAME} in the state folder"},
        )
    if isinstance(error, ConfigError):
        return CallError("CONFIG_INVALID", error.message, details=error.details())
    if isinstance(error, (store_module.StoreError, sqlite3.Error)):
        return unavailable(error)
    return error


def stop_worker(call: Call) -> dict[str, Any]:
    handle = call.arguments.get("session")
    if not handle:
        raise CallError(
            "BAD_ARGUMENTS",
            "stop needs session: the id or alias of the worker to stop",
            details={"argument": "session"},
        )
    router = call.router
    operation_id = call.operation_id()
    # The receipt is asked first: a stop that already happened has ended the
    # session, and resolving the name again would only say so.
    digest = store_module.digest_arguments({"action": "stop", "session": str(handle)})
    with router.store() as store:
        if store is not None:
            replayed = begin(store, operation_id, digest)
            if replayed is not None:
                return replayed
        progress = {"asked": False}
        try:
            result = stop_one(call, store, str(handle), progress)
            if store is not None:
                store.finish_operation(operation_id, outcome=result)
        except Exception as error:
            if store is not None:
                if progress["asked"]:
                    # The stop was written, so it may well have happened. The id
                    # is closed rather than freed for a second try that would
                    # only find the session ended.
                    abandon(store, operation_id)
                else:
                    stored(lambda: store.drop_operation(operation_id))
            if isinstance(error, (store_module.StoreError, sqlite3.Error)):
                raise unavailable(error) from None
            raise
    return result


def stop_one(call: Call, store: Any, handle: str, progress: dict[str, bool]) -> dict[str, Any]:
    router = call.router
    record = named(router.records(include_gone=True), handle)
    if record.state == store_module.SESSION_GONE:
        raise dead(record.session_id, record.alias, [])
    if record.kind != "hython":
        raise CallError(
            "NOT_A_WORKER",
            f"session {record.alias} has a user interface; this tool never closes one",
            details={"session_id": record.session_id, "alias": record.alias, "kind": record.kind},
        )
    workers = stored(store.list_workers) if store is not None else []
    worker = next((w for w in workers if w.session_id == record.session_id), None)
    if worker is None:
        raise not_a_worker(record)
    if not call.arguments.get("force"):
        refuse_if_in_use(router, record, worker)
    progress["asked"] = True
    try:
        stopped = pool.stop_worker(
            pool.PoolConfig(home=router.home), store, worker.token, grace_s=STOP_GRACE_S
        )
    except pool.UnknownWorker:
        raise not_a_worker(record) from None
    except (pool.PoolError, store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None
    if stopped.ended:
        # A worker that had to be ended here never got to end its own row, and
        # the next reader would find its process missing and call it crashed.
        # It was stopped on purpose, so the row says so.
        try:
            store.end_session(record.session_id, how=store_module.SESSION_GONE)
        except store_module.UnknownRecord:
            pass
        except (store_module.StoreError, sqlite3.Error) as error:
            raise unavailable(error) from None
    router.forget(record.session_id)
    return {
        "stopped": {
            "session_id": record.session_id,
            "alias": record.alias,
            "ended": stopped.ended,
            "killed": stopped.killed,
            "note": stopped.note or None,
        },
        "pool": pool_summary(stored(store.list_workers), settings(call)),
    }


def named(
    records: list[SessionRecord], handle: str | None, default: str | None = None
) -> SessionRecord:
    """The session a caller means, in whatever state it is.

    By id first, then the newest under an alias, preferring one that has not
    ended. With no name, the usual rules for a call that names none.
    """
    if not handle:
        return choose(records, None, default=default)
    handle = handle.strip()
    for record in records:
        if record.session_id == handle:
            return record
    under = sorted((r for r in records if r.alias == handle), key=lambda r: r.started_at)
    open_ones = [r for r in under if r.state != store_module.SESSION_GONE]
    if open_ones or under:
        return (open_ones or under)[-1]
    # Nobody by that name: the rules say so, with the nearest names.
    return choose(records, handle)


def refuse_if_in_use(router: Router, record: SessionRecord, worker: WorkerRecord) -> None:
    """`WORKER_BUSY` for a worker a job holds or a call is running on."""
    details: dict[str, Any] = {"session_id": record.session_id, "alias": record.alias}
    if worker.job_id or worker.state == "leased":
        details["job_id"] = worker.job_id
        raise CallError(
            "WORKER_BUSY",
            f"worker {record.alias} is held by job {worker.job_id or 'unnamed'}",
            details=details,
        )
    if record.state not in LIVE_STATES:
        return
    try:
        _target, health, state = look(router, record)
    except CallError:
        # A health read that failed says nothing about a job or a call, so it
        # does not stand in the way of a stop.
        return
    if state == "busy" and health is not None:
        details["current_op"] = health.get("current_op")
        raise CallError(
            "WORKER_BUSY",
            f"worker {record.alias} is running {health.get('current_op') or 'a call'}",
            details=details,
        )


def not_a_worker(record: SessionRecord) -> CallError:
    return CallError(
        "NOT_A_WORKER",
        f"session {record.alias} was not started by the pool; stop it where it was started",
        details={"session_id": record.session_id, "alias": record.alias},
    )


# Section: receipts for changes the server makes itself


def begin(store: Any, operation_id: str, digest: str) -> dict[str, Any] | None:
    """Claim an operation id. Returns the first answer when it already ran.

    As the bridge's receipts do: an id whose earlier attempt stopped without
    recording what it did is closed as abandoned and not run again, because
    that attempt may have started or stopped a worker before it went.
    """
    try:
        claim = store.begin_operation(operation_id, digest)
    except store_module.OperationMismatch as error:
        raise CallError(
            "OPERATION_MISMATCH", str(error), details={"operation_id": operation_id}
        ) from None
    except (store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None
    if claim.outcome_unknown:
        if claim.claimed:
            abandon(store, operation_id)
            raise unknown(operation_id, "an earlier attempt with this id stopped part way")
        raise unknown(operation_id, "another call with this id is still running")
    if claim.claimed:
        return None
    if claim.record.state == store_module.OPERATION_ABANDONED:
        raise unknown(operation_id, "an earlier attempt with this id stopped part way")
    outcome = claim.record.outcome
    return {**outcome, "replayed": True} if isinstance(outcome, Mapping) else None


def abandon(store: Any, operation_id: str) -> None:
    """Close a receipt whose work may have happened, so it is never run again."""
    try:
        store.finish_operation(
            operation_id,
            state=store_module.OPERATION_ABANDONED,
            error={"reason": "the attempt stopped without recording what it did"},
        )
    except (store_module.StoreError, sqlite3.Error):
        pass


def unknown(operation_id: str, why: str) -> CallError:
    return CallError(
        "OUTCOME_UNKNOWN",
        f"{why}, so whether it happened is not known",
        hint="list the sessions to see what is running, then use a new operation_id",
        details={"operation_id": operation_id},
    )


def stored(action: Callable[[], Any]) -> Any:
    """One store read or write, with a failure coded as the store's."""
    try:
        return action()
    except (store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None


def unavailable(error: BaseException) -> CallError:
    return CallError(
        "STORE_UNAVAILABLE",
        "the coordination store could not be read",
        details={"exception": type(error).__name__},
    )


ACTION_HANDLERS: dict[str, Callable[[Call], dict[str, Any]]] = {
    "list": list_sessions,
    "info": session_info,
    "start": start_worker,
    "stop": stop_worker,
}


HOU_SESSIONS = ToolSpec(
    name="hou_sessions",
    description=(
        "List sessions, or start and stop hython workers. state: live, busy, unresponsive, "
        "crashed, gone. A session_id lasts until its process exits; an alias like w1 may later "
        "name a new one. stop never closes a GUI Houdini."
    ),
    input_schema=inputs(
        {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "session": SESSION,
            "detail": DETAIL,
            "weight": {
                "type": "string",
                "enum": list(pool.WEIGHTS),
                "description": "start only. heavy may use every core.",
            },
            "force": {
                "type": "boolean",
                "description": "stop only: stop it even while in use.",
            },
            "operation_id": OPERATION_ID,
        }
    ),
    output_schema=outputs({}),
    handler=sessions,
    open_world=False,
)
