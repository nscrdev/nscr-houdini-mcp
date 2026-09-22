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
- `info` is one session in full, with its health.
- `start` starts a hython worker through the pool, under the config's cap
  and with the config's hython and ports.
- `stop` stops a worker the pool started. A Houdini with a user interface is
  somebody's working session and is never closed from here.

`start` and `stop` change what runs on the machine, so each takes a receipt
under its operation id in the store. The same id sent again after a lost reply
gets the first answer rather than a second worker. A start or stop that failed
changed nothing, so its receipt is dropped and the id can be used again.

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
from nscr_houdini_mcp.bridge.app import LOG_DIR_NAME
from nscr_houdini_mcp.config import Config, ConfigError, resolve_hython
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.router import LIVE_STATES, Router, Target, choose
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

# What a failed health answer says about the session behind it.
SILENT_CODES = frozenset(
    {"SESSION_UNREACHABLE", "SESSION_UNRESPONSIVE", "REPLY_NOT_AUTHENTIC", "BAD_REPLY"}
)


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
    target = call.target()
    health = call.health()
    state = "busy" if health.get("busy") else target.record.state
    worker = None
    with call.router.store() as store:
        if store is not None:
            worker = stored(lambda: newest_workers(store.list_workers(active_only=False))).get(
                target.session_id
            )
    return {
        "session": session_row(
            target.record, state, worker, target, health, now=time.time(), full=True
        )
    }


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
            return None, None, "crashed"
        if error.code in SILENT_CODES:
            return None, {"error": error.code}, "unresponsive"
        raise
    return target, health, "busy" if health.get("busy") else "live"


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
            stored(lambda: store.drop_operation(operation_id))
            raise start_failure(error, store, config) from None
        session = stored(lambda: store.get_session(record.session_id or ""))
        active = stored(store.list_workers)
        result = {
            "session": started_row(record, session),
            "pool": pool_summary(active, config),
        }
        stored(lambda: store.finish_operation(operation_id, outcome=result))
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
    if isinstance(error, store_module.PoolFull):
        active = stored(store.list_workers)
        return CallError(
            "POOL_FULL",
            str(error),
            details={"pool": pool_summary(active, config)},
        )
    if isinstance(error, pool.HythonNotFound):
        return CallError("HYTHON_NOT_FOUND", str(error))
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
        try:
            result = stop_one(call, store, str(handle))
        except Exception:
            if store is not None:
                stored(lambda: store.drop_operation(operation_id))
            raise
        stored(lambda: store.finish_operation(operation_id, outcome=result))
    return result


def stop_one(call: Call, store: Any, handle: str) -> dict[str, Any]:
    router = call.router
    record = named(router, handle)
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


def named(router: Router, handle: str) -> SessionRecord:
    """The session a stop means. One whose port went quiet can still be stopped."""
    records = router.records(include_gone=True)
    try:
        return choose(records, handle)
    except CallError as error:
        if error.code != "SESSION_UNRESPONSIVE":
            raise
        wanted = error.details.get("session_id")
        return next(r for r in records if r.session_id == wanted)


def not_a_worker(record: SessionRecord) -> CallError:
    return CallError(
        "NOT_A_WORKER",
        f"session {record.alias} was not started by the pool; stop it where it was started",
        details={"session_id": record.session_id, "alias": record.alias},
    )


# Section: receipts for changes the server makes itself


def begin(store: Any, operation_id: str, digest: str) -> dict[str, Any] | None:
    """Claim an operation id. Returns the first answer when it already ran."""
    try:
        claim = store.begin_operation(operation_id, digest)
    except store_module.OperationMismatch as error:
        raise CallError(
            "OPERATION_MISMATCH", str(error), details={"operation_id": operation_id}
        ) from None
    except (store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None
    if claim.claimed:
        return None
    if claim.outcome_unknown:
        raise CallError(
            "OUTCOME_UNKNOWN",
            "another call with this operation id is still running",
            details={"operation_id": operation_id},
        )
    outcome = claim.record.outcome
    return {**outcome, "replayed": True} if isinstance(outcome, Mapping) else None


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
        "name a new one. stop needs session and never closes a GUI Houdini."
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
            "operation_id": OPERATION_ID,
        }
    ),
    output_schema=outputs({}),
    handler=sessions,
    open_world=False,
)
