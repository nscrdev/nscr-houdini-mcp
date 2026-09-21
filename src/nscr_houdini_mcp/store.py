"""Local coordination store shared by every server process on one machine.

One client means one server process, so several server processes can be live at
once against the same Houdini sessions. Anything they must agree on lives here
instead of in memory: which sessions exist, how many workers are allowed to
run, which mutations already happened, jobs, version numbers and output runs.

The file is SQLite in WAL mode on a local disk, never on a share. There is no
daemon. Allocations that decide a winner run under `BEGIN IMMEDIATE`, so the
decision is one transaction rather than a read followed by a write.

Transactions are short on purpose. Every public method opens a transaction,
finishes its work and commits before it returns, and the transaction helper
itself is private, so there is no supported way to hold one open across a
hython start, a cook or a render.

This module never imports `hou`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

APP_DIR_NAME = "nscr-houdini-mcp"
HOME_ENV_VAR = "NSCR_MCP_HOME"
STORE_FILE_NAME = "coord.sqlite"

SCHEMA_VERSION = 1

SESSION_KINDS = frozenset({"gui", "hython"})
SESSION_STATES = frozenset({"live", "busy", "unresponsive", "crashed", "gone"})
SESSION_GONE = "gone"

# States that still hold a slot against the pool cap. A reservation counts from
# the moment it is made, before hython has started.
WORKER_ACTIVE_STATES = ("reserved", "starting", "running", "leased", "stopping")
WORKER_FINAL_STATES = ("failed", "stopped")
WORKER_STATES = frozenset(WORKER_ACTIVE_STATES + WORKER_FINAL_STATES)

OPERATION_STATES = frozenset({"running", "done", "failed"})
JOB_STATES = frozenset({"queued", "running", "done", "failed", "cancelled", "lost"})
JOB_FINAL_STATES = frozenset({"done", "failed", "cancelled", "lost"})

MAX_ALIAS_INDEX = 4096


class StoreError(Exception):
    """Base class for coordination store failures."""


class PoolFull(StoreError):
    """No worker slot is free under the current cap."""


class AliasInUse(StoreError):
    """The requested alias already belongs to a live session."""


class UnknownRecord(StoreError):
    """A session, reservation, job or run id is not in the store."""


class OperationMismatch(StoreError):
    """An operation id came back with different arguments than the first time."""


def default_home() -> Path:
    """Per user state folder for this tool, overridable by environment."""
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_STATE_HOME")
        root = Path(base) if base else Path.home() / ".local" / "state"
    return root / APP_DIR_NAME


def default_store_path() -> Path:
    """Where the store lives when config passes no path of its own."""
    return default_home() / STORE_FILE_NAME


def digest_arguments(payload: Any) -> str:
    """Stable digest of one call's arguments, for operation receipts."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=UTC).isoformat(timespec="seconds")


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True, default=str)


def _load(text: str | None) -> Any:
    return None if text is None else json.loads(text)


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    alias: str
    kind: str
    pid: int
    port: int | None
    state: str
    scene_epoch: int
    hip_path: str | None
    capabilities: Any
    started_at: float
    heartbeat_at: float

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> SessionRecord:
        return cls(
            session_id=row["session_id"],
            alias=row["alias"],
            kind=row["kind"],
            pid=row["pid"],
            port=row["port"],
            state=row["state"],
            scene_epoch=row["scene_epoch"],
            hip_path=row["hip_path"],
            capabilities=_load(row["capabilities"]),
            started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"],
        )


@dataclass(frozen=True)
class WorkerRecord:
    token: str
    alias: str
    state: str
    session_id: str | None
    job_id: str | None
    reserved_at: float
    leased_at: float

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> WorkerRecord:
        return cls(
            token=row["token"],
            alias=row["alias"],
            state=row["state"],
            session_id=row["session_id"],
            job_id=row["job_id"],
            reserved_at=row["reserved_at"],
            leased_at=row["leased_at"],
        )


@dataclass(frozen=True)
class OperationRecord:
    operation_id: str
    session_id: str | None
    scene_epoch: int | None
    digest: str
    state: str
    outcome: Any
    error: Any
    job_id: str | None
    created_at: float
    updated_at: float

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> OperationRecord:
        return cls(
            operation_id=row["operation_id"],
            session_id=row["session_id"],
            scene_epoch=row["scene_epoch"],
            digest=row["digest"],
            state=row["state"],
            outcome=_load(row["outcome"]),
            error=_load(row["error"]),
            job_id=row["job_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    session_id: str | None
    kind: str
    state: str
    weight: str
    progress: Any
    outputs: Any
    error: Any
    scene: Any
    created_at: float
    updated_at: float
    finished_at: float | None

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> JobRecord:
        return cls(
            job_id=row["job_id"],
            session_id=row["session_id"],
            kind=row["kind"],
            state=row["state"],
            weight=row["weight"],
            progress=_load(row["progress"]),
            outputs=_load(row["outputs"]),
            error=_load(row["error"]),
            scene=_load(row["scene"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
        )


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    kind: str
    name: str | None
    hip_family: str | None
    version: int | None
    session_id: str | None
    source_node: str | None
    job_id: str | None
    paths: Any
    scene: Any
    created_at: float

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> RunRecord:
        return cls(
            run_id=row["run_id"],
            kind=row["kind"],
            name=row["name"],
            hip_family=row["hip_family"],
            version=row["version"],
            session_id=row["session_id"],
            source_node=row["source_node"],
            job_id=row["job_id"],
            paths=_load(row["paths"]),
            scene=_load(row["scene"]),
            created_at=row["created_at"],
        )


# One entry per schema version. A step runs inside its own transaction and is
# never edited once it has shipped: a later change is a new step.
def _migrate_to_1(cur: sqlite3.Cursor) -> None:
    cur.executescript(
        """
        CREATE TABLE sessions (
            session_id   TEXT PRIMARY KEY,
            alias        TEXT NOT NULL,
            kind         TEXT NOT NULL,
            pid          INTEGER NOT NULL,
            port         INTEGER,
            state        TEXT NOT NULL,
            scene_epoch  INTEGER NOT NULL DEFAULT 0,
            hip_path     TEXT,
            capabilities TEXT,
            started_at   REAL NOT NULL,
            heartbeat_at REAL NOT NULL
        );
        CREATE UNIQUE INDEX sessions_live_alias
            ON sessions(alias) WHERE state <> 'gone';

        CREATE TABLE workers (
            token       TEXT PRIMARY KEY,
            alias       TEXT NOT NULL,
            state       TEXT NOT NULL,
            session_id  TEXT,
            job_id      TEXT,
            reserved_at REAL NOT NULL,
            leased_at   REAL NOT NULL
        );
        CREATE UNIQUE INDEX workers_live_alias
            ON workers(alias) WHERE state NOT IN ('failed', 'stopped');

        CREATE TABLE operations (
            operation_id TEXT PRIMARY KEY,
            session_id   TEXT,
            scene_epoch  INTEGER,
            digest       TEXT NOT NULL,
            state        TEXT NOT NULL,
            outcome      TEXT,
            error        TEXT,
            job_id       TEXT,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );

        CREATE TABLE jobs (
            job_id      TEXT PRIMARY KEY,
            session_id  TEXT,
            kind        TEXT NOT NULL,
            state       TEXT NOT NULL,
            weight      TEXT NOT NULL,
            progress    TEXT,
            outputs     TEXT,
            error       TEXT,
            scene       TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            finished_at REAL
        );
        CREATE INDEX jobs_by_state ON jobs(state, updated_at);

        CREATE TABLE versions (
            kind       TEXT NOT NULL,
            name       TEXT NOT NULL,
            hip_family TEXT NOT NULL,
            version    INTEGER NOT NULL,
            run_id     TEXT,
            created_at REAL NOT NULL,
            PRIMARY KEY (kind, name, hip_family, version)
        );

        CREATE TABLE runs (
            run_id      TEXT PRIMARY KEY,
            kind        TEXT NOT NULL,
            name        TEXT,
            hip_family  TEXT,
            version     INTEGER,
            session_id  TEXT,
            source_node TEXT,
            job_id      TEXT,
            paths       TEXT,
            scene       TEXT,
            created_at  REAL NOT NULL
        );
        """
    )


MIGRATIONS = (_migrate_to_1,)


class Store:
    """Handle on the coordination store. One per process or per thread.

    The connection is not shared between threads. Several processes opening the
    same file is the normal case and is what the store is for.
    """

    def __init__(self, path: Path | str | None = None, *, busy_timeout_s: float = 10.0) -> None:
        self.path = Path(path) if path is not None else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=busy_timeout_s, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._in_txn = False
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_s * 1000)}")
        self._apply_migrations()

    # -- lifetime ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transactions -----------------------------------------------------

    @contextmanager
    def _txn(self, *, write: bool) -> Iterator[sqlite3.Cursor]:
        """Short transaction. Private so no caller can hold one open.

        Writes take `BEGIN IMMEDIATE`, so a caller that reads a count and then
        decides on it wins or loses the whole decision, never half of it.
        """
        if self._in_txn:
            raise StoreError("store transactions do not nest")
        self._in_txn = True
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield cur
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()
        finally:
            self._in_txn = False
            cur.close()

    def _apply_migrations(self) -> None:
        with self._txn(write=True) as cur:
            current = cur.execute("PRAGMA user_version").fetchone()[0]
            for step, migrate in enumerate(MIGRATIONS[current:], start=current + 1):
                migrate(cur)
                cur.execute(f"PRAGMA user_version={step}")

    def schema_version(self) -> int:
        """Schema version of the open file."""
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    # -- sessions ---------------------------------------------------------

    def register_session(
        self,
        session_id: str,
        *,
        kind: str,
        pid: int,
        alias: str | None = None,
        alias_template: str | None = None,
        port: int | None = None,
        hip_path: str | None = None,
        scene_epoch: int = 0,
        capabilities: Any = None,
        state: str = "live",
    ) -> SessionRecord:
        """Add a session row and settle its alias in the same transaction.

        Pass `alias` for a name the caller already owns, or `alias_template`
        containing `{n}` (`"scene-{n}"`, `"w{n}"`) to take the lowest free one.
        A session id is never reused. An alias is free again once its session
        is gone.
        """
        if kind not in SESSION_KINDS:
            raise ValueError(f"unknown session kind: {kind}")
        if state not in SESSION_STATES:
            raise ValueError(f"unknown session state: {state}")
        if (alias is None) == (alias_template is None):
            raise ValueError("pass exactly one of alias or alias_template")
        now = _now()
        with self._txn(write=True) as cur:
            if alias_template is not None:
                taken = {
                    row["alias"]
                    for row in cur.execute(
                        "SELECT alias FROM sessions WHERE state <> ?", (SESSION_GONE,)
                    )
                }
                name = _first_free_alias(alias_template, taken)
            else:
                name = alias
                row = cur.execute(
                    "SELECT session_id FROM sessions WHERE alias = ? AND state <> ?",
                    (name, SESSION_GONE),
                ).fetchone()
                if row is not None:
                    raise AliasInUse(f"alias {name} belongs to session {row['session_id']}")
            cur.execute(
                "INSERT INTO sessions (session_id, alias, kind, pid, port, state, scene_epoch,"
                " hip_path, capabilities, started_at, heartbeat_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    name,
                    kind,
                    pid,
                    port,
                    state,
                    scene_epoch,
                    hip_path,
                    _dump(capabilities),
                    now,
                    now,
                ),
            )
            return SessionRecord._from_row(
                cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Session by id, gone or not. Ids are never reused."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return None if row is None else SessionRecord._from_row(row)

    def resolve_session(self, handle: str) -> SessionRecord | None:
        """Session by id, or by the alias of a session that is not gone."""
        found = self.get_session(handle)
        if found is not None:
            return found
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE alias = ? AND state <> ? ORDER BY started_at DESC",
            (handle, SESSION_GONE),
        ).fetchone()
        return None if row is None else SessionRecord._from_row(row)

    def list_sessions(self, *, include_gone: bool = False) -> list[SessionRecord]:
        """Sessions, oldest first."""
        sql = "SELECT * FROM sessions"
        args: tuple[Any, ...] = ()
        if not include_gone:
            sql += " WHERE state <> ?"
            args = (SESSION_GONE,)
        sql += " ORDER BY started_at"
        return [SessionRecord._from_row(row) for row in self._conn.execute(sql, args)]

    def touch_session(self, session_id: str, *, state: str | None = None) -> float:
        """Write a heartbeat, and a new state when one is given."""
        if state is not None and state not in SESSION_STATES:
            raise ValueError(f"unknown session state: {state}")
        now = _now()
        with self._txn(write=True) as cur:
            if state is None:
                cur.execute(
                    "UPDATE sessions SET heartbeat_at = ? WHERE session_id = ?", (now, session_id)
                )
            else:
                cur.execute(
                    "UPDATE sessions SET heartbeat_at = ?, state = ? WHERE session_id = ?",
                    (now, state, session_id),
                )
            if cur.rowcount == 0:
                raise UnknownRecord(f"no session {session_id}")
        return now

    def bump_scene_epoch(self, session_id: str, *, hip_path: str | None = None) -> int:
        """Count a scene open, new or reset. Returns the new epoch."""
        with self._txn(write=True) as cur:
            row = cur.execute(
                "SELECT scene_epoch FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise UnknownRecord(f"no session {session_id}")
            epoch = int(row["scene_epoch"]) + 1
            if hip_path is None:
                cur.execute(
                    "UPDATE sessions SET scene_epoch = ? WHERE session_id = ?", (epoch, session_id)
                )
            else:
                cur.execute(
                    "UPDATE sessions SET scene_epoch = ?, hip_path = ? WHERE session_id = ?",
                    (epoch, hip_path, session_id),
                )
        return epoch

    def end_session(self, session_id: str) -> None:
        """Mark a session gone, which frees its alias for a later process."""
        with self._txn(write=True) as cur:
            cur.execute(
                "UPDATE sessions SET state = ?, heartbeat_at = ? WHERE session_id = ?",
                (SESSION_GONE, _now(), session_id),
            )
            if cur.rowcount == 0:
                raise UnknownRecord(f"no session {session_id}")

    # -- workers ----------------------------------------------------------

    def reserve_worker(
        self,
        *,
        cap: int,
        token: str,
        alias_template: str = "w{n}",
        job_id: str | None = None,
    ) -> WorkerRecord:
        """Take a slot under the pool cap, or raise `PoolFull`.

        Counting and inserting are one transaction, so two processes that see
        the same count cannot both get the last slot. A reservation counts from
        here, before hython starts, and stops counting only when the worker
        ends up failed or stopped.
        """
        if cap < 1:
            raise ValueError("cap must be at least 1")
        now = _now()
        placeholders = ", ".join("?" * len(WORKER_ACTIVE_STATES))
        with self._txn(write=True) as cur:
            rows = cur.execute(
                f"SELECT alias FROM workers WHERE state IN ({placeholders})",
                WORKER_ACTIVE_STATES,
            ).fetchall()
            if len(rows) >= cap:
                raise PoolFull(f"{len(rows)} of {cap} worker slots are in use")
            alias = _first_free_alias(alias_template, {row["alias"] for row in rows})
            cur.execute(
                "INSERT INTO workers (token, alias, state, session_id, job_id, reserved_at,"
                " leased_at) VALUES (?, ?, 'reserved', NULL, ?, ?, ?)",
                (token, alias, job_id, now, now),
            )
            return WorkerRecord._from_row(
                cur.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            )

    def set_worker_state(
        self,
        token: str,
        state: str,
        *,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> WorkerRecord:
        """Move a reservation on. The token proves who owns the slot."""
        if state not in WORKER_STATES:
            raise ValueError(f"unknown worker state: {state}")
        with self._txn(write=True) as cur:
            row = cur.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            if row is None:
                raise UnknownRecord(f"no worker reservation {token}")
            cur.execute(
                "UPDATE workers SET state = ?, session_id = ?, job_id = ?, leased_at = ?"
                " WHERE token = ?",
                (
                    state,
                    session_id if session_id is not None else row["session_id"],
                    job_id if job_id is not None else row["job_id"],
                    _now(),
                    token,
                ),
            )
            return WorkerRecord._from_row(
                cur.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            )

    def release_worker(self, token: str, *, state: str = "stopped") -> WorkerRecord:
        """Give a slot back. Use `failed` when the start never came up."""
        if state not in WORKER_FINAL_STATES:
            raise ValueError(f"not a final worker state: {state}")
        return self.set_worker_state(token, state)

    def touch_worker_lease(self, token: str) -> float:
        """Renew the idle lease. Routing a call to a worker renews it."""
        now = _now()
        with self._txn(write=True) as cur:
            cur.execute("UPDATE workers SET leased_at = ? WHERE token = ?", (now, token))
            if cur.rowcount == 0:
                raise UnknownRecord(f"no worker reservation {token}")
        return now

    def get_worker(self, token: str) -> WorkerRecord | None:
        """One reservation by its owner token."""
        row = self._conn.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
        return None if row is None else WorkerRecord._from_row(row)

    def list_workers(self, *, active_only: bool = True) -> list[WorkerRecord]:
        """Reservations, oldest first."""
        sql = "SELECT * FROM workers"
        args: tuple[Any, ...] = ()
        if active_only:
            sql += f" WHERE state IN ({', '.join('?' * len(WORKER_ACTIVE_STATES))})"
            args = WORKER_ACTIVE_STATES
        sql += " ORDER BY reserved_at"
        return [WorkerRecord._from_row(row) for row in self._conn.execute(sql, args)]

    def idle_workers(self, max_idle_s: float) -> list[WorkerRecord]:
        """Workers whose lease is older than the idle limit.

        An expired lease only says a worker may exit by itself. It never hands
        a reserved worker to somebody else.
        """
        cutoff = _now() - max_idle_s
        placeholders = ", ".join("?" * len(WORKER_ACTIVE_STATES))
        rows = self._conn.execute(
            f"SELECT * FROM workers WHERE leased_at < ? AND state IN ({placeholders})"
            " AND job_id IS NULL ORDER BY leased_at",
            (cutoff, *WORKER_ACTIVE_STATES),
        )
        return [WorkerRecord._from_row(row) for row in rows]

    # -- operation receipts ----------------------------------------------

    def begin_operation(
        self,
        operation_id: str,
        digest: str,
        *,
        session_id: str | None = None,
        scene_epoch: int | None = None,
    ) -> tuple[OperationRecord, bool]:
        """Claim an operation id, or hand back what it did the first time.

        Returns the record and whether this call is the one that claimed it.
        A retry after a lost reply passes the same id and the same digest and
        gets the stored outcome. The same id with different arguments is a
        different call by mistake, so it raises `OperationMismatch`.
        """
        now = _now()
        with self._txn(write=True) as cur:
            row = cur.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if row is not None:
                if row["digest"] != digest:
                    raise OperationMismatch(
                        f"operation {operation_id} was recorded with different arguments"
                    )
                return OperationRecord._from_row(row), False
            cur.execute(
                "INSERT INTO operations (operation_id, session_id, scene_epoch, digest, state,"
                " outcome, error, job_id, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, 'running', NULL, NULL, NULL, ?, ?)",
                (operation_id, session_id, scene_epoch, digest, now, now),
            )
            row = cur.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            return OperationRecord._from_row(row), True

    def finish_operation(
        self,
        operation_id: str,
        *,
        state: str = "done",
        outcome: Any = None,
        error: Any = None,
        job_id: str | None = None,
    ) -> OperationRecord:
        """Store the outcome so a retry can be answered without redoing work."""
        if state not in OPERATION_STATES:
            raise ValueError(f"unknown operation state: {state}")
        with self._txn(write=True) as cur:
            cur.execute(
                "UPDATE operations SET state = ?, outcome = ?, error = ?, job_id = ?,"
                " updated_at = ? WHERE operation_id = ?",
                (state, _dump(outcome), _dump(error), job_id, _now(), operation_id),
            )
            if cur.rowcount == 0:
                raise UnknownRecord(f"no operation {operation_id}")
            return OperationRecord._from_row(
                cur.execute(
                    "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
                ).fetchone()
            )

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        """One receipt by id."""
        row = self._conn.execute(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        return None if row is None else OperationRecord._from_row(row)

    def prune_operations(self, max_age_s: float) -> int:
        """Drop receipts older than the retention window. Returns the count."""
        cutoff = _now() - max_age_s
        with self._txn(write=True) as cur:
            cur.execute("DELETE FROM operations WHERE updated_at < ?", (cutoff,))
            return cur.rowcount

    # -- jobs -------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        *,
        kind: str,
        session_id: str | None = None,
        state: str = "queued",
        weight: str = "light",
        scene: Any = None,
        progress: Any = None,
    ) -> JobRecord:
        """Record an accepted job, including the scene identity it consumes."""
        if state not in JOB_STATES:
            raise ValueError(f"unknown job state: {state}")
        now = _now()
        with self._txn(write=True) as cur:
            cur.execute(
                "INSERT INTO jobs (job_id, session_id, kind, state, weight, progress, outputs,"
                " error, scene, created_at, updated_at, finished_at)"
                " VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, NULL)",
                (job_id, session_id, kind, state, weight, _dump(progress), _dump(scene), now, now),
            )
            return JobRecord._from_row(
                cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            )

    def update_job(
        self,
        job_id: str,
        *,
        state: str | None = None,
        progress: Any = None,
        outputs: Any = None,
        error: Any = None,
    ) -> JobRecord:
        """Write progress, outputs so far or a final state."""
        if state is not None and state not in JOB_STATES:
            raise ValueError(f"unknown job state: {state}")
        now = _now()
        with self._txn(write=True) as cur:
            row = cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise UnknownRecord(f"no job {job_id}")
            new_state = state or row["state"]
            finished = now if new_state in JOB_FINAL_STATES else row["finished_at"]
            cur.execute(
                "UPDATE jobs SET state = ?, progress = ?, outputs = ?, error = ?, updated_at = ?,"
                " finished_at = ? WHERE job_id = ?",
                (
                    new_state,
                    _dump(progress) if progress is not None else row["progress"],
                    _dump(outputs) if outputs is not None else row["outputs"],
                    _dump(error) if error is not None else row["error"],
                    now,
                    finished,
                    job_id,
                ),
            )
            return JobRecord._from_row(
                cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        """One job by id."""
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else JobRecord._from_row(row)

    def list_jobs(
        self,
        *,
        session_id: str | None = None,
        states: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        """Recent jobs, newest first."""
        sql = "SELECT * FROM jobs"
        clauses: list[str] = []
        args: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            args.append(session_id)
        if states:
            clauses.append(f"state IN ({', '.join('?' * len(states))})")
            args.extend(states)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [JobRecord._from_row(row) for row in self._conn.execute(sql, args)]

    def prune_jobs(self, max_age_s: float) -> int:
        """Drop job rows older than the retention window. Returns the count."""
        cutoff = _now() - max_age_s
        with self._txn(write=True) as cur:
            cur.execute("DELETE FROM jobs WHERE updated_at < ?", (cutoff,))
            return cur.rowcount

    # -- version allocation ----------------------------------------------

    def allocate_version(
        self,
        *,
        kind: str,
        name: str,
        hip_family: str,
        run_id: str | None = None,
    ) -> int:
        """Take the next version number for this kind, name and hip family.

        One transaction, so a number is handed out once and never reused, even
        when several processes ask at the same moment. The caller still creates
        the `v<ver>` folder with an exclusive mkdir, which is the last guard
        when the scene folder is shared between machines.
        """
        with self._txn(write=True) as cur:
            row = cur.execute(
                "SELECT MAX(version) AS top FROM versions"
                " WHERE kind = ? AND name = ? AND hip_family = ?",
                (kind, name, hip_family),
            ).fetchone()
            version = int(row["top"] or 0) + 1
            cur.execute(
                "INSERT INTO versions (kind, name, hip_family, version, run_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (kind, name, hip_family, version, run_id, _now()),
            )
        return version

    def latest_version(self, *, kind: str, name: str, hip_family: str) -> int:
        """Highest number handed out so far, or 0 when there is none."""
        row = self._conn.execute(
            "SELECT MAX(version) AS top FROM versions WHERE kind = ? AND name = ? AND"
            " hip_family = ?",
            (kind, name, hip_family),
        ).fetchone()
        return int(row["top"] or 0)

    # -- runs -------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        *,
        kind: str,
        paths: Any,
        name: str | None = None,
        hip_family: str | None = None,
        version: int | None = None,
        session_id: str | None = None,
        source_node: str | None = None,
        job_id: str | None = None,
        scene: Any = None,
    ) -> RunRecord:
        """Record one output run with its expanded paths frozen at accept time."""
        with self._txn(write=True) as cur:
            cur.execute(
                "INSERT INTO runs (run_id, kind, name, hip_family, version, session_id,"
                " source_node, job_id, paths, scene, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    kind,
                    name,
                    hip_family,
                    version,
                    session_id,
                    source_node,
                    job_id,
                    _dump(paths),
                    _dump(scene),
                    _now(),
                ),
            )
            return RunRecord._from_row(
                cur.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        """One run by id."""
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else RunRecord._from_row(row)

    def list_runs(self, *, kind: str | None = None, limit: int = 50) -> list[RunRecord]:
        """Recent runs, newest first."""
        sql = "SELECT * FROM runs"
        args: list[Any] = []
        if kind is not None:
            sql += " WHERE kind = ?"
            args.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [RunRecord._from_row(row) for row in self._conn.execute(sql, args)]

    # -- readable exports -------------------------------------------------

    def run_export(self, run_id: str) -> dict[str, Any]:
        """Run record as plain data, for the sidecar beside the scene file."""
        record = self.get_run(run_id)
        if record is None:
            raise UnknownRecord(f"no run {run_id}")
        return _export_dict(record, {"created_at": "created"})

    def job_export(self, job_id: str) -> dict[str, Any]:
        """Job record as plain data, for the readable copy of a finished job."""
        record = self.get_job(job_id)
        if record is None:
            raise UnknownRecord(f"no job {job_id}")
        return _export_dict(
            record,
            {"created_at": "created", "updated_at": "updated", "finished_at": "finished"},
        )


def write_export(record: Mapping[str, Any], path: Path | str) -> Path:
    """Write an export next to a scene file, utf-8 and readable by a person."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_text(text, encoding="utf-8")
    return target


def _export_dict(record: Any, times: Mapping[str, str]) -> dict[str, Any]:
    """Record fields plus a readable timestamp beside each stored one."""
    data = asdict(record)
    for field, label in times.items():
        data[f"{label}_utc"] = _iso(data.get(field))
    return data


def _first_free_alias(template: str, taken: Iterable[str]) -> str:
    """Lowest `{n}` from 1 up that the template does not already produce."""
    if "{n}" not in template:
        raise ValueError("alias template must contain {n}")
    used = set(taken)
    for index in range(1, MAX_ALIAS_INDEX + 1):
        candidate = template.format(n=index)
        if candidate not in used:
            return candidate
    raise StoreError(f"no free alias for template {template}")
