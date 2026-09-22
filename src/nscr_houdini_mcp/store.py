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

Time and liveness. A process that crashes cannot clean up after itself, so the
first question asked about any held slot is whether its owner process is still
alive. Wall clock ages are the second signal and only a budget: a clock step
backwards would make an age negative, so ages are clamped at zero and a step
forward can only make something look older, which at worst reclaims a slot
early from an owner that is already gone. Monotonic clocks are not stored: they
restart with the machine and are not comparable between processes.

This module never imports `hou`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

APP_DIR_NAME = "nscr-houdini-mcp"
HOME_ENV_VAR = "NSCR_MCP_HOME"
STORE_FILE_NAME = "coord.sqlite"

SCHEMA_VERSION = 2

SESSION_KINDS = frozenset({"gui", "hython"})
SESSION_STATES = frozenset({"live", "busy", "unresponsive", "crashed", "gone"})
SESSION_GONE = "gone"

# States that still hold a slot against the pool cap. A reservation counts from
# the moment it is made, before hython has started.
WORKER_ACTIVE_STATES = ("reserved", "starting", "running", "leased", "stopping")
WORKER_STARTING_STATES = ("reserved", "starting")
WORKER_FINAL_STATES = ("failed", "stopped")
WORKER_STATES = frozenset(WORKER_ACTIVE_STATES + WORKER_FINAL_STATES)

OPERATION_STATES = frozenset({"running", "done", "failed"})
JOB_STATES = frozenset({"queued", "running", "done", "failed", "cancelled", "lost"})
JOB_FINAL_STATES = frozenset({"done", "failed", "cancelled", "lost"})
JOB_LIVE_STATES = ("queued", "running")

MAX_ALIAS_INDEX = 4096

DEFAULT_START_BUDGET_S = 180.0
DEFAULT_OPERATION_LEASE_S = 300.0

# Folder names that usually mean a synced or mounted location. The store must
# stay on a local disk, so a match is worth saying out loud, not worth failing
# over: the check is a name comparison and nothing else.
SHARED_FOLDER_NAMES = (
    "dropbox",
    "onedrive",
    "google drive",
    "googledrive",
    "icloud drive",
    "com~apple~clouddocs",
    "nextcloud",
    "owncloud",
    "box sync",
    "pcloud",
    "creative cloud files",
)


class StoreError(Exception):
    """Base class for coordination store failures."""


class StoreBusy(StoreError):
    """The store stayed locked by other processes for longer than allowed."""


class DuplicateRecord(StoreError):
    """An id or a name that must be unique is already in the store."""


class UnknownRecord(StoreError):
    """A session, reservation, job or run id is not in the store."""


class PoolFull(StoreError):
    """No worker slot is free under the current cap."""


class AliasInUse(DuplicateRecord):
    """The requested alias already belongs to a live session."""


class OperationMismatch(StoreError):
    """An operation id came back with different arguments than the first time."""


class SceneReplaced(StoreError):
    """An operation id came back against a scene that has since been replaced."""

    def __init__(self, message: str, *, recorded_epoch: int, current_epoch: int) -> None:
        super().__init__(message)
        self.recorded_epoch = recorded_epoch
        self.current_epoch = current_epoch


class SchemaTooNew(StoreError):
    """The file on disk was written by a newer build of this package."""


class UndigestableArgument(StoreError):
    """An argument has no stable text form, so it cannot go in a digest."""


class _Clear:
    """Sentinel: write NULL, as against leaving a stored value alone."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "CLEAR"


CLEAR = _Clear()


def default_home() -> Path:
    """Per user state folder for this tool.

    Every component asks here for its folder, so the store, the session
    registry, the config file and the logs stay together and move together
    when the environment override is set.
    """
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


def shared_location_warning(path: Path | str) -> str | None:
    """One line when the path looks synced or mounted, otherwise nothing.

    SQLite locking is not reliable on a share, and a synced folder copies the
    file behind its own back. The caller decides what to do with the line.
    """
    text = str(path)
    if text.startswith("\\\\") or text.startswith("//"):
        return f"{text} looks like a network path. Keep the store on a local disk."
    parts = Path(text).parts
    lowered = [part.lower() for part in parts]
    for name in SHARED_FOLDER_NAMES:
        if any(name in part for part in lowered):
            return f"{text} looks like a synced folder. Keep the store on a local disk."
    if sys.platform != "win32" and len(parts) > 2 and lowered[1] in {"volumes", "net", "mnt"}:
        return f"{text} looks like a mounted volume. Keep the store on a local disk."
    return None


def process_is_alive(pid: int | None) -> bool:
    """Whether a pid is running, on every supported system.

    Pids are reused eventually, so this answers the cheap question and the
    owner token answers the exact one.
    """
    if pid is None or pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_process_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process, so it exists.
        return True
    except OSError:
        return True
    return True


def _windows_process_is_alive(pid: int) -> bool:
    """Windows has no signal 0, so ask the kernel for the process directly."""
    import ctypes

    process_query_limited_information = 0x1000
    error_access_denied = 5
    still_active = 259

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        # Access denied means the process is there and belongs to somebody else.
        return kernel32.GetLastError() == error_access_denied
    try:
        code = ctypes.c_ulong()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return code.value == still_active
        return True
    finally:
        kernel32.CloseHandle(handle)


def digest_arguments(payload: Any) -> str:
    """Stable digest of one call's arguments, for operation receipts.

    The same arguments must digest the same in any process, so the value is
    put in a canonical form first rather than handed to a fallback that prints
    whatever an object happens to print. Arguments arrive over a JSON
    transport, so `1` and `1.0` are the same argument, and a type that JSON
    cannot carry is refused instead of being guessed at.
    """
    text = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """Canonical form of one argument value, or `UndigestableArgument`."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise UndigestableArgument(f"cannot digest the float {value!r}")
        return int(value) if value.is_integer() else value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        items = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UndigestableArgument(f"cannot digest the key {key!r}")
            items[key] = _canonical(item)
        return items
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        # A set has no order of its own, and iteration order changes between
        # processes, so sort the members by their own canonical text.
        members = [_canonical(item) for item in value]
        return sorted(members, key=lambda member: json.dumps(member, sort_keys=True))
    raise UndigestableArgument(f"cannot digest a value of type {type(value).__name__}")


def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=UTC).isoformat(timespec="seconds")


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True, default=str)


def _load(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _age(now: float, then: float | None) -> float:
    """Age in seconds, never negative, so a clock step cannot rewind a lease."""
    if then is None:
        return 0.0
    return max(0.0, now - then)


def _is_busy(error: sqlite3.Error) -> bool:
    text = str(error).lower()
    return "locked" in text or "busy" in text


def _translate(error: sqlite3.Error) -> StoreError:
    """Turn a raw database failure into one of this module's errors."""
    if isinstance(error, sqlite3.IntegrityError):
        return DuplicateRecord(str(error))
    if isinstance(error, sqlite3.OperationalError) and _is_busy(error):
        return StoreBusy(str(error))
    return StoreError(str(error))


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
    owner_pid: int | None
    start_deadline: float | None
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
            owner_pid=row["owner_pid"],
            start_deadline=row["start_deadline"],
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
    owner_pid: int | None
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
            owner_pid=row["owner_pid"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class OperationClaim:
    """What a caller learned by presenting an operation id.

    `claimed` says this caller may run the work. `outcome_unknown` says an
    earlier attempt ran and its result is not recorded, so the work may
    already be half done: with `claimed` it was abandoned and has been taken
    over, without it another process is still on it.
    """

    record: OperationRecord
    claimed: bool
    outcome_unknown: bool


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
    cancel_requested: bool
    worker_pid: int | None
    heartbeat_at: float | None
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
            cancel_requested=bool(row["cancel_requested"]),
            worker_pid=row["worker_pid"],
            heartbeat_at=row["heartbeat_at"],
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


# One entry per schema version, applied in order inside one transaction each.
# A step is never edited once it has shipped: a later change is a new step.
_SCHEMA_1 = (
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
    )
    """,
    "CREATE UNIQUE INDEX sessions_live_alias ON sessions(alias) WHERE state <> 'gone'",
    """
    CREATE TABLE workers (
        token       TEXT PRIMARY KEY,
        alias       TEXT NOT NULL,
        state       TEXT NOT NULL,
        session_id  TEXT,
        job_id      TEXT,
        reserved_at REAL NOT NULL,
        leased_at   REAL NOT NULL
    )
    """,
    "CREATE UNIQUE INDEX workers_live_alias"
    " ON workers(alias) WHERE state NOT IN ('failed', 'stopped')",
    """
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
    )
    """,
    """
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
    )
    """,
    "CREATE INDEX jobs_by_state ON jobs(state, updated_at)",
    """
    CREATE TABLE versions (
        kind       TEXT NOT NULL,
        name       TEXT NOT NULL,
        hip_family TEXT NOT NULL,
        version    INTEGER NOT NULL,
        run_id     TEXT,
        created_at REAL NOT NULL,
        PRIMARY KEY (kind, name, hip_family, version)
    )
    """,
    """
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
    )
    """,
)

# Owners and budgets, so a process that dies stops holding what it took.
_SCHEMA_2 = (
    "ALTER TABLE workers ADD COLUMN owner_pid INTEGER",
    "ALTER TABLE workers ADD COLUMN start_deadline REAL",
    "ALTER TABLE operations ADD COLUMN owner_pid INTEGER",
    "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE jobs ADD COLUMN worker_pid INTEGER",
    "ALTER TABLE jobs ADD COLUMN heartbeat_at REAL",
)

MIGRATIONS = (_SCHEMA_1, _SCHEMA_2)


class Store:
    """Handle on the coordination store. One per process or per thread.

    The connection is not shared between threads. Several processes opening the
    same file is the normal case and is what the store is for.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        busy_timeout_s: float = 10.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.location_warning = shared_location_warning(self.path)
        self._clock = clock or time.time
        self._busy_timeout_s = busy_timeout_s
        self._conn = sqlite3.connect(str(self.path), timeout=busy_timeout_s, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._in_txn = False
        try:
            # The busy handler comes first, before anything that can block.
            self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_s * 1000)}")
            # Switching the journal takes a lock the busy handler does not
            # cover, and it is the first move of every process on a fresh file.
            self._retry_while_busy(lambda: self._conn.execute("PRAGMA journal_mode=WAL"))
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._retry_while_busy(self._apply_migrations)
        except BaseException:
            # An open that did not finish leaves no handle on the file.
            self._conn.close()
            raise

    # -- lifetime ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _now(self) -> float:
        return self._clock()

    # -- transactions -----------------------------------------------------

    @contextmanager
    def _txn(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        """Short transaction. Private so no caller can hold one open.

        Writes take `BEGIN IMMEDIATE`, so a caller that reads a count and then
        decides on it wins or loses the whole decision, never half of it. The
        open flag is set only once the statement has been accepted, so a
        failed start leaves the handle usable.
        """
        if self._in_txn:
            raise StoreError("store transactions do not nest")
        try:
            self._conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        except sqlite3.Error as error:
            raise _translate(error) from error
        self._in_txn = True
        try:
            yield self._conn
        except BaseException as error:
            self._conn.rollback()
            if isinstance(error, sqlite3.Error):
                raise _translate(error) from error
            raise
        else:
            self._conn.commit()
        finally:
            self._in_txn = False

    def _retry_while_busy(self, action: Callable[[], Any]) -> Any:
        """Repeat an action that other processes can lock out, then give up.

        Uses a monotonic clock for the wait itself, which is safe because the
        value never leaves this call.
        """
        deadline = time.monotonic() + self._busy_timeout_s
        delay = 0.01
        while True:
            try:
                return action()
            except (sqlite3.OperationalError, StoreBusy) as error:
                if isinstance(error, sqlite3.OperationalError) and not _is_busy(error):
                    raise _translate(error) from error
                if time.monotonic() >= deadline:
                    raise StoreBusy(f"{self.path} stayed locked by other processes") from error
                time.sleep(delay)
                delay = min(delay * 2, 0.1)

    def _apply_migrations(self) -> None:
        with self._txn(write=True) as db:
            current = int(db.execute("PRAGMA user_version").fetchone()[0])
            if current > len(MIGRATIONS):
                raise SchemaTooNew(
                    f"{self.path} is at schema {current}, this build knows {SCHEMA_VERSION}"
                )
            for step, statements in enumerate(MIGRATIONS[current:], start=current + 1):
                for statement in statements:
                    db.execute(statement)
                db.execute(f"PRAGMA user_version={step}")

    def schema_version(self) -> int:
        """Schema version of the open file."""
        return int(self._read_one("PRAGMA user_version")[0])

    # -- reads ------------------------------------------------------------

    def _read_all(self, sql: str, args: Sequence[Any] = ()) -> list[sqlite3.Row]:
        try:
            return self._conn.execute(sql, args).fetchall()
        except sqlite3.Error as error:
            raise _translate(error) from error

    def _read_one(self, sql: str, args: Sequence[Any] = ()) -> sqlite3.Row | None:
        try:
            return self._conn.execute(sql, args).fetchone()
        except sqlite3.Error as error:
            raise _translate(error) from error

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
        now = self._now()
        with self._txn(write=True) as db:
            self._reclaim_sessions(db, now)
            if alias_template is not None:
                taken = {
                    row["alias"]
                    for row in db.execute(
                        "SELECT alias FROM sessions WHERE state <> ?", (SESSION_GONE,)
                    )
                }
                name = _first_free_alias(alias_template, taken)
            else:
                name = alias
                row = db.execute(
                    "SELECT session_id FROM sessions WHERE alias = ? AND state <> ?",
                    (name, SESSION_GONE),
                ).fetchone()
                if row is not None:
                    raise AliasInUse(f"alias {name} belongs to session {row['session_id']}")
            db.execute(
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
                db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Session by id, gone or not. Ids are never reused."""
        row = self._read_one("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return None if row is None else SessionRecord._from_row(row)

    def resolve_session(self, handle: str) -> SessionRecord | None:
        """Session by id, or by the alias of a session that is not gone."""
        found = self.get_session(handle)
        if found is not None:
            return found
        row = self._read_one(
            "SELECT * FROM sessions WHERE alias = ? AND state <> ?"
            " ORDER BY started_at DESC, rowid DESC",
            (handle, SESSION_GONE),
        )
        return None if row is None else SessionRecord._from_row(row)

    def list_sessions(self, *, include_gone: bool = False) -> list[SessionRecord]:
        """Sessions, oldest first."""
        sql = "SELECT * FROM sessions"
        args: tuple[Any, ...] = ()
        if not include_gone:
            sql += " WHERE state <> ?"
            args = (SESSION_GONE,)
        sql += " ORDER BY started_at, rowid"
        return [SessionRecord._from_row(row) for row in self._read_all(sql, args)]

    def touch_session(self, session_id: str, *, state: str | None = None) -> float:
        """Write a heartbeat, and a new state when one is given."""
        if state is not None and state not in SESSION_STATES:
            raise ValueError(f"unknown session state: {state}")
        now = self._now()
        with self._txn(write=True) as db:
            if state is None:
                written = db.execute(
                    "UPDATE sessions SET heartbeat_at = ? WHERE session_id = ?", (now, session_id)
                )
            else:
                written = db.execute(
                    "UPDATE sessions SET heartbeat_at = ?, state = ? WHERE session_id = ?",
                    (now, state, session_id),
                )
            if written.rowcount == 0:
                raise UnknownRecord(f"no session {session_id}")
        return now

    def bump_scene_epoch(self, session_id: str, *, hip_path: str | None = None) -> int:
        """Count a scene open, new or reset. Returns the new epoch."""
        with self._txn(write=True) as db:
            row = db.execute(
                "SELECT scene_epoch FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise UnknownRecord(f"no session {session_id}")
            epoch = int(row["scene_epoch"]) + 1
            if hip_path is None:
                db.execute(
                    "UPDATE sessions SET scene_epoch = ? WHERE session_id = ?", (epoch, session_id)
                )
            else:
                db.execute(
                    "UPDATE sessions SET scene_epoch = ?, hip_path = ? WHERE session_id = ?",
                    (epoch, hip_path, session_id),
                )
        return epoch

    def reclaim_sessions(self) -> list[str]:
        """Mark sessions gone whose process is not there any more.

        A session that crashed cannot end its own row, and its alias would
        otherwise stay taken for good. `register_session` does this for itself,
        so this is for a caller that only wants to tidy up or to read an
        honest list afterwards. Returns the ids that were marked.
        """
        with self._txn(write=True) as db:
            return self._reclaim_sessions(db, self._now())

    def _reclaim_sessions(self, db: sqlite3.Connection, now: float) -> list[str]:
        rows = db.execute(
            "SELECT session_id, pid FROM sessions WHERE state <> ?", (SESSION_GONE,)
        ).fetchall()
        reclaimed: list[str] = []
        for row in rows:
            if process_is_alive(row["pid"]):
                continue
            db.execute(
                "UPDATE sessions SET state = ?, heartbeat_at = ? WHERE session_id = ?",
                (SESSION_GONE, now, row["session_id"]),
            )
            reclaimed.append(row["session_id"])
        return reclaimed

    def end_session(self, session_id: str) -> None:
        """Mark a session gone, which frees its alias for a later process."""
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE sessions SET state = ?, heartbeat_at = ? WHERE session_id = ?",
                (SESSION_GONE, self._now(), session_id),
            )
            if written.rowcount == 0:
                raise UnknownRecord(f"no session {session_id}")

    # -- workers ----------------------------------------------------------

    def reserve_worker(
        self,
        *,
        cap: int,
        token: str,
        alias_template: str = "w{n}",
        job_id: str | None = None,
        owner_pid: int | None = None,
        start_budget_s: float = DEFAULT_START_BUDGET_S,
    ) -> WorkerRecord:
        """Take a slot under the pool cap, or raise `PoolFull`.

        Slots held by processes that are gone are reclaimed first, in the same
        transaction, so a crash does not shrink the pool for good. Counting and
        inserting are that same transaction, so two processes that see the same
        count cannot both get the last slot. A reservation counts from here,
        before hython starts, and stops counting once the worker ends up failed
        or stopped.
        """
        if cap < 1:
            raise ValueError("cap must be at least 1")
        now = self._now()
        pid = os.getpid() if owner_pid is None else owner_pid
        placeholders = ", ".join("?" * len(WORKER_ACTIVE_STATES))
        with self._txn(write=True) as db:
            self._reclaim_workers(db, now)
            rows = db.execute(
                f"SELECT alias FROM workers WHERE state IN ({placeholders})",
                WORKER_ACTIVE_STATES,
            ).fetchall()
            if len(rows) >= cap:
                raise PoolFull(f"{len(rows)} of {cap} worker slots are in use")
            alias = _first_free_alias(alias_template, {row["alias"] for row in rows})
            db.execute(
                "INSERT INTO workers (token, alias, state, session_id, job_id, owner_pid,"
                " start_deadline, reserved_at, leased_at)"
                " VALUES (?, ?, 'reserved', NULL, ?, ?, ?, ?, ?)",
                (token, alias, job_id, pid, now + start_budget_s, now, now),
            )
            return WorkerRecord._from_row(
                db.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            )

    def reclaim_workers(self) -> list[str]:
        """Fail slots whose owner is gone or that never finished starting.

        Returns the tokens that were reclaimed. `reserve_worker` does this for
        itself, so this is for a caller that only wants to tidy up or report.
        """
        with self._txn(write=True) as db:
            return self._reclaim_workers(db, self._now())

    def _reclaim_workers(self, db: sqlite3.Connection, now: float) -> list[str]:
        placeholders = ", ".join("?" * len(WORKER_ACTIVE_STATES))
        rows = db.execute(
            f"SELECT * FROM workers WHERE state IN ({placeholders})", WORKER_ACTIVE_STATES
        ).fetchall()
        reclaimed: list[str] = []
        for row in rows:
            owner_gone = row["owner_pid"] is not None and not process_is_alive(row["owner_pid"])
            deadline = row["start_deadline"]
            never_started = (
                row["state"] in WORKER_STARTING_STATES and deadline is not None and now > deadline
            )
            if owner_gone or never_started:
                db.execute(
                    "UPDATE workers SET state = 'failed', job_id = NULL, leased_at = ?"
                    " WHERE token = ?",
                    (now, row["token"]),
                )
                reclaimed.append(row["token"])
        return reclaimed

    def set_worker_state(
        self,
        token: str,
        state: str,
        *,
        session_id: str | None | _Clear = None,
        job_id: str | None | _Clear = None,
        owner_pid: int | None = None,
        start_budget_s: float | None = None,
    ) -> WorkerRecord:
        """Move a reservation on. The token proves who owns the slot.

        A field left out keeps its stored value. Pass `CLEAR` to empty one, for
        example to hand a worker back to the pool when its job is done.
        """
        if state not in WORKER_STATES:
            raise ValueError(f"unknown worker state: {state}")
        now = self._now()
        with self._txn(write=True) as db:
            row = db.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            if row is None:
                raise UnknownRecord(f"no worker reservation {token}")
            deadline = row["start_deadline"]
            if start_budget_s is not None:
                deadline = now + start_budget_s
            db.execute(
                "UPDATE workers SET state = ?, session_id = ?, job_id = ?, owner_pid = ?,"
                " start_deadline = ?, leased_at = ? WHERE token = ?",
                (
                    state,
                    _settle(session_id, row["session_id"]),
                    _settle(job_id, row["job_id"]),
                    row["owner_pid"] if owner_pid is None else owner_pid,
                    deadline,
                    now,
                    token,
                ),
            )
            return WorkerRecord._from_row(
                db.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            )

    def release_worker(self, token: str, *, state: str = "stopped") -> WorkerRecord:
        """Give a slot back. Use `failed` when the start never came up."""
        if state not in WORKER_FINAL_STATES:
            raise ValueError(f"not a final worker state: {state}")
        return self.set_worker_state(token, state, job_id=CLEAR)

    def touch_worker_lease(self, token: str) -> float:
        """Renew the idle lease. Routing a call to a worker renews it."""
        now = self._now()
        with self._txn(write=True) as db:
            written = db.execute("UPDATE workers SET leased_at = ? WHERE token = ?", (now, token))
            if written.rowcount == 0:
                raise UnknownRecord(f"no worker reservation {token}")
        return now

    def get_worker(self, token: str) -> WorkerRecord | None:
        """One reservation by its owner token."""
        row = self._read_one("SELECT * FROM workers WHERE token = ?", (token,))
        return None if row is None else WorkerRecord._from_row(row)

    def list_workers(self, *, active_only: bool = True) -> list[WorkerRecord]:
        """Reservations, oldest first."""
        sql = "SELECT * FROM workers"
        args: tuple[Any, ...] = ()
        if active_only:
            sql += f" WHERE state IN ({', '.join('?' * len(WORKER_ACTIVE_STATES))})"
            args = WORKER_ACTIVE_STATES
        sql += " ORDER BY reserved_at, rowid"
        return [WorkerRecord._from_row(row) for row in self._read_all(sql, args)]

    def idle_workers(self, max_idle_s: float) -> list[WorkerRecord]:
        """Live workers with no job whose lease is older than the limit.

        An expired lease only says a worker may exit by itself. It never hands
        a reserved worker to somebody else, and a worker on a job never counts
        as idle however long the job runs. A worker whose owner is gone is a
        job for `reclaim_workers`, not an idle worker.
        """
        now = self._now()
        idle = []
        for record in self.list_workers():
            if record.job_id is not None:
                continue
            if record.owner_pid is not None and not process_is_alive(record.owner_pid):
                continue
            if _age(now, record.leased_at) >= max_idle_s:
                idle.append(record)
        return sorted(idle, key=lambda record: record.leased_at)

    # -- operation receipts ----------------------------------------------

    def begin_operation(
        self,
        operation_id: str,
        digest: str,
        *,
        session_id: str | None = None,
        scene_epoch: int | None = None,
        owner_pid: int | None = None,
        lease_s: float = DEFAULT_OPERATION_LEASE_S,
    ) -> OperationClaim:
        """Claim an operation id, or hand back what it did the first time.

        A retry after a lost reply passes the same id and the same digest and
        gets the stored outcome. The same id with different arguments, or from
        a different session, is a different call by mistake and raises
        `OperationMismatch`. An id presented against a scene that has moved on
        raises `SceneReplaced`, because the stored outcome describes a scene
        that is gone.

        A receipt left `running` by a process that died is taken over, and the
        claim says the outcome is unknown so the caller can check the scene
        before repeating the work.
        """
        now = self._now()
        pid = os.getpid() if owner_pid is None else owner_pid
        with self._txn(write=True) as db:
            row = db.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO operations (operation_id, session_id, scene_epoch, digest, state,"
                    " outcome, error, job_id, owner_pid, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, 'running', NULL, NULL, NULL, ?, ?, ?)",
                    (operation_id, session_id, scene_epoch, digest, pid, now, now),
                )
                stored = db.execute(
                    "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
                ).fetchone()
                return OperationClaim(OperationRecord._from_row(stored), True, False)

            if row["digest"] != digest:
                raise OperationMismatch(
                    f"operation {operation_id} was recorded with different arguments"
                )
            if session_id is not None and row["session_id"] not in (None, session_id):
                raise OperationMismatch(
                    f"operation {operation_id} belongs to session {row['session_id']}"
                )
            if (
                scene_epoch is not None
                and row["scene_epoch"] is not None
                and row["scene_epoch"] != scene_epoch
            ):
                raise SceneReplaced(
                    f"operation {operation_id} was recorded against an earlier scene",
                    recorded_epoch=int(row["scene_epoch"]),
                    current_epoch=scene_epoch,
                )
            if row["state"] != "running":
                return OperationClaim(OperationRecord._from_row(row), False, False)

            owner_live = process_is_alive(row["owner_pid"]) if row["owner_pid"] else False
            if owner_live and _age(now, row["updated_at"]) <= lease_s:
                # Somebody else is on it. The outcome is not known yet.
                return OperationClaim(OperationRecord._from_row(row), False, True)
            db.execute(
                "UPDATE operations SET owner_pid = ?, updated_at = ? WHERE operation_id = ?",
                (pid, now, operation_id),
            )
            taken = db.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            return OperationClaim(OperationRecord._from_row(taken), True, True)

    def touch_operation(self, operation_id: str) -> float:
        """Renew the lease on a receipt that is still being worked on."""
        now = self._now()
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE operations SET updated_at = ? WHERE operation_id = ?", (now, operation_id)
            )
            if written.rowcount == 0:
                raise UnknownRecord(f"no operation {operation_id}")
        return now

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
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE operations SET state = ?, outcome = ?, error = ?, job_id = ?,"
                " updated_at = ? WHERE operation_id = ?",
                (state, _dump(outcome), _dump(error), job_id, self._now(), operation_id),
            )
            if written.rowcount == 0:
                raise UnknownRecord(f"no operation {operation_id}")
            return OperationRecord._from_row(
                db.execute(
                    "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
                ).fetchone()
            )

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        """One receipt by id."""
        row = self._read_one("SELECT * FROM operations WHERE operation_id = ?", (operation_id,))
        return None if row is None else OperationRecord._from_row(row)

    def prune_operations(self, max_age_s: float) -> int:
        """Drop receipts older than the retention window. Returns the count."""
        cutoff = self._now() - max_age_s
        with self._txn(write=True) as db:
            return db.execute("DELETE FROM operations WHERE updated_at < ?", (cutoff,)).rowcount

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
        worker_pid: int | None = None,
    ) -> JobRecord:
        """Record an accepted job, including the scene identity it consumes."""
        if state not in JOB_STATES:
            raise ValueError(f"unknown job state: {state}")
        now = self._now()
        with self._txn(write=True) as db:
            db.execute(
                "INSERT INTO jobs (job_id, session_id, kind, state, weight, progress, outputs,"
                " error, scene, cancel_requested, worker_pid, heartbeat_at, created_at,"
                " updated_at, finished_at)"
                " VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, 0, ?, ?, ?, ?, NULL)",
                (
                    job_id,
                    session_id,
                    kind,
                    state,
                    weight,
                    _dump(progress),
                    _dump(scene),
                    worker_pid,
                    now,
                    now,
                    now,
                ),
            )
            return JobRecord._from_row(
                db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            )

    def update_job(
        self,
        job_id: str,
        *,
        state: str | None = None,
        progress: Any = None,
        outputs: Any = None,
        error: Any = None,
        worker_pid: int | None = None,
    ) -> JobRecord:
        """Write progress, outputs so far or a final state."""
        if state is not None and state not in JOB_STATES:
            raise ValueError(f"unknown job state: {state}")
        now = self._now()
        with self._txn(write=True) as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise UnknownRecord(f"no job {job_id}")
            new_state = state or row["state"]
            finished = now if new_state in JOB_FINAL_STATES else row["finished_at"]
            db.execute(
                "UPDATE jobs SET state = ?, progress = ?, outputs = ?, error = ?, worker_pid = ?,"
                " heartbeat_at = ?, updated_at = ?, finished_at = ? WHERE job_id = ?",
                (
                    new_state,
                    _dump(progress) if progress is not None else row["progress"],
                    _dump(outputs) if outputs is not None else row["outputs"],
                    _dump(error) if error is not None else row["error"],
                    row["worker_pid"] if worker_pid is None else worker_pid,
                    now,
                    now,
                    finished,
                    job_id,
                ),
            )
            return JobRecord._from_row(
                db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            )

    def touch_job(self, job_id: str, *, worker_pid: int | None = None) -> float:
        """Heartbeat from whoever is running the job."""
        now = self._now()
        with self._txn(write=True) as db:
            row = db.execute("SELECT worker_pid FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise UnknownRecord(f"no job {job_id}")
            db.execute(
                "UPDATE jobs SET heartbeat_at = ?, worker_pid = ? WHERE job_id = ?",
                (now, row["worker_pid"] if worker_pid is None else worker_pid, job_id),
            )
        return now

    def request_job_cancel(self, job_id: str) -> JobRecord:
        """Ask for a job to stop. Whoever runs it reads the flag and acts."""
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
                (self._now(), job_id),
            )
            if written.rowcount == 0:
                raise UnknownRecord(f"no job {job_id}")
            return JobRecord._from_row(
                db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            )

    def stale_jobs(self, max_silence_s: float) -> list[JobRecord]:
        """Unfinished jobs whose runner is gone or has said nothing for a while.

        These are the candidates for `lost`. The caller decides, because only
        it knows whether the outputs written so far are worth keeping.
        """
        now = self._now()
        rows = self._read_all(
            f"SELECT * FROM jobs WHERE state IN ({', '.join('?' * len(JOB_LIVE_STATES))})"
            " ORDER BY created_at, rowid",
            JOB_LIVE_STATES,
        )
        stale = []
        for row in rows:
            record = JobRecord._from_row(row)
            runner_gone = record.worker_pid is not None and not process_is_alive(record.worker_pid)
            silent = _age(now, record.heartbeat_at) >= max_silence_s
            if runner_gone or silent:
                stale.append(record)
        return stale

    def get_job(self, job_id: str) -> JobRecord | None:
        """One job by id."""
        row = self._read_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
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
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        return [JobRecord._from_row(row) for row in self._read_all(sql, args)]

    def prune_jobs(self, max_age_s: float) -> int:
        """Drop job rows older than the retention window. Returns the count."""
        cutoff = self._now() - max_age_s
        with self._txn(write=True) as db:
            return db.execute("DELETE FROM jobs WHERE updated_at < ?", (cutoff,)).rowcount

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
        with self._txn(write=True) as db:
            row = db.execute(
                "SELECT MAX(version) AS top FROM versions"
                " WHERE kind = ? AND name = ? AND hip_family = ?",
                (kind, name, hip_family),
            ).fetchone()
            version = int(row["top"] or 0) + 1
            db.execute(
                "INSERT INTO versions (kind, name, hip_family, version, run_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (kind, name, hip_family, version, run_id, self._now()),
            )
        return version

    def attach_version_run(
        self, *, kind: str, name: str, hip_family: str, version: int, run_id: str
    ) -> None:
        """Point a version at its run, for a number taken before the run existed."""
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE versions SET run_id = ? WHERE kind = ? AND name = ? AND hip_family = ?"
                " AND version = ?",
                (run_id, kind, name, hip_family, version),
            )
            if written.rowcount == 0:
                raise UnknownRecord(f"no version {version} for {kind} {name}")

    def latest_version(self, *, kind: str, name: str, hip_family: str) -> int:
        """Highest number handed out so far, or 0 when there is none."""
        row = self._read_one(
            "SELECT MAX(version) AS top FROM versions WHERE kind = ? AND name = ? AND"
            " hip_family = ?",
            (kind, name, hip_family),
        )
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
        with self._txn(write=True) as db:
            db.execute(
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
                    self._now(),
                ),
            )
            return RunRecord._from_row(
                db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        """One run by id."""
        row = self._read_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return None if row is None else RunRecord._from_row(row)

    def list_runs(self, *, kind: str | None = None, limit: int = 50) -> list[RunRecord]:
        """Recent runs, newest first."""
        sql = "SELECT * FROM runs"
        args: list[Any] = []
        if kind is not None:
            sql += " WHERE kind = ?"
            args.append(kind)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        return [RunRecord._from_row(row) for row in self._read_all(sql, args)]

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


def _settle(given: Any, stored: Any) -> Any:
    """None keeps what is stored, `CLEAR` empties it, anything else replaces it."""
    if given is None:
        return stored
    if isinstance(given, _Clear):
        return None
    return given


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
