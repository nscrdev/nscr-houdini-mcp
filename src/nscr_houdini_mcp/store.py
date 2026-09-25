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
import select
import sqlite3
import subprocess
import sys
import threading
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

SCHEMA_VERSION = 12

SESSION_KINDS = frozenset({"gui", "hython"})
SESSION_STATES = frozenset({"live", "busy", "unresponsive", "crashed", "gone"})
SESSION_GONE = "gone"
SESSION_LIVE = "live"
# The process is there and writing heartbeats, and its port is not answering.
SESSION_UNRESPONSIVE = "unresponsive"
# How a session ended: on its own terms, or with its process found missing.
SESSION_CRASHED = "crashed"
SESSION_ENDINGS = frozenset({SESSION_GONE, SESSION_CRASHED})

# States that still hold a slot against the pool cap. A reservation counts from
# the moment it is made, before hython has started.
WORKER_ACTIVE_STATES = ("reserved", "starting", "running", "leased", "stopping")
WORKER_STARTING_STATES = ("reserved", "starting")
# A worker that is stopping still holds its slot, but is never taken for a job:
# the job would be lost when the process goes.
WORKER_LEASABLE_STATES = ("reserved", "starting", "running", "leased")
WORKER_FINAL_STATES = ("failed", "stopped")
WORKER_STATES = frozenset(WORKER_ACTIVE_STATES + WORKER_FINAL_STATES)

# A receipt whose attempt stopped without recording anything. It is final:
# the work may have happened, so the id is never run again.
OPERATION_ABANDONED = "abandoned"
OPERATION_STATES = frozenset({"running", "done", "failed", OPERATION_ABANDONED})
JOB_STATES = frozenset({"queued", "running", "done", "failed", "cancelled", "lost"})
JOB_FINAL_STATES = frozenset({"done", "failed", "cancelled", "lost"})
JOB_LIVE_STATES = ("queued", "running")

# Where a job may go from where it is. A job that has ended stays ended, with
# one exception: a job found `lost` whose work then turns out to have ended
# after all takes how it really ended, as a late finish.
JOB_MOVES = {
    "queued": frozenset({"running", "done", "failed", "cancelled", "lost"}),
    "running": frozenset({"done", "failed", "cancelled", "lost"}),
}
LATE_FINISHES = frozenset({"done", "failed", "cancelled"})

# What a job that was still going says once its session is known to have ended.
SESSION_ENDED_ERROR = {"code": "SESSION_ENDED", "message": "the session running this job ended"}

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


class WorkerTaken(StoreError):
    """That worker is already on another job."""


class JobIdTaken(DuplicateRecord):
    """A job is kept under that id, and it is too soon to take the id again."""


class JobMoveRefused(StoreError):
    """A job cannot move from the state it is in to the one asked for."""


class AliasInUse(DuplicateRecord):
    """The requested alias already belongs to a live session."""


class OperationMismatch(StoreError):
    """An operation id came back with different arguments than the first time."""


class ParmHeld(StoreError):
    """An output parameter is already frozen by another run."""

    def __init__(self, message: str, *, run_id: str | None) -> None:
        super().__init__(message)
        self.run_id = run_id


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


# How long a process listing may take before the answer is not worth waiting
# for. A pid is not an identity on its own: numbers are handed out again, so a
# session file left by a process that crashed can name a pid that belongs to
# something else by now. A pid and the moment its process started is an
# identity that holds, and every system will say the second part in some form
# of its own. `None` means this system would not say, and a caller that gets
# `None` has learned nothing and must not pretend otherwise.
PS_TIMEOUT_S = 5.0


def process_start_stamp(pid: int | None = None) -> str | None:
    """When a process started, in whatever form this system reports it."""
    number = os.getpid() if pid is None else pid
    if number <= 0:
        return None
    if sys.platform == "win32":
        return _windows_start(number)
    if sys.platform == "linux":
        return _linux_start(number)
    if sys.platform == "darwin":
        return _known_starts.stamp(number)
    return None


def same_process(pid: int | None, stamp: str | None) -> bool | None:
    """Whether this pid is still the process that recorded that stamp.

    `True` and `False` are answers. `None` says the question could not be
    settled here, which happens when nothing recorded a stamp or when the
    system will not give one.
    """
    if not process_is_alive(pid):
        return False
    if not stamp:
        return None
    current = process_start_stamp(pid)
    if current is None:
        return None
    return current == stamp


def _linux_start(pid: int) -> str | None:
    """Field 22 of the process stat file: start time in clock ticks.

    The name of the program sits in brackets and may itself contain brackets
    and spaces, so the fields are counted from the last closing bracket.
    """
    try:
        text = (
            open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace")  # noqa: SIM115
            .read()
            .strip()
        )
    except OSError:
        return None
    tail = text.rpartition(")")[2].split()
    # After the name come state and 19 more fields before start time.
    if len(tail) < 20:
        return None
    return tail[19]


def _ps_start(pid: int) -> str | None:
    """Ask the process listing, which every system of this kind ships."""
    try:
        finished = subprocess.run(  # noqa: S603 - a fixed command with a number
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    stamp = finished.stdout.strip()
    return stamp or None


class _KnownStarts:
    """Start stamps already read from the process listing, kept while their
    process runs.

    Asking the listing starts a program, which costs milliseconds, and every
    read of the session list asks once for each live session. A stamp is kept
    together with a kernel watch on its process's exit, set before the listing
    is read, so a kept stamp is handed out only while that very process has not
    ended. A pid that has been given to a new process is read afresh. Where the
    kernel offers no such watch, or refuses one, nothing is kept and every
    question goes to the listing as before.

    A process that ends while it is being read has no stamp to give: the
    listing may have read it or whatever took its pid after it, so the pid is
    read once more under a new watch, and the answer is nothing when that
    races too. One read per pid is under way at a time, and the others asking
    for the same pid wait for it. Only a few watches are opened at once; a
    read beyond that asks the listing without one.
    """

    # Each kept stamp holds one open watch, so the number is bounded.
    LIMIT = 64
    # Watches opened for reads under way at once.
    READING = 8
    # Tries at a pid whose process ends while it is read.
    TRIES = 2

    def __init__(self) -> None:
        self.forget_in_child()

    def forget_in_child(self) -> None:
        """Start empty, without closing anything: for a fresh or forked process.

        A forked child inherits the numbers of the parent's watches but not
        the watches, and may already have given those numbers to files of its
        own, so they are dropped rather than closed. The lock is made again
        because the parent may have held it at the moment of the fork.
        """
        self._owner = os.getpid()
        self._lock = threading.Lock()
        self._kept: dict[int, tuple[str, Any]] = {}
        self._reading: dict[int, threading.Event] = {}
        self._slots = threading.BoundedSemaphore(self.READING)

    def stamp(self, pid: int) -> str | None:
        if self._owner != os.getpid():
            # A fork the hook did not see, such as one made without it.
            self.forget_in_child()
        while True:
            with self._lock:
                kept = self._kept.get(pid)
                if kept is not None:
                    if not _has_exited(kept[1]):
                        return kept[0]
                    del self._kept[pid]
                    kept[1].close()
                under_way = self._reading.get(pid)
                if under_way is None:
                    self._reading[pid] = threading.Event()
                    break
            # Another thread is reading this pid: its answer is kept for this
            # one, or, when it could not be kept, this one reads next.
            under_way.wait(PS_TIMEOUT_S + 1.0)
        try:
            return self._read(pid)
        finally:
            with self._lock:
                self._reading.pop(pid).set()

    def _read(self, pid: int) -> str | None:
        for _ in range(self.TRIES):
            if not self._slots.acquire(blocking=False):
                return _ps_start(pid)
            try:
                watch = _watch_exit(pid)
                if watch is _GONE:
                    # The kernel will not watch it, yet the pid is taken, as
                    # by a process that has exited and not been reaped: the
                    # listing still tells a different process apart.
                    return _ps_start(pid) if process_is_alive(pid) else None
                stamp = _ps_start(pid)
                if watch is None:
                    return stamp
                if stamp is None:
                    watch.close()
                    return None
                if _has_exited(watch):
                    # The listing may have read the process that ended or one
                    # that took its pid since: neither stamp can be trusted.
                    watch.close()
                    continue
                self._keep(pid, stamp, watch)
                return stamp
            finally:
                self._slots.release()
        return None

    def _keep(self, pid: int, stamp: str, watch: Any) -> None:
        with self._lock:
            replaced = self._kept.pop(pid, None)
            if replaced is not None:
                replaced[1].close()
            # Processes that have ended go first, then the oldest kept. A read
            # happens once per new process, so the sweep is rare.
            for ended in [key for key, (_, held) in self._kept.items() if _has_exited(held)]:
                self._kept.pop(ended)[1].close()
            while len(self._kept) >= self.LIMIT:
                self._kept.pop(next(iter(self._kept)))[1].close()
            self._kept[pid] = (stamp, watch)


# What `_watch_exit` says when there is no such process to watch.
_GONE = object()


def _watch_exit(pid: int) -> Any:
    """A kernel queue that will hold an event once `pid` exits.

    `_GONE` when no process has that pid, and nothing when this system has no
    such watch or will not open one now, such as when it is out of handles.
    """
    make = getattr(select, "kqueue", None)
    if make is None:
        return None
    try:
        queue = make()
    except OSError:
        return None
    try:
        event = select.kevent(
            pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT,
        )
        queue.control([event], 0, 0)
    except ProcessLookupError:
        queue.close()
        return _GONE
    except (OSError, ValueError, OverflowError):
        queue.close()
        return None
    return queue


def _has_exited(queue: Any) -> bool:
    """Whether the process a queue watches has exited. Never waits."""
    try:
        return bool(queue.control(None, 1, 0))
    except (OSError, ValueError):
        # A watch that cannot be read proves nothing: read the listing again.
        return True


_known_starts = _KnownStarts()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_known_starts.forget_in_child)


def _windows_start(pid: int) -> str | None:
    """Creation time from the kernel, as a plain number."""
    import ctypes
    import ctypes.wintypes

    process_query_limited_information = 0x1000
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        created = ctypes.wintypes.FILETIME()
        exited = ctypes.wintypes.FILETIME()
        kernel = ctypes.wintypes.FILETIME()
        user = ctypes.wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        if not ok:
            return None
        return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
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


def _flag(value: Any) -> bool | None:
    """A column that holds yes, no or nothing yet."""
    return None if value is None else bool(value)


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
    pid_start: str | None
    port: int | None
    state: str
    scene_epoch: int
    hip_path: str | None
    capabilities: Any
    started_at: float
    heartbeat_at: float
    # Whether this session's own port answered it, and when it last asked.
    # Nothing when it has not asked yet.
    transport_ok: bool | None = None
    transport_checked_at: float | None = None
    # How an ended session ended: `gone` when it ended its own row, `crashed`
    # when its process was found missing. Nothing while it is running.
    ended_as: str | None = None
    # The name a session had before it took its scene's. Held for it while it
    # runs, so a caller still using that name never reaches another session.
    previous_alias: str | None = None

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> SessionRecord:
        return cls(
            session_id=row["session_id"],
            alias=row["alias"],
            kind=row["kind"],
            pid=row["pid"],
            pid_start=row["pid_start"],
            port=row["port"],
            state=row["state"],
            scene_epoch=row["scene_epoch"],
            hip_path=row["hip_path"],
            capabilities=_load(row["capabilities"]),
            started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"],
            transport_ok=_flag(row["transport_ok"]),
            transport_checked_at=row["transport_checked_at"],
            ended_as=row["ended_as"],
            previous_alias=row["previous_alias"],
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
    pid: int | None = None
    pid_start: str | None = None
    capabilities: Any = None
    weight: float = 1.0
    lessee_pid: int | None = None
    lessee_start: str | None = None

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
            pid=row["pid"],
            pid_start=row["pid_start"],
            capabilities=_load(row["capabilities"]),
            weight=float(row["weight"]),
            lessee_pid=row["lessee_pid"],
            lessee_start=row["lessee_start"],
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
    # The operation the job runs under, what it runs, and when it began to.
    operation_id: str | None = None
    spec: Any = None
    started_at: float | None = None
    # Where the readable copy and a spilled answer were written, when they were.
    export_path: str | None = None
    spill_path: str | None = None
    # Whether a caller was handed the job to follow, rather than its answer.
    promoted: bool = False
    # Where the row sits in the table, for a list that goes on from a row.
    seq: int | None = None

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> JobRecord:
        keys = row.keys()
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
            operation_id=row["operation_id"],
            spec=_load(row["spec"]),
            started_at=row["started_at"],
            export_path=row["export_path"],
            spill_path=row["spill_path"],
            promoted=bool(row["promoted"]),
            seq=row["seq"] if "seq" in keys else None,
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
    # Where the row sits in the table, for a read that pages through runs made
    # in the same instant. Nothing when the read did not ask for it.
    seq: int | None = None

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
            seq=row["seq"] if "seq" in row.keys() else None,
        )


FROZEN_PREPARED = "prepared"
FROZEN_ACTIVE = "active"


@dataclass(frozen=True)
class FrozenParm:
    """An output parameter a run set to its own path, owed its own value back.

    `frozen` is the path the run wrote on the node. `original` is what the
    parameter held before, as text, or `original_expression` and
    `original_language` when it held an expression, and that is what goes
    back when the run is over. `template` is the run's own line with its
    Houdini variables, kept for a reader. `node_sid` is the node's session
    id, which follows the node through a rename in the session that froze
    it. `hip_key` is the scene the parameter was frozen in, for the session
    that opens the scene next when the one that froze it has gone.
    """

    session_id: str
    node_path: str
    parm_name: str
    template: str
    frozen: str
    run_id: str | None
    hip_key: str | None
    created_at: float
    node_sid: int | None = None
    original: str | None = None
    original_expression: str | None = None
    original_language: str | None = None
    # Whose freeze this is: only a call holding the token may give it back.
    token: str | None = None
    # `prepared` from the moment it is written until the run's path is on the
    # node, `active` after. A prepared record is one whose run never took
    # hold, so its parameter gets its own value back whatever it holds.
    state: str = FROZEN_PREPARED

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> FrozenParm:
        return cls(
            session_id=row["session_id"],
            node_path=row["node_path"],
            parm_name=row["parm_name"],
            template=row["template"],
            frozen=row["frozen"],
            run_id=row["run_id"],
            hip_key=row["hip_key"],
            created_at=row["created_at"],
            node_sid=row["node_sid"],
            original=row["original"],
            original_expression=row["original_expression"],
            original_language=row["original_language"],
            token=row["token"],
            state=row["state"],
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

# A pid on its own is not an identity, so a session records when its process
# started as well. A row whose pid is alive but started at another moment
# belongs to a process that took the number over.
_SCHEMA_3 = ("ALTER TABLE sessions ADD COLUMN pid_start TEXT",)

# What a worker turned into once it was running: the process it is, what that
# process can do, and how much of the pool budget its work takes.
_SCHEMA_4 = (
    "ALTER TABLE workers ADD COLUMN pid INTEGER",
    "ALTER TABLE workers ADD COLUMN pid_start TEXT",
    "ALTER TABLE workers ADD COLUMN capabilities TEXT",
    "ALTER TABLE workers ADD COLUMN weight REAL NOT NULL DEFAULT 1",
)

# Who took a worker for a job. The worker owns its slot, so without this a
# server that died while holding a lease would hold it for ever: the slot
# looks busy to the pool and the row is never idle.
_SCHEMA_5 = (
    "ALTER TABLE workers ADD COLUMN lessee_pid INTEGER",
    "ALTER TABLE workers ADD COLUMN lessee_start TEXT",
)

# Whether a session's own port answered it the last time it asked, and when.
# A session can be alive and writing heartbeats while nothing can reach it, so
# a heartbeat on its own says only that the process is running.
_SCHEMA_6 = (
    "ALTER TABLE sessions ADD COLUMN transport_ok INTEGER",
    "ALTER TABLE sessions ADD COLUMN transport_checked_at REAL",
)

# How a session ended, written when it is marked ended. The signs it could be
# read from later, the process and the session file, are gone by then or are
# cleared by the next reader, so it is kept at the one moment it is known.
_SCHEMA_7 = ("ALTER TABLE sessions ADD COLUMN ended_as TEXT",)

# What a job runs and under which operation, so a job can be followed from
# the call that started it, and when it began running as opposed to when it
# was accepted.
_SCHEMA_8 = (
    "ALTER TABLE jobs ADD COLUMN operation_id TEXT",
    "ALTER TABLE jobs ADD COLUMN spec TEXT",
    "ALTER TABLE jobs ADD COLUMN started_at REAL",
    "CREATE INDEX jobs_by_session ON jobs(session_id, state)",
)

# Where a job's readable copy and spilled answer went, whether it was handed
# out to follow, the indexes the lists and the retention read by, and the
# claim several processes take turns on for upkeep.
_SCHEMA_9 = (
    "ALTER TABLE jobs ADD COLUMN export_path TEXT",
    "ALTER TABLE jobs ADD COLUMN spill_path TEXT",
    "ALTER TABLE jobs ADD COLUMN promoted INTEGER NOT NULL DEFAULT 0",
    "DROP INDEX IF EXISTS jobs_by_session",
    "CREATE INDEX jobs_by_created ON jobs(created_at)",
    "CREATE INDEX jobs_by_state_created ON jobs(state, created_at)",
    "CREATE INDEX jobs_by_session_state_created ON jobs(session_id, state, created_at)",
    "CREATE INDEX jobs_by_state_finished ON jobs(state, finished_at)",
    """
    CREATE TABLE sweeps (
        name     TEXT PRIMARY KEY,
        holder   TEXT,
        taken_at REAL NOT NULL
    )
    """,
)

# Output parameters a run has set to its own path, each owed its own value back
# when the run is over, and the index a scene's list of its runs reads by.
_FROZEN_PARMS_TABLE = (
    """
    CREATE TABLE frozen_parms (
        session_id TEXT NOT NULL,
        node_path  TEXT NOT NULL,
        parm_name  TEXT NOT NULL,
        template   TEXT NOT NULL,
        frozen     TEXT NOT NULL,
        run_id     TEXT,
        hip_key    TEXT,
        created_at REAL NOT NULL,
        node_sid   INTEGER,
        original   TEXT,
        original_expression TEXT,
        original_language   TEXT,
        token      TEXT,
        state      TEXT NOT NULL DEFAULT 'prepared',
        PRIMARY KEY (session_id, node_path, parm_name)
    )
    """,
    "CREATE INDEX frozen_parms_by_scene ON frozen_parms(hip_key)",
)

FROZEN_PARM_COLUMNS = frozenset(
    {
        "session_id",
        "node_path",
        "parm_name",
        "template",
        "frozen",
        "run_id",
        "hip_key",
        "created_at",
        "node_sid",
        "original",
        "original_expression",
        "original_language",
        "token",
        "state",
    }
)

_SCHEMA_10 = (
    *_FROZEN_PARMS_TABLE,
    "CREATE INDEX runs_by_family ON runs(hip_family, created_at)",
)

# What a node or a job wrote, found without reading every run.
_SCHEMA_11 = (
    "CREATE INDEX runs_by_node ON runs(source_node, created_at)",
    "CREATE INDEX runs_by_job ON runs(job_id)",
)

# The name a renamed session had, held for it while it runs.
_SCHEMA_12 = ("ALTER TABLE sessions ADD COLUMN previous_alias TEXT",)

MIGRATIONS = (
    _SCHEMA_1,
    _SCHEMA_2,
    _SCHEMA_3,
    _SCHEMA_4,
    _SCHEMA_5,
    _SCHEMA_6,
    _SCHEMA_7,
    _SCHEMA_8,
    _SCHEMA_9,
    _SCHEMA_10,
    _SCHEMA_11,
    _SCHEMA_12,
)


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
            self._retry_while_busy(self._settle_frozen_parms)
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
                if isinstance(error, sqlite3.OperationalError) and not (
                    _is_busy(error)
                    or (
                        sys.platform == "win32"
                        and getattr(error, "sqlite_errorcode", None)
                        == sqlite3.SQLITE_IOERR_TRUNCATE
                    )
                ):
                    raise _translate(error) from error
                if time.monotonic() >= deadline:
                    raise StoreBusy(f"{self.path} stayed locked by other processes") from error
                time.sleep(delay)
                delay = min(delay * 2, 0.1)

    def _apply_migrations(self) -> None:
        # A file that is already current is only read, so opening the store
        # takes no write lock and a reader is never held up by a writer.
        current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if current == len(MIGRATIONS):
            return
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

    def _settle_frozen_parms(self) -> None:
        """Make the frozen parameter table again when it is short of columns.

        A store opened by a build from before the table had its last shape
        has the table without them. It holds records that last as long as a
        run, so it is made again rather than moved forward.
        """
        # Read first: a store open must not wait on another process's write
        # lock when the table already has its shape, which is nearly always.
        with self._txn(write=False) as db:
            have = {row["name"] for row in db.execute("PRAGMA table_info(frozen_parms)")}
        if not have or FROZEN_PARM_COLUMNS <= have:
            return
        with self._txn(write=True) as db:
            have = {row["name"] for row in db.execute("PRAGMA table_info(frozen_parms)")}
            if not have or FROZEN_PARM_COLUMNS <= have:
                return
            db.execute("DROP TABLE frozen_parms")
            for statement in _FROZEN_PARMS_TABLE:
                db.execute(statement)

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
        pid_start: str | None = None,
        alias: str | None = None,
        alias_template: str | None = None,
        port: int | None = None,
        hip_path: str | None = None,
        scene_epoch: int = 0,
        capabilities: Any = None,
        state: str = SESSION_LIVE,
    ) -> SessionRecord:
        """Add a session row and settle its alias in the same transaction.

        Pass `alias` for a name the caller already owns, or `alias_template`
        containing `{n}` (`"scene-{n}"`, `"w{n}"`) to take the lowest free one.
        A session id is never reused. An alias is free again once its session
        is gone.

        `pid_start` is when the process started, from `process_start_stamp`.
        With it a row whose pid has been handed to something else is told from
        one whose process is still there, so a name is freed when its session
        really has gone and held when it has not.
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
                name = _first_free_alias(alias_template, _names_held(db))
            else:
                name = alias
                row = db.execute(
                    "SELECT session_id FROM sessions"
                    " WHERE (alias = ? OR previous_alias = ?) AND state <> ?",
                    (name, name, SESSION_GONE),
                ).fetchone()
                if row is not None:
                    raise AliasInUse(f"alias {name} belongs to session {row['session_id']}")
            db.execute(
                "INSERT INTO sessions (session_id, alias, kind, pid, pid_start, port, state,"
                " scene_epoch, hip_path, capabilities, started_at, heartbeat_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    name,
                    kind,
                    pid,
                    pid_start,
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

    def rename_session(
        self, session_id: str, *, alias_template: str, hip_path: str | None = None
    ) -> SessionRecord:
        """Give a live session the lowest free name the template makes.

        For a session that came up before its scene did, and takes the scene's
        name once it arrives, along with the scene's file. The name it had is
        kept as its previous name and held for it until it ends, so no other
        session is handed a name a caller may still be using for this one.
        """
        now = self._now()
        with self._txn(write=True) as db:
            self._reclaim_sessions(db, now)
            row = db.execute(
                "SELECT alias FROM sessions WHERE session_id = ? AND state <> ?",
                (session_id, SESSION_GONE),
            ).fetchone()
            if row is None:
                raise UnknownRecord(f"no live session {session_id}")
            name = _first_free_alias(alias_template, _names_held(db, but=session_id))
            previous = None if name == row["alias"] else row["alias"]
            db.execute(
                "UPDATE sessions SET alias = ?, previous_alias = COALESCE(?, previous_alias),"
                " hip_path = COALESCE(?, hip_path) WHERE session_id = ?",
                (name, previous, hip_path, session_id),
            )
            return SessionRecord._from_row(
                db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Session by id, gone or not. Ids are never reused."""
        row = self._read_one("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return None if row is None else SessionRecord._from_row(row)

    def resolve_session(self, handle: str) -> SessionRecord | None:
        """Session by id, or by the alias of a session that is not gone.

        A name a session had before it was renamed still finds it while it
        runs, since nobody else can be given that name meanwhile.
        """
        found = self.get_session(handle)
        if found is not None:
            return found
        row = self._read_one(
            "SELECT * FROM sessions WHERE alias = ? AND state <> ?"
            " ORDER BY started_at DESC, rowid DESC",
            (handle, SESSION_GONE),
        ) or self._read_one(
            "SELECT * FROM sessions WHERE previous_alias = ? AND state <> ?"
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

    def touch_session(
        self,
        session_id: str,
        *,
        state: str | None = None,
        transport_ok: bool | None = None,
        transport_checked_at: float | None = None,
    ) -> float:
        """Write a heartbeat, and a new state when one is given.

        A session that has checked its own port passes what it found. A
        heartbeat on its own only says the process is running, so a session
        whose port stopped answering writes `unresponsive` here and a server
        reading the row sees that rather than a live session.
        """
        if state is not None and state not in SESSION_STATES:
            raise ValueError(f"unknown session state: {state}")
        now = self._now()
        sets = ["heartbeat_at = ?"]
        values: list[Any] = [now]
        if state is not None:
            sets.append("state = ?")
            values.append(state)
        if transport_ok is not None:
            sets.append("transport_ok = ?")
            values.append(1 if transport_ok else 0)
            sets.append("transport_checked_at = ?")
            values.append(now if transport_checked_at is None else transport_checked_at)
        values.append(session_id)
        values.append(SESSION_GONE)
        with self._txn(write=True) as db:
            # A session that has ended stays ended. A beat written by a thread
            # that had not noticed the session was going would otherwise bring
            # the row back with nothing behind it.
            written = db.execute(
                f"UPDATE sessions SET {', '.join(sets)} WHERE session_id = ? AND state <> ?",
                tuple(values),
            )
            if written.rowcount == 0:
                raise UnknownRecord(f"no live session {session_id}")
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
        """Mark sessions gone whose process is not there any more, as crashed.

        A session that crashed cannot end its own row, and its alias would
        otherwise stay taken for good. `register_session` does this for itself,
        so this is for a caller that only wants to tidy up or to read an
        honest list afterwards. Returns the ids that were marked.
        """
        with self._txn(write=True) as db:
            return self._reclaim_sessions(db, self._now())

    def _reclaim_sessions(self, db: sqlite3.Connection, now: float) -> list[str]:
        rows = db.execute(
            "SELECT session_id, pid, pid_start FROM sessions WHERE state <> ?", (SESSION_GONE,)
        ).fetchall()
        reclaimed: list[str] = []
        for row in rows:
            # `None` means this system would not say which process a pid is,
            # and a session is never taken from a caller on a guess.
            if same_process(row["pid"], row["pid_start"]) is not False:
                continue
            db.execute(
                "UPDATE sessions SET state = ?, ended_as = ?, heartbeat_at = ?"
                " WHERE session_id = ?",
                (SESSION_GONE, SESSION_CRASHED, now, row["session_id"]),
            )
            self._lose_session_jobs(db, row["session_id"], now)
            reclaimed.append(row["session_id"])
        return reclaimed

    def set_scene_epoch(self, session_id: str, epoch: int, *, hip_path: str | None = None) -> int:
        """Write the epoch a session says it is on, and the scene it is in.

        The session that owns the scene owns the count, so a bridge writes the
        number it holds rather than asking this to add one: what happened in
        Houdini decides, not how many times the row was touched.
        """
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE sessions SET scene_epoch = ?, hip_path = ? WHERE session_id = ?",
                (epoch, hip_path, session_id),
            )
            if written.rowcount == 0:
                raise UnknownRecord(f"no session {session_id}")
        return epoch

    def end_session(self, session_id: str, *, how: str = SESSION_GONE) -> None:
        """Mark a session gone, which frees its alias for a later process.

        `how` says how it ended. A session that ends its own row, or one that
        was stopped on purpose, ended `gone`, even when a reader had already
        found its process missing and called it `crashed`.
        """
        if how not in SESSION_ENDINGS:
            raise ValueError(f"unknown way to end: {how}")
        now = self._now()
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE sessions SET state = ?, ended_as = ?, heartbeat_at = ?"
                " WHERE session_id = ?",
                (SESSION_GONE, how, now, session_id),
            )
            if written.rowcount == 0:
                raise UnknownRecord(f"no session {session_id}")
            self._lose_session_jobs(db, session_id, now)

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
        weight: float = 1.0,
        weight_budget: float | None = None,
    ) -> WorkerRecord:
        """Take a slot under the pool cap, or raise `PoolFull`.

        Slots held by processes that are gone are reclaimed first, in the same
        transaction, so a crash does not shrink the pool for good. Counting and
        inserting are that same transaction, so two processes that see the same
        count cannot both get the last slot. A reservation counts from here,
        before hython starts, and stops counting once the worker ends up failed
        or stopped.

        `weight` says how much of the machine this reservation means to use.
        With a `weight_budget` the weights already held are added up too, so a
        heavy job is refused while lighter ones have the machine, although a
        slot under the cap is free.
        """
        if cap < 1:
            raise ValueError("cap must be at least 1")
        if weight <= 0:
            raise ValueError("weight must be more than zero")
        now = self._now()
        pid = os.getpid() if owner_pid is None else owner_pid
        placeholders = ", ".join("?" * len(WORKER_ACTIVE_STATES))
        with self._txn(write=True) as db:
            self._reclaim_workers(db, now)
            rows = db.execute(
                f"SELECT alias, weight FROM workers WHERE state IN ({placeholders})",
                WORKER_ACTIVE_STATES,
            ).fetchall()
            if len(rows) >= cap:
                raise PoolFull(f"{len(rows)} of {cap} worker slots are in use")
            if weight_budget is not None:
                held = sum(float(row["weight"]) for row in rows)
                if held + weight > weight_budget:
                    raise PoolFull(
                        f"a weight of {weight:g} does not fit beside {held:g}"
                        f" under a budget of {weight_budget:g}"
                    )
            alias = _first_free_alias(alias_template, {row["alias"] for row in rows})
            db.execute(
                "INSERT INTO workers (token, alias, state, session_id, job_id, owner_pid,"
                " start_deadline, reserved_at, leased_at, weight)"
                " VALUES (?, ?, 'reserved', NULL, ?, ?, ?, ?, ?, ?)",
                (token, alias, job_id, pid, now + start_budget_s, now, now, float(weight)),
            )
            return WorkerRecord._from_row(
                db.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            )

    def reclaim_workers(self) -> list[str]:
        """Fail slots whose owner is gone or that never finished starting.

        A worker that is itself fine but was taken for a job by a server that
        has since died is handed back to the pool instead, warm and free.

        Returns the tokens whose slot was freed, which does not include those
        handed back. `reserve_worker` does this for itself, so this is for a
        caller that only wants to tidy up or report.
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
            # A worker that came up records the process it is. That process
            # going away frees the slot whoever started it is still running.
            worker_gone = (
                row["pid"] is not None and same_process(row["pid"], row["pid_start"]) is False
            )
            deadline = row["start_deadline"]
            never_started = (
                row["state"] in WORKER_STARTING_STATES and deadline is not None and now > deadline
            )
            if owner_gone or worker_gone or never_started:
                db.execute(
                    "UPDATE workers SET state = 'failed', job_id = NULL, lessee_pid = NULL,"
                    " lessee_start = NULL, leased_at = ? WHERE token = ?",
                    (now, row["token"]),
                )
                reclaimed.append(row["token"])
                continue
            self._reclaim_lease(db, row, now)
        return reclaimed

    def _reclaim_lease(self, db: sqlite3.Connection, row: sqlite3.Row, now: float) -> None:
        """Hand a worker back when the server that took it is not there.

        The worker itself is fine and stays in the pool. What is dropped is
        the claim on it, which nobody is coming back for. A lease with no
        recorded holder is left alone: it was taken by something this cannot
        ask about, and taking work off a caller on a guess is worse.
        """
        if row["job_id"] is None or row["lessee_pid"] is None:
            return
        if same_process(row["lessee_pid"], row["lessee_start"]) is not False:
            return
        db.execute(
            "UPDATE workers SET state = 'running', job_id = NULL, lessee_pid = NULL,"
            " lessee_start = NULL, leased_at = ? WHERE token = ?",
            (now, row["token"]),
        )

    def set_worker_state(
        self,
        token: str,
        state: str,
        *,
        session_id: str | None | _Clear = None,
        job_id: str | None | _Clear = None,
        owner_pid: int | None = None,
        start_budget_s: float | None = None,
        pid: int | None = None,
        pid_start: str | None = None,
        capabilities: Any = None,
        weight: float | None = None,
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
            settled_job = _settle(job_id, row["job_id"])
            # No job means no lessee: whoever held it has let it go.
            lessee = (row["lessee_pid"], row["lessee_start"]) if settled_job else (None, None)
            db.execute(
                "UPDATE workers SET state = ?, session_id = ?, job_id = ?, owner_pid = ?,"
                " start_deadline = ?, leased_at = ?, pid = ?, pid_start = ?, capabilities = ?,"
                " weight = ?, lessee_pid = ?, lessee_start = ? WHERE token = ?",
                (
                    state,
                    _settle(session_id, row["session_id"]),
                    settled_job,
                    row["owner_pid"] if owner_pid is None else owner_pid,
                    deadline,
                    now,
                    row["pid"] if pid is None else pid,
                    row["pid_start"] if pid_start is None else pid_start,
                    row["capabilities"] if capabilities is None else _dump(capabilities),
                    row["weight"] if weight is None else float(weight),
                    lessee[0],
                    lessee[1],
                    token,
                ),
            )
            return WorkerRecord._from_row(
                db.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            )

    def lease_worker(
        self,
        token: str,
        *,
        job_id: str,
        lessee_pid: int | None = None,
        lessee_start: str | None = None,
    ) -> WorkerRecord:
        """Take a warm worker for one job, in one write, or `WorkerTaken`.

        Reading who holds a worker and then writing that it is yours is two
        decisions, and two servers can both make the first one. So the guard
        is the write: it only lands on a worker that no other job holds, and a
        write that lands on nothing means somebody else got there first.

        The pid recorded here is the server that took the worker, not the
        worker itself. A server that dies mid job would otherwise hold the
        lease for good, because the worker's own process is still alive.
        """
        now = self._now()
        pid = os.getpid() if lessee_pid is None else lessee_pid
        stamp = process_start_stamp(pid) if lessee_start is None else lessee_start
        placeholders = ", ".join("?" * len(WORKER_LEASABLE_STATES))
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE workers SET state = 'leased', job_id = ?, lessee_pid = ?,"
                " lessee_start = ?, leased_at = ? WHERE token = ?"
                f" AND state IN ({placeholders})"
                " AND (job_id IS NULL OR job_id = ?)",
                (job_id, pid, stamp, now, token, *WORKER_LEASABLE_STATES, job_id),
            )
            row = db.execute("SELECT * FROM workers WHERE token = ?", (token,)).fetchone()
            if row is None:
                raise UnknownRecord(f"no worker reservation {token}")
            if written.rowcount == 0:
                raise WorkerTaken(
                    f"worker {row['alias']} is {row['state']} and holds job {row['job_id']}"
                )
            return WorkerRecord._from_row(row)

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

    def hold_worker_for_job(self, session_id: str, job_id: str) -> bool:
        """Put a job a session runs on its worker row, so the worker is not idle.

        A worker a server already took for a job of its own keeps that job.
        A session that is not a worker has no row, and nothing changes.
        Returns whether the job was put on the row.
        """
        now = self._now()
        states = ", ".join("?" * len(WORKER_ACTIVE_STATES))
        with self._txn(write=True) as db:
            written = db.execute(
                f"UPDATE workers SET job_id = ?, leased_at = ? WHERE session_id = ?"
                f" AND state IN ({states}) AND (job_id IS NULL OR job_id = ?)",
                (job_id, now, session_id, *WORKER_ACTIVE_STATES, job_id),
            )
            return written.rowcount > 0

    def renew_worker_of_session(self, session_id: str) -> bool:
        """Renew the idle lease of the worker a session is, while it works."""
        now = self._now()
        states = ", ".join("?" * len(WORKER_ACTIVE_STATES))
        with self._txn(write=True) as db:
            written = db.execute(
                f"UPDATE workers SET leased_at = ? WHERE session_id = ? AND state IN ({states})",
                (now, session_id, *WORKER_ACTIVE_STATES),
            )
            return written.rowcount > 0

    def free_worker_of_job(self, session_id: str, job_id: str) -> bool:
        """Take a finished job off its worker row. The idle wait starts again now."""
        now = self._now()
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE workers SET job_id = NULL, leased_at = ?"
                " WHERE session_id = ? AND job_id = ?",
                (now, session_id, job_id),
            )
            return written.rowcount > 0

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
        with self._txn(write=True) as db:
            return self._finish_operation(
                db,
                self._now(),
                operation_id=operation_id,
                state=state,
                outcome=outcome,
                error=error,
                job_id=job_id,
            )

    @staticmethod
    def _finish_operation(
        db: sqlite3.Connection,
        now: float,
        *,
        operation_id: str,
        state: str = "done",
        outcome: Any = None,
        error: Any = None,
        job_id: str | None = None,
    ) -> OperationRecord:
        if state not in OPERATION_STATES:
            raise ValueError(f"unknown operation state: {state}")
        written = db.execute(
            "UPDATE operations SET state = ?, outcome = ?, error = ?, job_id = ?,"
            " updated_at = ? WHERE operation_id = ?",
            (state, _dump(outcome), _dump(error), job_id, now, operation_id),
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

    def drop_operation(self, operation_id: str) -> bool:
        """Take one receipt off, for work that turned out never to run.

        A receipt claimed by an attempt that was refused before the tool was
        reached would otherwise answer every later attempt with an outcome
        nobody knows, although nothing ever happened.
        """
        with self._txn(write=True) as db:
            written = db.execute("DELETE FROM operations WHERE operation_id = ?", (operation_id,))
            return written.rowcount > 0

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
        operation_id: str | None = None,
        spec: Any = None,
        replace_after_s: float | None = None,
    ) -> JobRecord:
        """Record an accepted job, including the scene identity it consumes.

        An id already in the table is refused with `JobIdTaken`, unless
        `replace_after_s` is given and the row there is a job that ended at
        least that long ago: then the id is taken again for a new run and the
        old row goes.
        """
        if state not in JOB_STATES:
            raise ValueError(f"unknown job state: {state}")
        now = self._now()
        started = now if state == "running" else None
        with self._txn(write=True) as db:
            old = db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if old is not None:
                done_long_ago = (
                    replace_after_s is not None
                    and old["state"] in JOB_FINAL_STATES
                    and old["finished_at"] is not None
                    and _age(now, old["finished_at"]) >= replace_after_s
                )
                if not done_long_ago:
                    raise JobIdTaken(f"job {job_id} is kept and is {old['state']}")
                db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            db.execute(
                "INSERT INTO jobs (job_id, session_id, kind, state, weight, progress, outputs,"
                " error, scene, cancel_requested, worker_pid, heartbeat_at, created_at,"
                " updated_at, finished_at, operation_id, spec, started_at)"
                " VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    now if state in JOB_FINAL_STATES else None,
                    operation_id,
                    _dump(spec),
                    started,
                ),
            )
            return self._job_row(db, job_id)

    def update_job(
        self,
        job_id: str,
        *,
        state: str | None = None,
        progress: Any = None,
        outputs: Any = None,
        error: Any = None,
        worker_pid: int | None = None,
        scene: Any = None,
        late: bool = False,
    ) -> JobRecord:
        """Write progress, outputs so far or a final state, under the job rules.

        A job moves only as `JOB_MOVES` allows, and anything else raises
        `JobMoveRefused`. A row that has ended takes no more writes: progress
        and the rest are left as they are, and so is when it ended. The one
        move out of an ending is a late finish, `late` set, from `lost` to
        how the work really ended: it writes the error it is given, or none,
        in the same step, so the reason it was lost does not stay behind.
        """
        if state is not None and state not in JOB_STATES:
            raise ValueError(f"unknown job state: {state}")
        with self._txn(write=True) as db:
            return self._update_job(
                db,
                job_id,
                self._now(),
                state=state,
                progress=progress,
                outputs=outputs,
                error=error,
                worker_pid=worker_pid,
                scene=scene,
                late=late,
            )

    def _update_job(
        self,
        db: sqlite3.Connection,
        job_id: str,
        now: float,
        *,
        state: str | None,
        progress: Any = None,
        outputs: Any = None,
        error: Any = None,
        worker_pid: int | None = None,
        scene: Any = None,
        late: bool = False,
    ) -> JobRecord:
        """`update_job` inside a transaction the caller holds."""
        row = db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise UnknownRecord(f"no job {job_id}")
        current = row["state"]
        moving = state is not None and state != current
        if not moving:
            if current in JOB_FINAL_STATES:
                return JobRecord._from_row(row)
            new_state, finished, new_error = current, row["finished_at"], _keep(error, row)
        elif state in JOB_MOVES.get(current, ()):
            new_state, new_error = state, _keep(error, row)
            finished = now if state in JOB_FINAL_STATES else None
        elif late and current == "lost" and state in LATE_FINISHES:
            # How the work really ended, over a loss that was only a guess.
            new_state, finished, new_error = state, now, _dump(error)
        else:
            raise JobMoveRefused(f"job {job_id} cannot move from {current} to {state}")
        started = row["started_at"]
        if started is None and new_state == "running":
            started = now
        db.execute(
            "UPDATE jobs SET state = ?, progress = ?, outputs = ?, error = ?, worker_pid = ?,"
            " scene = ?, heartbeat_at = ?, updated_at = ?, finished_at = ?, started_at = ?"
            " WHERE job_id = ?",
            (
                new_state,
                _dump(progress) if progress is not None else row["progress"],
                _dump(outputs) if outputs is not None else row["outputs"],
                new_error,
                row["worker_pid"] if worker_pid is None else worker_pid,
                _dump(scene) if scene is not None else row["scene"],
                now,
                now,
                finished,
                started,
                job_id,
            ),
        )
        return self._job_row(db, job_id)

    def start_job(self, job_id: str, *, scene: Any = None) -> JobRecord | None:
        """Move a queued job to running, and only a queued one.

        Nothing when the row is not queued: a job the session has already
        marked, or one found lost meanwhile, is never brought back to running.
        """
        now = self._now()
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE jobs SET state = 'running', started_at = COALESCE(started_at, ?),"
                " scene = COALESCE(?, scene), heartbeat_at = ?, updated_at = ?"
                " WHERE job_id = ? AND state = 'queued'",
                (now, _dump(scene), now, now, job_id),
            )
            return self._job_row(db, job_id) if written.rowcount else None

    def beat_job(
        self,
        job_id: str,
        *,
        progress: Any = None,
        worker_pid: int | None = None,
        repair: Mapping[str, Any] | None = None,
    ) -> JobRecord | None:
        """The heartbeat of a running job, with its latest progress.

        One write that also mends the row: a queued row moves to running, and
        a missing one is made again from `repair`, which holds what
        `create_job` would have been given. A row that has ended is left as
        it is. Nothing when there is no row and nothing to make it from.
        """
        now = self._now()
        with self._txn(write=True) as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                if repair is None:
                    return None
                self._make_job_again(db, job_id, now, repair, progress, worker_pid)
                return self._job_row(db, job_id)
            if row["state"] in JOB_FINAL_STATES:
                return JobRecord._from_row(row)
            db.execute(
                "UPDATE jobs SET state = 'running', started_at = COALESCE(started_at, ?),"
                " progress = COALESCE(?, progress), worker_pid = COALESCE(?, worker_pid),"
                " heartbeat_at = ?, updated_at = ? WHERE job_id = ?",
                (now, _dump(progress), worker_pid, now, now, job_id),
            )
            return self._job_row(db, job_id)

    @staticmethod
    def _make_job_again(
        db: sqlite3.Connection,
        job_id: str,
        now: float,
        repair: Mapping[str, Any],
        progress: Any = None,
        worker_pid: int | None = None,
    ) -> None:
        """Write a running job's row again from what its runner knows of it."""
        db.execute(
            "INSERT INTO jobs (job_id, session_id, kind, state, weight, progress,"
            " scene, cancel_requested, worker_pid, heartbeat_at, created_at, updated_at,"
            " operation_id, spec, started_at)"
            " VALUES (?, ?, ?, 'running', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                repair.get("session_id"),
                repair.get("kind") or "unknown",
                repair.get("weight") or "light",
                _dump(progress),
                _dump(repair.get("scene")),
                worker_pid,
                now,
                now,
                now,
                repair.get("operation_id"),
                _dump(repair.get("spec")),
                now,
            ),
        )

    def finish_job(
        self,
        job_id: str,
        *,
        state: str,
        progress: Any = None,
        outputs: Any = None,
        error: Any = None,
        repair: Mapping[str, Any] | None = None,
        operation: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        """Write how a job ended, and the receipt of its call with it, in one step.

        `operation` holds what `finish_operation` takes, with its id, so the
        receipt and the job can never disagree about whether the call ended.
        A missing row is made again from `repair` first, and a row found
        `lost` meanwhile takes the real ending as a late finish.
        """
        if state not in JOB_FINAL_STATES:
            raise ValueError(f"not a final job state: {state}")
        now = self._now()
        with self._txn(write=True) as db:
            if operation is not None:
                self._finish_operation(db, now, **operation)
            row = db.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                if repair is None:
                    raise UnknownRecord(f"no job {job_id}")
                self._make_job_again(db, job_id, now, repair)
                current = "running"
            else:
                current = row["state"]
            return self._update_job(
                db,
                job_id,
                now,
                state=state,
                progress=progress,
                outputs=outputs,
                error=error,
                late=current == "lost",
            )

    @staticmethod
    def _job_row(db: sqlite3.Connection, job_id: str) -> JobRecord:
        return JobRecord._from_row(
            db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        )

    def touch_job(self, job_id: str, *, worker_pid: int | None = None) -> float:
        """Heartbeat from whoever is running the job. A job that has ended is left alone."""
        now = self._now()
        with self._txn(write=True) as db:
            row = db.execute(
                "SELECT worker_pid, state FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise UnknownRecord(f"no job {job_id}")
            if row["state"] in JOB_FINAL_STATES:
                return now
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

    def lose_jobs(self, job_ids: Sequence[str], *, error: Any = None) -> list[JobRecord]:
        """Mark jobs `lost` that have not finished, keeping what they wrote so far.

        A job that finished in the meantime keeps its own ending. Returns the
        rows that were marked.
        """
        if not job_ids:
            return []
        now = self._now()
        marks = ", ".join("?" * len(job_ids))
        states = ", ".join("?" * len(JOB_LIVE_STATES))
        with self._txn(write=True) as db:
            rows = db.execute(
                f"SELECT job_id FROM jobs WHERE job_id IN ({marks}) AND state IN ({states})",
                (*job_ids, *JOB_LIVE_STATES),
            ).fetchall()
            lost = [row["job_id"] for row in rows]
            for job_id in lost:
                self._lose_job(db, job_id, now, error)
            return [
                JobRecord._from_row(
                    db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                )
                for job_id in lost
            ]

    def _lose_session_jobs(self, db: sqlite3.Connection, session_id: str, now: float) -> None:
        """Every unfinished job of a session that has ended is lost with it."""
        states = ", ".join("?" * len(JOB_LIVE_STATES))
        rows = db.execute(
            f"SELECT job_id FROM jobs WHERE session_id = ? AND state IN ({states})",
            (session_id, *JOB_LIVE_STATES),
        ).fetchall()
        for row in rows:
            self._lose_job(db, row["job_id"], now, SESSION_ENDED_ERROR)

    @staticmethod
    def _lose_job(db: sqlite3.Connection, job_id: str, now: float, error: Any) -> None:
        db.execute(
            "UPDATE jobs SET state = 'lost', error = COALESCE(?, error), updated_at = ?,"
            " finished_at = ? WHERE job_id = ?",
            (_dump(error), now, now, job_id),
        )

    def note_job_paths(
        self, job_id: str, *, export_path: str | None = None, spill_path: str | None = None
    ) -> None:
        """Say where a job's readable copy or spilled answer was written.

        The one write a job that has ended still takes: it changes nothing
        about the job, only where to find what it left.
        """
        with self._txn(write=True) as db:
            db.execute(
                "UPDATE jobs SET export_path = COALESCE(?, export_path),"
                " spill_path = COALESCE(?, spill_path) WHERE job_id = ?",
                (export_path, spill_path, job_id),
            )

    def promote_job(self, job_id: str) -> None:
        """Mark a job as one a caller was handed to follow."""
        with self._txn(write=True) as db:
            db.execute("UPDATE jobs SET promoted = 1 WHERE job_id = ?", (job_id,))

    def drop_job(self, job_id: str) -> bool:
        """Take a job row off, for work that was accepted and never ran."""
        with self._txn(write=True) as db:
            return db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,)).rowcount > 0

    def get_job(self, job_id: str) -> JobRecord | None:
        """One job by id."""
        row = self._read_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return None if row is None else JobRecord._from_row(row)

    def list_jobs(
        self,
        *,
        session_id: str | None = None,
        session_ids: Sequence[str] | None = None,
        states: Sequence[str] | None = None,
        limit: int = 50,
        before: tuple[float, int] | None = None,
    ) -> list[JobRecord]:
        """Recent jobs, newest first.

        `before` is the `created_at` and `seq` of the last row a caller has,
        and the list goes on from the row after it.
        """
        sql = "SELECT rowid AS seq, * FROM jobs"
        clauses: list[str] = []
        args: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            args.append(session_id)
        if session_ids is not None:
            clauses.append(f"session_id IN ({', '.join('?' * len(session_ids)) or 'NULL'})")
            args.extend(session_ids)
        if states:
            clauses.append(f"state IN ({', '.join('?' * len(states))})")
            args.extend(states)
        if before is not None:
            clauses.append("(created_at < ? OR (created_at = ? AND rowid < ?))")
            args.extend([before[0], before[0], before[1]])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        return [JobRecord._from_row(row) for row in self._read_all(sql, args)]

    def prune_jobs(self, max_age_s: float) -> int:
        """Drop jobs that ended longer ago than the retention window. Returns the count."""
        return len(self.prune_final_jobs(max_age_s))

    def prune_final_jobs(self, max_age_s: float) -> list[JobRecord]:
        """Drop jobs that ended longer ago than the retention window.

        Only jobs that have ended, by when they ended: a job that is still
        running is kept however long it runs. Returns the rows that went, so
        whatever was written beside them can go too.
        """
        cutoff = self._now() - max_age_s
        finals = ", ".join("?" * len(JOB_FINAL_STATES))
        with self._txn(write=True) as db:
            rows = db.execute(
                f"SELECT * FROM jobs WHERE state IN ({finals}) AND finished_at < ?",
                (*sorted(JOB_FINAL_STATES), cutoff),
            ).fetchall()
            for row in rows:
                db.execute("DELETE FROM jobs WHERE job_id = ?", (row["job_id"],))
            return [JobRecord._from_row(row) for row in rows]

    def take_sweep(self, name: str, every_s: float, *, holder: str | None = None) -> bool:
        """Claim one round of upkeep, at most once every `every_s` per store.

        Several processes share one store and each would sweep it, so the
        round is claimed here first: the claim is a row with when it was last
        taken, and only the process that moves it on does the round.
        """
        now = self._now()
        with self._txn(write=True) as db:
            row = db.execute("SELECT taken_at FROM sweeps WHERE name = ?", (name,)).fetchone()
            # A claim from the future, after the clock stepped back, is spent.
            if row is not None and 0.0 <= now - row["taken_at"] < every_s:
                return False
            db.execute(
                "INSERT OR REPLACE INTO sweeps (name, holder, taken_at) VALUES (?, ?, ?)",
                (name, holder or str(os.getpid()), now),
            )
            return True

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

    def skip_versions_to(self, *, kind: str, name: str, hip_family: str, version: int) -> int:
        """Make sure the next number handed out is above `version`.

        For a sequence that was started somewhere this store never saw, such as
        scene files saved by hand before the first save through here. The
        number is taken with no run on it, so it keeps its place and is never
        handed out. Returns the highest number taken, which may be higher
        already.
        """
        with self._txn(write=True) as db:
            row = db.execute(
                "SELECT MAX(version) AS top FROM versions"
                " WHERE kind = ? AND name = ? AND hip_family = ?",
                (kind, name, hip_family),
            ).fetchone()
            top = int(row["top"] or 0)
            if top >= version:
                return top
            db.execute(
                "INSERT INTO versions (kind, name, hip_family, version, run_id, created_at)"
                " VALUES (?, ?, ?, ?, NULL, ?)",
                (kind, name, hip_family, version, self._now()),
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

    def disown_version(self, *, kind: str, name: str, hip_family: str, version: int) -> bool:
        """Take a run's name off a number it did not get to use.

        The number stays taken. Whoever won the race owns the folder it points
        at, so handing the number out again would only fail in the same way.
        What is dropped is the claim that this run wrote it.
        """
        with self._txn(write=True) as db:
            written = db.execute(
                "UPDATE versions SET run_id = NULL WHERE kind = ? AND name = ? AND"
                " hip_family = ? AND version = ?",
                (kind, name, hip_family, version),
            )
            return written.rowcount > 0

    def reap_versions(self, max_age_s: float) -> int:
        """Clear the run id from old numbers whose run never arrived.

        A process that stops between taking a number and recording its run
        leaves a name pointing at nothing. After the grace period the number
        keeps its place in the sequence and loses the name. Returns the count.
        """
        cutoff = self._now() - max_age_s
        with self._txn(write=True) as db:
            return db.execute(
                "UPDATE versions SET run_id = NULL WHERE run_id IS NOT NULL AND created_at <= ?"
                " AND run_id NOT IN (SELECT run_id FROM runs)",
                (cutoff,),
            ).rowcount

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

    def drop_run(self, run_id: str) -> bool:
        """Take off the record of a run whose output was never written."""
        with self._txn(write=True) as db:
            return db.execute("DELETE FROM runs WHERE run_id = ?", (run_id,)).rowcount > 0

    def set_run_paths(self, run_id: str, paths: Any) -> bool:
        """Replace the paths a run records, once it is known what it wrote."""
        with self._txn(write=True) as db:
            return (
                db.execute(
                    "UPDATE runs SET paths = ? WHERE run_id = ?", (_dump(paths), run_id)
                ).rowcount
                > 0
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

    def runs_made_by(
        self,
        *,
        job_id: str | None = None,
        source_node: str | None = None,
        hip_family: str | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """The runs of one job, or of one node, newest first.

        `hip_family` narrows them to one scene family before the limit, since
        the same node path is in every scene.
        """
        if (job_id is None) == (source_node is None):
            raise ValueError("name a job or a node, not both")
        column, value = ("job_id", job_id) if job_id is not None else ("source_node", source_node)
        clauses, args = [f"{column} = ?"], [value]
        if hip_family is not None:
            clauses.append("hip_family = ?")
            args.append(hip_family)
        sql = (
            f"SELECT * FROM runs WHERE {' AND '.join(clauses)}"
            " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        )
        return [RunRecord._from_row(row) for row in self._read_all(sql, [*args, limit])]

    def find_runs(
        self,
        *,
        hip_family: str,
        kind: str | None = None,
        name_glob: str | None = None,
        since: float | None = None,
        session_id: str | None = None,
        before: tuple[float, int] | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """One scene family's runs, newest first, a batch at a time.

        `before` is the `created_at` and `seq` of the last run of the batch
        before, so the next batch starts after it whatever was added since,
        and two runs made in the same instant keep the order they were made
        in. `name_glob` matches the way a shell does, and case counts.
        """
        clauses = ["hip_family = ?"]
        args: list[Any] = [hip_family]
        if kind is not None:
            clauses.append("kind = ?")
            args.append(kind)
        if name_glob is not None:
            clauses.append("name GLOB ?")
            args.append(name_glob)
        if since is not None:
            clauses.append("created_at >= ?")
            args.append(since)
        if session_id is not None:
            clauses.append("session_id = ?")
            args.append(session_id)
        if before is not None:
            clauses.append("(created_at < ? OR (created_at = ? AND rowid < ?))")
            args.extend([before[0], before[0], before[1]])
        sql = (
            f"SELECT rowid AS seq, * FROM runs WHERE {' AND '.join(clauses)}"
            " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        )
        args.append(limit)
        return [RunRecord._from_row(row) for row in self._read_all(sql, args)]

    # -- frozen output parameters ------------------------------------------

    def freeze_parm(
        self,
        *,
        session_id: str,
        node_path: str,
        parm_name: str,
        template: str,
        frozen: str,
        run_id: str | None = None,
        hip_key: str | None = None,
        node_sid: int | None = None,
        original: str | None = None,
        original_expression: str | None = None,
        original_language: str | None = None,
        token: str | None = None,
    ) -> FrozenParm:
        """Record that a run is about to set a parameter to its own path.

        Written, `prepared`, before the parameter is touched, with what it
        holds now, so a process that dies part way leaves a record of what to
        put back rather than a machine path nobody knows of. A parameter
        already frozen under another token is `ParmHeld`, naming the run that
        holds it. The same token freezing it again, for a later run of the
        same call, takes the record over, and what it owes back stays the
        value from before the first: the path in between was never the
        parameter's own.
        """
        with self._txn(write=True) as db:
            held = db.execute(
                "SELECT token, run_id FROM frozen_parms WHERE session_id = ? AND node_path = ?"
                " AND parm_name = ?",
                (session_id, node_path, parm_name),
            ).fetchone()
            if held is not None and held["token"] != token:
                raise ParmHeld(
                    f"{node_path}/{parm_name} is held by run {held['run_id']}",
                    run_id=held["run_id"],
                )
            db.execute(
                "INSERT INTO frozen_parms (session_id, node_path, parm_name, template, frozen,"
                " run_id, hip_key, created_at, node_sid, original, original_expression,"
                " original_language, token, state)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (session_id, node_path, parm_name) DO UPDATE SET"
                " template = excluded.template, frozen = excluded.frozen,"
                " run_id = excluded.run_id, hip_key = excluded.hip_key,"
                " created_at = excluded.created_at, node_sid = excluded.node_sid,"
                " state = excluded.state",
                (
                    session_id,
                    node_path,
                    parm_name,
                    template,
                    frozen,
                    run_id,
                    hip_key,
                    self._now(),
                    node_sid,
                    original,
                    original_expression,
                    original_language,
                    token,
                    FROZEN_PREPARED,
                ),
            )
            return FrozenParm._from_row(
                db.execute(
                    "SELECT * FROM frozen_parms WHERE session_id = ? AND node_path = ?"
                    " AND parm_name = ?",
                    (session_id, node_path, parm_name),
                ).fetchone()
            )

    def get_frozen_parm(self, session_id: str, node_path: str, parm_name: str) -> FrozenParm | None:
        """One frozen parameter, by the session that froze it."""
        row = self._read_one(
            "SELECT * FROM frozen_parms WHERE session_id = ? AND node_path = ? AND parm_name = ?",
            (session_id, node_path, parm_name),
        )
        return None if row is None else FrozenParm._from_row(row)

    def list_frozen_parms(
        self, *, session_id: str | None = None, hip_key: str | None = None
    ) -> list[FrozenParm]:
        """Frozen parameters, by node path, for a session, a scene or both."""
        clauses: list[str] = []
        args: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            args.append(session_id)
        if hip_key is not None:
            clauses.append("hip_key = ?")
            args.append(hip_key)
        sql = "SELECT * FROM frozen_parms"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY node_path, parm_name, session_id"
        return [FrozenParm._from_row(row) for row in self._read_all(sql, args)]

    def activate_frozen_parm(
        self, session_id: str, node_path: str, parm_name: str, *, token: str | None
    ) -> bool:
        """Mark a record `active` once the run's path is on the parameter."""
        with self._txn(write=True) as db:
            return (
                db.execute(
                    "UPDATE frozen_parms SET state = ? WHERE session_id = ? AND node_path = ?"
                    " AND parm_name = ? AND token IS ?",
                    (FROZEN_ACTIVE, session_id, node_path, parm_name, token),
                ).rowcount
                > 0
            )

    def thaw_parm(
        self,
        session_id: str,
        node_path: str,
        parm_name: str,
        *,
        token: str | None = None,
        state: str | None = None,
        any_token: bool = False,
    ) -> bool:
        """Take off the record of a parameter that has had its own value back.

        Only the record under this token, and in this state when one is
        named, goes: a record another freeze has taken over since stays.
        `any_token` is for a record whose session has ended and was read a
        moment ago, and for a check.
        """
        clauses = ["session_id = ?", "node_path = ?", "parm_name = ?"]
        args: list[Any] = [session_id, node_path, parm_name]
        if not any_token:
            clauses.append("token IS ?")
            args.append(token)
        if state is not None:
            clauses.append("state = ?")
            args.append(state)
        with self._txn(write=True) as db:
            return (
                db.execute(f"DELETE FROM frozen_parms WHERE {' AND '.join(clauses)}", args).rowcount
                > 0
            )

    def session_is_over(self, session_id: str) -> bool:
        """Whether a session has ended, or its process is shown to be gone.

        A process this system will not name is taken to be running, so a
        record is never taken from a session on a guess.
        """
        record = self.get_session(session_id)
        if record is None or record.state == SESSION_GONE:
            return True
        return same_process(record.pid, record.pid_start) is False

    def orphan_frozen_parms(self, *, hip_key: str | None = None) -> list[FrozenParm]:
        """Frozen parameters whose session is over, so nobody is left to restore them.

        With `hip_key`, only those frozen in that scene: the next session to
        open it restores them.
        """
        rows = self.list_frozen_parms(hip_key=hip_key)
        over: dict[str, bool] = {}
        orphans = []
        for row in rows:
            if row.session_id not in over:
                over[row.session_id] = self.session_is_over(row.session_id)
            if over[row.session_id]:
                orphans.append(row)
        return orphans

    def forget_unrestorable_frozen_parms(self) -> int:
        """Drop the records of an ended session's parameters in a scene never saved.

        No session can open that scene again, so nothing will ever be owed
        back. Returns the count.
        """
        dropped = 0
        for row in self.orphan_frozen_parms():
            if row.hip_key is None and self.thaw_parm(
                row.session_id, row.node_path, row.parm_name, token=row.token
            ):
                dropped += 1
        return dropped

    # -- readable exports -------------------------------------------------

    def run_export(self, run_id: str) -> dict[str, Any]:
        """Run record as plain data, for the sidecar beside the scene file."""
        record = self.get_run(run_id)
        if record is None:
            raise UnknownRecord(f"no run {run_id}")
        exported = _export_dict(record, {"created_at": "created"})
        # Where the row sat in the table says nothing to a reader of the file.
        exported.pop("seq", None)
        return exported

    def job_export(self, job_id: str) -> dict[str, Any]:
        """Job record as plain data, for the readable copy of a finished job."""
        record = self.get_job(job_id)
        if record is None:
            raise UnknownRecord(f"no job {job_id}")
        exported = _export_dict(
            record,
            {
                "created_at": "created",
                "started_at": "started",
                "updated_at": "updated",
                "finished_at": "finished",
            },
        )
        # Where the row sat in the table says nothing to a reader of the file.
        exported.pop("seq", None)
        return exported


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


def _keep(error: Any, row: sqlite3.Row) -> Any:
    """A new error when one is given, otherwise the stored one."""
    return _dump(error) if error is not None else row["error"]


def _settle(given: Any, stored: Any) -> Any:
    """None keeps what is stored, `CLEAR` empties it, anything else replaces it."""
    if given is None:
        return stored
    if isinstance(given, _Clear):
        return None
    return given


def _names_held(db: sqlite3.Connection, *, but: str | None = None) -> set[str]:
    """Every name a session that has not ended answers to, now or from before a rename."""
    held: set[str] = set()
    for row in db.execute(
        "SELECT session_id, alias, previous_alias FROM sessions WHERE state <> ?",
        (SESSION_GONE,),
    ):
        if row["session_id"] == but:
            continue
        held.add(row["alias"])
        if row["previous_alias"]:
            held.add(row["previous_alias"])
    return held


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
