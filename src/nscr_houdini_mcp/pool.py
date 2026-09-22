"""The hython workers this machine may run, and the rules for taking one.

A worker is a Houdini of its own with a bridge in it. The pool decides how
many may exist, starts them, hands them to jobs, and lets the ones nobody
wants go away again.

Admission. Every slot is taken through one transaction in the coordination
store, which counts the workers that are still starting as well as the ones
that are up. Several server processes share one machine, so the decision
cannot be a count in memory. A reservation is held from before hython starts
until the work it was taken for has cleaned up, and a start that fails hands
its slot straight back.

Lifetime. A worker does not belong to the process that started it. A stdio
server dies with its client, and throwing the warm pool away on every client
restart would cost a cold start each time. So the worker is started detached,
it records itself as the owner of its own slot, and it watches its lease in
the store from a small thread of its own. Any server that routes a call to it
renews that lease. When nobody has wanted it for `max_idle_s`, it ends itself.
There is no daemon anywhere in this.

Weights. Three workers can each ask for every core, so a reservation carries a
weight and the pool holds a budget. A heavy job is refused while the machine
is already busy, even when a slot under the cap is free.

This module never imports `hou`. The worker it starts does.
"""

from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import app as bridge_app
from nscr_houdini_mcp.bridge import client, registry, security
from nscr_houdini_mcp.bridge.net import DEFAULT_PORT_RANGE
from nscr_houdini_mcp.store import (
    Store,
    WorkerRecord,
    process_is_alive,
    process_start_stamp,
    same_process,
)

# The path to hython, when the machine is not to be searched.
HYTHON_ENV_VAR = "NSCR_MCP_HYTHON"

# The module the worker runs. It is the same entry point a person can start by
# hand, with a token and an idle limit added.
WORKER_MODULE = "nscr_houdini_mcp.bridge.main"

WORKER_ALIAS_TEMPLATE = "w{n}"

# How many workers may exist beside whatever Houdini the artist has open. Four
# extra workers were seen to license on one machine and the ceiling was not
# found, so this stays low on purpose.
DEFAULT_CAP = 3

# How long a worker nobody has asked for stays warm. A cold start costs a few
# seconds, so keeping one for half an hour is cheap.
DEFAULT_MAX_IDLE_S = 30.0 * 60.0

DEFAULT_START_TIMEOUT_S = 180.0
DEFAULT_PROBE_TIMEOUT_S = 60.0
DEFAULT_STOP_GRACE_S = 30.0

# How long the store may stay unreadable before a worker stops watching its
# lease. Long enough that a busy machine, a backup or a lock held by another
# process is only a gap in the watch, not the end of it.
LEASE_TROUBLE_S = 600.0

# The longest a worker sleeps between looks at its own row. It decides how
# quickly a stop request is noticed, so it is short enough for a person
# waiting at a terminal and long enough to cost nothing over an hour.
MAX_LEASE_TICK_S = 5.0

# What the pool says when there is no room. It is one word on purpose: a
# caller reads it and decides whether to wait or to do the work in session.
POOL_FULL = "POOL_FULL"

# What a job means when it says how big it is. The budget defaults to the cap,
# so one heavy job fills a default pool by itself.
WEIGHTS = {"light": 1.0, "heavy": 3.0}
DEFAULT_WEIGHT = "light"

# The tool a fresh worker is asked about itself with.
CAPABILITIES_TOOL = "bridge.capabilities"

# The worker's own token. It goes in the environment rather than on the
# command line, because a command line is readable by every account on the
# machine and the token is what proves who may move that reservation on.
TOKEN_ENV_VAR = "NSCR_MCP_WORKER_TOKEN"

# Houdini's own thread control, which the pool decides rather than inherits.
THREADS_ENV_VAR = "HOUDINI_MAXTHREADS"

# How large one worker's log may get before it is rolled over, and how many
# rolled files are kept. A worker that runs for days and says something on
# every job must not fill the state folder.
LOG_MAX_BYTES = 8 * 1024 * 1024
LOG_KEEP = 2

# How long a process is given to go after it has been ended outright. The kill
# is not a request, so this is short.
KILL_WAIT_S = 5.0


class PoolError(Exception):
    """Base class for worker pool failures."""


class HythonNotFound(PoolError):
    """No hython on this machine, or none where it was said to be."""


class WorkerStartFailed(PoolError):
    """The process started but no bridge announced itself."""


class UnknownWorker(PoolError):
    """No live worker answers to that name."""


class WorkerBusy(PoolError):
    """That worker is already on another job."""


@dataclass(frozen=True)
class PoolConfig:
    """Everything the pool needs to know before it starts a worker."""

    home: Path
    cap: int = DEFAULT_CAP
    max_idle_s: float = DEFAULT_MAX_IDLE_S
    weight_budget: float | None = None
    # Passed on as `HOUDINI_MAXTHREADS`, so one worker cannot take the machine.
    max_threads: int | None = None
    hython: Path | str | None = None
    port_range: tuple[int, int] = DEFAULT_PORT_RANGE
    start_timeout_s: float = DEFAULT_START_TIMEOUT_S
    # Whether the worker carries the tool that exists to be driven in tests.
    selfcheck: bool = False

    @property
    def budget(self) -> float:
        """How much weight the whole pool may hold at once."""
        return float(self.cap) if self.weight_budget is None else float(self.weight_budget)


def _still_running() -> int | None:
    """The exit code of a process nothing is watching: never known here."""
    return None


@dataclass(frozen=True)
class Launched:
    """A process that was started, and how to ask whether it is still there."""

    pid: int
    poll: Callable[[], int | None] = _still_running
    # Ends the process through the handle that started it, which is the one
    # way to be sure of reaching that process and no other. Nothing when the
    # starter holds no handle.
    kill: Callable[[], None] | None = None
    # Waits for that same process to end, for up to the seconds given.
    wait: Callable[[float], Any] | None = None


@dataclass(frozen=True)
class Stopped:
    """What became of a worker that was asked to stop."""

    record: WorkerRecord
    killed: bool
    ended: bool
    # Why it was not ended, when it was not. Empty when there is nothing to say.
    note: str = ""


# Section: where things live


def store_path(home: Path | str) -> Path:
    """The coordination store under a state folder."""
    return Path(home) / store_module.STORE_FILE_NAME


def open_store(home: Path | str, **rest: Any) -> Store:
    """A store handle on the state folder the pool was given."""
    return Store(store_path(home), **rest)


def log_path(home: Path | str, alias: str) -> Path:
    """Where one worker's output is kept, named so a person can find it."""
    return Path(home) / bridge_app.LOG_DIR_NAME / f"worker-{alias}.log"


def weight_of(value: str | float | None) -> float:
    """A weight from a name or a number. Unknown names are an error."""
    if value is None:
        return WEIGHTS[DEFAULT_WEIGHT]
    if isinstance(value, str):
        try:
            return WEIGHTS[value]
        except KeyError:
            raise ValueError(f"unknown weight: {value}") from None
    return float(value)


def _executable(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def hython_path(configured: Path | str | None = None) -> Path:
    """The hython to start workers with, or `HythonNotFound`.

    A configured path, or the environment override, is taken as given: a wrong
    one is an error rather than a reason to go looking somewhere else. After
    that the installs this machine has are tried, newest first.
    """
    named = configured or os.environ.get(HYTHON_ENV_VAR)
    if named:
        path = Path(named).expanduser()
        if not path.is_file():
            raise HythonNotFound(f"{path} is not a file")
        return path
    for found in install_module.find_installs():
        candidate = found.hfs / "bin" / _executable("hython")
        if candidate.is_file():
            return candidate
    raise HythonNotFound("no hython found, name one in config")


# Section: starting a worker


def worker_command(config: PoolConfig, *, alias: str, hython: Path) -> list[str]:
    """The command line one worker is started with.

    The token is not here. Anyone with an account on this machine can read
    another account's command lines, and the token is what proves who owns the
    reservation, so it travels in the environment instead.
    """
    return [
        str(hython),
        "-m",
        WORKER_MODULE,
        "--home",
        str(config.home),
        "--port",
        str(config.port_range[0]),
        "--max-port",
        str(config.port_range[1]),
        "--alias",
        alias,
        "--max-idle-s",
        str(config.max_idle_s),
    ]


def worker_env(
    config: PoolConfig,
    *,
    token: str | None = None,
    weight: float | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment a worker is started in.

    The rest of this process's environment is passed on deliberately: a worker
    has to see the same licensing, path and package settings as the shell the
    tool was started from, or it is a different Houdini from the artist's. The
    few things the pool decides are decided here, and the thread cap is one of
    them rather than whatever happened to be inherited.
    """
    environment = dict(os.environ if base is None else base)
    # The package has to be importable inside Houdini's own interpreter.
    source_root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else os.pathsep.join([source_root, existing])
    )
    environment[store_module.HOME_ENV_VAR] = str(config.home)
    threads = thread_cap(config, weight)
    if threads is None:
        # Houdini's own default, whatever this shell was carrying.
        environment.pop(THREADS_ENV_VAR, None)
    else:
        environment[THREADS_ENV_VAR] = str(threads)
    if token is not None:
        environment[TOKEN_ENV_VAR] = token
    if config.selfcheck:
        environment[bridge_app.SELFCHECK_ENV_VAR] = "1"
    return environment


def thread_cap(config: PoolConfig, weight: float | None = None) -> int | None:
    """How many threads one worker may use, or nothing for Houdini's default.

    A configured cap is taken as given. Otherwise the weight decides: a heavy
    worker is the one job that is meant to have the machine, and a light one
    is left at the default rather than at whatever the shell was carrying.
    """
    if config.max_threads is not None:
        return config.max_threads
    if weight is not None and weight >= WEIGHTS["heavy"]:
        return os.cpu_count() or 1
    return None


# The processes this one started. A worker outlives whoever started it, but
# while that starter is still running the worker stays its child, and a child
# that has ended keeps its number on the process table until somebody reads
# its exit code. Holding the handles is how a worker that ended stops looking
# alive to the process that started it.
_STARTED: list[subprocess.Popen[Any]] = []


def reap_started() -> None:
    """Take the workers this process started and that have ended off the table."""
    for process in list(_STARTED):
        if process.poll() is not None:
            _STARTED.remove(process)


def open_log(path: Path, *, max_bytes: int = LOG_MAX_BYTES, keep: int = LOG_KEEP) -> Path:
    """Make sure a worker's log is there, private, and not growing for ever.

    A worker can run for days, so the file is rolled over once it is large:
    the current one becomes `.1`, the one before that `.2`, and the oldest is
    dropped. The file is created here rather than by the child, so its
    permissions are settled before anything is written into it.
    """
    security.private_dir(path.parent)
    if path.is_file() and path.stat().st_size >= max_bytes:
        oldest = path.with_suffix(f".{keep}{path.suffix}")
        oldest.unlink(missing_ok=True)
        for number in range(keep - 1, 0, -1):
            older = path.with_suffix(f".{number}{path.suffix}")
            if older.is_file():
                older.replace(path.with_suffix(f".{number + 1}{path.suffix}"))
        path.replace(path.with_suffix(f".1{path.suffix}"))
    if not path.exists():
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        os.close(os.open(path, flags, security.PRIVATE_FILE_MODE))
    return path


def spawn_detached(
    command: Sequence[str], *, log: Path, env: Mapping[str, str] | None = None
) -> Launched:
    """Start a process that outlives this one, with its output in a file.

    The worker must survive the server that asked for it, so it is put in a
    session or a process group of its own and inherits no handle on this
    process. Its output goes to a file rather than a pipe: a pipe nobody reads
    fills up and stops the process writing it, and there is nobody to read it
    once this process has gone.
    """
    open_log(log)
    extra: dict[str, Any] = {}
    if sys.platform == "win32":
        creation = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        )
        extra["creationflags"] = creation
    else:
        extra["start_new_session"] = True
    handle = log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(  # noqa: S603 - the binary is ours to name
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            env=dict(env) if env is not None else None,
            **extra,
        )
    finally:
        # The child holds its own handle on the file from here on.
        handle.close()
    _STARTED.append(process)
    return Launched(process.pid, process.poll, process.kill, process.wait)


def wait_for_entry(
    home: Path | str,
    launched: Launched,
    *,
    alias: str,
    since: float,
    timeout_s: float = DEFAULT_START_TIMEOUT_S,
    poll_s: float = 0.25,
) -> dict[str, Any]:
    """Watch for the session file the worker writes when it is ready.

    A file left by a session that crashed can name the same pid and the same
    name as the worker being waited for, and adopting one would hand a caller
    a port that anything could be answering on. So a file counts only when it
    is this name, this process, written since this start, and its process is
    not known to be gone.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        for entry in registry.list_entries(Path(home)):
            if _is_the_worker(entry, launched, alias=alias, since=since):
                return entry
        code = launched.poll()
        if code is not None:
            raise WorkerStartFailed(f"the worker exited with {code} before its bridge started")
        if time.monotonic() >= deadline:
            raise WorkerStartFailed(f"no worker bridge after {timeout_s:g} seconds")
        time.sleep(poll_s)


def _is_the_worker(
    entry: Mapping[str, Any], launched: Launched, *, alias: str, since: float
) -> bool:
    """Whether one session file was written by the worker just started."""
    if entry.get("pid") != launched.pid or entry.get("alias") != alias:
        return False
    started_at = entry.get("started_at")
    if not isinstance(started_at, (int, float)) or started_at < since:
        return False
    return registry.entry_is_live(entry) is not False


def probe_capabilities(
    entry: Mapping[str, Any], *, timeout_s: float = DEFAULT_PROBE_TIMEOUT_S
) -> dict[str, Any]:
    """Ask a fresh worker what it can do, and record the answer as data.

    A probe that comes back with nothing useful is not a failed start: the
    worker is there and works, and the row says what could not be read.
    """
    try:
        session = client.Session.from_entry(entry)
        answer = client.call(session, CAPABILITIES_TOOL, timeout_s=timeout_s)
    except (client.BridgeUnreachable, client.BridgeNotAuthentic, OSError, ValueError) as error:
        # A worker that is running but did not answer this one read is still a
        # worker. Losing its slot over a probe would be the worse answer.
        return {"probe": f"the worker did not answer the capability tool: {error}"}
    payload = answer.payload if isinstance(answer.payload, dict) else {}
    data = payload.get("data")
    if answer.status != 200 or not isinstance(data, dict):
        return {"probe": "the worker did not answer the capability tool"}
    return dict(data)


def start_worker(
    config: PoolConfig,
    store: Store,
    *,
    token: str | None = None,
    weight: str | float | None = None,
    job_id: str | None = None,
    hython: Path | str | None = None,
    spawn: Callable[..., Launched] = spawn_detached,
    probe: Callable[..., dict[str, Any]] = probe_capabilities,
    timeout_s: float | None = None,
) -> WorkerRecord:
    """Take a slot, start a worker in it, and record what came up.

    Raises `store.PoolFull` when there is no room, and `WorkerStartFailed`
    when hython came up with no bridge in it. Either way the slot is the
    pool's again before this returns.

    A failure after the process was started, a store that would not record the
    running worker included, ends that process where it can be shown to be
    the one started, and removes its session file. What became of it is put
    on the error as `spawned_pid` and `spawned_ended`, so a caller holding a
    receipt knows whether a worker may still be running.
    """
    chosen = Path(hython) if hython is not None else hython_path(config.hython)
    reserved = store.reserve_worker(
        cap=config.cap,
        token=token or _token(),
        alias_template=WORKER_ALIAS_TEMPLATE,
        job_id=job_id,
        weight=weight_of(weight),
        weight_budget=config.budget,
    )
    launched: Launched | None = None
    stamp: str | None = None
    entry: dict[str, Any] | None = None
    try:
        store.set_worker_state(reserved.token, "starting")
        since = time.time()
        launched = spawn(
            worker_command(config, alias=reserved.alias, hython=chosen),
            log=log_path(config.home, reserved.alias),
            env=worker_env(config, token=reserved.token, weight=reserved.weight),
        )
        stamp = process_start_stamp(launched.pid)
        entry = wait_for_entry(
            config.home,
            launched,
            alias=reserved.alias,
            since=since,
            timeout_s=config.start_timeout_s if timeout_s is None else timeout_s,
        )
        capabilities = probe(entry)
        # From here the worker owns its own slot. The process that started it
        # may go away without taking the warm worker with it.
        record = store.set_worker_state(
            reserved.token,
            "running",
            session_id=str(entry["session_id"]),
            owner_pid=launched.pid,
            pid=launched.pid,
            pid_start=stamp,
            capabilities=capabilities,
        )
        # A worker started for a job is taken the same way any other is, so
        # the row says which process is holding it.
        return store.lease_worker(record.token, job_id=job_id) if job_id else record
    except BaseException as error:
        ended = True
        if launched is not None:
            ended = end_spawned(launched, stamp)
            if ended and entry is not None:
                registry.remove_entry(Path(config.home), str(entry.get("session_id") or ""))
            _note_spawned(error, launched.pid, ended)
        # The slot goes back whatever went wrong, including a caller that gave
        # up on the start, so a failure cannot shrink the pool. A process that
        # could not be ended keeps its slot instead: the row names it and
        # stays stopping, and the reaper frees it once the process has gone.
        # A store that cannot take either now is left to the reaper rather
        # than hiding why the start failed.
        try:
            if ended:
                store.release_worker(reserved.token, state="failed")
            else:
                assert launched is not None
                store.set_worker_state(
                    reserved.token, "stopping", pid=launched.pid, pid_start=stamp
                )
        except store_module.StoreError:
            pass
        raise


def end_spawned(launched: Launched, stamp: str | None) -> bool:
    """End a process a failed start left behind. Says whether it has ended.

    Only a process that can be shown to be the one started is ended: through
    the handle that started it, or by pid when its start stamp matches. Never
    this process itself.
    """
    if launched.poll() is not None:
        return True
    if launched.kill is not None:
        # The handle that started the process can only reach that process,
        # so it needs no stamp to prove who the pid is.
        try:
            launched.kill()
        except OSError:
            return launched.poll() is not None
        if launched.wait is not None:
            # A kill is asked for, not done, and on some systems the process
            # is still there for a moment after. Only its exit proves it.
            try:
                launched.wait(KILL_WAIT_S)
            except (subprocess.TimeoutExpired, OSError):
                pass
            return launched.poll() is not None
    elif launched.pid == os.getpid():
        return False
    elif not kill_process(launched.pid, stamp):
        return same_process(launched.pid, stamp) is False
    deadline = time.monotonic() + KILL_WAIT_S
    while time.monotonic() < deadline:
        reap_started()
        if launched.poll() is not None or not process_is_alive(launched.pid):
            return True
        time.sleep(0.05)
    return False


def _note_spawned(error: BaseException, pid: int, ended: bool) -> None:
    try:
        error.spawned_pid = pid  # type: ignore[attr-defined]
        error.spawned_ended = ended  # type: ignore[attr-defined]
    except AttributeError:
        pass


def _token() -> str:
    """A fresh owner token for one reservation."""
    return f"wk-{secrets.token_hex(8)}"


# Section: using and letting go of a worker


def find_worker(store: Store, handle: str, *, include_stopping: bool = False) -> WorkerRecord:
    """One live worker by alias, by token or by session id.

    A worker that is stopping is not found unless asked for: it is on its way
    out, and taking it for a job or renewing its lease would only lose the
    work when it goes. A stop asks for it, so a stop can be sent again.
    """
    for record in store.list_workers():
        if record.state == "stopping" and not include_stopping:
            continue
        if handle in (record.alias, record.token, record.session_id):
            return record
    raise UnknownWorker(f"no live worker {handle}")


def reserve(store: Store, handle: str, *, job_id: str) -> WorkerRecord:
    """Take a warm worker for one job. It is held until the job cleans up.

    Finding the worker and taking it are two steps, so the taking is the one
    that decides: it is a single write that only lands on a worker no other
    job holds. Two servers that both picked the same warm worker end with one
    owner and one clear refusal.
    """
    record = find_worker(store, handle)
    try:
        return store.lease_worker(record.token, job_id=job_id)
    except store_module.WorkerTaken as error:
        raise WorkerBusy(str(error)) from error


def release(store: Store, handle: str) -> WorkerRecord:
    """Hand a worker back to the pool, warm, with no job on it."""
    record = find_worker(store, handle)
    return store.set_worker_state(record.token, "running", job_id=store_module.CLEAR)


def touch(store: Store, handle: str) -> float:
    """Renew a worker's idle lease. Any server routing a call does this."""
    return store.touch_worker_lease(find_worker(store, handle).token)


def worker_is_alive(record: WorkerRecord) -> bool:
    """Whether the process a worker row names is still that worker.

    A pid alone is not an identity, so the start stamp settles it where the
    system will say. Where it will not, a live pid counts as the worker.
    """
    if record.pid is None:
        return False
    answer = same_process(record.pid, record.pid_start)
    return process_is_alive(record.pid) if answer is None else answer


def kill_process(pid: int, pid_start: str | None = None) -> bool:
    """End a process that did not end itself, if it is still that process.

    Pid numbers are handed out again, so a kill goes ahead only when the
    process running under that number is provably the one that was started.
    Where that cannot be proved, nothing is ended: a worker left running is a
    worker a person can end, and the wrong process ended is not undoable.
    """
    if pid <= 0 or same_process(pid, pid_start) is not True:
        return False
    if sys.platform == "win32":
        return _windows_kill(pid)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _windows_kill(pid: int) -> bool:
    """There is no signal to send, so ask the kernel to end it."""
    import ctypes

    process_terminate = 0x0001
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(process_terminate, False, pid)
    if not handle:
        return False
    try:
        return bool(kernel32.TerminateProcess(handle, 1))
    finally:
        kernel32.CloseHandle(handle)


def stop_worker(
    config: PoolConfig,
    store: Store,
    handle: str,
    *,
    grace_s: float = DEFAULT_STOP_GRACE_S,
    poll_s: float = 0.5,
) -> Stopped:
    """Ask a worker to stop, then make sure it has.

    The ask is its own row: the worker watches that row and ends itself, which
    works from any process on the machine and needs no pipe to the worker. A
    worker that has not gone when the grace period is over is ended here, if
    it can be proved to still be the same process. Once it has gone its
    session file goes with it, because a file left behind names a port
    anything could be answering on. A worker that has not gone keeps its
    slot, so the pool never counts a running process as free room.
    """
    record = find_worker(store, handle, include_stopping=True)
    store.set_worker_state(record.token, "stopping")
    _wait_for_the_end(record, grace_s, poll_s)
    killed = False
    note = ""
    if worker_is_alive(record):
        killed = kill_process(int(record.pid or 0), record.pid_start)
        if killed:
            _wait_for_the_end(record, KILL_WAIT_S, poll_s)
        else:
            note = f"the process {record.pid} could not be shown to be this worker"
    ended = not worker_is_alive(record)
    if not ended:
        # The process is still there, so its slot is still taken. The row
        # stays `stopping`: the worker releases it when it goes, or the
        # reaper does once the process is shown to have ended.
        current = store.get_worker(record.token) or record
        note = note or "the process did not end in time; the slot stays taken until it does"
        return Stopped(current, killed=killed, ended=False, note=note)
    if record.session_id:
        registry.remove_entry(Path(config.home), record.session_id)
    stopped = store.release_worker(record.token, state="stopped")
    return Stopped(stopped, killed=killed, ended=True, note=note)


def _wait_for_the_end(record: WorkerRecord, grace_s: float, poll_s: float) -> None:
    """Wait for one worker's process to go, for as long as it is allowed."""
    deadline = time.monotonic() + grace_s
    while True:
        reap_started()
        if not worker_is_alive(record) or time.monotonic() >= deadline:
            return
        time.sleep(poll_s)


# Section: what a worker watches from inside itself


def lease_tick(max_idle_s: float) -> float:
    """How long the worker sleeps between looks at its row."""
    return max(0.25, min(MAX_LEASE_TICK_S, max_idle_s / 4.0))


def watch_lease(
    store: Store,
    token: str,
    *,
    max_idle_s: float,
    stop: Any,
    interval_s: float | None = None,
    clock: Callable[[], float] = time.time,
    trouble_s: float = LEASE_TROUBLE_S,
    log: Callable[[str], None] | None = None,
) -> str:
    """Watch one worker's own row until it should end, and say why.

    This runs inside the worker, on a thread of its own, so nothing on the
    machine has to stay alive to look after the pool. Five answers: `asked`
    when a server set the row to stopping, `idle` when nobody has wanted this
    worker for long enough, `gone` when the row has been ended or removed
    under it, `stopped` when the process is going down for its own reasons,
    and `unreadable` when the store has not been readable for a long time.

    A worker holding a job is never idle, however long the job runs.

    A store that another process has locked for a moment, or a read that
    failed once, is not a reason to stop watching: the pass is given up on,
    said out loud, and tried again on the next tick. Only a store that stays
    unreadable for `trouble_s` ends the watch, because a worker nobody can
    reach through the store is a worker nobody can stop.
    """
    tick = lease_tick(max_idle_s) if interval_s is None else interval_s
    note = log or (lambda _line: None)
    trouble_since: float | None = None
    while True:
        try:
            reason = _lease_pass(store, token, max_idle_s=max_idle_s, clock=clock)
        except store_module.StoreError as error:
            if trouble_since is None:
                trouble_since = clock()
                note(f"the lease could not be read: {error}")
            elif max(0.0, clock() - trouble_since) >= trouble_s:
                note(f"the lease has not been readable for {trouble_s:g} seconds")
                return "unreadable"
            reason = None
        else:
            if trouble_since is not None:
                note("the lease can be read again")
                trouble_since = None
        if reason is not None:
            return reason
        if stop.wait(tick):
            return "stopped"


def _lease_pass(
    store: Store, token: str, *, max_idle_s: float, clock: Callable[[], float]
) -> str | None:
    """One look at the row. `None` means carry on watching."""
    record = store.get_worker(token)
    if record is None or record.state in store_module.WORKER_FINAL_STATES:
        return "gone"
    if record.state == "stopping":
        # The row stays stopping, and the slot taken, until the process is
        # about to exit: the worker releases it then, or whoever stops it
        # does once the process has gone, or the reaper.
        return "asked"
    # A clock that stepped backwards must not make a worker look fresh or
    # old, so the age is never negative.
    idle_for = max(0.0, clock() - record.leased_at)
    if record.job_id is None and max_idle_s > 0 and idle_for >= max_idle_s:
        store.set_worker_state(token, "stopping")
        return "idle"
    return None


def release_on_exit(home: Path | str, token: str) -> None:
    """Give this worker's slot back as its process is about to exit.

    Called last, after the bridge has stopped, so the slot is never free while
    the process that holds it is still there to do work.
    """
    with open_store(home) as store:
        record = store.get_worker(token)
        if record is not None and record.state not in store_module.WORKER_FINAL_STATES:
            store.release_worker(token, state="stopped")


# Section: what a person is shown


def capability_summary(capabilities: Any) -> str:
    """One short line about a worker, from whatever the probe recorded."""
    if not isinstance(capabilities, Mapping):
        return "-"
    parts: list[str] = []
    version = capabilities.get("houdini_version")
    if version:
        parts.append(str(version))
    license_name = capabilities.get("license")
    if license_name:
        parts.append(str(license_name))
    renderers = capabilities.get("renderers")
    if isinstance(renderers, Sequence) and not isinstance(renderers, str) and renderers:
        parts.append("renderers " + ",".join(str(name) for name in renderers))
    capture = capabilities.get("capture")
    if isinstance(capture, Sequence) and not isinstance(capture, str) and capture:
        parts.append("capture " + ",".join(str(name) for name in capture))
    if capabilities.get("cancellation"):
        parts.append("cancel")
    return "  ".join(parts) if parts else "-"


def summary(record: WorkerRecord, *, now: float | None = None) -> dict[str, Any]:
    """One worker as plain data, for a listing."""
    moment = time.time() if now is None else now
    return {
        "alias": record.alias,
        "session_id": record.session_id or "-",
        "pid": record.pid if record.pid is not None else "-",
        "state": record.state,
        "job": record.job_id or "-",
        "weight": record.weight,
        "lease_age_s": max(0.0, moment - record.leased_at),
        "alive": worker_is_alive(record) if record.pid is not None else None,
        "capabilities": capability_summary(record.capabilities),
    }


def list_workers(store: Store, *, now: float | None = None) -> list[dict[str, Any]]:
    """Every live worker, reclaiming the ones whose process has gone first."""
    reap_started()
    store.reclaim_workers()
    moment = time.time() if now is None else now
    return [summary(record, now=moment) for record in store.list_workers()]
