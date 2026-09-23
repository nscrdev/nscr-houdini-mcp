"""The job row a session keeps for the work it is running.

The running operation is the job. When the session takes a call whose tool
runs as a job, it writes the row (`queued`), marks it `running` once a thread
has picked the work up, and writes how it ended before the session is given
to the next caller, the same moment the receipt is settled. While the work
runs, one small thread per job does three things:

- Writes the latest progress note to the row, at most once a second however
  often the code reports.
- Reads the row's cancel request, after a progress note and at least every
  two seconds, and sets the call's own cancel flag when it finds one. That is
  what `mcp.cancelled()` looks at, so a job cancelled from any server process
  sees it even when the direct request to this session never arrived.
- Keeps the row's heartbeat fresh, so the job is not taken for one whose
  session has gone quiet, and renews the idle lease of the worker it runs
  in. The job sits on the worker's row from accept to finish, so a worker
  running a long job is never stopped for being idle.

How it ends. Work that saw the cancel flag and stopped is `cancelled`. Work
that finished without ever looking is `done`, with `cancel_requested` still
on the row. Work that raised, or a call that failed, is `failed` with the
error. A finished job leaves a readable copy of its row beside the scene.

The row has to be there before the work may run: the write at accept is tried
a few times over a busy store, and the call is refused when it cannot land,
as it is when a job under the same id is still kept. After that the writes
mend the row rather than give up on it: a heartbeat makes a missing row again
and moves a queued one on, and the ending is written over a row found `lost`
meanwhile as a late finish. Every write goes through the store's rules for
how a job may move, so nothing brings an ended job back.

This module never imports `hou` itself; the one it is handed is read only on
the thread that runs the work.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from nscr_houdini_mcp import jobs as job_rules
from nscr_houdini_mcp import store as store_module

# At most one progress write a second, and a look at the cancel request at
# least every two seconds.
PROGRESS_WRITE_S = 1.0
CANCEL_POLL_S = 2.0

# The pauses between tries at a write that has to land, such as the row a
# job needs before its work may run.
ACCEPT_BACKOFF_S = (0.05, 0.2, 0.5)

# How much of a Python answer the row keeps. The whole answer is on the
# receipt under the operation id.
RESULT_KEPT_CHARS = 4000
STDOUT_KEPT_CHARS = 2000


class JobNotAccepted(Exception):
    """The job row could not be written, so the call must not run."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class _Job:
    job_id: str
    kind: str
    # What the row says about the job, kept here so a heartbeat can make the
    # row again when it is missing.
    repair: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False
    written: Any = None


class JobKeeper:
    """Writes the rows of the jobs one session runs."""

    def __init__(
        self,
        open_store: Callable[[], store_module.Store],
        *,
        session_id: str = "",
        pid: int | None = None,
        home: Any = None,
        log: Callable[[str], None] | None = None,
        write_every_s: float = PROGRESS_WRITE_S,
        poll_every_s: float = CANCEL_POLL_S,
        backoff_s: tuple[float, ...] = ACCEPT_BACKOFF_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._open_store = open_store
        self._session_id = session_id
        self._pid = os.getpid() if pid is None else pid
        self._home = home
        self._log = log or (lambda text: None)
        self._write_every_s = write_every_s
        self._poll_every_s = poll_every_s
        self._backoff_s = backoff_s
        self._clock = clock
        self._jobs: dict[int, _Job] = {}
        self._lock = threading.Lock()

    # Section: the three moments

    def accept(
        self,
        running: Any,
        *,
        kind: str,
        spec: Mapping[str, Any] | None,
        identity: Mapping[str, Any],
    ) -> str:
        """Write the row for work the session has taken, and start watching it.

        The row is written before the work may run, retried a few times over
        a busy store, and `JobNotAccepted` when it cannot be: a caller must
        never be handed a job that has no row. An id whose job is still kept
        is refused the same way, so an old job and its copy are never
        written over.
        """
        job_id = job_rules.job_id_for(running.operation_id)
        scene = {
            "session_id": identity.get("session_id") or self._session_id,
            "alias": identity.get("alias"),
            "scene_epoch": identity.get("scene_epoch"),
        }
        repair = {
            "kind": kind,
            "session_id": self._session_id or None,
            "operation_id": running.operation_id,
            "spec": dict(spec or {}),
            "scene": scene,
        }
        try:
            self._durably(
                lambda store: store.create_job(
                    job_id,
                    kind=kind,
                    session_id=self._session_id or None,
                    state="queued",
                    scene=scene,
                    worker_pid=self._pid,
                    operation_id=running.operation_id,
                    spec=repair["spec"],
                    # The same operation id taken again once its job has
                    # gone past keeping is a new run of it; before that the
                    # job and its answer stay.
                    replace_after_s=job_rules.KEEP_S,
                )
            )
        except store_module.JobIdTaken:
            raise JobNotAccepted(
                "JOB_ID_TAKEN", f"a job is still kept under {job_id}; use a new operation id"
            ) from None
        except (store_module.StoreError, sqlite3.Error, OSError) as error:
            raise JobNotAccepted(
                "STORE_UNAVAILABLE",
                f"the job row could not be written ({type(error).__name__}), so nothing ran",
            ) from None
        running.job_id = job_id
        job = _Job(job_id=job_id, kind=kind, repair=repair)
        with self._lock:
            self._jobs[id(running)] = job
        if self._session_id:
            self._write(lambda store: store.hold_worker_for_job(self._session_id, job_id))
        threading.Thread(
            target=self._watch, args=(running, job), name="nscr-mcp-job", daemon=True
        ).start()
        return job_id

    def started(self, running: Any, hou: Any) -> None:
        """The work has been picked up. On the thread that runs it, so `hou` is safe.

        Only a queued row moves to running here; a row found lost meanwhile
        stays lost until the work says how it really ended.
        """
        job = self._job(running)
        if job is None:
            return
        facts = {
            "hip_path": _quiet(lambda: str(hou.hipFile.path())) if hou is not None else None,
            "untitled": _quiet(lambda: bool(hou.hipFile.isNewFile())) if hou is not None else None,
            "houdini_version": _quiet(hou.applicationVersionString) if hou is not None else None,
        }
        with job.lock:
            job.repair["scene"].update({k: v for k, v in facts.items() if v is not None})
            scene = dict(job.repair["scene"])
            self._write(lambda store: store.start_job(job.job_id, scene=scene))

    def finish(self, running: Any, payload: Mapping[str, Any]) -> None:
        """Write how the work ended, and leave the readable copy beside the scene."""
        job = self._job(running, forget=True)
        if job is None:
            return
        state, outputs, error = ending(job.kind, running, payload)
        progress = job_rules.progress_of(latest(running)) or None
        with job.lock:
            job.closed = True
            try:
                self._durably(
                    lambda store: settle(
                        store, job, state=state, progress=progress, outputs=outputs, error=error
                    )
                )
            except Exception as error:  # noqa: BLE001 - logged; the receipt holds the answer
                self._log(f"could not write how job {job.job_id} ended: {error}")
        self._free(job)
        self._write(lambda store: job_rules.export(store, job.job_id, home=self._home))

    def asked_to_stop(self, running: Any) -> None:
        """Put a stop asked of the session directly on the row as well."""
        job = self._job(running)
        if job is None:
            return
        with job.lock:
            if not job.closed:
                self._write(lambda store: store.request_job_cancel(job.job_id))

    def drop(self, running: Any) -> None:
        """Take the row back for work that was never picked up."""
        job = self._job(running, forget=True)
        if job is None:
            return
        with job.lock:
            job.closed = True
            self._write(lambda store: store.drop_job(job.job_id))
        self._free(job)

    def _free(self, job: _Job) -> None:
        """Take the job off the worker row, which starts its idle wait again."""
        if self._session_id:
            self._write(lambda store: store.free_worker_of_job(self._session_id, job.job_id))

    # Section: while it runs

    def _watch(self, running: Any, job: _Job) -> None:
        """Progress, the cancel request and the heartbeat, until the work ends."""
        last = self._clock()
        while not running.ended.is_set():
            running.noted.wait(self._poll_every_s)
            if running.noted.is_set():
                # A burst of notes is written once a second, not once a note.
                wait = self._write_every_s - (self._clock() - last)
                if wait > 0 and running.ended.wait(wait):
                    break
                running.noted.clear()
            if running.ended.is_set():
                break
            self._beat(running, job)
            last = self._clock()

    def _beat(self, running: Any, job: _Job) -> None:
        with job.lock:
            if job.closed:
                return
            note = job_rules.progress_of(latest(running))
            fresh = note if note is not None and note != job.written else None
            repair = {**job.repair, "scene": dict(job.repair["scene"])}
            # One write that also mends the row: made again when it is
            # missing, moved on from queued, left alone once it has ended.
            record = self._write(
                lambda store: store.beat_job(
                    job.job_id, progress=fresh, worker_pid=self._pid, repair=repair
                )
            )
            if record is not None and fresh is not None:
                job.written = fresh
            if self._session_id:
                self._write(lambda store: store.renew_worker_of_session(self._session_id))
        if record is not None and record.cancel_requested and not running.cancel.is_set():
            self._log(f"job {job.job_id} was asked to stop")
            running.cancel.set()

    # Section: helpers

    def _job(self, running: Any, *, forget: bool = False) -> _Job | None:
        with self._lock:
            if forget:
                return self._jobs.pop(id(running), None)
            return self._jobs.get(id(running))

    def _write(self, change: Callable[[store_module.Store], Any]) -> Any:
        try:
            store = self._open_store()
        except Exception as error:  # noqa: BLE001 - the work goes on without its row
            self._log(f"could not open the store for a job: {type(error).__name__}: {error}")
            return None
        try:
            return change(store)
        except Exception as error:  # noqa: BLE001 - the work goes on without its row
            self._log(f"could not write a job row: {type(error).__name__}: {error}")
            return None
        finally:
            store.close()

    def _durably(self, change: Callable[[store_module.Store], Any]) -> Any:
        """One write that has to land, tried again over a busy store, or raised."""
        last: BaseException | None = None
        for pause in (*self._backoff_s, None):
            try:
                store = self._open_store()
                try:
                    return change(store)
                finally:
                    store.close()
            except store_module.JobIdTaken:
                raise
            except (store_module.StoreError, sqlite3.Error, OSError) as error:
                last = error
                self._log(f"a job write did not land: {type(error).__name__}: {error}")
                if pause is not None:
                    time.sleep(pause)
        assert last is not None
        raise last


def settle(
    store: store_module.Store,
    job: _Job,
    *,
    state: str,
    progress: Any,
    outputs: Any,
    error: Any,
) -> store_module.JobRecord:
    """Write how a job ended, whatever its row went through meanwhile.

    A missing row is made again first. A row found lost while the work ran
    takes the real ending as a late finish, which clears the loss.
    """
    if store.get_job(job.job_id) is None:
        store.beat_job(job.job_id, repair={**job.repair, "scene": dict(job.repair["scene"])})
    try:
        return store.update_job(
            job.job_id, state=state, progress=progress, outputs=outputs, error=error
        )
    except store_module.JobMoveRefused:
        return store.update_job(
            job.job_id, state=state, progress=progress, outputs=outputs, error=error, late=True
        )


def latest(running: Any) -> Any:
    progress = getattr(running, "progress", None)
    return dict(progress[-1]) if progress else None


def ending(kind: str, running: Any, payload: Mapping[str, Any]) -> tuple[str, dict[str, Any], Any]:
    """The final state, the outputs and the error, from the call's answer."""
    stopped = bool(getattr(running, "cancel_seen", False))
    outputs: dict[str, Any] = {"operation_id": running.operation_id}
    if not payload.get("ok"):
        error = payload.get("error")
        return ("cancelled" if stopped else "failed"), outputs, error
    data = payload.get("data")
    data = data if isinstance(data, Mapping) else {}
    error = data.get("error") if kind == "python" else None
    if kind == "python":
        outputs.update(python_outputs(data))
    if stopped:
        return "cancelled", outputs, error
    return ("failed" if error is not None else "done"), outputs, error


def python_outputs(data: Mapping[str, Any]) -> dict[str, Any]:
    """What a job row keeps of one Python answer: a short form of it."""
    kept: dict[str, Any] = {
        "namespace": data.get("namespace"),
        "duration_ms": data.get("duration_ms"),
    }
    result = data.get("result")
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
    if data.get("result_text_chars") is None and len(text) <= RESULT_KEPT_CHARS:
        kept["result"] = result
    else:
        kept["result_chars"] = int(data.get("result_text_chars") or len(text))
    stdout = str(data.get("stdout") or "")
    kept["stdout_tail"] = stdout[-STDOUT_KEPT_CHARS:]
    if len(stdout) > STDOUT_KEPT_CHARS:
        kept["stdout_chars"] = len(stdout)
    return kept


def _quiet(read: Callable[[], Any]) -> Any:
    try:
        return read()
    except Exception:  # noqa: BLE001 - a fact we cannot read is a fact we do not have
        return None
