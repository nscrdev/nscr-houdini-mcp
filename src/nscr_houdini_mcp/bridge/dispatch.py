"""Running one tool call, with the rules around it.

The lock in the app says one call at a time. This says the rest of it:

- Order. Calls that wait are served oldest first, so a caller that has waited
  is not passed over by one that has just arrived.
- Two budgets, kept apart. `wait_s` is how long a caller will wait for its
  turn and for the work to be picked up. `timeout_s` is how long it will wait
  for the work once it is running. A call that asks for a short wait and a
  long run is a normal thing to ask for.
- In a session with a user interface every tool runs on the main thread, reads
  included. Whether the main thread is taking work at all is decided here from
  the pulse, before anything is posted and without calling into `hou`, so a
  call that arrives during a cook is refused inside its own wait rather than
  waiting out the cook. A call that gives up is cancelled on its token, so the
  main thread finds it cancelled and never runs it late.
- A call that runs out of `timeout_s` gets `TIMEOUT` with `still_running` and
  the operation id. Nothing is interrupted: the work carries on holding the
  session, health reports it, and calls behind it wait or are told the session
  is busy. A read that timed out is also asked to stop, the same request
  `bridge.cancel` makes, since nobody is waiting for its answer.
- `skip_if_busy` answers at once rather than queueing at all. At once means
  inside a tenth of a second: such a call is refused when anything holds or
  waits for the session, when work is already queued for the main thread, or
  when the main thread has not run our code for longer than a tick. It is
  never held for a pickup budget to run out.
- A call that carries a scene epoch older than this session's is refused with
  `SCENE_REPLACED` and a summary of the scene there is now, before any tool
  runs. The paths in it belong to a scene that has been thrown away.
- A mutating call that carries an operation id takes a receipt before it runs
  and finishes it with the answer, so the same id sent again is answered from
  the receipt instead of doing the work twice. The answer is worked out and
  the receipt written on the thread that ran the work, before the session is
  given to the next caller, so a retry queued behind the work always finds
  the answer rather than a receipt that still says running. While the work
  runs, however long past its caller's timeout, the receipt's lease is renewed
  so nobody takes it over.
- Every mutating call runs inside one undo group, on the main thread where
  there is one, and a failure rolls the group's graph edits back. A change
  Houdini cannot undo, such as loading a scene, runs without a group and its
  reply says so.
- Every error carries a code from the table, and no exception text.

Cancelling is a request, not a stop. `bridge.cancel` sets a flag on the call
that holds the session, and a tool that polls the flag can return early. A
tool that does not poll it holds the session until it returns on its own, and
nothing here will take the session off it: killing work part way through an
edit is how a scene gets left half built. The tools this project ships poll
it; anything that runs arbitrary code cannot promise to.

A tool whose calls run as jobs gets a job row for each call it takes, written
by the job keeper: `queued` when the session has taken the call, `running`
once a thread has picked it up, and how it ended before the session is given
to the next caller, beside the receipt. The keeper also carries a cancel
request made through the store to the call's own flag.

Every call that may change the scene also moves the bridge's mark of whether
the scene has changes that are not on disk, which a headless session cannot
answer for itself.
"""

from __future__ import annotations

import secrets
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from nscr_houdini_mcp.bridge import encoding, host, marshal
from nscr_houdini_mcp.bridge import receipts as receipt_module
from nscr_houdini_mcp.bridge.envelope import Envelope, Reply, error_payload, ok_payload
from nscr_houdini_mcp.bridge.errors import BridgeError, did_you_mean, map_exception
from nscr_houdini_mcp.bridge.gate import Gate
from nscr_houdini_mcp.bridge.handlers import Tool, ToolRegistry, UnknownTool
from nscr_houdini_mcp.bridge.identity import Identity
from nscr_houdini_mcp.bridge.jobs import JobKeeper, JobNotAccepted
from nscr_houdini_mcp.bridge.tools import ToolContext
from nscr_houdini_mcp.bridge.undo import run_in_undo_group

# How long a call waits for its turn when it names no wait of its own. Short
# on purpose: pickup off a busy main thread runs to about a tenth of a second,
# so a second is a wide margin, and a caller that wants to queue behind a long
# job says so.
DEFAULT_WAIT_S = 1.0

# How long a call waits for work that is already running.
DEFAULT_TIMEOUT_S = 60.0

# The shortest the pickup wait can be, so a call with no wait at all still
# gives the thread that runs the work a moment to take it.
MIN_PICKUP_S = 0.25

# How long the main thread may have been away before a call that will not wait
# at all is told the session is busy. A caller that asked to be skipped wants
# an answer now, so this is far shorter than the stale limit a waiting call is
# judged by: anything longer than a tick means the answer would not be
# immediate, which is the one thing that caller asked for.
DEFAULT_SKIP_STALE_S = 0.1

OPERATION_ID_BYTES = 8

# The one tool that runs while another call holds the session.
CANCEL_TOOL = "bridge.cancel"

# Refusals a caller cannot act on without knowing what the scene is now.
SCENE_CODES = ("SCENE_REPLACED", "OUTCOME_UNKNOWN")

# How many progress notes the running call keeps for health to show. Only
# the latest few say anything a caller can use.
PROGRESS_KEPT = 8

# How often a running call renews the lease on its receipt. Well inside the
# store's lease, so a long call is never taken for one that died.
LEASE_RENEW_S = 60.0


@dataclass
class Running:
    """The call that holds the session, as the rest of the bridge sees it."""

    operation_id: str
    tool: str
    mutating: bool
    # What this call's undo entry is called.
    label: str = ""
    began: float = field(default_factory=time.monotonic)
    started_at: float = field(default_factory=time.time)
    timed_out: bool = False
    recorded: bool = False
    rolled_back: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    progress: deque = field(default_factory=lambda: deque(maxlen=PROGRESS_KEPT))
    # Set once the call has given the session back.
    ended: threading.Event = field(default_factory=threading.Event)
    # The job this call runs as, for a tool whose calls are jobs.
    job_id: str | None = None
    # What the call's receipt is bound to, so a retry can be told for the
    # same call while it still runs.
    digest: str | None = None
    # Whether the work found out it should stop, because the call was
    # cancelled or because the session is going down, and whether a progress
    # note has come since the job row was last written.
    cancel_seen: bool = False
    stop_seen: bool = False
    noted: threading.Event = field(default_factory=threading.Event)

    def elapsed_s(self) -> float:
        return round(max(0.0, time.monotonic() - self.began), 3)

    def note_progress(self, note: Mapping[str, Any]) -> None:
        """Keep one progress note from the running tool, with when it came."""
        self.progress.append({**note, "elapsed_s": self.elapsed_s()})
        self.noted.set()

    def saw_stop(self, *, cancelled: bool) -> None:
        if cancelled:
            self.cancel_seen = True
        else:
            self.stop_seen = True

    def as_dict(self) -> dict[str, Any]:
        said = {
            "operation_id": self.operation_id,
            "job_id": self.job_id,
            "tool": self.tool,
            "elapsed_s": self.elapsed_s(),
            "timed_out": self.timed_out,
            "cancel_asked": self.cancel.is_set(),
        }
        if self.progress:
            said["progress"] = dict(self.progress[-1])
        return said


class Dispatcher:
    """Runs one call at a time, in order, and says what happened."""

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        lock: Any,
        kind: str = host.HYTHON,
        session_id: str = "",
        identity: Identity | None = None,
        receipts: receipt_module.Receipts | None = None,
        hou: Any | None = None,
        log: Callable[[str], None] | None = None,
        main_loop: marshal.MainLoop | None = None,
        main_thread: marshal.MainThreadRunner | None = None,
        pulse: marshal.Pulse | None = None,
        stopping: threading.Event | None = None,
        wait_s: float = DEFAULT_WAIT_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        skip_stale_s: float = DEFAULT_SKIP_STALE_S,
        home: Any = None,
        open_store: Callable[[], Any] | None = None,
        lease_renew_s: float = LEASE_RENEW_S,
        jobs: JobKeeper | None = None,
    ) -> None:
        self.tools = tools
        self.kind = kind
        self.session_id = session_id
        self.wait_s = wait_s
        self.timeout_s = timeout_s
        self.skip_stale_s = skip_stale_s
        self.identity = identity or Identity(session_id=session_id, kind=kind)
        self.receipts = receipts or receipt_module.Receipts(None)
        self._hou = hou if hou is not None else host.houdini()
        self._log = log or (lambda text: None)
        self._main_loop = main_loop
        self._main_thread = main_thread
        self._pulse = pulse
        self._stopping = stopping or threading.Event()
        # The state folder and a way to open the store, for a tool that hands
        # out managed output paths. A bridge with neither hands out none.
        self._home = home
        self._open_store = open_store
        self._lease_renew_s = lease_renew_s
        # A bridge with no store keeps no job rows.
        self._jobs = jobs
        if jobs is None and open_store is not None:
            self._jobs = JobKeeper(open_store, session_id=session_id, home=home, log=self._log)
        self._gate = Gate(lock)
        self._running: Running | None = None
        self._last: dict[str, Any] | None = None
        if CANCEL_TOOL not in self.tools:
            self.tools.add(
                CANCEL_TOOL,
                self._ask_to_cancel,
                immediate=True,
                arguments=("operation_id",),
                summary="ask the running call to stop",
            )

    # Section: what the rest of the bridge asks

    @property
    def running(self) -> Running | None:
        return self._running

    def state(self) -> dict[str, Any]:
        """Busy state for the health endpoint. Reads no scene and no `hou`."""
        running = self._running
        gate = self._gate.state()
        return {
            "busy": running is not None,
            "queued": gate["waiting"],
            "main_thread": self._main_thread_state(),
            "current_op": None if running is None else running.tool,
            "current_op_id": None if running is None else running.operation_id,
            "current_job_id": None if running is None else running.job_id,
            "current_op_elapsed_s": None if running is None else running.elapsed_s(),
            "current_op_timed_out": False if running is None else running.timed_out,
            "current_op_progress": None if running is None else list(running.progress),
            "last_op": self._last,
        }

    def _main_thread_state(self) -> dict[str, Any]:
        """What the main thread is doing, from values held in memory."""
        state: dict[str, Any] = {}
        if self._pulse is not None:
            state.update(self._pulse.state())
        if self._main_thread is not None:
            state.update(self._main_thread.state())
        return state

    def _main_thread_away(self, wait_s: float) -> float | None:
        """How long the main thread has been away, when that is too long.

        Nothing in a session without a user interface, and nothing while the
        pulse is uninstalled: there the pickup budget alone bounds the call. A
        caller willing to wait longer than the pulse is stale is allowed to
        queue, because it has said it will wait for a busy session.
        """
        pulse = self._pulse
        if pulse is None:
            return None
        age = pulse.age_s()
        if age is None:
            return None
        return age if age > max(pulse.stale_s, wait_s) else None

    def _main_thread_slow(self) -> float | None:
        """Whether a call that will not wait would have to wait after all.

        Two things say it would: work already queued for the main thread, and
        a main thread that has not run our code for longer than a tick. The
        stale limit a waiting call is judged by is seconds long, which is the
        right answer for a caller that will wait and the wrong one for a
        caller that asked to be skipped instead.
        """
        runner = self._main_thread
        if runner is not None and runner.pending:
            return self._pulse_age() or 0.0
        pulse = self._pulse
        if pulse is None:
            return None
        age = pulse.age_s()
        if age is None:
            return None
        return age if age > self.skip_stale_s else None

    def _pulse_age(self) -> float | None:
        if self._pulse is None:
            return None
        age = self._pulse.age_s()
        return None if age is None else round(age, 3)

    # Section: one call

    def dispatch(self, envelope: Envelope) -> Reply:
        """Run one call and answer it."""
        operation_id = envelope.operation_id or _new_operation_id()
        trace = {"operation_id": operation_id, **self.identity.trace()}

        try:
            tool = self.tools.get(envelope.tool)
        except UnknownTool:
            return self._refuse(
                BridgeError(
                    "UNKNOWN_TOOL",
                    f"no tool named {envelope.tool}",
                    {
                        "tool": envelope.tool,
                        "did_you_mean": did_you_mean(envelope.tool, self.tools.names()),
                        "tools": self.tools.names(),
                    },
                    hint="call one of the names in the tool list",
                ),
                trace,
            )

        bad = _check_arguments(tool, envelope.arguments)
        if bad is not None:
            return self._refuse(bad, trace)

        if tool.immediate:
            # It takes no session and touches no scene, so it answers while
            # another call is running. That is the whole point of it.
            return self._answer_now(tool, envelope.arguments, trace)

        # The receipt is asked first, because a call that has already run is
        # answered from what it did, and a call that replaced the scene itself
        # would otherwise be refused by the scene guard for its own doing.
        # A receipt only answers a retry when the caller chose the id, so a
        # call that named none takes none: a fresh id could answer nothing.
        carried = envelope.scene_epoch
        wanted = envelope.operation_id if tool.mutating else None
        digest = _digest(tool, envelope.arguments) if wanted else None
        if wanted and digest:
            peeked = self.receipts.peek(wanted, digest, current_epoch=self.identity.scene_epoch)
            early = self._receipt_reply(peeked, tool, trace)
            if early is not None:
                return early
            # The same call sent again while it still runs, most often by a
            # caller following up on a job: it is told so now, with the job,
            # rather than queueing behind itself until its wait runs out.
            now_running = self._running
            if (
                now_running is not None
                and now_running.operation_id == wanted
                and now_running.digest in (None, digest)
            ):
                return self._still_running(tool, now_running, 0.0, trace)

        # The scene is checked here so a hopeless call is not queued at all,
        # and again on the thread that runs the work, because the scene can be
        # replaced while this call waits its turn.
        stale = self._scene_guard(carried, trace)
        if stale is not None:
            return stale

        wait_s = self.wait_s if envelope.wait_s is None else envelope.wait_s
        timeout_s = _first(envelope.timeout_s, tool.timeout_s, self.timeout_s)

        waited = time.monotonic()
        if envelope.skip_if_busy:
            # A main thread that is away, or one with work already queued for
            # it, is as busy as a session another call holds. Both cost a read
            # of a number held in memory, which is what keeps this answer
            # inside the hundred milliseconds the caller was promised.
            idle = self._main_thread_slow()
            if idle is not None:
                return self._busy(
                    trace,
                    waited=waited,
                    wait_s=wait_s,
                    cause="main thread busy",
                    main_thread_idle_s=round(idle, 3),
                    picked_up=False,
                )
        if not self._gate.enter(wait_s=wait_s, skip_if_busy=bool(envelope.skip_if_busy)):
            return self._busy(trace, waited=waited, wait_s=wait_s, cause="session busy")

        if wanted and digest:
            # The id is bound to the scene the caller wrote the call against,
            # not to whatever the scene has become while the call waited.
            verdict = self.receipts.claim(
                wanted,
                digest,
                scene_epoch=carried if carried is not None else self.identity.scene_epoch,
                current_epoch=self.identity.scene_epoch,
            )
            settled = self._receipt_reply(verdict, tool, trace)
            if settled is not None:
                self._gate.leave()
                return settled

        # The main thread is asked about here, after the session is in hand and
        # before the call becomes the running one, so health never shows an
        # operation that was refused before it began.
        idle = self._main_thread_away(wait_s)
        if idle is not None:
            self._gate.leave()
            if wanted:
                self.receipts.drop(wanted)
            return self._busy(
                trace,
                waited=waited,
                wait_s=wait_s,
                cause="main thread busy",
                main_thread_idle_s=round(idle, 3),
                picked_up=False,
            )

        running = Running(
            operation_id=operation_id,
            tool=tool.name,
            mutating=tool.mutating,
            label=tool.undo_label(envelope.arguments),
            digest=digest,
        )
        self._running = running
        context = ToolContext(
            hou=self._hou,
            kind=self.kind,
            session_id=self.session_id,
            scene_epoch=self.identity.scene_epoch,
            operation_id=operation_id,
            label=running.label,
            cancel=running.cancel,
            stopping=self._stopping,
            progress=running.note_progress,
            home=self._home,
            open_store=self._open_store,
            saw_stop=running.saw_stop,
            dirty=self.identity.dirty,
        )
        if tool.job_kind and self._jobs is not None:
            try:
                self._jobs.accept(
                    running,
                    kind=tool.job_kind,
                    spec=tool.job_spec(envelope.arguments) if tool.job_spec else None,
                    identity=self.identity.trace(),
                )
            except JobNotAccepted as refused:
                # A job nobody could follow must not run: the session goes
                # back and the receipt with it, so the same id can try again.
                self._never_ran(wanted, running)
                hint = (
                    "use a new operation_id"
                    if refused.code == "JOB_ID_TAKEN"
                    else "call again shortly; the store was busy or could not be written"
                )
                return self._refuse(
                    BridgeError(refused.code, refused.message, {"tool": tool.name}, hint=hint),
                    trace,
                )

        work = marshal.Work(
            lambda: self._work(tool, envelope.arguments, context, running, carried, wanted, trace)
        )
        if wanted:
            self._keep_lease(wanted, running)
        runner = marshal.choose_runner(
            self.kind,
            mutating=tool.mutating,
            hou=self._hou,
            main_loop=self._main_loop,
            main_thread=self._main_thread,
        )
        try:
            # Submitting to the main thread runner is a queue put and cannot
            # raise; the other runners can, and a user interface being torn
            # down refuses a posted callback.
            runner.submit(work)

            pickup_s = max(MIN_PICKUP_S, wait_s - (time.monotonic() - waited))
            if not work.started.wait(pickup_s) and work.cancel():
                self._never_ran(wanted, running)
                return self._busy(
                    trace,
                    waited=waited,
                    wait_s=wait_s,
                    cause="main thread busy",
                    main_thread_idle_s=self._pulse_age(),
                    picked_up=False,
                )
        except BaseException as error:  # noqa: BLE001 - a session held for ever is worse
            # The session goes back rather than staying busy with nothing
            # running in it.
            self._log(f"could not hand {tool.name} over: {type(error).__name__}: {error}")
            if work.cancel():
                self._never_ran(wanted, running)
            return self._could_not_take(tool, type(error).__name__, trace)

        if isinstance(work.error, marshal.Rejected):
            # The work was given back rather than run: the interface would not
            # take the post. The caller hears that now instead of waiting out
            # its budget for work nobody is going to do.
            self._never_ran(wanted, running)
            return self._could_not_take(tool, type(work.error).__name__, trace)

        if not work.finished.wait(timeout_s):
            running.timed_out = True
            if not tool.mutating:
                # Nobody is waiting for a read any more, so it is asked to
                # stop at its next look at the flag rather than run on.
                running.cancel.set()
            # The work goes on, renewing its receipt, and writes its answer
            # there when it ends, before the session is given back.
            return self._still_running(tool, running, timeout_s, trace)

        return self._finished(tool, work, running, trace, wanted)

    def _still_running(
        self, tool: Tool, running: Running, timeout_s: float, trace: Mapping[str, Any]
    ) -> Reply:
        """`TIMEOUT` with `still_running`: the work goes on, and here is its job."""
        return Reply(
            200,
            {
                **error_payload(
                    "TIMEOUT",
                    f"{tool.name} is still running after {timeout_s:g} seconds",
                    hint="the work goes on, ask health for the session before calling again",
                    details={
                        "tool": tool.name,
                        "operation_id": running.operation_id,
                        "timeout_s": timeout_s,
                        "still_running": True,
                        **({"job_id": running.job_id} if running.job_id else {}),
                    },
                ),
                **self._said({**trace, "operation_id": running.operation_id}),
            },
        )

    # Section: answering

    def _finished(
        self,
        tool: Tool,
        work: marshal.Work,
        running: Running,
        trace: dict[str, Any],
        wanted: str | None,
    ) -> Reply:
        """The answer the work settled on, with which route reached it."""
        reply = work.result
        if work.error is not None or not isinstance(reply, Reply):
            # The work failed outside the tool, which leaves no answer and no
            # receipt behind it. Both are made here instead.
            reply = self._failed(tool, work.error or RuntimeError("no answer"), running, trace)
            self._settle_call(running, wanted, reply.payload)
            return reply
        if work.picked_by is not None:
            # Which route to the main thread reached the work first, so a check
            # against a real Houdini can tell the two apart.
            return Reply(reply.status, {**reply.payload, "picked_by": work.picked_by})
        return reply

    def _keep_lease(self, operation_id: str, running: Running) -> None:
        """Renew the receipt's lease until the call gives the session back.

        A call can run far past its caller's timeout. Without this its receipt
        would look abandoned to the next caller presenting the id once the
        store's lease ran out.
        """

        def renew() -> None:
            while not running.ended.wait(self._lease_renew_s):
                self.receipts.touch(operation_id)

        threading.Thread(target=renew, name="nscr-mcp-lease", daemon=True).start()

    def _said(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        """The trace as it is at the moment of answering.

        The session and the epoch are read again here rather than kept from
        the start of the call, because a call that replaced the scene has to
        hand back the epoch a caller should use next, not the one it used.
        """
        return {**trace, **self.identity.trace()}

    # Section: the scene and the receipt

    def _scene_guard(self, carried: int | None, trace: Mapping[str, Any]) -> Reply | None:
        """Refuse a call written against a scene this session has thrown away."""
        try:
            self._still_the_same_scene(carried)
        except BridgeError as replaced:
            return self._refuse(replaced, trace)
        return None

    def _still_the_same_scene(self, carried: int | None) -> None:
        """Raise when the scene is not the one the call was written against."""
        current = self.identity.scene_epoch
        if carried is None or carried == current:
            return
        raise BridgeError(
            "SCENE_REPLACED",
            "this session has replaced its scene since that call was written",
            {"carried_epoch": carried, "scene_epoch": current},
            hint="read the scene again, then send the call with the new epoch",
        )

    def _receipt_reply(
        self, verdict: receipt_module.Verdict, tool: Tool, trace: Mapping[str, Any]
    ) -> Reply | None:
        """Turn a receipt verdict into an answer, or nothing when it may run."""
        if verdict.may_run:
            return None
        if verdict.action == receipt_module.REPLAY:
            stored = verdict.outcome
            if isinstance(stored, Mapping):
                return Reply(200, {**stored, "replayed": True})
            # A receipt with no answer in it says nothing useful, so the call
            # is treated as one whose outcome nobody knows.
            return self._unknown(verdict, tool, trace)
        if verdict.action == receipt_module.MISMATCH:
            return self._refuse(
                BridgeError(
                    "OPERATION_MISMATCH",
                    "that operation id was used for a different call",
                    {"tool": tool.name, **verdict.state()},
                    hint="use a new operation id, or send the arguments the id was used with",
                ),
                trace,
            )
        if verdict.action == receipt_module.SCENE_REPLACED:
            return self._refuse(
                BridgeError(
                    "SCENE_REPLACED",
                    "that operation id belongs to a scene this session has replaced",
                    {
                        "recorded_epoch": verdict.recorded_epoch,
                        "scene_epoch": self.identity.scene_epoch,
                    },
                    hint="read the scene again and send the call with a new operation id",
                ),
                trace,
                scene=True,
            )
        return self._unknown(verdict, tool, trace)

    def _unknown(
        self, verdict: receipt_module.Verdict, tool: Tool, trace: Mapping[str, Any]
    ) -> Reply:
        return self._refuse(
            BridgeError(
                "OUTCOME_UNKNOWN",
                "that operation id may already have changed the scene",
                {"tool": tool.name, "reason": verdict.reason, "receipt": verdict.state()},
                hint="read the scene, decide what is already there, then call with a new id",
            ),
            trace,
            scene=True,
        )

    def _answer_now(
        self, tool: Tool, arguments: Mapping[str, Any], trace: Mapping[str, Any]
    ) -> Reply:
        """Run a tool that takes no session, here on the calling thread."""
        began = time.monotonic()
        try:
            data = tool.run(arguments, ToolContext(kind=self.kind, session_id=self.session_id))
        except BaseException as error:  # noqa: BLE001 - one failed tool, not a failed bridge
            return self._failed(tool, error, None, trace)
        converted = encoding.convert(data)
        payload = {
            **ok_payload(converted.value, timing_ms=(time.monotonic() - began) * 1000.0),
            **self._said(trace),
        }
        if converted.lossy:
            payload["lossy"] = True
            payload["cut"] = converted.cut
        return Reply(200, payload)

    def _ask_to_cancel(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Ask the call that holds the session to stop.

        It is a flag, not a stop. A tool that polls it returns early; one that
        does not runs to the end, and the answer says so.
        """
        running = self._running
        if running is None:
            return {"asked": False, "reason": "nothing is running"}
        wanted = arguments.get("operation_id")
        if wanted and str(wanted) != running.operation_id:
            return {
                "asked": False,
                "reason": "that call is not the one running",
                "current_op_id": running.operation_id,
            }
        running.cancel.set()
        if running.job_id and self._jobs is not None:
            self._jobs.asked_to_stop(running)
        return {
            "asked": True,
            "operation_id": running.operation_id,
            "tool": running.tool,
            "note": "a tool that does not poll the flag runs to the end",
        }

    # Section: running the work

    def _work(
        self,
        tool: Tool,
        arguments: Mapping[str, Any],
        context: ToolContext,
        running: Running,
        carried: int | None = None,
        wanted: str | None = None,
        trace: Mapping[str, Any] | None = None,
    ) -> Reply:
        """The whole of one call, on whichever thread it was given to.

        In a session with a user interface that thread is the main thread, for
        a read as much as for a mutation. The undo group is still only for a
        tool that changes the scene.

        The answer is made here, while the session is still held: the value is
        converted on this thread, where reading a node is safe, and the receipt
        is written before the next caller can present the same id.
        """
        began = time.monotonic()
        value: Any = None
        error: BaseException | None = None
        marked = False
        try:
            try:
                # The last look at the scene, here on the thread that is about
                # to touch it. A call can wait a long time for its turn, and
                # the session may have loaded another scene while it waited:
                # the paths in its arguments would then mean something else.
                self._still_the_same_scene(carried)
                if running.job_id and self._jobs is not None:
                    self._jobs.started(running, self._hou)
                if tool.mutating:
                    self.identity.dirty.began(tool.name)
                    marked = True
                if not tool.mutating or not tool.undoable or self._hou is None:
                    value = tool.run(arguments, context)
                else:
                    outcome = run_in_undo_group(
                        lambda: tool.run(arguments, context),
                        label=running.label or tool.undo_label(arguments),
                        hou=self._hou,
                    )
                    running.recorded = outcome.recorded
                    running.rolled_back = outcome.rolled_back
                    if outcome.error is not None:
                        raise outcome.error
                    value = outcome.value
            except BaseException as raised:  # noqa: BLE001 - becomes the coded answer
                error = raised
            if marked:
                self.identity.dirty.ended(tool.name, ok=error is None)
            timing_ms = (time.monotonic() - began) * 1000.0
            reply = self._settle(tool, value, error, running, dict(trace or {}), timing_ms)
            # Written before the session is given back, so a retry queued
            # behind the work and a caller following its job both find the
            # answer rather than a call that still says running.
            self._settle_call(running, wanted, reply.payload)
            return reply
        finally:
            self._release(running)

    def _settle_call(
        self, running: Running, wanted: str | None, payload: Mapping[str, Any]
    ) -> None:
        """Write the call's receipt, and its job's ending in the same step.

        When the joint write cannot land, the receipt is written on its own,
        and the sweep that finds the job still running takes the ending from
        the receipt.
        """
        if running.job_id and self._jobs is not None:
            operation = self.receipts.settlement(wanted, payload) if wanted else None
            if self._jobs.finish(running, payload, operation=operation) or not wanted:
                return
        if wanted:
            self.receipts.finish(wanted, payload)

    def _never_ran(self, wanted: str | None, running: Running) -> None:
        """Give the session back, and take back the receipt with it.

        The work was cancelled before any thread took it, so the tool was
        never reached. Leaving the receipt behind would answer every later
        attempt at that id with an outcome nobody knows, for work that never
        happened. The caller is free to send it again.
        """
        self._release(running)
        if wanted:
            self.receipts.drop(wanted)
        if running.job_id and self._jobs is not None:
            self._jobs.drop(running)

    def _release(self, running: Running) -> None:
        """Note how the call ended and give the session to the next caller."""
        if self._running is running:
            self._last = {**running.as_dict(), "finished_at": time.time()}
            self._running = None
        running.ended.set()
        self._gate.leave()

    # Section: answers

    def _settle(
        self,
        tool: Tool,
        value: Any,
        error: BaseException | None,
        running: Running,
        trace: dict[str, Any],
        timing_ms: float,
    ) -> Reply:
        """One call's answer, from what its tool returned or raised."""
        if error is not None:
            return self._failed(tool, error, running, trace)

        converted = encoding.convert(value, **(tool.caps or {}))
        payload = {**ok_payload(converted.value, timing_ms=timing_ms), **self._said(trace)}
        if running.job_id:
            payload["job_id"] = running.job_id
        if converted.lossy:
            payload["lossy"] = True
            payload["cut"] = converted.cut
        label = running.label or tool.undo_label()
        if tool.mutating and tool.undoable:
            payload["undo"] = {
                "label": label,
                "recorded": running.recorded,
                "rolled_back": running.rolled_back,
            }
        elif tool.mutating:
            payload["undo"] = {"label": label, "undoable": False}
        return Reply(200, payload)

    def _failed(
        self, tool: Tool, error: BaseException, running: Running | None, trace: Mapping[str, Any]
    ) -> Reply:
        # The text of an exception can hold paths and scene contents, so it
        # goes to the local log and the caller gets a code and a type.
        self._log(
            f"tool {tool.name} raised: {type(error).__name__}: {error}\n"
            + "".join(traceback.format_exception(error)[-8:])
        )
        coded = map_exception(error, tool=tool.name)
        if tool.mutating and tool.undoable and running is not None:
            coded.details.setdefault("rolled_back", running.rolled_back)
            coded.details.setdefault("undo_recorded", running.recorded)
        return self._refuse(coded, trace)

    def _could_not_take(self, tool: Tool, exception: str, trace: Mapping[str, Any]) -> Reply:
        """The work was never handed over, and nobody is going to run it."""
        return self._refuse(
            BridgeError(
                "TOOL_FAILED",
                "the session could not take the work",
                {"tool": tool.name, "exception": exception},
                hint="ask health whether this session is still there",
            ),
            trace,
        )

    def _refuse(
        self, error: BridgeError, trace: Mapping[str, Any], *, scene: bool = False
    ) -> Reply:
        """One coded refusal, with the scene summary where it helps.

        The summary sits beside the error rather than inside it, because
        anything inside an error has the places on disk taken out of it and a
        caller being told its scene has gone needs to know which one is there
        now.
        """
        safe = error.safe()
        payload = {
            **error_payload(safe.code, safe.message, hint=safe.hint, details=safe.details),
            **self._said(trace),
        }
        if scene or safe.code in SCENE_CODES:
            payload["scene"] = self.identity.scene()
        return Reply(200, payload)

    def _busy(
        self,
        trace: Mapping[str, Any],
        *,
        waited: float,
        wait_s: float,
        cause: str,
        **extra: Any,
    ) -> Reply:
        """One busy refusal.

        The cause says which kind of busy it is, and `extra` carries how long
        the main thread has been away where that is the reason.
        """
        running = self._running
        return self._refuse(
            BridgeError(
                "SESSION_BUSY",
                "this session is running another call",
                {
                    "cause": cause,
                    **extra,
                    "current_op": None if running is None else running.tool,
                    "current_op_id": None if running is None else running.operation_id,
                    "elapsed_s": None if running is None else running.elapsed_s(),
                    "queued": self._gate.waiting(),
                    "waited_s": round(time.monotonic() - waited, 3),
                    "wait_s": wait_s,
                },
                hint="wait for the running call to finish, then send this one again",
            ),
            trace,
        )


# Section: arguments


def _check_arguments(tool: Tool, arguments: Mapping[str, Any]) -> BridgeError | None:
    """Refuse a call whose argument names are not the tool's, with the near ones."""
    if tool.arguments is None:
        return None
    unknown = [name for name in arguments if name not in tool.arguments]
    if unknown:
        return BridgeError(
            "BAD_ARGUMENTS",
            f"{tool.name} takes no argument named {unknown[0]}",
            {
                "tool": tool.name,
                "unknown": sorted(unknown),
                "did_you_mean": did_you_mean(unknown[0], tool.arguments),
                "arguments": list(tool.arguments),
            },
            hint="use one of the argument names this tool takes",
        )
    missing = [name for name in tool.required if arguments.get(name) is None]
    if missing:
        return BridgeError(
            "BAD_ARGUMENTS",
            f"{tool.name} needs {missing[0]}",
            {
                "tool": tool.name,
                "missing": sorted(missing),
                "required": list(tool.required),
                "arguments": list(tool.arguments),
            },
            hint="send every required argument",
        )
    return None


def _first(*values: float | None) -> float:
    """The first budget that was actually given. A zero is a value, not a gap."""
    for value in values:
        if value is not None:
            return float(value)
    return DEFAULT_TIMEOUT_S


def _new_operation_id() -> str:
    return f"op-{secrets.token_hex(OPERATION_ID_BYTES)}"


def _digest(tool: Tool, arguments: Mapping[str, Any]) -> str | None:
    """The digest an operation id is bound to, or nothing when there is none.

    An argument JSON cannot carry has no stable digest, so the call runs with
    no receipt rather than with one that could match the wrong arguments.
    Arguments a tool names as filled in by the sender rather than chosen by
    the caller are left out, so a retry from another sender still matches.
    """
    kept = {key: value for key, value in arguments.items() if key not in tool.digest_ignores}
    try:
        return receipt_module.digest_call(tool.name, kept)
    except Exception:  # noqa: BLE001 - no receipt is better than a wrong one
        return None
