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
  is busy.
- `skip_if_busy` answers at once rather than queueing at all.
- A call that carries a scene epoch older than this session's is refused with
  `SCENE_REPLACED` and a summary of the scene there is now, before any tool
  runs. The paths in it belong to a scene that has been thrown away.
- A mutating call that carries an operation id takes a receipt before it runs
  and finishes it with the answer, so the same id sent again is answered from
  the receipt instead of doing the work twice.
- Every mutating call runs inside one undo group, on the main thread where
  there is one, and a failure rolls the group's graph edits back.
- Every error carries a code from the table, and no exception text.

Cancelling is a request, not a stop. `bridge.cancel` sets a flag on the call
that holds the session, and a tool that polls the flag can return early. A
tool that does not poll it holds the session until it returns on its own, and
nothing here will take the session off it: killing work part way through an
edit is how a scene gets left half built. The tools this project ships poll
it; anything that runs arbitrary code cannot promise to.
"""

from __future__ import annotations

import secrets
import threading
import time
import traceback
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

OPERATION_ID_BYTES = 8

# The one tool that runs while another call holds the session.
CANCEL_TOOL = "bridge.cancel"

# Refusals a caller cannot act on without knowing what the scene is now.
SCENE_CODES = ("SCENE_REPLACED", "OUTCOME_UNKNOWN")


@dataclass
class Running:
    """The call that holds the session, as the rest of the bridge sees it."""

    operation_id: str
    tool: str
    mutating: bool
    began: float = field(default_factory=time.monotonic)
    started_at: float = field(default_factory=time.time)
    timed_out: bool = False
    recorded: bool = False
    rolled_back: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)

    def elapsed_s(self) -> float:
        return round(max(0.0, time.monotonic() - self.began), 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "tool": self.tool,
            "elapsed_s": self.elapsed_s(),
            "timed_out": self.timed_out,
            "cancel_asked": self.cancel.is_set(),
        }


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
    ) -> None:
        self.tools = tools
        self.kind = kind
        self.session_id = session_id
        self.wait_s = wait_s
        self.timeout_s = timeout_s
        self.identity = identity or Identity(session_id=session_id, kind=kind)
        self.receipts = receipts or receipt_module.Receipts(None)
        self._hou = hou if hou is not None else host.houdini()
        self._log = log or (lambda text: None)
        self._main_loop = main_loop
        self._main_thread = main_thread
        self._pulse = pulse
        self._stopping = stopping or threading.Event()
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
            "current_op_elapsed_s": None if running is None else running.elapsed_s(),
            "current_op_timed_out": False if running is None else running.timed_out,
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
            # A main thread that is away is as busy as a session another call
            # holds, and saying so costs one float read.
            idle = self._main_thread_away(0.0)
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

        running = Running(operation_id=operation_id, tool=tool.name, mutating=tool.mutating)
        self._running = running
        context = ToolContext(
            hou=self._hou,
            kind=self.kind,
            session_id=self.session_id,
            scene_epoch=self.identity.scene_epoch,
            operation_id=operation_id,
            label=tool.undo_label(),
            cancel=running.cancel,
            stopping=self._stopping,
        )

        work = marshal.Work(lambda: self._work(tool, envelope.arguments, context, running, carried))
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
            if wanted:
                # The work is still going, so the receipt says so rather than
                # looking abandoned to the next caller that presents the id,
                # and it is finished with the answer whenever the work ends.
                self.receipts.touch(wanted)
                self._record_when_it_ends(wanted, tool, work, running, dict(trace))
            return Reply(
                200,
                {
                    **error_payload(
                        "TIMEOUT",
                        f"{tool.name} is still running after {timeout_s:g} seconds",
                        hint="the work goes on, ask health for the session before calling again",
                        details={
                            "tool": tool.name,
                            "operation_id": operation_id,
                            "timeout_s": timeout_s,
                            "still_running": True,
                        },
                    ),
                    **self._said(trace),
                },
            )

        reply = self._answer(tool, work, running, trace)
        if wanted:
            self.receipts.finish(wanted, reply.payload)
        return reply

    # Section: answering

    def _record_when_it_ends(
        self,
        operation_id: str,
        tool: Tool,
        work: marshal.Work,
        running: Running,
        trace: dict[str, Any],
    ) -> None:
        """Finish the receipt of a call whose work outlived it.

        The caller has been told the work goes on. What it ends up doing still
        belongs under its operation id, so the same id sent again gets the
        answer rather than being told that nobody knows.
        """

        def record() -> None:
            work.finished.wait()
            try:
                self.receipts.finish(operation_id, self._answer(tool, work, running, trace).payload)
            except Exception as error:  # noqa: BLE001 - a missed receipt is not a failed call
                self._log(f"could not record {operation_id}: {type(error).__name__}: {error}")

        threading.Thread(target=record, name="nscr-mcp-receipt", daemon=True).start()

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
    ) -> Any:
        """The whole of one call, on whichever thread it was given to.

        In a session with a user interface that thread is the main thread, for
        a read as much as for a mutation. The undo group is still only for a
        tool that changes the scene.
        """
        try:
            # The last look at the scene, here on the thread that is about to
            # touch it. A call can wait a long time for its turn, and the
            # session may have loaded another scene while it waited: the paths
            # in its arguments would then mean something else entirely.
            self._still_the_same_scene(carried)
            if not tool.mutating or self._hou is None:
                return tool.run(arguments, context)
            outcome = run_in_undo_group(
                lambda: tool.run(arguments, context),
                label=tool.undo_label(),
                hou=self._hou,
            )
            running.recorded = outcome.recorded
            running.rolled_back = outcome.rolled_back
            if outcome.error is not None:
                raise outcome.error
            return outcome.value
        finally:
            self._release(running)

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

    def _release(self, running: Running) -> None:
        """Note how the call ended and give the session to the next caller."""
        if self._running is running:
            self._last = {**running.as_dict(), "finished_at": time.time()}
            self._running = None
        self._gate.leave()

    # Section: answers

    def _answer(self, tool: Tool, work: marshal.Work, running: Running, trace: dict) -> Reply:
        timing_ms = 0.0
        if work.started_at is not None and work.finished_at is not None:
            timing_ms = (work.finished_at - work.started_at) * 1000.0

        if work.error is not None:
            return self._failed(tool, work.error, running, trace)

        converted = encoding.convert(work.result)
        payload = {**ok_payload(converted.value, timing_ms=timing_ms), **self._said(trace)}
        if work.picked_by is not None:
            # Which route to the main thread reached the work first, so a check
            # against a real Houdini can tell the two apart.
            payload["picked_by"] = work.picked_by
        if converted.lossy:
            payload["lossy"] = True
            payload["cut"] = converted.cut
        if tool.mutating:
            payload["undo"] = {
                "label": tool.undo_label(),
                "recorded": running.recorded,
                "rolled_back": running.rolled_back,
            }
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
        if tool.mutating and running is not None:
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
    """
    try:
        return receipt_module.digest_call(tool.name, arguments)
    except Exception:  # noqa: BLE001 - no receipt is better than a wrong one
        return None
