"""The bridge that runs inside a Houdini process.

It starts a web server on loopback, mints a token, registers the session in
the coordination store and in a private file, keeps a heartbeat going, and
takes all of that down again when the process quits.

Two endpoints, both on paths of the bridge's own:

- `/nscr-mcp/health` answers from values held in memory. It reads no scene,
  touches no `hou`, takes no lock and opens no file, so it still answers while
  the session is busy.
- `/nscr-mcp/call` takes one request envelope and hands it to the dispatcher,
  which runs one call at a time in the whole process.

What this protects against, and what it does not. Anything running as the
same person on the same machine is already inside: it can read that person's
files, including the token file, and it could drive Houdini without this
bridge at all. So the checks below are not a wall against the owner's own
programs. They are a wall against a web page in a browser on this machine,
against another person with an account on it, and against the network. The
signing on top of that is for the case where this Houdini has crashed and
something else has taken its port: a caller sends nothing reusable and can
tell it is not talking to the bridge.

What every request meets, in this order, before anything is parsed:

1. `Origin` or `Referer` present: refused. Only a browser sends those.
2. The address the request arrived on must be loopback. Anything else and the
   bridge refuses the request and shuts itself down.
3. `Host` must name this bridge's own loopback address and port.
4. The body must be JSON, and no larger than the configured cap.
5. The request must carry a signature made with this session's token.
6. Only then is the body decoded, with a nesting limit.

One call at a time is not a preference. Two handler threads working the object
model at once ends the process for good: no exception, no crash, full CPU and
no answers ever again. So every tool runs under one process wide lock, and
nothing here offers a way around it.

In a session with a user interface every tool runs on the process main thread,
reads included. Any `hou` call from another thread takes Houdini's object model
lock, which the main thread holds for the whole of a cook, so a request thread
that called into `hou` would be stuck there for as long as the cook lasts. The
health endpoint stays answerable because it touches none of it.

For the same reason, nothing drives a bridge from inside its own process. The
caller is always another process.

The order calls are served in, the wait and timeout budgets, the undo group,
the scene guard, the receipts for repeated operation ids and the error codes
belong to the dispatcher. Who this session is and which scene it is holding
belong to the identity.
"""

from __future__ import annotations

import atexit
import os
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import (
    client,
    host,
    liveness,
    marshal,
    registry,
    signing,
)
from nscr_houdini_mcp.bridge import (
    receipts as receipt_module,
)
from nscr_houdini_mcp.bridge.dispatch import DEFAULT_TIMEOUT_S, DEFAULT_WAIT_S, Dispatcher
from nscr_houdini_mcp.bridge.envelope import (
    MAX_DEPTH,
    Envelope,
    EnvelopeError,
    Reply,
    error_payload,
    load_json,
    ok_payload,
    parse_envelope,
)
from nscr_houdini_mcp.bridge.handlers import ToolRegistry, default_registry
from nscr_houdini_mcp.bridge.identity import Identity, alias_template
from nscr_houdini_mcp.bridge.net import (
    DEFAULT_PORT_RANGE,
    LOOPBACK,
    is_loopback,
    pick_port,
    port_is_free,
    prove_loopback_only,
)
from nscr_houdini_mcp.bridge.security import (
    browser_header,
    check_home,
    host_allowed,
    mint_token,
)
from nscr_houdini_mcp.bridge.serving import (
    CALL_PATH,
    HEALTH_PATH,
    JSON_TYPE,
    STDLIB,
    Backend,
    RawReply,
    RawRequest,
    make_backend,
)

SESSION_ID_BYTES = 16

DEFAULT_HEARTBEAT_S = 10.0

# How long a call waits for its turn when it names no wait of its own, and how
# long it waits for work that is already running. A call may ask for less, or
# for none at all.
DEFAULT_DISPATCH_WAIT_S = DEFAULT_WAIT_S
DEFAULT_DISPATCH_TIMEOUT_S = DEFAULT_TIMEOUT_S

# The largest request body the bridge will read. An envelope is small; this is
# room for a long piece of code as an argument and nothing more.
DEFAULT_MAX_BODY_BYTES = 1024 * 1024

# How long a call has to notice the session is going down before a process
# that exists only to be this bridge ends itself.
DEFAULT_SHUTDOWN_GRACE_S = 5.0

# How long the session waits for its own port to answer the self check. Health
# is answered from memory, so anything near this means the port is not being
# served at all.
DEFAULT_SELF_CHECK_TIMEOUT_S = 5.0

# Set in a worker this project starts. The self check tool makes nodes and can
# park a session for a minute, so it is off unless a process was started to be
# tested against, and off in a session with a user interface either way.
SELFCHECK_ENV_VAR = "NSCR_MCP_SELFCHECK"

# How many ports to try when the server refuses the one it was handed. The
# range is walked here because the run call has no port range argument of its
# own, and because a server object that has run once cannot run again.
START_ATTEMPTS = 5

# One process, one Houdini, one call at a time. Module level rather than per
# bridge, because the object model is shared by everything in the process.
_HOUDINI_LOCK = threading.Lock()


def selfcheck_wanted(kind: str) -> bool:
    """Whether this session should carry the self check tool."""
    if kind == host.GUI:
        return False
    return os.environ.get(SELFCHECK_ENV_VAR, "").strip().lower() not in ("", "0", "false", "no")


def houdini_lock() -> threading.Lock:
    """The lock every Houdini touching call is taken under."""
    return _HOUDINI_LOCK


LOG_DIR_NAME = "logs"

# The self check argument that holds an answer back until the caller has given
# up on it, so the lost reply case can be tried end to end. It is only ever
# read in a session that carries the self check tool, which is a worker this
# project started to be driven.
DROP_REPLY_ARG = "drop_reply"

# How long such an answer is held. Longer than any sensible client read
# budget, and short enough that the thread is not parked for the session's
# whole life.
DEFAULT_DROP_REPLY_S = 30.0


class BridgeStartError(Exception):
    """The bridge could not start, or could not start safely."""


@dataclass(frozen=True)
class BridgeConfig:
    """Everything a bridge needs to know before it starts."""

    home: Path | None = None
    store_path: Path | None = None
    port_range: tuple[int, int] = DEFAULT_PORT_RANGE
    address: str = LOOPBACK
    alias: str | None = None
    alias_template: str | None = None
    kind: str | None = None
    heartbeat_s: float = DEFAULT_HEARTBEAT_S
    # How long the main thread may go without running our code before a call
    # that will not wait that long is refused at once.
    main_thread_stale_s: float = marshal.DEFAULT_STALE_S
    dispatch_wait_s: float = DEFAULT_DISPATCH_WAIT_S
    dispatch_timeout_s: float = DEFAULT_DISPATCH_TIMEOUT_S
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_depth: int = MAX_DEPTH
    # Which web server answers the port. The standard library one serves
    # several client connections at once; Houdini's own is the fallback.
    transport: str = STDLIB
    server_name: str = "nscr_mcp_bridge"
    in_background: bool = True
    verify_loopback: bool = True
    # Whether this process exists to be this bridge. When it does, a call that
    # will not stop does not get to keep the process alive for ever.
    owns_process: bool = False
    shutdown_grace_s: float = DEFAULT_SHUTDOWN_GRACE_S
    # How long the session's own health request may take before the port
    # counts as not answering.
    self_check_timeout_s: float = DEFAULT_SELF_CHECK_TIMEOUT_S
    # How long an answer is held when a call asks for it to be dropped.
    drop_reply_s: float = DEFAULT_DROP_REPLY_S
    facts: dict[str, Any] = field(default_factory=dict)


class Bridge:
    """One bridge in one Houdini process."""

    def __init__(
        self,
        config: BridgeConfig | None = None,
        *,
        backend: Backend | None = None,
        tools: ToolRegistry | None = None,
        hou: Any | None = None,
    ) -> None:
        self.config = config or BridgeConfig()
        # Handed in by the tests, read from the process otherwise.
        self._hou = hou if hou is not None else host.houdini()
        self.home = Path(self.config.home) if self.config.home else store_module.default_home()
        check_home(self.home)
        self.store_path = (
            Path(self.config.store_path)
            if self.config.store_path
            else self.home / store_module.STORE_FILE_NAME
        )
        self.kind_for_tools = self.config.kind or host.session_kind()
        self.tools = (
            tools
            if tools is not None
            else default_registry(selfcheck=selfcheck_wanted(self.kind_for_tools))
        )
        self.kind = self.kind_for_tools
        self.facts = dict(self.config.facts) if self.config.facts else host.describe()
        self.pid = os.getpid()
        self.pid_start = liveness.process_start_stamp()
        self.session_id = secrets.token_hex(SESSION_ID_BYTES)
        # The name is settled once, at start, and never changes afterwards.
        # A scene saved under another name makes the name out of date, which
        # every reply says, rather than moving it under a caller's feet.
        self.identity = Identity(
            session_id=self.session_id,
            kind=self.kind,
            hip_path=self.facts.get("hip_path"),
            tracks_hip=self.kind == host.GUI and not self.config.alias,
            hou=self._hou,
            on_change=self._scene_replaced,
            log=self._log,
        )
        self.port: int | None = None
        self.started_at: float | None = None
        self.privacy: dict[str, Any] | None = None
        # What the last request this session sent to its own port found.
        # Nothing until the first one has been sent.
        self.transport_ok: bool | None = None
        self.transport_checked_at: float | None = None

        self._token = mint_token()
        self._verifier = signing.Verifier(self._token, self.session_id)
        self._backend = backend
        self._own_backend = backend is None
        self._lock = threading.Lock()
        self._running = False
        # Set while the bridge is going down, so a tool that looks at it can
        # stop early instead of holding the process open.
        self.stopping = threading.Event()
        # The thread that owns the process runs this while the bridge is up,
        # because a scene edit has to happen on the main thread to be one
        # undo step. A bridge whose owner never runs it still works, and says
        # in every mutating reply that nothing was recorded.
        self.main_loop = marshal.MainLoop()
        # With a user interface the main thread is Houdini's own, and it is
        # reached through a queue and one poster thread rather than by calling
        # into `hou` from whichever thread took the request.
        self.pulse = marshal.Pulse(stale_s=self.config.main_thread_stale_s, log=self._log)
        self.main_thread = (
            marshal.MainThreadRunner(self._hou, pulse=self.pulse, log=self._log)
            if self.kind == host.GUI and self._hou is not None
            else None
        )
        self.dispatcher = Dispatcher(
            self.tools,
            lock=_HOUDINI_LOCK,
            kind=self.kind,
            session_id=self.session_id,
            identity=self.identity,
            receipts=receipt_module.Receipts(
                self._open_store, session_id=self.session_id, owner_pid=self.pid, log=self._log
            ),
            hou=self._hou,
            log=self._log,
            main_loop=self.main_loop,
            main_thread=self.main_thread,
            pulse=self.pulse,
            stopping=self.stopping,
            wait_s=self.config.dispatch_wait_s,
            timeout_s=self.config.dispatch_timeout_s,
        )
        self._heartbeat_at = 0.0
        self._heartbeat_stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        self._quit_hook = host.QuitHook(self.stop, hou=self._hou)
        self._entry: dict[str, Any] = {}
        # One writer at a time for the session file, so two threads updating
        # different parts of it cannot land one on top of the other.
        self._entry_lock = threading.Lock()
        # One self check at a time, so the answer and the file move together.
        self._transport_lock = threading.Lock()
        self.problems: list[str] = []

    # Section: identity

    @property
    def alias(self) -> str | None:
        """The readable name this session was given when it started."""
        return self.identity.alias

    @property
    def scene_epoch(self) -> int:
        """How many times this process has replaced its scene."""
        return self.identity.scene_epoch

    def _scene_replaced(self, epoch: int, hip_path: str | None) -> None:
        """Write the new epoch where other processes read it.

        Called from Houdini's own scene event, on the main thread, so it does
        the least it can: one store row and one file.
        """
        self._log(f"the scene was replaced, epoch {epoch}")
        with self._open_store() as store:
            store.set_scene_epoch(self.session_id, epoch, hip_path=hip_path)
        self._write_entry(scene_epoch=epoch, hip_path=hip_path)

    # Lifetime

    def start(self) -> store_module.SessionRecord:
        """Start the server, prove it is private, then announce the session."""
        with self._lock:
            if self._running:
                raise BridgeStartError("this bridge is already running")
            backend, port = self._listen()
            self._backend = backend
            self.port = port

            proof = prove_loopback_only(port)
            self.privacy = proof.as_dict()
            if self.config.verify_loopback and not proof.private:
                backend.stop()
                self.port = None
                raise BridgeStartError(
                    f"port {port} is held or answered on {', '.join(proof.reachable)},"
                    " not loopback alone"
                )
            if not proof.proven:
                self._note(f"the port could not be proven private: {proof.note}")

            self.started_at = time.time()
            self._heartbeat_at = self.started_at
            try:
                record = self._announce(port)
            except BaseException:
                # A start that did not finish leaves nothing behind: no token
                # file, no session row, no listening port.
                self._undo_announce()
                _try(backend.stop)
                self.port = None
                self.started_at = None
                raise

            self._running = True

        if self._hou is not None:
            # Callbacks an earlier bridge in this process left to come off go
            # now, so they are never registered next to this bridge's own.
            # Like the steps below this calls into `hou`, which belongs here on
            # the thread that starts the bridge while the session is idle.
            _try(lambda: host.clear_leftovers(self._hou, log=self._log))

        if self.main_thread is not None:
            # Both steps call into `hou`, so they belong here on the thread
            # that starts the bridge while the session is idle, never on a
            # thread answering a request. A pulse that could not be installed
            # is noted and the bridge still starts: every call is then bounded
            # by its pickup budget alone.
            failure = _try(lambda: self.pulse.install(self._hou))
            if failure is not None:
                self._note(f"could not start watching the main thread: {failure}")
            self.main_thread.start()

        self._heartbeat = threading.Thread(
            target=self._beat, name="nscr-mcp-heartbeat", daemon=True
        )
        self._heartbeat.start()
        atexit.register(self.stop)
        self._quit_hook.install()
        # From here the session follows its own scene: the first summary is
        # taken now, and the epoch moves whenever the scene is replaced.
        self.identity.refresh()
        self.identity.watch()
        return record

    def stop(self) -> list[str]:
        """Take the session out of the store, the file off disk, the port down.

        Every step runs whatever the ones before it did, because a step that
        fails must not leave the rest undone: a token file nobody clears is
        worse than a port nobody closes. Whatever went wrong is collected and
        handed back, and written to the session log. Stopping never raises,
        because it usually runs while the process is already quitting.

        Safe to call more than once.
        """
        with self._lock:
            if not self._running:
                return []
            self._running = False

        problems: list[str] = []
        self.stopping.set()
        self._heartbeat_stop.set()
        self._leave_anyway()
        # The scene callbacks are told to go first, so the main thread's next
        # visit, from the pulse or from the last post, finds them waiting.
        for what, step in (
            ("stop following the scene", self.identity.unwatch),
            ("take the quit hook off", self._quit_hook.remove),
            ("stop posting to the main thread", self._stop_posting),
            ("stop watching the main thread", self.pulse.uninstall),
            ("stop being called at exit", lambda: atexit.unregister(self.stop)),
            ("end the session row", self._end_session_row),
            ("remove the session file", self._remove_session_file),
            ("stop the server", self._stop_backend),
            ("take our callbacks off", self._take_callbacks_off),
        ):
            if step is None:
                continue
            failure = _try(step)
            if failure is not None:
                problems.append(f"could not {what}: {failure}")
        self.port = None
        for problem in problems:
            self._log(problem)
        return problems

    def _stop_posting(self) -> None:
        """Close the runner, posting the removal of our callbacks on the way out.

        That post and the pulse's last tick both reach the main thread, and
        whichever lands first takes the callbacks off. Neither waits for a
        scene event.
        """
        if self.main_thread is None:
            return
        hou, log = self._hou, self._log
        self.main_thread.stop(last=lambda: host.clear_leftovers(hou, log=log))

    def _take_callbacks_off(self) -> None:
        """Without a user interface, take our callbacks off here and now.

        No event loop holds the object model lock in such a session, so the
        call is free from this thread. With one, the main thread does it.
        """
        if self.main_thread is None and self._hou is not None:
            host.clear_leftovers(self._hou, log=self._log)

    def _leave_anyway(self) -> None:
        """End the process by force if a call will not let go of it.

        Only where the bridge is what the process is for. A call that ignores
        the stopping flag would otherwise keep a Houdini alive with nobody
        left to talk to it, which is the thing this whole design is trying not
        to leave behind. A session somebody is working in is never ended this
        way: there the call finishes and the bridge goes quiet.
        """
        if not self.config.owns_process or self.dispatcher.running is None:
            return
        self._log(
            f"a call is still running at shutdown, ending the process in "
            f"{self.config.shutdown_grace_s:g} seconds"
        )

        def end() -> None:
            time.sleep(self.config.shutdown_grace_s)
            if self.dispatcher.running is None:
                return
            self._log("the call did not stop, ending the process")
            os._exit(0)

        threading.Thread(target=end, name="nscr-mcp-shutdown", daemon=True).start()

    def _end_session_row(self) -> None:
        with self._open_store() as store:
            store.end_session(self.session_id)

    def _remove_session_file(self) -> None:
        """Forget the file's contents, then take it off disk.

        Both happen under the lock every writer takes, so a self check or a
        scene event that finishes after this finds nothing to write and cannot
        put the file of a stopped session back.
        """
        with self._entry_lock:
            self._entry = {}
            registry.remove_entry(self.home, self.session_id)

    def _stop_backend(self) -> None:
        if self._backend is not None:
            self._backend.stop()

    def _undo_announce(self) -> None:
        """Clear whatever a half finished announcement managed to write."""
        for step in (self._remove_session_file, self._end_session_row):
            _try(step)

    def __enter__(self) -> Bridge:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._running

    # Endpoints

    def handle_health(self, request: RawRequest) -> RawReply:
        """Liveness, from memory. No scene, no `hou`, no lock, no disk.

        `main_thread.pulse_age_s` is the number a caller reads to see whether
        the main thread is taking work: it is a float in memory, stamped the
        last time the main thread ran our code, and it climbs while Houdini is
        cooking. How soon an idle main thread picks work up follows Houdini's
        own event loop, and a sleeping display slows that to about 200 ms.
        `heartbeat_age_s` answers a different question, whether this bridge is
        still writing where other processes can see it.
        """
        refused = self._front(request)
        if refused is not None:
            return refused
        now = time.time()
        return self._answer(
            request,
            Reply(
                200,
                ok_payload(
                    {
                        "status": "ok",
                        "session_id": self.session_id,
                        "alias": self.alias,
                        "kind": self.kind,
                        "pid": self.pid,
                        "port": self.port,
                        "scene_epoch": self.scene_epoch,
                        "scene": self.identity.scene(),
                        "alias_drift": self.identity.drift(),
                        "started_at": self.started_at,
                        "heartbeat_age_s": round(max(0.0, now - self._heartbeat_at), 3),
                        **self.transport_state(),
                        "privacy": self.privacy,
                        "tools": self.tools.names(),
                        **self.dispatcher.state(),
                    }
                ),
            ),
        )

    def handle_call(self, request: RawRequest) -> RawReply:
        """Dispatch one request envelope to one tool.

        Nothing gets out of here uncaught. A handler that raises gives the web
        server a 500 the bridge never signed, which a caller is right to read
        as somebody else sitting on the port. A coded answer is better than
        that, whatever went wrong.
        """
        refused = self._front(request)
        if refused is not None:
            return refused
        try:
            payload = load_json(request.body, max_depth=self.config.max_depth)
            envelope = parse_envelope(payload)
        except EnvelopeError as error:
            return self._answer(
                request, Reply(400, error_payload(error.code, str(error), details=error.details))
            )
        try:
            reply = self._dispatch(envelope)
        except BaseException as error:  # noqa: BLE001 - an unsigned 500 is worse
            self._log(
                f"dispatching {envelope.tool} raised: {type(error).__name__}: {error}\n"
                + "".join(traceback.format_exception(error)[-8:])
            )
            reply = Reply(
                200,
                {
                    **error_payload(
                        "TOOL_FAILED",
                        "the bridge could not finish this call",
                        hint="the bridge log for this session has the detail",
                        details={"tool": envelope.tool, "exception": type(error).__name__},
                    ),
                    "operation_id": envelope.operation_id,
                    "scene_epoch": self.scene_epoch,
                },
            )
        if self._drop_wanted(envelope, reply):
            self._hold_back(envelope)
        return self._answer(request, reply)

    # Losing an answer on purpose

    def _drop_wanted(self, envelope: Envelope, reply: Reply) -> bool:
        """Whether this call asked for its answer to go missing.

        Only a session carrying the self check tool will do it, which is a
        worker this project started to be driven. A session somebody is
        working in never has it, so nothing a user does can reach this.

        An answer that came from a receipt is never held back: the caller that
        lost the first one is asking for that answer, and losing it again
        would leave it nowhere to go.
        """
        if "bridge.selfcheck" not in self.tools:
            return False
        if reply.payload.get("replayed"):
            return False
        return bool(envelope.arguments.get(DROP_REPLY_ARG))

    def _hold_back(self, envelope: Envelope) -> None:
        """Hold an answer until the caller has stopped waiting for it.

        This is the failure the receipts exist for: the work has run and
        changed the scene, and the caller learns nothing. It waits on the
        shutdown flag rather than sleeping, so a session told to stop is not
        held open by it.
        """
        self._log(
            f"holding back the answer to {envelope.tool} for "
            f"{self.config.drop_reply_s:g} seconds, as the call asked"
        )
        self.stopping.wait(self.config.drop_reply_s)

    # Dispatch

    def _dispatch(self, envelope: Envelope) -> Reply:
        """Check the call is for this session, then hand it to the dispatcher."""
        if envelope.session_id is not None and envelope.session_id != self.session_id:
            return Reply(
                200,
                {
                    **error_payload(
                        "UNKNOWN_SESSION",
                        "this bridge is a different session",
                        hint="read the session id from the health endpoint and call again",
                        details={"session_id": self.session_id},
                    ),
                    "operation_id": envelope.operation_id,
                    "scene_epoch": self.scene_epoch,
                },
            )
        return self.dispatcher.dispatch(envelope)

    # The front of every request

    def _front(self, request: RawRequest) -> RawReply | None:
        """Turn a request away, or let it through to be read.

        Nothing is parsed here. The refusal bodies say only that the request
        was refused; the reason goes to the local log.
        """
        headers = request.headers
        offender = browser_header(headers)
        if offender is not None:
            return self._refuse(request, 403, "FORBIDDEN", f"requests with {offender} are refused")

        if request.server_address is not None and not is_loopback(request.server_address):
            # The bind did not do what it was told. Refuse, then close the
            # door rather than keep serving the network.
            self._note(f"a request arrived on {request.server_address}, which is not loopback")
            threading.Thread(target=self.stop, name="nscr-mcp-close", daemon=True).start()
            return self._refuse(request, 403, "FORBIDDEN", "this port is for loopback only")

        if self.port is not None and not host_allowed(headers, self.port):
            return self._refuse(request, 403, "FORBIDDEN", "the host is not this bridge")

        if request.method.upper() != "POST":
            return self._refuse(request, 405, "METHOD_REFUSED", "this endpoint takes POST")

        kind = (request.content_type or headers.get("content-type", "")).split(";")[0].strip()
        if kind.lower() != JSON_TYPE:
            return self._refuse(
                request, 415, "BODY_REFUSED", f"this endpoint takes {JSON_TYPE} only"
            )

        if len(request.body) > self.config.max_body_bytes:
            return self._refuse(
                request,
                413,
                "BODY_REFUSED",
                f"the body is larger than {self.config.max_body_bytes} bytes",
            )

        try:
            self._verifier.check(
                headers, method=request.method, path=request.path, body=request.body
            )
        except signing.SignatureRefused as refused:
            self._log(f"refused a request to {request.path}: {refused.reason}")
            return self._refuse(request, 401, "UNAUTHORIZED", "the request was not signed for me")
        return None

    def _refuse(self, request: RawRequest, status: int, code: str, message: str) -> RawReply:
        return self._answer(request, Reply(status, error_payload(code, message)))

    def _answer(self, request: RawRequest, reply: Reply) -> RawReply:
        """Sign an answer, so a caller can tell this bridge from a squatter."""
        raw = RawReply.of(reply)
        nonce = self._verifier.nonce_of(request.headers)
        return RawReply(
            raw.status,
            raw.body,
            {signing.SIGNATURE_HEADER: self._verifier.sign_answer(nonce, raw.status, raw.body)},
        )

    # Internals

    def _listen(self) -> tuple[Backend, int]:
        """Take a port in the range, walking it when the server refuses one.

        Houdini's own server has no port range argument on its run call, and a
        server object that has run once can never run again, so a refused port
        means a fresh server object and the next free port. The standard
        library one walks the range itself and takes one attempt. A backend
        handed in from outside is used as it is: whoever passed it owns it.
        """
        start_port, end_port = self.config.port_range
        taken: set[int] = set()
        last: Exception | None = None
        attempts = START_ATTEMPTS if self._own_backend else 1
        for attempt in range(attempts):
            wanted = pick_port(
                (start_port, end_port),
                is_free=lambda port: port not in taken and port_is_free(port),
            )
            taken.add(wanted)
            backend = self._backend or make_backend(
                self.config.transport,
                f"{self.config.server_name}_{attempt}" if attempt else self.config.server_name,
            )
            backend.configure(address=self.config.address, port=wanted, max_port=end_port)
            backend.set_max_body(self.config.max_body_bytes)
            backend.register(HEALTH_PATH, self.handle_health)
            backend.register(CALL_PATH, self.handle_call)
            try:
                port = backend.start(wanted, in_background=self.config.in_background)
            except Exception as error:  # noqa: BLE001 - a busy port is not a failed bridge
                last = error
                self._note(f"port {wanted}: {error}")
                if not self._own_backend:
                    raise
                continue
            if start_port <= port <= end_port:
                return backend, port
            backend.stop()
            raise BridgeStartError(f"the server took port {port}, outside {self.config.port_range}")
        raise BridgeStartError(f"no port in {self.config.port_range} could be served: {last}")

    def _announce(self, port: int) -> store_module.SessionRecord:
        """Take an alias, write the session row, then the private file."""
        with self._open_store() as store:
            record = store.register_session(
                self.session_id,
                kind=self.kind,
                pid=self.pid,
                pid_start=self.pid_start,
                alias=self.config.alias,
                alias_template=None if self.config.alias else self._alias_template(),
                port=port,
                hip_path=self.facts.get("hip_path"),
                scene_epoch=self.scene_epoch,
                capabilities={
                    "houdini_version": self.facts.get("houdini_version"),
                    "hfs": self.facts.get("hfs"),
                    "privacy": self.privacy,
                },
            )
        self.identity.settle_alias(record.alias)
        self._entry = {
            "session_id": self.session_id,
            "alias": record.alias,
            "kind": self.kind,
            "pid": self.pid,
            "pid_start": self.pid_start,
            "port": port,
            "address": self.config.address,
            "health_path": HEALTH_PATH,
            "call_path": CALL_PATH,
            "token": self._token,
            "scene_epoch": self.scene_epoch,
            "started_at": self.started_at,
            **self.transport_state(),
            **self.facts,
        }
        registry.write_entry(self.home, self._entry)
        return record

    def _alias_template(self) -> str:
        if self.config.alias_template:
            return self.config.alias_template
        return alias_template(self.kind, self.facts.get("hip_path"))

    def _open_store(self) -> store_module.Store:
        """A store handle for this thread. Handles are never shared."""
        return store_module.Store(self.store_path)

    def _beat(self) -> None:
        """Prove the port answers, then write a heartbeat, until the bridge stops.

        The first round runs at once, so a session says something true about
        its own port from the moment it is up rather than one interval later.
        """
        interval = max(1.0, self.config.heartbeat_s)
        self._round()
        while not self._heartbeat_stop.wait(interval):
            self._round()

    def _round(self) -> None:
        """One self check and one heartbeat, neither able to stop the other."""
        if self._heartbeat_stop.is_set():
            return
        self.check_transport()
        try:
            with self._open_store() as store:
                self._heartbeat_at = store.touch_session(
                    self.session_id,
                    state=store_module.SESSION_LIVE
                    if self.transport_ok is not False
                    else store_module.SESSION_UNRESPONSIVE,
                    transport_ok=self.transport_ok,
                    transport_checked_at=self.transport_checked_at,
                )
        except Exception as error:  # noqa: BLE001 - a missed beat is not a crash
            self._note(f"heartbeat: {error}")

    # Section: proving the port answers

    def check_transport(self) -> bool | None:
        """Send one signed request to this session's own health endpoint.

        A bridge can be alive, writing heartbeats and answering from memory
        while nothing can reach its port: the thread that serves it can be
        gone without the process noticing. Health read from inside the process
        would say everything is fine, so this asks the way a caller would,
        over the socket, and records what came back.

        Nothing here touches `hou`, and the health endpoint reads no scene, so
        this answers while the session is busy with a call.
        """
        if not self._running or self.port is None:
            return None
        session = client.Session(
            session_id=self.session_id,
            token=self._token,
            port=self.port,
            address=self.config.address,
        )
        ok = False
        try:
            answer = client.health(session, timeout_s=self.config.self_check_timeout_s)
            ok = answer.status == 200 and bool(answer.payload.get("ok"))
        except (client.BridgeUnreachable, client.BridgeNotAuthentic) as error:
            self._log(f"this session's own port did not answer: {error}")
        except Exception as error:  # noqa: BLE001 - a failed check is a failed check
            self._log(f"could not check this session's own port: {type(error).__name__}: {error}")
        at = time.time()
        # The file is written before the answer is published, so a reader that
        # has seen this session say its port answers finds the same in the
        # file. Two checks at once are kept apart for the same reason.
        with self._transport_lock:
            was = self.transport_ok
            if was is True and not ok:
                self._log("this session stopped answering on its own port")
            elif was is False and ok:
                self._log("this session is answering on its own port again")
            if was is not ok:
                self._write_entry(**self._transport_fields(ok, at))
            self.transport_ok = ok
            self.transport_checked_at = at
        return ok

    @staticmethod
    def _transport_fields(ok: bool | None, at: float | None) -> dict[str, Any]:
        """One self check, as it is written down."""
        return {
            "last_self_check_ok": ok,
            "last_self_check_at": at,
            "last_self_check_age_s": None if at is None else round(max(0.0, time.time() - at), 3),
        }

    def transport_state(self) -> dict[str, Any]:
        """What the last self check found, for health and for the session file."""
        return self._transport_fields(self.transport_ok, self.transport_checked_at)

    def _write_entry(self, **changes: Any) -> None:
        """Write the session file again, with whatever has changed in it."""
        with self._entry_lock:
            if not self._entry:
                return
            self._entry = {**self._entry, **changes}
            try:
                registry.write_entry(self.home, self._entry)
            except Exception as error:  # noqa: BLE001 - the file is not the session
                self._note(f"could not write the session file: {error}")

    def log_path(self) -> Path:
        """Where this session's detail goes. The token is never written here."""
        return self.home / LOG_DIR_NAME / f"{self.session_id}.log"

    def _log(self, text: str) -> None:
        """Append one note to the session log, and never fail over it."""
        self._note(text.splitlines()[0] if text else "")
        try:
            path = self.log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with path.open("a", encoding="utf-8") as stream:
                stream.write(f"{stamp} {text}\n")
        except OSError:
            pass

    def _note(self, problem: str) -> None:
        """Keep the last few problems where a status call can find them."""
        self.problems.append(problem)
        del self.problems[:-10]


def _try(step: Any) -> str | None:
    """Run a cleanup step. Returns what went wrong, or nothing."""
    try:
        step()
    except Exception as error:  # noqa: BLE001 - one failed step, not a failed cleanup
        return f"{type(error).__name__}: {error}"
    return None
