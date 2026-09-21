"""The bridge that runs inside a Houdini process.

It starts a web server on loopback, mints a token, registers the session in
the coordination store and in a private file, keeps a heartbeat going, and
takes all of that down again when the process quits.

Two endpoints:

- `mcp.health` answers from values held in memory. It reads no scene, touches
  no `hou`, takes no lock and opens no file, so it still answers while the
  session is busy.
- `mcp.call` takes one request envelope and dispatches it to a tool, one call
  at a time in the whole process.

One at a time is not a preference. Two handler threads working the object
model at once was measured wedging the process for good: no exception, no
crash, full CPU and no answers ever again. So every tool runs under one
process wide lock, and nothing here offers a way around it.

For the same reason, nothing drives a bridge from inside its own process. The
caller is always another process.

What is deliberately not here: the queue with its own ordering, the wait and
timeout policy, the undo group, the receipt table and the full error code
table. Those sit on top of this dispatch point.
"""

from __future__ import annotations

import atexit
import os
import re
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import host, registry
from nscr_houdini_mcp.bridge.envelope import (
    TOKEN_HEADER,
    EnvelopeError,
    Reply,
    error_payload,
    ok_payload,
    parse_envelope,
)
from nscr_houdini_mcp.bridge.handlers import ToolRegistry, UnknownTool, default_registry
from nscr_houdini_mcp.bridge.net import (
    DEFAULT_PORT_RANGE,
    LOOPBACK,
    pick_port,
    port_is_free,
    reachable_from_outside,
)
from nscr_houdini_mcp.bridge.security import (
    browser_header,
    mint_token,
    token_matches,
)
from nscr_houdini_mcp.bridge.serving import Backend, HwebserverBackend

SESSION_ID_BYTES = 16

DEFAULT_HEARTBEAT_S = 10.0

# How long a call may wait for the one at a time lock before it is turned away.
DEFAULT_DISPATCH_WAIT_S = 30.0

# How many ports to try when the server refuses the one it was handed. The
# range is walked here because the run call has no port range argument of its
# own, and because a server object that has run once cannot run again.
START_ATTEMPTS = 5

# One process, one Houdini, one call at a time. Module level rather than per
# bridge, because the object model is shared by everything in the process.
_HOUDINI_LOCK = threading.Lock()


def houdini_lock() -> threading.Lock:
    """The lock every Houdini touching call is taken under."""
    return _HOUDINI_LOCK


# Alias templates. A worker is addressed by number, a session with a scene by
# the scene name, and a scene with no name yet by a plain word.
WORKER_ALIAS = "w{n}"
UNTITLED_ALIAS = "scene-{n}"

ALIAS_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class BridgeError(Exception):
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
    dispatch_wait_s: float = DEFAULT_DISPATCH_WAIT_S
    server_name: str = "nscr_mcp_bridge"
    in_background: bool = True
    verify_loopback: bool = True
    facts: dict[str, Any] = field(default_factory=dict)


class Bridge:
    """One bridge in one Houdini process."""

    def __init__(
        self,
        config: BridgeConfig | None = None,
        *,
        backend: Backend | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.config = config or BridgeConfig()
        self.home = Path(self.config.home) if self.config.home else store_module.default_home()
        self.store_path = (
            Path(self.config.store_path)
            if self.config.store_path
            else self.home / store_module.STORE_FILE_NAME
        )
        self.tools = tools if tools is not None else default_registry()
        self.kind = self.config.kind or host.session_kind()
        self.facts = dict(self.config.facts) if self.config.facts else host.describe()
        self.pid = os.getpid()
        self.session_id = secrets.token_hex(SESSION_ID_BYTES)
        self.scene_epoch = 0
        self.alias: str | None = None
        self.port: int | None = None
        self.started_at: float | None = None

        self._token = mint_token()
        self._backend = backend
        self._own_backend = backend is None
        self._lock = threading.Lock()
        self._running = False
        self._current_tool: str | None = None
        self._busy_since: float | None = None
        self._heartbeat_at = 0.0
        self._heartbeat_stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        self._remove_quit_hook = None
        self.problems: list[str] = []

    # -- lifetime ---------------------------------------------------------

    def start(self) -> store_module.SessionRecord:
        """Start the server, prove it is private, then announce the session."""
        with self._lock:
            if self._running:
                raise BridgeError("this bridge is already running")
            backend, port = self._listen()
            self._backend = backend
            self.port = port

            if self.config.verify_loopback:
                reachable = reachable_from_outside(port)
                if reachable:
                    backend.stop()
                    self.port = None
                    raise BridgeError(
                        f"port {port} answered on {', '.join(reachable)}, not loopback alone"
                    )

            self.started_at = time.time()
            self._heartbeat_at = self.started_at
            try:
                record = self._announce(port)
            except BaseException:
                backend.stop()
                self.port = None
                raise

            self._running = True

        self._heartbeat = threading.Thread(
            target=self._beat, name="nscr-mcp-heartbeat", daemon=True
        )
        self._heartbeat.start()
        atexit.register(self.stop)
        self._remove_quit_hook = host.install_quit_hook(self.stop)
        return record

    def stop(self) -> None:
        """Take the session out of the store, the file off disk, the port down.

        Safe to call more than once, and safe to call while the process is
        already quitting, which is where it usually runs.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._heartbeat_stop.set()
        if self._remove_quit_hook is not None:
            self._remove_quit_hook()
            self._remove_quit_hook = None
        atexit.unregister(self.stop)

        try:
            with self._open_store() as store:
                store.end_session(self.session_id)
        except Exception as error:  # noqa: BLE001 - quitting must not raise
            self._note(f"could not end the session row: {error}")
        try:
            registry.remove_entry(self.home, self.session_id)
        except OSError as error:
            self._note(f"could not remove the session file: {error}")
        if self._backend is not None:
            try:
                self._backend.stop()
            except Exception as error:  # noqa: BLE001 - quitting must not raise
                self._note(f"could not stop the server: {error}")
        self.port = None

    def __enter__(self) -> Bridge:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._running

    # -- endpoints --------------------------------------------------------

    def health(self, headers: Mapping[str, str], payload: Mapping[str, Any]) -> Reply:
        """Liveness, from memory. No scene, no `hou`, no disk."""
        refused = self._refuse(headers, payload)
        if refused is not None:
            return refused
        now = time.time()
        return Reply(
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
                    "started_at": self.started_at,
                    "heartbeat_age_s": round(max(0.0, now - self._heartbeat_at), 3),
                    "busy": self._busy_since is not None,
                    "current_op": self._current_tool,
                    "current_op_elapsed_s": self._elapsed_s(),
                    "tools": self.tools.names(),
                }
            ),
        )

    def call(self, headers: Mapping[str, str], payload: Mapping[str, Any]) -> Reply:
        """Dispatch one request envelope to one tool."""
        refused = self._refuse(headers, payload)
        if refused is not None:
            return refused
        try:
            envelope = parse_envelope(payload)
        except EnvelopeError as error:
            return Reply(400, error_payload(error.code, str(error), details=error.details))

        trace = {"operation_id": envelope.operation_id, "scene_epoch": envelope.scene_epoch}
        if envelope.session_id is not None and envelope.session_id != self.session_id:
            return Reply(
                200,
                {
                    **error_payload(
                        "SESSION_UNKNOWN",
                        "this bridge is a different session",
                        hint="read the session id from mcp.health and call again",
                        details={"session_id": self.session_id},
                    ),
                    **trace,
                },
            )
        try:
            handler = self.tools.get(envelope.tool)
        except UnknownTool:
            return Reply(
                200,
                {
                    **error_payload(
                        "TOOL_UNKNOWN",
                        f"no tool named {envelope.tool}",
                        details={"tools": self.tools.names()},
                    ),
                    **trace,
                },
            )

        waited = time.perf_counter()
        if not _HOUDINI_LOCK.acquire(timeout=max(0.0, self.config.dispatch_wait_s)):
            return Reply(
                200,
                {
                    **error_payload(
                        "SESSION_BUSY",
                        "this session is running another call",
                        hint="wait for the running call to finish, then send this one again",
                        details={
                            "current_op": self._current_tool,
                            "elapsed_s": self._elapsed_s(),
                            "waited_s": round(time.perf_counter() - waited, 3),
                        },
                    ),
                    **trace,
                },
            )
        began = time.perf_counter()
        self._current_tool = envelope.tool
        self._busy_since = time.time()
        try:
            data = handler(envelope.arguments)
        except Exception as error:  # noqa: BLE001 - one failed tool, not a failed bridge
            return Reply(
                200,
                {
                    **error_payload(
                        "TOOL_FAILED",
                        f"{type(error).__name__}: {error}",
                        details={"tool": envelope.tool},
                    ),
                    **trace,
                },
            )
        finally:
            self._current_tool = None
            self._busy_since = None
            _HOUDINI_LOCK.release()
        timing_ms = (time.perf_counter() - began) * 1000.0
        return Reply(200, {**ok_payload(data, timing_ms=timing_ms), **trace})

    # -- internals --------------------------------------------------------

    def _listen(self) -> tuple[Backend, int]:
        """Take a port in the range, walking it when the server refuses one.

        The run call has no port range argument, and a server object that has
        run once can never run again, so a refused port means a fresh server
        object and the next free port. A backend handed in from outside is
        used as it is: whoever passed it owns it.
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
            backend = self._backend or HwebserverBackend(
                f"{self.config.server_name}_{attempt}" if attempt else self.config.server_name
            )
            backend.configure(address=self.config.address, port=wanted, max_port=end_port)
            backend.register("health", self.health)
            backend.register("call", self.call)
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
            raise BridgeError(f"the server took port {port}, outside {self.config.port_range}")
        raise BridgeError(f"no port in {self.config.port_range} could be served: {last}")

    def _refuse(self, headers: Mapping[str, str], payload: Mapping[str, Any]) -> Reply | None:
        """Turn away a browser or an unauthenticated caller, or let it through.

        The transport authenticates nobody, so this runs at the top of every
        handler. The refusal bodies say nothing a caller could learn from.
        """
        offender = browser_header(headers)
        if offender is not None:
            return Reply(403, error_payload("FORBIDDEN", f"requests with {offender} are refused"))
        presented: Any = headers.get(TOKEN_HEADER)
        if presented is None and isinstance(payload, Mapping):
            presented = payload.get("token")
        if not token_matches(presented, self._token):
            return Reply(401, error_payload("UNAUTHORIZED", "token missing or wrong"))
        return None

    def _announce(self, port: int) -> store_module.SessionRecord:
        """Take an alias, write the session row, then the private file."""
        with self._open_store() as store:
            record = store.register_session(
                self.session_id,
                kind=self.kind,
                pid=self.pid,
                alias=self.config.alias,
                alias_template=None if self.config.alias else self._alias_template(),
                port=port,
                hip_path=self.facts.get("hip_path"),
                scene_epoch=self.scene_epoch,
                capabilities={
                    "houdini_version": self.facts.get("houdini_version"),
                    "hfs": self.facts.get("hfs"),
                },
            )
        self.alias = record.alias
        registry.write_entry(
            self.home,
            {
                "session_id": self.session_id,
                "alias": record.alias,
                "kind": self.kind,
                "pid": self.pid,
                "port": port,
                "address": self.config.address,
                "token": self._token,
                "scene_epoch": self.scene_epoch,
                "started_at": self.started_at,
                **self.facts,
            },
        )
        return record

    def _alias_template(self) -> str:
        if self.config.alias_template:
            return self.config.alias_template
        if self.kind != host.GUI:
            return WORKER_ALIAS
        hip = self.facts.get("hip_path")
        stem = ALIAS_SAFE.sub("-", Path(hip).stem).strip("-") if hip else ""
        return f"{stem}-{{n}}" if stem else UNTITLED_ALIAS

    def _open_store(self) -> store_module.Store:
        """A store handle for this thread. Handles are never shared."""
        return store_module.Store(self.store_path)

    def _beat(self) -> None:
        """Write a heartbeat until the bridge stops."""
        interval = max(1.0, self.config.heartbeat_s)
        while not self._heartbeat_stop.wait(interval):
            try:
                with self._open_store() as store:
                    self._heartbeat_at = store.touch_session(self.session_id)
            except Exception as error:  # noqa: BLE001 - a missed beat is not a crash
                self._note(f"heartbeat: {error}")

    def _elapsed_s(self) -> float | None:
        """How long the running call has been running, read without a lock."""
        since = self._busy_since
        return None if since is None else round(max(0.0, time.time() - since), 3)

    def _note(self, problem: str) -> None:
        """Keep the last few problems where a status call can find them."""
        self.problems.append(problem)
        del self.problems[:-10]
