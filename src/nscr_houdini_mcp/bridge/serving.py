"""The seam between the bridge and a web server.

Everything a web server does for the bridge is four calls: settings, register,
start, stop. Keeping them behind one small class means the request handling
can be tested in plain Python, and it puts the rules each server imposes in
one readable place.

Two servers sit behind that seam. The default is `StdlibBackend`, a threading
HTTP server from the standard library, because it serves several client
connections at once and keeps running. `HwebserverBackend` is Houdini's own
server, kept as a fallback; it answers one client at a time.

Neither server thread ever touches `hou`. A handler hands the work to the
bridge, which runs it on the main thread in a session with a user interface
and one call at a time everywhere.

Rules the standard library server follows:

- Bind loopback by address, and walk the port range until one is free.
- A body needs a length, and a length over the cap is refused before the body
  is read. A chunked body is refused: nothing this bridge answers sends one.
- Keep alive is on, with an idle timeout per connection, and there is a hard
  cap on how many connections are served at once. Past the cap a connection is
  told the server is busy and closed.
- The answer names no build in its `Server` header.
- Address reuse is off on Windows, where it would let another program take a
  port this one is listening on.

Rules Houdini's own server imposes, when it is the one in use:

Why a raw URL handler and not the built in API route. The API route reads the
posted form and parses its JSON field before any handler code runs, so an
unsigned request from anywhere reaches a parser the bridge does not control.
A raw handler is given the request untouched: the bridge reads the headers,
refuses what it does not want, caps the size and then parses the body itself.
No API function is registered, so that route does not exist on this server.

The rest of them:

- Build on an explicit named server object. The module level functions resolve
  through a thread local, so they would register onto, and restart, whichever
  server another tool in the same Houdini already owns.
- Register every handler before the server runs. One registered afterwards is
  unreachable.
- Set the bind address and the port range through the port settings. The run
  call takes neither.
- Set a cross origin whitelist, because an unset one reflects any origin back.
  It does not stop the request, so the handler checks the header as well.
- One server, one run, per process. A second run after a shutdown leaves
  sockets behind that nothing can reclaim.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

from nscr_houdini_mcp.bridge.envelope import Reply, error_payload
from nscr_houdini_mcp.bridge.security import normalise_headers

# The two servers a bridge can be given, by name.
STDLIB = "stdlib"
HWEBSERVER = "hwebserver"
TRANSPORTS = (STDLIB, HWEBSERVER)

# The port settings this bridge writes are its own, under its own server
# object. "main" is the name the settings and the handlers agree on.
PORT_NAME = "main"

# A non empty whitelist turns the origin reflection off. Nothing real is meant
# to match it: the bridge answers no browser.
CORS_DENY_LIST = ("http://cors.invalid",)

HEALTH_PATH = "/nscr-mcp/health"
CALL_PATH = "/nscr-mcp/call"

JSON_TYPE = "application/json"


@dataclass(frozen=True)
class RawRequest:
    """One request, before anything has been read out of its body."""

    method: str
    path: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    content_type: str = ""
    server_address: str | None = None
    client_address: str | None = None


@dataclass(frozen=True)
class RawReply:
    """One answer, with the headers the bridge adds to it."""

    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)
    content_type: str = JSON_TYPE

    @classmethod
    def of(cls, reply: Reply, headers: Mapping[str, str] | None = None) -> RawReply:
        body = json.dumps(reply.payload, default=str).encode("utf-8")
        return cls(reply.status, body, dict(headers or {}))


Endpoint = Callable[[RawRequest], RawReply]


class Backend(Protocol):
    """What the bridge needs from a web server."""

    def configure(self, *, address: str, port: int, max_port: int) -> None: ...

    def set_max_body(self, limit: int) -> None: ...

    def register(self, path: str, endpoint: Endpoint) -> None: ...

    def start(self, port: int, *, in_background: bool = True) -> int: ...

    def stop(self) -> None: ...


class HwebserverBackend:
    """Houdini's own web server, driven through one named server object.

    One client at a time; several client connections at once can end the
    process.
    """

    def __init__(self, server_name: str) -> None:
        import hwebserver

        self._hwebserver = hwebserver
        self._server = hwebserver.Server(server_name)
        self._reported_port: int | None = None
        self._max_body: int | None = None
        self._started = False

    def configure(self, *, address: str, port: int, max_port: int) -> None:
        self._server.setSettingsForPort(
            {"ADDRESS": address, "PORT": port, "MAX_PORT": max_port},
            PORT_NAME,
        )
        self._server.setCORSWhitelist(list(CORS_DENY_LIST), PORT_NAME)

    def set_max_body(self, limit: int) -> None:
        """Have the server refuse an oversized body before a handler sees it.

        The handler checks the size as well. This one saves the server from
        holding fifty megabytes in memory to be told no.
        """
        self._max_body = limit

    def register(self, path: str, endpoint: Endpoint) -> None:
        if self._started:
            raise RuntimeError("handlers must be registered before the server runs")

        @self._server.urlHandler(path)
        def handle(request):
            return self._respond(endpoint(_read(request)))

        # The decorator hands back nothing, and the handler is held by the
        # server. This keeps a name on it so a reader can see it is wired up.
        self._handlers = getattr(self, "_handlers", {})
        self._handlers[path] = handle

    def start(self, port: int, *, in_background: bool = True) -> int:
        if self._started:
            raise RuntimeError("one server, one run, per process")
        self._started = True
        extra = {} if self._max_body is None else {"max_request_size": self._max_body}
        self._server.run(
            port=port, in_background=in_background, port_callback=self._note_port, **extra
        )
        return self._reported_port or port

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._server.requestShutdown()

    def _note_port(self, *args: Any) -> None:
        """Record the port the server really took.

        It is called with the port name and the port, both as text, so the
        signature takes whatever arrives and keeps the part that is a number.
        """
        for value in args:
            try:
                self._reported_port = int(value)
            except (TypeError, ValueError):
                continue

    def _respond(self, reply: RawReply):
        response = self._hwebserver.Response(reply.body, reply.status, reply.content_type)
        for name, value in reply.headers.items():
            response.setHeader(name, value)
        # The default value names the exact Houdini and system build. Nothing
        # asked for that, and this endpoint answers one client.
        try:
            response.setHeader("Server", "-")
        except Exception:  # noqa: BLE001 - an older build may not allow it
            pass
        return response


def _read(request: Any) -> RawRequest:
    """Copy out of the web server's request object without parsing anything."""
    return RawRequest(
        method=_ask(request, "method", "POST"),
        path=_ask(request, "path", ""),
        headers=normalise_headers(_request_headers(request)),
        body=_ask(request, "body", b"") or b"",
        content_type=_ask(request, "contentType", "") or "",
        server_address=_address(_ask(request, "serverAddress", None)),
        client_address=_address(_ask(request, "clientAddress", None)),
    )


def _ask(request: Any, name: str, fallback: Any) -> Any:
    reader = getattr(request, name, None)
    if not callable(reader):
        return fallback
    try:
        return reader()
    except Exception:  # noqa: BLE001 - a fact we cannot read is a fact we do not have
        return fallback


def _address(value: Any) -> str | None:
    """The host part of an address the web server reports, however it shapes it."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return str(value)


def _request_headers(request: Any) -> Mapping[str, object]:
    """Headers from a request object, whether `headers` is a method or a map."""
    raw = getattr(request, "headers", None)
    if callable(raw):
        raw = raw()
    if raw is None:
        return {}
    items = getattr(raw, "items", None)
    return dict(items()) if callable(items) else {}


# How long a connection may sit with nothing on it before it is closed.
DEFAULT_IDLE_TIMEOUT_S = 30.0

# How many connections are served at once. One caller uses one at a time, so
# this is room for a handful of callers and their health polls, and a wall
# against a program that opens sockets and leaves them open.
DEFAULT_MAX_CONNECTIONS = 32

# How many connections the system holds for us between accepts.
LISTEN_BACKLOG = 64

# A refused body is read and thrown away for this long, so the caller can
# finish sending and then read the refusal instead of a closed connection.
DRAIN_S = 2.0
DRAIN_STEP_S = 0.2
DRAIN_BYTES = 1 << 16

# How often the serving thread looks at whether it has been told to stop, and
# how long stopping then waits for it.
POLL_S = 0.1
STOP_WAIT_S = 1.0

# What a connection past the cap is told, written to the socket by hand
# because no handler ever runs for it.
BUSY_BODY = b'{"ok":false,"error":{"code":"SERVER_BUSY","message":"too many connections"}}'

# Once the refusal is written the write side is closed and the request nobody
# wanted is read out for this long. Windows resets a connection that is closed
# with bytes still unread, and the reset takes the refusal with it before the
# caller has read it.
BUSY_LINGER_S = 1.0
BUSY_LINGER_STEP_S = 0.2


class StdlibBackend:
    """A threading HTTP server from the standard library, on loopback.

    One thread per connection, keep alive with an idle timeout, and a cap on
    how many connections are served at once.
    """

    def __init__(
        self,
        *,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.idle_timeout_s = idle_timeout_s
        self.max_connections = max_connections
        self._log = log
        self._address = "127.0.0.1"
        self._max_port = 0
        self._max_body: int | None = None
        self._endpoints: dict[str, Endpoint] = {}
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._started = False
        self._lock = threading.Lock()
        self._live = 0
        self._sockets: set[socket.socket] = set()

    # The seam

    def configure(self, *, address: str, port: int, max_port: int) -> None:
        """The address to bind and the top of the range. Start names the port."""
        self._address = address
        self._max_port = max_port

    def set_max_body(self, limit: int) -> None:
        """Refuse a body over this size before any of it is read."""
        self._max_body = limit

    def register(self, path: str, endpoint: Endpoint) -> None:
        if self._started:
            raise RuntimeError("handlers must be registered before the server runs")
        self._endpoints[path] = endpoint

    def start(self, port: int, *, in_background: bool = True) -> int:
        """Bind the first free port from here up, then serve.

        In the background the serving runs on a thread of its own, which is
        what the bridge asks for. Serving in this thread is offered for a
        process that has nothing else to do, and returns when the server stops.
        """
        if self._started:
            raise RuntimeError("this server has already been started")
        server = self._bind(port)
        self._server = server
        self._started = True
        bound = int(server.server_address[1])
        if not in_background:
            server.serve_forever(poll_interval=POLL_S)
            return bound
        self._thread = threading.Thread(
            target=server.serve_forever, args=(POLL_S,), name="nscr-mcp-server", daemon=True
        )
        self._thread.start()
        return bound

    def stop(self) -> None:
        """Close the listening socket and every connection still open.

        Idle connections are held open by keep alive, so each one is shut down
        by hand rather than waited out.
        """
        server, self._server = self._server, None
        thread, self._thread = self._thread, None
        self._started = False
        if server is None:
            return
        server.shutdown()
        with self._lock:
            open_sockets = list(self._sockets)
            self._sockets.clear()
        for connection in open_sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        server.server_close()
        if thread is not None:
            thread.join(STOP_WAIT_S)

    # What the handlers ask for

    @property
    def port(self) -> int | None:
        server = self._server
        return None if server is None else int(server.server_address[1])

    @property
    def max_body(self) -> int | None:
        return self._max_body

    def endpoint(self, path: str) -> Endpoint | None:
        return self._endpoints.get(path)

    def note(self, text: str) -> None:
        if self._log is not None:
            self._log(text)

    def hold(self, connection: socket.socket) -> bool:
        """Take a slot for one connection, or say there is none."""
        with self._lock:
            if self._live >= self.max_connections:
                return False
            self._live += 1
            self._sockets.add(connection)
        return True

    def release(self, connection: socket.socket) -> None:
        with self._lock:
            self._live = max(0, self._live - 1)
            self._sockets.discard(connection)

    # Internals

    def _bind(self, port: int) -> _Server:
        last: OSError | None = None
        end = max(port, self._max_port)
        for wanted in range(port, end + 1):
            try:
                return _Server((self._address, wanted), _Handler, self)
            except OSError as error:
                last = error
        raise OSError(f"no free port between {port} and {end}: {last}")


class _Server(ThreadingHTTPServer):
    """The listening socket, with a cap on the connections it serves."""

    daemon_threads = True
    request_queue_size = LISTEN_BACKLOG
    # On Linux and macOS this only lets a port be taken back after the
    # connections on it have drained, which nothing can use to steal a port
    # something else is listening on. On Windows it can, so it is off there.
    allow_reuse_address = sys.platform != "win32"
    allow_reuse_port = False

    def __init__(self, address: tuple[str, int], handler: Any, backend: StdlibBackend) -> None:
        self.backend = backend
        super().__init__(address, handler)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        if not self.backend.hold(request):
            self.backend.note("a connection arrived over the cap and was turned away")
            _say_busy(request)
            self.shutdown_request(request)
            return
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.backend.release(request)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A connection that broke is not a server fault, and is not printed."""
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            return
        self.backend.note(f"a connection ended with {type(error).__name__}: {error}")


class _Handler(BaseHTTPRequestHandler):
    """One connection: read a request, hand it over, write the answer."""

    protocol_version = "HTTP/1.1"
    server_version = "-"
    sys_version = ""

    def version_string(self) -> str:
        """Name no build. The default says which Houdini and which system."""
        return self.server_version

    def setup(self) -> None:
        self.timeout = self.server.backend.idle_timeout_s
        super().setup()

    def handle(self) -> None:
        """A caller that goes away mid request closes the connection, quietly.

        Windows reports that as a connection reset rather than an empty read,
        so both arrive here as a connection error.
        """
        try:
            super().handle()
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    def log_message(self, *args: Any) -> None:
        """Say nothing. The bridge writes its own log."""

    def do_GET(self) -> None:  # noqa: N802 - the name the server looks for
        self._serve()

    def do_POST(self) -> None:  # noqa: N802
        self._serve()

    def do_PUT(self) -> None:  # noqa: N802
        self._serve()

    def do_PATCH(self) -> None:  # noqa: N802
        self._serve()

    def do_DELETE(self) -> None:  # noqa: N802
        self._serve()

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._serve()

    def _serve(self) -> None:
        backend = self.server.backend
        path = self.path.split("?", 1)[0]
        endpoint = backend.endpoint(path)
        if endpoint is None:
            self._refuse(404, "NOT_FOUND", "there is nothing on that path")
            return
        body = self._body(backend.max_body)
        if body is None:
            return
        reply = endpoint(
            RawRequest(
                method=self.command,
                path=path,
                headers=normalise_headers(dict(self.headers.items())),
                body=body,
                content_type=self.headers.get("content-type", "") or "",
                server_address=self._arrived_on(),
                client_address=self.client_address[0] if self.client_address else None,
            )
        )
        self._write(reply.status, reply.body, reply.content_type, reply.headers)

    def _body(self, max_body: int | None) -> bytes | None:
        """The body, or nothing when the request was refused over it."""
        if "chunked" in (self.headers.get("transfer-encoding") or "").lower():
            self._refuse(400, "BODY_REFUSED", "this endpoint takes a body with a length")
            return None
        given = self.headers.get("content-length")
        if given is None:
            if self.command in ("POST", "PUT", "PATCH"):
                self._refuse(411, "BODY_REFUSED", "this endpoint needs a content length")
                return None
            return b""
        try:
            length = int(given)
        except ValueError:
            length = -1
        if length < 0:
            self._refuse(400, "BODY_REFUSED", "the content length is not a size")
            return None
        if max_body is not None and length > max_body:
            self._refuse(413, "BODY_REFUSED", f"the body is larger than {max_body} bytes")
            return None
        return self.rfile.read(length) if length else b""

    def _arrived_on(self) -> str | None:
        """The address this request came in on, for the loopback check."""
        try:
            return str(self.connection.getsockname()[0])
        except OSError:
            return None

    def _refuse(self, status: int, code: str, message: str) -> None:
        """Turn a request away before it reaches the bridge, and close.

        The body is drained afterwards so a caller that is still sending a
        large one reads this answer rather than a broken connection.
        """
        self.close_connection = True
        raw = RawReply.of(Reply(status, error_payload(code, message)))
        self._write(raw.status, raw.body, raw.content_type, {"Connection": "close"})
        if self._announced():
            self._drain()

    def _write(
        self, status: int, body: bytes, content_type: str, headers: Mapping[str, str]
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type or JSON_TYPE)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    def _announced(self) -> bool:
        """Whether the request said it was sending a body."""
        if "chunked" in (self.headers.get("transfer-encoding") or "").lower():
            return True
        try:
            return int(self.headers.get("content-length") or 0) > 0
        except ValueError:
            return False

    def _drain(self) -> None:
        """Read what is left on the connection and throw it away.

        It stops at the first quiet moment, so a caller that has finished
        sending is not waited on, and at the time cap whatever happens.
        """
        end = time.monotonic() + DRAIN_S
        try:
            self.connection.settimeout(DRAIN_STEP_S)
            while time.monotonic() < end:
                if not self.connection.recv(DRAIN_BYTES):
                    return
        except OSError:
            return


def _say_busy(connection: socket.socket) -> None:
    """Tell a connection past the cap, without a handler and without a thread."""
    head = (
        "HTTP/1.1 503 Service Unavailable\r\n"
        "Server: -\r\n"
        f"Content-Type: {JSON_TYPE}\r\n"
        f"Content-Length: {len(BUSY_BODY)}\r\n"
        "Connection: close\r\n\r\n"
    )
    try:
        connection.sendall(head.encode("ascii") + BUSY_BODY)
        connection.shutdown(socket.SHUT_WR)
    except OSError:
        return
    _linger(connection)


def _linger(connection: socket.socket) -> None:
    """Read out the request nobody wanted, so closing does not reset it.

    The caller is still sending, or has sent and not yet read. Either way the
    bytes sit unread on our side, and on Windows that turns the close into a
    reset the caller sees instead of the refusal. Waiting for the end of what
    it sent, or for a quiet moment, leaves nothing for the close to throw away.
    """
    end = time.monotonic() + BUSY_LINGER_S
    try:
        connection.settimeout(BUSY_LINGER_STEP_S)
        while time.monotonic() < end:
            if not connection.recv(DRAIN_BYTES):
                return
    except OSError:
        return


def make_backend(transport: str, server_name: str) -> Backend:
    """The server a bridge was configured to use."""
    if transport == STDLIB:
        return StdlibBackend()
    if transport == HWEBSERVER:
        return HwebserverBackend(server_name)
    raise ValueError(f"no such transport: {transport}")


class RecordingBackend:
    """A backend that keeps the handlers in memory, for tests and dry runs."""

    def __init__(self) -> None:
        self.endpoints: dict[str, Endpoint] = {}
        self.settings: dict[str, Any] = {}
        self.max_body: int | None = None
        self.port: int | None = None
        self.running = False
        self.stops = 0
        self._lock = threading.Lock()

    def configure(self, *, address: str, port: int, max_port: int) -> None:
        self.settings = {"address": address, "port": port, "max_port": max_port}

    def set_max_body(self, limit: int) -> None:
        self.max_body = limit

    def register(self, path: str, endpoint: Endpoint) -> None:
        if self.running:
            raise RuntimeError("handlers must be registered before the server runs")
        self.endpoints[path] = endpoint

    def start(self, port: int, *, in_background: bool = True) -> int:
        with self._lock:
            if self.running:
                raise RuntimeError("one server, one run, per process")
            self.running = True
            self.port = port
        return port

    def stop(self) -> None:
        with self._lock:
            self.running = False
            self.stops += 1

    def send(self, request: RawRequest) -> RawReply:
        """Hand a request to the registered handler, as the web server would.

        The header names are lowercased first, which is what the real seam
        does when it copies them out of the web server's request object.
        """
        arrived = RawRequest(
            method=request.method,
            path=request.path,
            headers=normalise_headers(request.headers),
            body=request.body,
            content_type=request.content_type,
            server_address=request.server_address,
            client_address=request.client_address,
        )
        return self.endpoints[arrived.path](arrived)
