"""The seam between the bridge and Houdini's web server.

Everything the web server does for the bridge is four calls: settings,
register, start, stop. Keeping them behind one small class means the request
handling can be tested in plain Python, and it puts the rules the web server
imposes in one readable place.

Why a raw URL handler and not the built in API route. The API route reads the
posted form and parses its JSON field before any handler code runs, so an
unsigned request from anywhere reaches a parser the bridge does not control.
A raw handler is given the request untouched: the bridge reads the headers,
refuses what it does not want, caps the size and then parses the body itself.
No API function is registered, so that route does not exist on this server.

The rest of the rules:

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
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from nscr_houdini_mcp.bridge.envelope import Reply
from nscr_houdini_mcp.bridge.security import normalise_headers

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
    """Houdini's own web server, driven through one named server object."""

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
