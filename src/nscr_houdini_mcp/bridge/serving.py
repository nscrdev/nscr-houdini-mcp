"""The seam between the bridge and Houdini's web server.

Everything the web server can do for the bridge is four calls: settings,
register, start, stop. Keeping them behind one small class means the request
handling can be tested in plain Python, and means the rules that were learned
the hard way live in one readable place:

- Build on an explicit named server object. The module level functions resolve
  through a thread local, so they would register onto, and restart, whichever
  server another tool in the same Houdini already owns.
- Register every handler before the server runs. One registered afterwards is
  unreachable.
- Set the bind address and the port range through the port settings. The run
  call takes neither.
- Set a cross origin whitelist, because an unset one reflects any origin back.
  It does not stop the request, so the handler checks for the header as well.
- One server, one run, per process. A second run after a shutdown leaves
  sockets behind that nothing can reclaim.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from nscr_houdini_mcp.bridge.envelope import Reply
from nscr_houdini_mcp.bridge.security import normalise_headers

# The port settings this bridge writes are its own, under its own server
# object. "main" is the name the settings and the handlers agree on.
PORT_NAME = "main"

# A non empty whitelist turns the origin reflection off. Nothing real is meant
# to match it: the bridge answers no browser.
CORS_DENY_LIST = ("http://cors.invalid",)

# Handlers take the request headers and the posted payload and hand back a
# status and a body.
Endpoint = Callable[[Mapping[str, str], Mapping[str, Any]], Reply]


class Backend(Protocol):
    """What the bridge needs from a web server."""

    def configure(self, *, address: str, port: int, max_port: int) -> None: ...

    def register(self, name: str, endpoint: Endpoint) -> None: ...

    def start(self, port: int, *, in_background: bool = True) -> int: ...

    def stop(self) -> None: ...


class HwebserverBackend:
    """Houdini's own web server, driven through one named server object."""

    namespace = "mcp"

    def __init__(self, server_name: str) -> None:
        import hwebserver

        self._hwebserver = hwebserver
        self._server = hwebserver.Server(server_name)
        self._port: int | None = None
        self._reported_port: int | None = None
        self._started = False

    def configure(self, *, address: str, port: int, max_port: int) -> None:
        self._server.setSettingsForPort(
            {"ADDRESS": address, "PORT": port, "MAX_PORT": max_port},
            PORT_NAME,
        )
        self._server.setCORSWhitelist(list(CORS_DENY_LIST), PORT_NAME)

    def register(self, name: str, endpoint: Endpoint) -> None:
        if self._started:
            raise RuntimeError("handlers must be registered before the server runs")

        def api(request, envelope=None):
            headers = normalise_headers(_request_headers(request))
            reply = endpoint(headers, envelope if isinstance(envelope, Mapping) else {})
            return self._response(reply)

        api.__name__ = name
        api.__qualname__ = name
        self._server.apiFunction(namespace=self.namespace)(api)

    def start(self, port: int, *, in_background: bool = True) -> int:
        if self._started:
            raise RuntimeError("one server, one run, per process")
        self._started = True
        self._server.run(
            port=port,
            in_background=in_background,
            port_callback=self._note_port,
        )
        self._port = port
        return self._reported_port or port

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._server.requestShutdown()

    def _note_port(self, *args: Any) -> None:
        """Record the port the server really took, whatever shape it reports."""
        for value in args:
            try:
                self._reported_port = int(value)
            except (TypeError, ValueError):
                continue
            return

    def _response(self, reply: Reply):
        body = json.dumps(reply.payload, default=str).encode("utf-8")
        response = self._hwebserver.Response(body, reply.status, "application/json")
        # The default value names the exact Houdini and system build. Nothing
        # asked for that, and this endpoint answers one client.
        try:
            response.setHeader("Server", "-")
        except Exception:  # noqa: BLE001 - an older build may not allow it
            pass
        return response


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
        self.port: int | None = None
        self.running = False
        self.stops = 0
        self._lock = threading.Lock()

    def configure(self, *, address: str, port: int, max_port: int) -> None:
        self.settings = {"address": address, "port": port, "max_port": max_port}

    def register(self, name: str, endpoint: Endpoint) -> None:
        if self.running:
            raise RuntimeError("handlers must be registered before the server runs")
        self.endpoints[name] = endpoint

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

    def post(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Reply:
        """Call a registered handler the way the web server would.

        `arguments` are the posted keyword arguments, so the envelope arrives
        under the same name a real request puts it under.
        """
        envelope = (arguments or {}).get("envelope")
        return self.endpoints[name](
            normalise_headers(headers),
            envelope if isinstance(envelope, Mapping) else {},
        )
