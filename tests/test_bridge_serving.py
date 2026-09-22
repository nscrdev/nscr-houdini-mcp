"""What the server that answers the port does before the bridge sees anything.

These tests talk to a real socket, so what is checked here is the transport:
the paths it knows, the length rules, the size cap, keep alive, the cap on
connections and what stopping closes. The bridge pipeline itself is checked in
the app tests, against this server and against the recording one.
"""

from __future__ import annotations

import http.client
import socket
import time
from typing import Any

import pytest

import support
from nscr_houdini_mcp.bridge.serving import (
    HWEBSERVER,
    JSON_TYPE,
    STDLIB,
    HwebserverBackend,
    RawReply,
    RawRequest,
    StdlibBackend,
    make_backend,
)

PATH = "/nscr-mcp/health"
TIMEOUT_S = 10.0


class Seen:
    """The requests that reached the endpoint."""

    def __init__(self) -> None:
        self.requests: list[RawRequest] = []

    def __call__(self, request: RawRequest) -> RawReply:
        self.requests.append(request)
        return RawReply(200, b'{"ok":true}')


def serve(**settings: Any) -> tuple[StdlibBackend, Seen, int]:
    """One server on a port of its own, with one endpoint on it."""
    max_body = settings.pop("max_body", 1024)
    backend = StdlibBackend(**settings)
    start, end = support.APP_PORTS
    backend.configure(address="127.0.0.1", port=start, max_port=end)
    backend.set_max_body(max_body)
    seen = Seen()
    backend.register(PATH, seen)
    port = backend.start(start)
    return backend, seen, port


@pytest.fixture
def server() -> Any:
    backend, seen, port = serve()
    try:
        yield backend, seen, port
    finally:
        backend.stop()


def open_connection(port: int) -> http.client.HTTPConnection:
    return http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT_S)


def post(connection: http.client.HTTPConnection, path: str = PATH, body: bytes = b"{}") -> Any:
    connection.request("POST", path, body=body, headers={"Content-Type": JSON_TYPE})
    return connection.getresponse()


def test_a_request_reaches_the_endpoint_with_the_address_it_arrived_on(server: Any) -> None:
    backend, seen, port = server
    connection = open_connection(port)
    try:
        answer = post(connection)
        assert answer.status == 200
        assert answer.read() == b'{"ok":true}'
    finally:
        connection.close()
    arrived = seen.requests[0]
    assert arrived.method == "POST"
    assert arrived.path == PATH
    assert arrived.server_address == "127.0.0.1"
    assert arrived.client_address == "127.0.0.1"
    assert arrived.headers["host"] == f"127.0.0.1:{port}"
    assert arrived.content_type == JSON_TYPE
    assert arrived.body == b"{}"


def test_a_path_nothing_is_registered_on_is_not_there(server: Any) -> None:
    backend, seen, port = server
    connection = open_connection(port)
    try:
        answer = post(connection, path="/api")
        assert answer.status == 404
        assert b"NOT_FOUND" in answer.read()
    finally:
        connection.close()
    assert seen.requests == []


def test_the_answer_names_no_build(server: Any) -> None:
    backend, seen, port = server
    connection = open_connection(port)
    try:
        answer = post(connection)
        answer.read()
        assert answer.getheader("Server") == "-"
    finally:
        connection.close()


def test_a_query_is_not_part_of_the_path(server: Any) -> None:
    backend, seen, port = server
    connection = open_connection(port)
    try:
        assert post(connection, path=f"{PATH}?x=1").status == 200
    finally:
        connection.close()
    assert seen.requests[0].path == PATH


def test_a_body_without_a_length_is_refused(server: Any) -> None:
    backend, seen, port = server
    connection = open_connection(port)
    try:
        connection.putrequest("POST", PATH)
        connection.putheader("Content-Type", JSON_TYPE)
        connection.endheaders()
        answer = connection.getresponse()
        assert answer.status == 411
        assert b"BODY_REFUSED" in answer.read()
    finally:
        connection.close()
    assert seen.requests == []


def test_a_chunked_body_is_refused(server: Any) -> None:
    backend, seen, port = server
    connection = open_connection(port)
    try:
        connection.putrequest("POST", PATH)
        connection.putheader("Content-Type", JSON_TYPE)
        connection.putheader("Transfer-Encoding", "chunked")
        connection.endheaders()
        connection.send(b"2\r\n{}\r\n0\r\n\r\n")
        answer = connection.getresponse()
        assert answer.status == 400
        assert b"BODY_REFUSED" in answer.read()
    finally:
        connection.close()
    assert seen.requests == []


def test_a_body_over_the_cap_is_refused_and_never_read(server: Any) -> None:
    backend, seen, port = server
    connection = open_connection(port)
    try:
        answer = post(connection, body=b"x" * (1024 * 1024))
        assert answer.status == 413
        assert b"BODY_REFUSED" in answer.read()
    finally:
        connection.close()
    assert seen.requests == []


def test_the_session_still_answers_after_a_body_it_refused(server: Any) -> None:
    backend, seen, port = server
    first = open_connection(port)
    try:
        post(first, body=b"x" * (2 * 1024 * 1024)).read()
    finally:
        first.close()
    second = open_connection(port)
    try:
        assert post(second).status == 200
    finally:
        second.close()


def test_one_connection_carries_several_requests(server: Any) -> None:
    backend, seen, port = server
    connection = open_connection(port)
    try:
        for _ in range(5):
            answer = post(connection)
            assert answer.status == 200
            answer.read()
        held = connection.sock.getsockname()
    finally:
        connection.close()
    assert len(seen.requests) == 5
    assert held is not None


def test_a_connection_left_open_is_closed_after_the_idle_time() -> None:
    backend, seen, port = serve(idle_timeout_s=0.3)
    try:
        connection = open_connection(port)
        post(connection).read()
        time.sleep(1.0)
        with pytest.raises(Exception):  # noqa: B017 - any broken connection will do
            post(connection).read()
        connection.close()
    finally:
        backend.stop()


def turned_away(connection: http.client.HTTPConnection) -> bool:
    """Whether the server refused this connection.

    Over the cap it writes a 503 and hangs up. The hang up can reach a caller
    before the answer does, so a broken connection counts as turned away too:
    either way nothing of the request was served.
    """
    try:
        answer = post(connection)
        return answer.status == 503 and b"SERVER_BUSY" in answer.read()
    except (OSError, http.client.HTTPException):
        return True


def test_connections_past_the_cap_are_turned_away() -> None:
    backend, seen, port = serve(max_connections=2)
    held = []
    try:
        for _ in range(2):
            connection = open_connection(port)
            post(connection).read()
            held.append(connection)
        over = open_connection(port)
        held.append(over)
        assert turned_away(over)
        # The cap turned one connection away, not the server: what it was
        # already serving is answered as before.
        answer = post(held[0])
        assert answer.status == 200
        answer.read()
    finally:
        for connection in held:
            connection.close()
        backend.stop()


def test_a_slot_comes_back_when_a_connection_closes() -> None:
    backend, seen, port = serve(max_connections=1)
    try:
        for _ in range(3):
            connection = open_connection(port)
            assert post(connection).status == 200
            connection.close()
            # The thread holding the slot ends just after the socket does.
            deadline = time.monotonic() + 5.0
            while backend._live and time.monotonic() < deadline:
                time.sleep(0.01)
    finally:
        backend.stop()


def test_stopping_closes_the_port_and_a_connection_left_open() -> None:
    backend, seen, port = serve()
    connection = open_connection(port)
    post(connection).read()
    started = time.monotonic()
    backend.stop()
    assert time.monotonic() - started < 1.0
    with pytest.raises(OSError):
        probe = socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT_S)
        probe.close()
    connection.close()


def test_the_port_range_is_walked_until_one_is_free() -> None:
    start, end = support.APP_PORTS
    taken = socket.socket()
    # The same flag the server sets, so a port another test has just finished
    # with can be taken here rather than waiting for it to drain.
    taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    taken.bind(("127.0.0.1", start))
    taken.listen(1)
    try:
        backend = StdlibBackend()
        backend.configure(address="127.0.0.1", port=start, max_port=end)
        backend.register(PATH, Seen())
        try:
            assert backend.start(start) > start
        finally:
            backend.stop()
    finally:
        taken.close()


def test_a_handler_cannot_be_added_once_the_server_runs(server: Any) -> None:
    backend, seen, port = server
    with pytest.raises(RuntimeError):
        backend.register("/late", Seen())
    with pytest.raises(RuntimeError):
        backend.start(port)


def test_the_transport_is_picked_by_name() -> None:
    assert isinstance(make_backend(STDLIB, "whatever"), StdlibBackend)
    with pytest.raises(ValueError):
        make_backend("carrier pigeon", "whatever")


def test_the_fallback_transport_needs_houdini() -> None:
    """Houdini's own server is only there inside Houdini."""
    try:
        import hwebserver  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            make_backend(HWEBSERVER, "whatever")
        return
    assert isinstance(make_backend(HWEBSERVER, "whatever"), HwebserverBackend)
