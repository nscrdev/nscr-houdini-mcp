from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import support
from fake_hou import Scene
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client as client_module
from nscr_houdini_mcp.bridge import host, registry, security, signing
from nscr_houdini_mcp.bridge.app import Bridge, BridgeConfig, BridgeStartError, houdini_lock
from nscr_houdini_mcp.bridge.handlers import ToolRegistry, default_registry
from nscr_houdini_mcp.bridge.serving import (
    CALL_PATH,
    HEALTH_PATH,
    JSON_TYPE,
    Endpoint,
    RawReply,
    RawRequest,
    RecordingBackend,
    StdlibBackend,
)

SOCKET_TIMEOUT_S = 30.0


class SocketDriver:
    """The same seam, over a real socket.

    It stands in for the recording backend in the tests that send requests, so
    the pipeline is tried once in plain Python and once through the server
    that answers the port: the headers, the length rules and the sizes are
    then the ones a caller really meets.
    """

    def __init__(self) -> None:
        self.backend = StdlibBackend()
        self.port: int | None = None

    def configure(self, *, address: str, port: int, max_port: int) -> None:
        self.backend.configure(address=address, port=port, max_port=max_port)

    def set_max_body(self, limit: int) -> None:
        self.backend.set_max_body(limit)

    def register(self, path: str, endpoint: Endpoint) -> None:
        self.backend.register(path, endpoint)

    def start(self, port: int, *, in_background: bool = True) -> int:
        self.port = self.backend.start(port, in_background=in_background)
        return self.port

    def stop(self) -> None:
        self.backend.stop()

    def send(self, request: RawRequest) -> RawReply:
        """Send one request exactly as it was built, and read the answer."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=SOCKET_TIMEOUT_S)
        try:
            connection.putrequest(
                request.method, request.path, skip_host=True, skip_accept_encoding=True
            )
            headers = dict(request.headers)
            if request.content_type:
                headers["content-type"] = request.content_type
            headers["content-length"] = str(len(request.body))
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders(request.body)
            answer = connection.getresponse()
            body = answer.read()
            return RawReply(
                answer.status,
                body,
                {str(name).lower(): str(value) for name, value in answer.getheaders()},
                answer.headers.get("content-type", ""),
            )
        finally:
            connection.close()


@pytest.fixture(params=["recording", "stdlib"])
def driver(request: Any) -> str:
    """Run a pipeline test against both of the ways a request can arrive."""
    return request.param


def make_bridge(home: Path, *, driver: str = "recording", **overrides: Any) -> tuple[Bridge, Any]:
    backend: Any = SocketDriver() if driver == "stdlib" else RecordingBackend()
    settings: dict[str, Any] = {
        "home": home,
        "kind": "hython",
        "verify_loopback": False,
        "heartbeat_s": 3600.0,
        "facts": {"houdini_version": "22.0.0", "hfs": "/hfs", "hip_path": None},
    }
    if driver == "stdlib":
        settings["port_range"] = support.APP_PORTS
    settings.update(overrides)
    hou = settings.pop("hou", None)
    bridge = Bridge(BridgeConfig(**settings), backend=backend, hou=hou)
    return bridge, backend


def build(
    bridge: Bridge,
    path: str,
    payload: Any = None,
    *,
    token: str | None = None,
    headers: Mapping[str, str] | None = None,
    content_type: str = JSON_TYPE,
    method: str = "POST",
    server_address: str | None = "127.0.0.1",
    sign: bool = True,
    body: bytes | None = None,
) -> RawRequest:
    """One request as the web server would hand it over."""
    raw = (
        json.dumps(payload if payload is not None else {}).encode("utf-8") if body is None else body
    )
    sent = {"host": f"127.0.0.1:{bridge.port}"}
    if sign:
        sent.update(
            signing.sign_request(
                token if token is not None else bridge._token,
                method=method,
                path=path,
                session_id=bridge.session_id,
                body=raw,
            )
        )
    sent.update(headers or {})
    return RawRequest(
        method=method,
        path=path,
        headers=sent,
        body=raw,
        content_type=content_type,
        server_address=server_address,
        client_address="127.0.0.1",
    )


def envelope(**fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool": "bridge.ping", "arguments": {"echo": 1}}
    payload.update(fields)
    return payload


def body_of(reply: Any) -> Any:
    return json.loads(reply.body)


def send(bridge: Bridge, backend: RecordingBackend, path: str, payload: Any = None, **rest: Any):
    return backend.send(build(bridge, path, payload, **rest))


def test_starting_writes_a_session_row_and_a_private_file(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    record = bridge.start()
    try:
        assert record.kind == "hython"
        assert record.alias == "w1"
        assert record.port == bridge.port
        assert backend.settings == {
            "address": "127.0.0.1",
            "port": bridge.port,
            "max_port": 18199,
        }
        assert sorted(backend.endpoints) == sorted([CALL_PATH, HEALTH_PATH])

        entries = registry.list_entries(tmp_path)
        assert [entry["session_id"] for entry in entries] == [bridge.session_id]
        assert entries[0]["port"] == bridge.port
        assert entries[0]["pid"] == bridge.pid
        assert security.is_private(registry.entry_path(tmp_path, bridge.session_id))

        with store_module.Store(bridge.store_path) as store:
            assert [row.session_id for row in store.list_sessions()] == [bridge.session_id]
    finally:
        bridge.stop()


def test_stopping_takes_the_session_away_and_can_be_repeated(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    bridge.stop()
    bridge.stop()

    assert registry.list_entries(tmp_path) == []
    assert backend.running is False
    assert backend.stops == 1
    with store_module.Store(bridge.store_path) as store:
        assert store.list_sessions() == []
        assert store.get_session(bridge.session_id).state == "gone"


def test_a_self_check_that_ends_after_stopping_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first self check is held open until the bridge has stopped, so its
    # answer lands after the session file is gone, every time.
    checking = threading.Event()
    stopped = threading.Event()

    def health(*_args: Any, **_kwargs: Any) -> Any:
        checking.set()
        stopped.wait(SOCKET_TIMEOUT_S)
        raise client_module.BridgeUnreachable("nothing answered")

    monkeypatch.setattr(client_module, "health", health)
    bridge, _ = make_bridge(tmp_path)
    bridge.start()
    assert checking.wait(SOCKET_TIMEOUT_S)
    bridge.stop()
    stopped.set()
    assert bridge._heartbeat is not None
    bridge._heartbeat.join(SOCKET_TIMEOUT_S)

    assert not bridge._heartbeat.is_alive()
    assert bridge.transport_ok is False
    assert registry.list_entries(tmp_path) == []


def test_a_gui_session_is_named_after_its_scene(tmp_path: Path) -> None:
    bridge, _ = make_bridge(
        tmp_path,
        kind="gui",
        facts={"hip_path": str(tmp_path / "hero shot_v003.hip")},
    )
    record = bridge.start()
    try:
        assert record.alias == "hero-shot_v003-1"
    finally:
        bridge.stop()


def test_a_session_id_is_random_and_the_alias_comes_back(tmp_path: Path) -> None:
    first, _ = make_bridge(tmp_path)
    first.start()
    second, _ = make_bridge(tmp_path)
    second.start()
    try:
        assert first.session_id != second.session_id
        assert [first.alias, second.alias] == ["w1", "w2"]
    finally:
        second.stop()
        first.stop()
    third, _ = make_bridge(tmp_path)
    third.start()
    try:
        assert third.alias == "w1"
        assert third.session_id not in (first.session_id, second.session_id)
    finally:
        third.stop()


def test_two_sessions_on_the_same_scene_get_different_ids_and_names(tmp_path: Path) -> None:
    facts = {"hip_path": str(tmp_path / "shot_010.hip")}
    first, _ = make_bridge(tmp_path, kind="gui", facts=facts)
    second, _ = make_bridge(tmp_path, kind="gui", facts=facts)
    first.start()
    second.start()
    try:
        assert first.session_id != second.session_id
        assert [first.alias, second.alias] == ["shot_010-1", "shot_010-2"]
    finally:
        second.stop()
        first.stop()


def test_a_replaced_scene_is_written_where_other_processes_read_it(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        bridge.identity.bump("cleared")

        assert bridge.scene_epoch == 1
        with store_module.Store(tmp_path / store_module.STORE_FILE_NAME) as store:
            assert store.get_session(bridge.session_id).scene_epoch == 1
        entry = registry.read_entry(registry.entry_path(tmp_path, bridge.session_id))
        assert entry["scene_epoch"] == 1
        data = body_of(send(bridge, backend, HEALTH_PATH))["data"]
        assert data["scene_epoch"] == 1
        assert data["scene"]["changed"] == "cleared"
    finally:
        bridge.stop()


def test_a_call_written_against_a_scene_that_has_gone_is_refused(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        bridge.identity.bump("loaded")
        reply = send(bridge, backend, CALL_PATH, envelope(scene_epoch=0))
        payload = body_of(reply)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "SCENE_REPLACED"
        assert payload["scene"]["scene_epoch"] == 1
    finally:
        bridge.stop()


def test_an_answer_can_be_held_back_so_the_lost_reply_case_can_be_tried(tmp_path: Path) -> None:
    """The work runs, and the caller is left with nothing. That is the point.

    Only a session carrying the self check tool will do it, which is a worker
    this project started to be driven.
    """
    tools = ToolRegistry()
    ran: list[str] = []
    tools.add(
        "bridge.selfcheck",
        lambda arguments: ran.append("once") or {"created": ["/obj/geo1"]},
        arguments=("drop_reply",),
        mutating=True,
    )
    bridge, backend = make_bridge(tmp_path, drop_reply_s=0.2)
    bridge.tools = tools
    bridge.dispatcher.tools = tools
    bridge.start()
    try:
        began = time.monotonic()
        send(
            bridge,
            backend,
            CALL_PATH,
            envelope(tool="bridge.selfcheck", arguments={"drop_reply": True}),
        )
        took = time.monotonic() - began
        # Some clocks tick coarsely, so allow a sliver under the hold.
        assert took >= 0.19
        assert ran == ["once"]
    finally:
        bridge.stop()


def test_an_answer_is_only_held_back_where_the_self_check_lives(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path, drop_reply_s=30.0)
    bridge.start()
    try:
        began = time.monotonic()
        reply = send(
            bridge,
            backend,
            CALL_PATH,
            envelope(arguments={"echo": 1, "drop_reply": True}),
        )
        assert time.monotonic() - began < 5.0
        assert body_of(reply)["ok"] is False
        assert body_of(reply)["error"]["code"] == "BAD_ARGUMENTS"
    finally:
        bridge.stop()


def test_health_answers_from_memory(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = send(bridge, backend, HEALTH_PATH)
        assert reply.status == 200
        data = body_of(reply)["data"]
        assert data["status"] == "ok"
        assert data["session_id"] == bridge.session_id
        assert data["alias"] == "w1"
        assert data["port"] == bridge.port
        assert data["scene_epoch"] == 0
        assert data["busy"] is False
        assert data["tools"][0] == "bridge.ping"
        assert "node.create" in data["tools"]
    finally:
        bridge.stop()


def test_a_call_reaches_its_tool_and_carries_the_trace_back(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = send(
            bridge,
            backend,
            CALL_PATH,
            envelope(session_id=bridge.session_id, scene_epoch=0, operation_id="op-1"),
        )
        assert reply.status == 200
        payload = body_of(reply)
        assert payload["ok"] is True
        assert payload["data"] == {"pong": True, "echo": 1}
        assert payload["operation_id"] == "op-1"
        assert payload["scene_epoch"] == 0
        assert payload["timing_ms"] >= 0
    finally:
        bridge.stop()


# Requests that never reach a tool


@pytest.mark.parametrize("path", [HEALTH_PATH, CALL_PATH])
def test_an_unsigned_request_is_refused(tmp_path: Path, path: str, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = backend.send(build(bridge, path, envelope(), sign=False))
        assert reply.status == 401
        assert body_of(reply)["error"]["code"] == "UNAUTHORIZED"
        assert bridge._token not in reply.body.decode("utf-8")
    finally:
        bridge.stop()


def test_a_signature_from_another_token_is_refused(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, envelope(), token=security.mint_token())
        assert reply.status == 401
    finally:
        bridge.stop()


def test_a_signature_over_a_different_body_is_refused(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        request = build(bridge, CALL_PATH, envelope())
        swapped = RawRequest(
            request.method,
            request.path,
            request.headers,
            json.dumps(envelope(tool="bridge.other")).encode("utf-8"),
            request.content_type,
            request.server_address,
            request.client_address,
        )
        assert backend.send(swapped).status == 401
    finally:
        bridge.stop()


def test_the_same_signature_cannot_be_sent_twice(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        request = build(bridge, CALL_PATH, envelope())
        assert backend.send(request).status == 200
        assert backend.send(request).status == 401
    finally:
        bridge.stop()


def test_a_flood_of_signed_requests_gets_a_coded_answer_with_the_limit_and_the_wait(
    tmp_path: Path, driver: str
) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        assert send(bridge, backend, CALL_PATH, envelope()).status == 200
        # Fill the table the way a flood would. The bridge's own self check
        # spends nonces too, in the background, so the fill goes until the
        # table refuses rather than to a count worked out beforehand.
        nonces = bridge._verifier.nonces
        limit = nonces.limit = len(nonces) + 5
        with pytest.raises(signing.FloodGuard):
            for index in range(limit + 1):
                nonces.claim(f"flood-{index:04d}", time.time())
        reply = send(bridge, backend, CALL_PATH, envelope())
        assert reply.status == 429
        error = body_of(reply)["error"]
        assert error["code"] == "FLOOD_GUARD"
        window = signing.DEFAULT_SKEW_S
        assert error["details"]["limit"] == limit
        assert error["details"]["window_s"] == window
        assert 1 <= error["details"]["retry_after_s"] <= window
        assert f"at most {limit} requests in any {window} seconds" in error["hint"]
        window = error["details"]["retry_after_s"]
        assert f"wait {window} seconds" in error["hint"]
        assert "not signed" not in error["message"]
    finally:
        bridge.stop()


def test_a_signature_from_outside_the_time_window_is_refused(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        body = json.dumps(envelope()).encode("utf-8")
        old = signing.sign_request(
            bridge._token,
            method="POST",
            path=CALL_PATH,
            session_id=bridge.session_id,
            body=body,
            now=0.0,
        )
        request = build(bridge, CALL_PATH, envelope(), sign=False, headers=old)
        assert backend.send(request).status == 401
    finally:
        bridge.stop()


@pytest.mark.parametrize("header", ["Origin", "Referer", "origin"])
def test_a_request_from_a_page_is_refused_even_when_signed(
    tmp_path: Path, header: str, driver: str
) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, envelope(), headers={header: "http://x.example"})
        assert reply.status == 403
        assert body_of(reply)["error"]["code"] == "FORBIDDEN"
    finally:
        bridge.stop()


@pytest.mark.parametrize("host", ["evil.example:18100", "", "127.0.0.1", "127.0.0.1:1"])
def test_another_host_is_refused_even_when_signed(tmp_path: Path, host: str, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, envelope(), headers={"host": host})
        assert reply.status == 403
    finally:
        bridge.stop()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_this_machine_by_either_loopback_name_is_allowed(
    tmp_path: Path, host: str, driver: str
) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        headers = {"host": f"{host}:{bridge.port}"}
        assert send(bridge, backend, HEALTH_PATH, headers=headers).status == 200
    finally:
        bridge.stop()


def test_a_form_post_is_refused_before_anything_is_read(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = backend.send(
            build(
                bridge,
                CALL_PATH,
                content_type="application/x-www-form-urlencoded",
                body=b"json=" + b"[" * 5000 + b"]" * 5000,
                sign=False,
            )
        )
        assert reply.status == 415
        assert body_of(reply)["error"]["code"] == "BODY_REFUSED"
    finally:
        bridge.stop()


def test_a_body_over_the_cap_is_refused_before_it_is_read(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver, max_body_bytes=1024)
    bridge.start()
    try:
        reply = backend.send(build(bridge, CALL_PATH, body=b"x" * 2048, sign=False))
        assert reply.status == 413
    finally:
        bridge.stop()


def test_a_body_nested_past_the_limit_is_refused_without_parsing(
    tmp_path: Path, driver: str
) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = backend.send(build(bridge, CALL_PATH, body=b"[" * 5000 + b"]" * 5000))
        assert reply.status == 400
        assert body_of(reply)["error"]["code"] == "BODY_REFUSED"
    finally:
        bridge.stop()


def test_brackets_inside_an_argument_do_not_count_as_nesting(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        code = "{" * 200 + "[" * 200
        reply = send(bridge, backend, CALL_PATH, envelope(arguments={"echo": code}))
        assert reply.status == 200
        assert body_of(reply)["data"]["echo"] == code
    finally:
        bridge.stop()


def test_a_method_other_than_post_is_refused(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        assert backend.send(build(bridge, HEALTH_PATH, method="GET")).status == 405
    finally:
        bridge.stop()


def test_a_request_that_did_not_arrive_on_loopback_closes_the_bridge(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        reply = send(bridge, backend, HEALTH_PATH, server_address="192.0.2.7")
        assert reply.status == 403
        # A slow runner can take a while to close the server and drop the
        # entry, so give it up to fifteen seconds rather than five.
        for _ in range(1500):
            if not bridge.running and registry.list_entries(tmp_path) == []:
                break
            threading.Event().wait(0.01)
        assert bridge.running is False
        assert registry.list_entries(tmp_path) == []
    finally:
        bridge.stop()


def test_every_answer_is_signed_so_a_squatter_cannot_pass_for_the_bridge(
    tmp_path: Path, driver: str
) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        request = build(bridge, HEALTH_PATH)
        reply = backend.send(request)
        nonce = request.headers[signing.NONCE_HEADER]
        assert reply.headers[signing.SIGNATURE_HEADER] == signing.response_signature(
            bridge._token, nonce=nonce, status=reply.status, body=reply.body
        )
        assert not signing.equal(
            reply.headers[signing.SIGNATURE_HEADER],
            signing.response_signature(
                security.mint_token(), nonce=nonce, status=reply.status, body=reply.body
            ),
        )
    finally:
        bridge.stop()


# Calls that reach dispatch


def test_a_request_that_cannot_be_read_comes_back_as_a_bad_request(
    tmp_path: Path, driver: str
) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, {"tool": ""})
        assert reply.status == 400
        assert body_of(reply)["error"]["code"] == "BAD_ENVELOPE"
    finally:
        bridge.stop()


def test_another_session_id_and_an_unknown_tool_fail_without_failing_the_call(
    tmp_path: Path,
    driver: str,
) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        wrong = send(bridge, backend, CALL_PATH, envelope(session_id="somebody-else"))
        assert wrong.status == 200
        assert body_of(wrong)["error"]["code"] == "UNKNOWN_SESSION"
        assert body_of(wrong)["error"]["details"]["session_id"] == bridge.session_id

        unknown = send(bridge, backend, CALL_PATH, envelope(tool="nothing.here"))
        assert unknown.status == 200
        assert body_of(unknown)["error"]["code"] == "UNKNOWN_TOOL"
        assert body_of(unknown)["error"]["details"]["tools"] == bridge.tools.names()
    finally:
        bridge.stop()


def test_a_bridge_that_breaks_below_the_tool_still_signs_a_coded_answer(tmp_path: Path) -> None:
    """Nothing leaves this endpoint unsigned.

    An unsigned 500 from the web server is what a caller is meant to read as
    somebody else sitting on the port, so a bridge that breaks says so in its
    own words, with its own signature on it.
    """
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:

        def explode(envelope: Any) -> Any:
            raise RuntimeError("the floor gave way in /Users/somebody/scenes")

        bridge.dispatcher.dispatch = explode
        reply = send(bridge, backend, CALL_PATH, envelope())
        assert reply.status == 200
        body = body_of(reply)
        assert body["error"]["code"] == "TOOL_FAILED"
        assert body["error"]["details"]["exception"] == "RuntimeError"
        assert "somebody" not in reply.body.decode("utf-8")
        assert signing.SIGNATURE_HEADER in reply.headers
        assert "the floor gave way" in bridge.log_path().read_text(encoding="utf-8")
        assert bridge.running is True
    finally:
        bridge.stop()


def test_a_tool_that_raises_gives_back_a_type_and_logs_the_rest(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.add("bridge.explode", _explode)
    backend = RecordingBackend()
    bridge = Bridge(
        BridgeConfig(home=tmp_path, kind="hython", verify_loopback=False, heartbeat_s=3600.0),
        backend=backend,
        tools=tools,
    )
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, {"tool": "bridge.explode"})
        assert reply.status == 200
        error = body_of(reply)["error"]
        assert error["code"] == "TOOL_FAILED"
        assert error["details"]["exception"] == "ValueError"
        # The message the tool raised names a path, so it stays out of the reply.
        assert "secret-scene" not in reply.body.decode("utf-8")
        written = bridge.log_path().read_text(encoding="utf-8")
        assert "secret-scene" in written
        assert bridge._token not in written
        assert bridge.running is True
    finally:
        bridge.stop()


def test_the_lock_is_let_go_when_a_tool_fails(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.add("bridge.explode", _explode)
    backend = RecordingBackend()
    bridge = Bridge(
        BridgeConfig(home=tmp_path, kind="hython", verify_loopback=False, heartbeat_s=3600.0),
        backend=backend,
        tools=tools,
    )
    bridge.start()
    try:
        for _ in range(3):
            reply = send(bridge, backend, CALL_PATH, {"tool": "bridge.explode"})
            assert body_of(reply)["error"]["code"] == "TOOL_FAILED"
        health = send(bridge, backend, HEALTH_PATH)
        assert body_of(health)["data"]["busy"] is False
    finally:
        bridge.stop()


def test_one_call_at_a_time_and_health_still_answers(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    tools = ToolRegistry()

    def slow(arguments: Mapping[str, Any]) -> Any:
        started.set()
        release.wait(10.0)
        return {"done": True}

    tools.add("bridge.slow", slow)
    backend = RecordingBackend()
    bridge = Bridge(
        BridgeConfig(
            home=tmp_path,
            kind="hython",
            verify_loopback=False,
            heartbeat_s=3600.0,
            dispatch_wait_s=0.2,
        ),
        backend=backend,
        tools=tools,
    )
    bridge.start()
    first: list[Any] = []

    def run_first() -> None:
        first.append(send(bridge, backend, CALL_PATH, {"tool": "bridge.slow"}))

    worker = threading.Thread(target=run_first)
    worker.start()
    try:
        assert started.wait(10.0)

        health = send(bridge, backend, HEALTH_PATH)
        data = body_of(health)["data"]
        assert health.status == 200
        assert data["busy"] is True
        assert data["current_op"] == "bridge.slow"
        assert data["current_op_elapsed_s"] >= 0

        second = send(bridge, backend, CALL_PATH, {"tool": "bridge.slow"})
        assert second.status == 200
        error = body_of(second)["error"]
        assert error["code"] == "SESSION_BUSY"
        assert error["details"]["current_op"] == "bridge.slow"
        assert error["details"]["elapsed_s"] >= 0
    finally:
        release.set()
        worker.join(10.0)
        bridge.stop()

    assert body_of(first[0])["data"] == {"done": True}
    assert houdini_lock().acquire(timeout=1.0)
    houdini_lock().release()


def test_a_call_that_will_not_wait_is_answered_at_once(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    tools = ToolRegistry()

    def slow(arguments: Mapping[str, Any]) -> Any:
        started.set()
        release.wait(10.0)
        return {"done": True}

    tools.add("bridge.slow", slow)
    backend = RecordingBackend()
    bridge = Bridge(
        BridgeConfig(
            home=tmp_path,
            kind="hython",
            verify_loopback=False,
            heartbeat_s=3600.0,
            dispatch_wait_s=30.0,
        ),
        backend=backend,
        tools=tools,
    )
    bridge.start()
    worker = threading.Thread(
        target=lambda: send(bridge, backend, CALL_PATH, {"tool": "bridge.slow"})
    )
    worker.start()
    try:
        assert started.wait(10.0)
        reply = send(bridge, backend, CALL_PATH, {"tool": "bridge.slow", "wait_s": 0})
        error = body_of(reply)["error"]
        assert error["code"] == "SESSION_BUSY"
        assert error["details"]["waited_s"] < 1.0
        assert error["details"]["wait_s"] == 0
    finally:
        release.set()
        worker.join(10.0)
        bridge.stop()


# Starting safely


def test_a_port_the_server_took_outside_the_range_stops_the_bridge(tmp_path: Path) -> None:
    class Wanderer(RecordingBackend):
        def start(self, port: int, *, in_background: bool = True) -> int:
            super().start(port, in_background=in_background)
            return 9999

    backend = Wanderer()
    bridge = Bridge(
        BridgeConfig(home=tmp_path, kind="hython", verify_loopback=False),
        backend=backend,
    )
    with pytest.raises(BridgeStartError):
        bridge.start()
    assert backend.stops == 1
    assert registry.list_entries(tmp_path) == []


def test_a_port_that_is_not_private_stops_the_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nscr_houdini_mcp.bridge.net import PrivacyProof

    monkeypatch.setattr(
        "nscr_houdini_mcp.bridge.app.prove_loopback_only",
        lambda port, **rest: PrivacyProof(False, True, ("192.0.2.7",), ("192.0.2.7",), "held"),
    )
    bridge, backend = make_bridge(tmp_path, verify_loopback=True)
    with pytest.raises(BridgeStartError) as raised:
        bridge.start()
    assert "192.0.2.7" in str(raised.value)
    assert backend.stops == 1
    assert bridge.port is None
    assert registry.list_entries(tmp_path) == []
    with store_module.Store(bridge.store_path) as store:
        assert store.list_sessions() == []


def test_a_port_that_could_not_be_proven_starts_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nscr_houdini_mcp.bridge.net import PrivacyProof

    monkeypatch.setattr(
        "nscr_houdini_mcp.bridge.app.prove_loopback_only",
        lambda port, **rest: PrivacyProof(True, False, (), (), "nothing to test against"),
    )
    bridge, backend = make_bridge(tmp_path, verify_loopback=True)
    bridge.start()
    try:
        assert bridge.privacy["proven"] is False
        data = body_of(send(bridge, backend, HEALTH_PATH))["data"]
        assert data["privacy"]["proven"] is False
        assert any("not be proven" in problem for problem in bridge.problems)
    finally:
        bridge.stop()


def test_handlers_cannot_be_added_once_the_server_runs(tmp_path: Path, driver: str) -> None:
    bridge, backend = make_bridge(tmp_path, driver=driver)
    bridge.start()
    try:
        with pytest.raises(RuntimeError):
            backend.register("/late", lambda request: None)
        with pytest.raises(RuntimeError):
            backend.start(bridge.port)
    finally:
        bridge.stop()


def test_the_tool_table_refuses_a_name_twice() -> None:
    tools = default_registry()
    with pytest.raises(ValueError):
        tools.add("bridge.ping", lambda arguments: None)
    names = tools.names()
    assert names == sorted(set(names), key=names.index)
    assert "bridge.ping" in tools
    assert len(tools) == len(names)


def _explode(arguments: Mapping[str, Any]) -> Any:
    raise ValueError("could not open /work/secret-scene_v012.hip")


# Each guard is the thing doing the refusing


def counting_bridge(tmp_path: Path, **overrides: Any) -> tuple[Bridge, RecordingBackend, list]:
    """A bridge whose one tool records every time it is reached."""
    reached: list[Mapping[str, Any]] = []
    tools = ToolRegistry()
    tools.add(
        "bridge.count", lambda arguments: reached.append(arguments) or {"count": len(reached)}
    )
    backend = RecordingBackend()
    settings: dict[str, Any] = {
        "home": tmp_path,
        "kind": "hython",
        "verify_loopback": False,
        "heartbeat_s": 3600.0,
    }
    settings.update(overrides)
    bridge = Bridge(BridgeConfig(**settings), backend=backend, tools=tools)
    bridge.start()
    return bridge, backend, reached


def test_the_browser_header_check_is_what_refuses_a_page(tmp_path: Path) -> None:
    bridge, backend, reached = counting_bridge(tmp_path)
    try:
        args = {"tool": "bridge.count"}
        headers = {"origin": "http://evil.example"}
        refused = send(bridge, backend, CALL_PATH, args, headers=headers)
        assert refused.status == 403
        assert reached == []

        # With the check taken away the same request runs the tool, so the
        # check is what stopped it and not something else.
        import nscr_houdini_mcp.bridge.app as app_module

        original = app_module.browser_header
        app_module.browser_header = lambda headers: None
        try:
            allowed = send(bridge, backend, CALL_PATH, args, headers=headers)
        finally:
            app_module.browser_header = original
        assert allowed.status == 200
        assert len(reached) == 1
    finally:
        bridge.stop()


def test_the_host_check_is_what_refuses_another_name(tmp_path: Path) -> None:
    bridge, backend, reached = counting_bridge(tmp_path)
    try:
        args = {"tool": "bridge.count"}
        headers = {"host": "evil.example:1"}
        assert send(bridge, backend, CALL_PATH, args, headers=headers).status == 403
        assert reached == []

        import nscr_houdini_mcp.bridge.app as app_module

        original = app_module.host_allowed
        app_module.host_allowed = lambda headers, port: True
        try:
            assert send(bridge, backend, CALL_PATH, args, headers=headers).status == 200
        finally:
            app_module.host_allowed = original
        assert len(reached) == 1
    finally:
        bridge.stop()


def test_the_signature_check_is_what_refuses_an_unsigned_request(tmp_path: Path) -> None:
    bridge, backend, reached = counting_bridge(tmp_path)
    try:
        args = {"tool": "bridge.count"}
        assert backend.send(build(bridge, CALL_PATH, args, sign=False)).status == 401
        assert reached == []

        original = bridge._verifier.check
        bridge._verifier.check = lambda *rest, **more: None
        try:
            assert backend.send(build(bridge, CALL_PATH, args, sign=False)).status == 200
        finally:
            bridge._verifier.check = original
        assert len(reached) == 1
    finally:
        bridge.stop()


def test_the_size_cap_is_what_refuses_a_large_body(tmp_path: Path) -> None:
    bridge, backend, reached = counting_bridge(tmp_path, max_body_bytes=200)
    try:
        args = {"tool": "bridge.count", "arguments": {"echo": "x" * 500}}
        assert send(bridge, backend, CALL_PATH, args).status == 413
        assert reached == []
    finally:
        bridge.stop()

    wide, backend, reached = counting_bridge(tmp_path, max_body_bytes=100_000)
    try:
        assert send(wide, backend, CALL_PATH, args).status == 200
        assert len(reached) == 1
    finally:
        wide.stop()


def test_the_depth_limit_is_what_refuses_a_nested_body(tmp_path: Path) -> None:
    nested = b'{"tool": "bridge.count", "arguments": {"echo": ' + b"[" * 200 + b"]" * 200 + b"}}"
    bridge, backend, reached = counting_bridge(tmp_path)
    try:
        reply = backend.send(build(bridge, CALL_PATH, body=nested))
        assert reply.status == 400
        assert body_of(reply)["error"]["code"] == "BODY_REFUSED"
        assert reached == []
    finally:
        bridge.stop()

    deep, backend, reached = counting_bridge(tmp_path, max_depth=1000)
    try:
        assert backend.send(build(deep, CALL_PATH, body=nested)).status == 200
        assert len(reached) == 1
    finally:
        deep.stop()


def test_the_content_type_check_is_what_refuses_a_form(tmp_path: Path) -> None:
    bridge, backend, reached = counting_bridge(tmp_path)
    try:
        reply = backend.send(
            build(bridge, CALL_PATH, {"tool": "bridge.count"}, content_type="text/plain")
        )
        assert reply.status == 415
        assert reached == []
        assert send(bridge, backend, CALL_PATH, {"tool": "bridge.count"}).status == 200
        assert len(reached) == 1
    finally:
        bridge.stop()


def test_a_start_that_fails_part_way_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*rest: Any, **more: Any) -> None:
        raise OSError("the disk said no")

    bridge, backend = make_bridge(tmp_path)
    monkeypatch.setattr("nscr_houdini_mcp.bridge.app.registry.write_entry", refuse)
    with pytest.raises(OSError):
        bridge.start()
    assert registry.list_entries(tmp_path) == []
    assert backend.stops == 1
    assert bridge.port is None
    assert bridge.running is False
    with store_module.Store(bridge.store_path) as store:
        assert store.list_sessions() == []


def test_every_step_of_stopping_runs_even_when_one_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    monkeypatch.setattr(
        bridge, "_end_session_row", lambda: (_ for _ in ()).throw(OSError("the store said no"))
    )
    problems = bridge.stop()
    assert any("end the session row" in problem for problem in problems)
    assert registry.list_entries(tmp_path) == []
    assert backend.stops == 1
    assert bridge.running is False


def test_stopping_a_bridge_that_is_already_stopped_says_nothing_went_wrong(
    tmp_path: Path,
) -> None:
    bridge, _ = make_bridge(tmp_path)
    bridge.start()
    assert bridge.stop() == []
    assert bridge.stop() == []


# Section: proving the port answers


def test_a_session_whose_port_answers_says_so_everywhere(tmp_path: Path) -> None:
    """Health read from inside the process cannot prove the port is served.

    So the session asks its own port the way a caller would, and what it finds
    goes into health, into the session file and into the store row.
    """
    bridge, _ = make_bridge(tmp_path, driver="stdlib", heartbeat_s=3600.0)
    bridge.start()
    try:
        assert bridge.check_transport() is True
        state = bridge.transport_state()
        assert state["last_self_check_ok"] is True
        assert state["last_self_check_age_s"] < 5.0

        entry = registry.find_entry(tmp_path, bridge.session_id)
        assert entry["last_self_check_ok"] is True

        session = client_module.Session.open(tmp_path, bridge.session_id)
        assert session.transport_ok is True
        assert session.deaf() is False

        bridge._round()
        with store_module.Store(bridge.store_path) as store:
            record = store.get_session(bridge.session_id)
        assert record.state == "live"
        assert record.transport_ok is True
        assert record.transport_checked_at is not None
    finally:
        bridge.stop()


def test_a_session_nothing_can_reach_reports_itself_unresponsive(tmp_path: Path) -> None:
    """The process is fine, the heartbeat is fine, and the port is deaf."""
    bridge, backend = make_bridge(tmp_path, driver="stdlib", heartbeat_s=3600.0)
    bridge.start()
    try:
        assert bridge.check_transport() is True

        # The server goes away under the bridge, which is what a transport
        # that has died leaves behind: everything else still works.
        backend.stop()

        assert bridge.check_transport() is False
        assert bridge.transport_state()["last_self_check_ok"] is False

        bridge._round()
        with store_module.Store(bridge.store_path) as store:
            record = store.get_session(bridge.session_id)
        assert record.state == "unresponsive"
        assert record.transport_ok is False

        session = client_module.Session.open(tmp_path, bridge.session_id)
        assert session.transport_ok is False
        assert session.deaf() is True
    finally:
        bridge.stop()


def test_health_carries_the_self_check(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path, driver="stdlib", heartbeat_s=3600.0)
    bridge.start()
    try:
        bridge.check_transport()
        data = body_of(send(bridge, backend, HEALTH_PATH))["data"]
        assert data["last_self_check_ok"] is True
        assert data["last_self_check_at"] is not None
        assert data["last_self_check_age_s"] >= 0.0
    finally:
        bridge.stop()


def test_a_session_that_has_not_asked_yet_says_nothing_either_way(tmp_path: Path) -> None:
    """Nothing is not the same as a failure, and is never reported as one."""
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        bridge.transport_ok = None
        bridge.transport_checked_at = None
        data = body_of(send(bridge, backend, HEALTH_PATH))["data"]
        assert data["last_self_check_ok"] is None
        assert data["last_self_check_age_s"] is None
        assert client_module.Session("s", "t", 1).deaf() is False
    finally:
        bridge.stop()


def test_nothing_listening_on_the_port_is_a_failed_check(tmp_path: Path) -> None:
    """The recording backend serves no socket, and the check says so."""
    bridge, _ = make_bridge(tmp_path, heartbeat_s=3600.0)
    bridge.start()
    try:
        assert bridge.check_transport() is False
    finally:
        bridge.stop()


# Section: the main thread of a session with a user interface


def test_health_reports_the_main_thread_and_answers_under_budget_while_it_is_held(
    tmp_path: Path,
) -> None:
    """Health is the one answer a cook cannot delay, and it says what is going on."""
    scene = Scene()
    bridge, backend = make_bridge(tmp_path, kind="gui", hou=scene.module(), main_thread_stale_s=0.2)
    scene.ui.start()
    bridge.start()
    begun, release = scene.ui.hold()
    try:
        # The hold lasts until this test lets it go, so the reads below cannot
        # outrun it however slow the machine is.
        assert begun.wait(5.0)

        slowest = 0.0
        ages: list[float] = []
        away: list[bool] = []
        for _ in range(25):
            began = time.monotonic()
            reply = send(bridge, backend, HEALTH_PATH)
            slowest = max(slowest, time.monotonic() - began)
            main_thread = body_of(reply)["data"]["main_thread"]
            ages.append(main_thread["pulse_age_s"])
            away.append(main_thread["away"])
            time.sleep(0.02)

        assert slowest < 0.05
        assert ages[-1] > ages[0]
        assert away[-1] is True
        assert away[0] is False
    finally:
        release.set()
        bridge.stop()
        scene.ui.stop()


def test_a_gui_bridge_installs_the_pulse_at_start_and_removes_it_at_stop(
    tmp_path: Path,
) -> None:
    scene = Scene()
    bridge, _ = make_bridge(tmp_path, kind="gui", hou=scene.module())
    scene.ui.start()
    bridge.start()
    try:
        assert len(scene.ui.eventLoopCallbacks()) == 1
        assert bridge.main_thread is not None
        assert bridge.dispatcher.state()["main_thread"]["installed"] is True
        scene.ui.cook(0.3)
        _wait_for(lambda: scene.ui.ran_on != [])
        assert bridge.stop() == []
        # Stopping says nothing to Houdini. The callback takes itself off the
        # next time the main thread runs it.
        assert bridge.pulse.installed is False
        _wait_for(lambda: scene.ui.eventLoopCallbacks() == ())
    finally:
        scene.ui.stop()


def test_stopping_during_a_cook_returns_at_once_and_says_nothing_to_houdini(
    tmp_path: Path,
) -> None:
    """Nothing on the stop path waits for the object model lock.

    Every `hou` call stopping would make is a wait for a cook to end, so the
    stop sets flags instead and the callbacks take themselves off from the
    main thread when it is free again.
    """
    scene = Scene()
    bridge, _ = make_bridge(tmp_path, kind="gui", hou=scene.module())
    scene.ui.start()
    bridge.start()
    try:
        scene_callbacks = len(scene.hipFile._callbacks)
        assert scene_callbacks >= 1
        assert len(scene.ui.eventLoopCallbacks()) == 1

        scene.ui.cook(3.0)
        _wait_for(lambda: scene.ui.ran_on != [])

        began = time.monotonic()
        assert bridge.stop() == []
        assert time.monotonic() - began < 1.0

        # Nothing was taken off yet, and nothing fires while the cook runs.
        assert bridge.pulse.installed is False
        assert bridge.pulse.registered is True
        before = scene.ui.ticks

        # Once the cook ends the callbacks take themselves off, and the pulse
        # stamps nothing after that.
        _wait_for(lambda: scene.ui.eventLoopCallbacks() == (), timeout_s=15.0)
        assert bridge.pulse.registered is False
        assert bridge.pulse.age_s() is None
        assert scene.ui.ticks > before

        scene.hipFile.clear()
        _wait_for(lambda: len(scene.hipFile._callbacks) == 0, timeout_s=15.0)
        assert bridge.scene_epoch == 0
    finally:
        scene.ui.stop()


def _ours(scene: Scene) -> tuple[int, int]:
    """What Houdini holds: scene event callbacks, then event loop callbacks."""
    return len(scene.hipFile._callbacks), len(scene.ui.eventLoopCallbacks())


@pytest.mark.parametrize("way", ["either", "post", "tick"])
def test_a_stopped_gui_bridge_takes_its_callbacks_off_without_a_scene_event(
    tmp_path: Path, way: str
) -> None:
    """The main thread's next visit takes everything of ours off.

    Either the pulse's last tick or the runner's last post does it, whichever
    lands first, and each is enough alone. No scene is loaded or cleared.
    """
    scene = Scene()
    module = scene.module()
    ui = scene.ui
    if way == "post":
        # No loop callbacks here, so the last post is the only way in.
        module.ui = SimpleNamespace(postEventCallback=ui.postEventCallback)
    elif way == "tick":
        # Posts that never land, so the pulse is the only way in.
        module.ui = SimpleNamespace(
            postEventCallback=lambda callback: None,
            addEventLoopCallback=ui.addEventLoopCallback,
            removeEventLoopCallback=ui.removeEventLoopCallback,
        )
    baseline = _ours(scene)
    bridge, _ = make_bridge(tmp_path, kind="gui", hou=module)
    ui.start()
    try:
        bridge.start()
        pulses = 0 if way == "post" else 1
        assert _ours(scene) == (baseline[0] + 2, baseline[1] + pulses)
        assert bridge.stop() == []
        _wait_for(lambda: _ours(scene) == baseline, timeout_s=2.0)
        assert host.leftovers(module) == 0
        assert bridge.scene_epoch == 0
    finally:
        ui.stop()


def test_bridges_started_and_stopped_in_turn_leave_nothing_behind(tmp_path: Path) -> None:
    scene = Scene()
    module = scene.module()
    baseline = _ours(scene)
    scene.ui.start()
    try:
        for _ in range(3):
            bridge, _ = make_bridge(tmp_path, kind="gui", hou=module)
            bridge.start()
            assert _ours(scene) == (baseline[0] + 2, baseline[1] + 1)
            assert bridge.stop() == []
            _wait_for(lambda: _ours(scene) == baseline, timeout_s=2.0)
        assert host.leftovers(module) == 0
    finally:
        scene.ui.stop()


def test_a_bridge_started_before_the_main_thread_came_back_never_doubles_up(
    tmp_path: Path,
) -> None:
    """A stop and a start in one go, as a script run on the main thread does.

    The main thread has not visited in between, so the first bridge's
    callbacks are still registered when the second starts. The second takes
    them off before it registers its own.
    """
    scene = Scene()
    module = scene.module()
    baseline = _ours(scene)
    first, _ = make_bridge(tmp_path, kind="gui", hou=module)
    first.start()
    assert first.stop() == []
    assert _ours(scene) == (baseline[0] + 2, baseline[1] + 1)

    second, _ = make_bridge(tmp_path, kind="gui", hou=module)
    second.start()
    try:
        assert _ours(scene) == (baseline[0] + 2, baseline[1] + 1)
        scene.hipFile.clear()
        assert second.scene_epoch == 1
        assert first.scene_epoch == 0
    finally:
        assert second.stop() == []

    scene.ui.start()
    try:
        _wait_for(lambda: _ours(scene) == baseline, timeout_s=2.0)
        assert host.leftovers(module) == 0
    finally:
        scene.ui.stop()


def test_a_bridge_with_no_user_interface_takes_its_callbacks_off_as_it_stops(
    tmp_path: Path,
) -> None:
    """No event loop holds the object model lock there, so nothing waits."""
    scene = Scene()
    module = scene.module()
    baseline = _ours(scene)
    bridge, _ = make_bridge(tmp_path, kind="hython", hou=module)
    bridge.start()
    assert _ours(scene)[0] == baseline[0] + 2
    assert bridge.stop() == []
    assert _ours(scene) == baseline
    assert host.leftovers(module) == 0


def test_a_quit_hook_installed_twice_or_taken_back_is_registered_once() -> None:
    scene = Scene()
    module = scene.module()
    quits: list[bool] = []
    hook = host.QuitHook(lambda: quits.append(True), hou=module)
    assert hook.install() is True
    assert hook.install() is True
    assert len(scene.hipFile._callbacks) == 1

    hook.remove()
    assert host.leftovers(module) == 1
    assert hook.install() is True
    assert len(scene.hipFile._callbacks) == 1
    assert host.leftovers(module) == 0

    scene.hipFile._fire("BeforeQuit")
    assert quits == [True]
    hook.remove()
    host.clear_leftovers(module)
    assert scene.hipFile._callbacks == []


def test_a_bridge_whose_pulse_cannot_install_still_starts_and_says_so(tmp_path: Path) -> None:
    """A session that will not be watched is still a session that answers."""
    scene = Scene()
    module = scene.module()
    module.ui = SimpleNamespace(postEventCallback=lambda callback: None)
    bridge, backend = make_bridge(tmp_path, kind="gui", hou=module)
    bridge.start()
    try:
        assert any("watching the main thread" in problem for problem in bridge.problems)
        assert bridge.dispatcher.state()["main_thread"]["installed"] is False
        assert body_of(send(bridge, backend, HEALTH_PATH))["data"]["status"] == "ok"
    finally:
        bridge.stop()


def _wait_for(ready: Any, timeout_s: float = 10.0) -> None:
    """Wait for something another thread is about to do."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.005)
    raise AssertionError("waited too long")
