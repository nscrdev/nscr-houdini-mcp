from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import registry, security, signing
from nscr_houdini_mcp.bridge.app import Bridge, BridgeConfig, BridgeError, houdini_lock
from nscr_houdini_mcp.bridge.handlers import ToolRegistry, default_registry
from nscr_houdini_mcp.bridge.serving import (
    CALL_PATH,
    HEALTH_PATH,
    JSON_TYPE,
    RawRequest,
    RecordingBackend,
)


def make_bridge(home: Path, **overrides: Any) -> tuple[Bridge, RecordingBackend]:
    backend = RecordingBackend()
    settings: dict[str, Any] = {
        "home": home,
        "kind": "hython",
        "verify_loopback": False,
        "heartbeat_s": 3600.0,
        "facts": {"houdini_version": "22.0.0", "hfs": "/hfs", "hip_path": None},
    }
    settings.update(overrides)
    bridge = Bridge(BridgeConfig(**settings), backend=backend)
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


def test_health_answers_from_memory(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
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
        assert data["tools"] == ["bridge.ping"]
    finally:
        bridge.stop()


def test_a_call_reaches_its_tool_and_carries_the_trace_back(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
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
def test_an_unsigned_request_is_refused(tmp_path: Path, path: str) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        reply = backend.send(build(bridge, path, envelope(), sign=False))
        assert reply.status == 401
        assert body_of(reply)["error"]["code"] == "UNAUTHORIZED"
        assert bridge._token not in reply.body.decode("utf-8")
    finally:
        bridge.stop()


def test_a_signature_from_another_token_is_refused(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, envelope(), token=security.mint_token())
        assert reply.status == 401
    finally:
        bridge.stop()


def test_a_signature_over_a_different_body_is_refused(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
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


def test_the_same_signature_cannot_be_sent_twice(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        request = build(bridge, CALL_PATH, envelope())
        assert backend.send(request).status == 200
        assert backend.send(request).status == 401
    finally:
        bridge.stop()


def test_a_signature_from_outside_the_time_window_is_refused(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
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
def test_a_request_from_a_page_is_refused_even_when_signed(tmp_path: Path, header: str) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, envelope(), headers={header: "http://x.example"})
        assert reply.status == 403
        assert body_of(reply)["error"]["code"] == "FORBIDDEN"
    finally:
        bridge.stop()


@pytest.mark.parametrize("host", ["evil.example:18100", "", "127.0.0.1", "127.0.0.1:1"])
def test_another_host_is_refused_even_when_signed(tmp_path: Path, host: str) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, envelope(), headers={"host": host})
        assert reply.status == 403
    finally:
        bridge.stop()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_this_machine_by_either_loopback_name_is_allowed(tmp_path: Path, host: str) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        headers = {"host": f"{host}:{bridge.port}"}
        assert send(bridge, backend, HEALTH_PATH, headers=headers).status == 200
    finally:
        bridge.stop()


def test_a_form_post_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
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


def test_a_body_over_the_cap_is_refused_before_it_is_read(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path, max_body_bytes=1024)
    bridge.start()
    try:
        reply = backend.send(build(bridge, CALL_PATH, body=b"x" * 2048, sign=False))
        assert reply.status == 413
    finally:
        bridge.stop()


def test_a_body_nested_past_the_limit_is_refused_without_parsing(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        reply = backend.send(build(bridge, CALL_PATH, body=b"[" * 5000 + b"]" * 5000))
        assert reply.status == 400
        assert body_of(reply)["error"]["code"] == "BODY_REFUSED"
    finally:
        bridge.stop()


def test_brackets_inside_an_argument_do_not_count_as_nesting(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        code = "{" * 200 + "[" * 200
        reply = send(bridge, backend, CALL_PATH, envelope(arguments={"echo": code}))
        assert reply.status == 200
        assert body_of(reply)["data"]["echo"] == code
    finally:
        bridge.stop()


def test_a_method_other_than_post_is_refused(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
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
        for _ in range(500):
            if not bridge.running and registry.list_entries(tmp_path) == []:
                break
            threading.Event().wait(0.01)
        assert bridge.running is False
        assert registry.list_entries(tmp_path) == []
    finally:
        bridge.stop()


def test_every_answer_is_signed_so_a_squatter_cannot_pass_for_the_bridge(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
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


def test_a_request_that_cannot_be_read_comes_back_as_a_bad_request(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        reply = send(bridge, backend, CALL_PATH, {"tool": ""})
        assert reply.status == 400
        assert body_of(reply)["error"]["code"] == "BAD_ENVELOPE"
    finally:
        bridge.stop()


def test_another_session_id_and_an_unknown_tool_fail_without_failing_the_call(
    tmp_path: Path,
) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        wrong = send(bridge, backend, CALL_PATH, envelope(session_id="somebody-else"))
        assert wrong.status == 200
        assert body_of(wrong)["error"]["code"] == "SESSION_UNKNOWN"
        assert body_of(wrong)["error"]["details"]["session_id"] == bridge.session_id

        unknown = send(bridge, backend, CALL_PATH, envelope(tool="nothing.here"))
        assert unknown.status == 200
        assert body_of(unknown)["error"]["code"] == "TOOL_UNKNOWN"
        assert body_of(unknown)["error"]["details"]["tools"] == ["bridge.ping"]
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
    with pytest.raises(BridgeError):
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
    with pytest.raises(BridgeError) as raised:
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


def test_handlers_cannot_be_added_once_the_server_runs(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
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
    assert tools.names() == ["bridge.ping"]
    assert "bridge.ping" in tools
    assert len(tools) == 1


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
