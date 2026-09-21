from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import registry, security
from nscr_houdini_mcp.bridge.app import Bridge, BridgeConfig, BridgeError, houdini_lock
from nscr_houdini_mcp.bridge.envelope import TOKEN_HEADER
from nscr_houdini_mcp.bridge.handlers import ToolRegistry, default_registry
from nscr_houdini_mcp.bridge.serving import RecordingBackend


def make_bridge(home: Path, **overrides: Any) -> tuple[Bridge, RecordingBackend]:
    backend = RecordingBackend()
    settings = {
        "home": home,
        "kind": "hython",
        "verify_loopback": False,
        "heartbeat_s": 3600.0,
        "facts": {"houdini_version": "22.0.0", "hfs": "/hfs", "hip_path": None},
    }
    settings.update(overrides)
    bridge = Bridge(BridgeConfig(**settings), backend=backend)
    return bridge, backend


def envelope(bridge: Bridge, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool": "bridge.ping", "arguments": {"echo": 1}}
    payload.update(fields)
    return payload


def token_headers(bridge: Bridge) -> dict[str, str]:
    return {TOKEN_HEADER: bridge._token}


def call(backend: RecordingBackend, bridge: Bridge, **fields: Any) -> Any:
    return backend.post("call", {"envelope": envelope(bridge, **fields)}, token_headers(bridge))


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
        assert sorted(backend.endpoints) == ["call", "health"]

        entries = registry.list_entries(tmp_path)
        assert [entry["session_id"] for entry in entries] == [bridge.session_id]
        assert entries[0]["port"] == bridge.port
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
        reply = backend.post("health", {}, token_headers(bridge))
        assert reply.status == 200
        data = reply.payload["data"]
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
        reply = call(
            backend,
            bridge,
            session_id=bridge.session_id,
            scene_epoch=0,
            operation_id="op-1",
        )
        assert reply.status == 200
        assert reply.payload["ok"] is True
        assert reply.payload["data"] == {"pong": True, "echo": 1}
        assert reply.payload["operation_id"] == "op-1"
        assert reply.payload["scene_epoch"] == 0
        assert reply.payload["timing_ms"] >= 0
    finally:
        bridge.stop()


@pytest.mark.parametrize("endpoint", ["health", "call"])
def test_no_token_and_a_wrong_token_are_both_refused(tmp_path: Path, endpoint: str) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        payload = {"envelope": envelope(bridge)} if endpoint == "call" else {}
        for headers in ({}, {TOKEN_HEADER: "wrong"}, {TOKEN_HEADER: bridge._token[:-1]}):
            reply = backend.post(endpoint, payload, headers)
            assert reply.status == 401
            assert reply.payload["error"]["code"] == "UNAUTHORIZED"
            assert bridge._token not in str(reply.payload)
    finally:
        bridge.stop()


@pytest.mark.parametrize("header", ["Origin", "Referer"])
def test_a_request_from_a_page_is_refused_even_with_the_token(tmp_path: Path, header: str) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        headers = {**token_headers(bridge), header: "http://evil.example"}
        reply = backend.post("call", {"envelope": envelope(bridge)}, headers)
        assert reply.status == 403
        assert reply.payload["error"]["code"] == "FORBIDDEN"
    finally:
        bridge.stop()


def test_the_token_may_travel_in_the_envelope(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        payload = {"envelope": envelope(bridge, token=bridge._token)}
        assert backend.post("call", payload, {}).status == 200
    finally:
        bridge.stop()


def test_a_request_that_cannot_be_read_comes_back_as_a_bad_request(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        reply = backend.post("call", {"envelope": {"tool": ""}}, token_headers(bridge))
        assert reply.status == 400
        assert reply.payload["error"]["code"] == "BAD_ENVELOPE"
    finally:
        bridge.stop()


def test_another_session_id_and_an_unknown_tool_fail_without_failing_the_call(
    tmp_path: Path,
) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        wrong_session = call(backend, bridge, session_id="somebody-else")
        assert wrong_session.status == 200
        assert wrong_session.payload["error"]["code"] == "SESSION_UNKNOWN"
        assert wrong_session.payload["error"]["details"]["session_id"] == bridge.session_id

        unknown = call(backend, bridge, tool="nothing.here")
        assert unknown.status == 200
        assert unknown.payload["error"]["code"] == "TOOL_UNKNOWN"
        assert unknown.payload["error"]["details"]["tools"] == ["bridge.ping"]
    finally:
        bridge.stop()


def test_a_tool_that_raises_is_reported_and_the_bridge_stays_up(tmp_path: Path) -> None:
    tools = ToolRegistry()

    def explode(arguments: Mapping[str, Any]) -> Any:
        raise ValueError("no")

    tools.add("bridge.explode", explode)
    backend = RecordingBackend()
    bridge = Bridge(
        BridgeConfig(home=tmp_path, kind="hython", verify_loopback=False, heartbeat_s=3600.0),
        backend=backend,
        tools=tools,
    )
    bridge.start()
    try:
        reply = backend.post(
            "call",
            {"envelope": {"tool": "bridge.explode"}},
            token_headers(bridge),
        )
        assert reply.status == 200
        assert reply.payload["error"]["code"] == "TOOL_FAILED"
        assert "ValueError: no" in reply.payload["error"]["message"]
        assert bridge.running is True
    finally:
        bridge.stop()


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


def test_a_port_the_network_can_reach_stops_the_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "nscr_houdini_mcp.bridge.app.reachable_from_outside",
        lambda port, **rest: ["192.0.2.7"],
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


def test_handlers_cannot_be_added_once_the_server_runs(tmp_path: Path) -> None:
    bridge, backend = make_bridge(tmp_path)
    bridge.start()
    try:
        with pytest.raises(RuntimeError):
            backend.register("late", lambda headers, payload: None)
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
    headers = token_headers(bridge)
    first: list[Any] = []

    def run_first() -> None:
        first.append(backend.post("call", {"envelope": {"tool": "bridge.slow"}}, headers))

    worker = threading.Thread(target=run_first)
    worker.start()
    try:
        assert started.wait(10.0)

        health = backend.post("health", {}, headers)
        assert health.status == 200
        assert health.payload["data"]["busy"] is True
        assert health.payload["data"]["current_op"] == "bridge.slow"

        second = backend.post("call", {"envelope": {"tool": "bridge.slow"}}, headers)
        assert second.status == 200
        assert second.payload["error"]["code"] == "SESSION_BUSY"
        assert second.payload["error"]["details"]["current_op"] == "bridge.slow"
    finally:
        release.set()
        worker.join(10.0)
        bridge.stop()

    assert first[0].payload["data"] == {"done": True}
    assert houdini_lock().acquire(timeout=1.0)
    houdini_lock().release()


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
            reply = backend.post(
                "call", {"envelope": {"tool": "bridge.explode"}}, token_headers(bridge)
            )
            assert reply.payload["error"]["code"] == "TOOL_FAILED"
        health = backend.post("health", {}, token_headers(bridge))
        assert health.payload["data"]["busy"] is False
    finally:
        bridge.stop()


def _explode(arguments: Mapping[str, Any]) -> Any:
    raise ValueError("no")
