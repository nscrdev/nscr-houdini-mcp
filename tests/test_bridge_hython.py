"""The bridge in a real Houdini.

Skipped, not failed, when there is no Houdini on this machine, which is the
case on the build machines. One hython at a time, started here and stopped
here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, net, registry, security
from nscr_houdini_mcp.bridge.launcher import HythonBridge, hython_available

pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
]

# Its own range, so a bridge started by hand keeps the port it has.
PORT_RANGE = (18300, 18349)


@pytest.fixture(scope="module")
def home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("bridge-home")


@pytest.fixture(scope="module")
def bridge(home: Path) -> Iterator[HythonBridge]:
    started = HythonBridge(home=home, port_range=PORT_RANGE)
    try:
        started.start()
        yield started
    finally:
        started.stop()


def test_health_answers_and_names_the_session(bridge: HythonBridge) -> None:
    answer = bridge.health()
    assert answer.status == 200
    data = answer.payload["data"]
    assert data["status"] == "ok"
    assert data["session_id"] == bridge.session_id
    assert data["kind"] == "hython"
    assert data["port"] == bridge.port
    assert data["busy"] is False
    assert data["tools"] == ["bridge.ping"]


def test_a_call_runs_a_tool(bridge: HythonBridge) -> None:
    answer = bridge.call(
        "bridge.ping",
        arguments={"echo": "hello"},
        session_id=bridge.session_id,
        operation_id="op-1",
    )
    assert answer.status == 200
    assert answer.payload["ok"] is True
    assert answer.payload["data"] == {"pong": True, "echo": "hello"}
    assert answer.payload["operation_id"] == "op-1"


@pytest.mark.parametrize("token", [None, "wrong"])
def test_no_token_and_a_wrong_token_are_refused(bridge: HythonBridge, token: Any) -> None:
    answer = client.post(
        bridge.port,
        "mcp.call",
        arguments={"envelope": {"tool": "bridge.ping"}},
        token=token,
    )
    assert answer.status == 401
    assert answer.payload["error"]["code"] == "UNAUTHORIZED"
    assert bridge.token not in str(answer.payload)


def test_health_is_refused_without_the_token(bridge: HythonBridge) -> None:
    assert client.post(bridge.port, "mcp.health").status == 401


@pytest.mark.parametrize("header", ["Origin", "Referer"])
def test_a_request_from_a_page_is_refused(bridge: HythonBridge, header: str) -> None:
    answer = client.post(
        bridge.port,
        "mcp.health",
        token=bridge.token,
        headers={header: "http://evil.example"},
    )
    assert answer.status == 403
    assert answer.payload["error"]["code"] == "FORBIDDEN"
    assert not [name for name in answer.headers if name.startswith("access-control-allow")]


def test_the_port_answers_on_loopback_alone(bridge: HythonBridge) -> None:
    assert net.can_connect(net.LOOPBACK, bridge.port) is True
    assert net.reachable_from_outside(bridge.port) == []
    # Stronger than a connection attempt: a bind that succeeds proves nothing
    # is listening there, and a firewall cannot flatter the result.
    assert net.addresses_holding_port(bridge.port) == []


def test_the_socket_listing_agrees(bridge: HythonBridge) -> None:
    """A second opinion, from whatever listing this machine will hand over.

    Some systems only show another process's sockets to an administrator, so
    this skips rather than failing. The bind probe above needs no permission
    and is the check that always runs.
    """
    listings = _listening_addresses(bridge.port)
    if listings is None:
        pytest.skip("no socket listing available to this user")
    assert listings, "the port is not in the socket listing"
    for address in listings:
        assert address.startswith("127.") or address == "::1", address


def test_no_answer_carries_a_cross_origin_header_or_names_the_build(
    bridge: HythonBridge,
) -> None:
    answer = bridge.health()
    assert answer.status == 200
    assert not [name for name in answer.headers if name.startswith("access-control-allow")]
    server = answer.headers.get("server", "")
    assert "22.0" not in server, server


def test_the_session_file_is_private_and_holds_the_token(bridge: HythonBridge, home: Path) -> None:
    path = registry.entry_path(home, bridge.session_id)
    assert security.is_private(path)
    entry = registry.read_entry(path)
    assert entry["token"] == bridge.token
    assert entry["kind"] == "hython"
    assert entry["houdini_version"]


def test_the_session_is_in_the_store_and_goes_when_the_process_quits(
    bridge: HythonBridge, home: Path
) -> None:
    store_path = home / store_module.STORE_FILE_NAME
    with store_module.Store(store_path) as store:
        live = store.resolve_session(bridge.session_id)
        assert live is not None
        assert live.state == "live"
        assert live.port == bridge.port
        assert live.pid == bridge.process.pid

    assert bridge.stop() == 0

    with store_module.Store(store_path) as store:
        assert store.list_sessions() == []
        assert store.get_session(bridge.session_id).state == "gone"
    assert registry.list_entries(home) == []
    with pytest.raises(client.BridgeUnreachable):
        client.post(bridge.port, "mcp.health", token=bridge.token, timeout_s=5.0)


def _listening_addresses(port: int) -> list[str] | None:
    """Addresses this machine says are listening on a port, if it can say."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError):
        return None
    found = []
    for connection in connections:
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port:
            found.append(connection.laddr.ip)
    return found
