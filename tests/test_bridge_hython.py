"""The bridge in a real Houdini.

Skipped, not failed, when there is no Houdini on this machine, which is the
case on the build machines. One hython at a time, started here and stopped
here, and never the one a person is working in.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, net, registry, security, signing
from nscr_houdini_mcp.bridge.launcher import HythonBridge, hython_available
from nscr_houdini_mcp.bridge.serving import CALL_PATH, HEALTH_PATH, JSON_TYPE

pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
]

# Its own range, so a bridge started by hand keeps the port it has.
PORT_RANGE = (18300, 18349)

FORM_TYPE = "application/x-www-form-urlencoded"

# How long a dropped answer is held, and how long the caller waits for one
# before it decides the answer is lost. The caller has to give up first.
HOLD_S = 8.0
GIVE_UP_S = 2.0


@pytest.fixture(scope="module")
def home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("bridge-home")


@pytest.fixture(scope="module")
def bridge(home: Path) -> Iterator[HythonBridge]:
    started = HythonBridge(
        home=home, port_range=PORT_RANGE, extra_args=["--drop-reply-s", str(HOLD_S)]
    )
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
    assert data["tools"][0] == "bridge.ping"
    assert {"scene.info", "node.create"} <= set(data["tools"])


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


# Tools that touch the scene


def scene_info(bridge: HythonBridge) -> dict:
    answer = bridge.call("scene.info")
    assert answer.payload["ok"] is True, answer.payload
    return answer.payload["data"]


def test_scene_info_reads_the_open_scene(bridge: HythonBridge) -> None:
    data = scene_info(bridge)
    assert data["houdini_version"].startswith("22.")
    assert data["frame"] == 1.0
    assert data["fps"] > 0
    assert data["kind"] == "hython"
    assert "/obj" in data["nodes"]
    assert data["scene_epoch"] == 0
    assert data["session_id"] == bridge.session_id
    # A headless session says yes to unsaved whatever has happened, so the
    # bridge does not pass that on.
    assert data["unsaved"] is None


def test_node_create_makes_a_node_that_is_really_there(bridge: HythonBridge) -> None:
    before = scene_info(bridge)
    answer = bridge.call(
        "node.create",
        arguments={"parent": "/obj", "type": "geo", "name": "made_here", "parms": {"tx": 2.0}},
    )
    assert answer.payload["ok"] is True, answer.payload
    data = answer.payload["data"]
    assert data["path"] == "/obj/made_here"
    assert data["type"] == "geo"
    assert data["parms_set"] == ["tx"]
    assert answer.payload["undo"]["recorded"] is True
    assert answer.payload["undo"]["rolled_back"] is False

    after = scene_info(bridge)
    assert after["nodes"]["/obj"] == before["nodes"]["/obj"] + 1
    # One call by the agent is one undo step for the artist.
    assert after["undo_entries"] == before["undo_entries"] + 1


def test_a_call_that_fails_part_way_takes_its_own_edits_back(bridge: HythonBridge) -> None:
    before = scene_info(bridge)
    answer = bridge.call("bridge.selfcheck", arguments={"creates": 2, "fail_at": 2}, timeout_s=30.0)
    error = answer.payload["error"]
    assert error["code"] == "TOOL_FAILED"
    assert error["details"]["rolled_back"] is True

    after = scene_info(bridge)
    assert after["nodes"]["/obj"] == before["nodes"]["/obj"]
    assert after["undo_entries"] == before["undo_entries"]


def test_a_call_that_fails_before_it_changes_anything_undoes_nothing(
    bridge: HythonBridge,
) -> None:
    before = scene_info(bridge)
    answer = bridge.call("bridge.selfcheck", arguments={"creates": 1, "fail_at": 1})
    assert answer.payload["error"]["details"]["rolled_back"] is False
    after = scene_info(bridge)
    assert after["undo_entries"] == before["undo_entries"]


def test_work_that_outlives_its_timeout_answers_and_carries_on(bridge: HythonBridge) -> None:
    answer = bridge.call("bridge.selfcheck", arguments={"sleep_s": 4.0}, timeout_s=0.5)
    error = answer.payload["error"]
    assert error["code"] == "TIMEOUT"
    assert error["details"]["still_running"] is True
    operation_id = error["details"]["operation_id"]

    health = bridge.health().payload["data"]
    assert health["busy"] is True
    assert health["current_op"] == "bridge.selfcheck"
    assert health["current_op_id"] == operation_id
    assert health["current_op_timed_out"] is True

    busy = bridge.call("bridge.ping", skip_if_busy=True)
    assert busy.payload["error"]["code"] == "SESSION_BUSY"

    # The work was never interrupted, so a call that waits for it gets through.
    later = bridge.call("bridge.ping", arguments={"echo": "after"}, wait_s=20.0)
    assert later.payload["ok"] is True, later.payload
    assert later.payload["data"]["echo"] == "after"
    assert bridge.health().payload["data"]["busy"] is False


def test_a_running_call_stops_when_it_is_asked_to(bridge: HythonBridge) -> None:
    answers: list[Any] = []
    holder = threading.Thread(
        target=lambda: answers.append(
            bridge.call("bridge.selfcheck", arguments={"sleep_s": 45.0}, timeout_s=60.0)
        )
    )
    holder.start()
    try:
        _until(lambda: bridge.health().payload["data"]["busy"] is True)
        operation_id = bridge.health().payload["data"]["current_op_id"]
        asked = bridge.call("bridge.cancel", arguments={"operation_id": operation_id})
        assert asked.payload["ok"] is True, asked.payload
        assert asked.payload["data"]["asked"] is True
    finally:
        holder.join(120.0)

    reply = answers[0]
    assert reply.payload["ok"] is True, reply.payload
    assert reply.payload["data"]["stopped_early"] is True
    assert reply.payload["data"]["slept_s"] < 45.0
    assert bridge.health().payload["data"]["busy"] is False


def test_the_self_check_is_only_in_a_worker_started_to_be_driven(bridge: HythonBridge) -> None:
    """It is on here because the launcher asked for it, and off by default."""
    assert "bridge.selfcheck" in bridge.health().payload["data"]["tools"]
    assert "bridge.cancel" in bridge.health().payload["data"]["tools"]


def test_a_misspelled_argument_name_comes_back_with_the_closest_one(
    bridge: HythonBridge,
) -> None:
    answer = bridge.call("node.create", arguments={"paren": "/obj", "type": "geo"})
    error = answer.payload["error"]
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["did_you_mean"][0] == "parent"
    assert error["hint"]


def test_a_parameter_name_the_node_does_not_have_comes_back_with_the_closest_ones(
    bridge: HythonBridge,
) -> None:
    before = scene_info(bridge)
    answer = bridge.call(
        "node.create",
        arguments={"parent": "/obj", "type": "geo", "parms": {"tix": 1.0}},
    )
    error = answer.payload["error"]
    assert error["code"] == "PARM_NOT_FOUND"
    assert "tx" in error["details"]["did_you_mean"]
    assert error["details"]["rolled_back"] is True
    after = scene_info(bridge)
    assert after["nodes"]["/obj"] == before["nodes"]["/obj"]


def test_a_name_that_is_not_a_tool_comes_back_with_the_closest_one(
    bridge: HythonBridge,
) -> None:
    answer = bridge.call("scene.inf")
    error = answer.payload["error"]
    assert error["code"] == "UNKNOWN_TOOL"
    assert error["details"]["did_you_mean"] == ["scene.info"]


def test_waiting_calls_are_served_in_the_order_they_arrived(bridge: HythonBridge) -> None:
    """Three calls queued behind one that is holding the session.

    Arrival order is known because each one is sent only after health shows
    the queue has grown. Each queued call takes long enough that the order
    they came back in cannot be a coin toss.
    """
    served: list[tuple[float, int]] = []
    held: list[Any] = []
    holder = threading.Thread(
        target=lambda: held.append(
            bridge.call("bridge.selfcheck", arguments={"sleep_s": 2.0}, timeout_s=60.0)
        )
    )
    holder.start()
    waiters = []
    try:
        _until(lambda: bridge.health().payload["data"]["busy"] is True)
        for index in range(3):
            waiter = threading.Thread(target=_queue_one, args=(bridge, index, served))
            waiters.append(waiter)
            waiter.start()
            _until(lambda index=index: bridge.health().payload["data"]["queued"] == index + 1)
    finally:
        holder.join(120.0)
        for waiter in waiters:
            waiter.join(120.0)

    assert held and held[0].payload["ok"] is True
    assert [index for _, index in sorted(served)] == [0, 1, 2]


def _queue_one(bridge: HythonBridge, index: int, served: list[tuple[float, int]]) -> None:
    answer = bridge.call(
        "bridge.selfcheck",
        arguments={"sleep_s": 0.3},
        wait_s=45.0,
        timeout_s=60.0,
    )
    assert answer.payload["ok"] is True, answer.payload
    served.append((time.monotonic(), index))


def _until(ready: Any, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.05)
    raise AssertionError("waited too long")


def test_a_session_told_to_stop_does_not_wait_for_a_long_call(home: Path) -> None:
    """A worker of its own, asked to stop while a long call is running.

    The call is asked to stop with the session, so the process goes away in
    about the time the shutdown allows and not in the time the call asked for.
    """
    worker = HythonBridge(home=home, port_range=(18350, 18369), alias="stopper")
    worker.start()
    running = threading.Thread(
        target=lambda: _quietly(
            lambda: worker.call("bridge.selfcheck", arguments={"sleep_s": 55.0}, timeout_s=60.0)
        )
    )
    running.start()
    try:
        _until(lambda: worker.health().payload["data"]["busy"] is True)
        began = time.monotonic()
        code = worker.stop()
        took = time.monotonic() - began
    finally:
        running.join(120.0)
    assert code is not None
    assert took < 30.0, took
    assert worker.process.poll() is not None


def _quietly(work: Any) -> None:
    """Run something whose failure is not what the test is about."""
    try:
        work()
    except Exception:  # noqa: BLE001 - the session is going down under it
        return


# A reply that never arrives


def test_a_lost_reply_is_answered_from_the_receipt_and_the_work_happens_once(
    bridge: HythonBridge,
) -> None:
    """The call runs, the answer is thrown away, the caller sends it again.

    The scene has to hold one node and not two, and the second send has to
    come back with what the first one did.
    """
    before = scene_info(bridge)
    operation_id = client.new_operation_id()
    arguments = {"creates": 1, "drop_reply": True}

    began = time.monotonic()
    answer = bridge.call(
        "bridge.selfcheck",
        arguments=arguments,
        operation_id=operation_id,
        http_timeout_s=GIVE_UP_S,
    )
    took = time.monotonic() - began

    # The first answer was lost and the second came from the receipt.
    assert took >= GIVE_UP_S
    assert answer.payload["ok"] is True, answer.payload
    assert answer.payload["replayed"] is True
    assert answer.payload["operation_id"] == operation_id
    created = answer.payload["data"]["created"]
    assert len(created) == 1

    after = scene_info(bridge)
    assert after["nodes"]["/obj"] == before["nodes"]["/obj"] + 1

    # Sending it a third time changes nothing at all.
    again = bridge.call("bridge.selfcheck", arguments=arguments, operation_id=operation_id)
    assert again.payload["data"]["created"] == created
    assert scene_info(bridge)["nodes"]["/obj"] == after["nodes"]["/obj"]

    # The same id with other arguments is a mistake, not a retry.
    wrong = bridge.call(
        "bridge.selfcheck", arguments={"creates": 2}, operation_id=operation_id, wait_s=10.0
    )
    assert wrong.payload["error"]["code"] == "OPERATION_MISMATCH"
    assert scene_info(bridge)["nodes"]["/obj"] == after["nodes"]["/obj"]


def test_a_call_with_no_operation_id_is_never_sent_again_by_itself(
    bridge: HythonBridge,
) -> None:
    before = scene_info(bridge)
    with pytest.raises(client.BridgeUnreachable):
        bridge.call(
            "bridge.selfcheck",
            arguments={"creates": 1, "drop_reply": True},
            http_timeout_s=GIVE_UP_S,
        )
    # It ran once, and nothing here tried it again.
    _until(lambda: bridge.health().payload["data"]["busy"] is False)
    assert scene_info(bridge)["nodes"]["/obj"] == before["nodes"]["/obj"] + 1


# A scene that has been replaced


def test_a_call_carrying_an_old_scene_epoch_is_refused_with_the_scene_there_is_now(
    bridge: HythonBridge, tmp_path: Path
) -> None:
    """Both ways a scene goes away: a new scene, and a load over the top."""
    epoch = scene_info(bridge)["scene_epoch"]
    bridge.call("bridge.selfcheck", arguments={"creates": 1}, wait_s=10.0)

    cleared = bridge.call("bridge.selfcheck", arguments={"new_scene": True}, wait_s=10.0)
    assert cleared.payload["ok"] is True, cleared.payload
    assert cleared.payload["scene_epoch"] == epoch + 1

    refused = bridge.call("scene.info", scene_epoch=epoch)
    error = refused.payload["error"]
    assert error["code"] == "SCENE_REPLACED"
    assert error["details"]["carried_epoch"] == epoch
    assert error["details"]["scene_epoch"] == epoch + 1
    summary = refused.payload["scene"]
    assert summary["nodes"]["/obj"] == 0
    assert summary["changed"] == "cleared"
    assert "hip_path" in summary

    # A load of a scene file replaces the scene as well, and counts once.
    scratch = tmp_path / "scratch.hip"
    saved = bridge.call(
        "bridge.selfcheck", arguments={"save_hip": str(scratch)}, wait_s=10.0, timeout_s=120.0
    )
    assert saved.payload["ok"] is True, saved.payload
    loaded = bridge.call(
        "bridge.selfcheck", arguments={"load_hip": str(scratch)}, wait_s=10.0, timeout_s=120.0
    )
    assert loaded.payload["scene_epoch"] == epoch + 2

    stale = bridge.call("scene.info", scene_epoch=epoch + 1)
    assert stale.payload["error"]["code"] == "SCENE_REPLACED"
    assert stale.payload["scene"]["changed"] == "loaded"
    assert scratch.name in stale.payload["scene"]["hip_path"]

    # And a call carrying the epoch there is now goes through.
    assert bridge.call("scene.info", scene_epoch=epoch + 2).payload["ok"] is True


# Nothing gets in without a signature


def test_an_unsigned_request_is_refused(bridge: HythonBridge) -> None:
    for path in (HEALTH_PATH, CALL_PATH):
        answer = client.request(bridge.port, path, body=b"{}")
        assert answer.status == 401
        assert answer.payload["error"]["code"] == "UNAUTHORIZED"


def test_a_signature_made_with_another_token_is_refused(bridge: HythonBridge) -> None:
    wrong = bridge.session._replace(token=security.mint_token())
    answer = client.post(wrong, HEALTH_PATH, {}, verify=False)
    assert answer.status == 401


def test_the_same_signed_request_cannot_be_sent_twice(bridge: HythonBridge) -> None:
    session = bridge.session
    body = b"{}"
    headers = signing.sign_request(
        session.token,
        method="POST",
        path=HEALTH_PATH,
        session_id=session.session_id,
        body=body,
    )
    first = client.request(bridge.port, HEALTH_PATH, body=body, headers=headers)
    second = client.request(bridge.port, HEALTH_PATH, body=body, headers=headers)
    assert first.status == 200
    assert second.status == 401


@pytest.mark.parametrize("header", ["Origin", "Referer"])
def test_a_request_from_a_page_is_refused(bridge: HythonBridge, header: str) -> None:
    answer = client.post(
        bridge.session, HEALTH_PATH, {}, headers={header: "http://evil.example"}, verify=False
    )
    assert answer.status == 403
    assert answer.payload["error"]["code"] == "FORBIDDEN"
    assert not [name for name in answer.headers if name.startswith("access-control-allow")]


def test_another_host_is_refused_even_when_signed(bridge: HythonBridge) -> None:
    answer = client.post(
        bridge.session, HEALTH_PATH, {}, headers={"Host": "evil.example:1"}, verify=False
    )
    assert answer.status == 403


def test_a_form_post_is_refused(bridge: HythonBridge) -> None:
    answer = client.request(bridge.port, CALL_PATH, body=b"json=%5B%5D", content_type=FORM_TYPE)
    assert answer.status in (401, 415)


# The shapes that used to end the process


def test_the_nesting_bomb_does_not_end_the_process(bridge: HythonBridge) -> None:
    """The body that killed a Houdini through the built in route.

    Sent three ways: to the route that used to exist, and to both of the
    bridge's own paths. The process must answer normally afterwards.
    """
    bomb = b"[" * 5000 + b"]" * 5000
    form = urllib.parse.urlencode({"json": "[" * 5000 + "]" * 5000}).encode("utf-8")

    old_route = client.request(bridge.port, "/api", body=form, content_type=FORM_TYPE)
    assert old_route.status == 404, "the built in route must not exist on this server"
    assert bridge.health().status == 200

    for path in (HEALTH_PATH, CALL_PATH):
        unsigned = client.request(bridge.port, path, body=bomb)
        assert unsigned.status == 401
        assert bridge.health().status == 200

    signed = client.post(bridge.session, CALL_PATH, {"tool": "bridge.ping"})
    assert signed.status == 200

    session = bridge.session
    headers = signing.sign_request(
        session.token, method="POST", path=CALL_PATH, session_id=session.session_id, body=bomb
    )
    answer = client.request(bridge.port, CALL_PATH, body=bomb, headers=headers)
    assert answer.status == 400
    assert answer.payload["error"]["code"] == "BODY_REFUSED"
    assert bridge.health().status == 200
    assert bridge.process.poll() is None


def test_a_very_large_body_does_not_end_the_process(bridge: HythonBridge) -> None:
    big = b"x" * (50 * 1024 * 1024)
    try:
        answer = client.request(bridge.port, CALL_PATH, body=big, timeout_s=60.0)
        assert answer.status in (401, 413)
    except client.BridgeUnreachable:
        # The server may close the connection on an oversized body rather than
        # answer it. Either is fine as long as the process lives.
        pass
    assert bridge.process.poll() is None
    assert bridge.health().status == 200


def test_a_body_that_never_arrives_does_not_stop_the_bridge(bridge: HythonBridge) -> None:
    """A request that promises a body and then sends nothing."""
    stalled = socket.create_connection((net.LOOPBACK, bridge.port), timeout=10.0)
    try:
        stalled.sendall(
            f"POST {CALL_PATH} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{bridge.port}\r\n"
            f"Content-Type: {JSON_TYPE}\r\n"
            "Content-Length: 1048576\r\n\r\n".encode()
        )
        stalled.sendall(b"{")
        assert bridge.health().status == 200
        assert bridge.call("bridge.ping").status == 200
    finally:
        stalled.close()
    assert bridge.process.poll() is None
    assert bridge.health().status == 200


# What the port and the files give away


def test_the_port_answers_on_loopback_alone(bridge: HythonBridge) -> None:
    assert net.can_connect(net.LOOPBACK, bridge.port) is True
    proof = net.prove_loopback_only(bridge.port)
    assert proof.private is True
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
        assert net.is_loopback(address), address


def test_no_answer_carries_a_cross_origin_header_or_names_the_build(
    bridge: HythonBridge,
) -> None:
    answer = bridge.health()
    assert answer.status == 200
    assert not [name for name in answer.headers if name.startswith("access-control-allow")]
    server = answer.headers.get("server", "")
    assert "22.0" not in server, server


def test_every_answer_is_signed_so_a_squatter_cannot_pass_for_the_bridge(
    bridge: HythonBridge,
) -> None:
    answer = bridge.health()
    assert signing.SIGNATURE_HEADER in answer.headers
    impostor = bridge.session._replace(token=security.mint_token())
    with pytest.raises(client.BridgeNotAuthentic):
        client.health(impostor)


def test_the_session_file_is_private_and_holds_the_token(bridge: HythonBridge, home: Path) -> None:
    path = registry.entry_path(home, bridge.session_id)
    assert security.is_private(path)
    entry = registry.read_entry(path)
    assert entry["token"] == bridge.session.token
    assert entry["kind"] == "hython"
    assert entry["houdini_version"]
    assert entry["pid"] == bridge.process.pid
    assert registry.entry_is_live(entry) is not False


def test_the_session_is_in_the_store_and_goes_when_the_process_quits(
    bridge: HythonBridge, home: Path
) -> None:
    store_path = home / store_module.STORE_FILE_NAME
    session = bridge.session
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
        client.health(session, timeout_s=5.0)


# Sessions of their own, started after the shared one has gone


def test_a_call_to_a_killed_session_never_reaches_the_one_that_replaced_it(
    home: Path,
) -> None:
    """A worker is killed and another starts under the same name.

    A caller holding the old id is refused here, with the id of the session
    answering to that name now. The new process is never asked.
    """
    worker = HythonBridge(home=home, port_range=(18370, 18379), alias="restarter")
    worker.start()
    old_id = worker.session_id
    worker.process.kill()
    worker.process.wait(timeout=60.0)

    replacement = HythonBridge(home=home, port_range=(18370, 18379), alias="restarter")
    replacement.start()
    try:
        assert replacement.session_id != old_id
        assert replacement.entry["alias"] == "restarter"

        with pytest.raises(client.SessionDead) as refused:
            client.Session.open(home, old_id)

        details = refused.value.details()
        assert details["code"] == "SESSION_DEAD"
        assert details["alias"] == "restarter"
        assert details["live_session_id"] == replacement.session_id

        # Nothing was sent, so the new session has run nothing at all.
        assert replacement.health().payload["data"]["last_op"] is None
        # And the name now resolves to the new session.
        assert client.Session.open(home, "restarter").session_id == replacement.session_id
    finally:
        replacement.stop()
    assert worker.process.poll() is not None
    assert replacement.process.poll() is not None


def test_two_sessions_on_the_same_scene_get_different_ids_and_names(home: Path) -> None:
    """The only check here that runs two Houdinis at once, and needs to."""
    pattern = ["--alias-template", "probe-{n}"]
    first = HythonBridge(home=home, port_range=(18380, 18389), extra_args=pattern)
    second = HythonBridge(home=home, port_range=(18390, 18399), extra_args=pattern)
    first.start()
    try:
        second.start()
        try:
            assert first.session_id != second.session_id
            assert [first.entry["alias"], second.entry["alias"]] == ["probe-1", "probe-2"]
            assert first.port != second.port
            for started in (first, second):
                data = started.health().payload["data"]
                assert data["status"] == "ok"
                assert data["session_id"] == started.session_id
        finally:
            second.stop()
    finally:
        first.stop()
    assert first.process.poll() is not None
    assert second.process.poll() is not None
    assert registry.find_entry(home, "probe-1") is None
    assert registry.find_entry(home, "probe-2") is None


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
