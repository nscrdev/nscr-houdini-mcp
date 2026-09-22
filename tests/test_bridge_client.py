"""What the calling end refuses to do, and what it does twice.

Nothing here opens a socket. The two rules being checked are both decided
before anything is sent: a call is never addressed to a session that has gone,
and a call is only ever sent again when sending it again is safe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.bridge.liveness import process_start_stamp

ALIAS = "w1"


def entry(home: Path, session_id: str, *, pid: int, port: int, pid_start: Any = None) -> None:
    registry.write_entry(
        home,
        {
            "session_id": session_id,
            "alias": ALIAS,
            "kind": "hython",
            "pid": pid,
            "pid_start": pid_start,
            "port": port,
            "address": "127.0.0.1",
            "token": "a token",
            "started_at": 1.0,
        },
    )


def row(home: Path, session_id: str, *, pid: int, port: int) -> store_module.Store:
    store = store_module.Store(home / store_module.STORE_FILE_NAME)
    store.register_session(session_id, kind="hython", pid=pid, alias=ALIAS, port=port)
    return store


# Section: a session that has gone


def test_a_live_session_is_found_by_id_and_by_name(tmp_path: Path) -> None:
    entry(tmp_path, "id-1", pid=os.getpid(), port=18100, pid_start=process_start_stamp())
    with row(tmp_path, "id-1", pid=os.getpid(), port=18100):
        pass

    by_id = client.Session.open(tmp_path, "id-1")
    by_name = client.Session.open(tmp_path, ALIAS)

    assert by_id == by_name
    assert by_id.port == 18100


def test_a_call_to_a_dead_session_is_refused_with_the_live_one(tmp_path: Path) -> None:
    """The dead process cannot answer, and the live one must not be handed
    work that was written for its predecessor."""
    dead_pid = 1 << 30
    with row(tmp_path, "id-old", pid=dead_pid, port=18100):
        pass
    entry(tmp_path, "id-new", pid=os.getpid(), port=18101, pid_start=process_start_stamp())
    with row(tmp_path, "id-new", pid=os.getpid(), port=18101):
        pass

    with pytest.raises(client.SessionDead) as refused:
        client.Session.open(tmp_path, "id-old")

    details = refused.value.details()
    assert details["code"] == "SESSION_DEAD"
    assert details["session_id"] == "id-old"
    assert details["alias"] == ALIAS
    assert details["live_session_id"] == "id-new"
    assert details["hint"]


def test_a_dead_session_with_no_successor_still_says_it_is_dead(tmp_path: Path) -> None:
    with row(tmp_path, "id-old", pid=1 << 30, port=18100):
        pass

    with pytest.raises(client.SessionDead) as refused:
        client.Session.open(tmp_path, "id-old")

    assert refused.value.details()["live_session_id"] is None


def test_a_handle_nothing_has_heard_of_is_not_a_dead_session(tmp_path: Path) -> None:
    with pytest.raises(client.SessionGone):
        client.Session.open(tmp_path, "id-nobody")


def test_a_session_file_left_by_a_crash_is_not_used(tmp_path: Path) -> None:
    """The port in it is free for anything to be sitting on by now."""
    entry(tmp_path, "id-old", pid=1 << 30, port=18100, pid_start="old")
    with row(tmp_path, "id-old", pid=1 << 30, port=18100):
        pass

    with pytest.raises(client.SessionDead):
        client.Session.open(tmp_path, "id-old")


# Section: sending a call again


class Attempts:
    """A stand in for the send, which fails as many times as it is told to."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.sent: list[dict[str, Any]] = []

    def __call__(self, session: Any, path: str, payload: Any, **rest: Any) -> client.Answer:
        self.sent.append(dict(payload))
        if len(self.sent) <= self.failures:
            raise client.BridgeUnreachable("the connection closed")
        return client.Answer(200, {"ok": True, "data": {"sends": len(self.sent)}}, {}, b"")


def session() -> client.Session:
    return client.Session(session_id="id-1", token="a token", port=18100)


def test_a_lost_reply_is_sent_again_under_the_same_operation_id(monkeypatch) -> None:
    attempts = Attempts(failures=1)
    monkeypatch.setattr(client, "post", attempts)

    answer = client.call(session(), "node.create", operation_id="op-1")

    assert answer.payload["ok"] is True
    assert len(attempts.sent) == 2
    # The same id, so the bridge answers the second send from its receipt.
    assert {sent["operation_id"] for sent in attempts.sent} == {"op-1"}


def test_a_call_with_no_operation_id_is_never_sent_again(monkeypatch) -> None:
    attempts = Attempts(failures=1)
    monkeypatch.setattr(client, "post", attempts)

    with pytest.raises(client.BridgeUnreachable):
        client.call(session(), "scene.info")

    assert len(attempts.sent) == 1


def test_a_lost_reply_is_sent_again_once_and_no_more(monkeypatch) -> None:
    attempts = Attempts(failures=2)
    monkeypatch.setattr(client, "post", attempts)

    with pytest.raises(client.BridgeUnreachable):
        client.call(session(), "node.create", operation_id="op-1")

    assert len(attempts.sent) == 2


def test_a_reply_that_arrived_is_never_sent_again(monkeypatch) -> None:
    """Whatever it says. A failure that was answered is an answer."""
    attempts = Attempts(failures=0)
    monkeypatch.setattr(client, "post", attempts)

    client.call(session(), "node.create", operation_id="op-1")

    assert len(attempts.sent) == 1


def test_a_caller_that_asks_for_no_retry_gets_none(monkeypatch) -> None:
    attempts = Attempts(failures=1)
    monkeypatch.setattr(client, "post", attempts)

    with pytest.raises(client.BridgeUnreachable):
        client.call(session(), "node.create", operation_id="op-1", retry_lost_reply=False)

    assert len(attempts.sent) == 1


def test_a_mutation_is_given_an_id_when_the_caller_passes_none(monkeypatch) -> None:
    attempts = Attempts(failures=1)
    monkeypatch.setattr(client, "post", attempts)

    client.mutate(session(), "node.create", arguments={"parent": "/obj"})

    ids = {sent["operation_id"] for sent in attempts.sent}
    assert len(ids) == 1
    assert next(iter(ids)).startswith("op-")


def test_two_mutations_are_given_different_ids() -> None:
    assert client.new_operation_id() != client.new_operation_id()
