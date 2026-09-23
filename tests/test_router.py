"""Which session a call reaches, on a stand in store and stand in session files.

Nothing here starts a Houdini or opens a port. The store is a list of rows the
test writes, the session files are a table of clients, and the call is a
function that records what it was asked to send.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp.bridge import client
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.router import SOCKET_MARGIN_S, Router, choose, socket_wait
from nscr_houdini_mcp.store import SessionRecord


def record(
    session_id: str,
    alias: str,
    *,
    state: str = "live",
    kind: str = "hython",
    started_at: float = 1.0,
    scene_epoch: int = 0,
    hip_path: str | None = None,
    previous_alias: str | None = None,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        alias=alias,
        kind=kind,
        pid=1000,
        pid_start="stamp",
        port=18000,
        state=state,
        scene_epoch=scene_epoch,
        hip_path=hip_path,
        capabilities=None,
        started_at=started_at,
        heartbeat_at=started_at,
        previous_alias=previous_alias,
    )


class FakeStore:
    """The two reads the router makes, over rows a test writes."""

    def __init__(self, rows: list[SessionRecord]) -> None:
        self.rows = rows
        self.reclaimed = 0
        self.asked_for_gone: list[bool] = []

    def __enter__(self) -> FakeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def reclaim_sessions(self) -> list[str]:
        self.reclaimed += 1
        return []

    def list_sessions(self, *, include_gone: bool = False) -> list[SessionRecord]:
        self.asked_for_gone.append(include_gone)
        return [r for r in self.rows if include_gone or r.state != "gone"]


class FakeFiles:
    """Session files by id. A missing id is a process that has gone."""

    def __init__(self, ids: list[str]) -> None:
        self.ids = set(ids)
        self.opened: list[str] = []

    def open(self, home: Any, handle: str, *, store_path: Any = None) -> client.Session:
        self.opened.append(handle)
        if handle not in self.ids:
            raise client.SessionDead(handle)
        return client.Session(session_id=handle, token="token", port=18000)


class Sent:
    """Records each call and answers it with a reply the test chose."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, session: client.Session, tool: str, **rest: Any) -> client.Answer:
        self.calls.append({"session": session.session_id, "tool": tool, **rest})
        reply = self.replies.pop(0) if self.replies else {"ok": True, "data": {}}
        if isinstance(reply, Exception):
            raise reply
        return client.Answer(200, reply, {})


def router_for(
    rows: list[SessionRecord],
    *,
    files: list[str] | None = None,
    default: str | None = None,
    send: Any = None,
) -> tuple[Router, FakeStore, FakeFiles, list[str]]:
    store = FakeStore(rows)
    opened = FakeFiles(files if files is not None else [r.session_id for r in rows])
    renewed: list[str] = []
    router = Router(
        home=Path("."),
        default_session=default,
        open_store=lambda path: store,
        open_session=opened.open,
        send=send or Sent(),
        renew_lease=lambda store, session_id: renewed.append(session_id),
    )
    return router, store, opened, renewed


# Section: the rules


def test_a_session_is_found_by_id_and_by_alias() -> None:
    rows = [record("s-1", "w1"), record("s-2", "acc-1", kind="gui")]
    assert choose(rows, "s-2").session_id == "s-2"
    assert choose(rows, "w1").session_id == "s-1"


def test_the_only_live_session_is_used_when_none_is_named() -> None:
    rows = [record("s-1", "w1"), record("s-0", "w1", state="gone", started_at=0.5)]
    assert choose(rows, None).session_id == "s-1"


def test_the_config_default_settles_several_live_sessions() -> None:
    rows = [record("s-1", "w1"), record("s-2", "acc-1", kind="gui")]
    assert choose(rows, None, default="acc-1").session_id == "s-2"
    assert choose(rows, None, default="s-1").session_id == "s-1"


def test_several_live_and_no_default_is_ambiguous_with_the_candidates() -> None:
    rows = [record("s-1", "w1"), record("s-2", "acc-1", kind="gui", hip_path="/p/acc.hip")]
    with pytest.raises(CallError) as caught:
        choose(rows, None)
    error = caught.value
    assert error.code == "SESSION_AMBIGUOUS"
    candidates = error.details["candidates"]
    assert [c["alias"] for c in candidates] == ["w1", "acc-1"]
    assert candidates[1] == {
        "session_id": "s-2",
        "alias": "acc-1",
        "kind": "gui",
        "state": "live",
        "hip_path": "/p/acc.hip",
    }


def test_a_default_that_is_not_live_does_not_settle_anything() -> None:
    rows = [record("s-1", "w1"), record("s-2", "w2")]
    with pytest.raises(CallError) as caught:
        choose(rows, None, default="w9")
    assert caught.value.code == "SESSION_AMBIGUOUS"
    assert caught.value.details["default_session"] == "w9"
    assert caught.value.details["default_live"] is False


def test_no_live_session_says_so() -> None:
    with pytest.raises(CallError) as caught:
        choose([record("s-0", "w1", state="gone")], None)
    assert caught.value.code == "NO_SESSION"


def test_a_dead_id_names_the_live_successor_under_its_alias() -> None:
    rows = [
        record("s-old", "w1", state="gone", started_at=1.0),
        record("s-new", "w1", started_at=2.0),
    ]
    with pytest.raises(CallError) as caught:
        choose(rows, "s-old")
    error = caught.value
    assert error.code == "SESSION_DEAD"
    assert error.details["live_session_id"] == "s-new"
    assert error.details["alias"] == "w1"
    assert "s-new" in (error.hint or "")


def test_a_crashed_session_is_dead_too() -> None:
    with pytest.raises(CallError) as caught:
        choose([record("s-1", "w1", state="crashed")], "s-1")
    assert caught.value.code == "SESSION_DEAD"
    assert caught.value.details["live_session_id"] is None


def test_an_alias_whose_process_has_gone_is_dead_with_no_successor() -> None:
    with pytest.raises(CallError) as caught:
        choose([record("s-1", "w1", state="gone")], "w1")
    assert caught.value.code == "SESSION_DEAD"
    assert caught.value.details["live_session_id"] is None


def test_an_alias_reaches_the_newest_process_under_it() -> None:
    rows = [
        record("s-old", "w1", state="gone", started_at=1.0),
        record("s-new", "w1", started_at=2.0),
    ]
    assert choose(rows, "w1").session_id == "s-new"


def test_the_name_a_renamed_session_had_still_reaches_it_and_says_so() -> None:
    """The same process the caller meant, so the call goes on, with a warning."""
    rows = [
        record("s-old", "untitled-1", state="gone", kind="gui", started_at=0.5),
        record("s-1", "shot_010-1", kind="gui", previous_alias="untitled-1"),
        record("s-2", "untitled-2", kind="gui", started_at=2.0),
    ]
    assert choose(rows, "untitled-1").session_id == "s-1"

    router, _, _, _ = router_for(rows)
    target = router.resolve("untitled-1")
    assert target.session_id == "s-1"
    [warning] = target.trace()["warnings"]
    assert warning["code"] == "ALIAS_RENAMED"
    assert warning["alias"] == "shot_010-1"
    assert warning["previous_alias"] == "untitled-1"
    # Named as it is now, or by id, there is nothing to say.
    assert "warnings" not in router.resolve("shot_010-1").trace()
    assert "warnings" not in router.resolve("s-1").trace()


def test_a_session_that_holds_a_name_now_wins_over_one_that_held_it_before() -> None:
    rows = [
        record("s-1", "shot_010-1", previous_alias="untitled-1", state="gone"),
        record("s-2", "untitled-1", started_at=2.0),
    ]
    assert choose(rows, "untitled-1").session_id == "s-2"


def test_an_unresponsive_session_is_refused_with_the_others_listed() -> None:
    rows = [record("s-1", "w1", state="unresponsive"), record("s-2", "w2")]
    with pytest.raises(CallError) as caught:
        choose(rows, "w1")
    error = caught.value
    assert error.code == "SESSION_UNRESPONSIVE"
    assert [row["alias"] for row in error.details["live"]] == ["w2"]


def test_an_unknown_name_offers_the_nearest() -> None:
    with pytest.raises(CallError) as caught:
        choose([record("s-1", "acc-1")], "acc1")
    assert caught.value.code == "SESSION_UNKNOWN"
    assert caught.value.details["did_you_mean"] == ["acc-1"]


# Section: the router around the rules


def test_a_machine_with_no_store_yet_has_no_session(tmp_path: Path) -> None:
    router = Router(tmp_path / "home")
    with pytest.raises(CallError) as caught:
        router.resolve(None)
    assert caught.value.code == "NO_SESSION"
    assert not (tmp_path / "home").exists()


def test_the_router_keeps_one_client_per_session() -> None:
    router, store, files, _ = router_for([record("s-1", "w1")])
    first = router.resolve(None)
    second = router.resolve("w1")
    assert first.session is second.session
    assert files.opened == ["s-1"]
    assert store.reclaimed == 2
    assert router.cached() == ["s-1"]


def test_a_session_file_that_is_gone_is_dead_and_not_kept() -> None:
    router, _, _, _ = router_for([record("s-1", "w1")], files=[])
    with pytest.raises(CallError) as caught:
        router.resolve("s-1")
    assert caught.value.code == "SESSION_DEAD"
    assert router.cached() == []


def test_a_worker_lease_is_renewed_and_a_gui_session_is_left_alone() -> None:
    router, _, _, renewed = router_for([record("s-1", "w1"), record("s-2", "g", kind="gui")])
    router.resolve("w1")
    router.resolve("g")
    assert renewed == ["s-1"]


def test_a_lost_reply_from_a_process_that_died_is_session_dead() -> None:
    rows = [record("s-1", "w1")]
    send = Sent(client.BridgeUnreachable("closed"))
    router, store, _, _ = router_for(rows, send=send)
    target = router.resolve("s-1")
    # The process ends while the call is out.
    store.rows = [record("s-1", "w1", state="gone")]
    with pytest.raises(CallError) as caught:
        router.call(target, "node.create", {}, operation_id="op-1")
    assert caught.value.code == "SESSION_DEAD"
    assert router.cached() == []


def test_a_lost_reply_from_a_live_session_says_to_resend_the_same_id() -> None:
    router, _, _, _ = router_for([record("s-1", "w1")], send=Sent(client.BridgeUnreachable("x")))
    target = router.resolve(None)
    with pytest.raises(CallError) as caught:
        router.call(target, "node.create", {}, operation_id="op-1")
    error = caught.value
    assert error.code == "SESSION_UNREACHABLE"
    assert error.details["operation_id"] == "op-1"
    assert "same operation_id" in (error.hint or "")


def test_an_answer_the_session_did_not_sign_drops_the_client() -> None:
    router, _, _, _ = router_for([record("s-1", "w1")], send=Sent(client.BridgeNotAuthentic("x")))
    target = router.resolve(None)
    with pytest.raises(CallError) as caught:
        router.call(target, "bridge.ping")
    assert caught.value.code == "REPLY_NOT_AUTHENTIC"
    assert router.cached() == []


def test_a_bridge_error_carries_its_code_trace_and_scene() -> None:
    refusal = {
        "ok": False,
        "error": {"code": "SCENE_REPLACED", "message": "the scene changed", "details": {"a": 1}},
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": 3,
        "operation_id": "op-9",
        "scene": {"hip_name": "b.hip"},
    }
    router, _, _, _ = router_for([record("s-1", "w1")], send=Sent(refusal))
    target = router.resolve(None)
    with pytest.raises(CallError) as caught:
        router.call(target, "node.create", {}, operation_id="op-9", scene_epoch=2)
    error = caught.value
    assert error.code == "SCENE_REPLACED"
    assert error.details == {"a": 1, "scene": {"hip_name": "b.hip"}}
    assert error.trace["scene_epoch"] == 3
    assert error.hint


def test_a_session_dead_reply_drops_the_kept_client() -> None:
    refusal = {"ok": False, "error": {"code": "SESSION_DEAD", "message": "gone"}}
    router, _, _, _ = router_for([record("s-1", "w1")], send=Sent(refusal))
    target = router.resolve(None)
    with pytest.raises(CallError):
        router.call(target, "bridge.ping")
    assert router.cached() == []


def test_the_budgets_and_ids_pass_through_to_the_client() -> None:
    send = Sent()
    router, _, _, _ = router_for([record("s-1", "w1")], send=send)
    target = router.resolve(None)
    router.call(
        target,
        "node.create",
        {"type": "box"},
        operation_id="op-1",
        scene_epoch=4,
        wait_s=7.0,
        timeout_s=30.0,
    )
    [sent] = send.calls
    assert sent["tool"] == "node.create"
    assert sent["arguments"] == {"type": "box"}
    assert sent["session_id"] == "s-1"
    assert sent["operation_id"] == "op-1"
    assert sent["scene_epoch"] == 4
    assert sent["wait_s"] == 7.0
    assert sent["timeout_s"] == 30.0


def test_a_lost_reply_is_sent_again_with_the_same_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real client under the router: one loss, one resend, the same id twice."""
    posted: list[dict[str, Any]] = []

    def post(session: client.Session, path: str, payload: Any = None, **rest: Any) -> Any:
        posted.append(dict(payload))
        if len(posted) == 1:
            raise client.BridgeUnreachable("the reply was lost")
        return client.Answer(200, {"ok": True, "data": {"made": 1}}, {})

    monkeypatch.setattr(client, "post", post)
    router, _, _, _ = router_for([record("s-1", "w1")], send=client.call)
    target = router.resolve(None)
    reply = router.call(target, "node.create", {"type": "box"}, operation_id="op-7")
    assert reply["data"] == {"made": 1}
    assert [p["operation_id"] for p in posted] == ["op-7", "op-7"]


def test_a_read_with_no_id_is_not_sent_again(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[Any] = []

    def post(session: client.Session, path: str, payload: Any = None, **rest: Any) -> Any:
        posted.append(payload)
        raise client.BridgeUnreachable("the reply was lost")

    monkeypatch.setattr(client, "post", post)
    router, _, _, _ = router_for([record("s-1", "w1")], send=client.call)
    target = router.resolve(None)
    with pytest.raises(CallError) as caught:
        router.call(target, "scene.info")
    assert caught.value.code == "SESSION_UNREACHABLE"
    assert len(posted) == 1


def test_a_call_with_no_timeout_waits_on_the_socket_past_the_bridge_default() -> None:
    send = Sent()
    router, _, _, _ = router_for([record("s-1", "w1")], send=send)
    target = router.resolve(None)
    router.call(target, "node.create", {}, operation_id="op-1")
    router.call(target, "scene.info", wait_s=20.0, timeout_s=90.0)
    first, second = send.calls
    # One second of queueing and a minute of running, the bridge's own
    # defaults, and a margin: a read that takes 30 s is slow, not lost.
    assert first["http_timeout_s"] == 1.0 + 60.0 + SOCKET_MARGIN_S
    assert first["http_timeout_s"] > 60.0
    assert second["http_timeout_s"] == 20.0 + 90.0 + SOCKET_MARGIN_S
    assert socket_wait(0.0, None) == 60.0 + SOCKET_MARGIN_S


def test_a_worker_lease_is_renewed_again_when_the_answer_comes_back() -> None:
    refusal = {"ok": False, "error": {"code": "SESSION_BUSY", "message": "busy"}}
    router, _, _, renewed = router_for([record("s-1", "w1")], send=Sent({"ok": True}, refusal))
    target = router.resolve(None)
    assert renewed == ["s-1"]
    router.call(target, "bridge.ping")
    assert renewed == ["s-1", "s-1"]
    with pytest.raises(CallError):
        router.call(target, "bridge.ping")
    assert renewed == ["s-1", "s-1", "s-1"]


def test_kept_clients_of_sessions_that_ended_are_dropped_on_the_next_resolve() -> None:
    router, store, _, _ = router_for([record("s-1", "w1"), record("s-2", "w2")])
    router.resolve("w1")
    router.resolve("w2")
    assert sorted(router.cached()) == ["s-1", "s-2"]
    store.rows = [record("s-1", "w1", state="gone"), record("s-2", "w2")]
    router.resolve(None)
    assert router.cached() == ["s-2"]


def test_ended_sessions_are_read_only_when_a_session_is_named() -> None:
    router, store, _, _ = router_for([record("s-1", "w1")])
    router.resolve(None)
    router.resolve("w1")
    assert store.asked_for_gone == [False, True]


def test_neither_the_session_nor_the_target_prints_the_token() -> None:
    router, _, _, _ = router_for([record("s-1", "w1")])
    target = router.resolve(None)
    for text in (repr(target), str(target), repr(target.session), f"{target.session}"):
        assert "token" not in text
        assert "s-1" in text
