"""`hou_sessions` through the server, on a real store and stand in sessions.

The store is a real one in a folder of the test's own, so the states come from
the store's own rules: a row whose process has gone is found and marked by the
store, not by the test. The sessions behind the rows are stand ins: the health
answer and the signed client are recorded, and the pool's start and stop are
replaced, so nothing here starts a Houdini.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import pool
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.config import Config
from nscr_houdini_mcp.router import Router
from nscr_houdini_mcp.server import build_server
from nscr_houdini_mcp.store import SessionRecord, Store, WorkerRecord, process_start_stamp
from nscr_houdini_mcp.tools.registry import TOOLS
from test_router import Sent
from test_server import talk, text_of

# A pid no system hands out, so the process behind the row is gone.
DEAD_PID = 2_000_000_000

CAPABILITIES = {"houdini_version": "22.0.368", "license": "Commercial", "renderers": ["karma"]}


class Bench:
    """A state folder with a real store, and stand ins for what is behind it."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.store_path = home / store_module.STORE_FILE_NAME
        self.reachable: set[str] = set()
        self.health: dict[str, Any] = {}
        self.renewed: list[str] = []
        self.forgotten: list[str] = []
        self.sent = Sent()
        self.stamp = process_start_stamp()
        self.config = Config(
            path=home / "config.toml",
            pool_cap=2,
            state_home=home,
            worker_ports=(18830, 18839),
            hython=home / "bin" / "hython",
        )

    def open(self, home: Any, handle: str, *, store_path: Any = None) -> client.Session:
        if handle not in self.reachable:
            raise client.SessionDead(handle)
        return client.Session(session_id=handle, token="token", port=18000)

    def ask_health(self, session: client.Session, **rest: Any) -> client.Answer:
        answer = self.health.get(session.session_id, {"status": "ok", "busy": False})
        if isinstance(answer, Exception):
            raise answer
        return client.Answer(200, {"ok": True, "data": answer}, {})

    def router(self, config: Config) -> Router:
        router = Router(
            self.home,
            open_session=self.open,
            send=self.sent,
            ask_health=self.ask_health,
            renew_lease=lambda store, session_id: self.renewed.append(session_id),
        )
        forget = router.forget

        def remember_forgetting(session_id: str) -> None:
            self.forgotten.append(session_id)
            forget(session_id)

        router.forget = remember_forgetting  # type: ignore[method-assign]
        return router

    def store(self, **rest: Any) -> Store:
        return Store(self.store_path, **rest)

    def session(
        self,
        session_id: str,
        alias: str,
        *,
        kind: str = "hython",
        pid: int | None = None,
        hip_path: str | None = None,
        clock: Any = None,
    ) -> SessionRecord:
        alive = pid is None
        with self.store(clock=clock) as store:
            record = store.register_session(
                session_id,
                kind=kind,
                pid=os.getpid() if alive else pid,
                pid_start=self.stamp if alive else "gone",
                alias=alias,
                port=18000,
                hip_path=hip_path,
                capabilities=CAPABILITIES if kind == "gui" else None,
            )
        if alive:
            self.reachable.add(session_id)
        return record

    def worker(self, session_id: str, token: str, *, pid: int | None = None) -> WorkerRecord:
        alive = pid is None
        with self.store() as store:
            store.reserve_worker(cap=8, token=token)
            return store.set_worker_state(
                token,
                "running",
                session_id=session_id,
                owner_pid=os.getpid() if alive else pid,
                pid=os.getpid() if alive else pid,
                pid_start=self.stamp if alive else "gone",
                capabilities=CAPABILITIES,
            )

    def serve(self) -> Any:
        return build_server(TOOLS, config_loader=lambda: self.config, router_factory=self.router)


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    home = tmp_path / "home"
    home.mkdir()
    return Bench(home)


def listed(bench: Bench, **arguments: Any) -> dict[str, dict[str, Any]]:
    _, [result] = talk(bench.serve(), ("hou_sessions", arguments))
    assert not result.is_error, text_of(result)
    return {row["alias"]: row for row in result.structured_content["sessions"]}


# Section: list


def test_an_empty_state_folder_lists_nothing(bench: Bench) -> None:
    _, [result] = talk(bench.serve(), ("hou_sessions", {}))
    assert not result.is_error
    body = result.structured_content
    assert body["sessions"] == []
    assert body["pool"]["cap"] == 2
    assert body["pool"]["running"] == 0


def test_list_says_live_and_busy_from_the_sessions_own_answer(bench: Bench) -> None:
    bench.session("s-1", "w1")
    bench.worker("s-1", "wk-1")
    bench.session("s-2", "acc-1", kind="gui", hip_path="/p/acc.hip")
    bench.health["s-2"] = {
        "status": "ok",
        "busy": True,
        "current_op": "node.create",
        "scene_epoch": 3,
        "scene": {"hip_path": "/p/acc_v002.hip", "nodes": {"/obj": 4}},
    }
    rows = listed(bench)
    assert rows["w1"]["state"] == "live"
    assert rows["w1"]["kind"] == "hython"
    assert rows["w1"]["job"] is None
    assert rows["w1"]["lease_age_s"] >= 0
    assert "22.0.368" in rows["w1"]["capabilities"]
    assert rows["acc-1"]["state"] == "busy"
    assert rows["acc-1"]["current_op"] == "node.create"
    assert rows["acc-1"]["hip_path"] == "/p/acc_v002.hip"
    assert rows["acc-1"]["scene_epoch"] == 3
    assert "job" not in rows["acc-1"]
    # Listing is not using: no worker's lease was renewed.
    assert bench.renewed == []


def test_a_killed_worker_lists_as_crashed_now_and_later(bench: Bench) -> None:
    bench.session("s-1", "w1", pid=DEAD_PID)
    bench.worker("s-1", "wk-1", pid=DEAD_PID)
    assert listed(bench)["w1"]["state"] == "crashed"
    # The store has marked the row gone by now; the pool's failed worker row
    # is what still tells a kill from a stop.
    with bench.store() as store:
        assert store.get_session("s-1").state == "gone"
        assert store.get_worker("wk-1").state == "failed"
    assert listed(bench)["w1"]["state"] == "crashed"


def test_a_gui_session_that_left_its_file_behind_lists_as_crashed(bench: Bench) -> None:
    bench.session("s-3", "shot-1", kind="gui", pid=DEAD_PID)
    registry.ensure_registry_dir(bench.home)
    registry.write_entry(bench.home, {"session_id": "s-3", "alias": "shot-1"})
    assert listed(bench)["shot-1"]["state"] == "crashed"
    assert listed(bench)["shot-1"]["state"] == "crashed"


def test_a_stopped_worker_lists_as_gone(bench: Bench) -> None:
    bench.session("s-1", "w1")
    bench.worker("s-1", "wk-1")
    with bench.store() as store:
        store.release_worker("wk-1", state="stopped")
        store.end_session("s-1")
    row = listed(bench)["w1"]
    assert row["state"] == "gone"
    assert row["ended_at"] is not None


def test_a_session_that_ended_long_ago_is_not_listed(bench: Bench) -> None:
    then = time.time() - 7200.0
    bench.session("s-old", "w9", clock=lambda: then)
    with bench.store(clock=lambda: then) as store:
        store.end_session("s-old")
    bench.session("s-1", "w1")
    assert set(listed(bench)) == {"w1"}


def test_a_port_that_does_not_answer_lists_as_unresponsive(bench: Bench) -> None:
    bench.session("s-1", "w1")
    bench.health["s-1"] = client.BridgeUnreachable("nothing on the port")
    bench.session("s-2", "w2")
    with bench.store() as store:
        store.touch_session("s-2", state="unresponsive", transport_ok=False)
    rows = listed(bench)
    assert rows["w1"]["state"] == "unresponsive"
    assert rows["w2"]["state"] == "unresponsive"


def test_full_detail_adds_health_and_the_whole_capability_record(bench: Bench) -> None:
    bench.session("s-1", "w1")
    bench.worker("s-1", "wk-1")
    bench.health["s-1"] = {"status": "ok", "busy": False, "queued": 0, "heartbeat_age_s": 1.0}
    row = listed(bench, detail="full")["w1"]
    assert row["pid"] == os.getpid()
    assert row["port"] == 18000
    assert row["capabilities"] == CAPABILITIES
    assert row["health"]["queued"] == 0
    assert row["health"]["round_trip_ms"] >= 0
    assert row["worker"]["state"] == "running"


# Section: info


def test_info_is_one_session_in_full_with_its_health(bench: Bench) -> None:
    bench.session("s-1", "w1")
    bench.worker("s-1", "wk-1")
    bench.session("s-2", "acc-1", kind="gui")
    bench.health["s-2"] = {"status": "ok", "busy": True, "current_op": "scene.open"}
    _, [result] = talk(bench.serve(), ("hou_sessions", {"action": "info", "session": "acc-1"}))
    assert not result.is_error, text_of(result)
    body = result.structured_content
    assert body["session"]["alias"] == "acc-1"
    assert body["session"]["state"] == "busy"
    assert body["session"]["health"]["current_op"] == "scene.open"
    assert body["trace"]["session_id"] == "s-2"


# Section: start


def fake_start(bench: Bench, seen: list[dict[str, Any]]) -> Any:
    def start(config: pool.PoolConfig, store: Store, **rest: Any) -> WorkerRecord:
        seen.append({"config": config, **rest})
        count = len(seen)
        bench.session(f"s-new-{count}", f"w{count}")
        return bench.worker(f"s-new-{count}", f"wk-new-{count}")

    return start


def test_start_takes_cap_ports_and_hython_from_config(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(pool, "start_worker", fake_start(bench, seen))
    _, [result] = talk(bench.serve(), ("hou_sessions", {"action": "start", "weight": "heavy"}))
    assert not result.is_error, text_of(result)
    [asked] = seen
    assert asked["config"].cap == 2
    assert asked["config"].port_range == (18830, 18839)
    assert asked["config"].hython == bench.home / "bin" / "hython"
    assert asked["config"].home == bench.home
    assert asked["weight"] == "heavy"
    body = result.structured_content
    assert body["session"]["alias"] == "w1"
    assert body["session"]["state"] == "live"
    assert body["pool"]["running"] == 1
    assert body["trace"]["operation_id"].startswith("op-")


def test_a_start_sent_again_under_its_id_starts_nothing_new(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(pool, "start_worker", fake_start(bench, seen))
    arguments = {"action": "start", "operation_id": "start-1"}
    _, [first, second] = talk(
        bench.serve(), ("hou_sessions", arguments), ("hou_sessions", arguments)
    )
    assert len(seen) == 1
    assert second.structured_content["replayed"] is True
    assert second.structured_content["session"] == first.structured_content["session"]
    assert second.structured_content["trace"]["operation_id"] == "start-1"


def test_a_full_pool_is_a_coded_error_and_frees_the_id(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(config: pool.PoolConfig, store: Store, **rest: Any) -> WorkerRecord:
        raise store_module.PoolFull("2 of 2 worker slots are in use")

    monkeypatch.setattr(pool, "start_worker", full)
    arguments = {"action": "start", "operation_id": "start-2"}
    _, [result] = talk(bench.serve(), ("hou_sessions", arguments))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "POOL_FULL"
    assert text_of(result).startswith("POOL_FULL: 2 of 2 worker slots are in use")
    assert "hint: " in text_of(result)
    assert result.structured_content["trace"]["operation_id"] == "start-2"

    # Nothing started, so the same id may try again once there is room.
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(pool, "start_worker", fake_start(bench, seen))
    _, [again] = talk(bench.serve(), ("hou_sessions", arguments))
    assert not again.is_error, text_of(again)
    assert len(seen) == 1


def test_no_hython_is_a_coded_error(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(config: pool.PoolConfig, store: Store, **rest: Any) -> WorkerRecord:
        raise pool.HythonNotFound("no hython found, name one in config")

    monkeypatch.setattr(pool, "start_worker", missing)
    _, [result] = talk(bench.serve(), ("hou_sessions", {"action": "start"}))
    assert result.structured_content["error"]["code"] == "HYTHON_NOT_FOUND"


# Section: stop


def fake_stop(bench: Bench, seen: list[str]) -> Any:
    def stop(config: pool.PoolConfig, store: Store, handle: str, **rest: Any) -> pool.Stopped:
        seen.append(handle)
        record = store.release_worker(handle, state="stopped")
        store.end_session(record.session_id)
        return pool.Stopped(record, killed=False, ended=True)

    return stop


def test_stop_refuses_a_session_with_a_user_interface(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(pool, "stop_worker", fake_stop(bench, seen))
    bench.session("s-2", "acc-1", kind="gui")
    _, [result] = talk(bench.serve(), ("hou_sessions", {"action": "stop", "session": "acc-1"}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "NOT_A_WORKER"
    assert "never closes" in text_of(result)
    assert seen == []


def test_stop_needs_a_session(bench: Bench) -> None:
    bench.session("s-1", "w1")
    bench.worker("s-1", "wk-1")
    _, [result] = talk(bench.serve(), ("hou_sessions", {"action": "stop"}))
    assert result.structured_content["error"]["code"] == "BAD_ARGUMENTS"
    assert "session" in text_of(result)


def test_stop_refuses_a_hython_the_pool_did_not_start(bench: Bench) -> None:
    bench.session("s-1", "hand-1")
    _, [result] = talk(bench.serve(), ("hou_sessions", {"action": "stop", "session": "hand-1"}))
    assert result.structured_content["error"]["code"] == "NOT_A_WORKER"


def test_stop_ends_a_worker_and_says_so(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(pool, "stop_worker", fake_stop(bench, seen))
    bench.session("s-1", "w1")
    bench.worker("s-1", "wk-1")
    arguments = {"action": "stop", "session": "w1", "operation_id": "stop-1"}
    _, [result, again] = talk(
        bench.serve(), ("hou_sessions", arguments), ("hou_sessions", arguments)
    )
    assert not result.is_error, text_of(result)
    stopped = result.structured_content["stopped"]
    assert stopped == {
        "session_id": "s-1",
        "alias": "w1",
        "ended": True,
        "killed": False,
        "note": None,
    }
    assert seen == ["wk-1"]
    assert "s-1" in bench.forgotten
    assert result.structured_content["trace"]["operation_id"] == "stop-1"
    # The same id again is answered from the receipt: nothing is stopped twice.
    assert again.structured_content["replayed"] is True
    assert seen == ["wk-1"]


def test_a_worker_whose_port_went_quiet_can_still_be_stopped(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(pool, "stop_worker", fake_stop(bench, seen))
    bench.session("s-1", "w1")
    bench.worker("s-1", "wk-1")
    with bench.store() as store:
        store.touch_session("s-1", state="unresponsive", transport_ok=False)
    _, [result] = talk(bench.serve(), ("hou_sessions", {"action": "stop", "session": "w1"}))
    assert not result.is_error, text_of(result)
    assert seen == ["wk-1"]


def test_stopping_an_ended_session_says_it_has_ended(bench: Bench) -> None:
    bench.session("s-1", "w1", pid=DEAD_PID)
    _, [result] = talk(bench.serve(), ("hou_sessions", {"action": "stop", "session": "s-1"}))
    assert result.structured_content["error"]["code"] == "SESSION_DEAD"
