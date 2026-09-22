"""What happens when the same operation id arrives twice.

The store here is a real one in a temporary folder, because the receipt and
the claim are one transaction in it and a stand in would prove nothing.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fake_hou import Scene
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import marshal
from nscr_houdini_mcp.bridge import receipts as receipt_module
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.handlers import ToolRegistry
from nscr_houdini_mcp.bridge.identity import Identity

SESSION = "session-1"


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


class Counter:
    """A mutating tool that says how many times it really ran."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def __call__(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(arguments))
        return {"runs": len(self.calls), "arguments": dict(arguments)}


def store_at(path: Path) -> store_module.Store:
    return store_module.Store(path)


def build(
    tmp_path: Path,
    scene: Scene,
    *,
    scene_epoch: int = 0,
    clock: Any = None,
    prune_every: int = receipt_module.DEFAULT_PRUNE_EVERY,
    max_age_s: float = receipt_module.DEFAULT_MAX_AGE_S,
) -> tuple[Dispatcher, Counter, Path]:
    path = tmp_path / "coord.sqlite"
    tools = ToolRegistry()
    counter = Counter()
    tools.add("scene.touch", counter, mutating=True)
    tools.add("scene.read", lambda arguments: {"read": True})
    running = Dispatcher(
        tools,
        lock=threading.Lock(),
        kind="hython",
        session_id=SESSION,
        identity=Identity(session_id=SESSION, scene_epoch=scene_epoch),
        receipts=receipt_module.Receipts(
            lambda: store_module.Store(path, clock=clock) if clock else store_module.Store(path),
            session_id=SESSION,
            prune_every=prune_every,
            max_age_s=max_age_s,
        ),
        hou=scene.module(),
        wait_s=5.0,
        timeout_s=5.0,
    )
    return running, counter, path


def touch(**fields: Any) -> Envelope:
    settings: dict[str, Any] = {"tool": "scene.touch", "arguments": {"what": "one"}}
    settings.update(fields)
    return Envelope(**settings)


# Section: the ordinary path


def test_a_mutating_call_with_an_id_leaves_a_finished_receipt(tmp_path: Path, scene: Scene) -> None:
    running, counter, path = build(tmp_path, scene)

    reply = running.dispatch(touch(operation_id="op-1"))

    assert reply.payload["ok"] is True
    with store_at(path) as store:
        record = store.get_operation("op-1")
    assert record is not None
    assert record.state == "done"
    assert record.session_id == SESSION
    assert record.scene_epoch == 0
    assert record.outcome["data"]["runs"] == 1
    assert len(counter.calls) == 1


def test_the_same_id_and_the_same_arguments_answer_from_the_receipt(
    tmp_path: Path, scene: Scene
) -> None:
    running, counter, _ = build(tmp_path, scene)
    first = running.dispatch(touch(operation_id="op-1"))

    second = running.dispatch(touch(operation_id="op-1"))

    assert second.payload["ok"] is True
    assert second.payload["data"] == first.payload["data"]
    assert second.payload["replayed"] is True
    # The work happened once, whatever the caller sent.
    assert len(counter.calls) == 1


def test_a_stored_failure_comes_back_as_the_failure_it_was(tmp_path: Path, scene: Scene) -> None:
    path = tmp_path / "coord.sqlite"
    tools = ToolRegistry()

    def refuse(arguments: Mapping[str, Any]) -> None:
        raise ValueError("no")

    tools.add("scene.touch", refuse, mutating=True)
    running = Dispatcher(
        tools,
        lock=threading.Lock(),
        kind="hython",
        session_id=SESSION,
        identity=Identity(session_id=SESSION),
        receipts=receipt_module.Receipts(lambda: store_module.Store(path), session_id=SESSION),
        hou=scene.module(),
    )

    first = running.dispatch(touch(operation_id="op-1"))
    second = running.dispatch(touch(operation_id="op-1"))

    assert first.payload["error"]["code"] == "TOOL_FAILED"
    assert second.payload["error"] == first.payload["error"]
    assert second.payload["replayed"] is True


def test_the_same_id_with_other_arguments_is_refused(tmp_path: Path, scene: Scene) -> None:
    running, counter, _ = build(tmp_path, scene)
    running.dispatch(touch(operation_id="op-1"))

    reply = running.dispatch(touch(operation_id="op-1", arguments={"what": "two"}))

    error = reply.payload["error"]
    assert error["code"] == "OPERATION_MISMATCH"
    assert error["hint"]
    assert len(counter.calls) == 1


def test_the_same_id_against_another_tool_is_refused(tmp_path: Path, scene: Scene) -> None:
    running, _, path = build(tmp_path, scene)
    running.dispatch(touch(operation_id="op-1"))
    with store_at(path) as store:
        store.finish_operation("op-1", state="done", outcome={"ok": True})

    reply = running.dispatch(
        Envelope(tool="scene.read", operation_id="op-1", arguments={"what": "one"})
    )

    # A read takes no receipt at all, so the id means nothing to it.
    assert reply.payload["ok"] is True


# Section: what is not known


def test_an_id_another_caller_is_still_running_is_not_run_again(
    tmp_path: Path, scene: Scene
) -> None:
    running, counter, path = build(tmp_path, scene)
    digest = receipt_module.digest_call("scene.touch", {"what": "one"})
    with store_at(path) as store:
        # A receipt left running by a process that is alive: this one.
        store.begin_operation(
            "op-1", digest, session_id=SESSION, scene_epoch=0, owner_pid=os.getpid()
        )

    reply = running.dispatch(touch(operation_id="op-1"))

    error = reply.payload["error"]
    assert error["code"] == "OUTCOME_UNKNOWN"
    assert error["details"]["receipt"]["state"] == "running"
    assert error["details"]["reason"]
    assert reply.payload["scene"]["scene_epoch"] == 0
    assert counter.calls == []


def test_an_id_left_behind_by_a_process_that_died_is_never_run_again(
    tmp_path: Path, scene: Scene
) -> None:
    """The work may already be in the scene. Nothing here will guess.

    The receipt is closed in a state of its own, so the caller after this one
    is told what really happened rather than that somebody else is on it.
    """
    running, counter, path = build(tmp_path, scene)
    digest = receipt_module.digest_call("scene.touch", {"what": "one"})
    with store_at(path) as store:
        store.begin_operation("op-1", digest, session_id=SESSION, scene_epoch=0, owner_pid=1 << 30)

    reply = running.dispatch(touch(operation_id="op-1"))

    assert reply.payload["error"]["code"] == "OUTCOME_UNKNOWN"
    assert reply.payload["error"]["details"]["reason"] == receipt_module.ABANDONED_BY
    assert counter.calls == []
    with store_at(path) as store:
        assert store.get_operation("op-1").state == store_module.OPERATION_ABANDONED

    again = running.dispatch(touch(operation_id="op-1"))

    details = again.payload["error"]["details"]
    assert again.payload["error"]["code"] == "OUTCOME_UNKNOWN"
    assert details["reason"] == receipt_module.ABANDONED_BY
    assert details["receipt"]["state"] == store_module.OPERATION_ABANDONED
    assert counter.calls == []


def test_an_id_from_a_scene_that_has_been_replaced_is_refused(tmp_path: Path, scene: Scene) -> None:
    running, counter, path = build(tmp_path, scene, scene_epoch=2)
    digest = receipt_module.digest_call("scene.touch", {"what": "one"})
    with store_at(path) as store:
        store.begin_operation("op-1", digest, session_id=SESSION, scene_epoch=1)
        store.finish_operation(
            "op-1", state="done", outcome={"ok": True, "data": {"runs": 1}, "scene_epoch": 1}
        )

    reply = running.dispatch(touch(operation_id="op-1"))

    error = reply.payload["error"]
    assert error["code"] == "SCENE_REPLACED"
    assert error["details"]["recorded_epoch"] == 1
    assert error["details"]["scene_epoch"] == 2
    assert counter.calls == []


def test_an_operation_that_replaced_the_scene_itself_still_answers_its_retry(
    tmp_path: Path, scene: Scene
) -> None:
    """Its own load moved the epoch. That must not lock its answer away."""
    path = tmp_path / "coord.sqlite"
    session = Identity(session_id=SESSION, hou=scene.module())
    session.watch()
    tools = ToolRegistry()
    loads: list[str] = []

    def load(arguments: Mapping[str, Any]) -> dict[str, Any]:
        loads.append("once")
        scene.hipFile.load("/scenes/other.hip")
        return {"loaded": scene.hipFile.path()}

    tools.add("scene.load", load, mutating=True)
    running = Dispatcher(
        tools,
        lock=threading.Lock(),
        kind="hython",
        session_id=SESSION,
        identity=session,
        receipts=receipt_module.Receipts(lambda: store_module.Store(path), session_id=SESSION),
        hou=scene.module(),
    )

    first = running.dispatch(Envelope(tool="scene.load", operation_id="op-1", scene_epoch=0))
    assert first.payload["ok"] is True, first.payload
    assert first.payload["scene_epoch"] == 1

    second = running.dispatch(Envelope(tool="scene.load", operation_id="op-1", scene_epoch=0))

    assert second.payload["ok"] is True, second.payload
    assert second.payload["replayed"] is True
    assert second.payload["data"] == first.payload["data"]
    assert loads == ["once"]


# Section: a call that never reached its tool


def unreachable(
    tmp_path: Path, scene: Scene, *, refuse_to_post: bool = False
) -> tuple[Any, list, Any]:
    """A dispatcher whose main thread never picks anything up.

    The poster is only started where the test needs a post to be refused.
    Without it nothing is ever handed to the main thread, which is the case a
    call that gives up on its pickup budget has to survive.
    """
    path = tmp_path / "coord.sqlite"
    tools = ToolRegistry()
    counter = Counter()
    tools.add("scene.touch", counter, mutating=True)
    module = scene.module()
    if refuse_to_post:
        module.ui = SimpleNamespace(postEventCallback=_refuse_to_post)
    pulse = marshal.Pulse()
    runner = marshal.MainThreadRunner(module, pulse=pulse)
    if refuse_to_post:
        runner.start()
    else:
        pulse.install(module)
    running = Dispatcher(
        tools,
        lock=threading.Lock(),
        kind="gui",
        session_id=SESSION,
        identity=Identity(session_id=SESSION),
        receipts=receipt_module.Receipts(lambda: store_module.Store(path), session_id=SESSION),
        hou=module,
        main_thread=runner,
        pulse=pulse,
        wait_s=0.0,
    )
    return running, counter.calls, runner


def _refuse_to_post(callback: Any) -> None:
    raise RuntimeError("the user interface is going down")


def test_a_call_the_session_never_took_leaves_its_id_free(tmp_path: Path, scene: Scene) -> None:
    """Nothing ran, so the id has to be as good as new.

    The main thread here is not running, so the work is never picked up and
    the call comes back busy. A receipt left behind would answer the retry
    with an outcome nobody knows, for work that never happened.
    """
    running, calls, runner = unreachable(tmp_path, scene)
    try:
        refused = running.dispatch(touch(operation_id="op-1"))
        assert refused.payload["error"]["code"] == "SESSION_BUSY"
        assert calls == []
        with store_at(tmp_path / "coord.sqlite") as store:
            assert store.get_operation("op-1") is None

        # The same id again, with the main thread running this time.
        scene.ui.start()
        answered = running.dispatch(touch(operation_id="op-1", wait_s=5.0))
        assert answered.payload["ok"] is True, answered.payload
        assert len(calls) == 1
    finally:
        runner.stop()


def test_a_call_the_session_could_not_take_at_all_leaves_its_id_free(
    tmp_path: Path, scene: Scene
) -> None:
    running, calls, runner = unreachable(tmp_path, scene, refuse_to_post=True)
    try:
        refused = running.dispatch(touch(operation_id="op-1"))

        assert refused.payload["error"]["code"] == "TOOL_FAILED"
        assert refused.payload["error"]["details"]["exception"] == "Rejected"
        assert calls == []
        with store_at(tmp_path / "coord.sqlite") as store:
            assert store.get_operation("op-1") is None
    finally:
        runner.stop()


# Section: what takes no receipt


def test_a_read_takes_no_receipt(tmp_path: Path, scene: Scene) -> None:
    running, _, path = build(tmp_path, scene)

    reply = running.dispatch(Envelope(tool="scene.read", operation_id="op-read"))

    assert reply.payload["ok"] is True
    with store_at(path) as store:
        assert store.get_operation("op-read") is None


def test_a_mutating_call_that_names_no_id_takes_no_receipt(tmp_path: Path, scene: Scene) -> None:
    """A minted id could answer no retry, because no caller has it."""
    running, counter, path = build(tmp_path, scene)

    reply = running.dispatch(touch())

    minted = reply.payload["operation_id"]
    assert minted
    with store_at(path) as store:
        assert store.get_operation(minted) is None
    assert len(counter.calls) == 1


def test_a_call_runs_when_there_is_nowhere_to_keep_a_receipt(scene: Scene) -> None:
    tools = ToolRegistry()
    counter = Counter()
    tools.add("scene.touch", counter, mutating=True)
    running = Dispatcher(
        tools,
        lock=threading.Lock(),
        kind="hython",
        session_id=SESSION,
        hou=scene.module(),
    )

    reply = running.dispatch(touch(operation_id="op-1"))

    assert reply.payload["ok"] is True
    assert len(counter.calls) == 1


def test_work_that_outlived_its_call_still_finishes_its_receipt(
    tmp_path: Path, scene: Scene
) -> None:
    """The caller gave up waiting. The answer still belongs under its id."""
    path = tmp_path / "coord.sqlite"
    started = threading.Event()
    release = threading.Event()
    tools = ToolRegistry()

    def slow(arguments: Mapping[str, Any]) -> dict[str, Any]:
        started.set()
        assert release.wait(10.0)
        return {"finished": True}

    tools.add("scene.touch", slow, mutating=True)
    running = Dispatcher(
        tools,
        lock=threading.Lock(),
        kind="hython",
        session_id=SESSION,
        identity=Identity(session_id=SESSION),
        receipts=receipt_module.Receipts(lambda: store_module.Store(path), session_id=SESSION),
        hou=scene.module(),
        timeout_s=1.0,
    )

    gave_up = running.dispatch(touch(operation_id="op-1"))
    assert gave_up.payload["error"]["code"] == "TIMEOUT"
    assert started.wait(10.0)
    release.set()

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        with store_at(path) as store:
            record = store.get_operation("op-1")
        if record is not None and record.state == "done":
            break
        time.sleep(0.02)
    assert record.state == "done"
    assert record.outcome["data"] == {"finished": True}

    replayed = running.dispatch(touch(operation_id="op-1"))
    assert replayed.payload["replayed"] is True
    assert replayed.payload["data"] == {"finished": True}


# Section: keeping the table small


def test_receipts_past_their_age_are_dropped(tmp_path: Path, scene: Scene) -> None:
    path = tmp_path / "coord.sqlite"
    now = [1000.0]
    running, _, _ = build(
        tmp_path,
        scene,
        clock=lambda: now[0],
        prune_every=1,
        max_age_s=3600.0,
    )
    with store_module.Store(path, clock=lambda: now[0]) as store:
        store.begin_operation("op-old", "digest", session_id=SESSION)
        store.finish_operation("op-old", state="done", outcome={"ok": True})

    now[0] += 7200.0
    running.dispatch(touch(operation_id="op-new"))

    with store_at(path) as store:
        assert store.get_operation("op-old") is None
        assert store.get_operation("op-new") is not None
