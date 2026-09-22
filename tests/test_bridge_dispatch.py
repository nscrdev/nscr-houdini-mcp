"""The rules around one tool call, against a stand in for Houdini.

What a real Houdini has to confirm is in the integration tests. What is here
is everything that can be decided without one: the order calls are served in,
the two budgets, where the work runs, what an undo group does when a call
fails, the error codes, and what the encoder will and will not carry.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from fake_hou import InvalidInput, Matrix4, ObjectWasDeleted, OperationFailed, Scene, Vector3
from nscr_houdini_mcp.bridge import encoding, errors, marshal
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.errors import BridgeError
from nscr_houdini_mcp.bridge.gate import Gate
from nscr_houdini_mcp.bridge.handlers import ToolRegistry, default_registry
from nscr_houdini_mcp.bridge.undo import run_in_undo_group
from thread_guard import Guard


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


def dispatcher(
    tools: ToolRegistry | None = None,
    *,
    kind: str = "hython",
    hou: Any = None,
    wait_s: float = 5.0,
    timeout_s: float = 5.0,
) -> Dispatcher:
    return Dispatcher(
        tools if tools is not None else default_registry(selfcheck=True),
        lock=threading.Lock(),
        kind=kind,
        session_id="session-1",
        hou=hou,
        wait_s=wait_s,
        timeout_s=timeout_s,
    )


@pytest.fixture
def gui() -> Iterator[Any]:
    """Dispatchers for a session with a user interface, taken down again.

    Every one of them owns what a bridge owns: a pulse watching the main
    thread and a runner with a poster thread, so the tests exercise the same
    path a real session does.
    """
    made: list[tuple[marshal.MainThreadRunner, marshal.Pulse]] = []

    def build(
        scene: Scene,
        *,
        tools: ToolRegistry | None = None,
        module: Any = None,
        wait_s: float = 5.0,
        timeout_s: float = 5.0,
        stale_s: float = marshal.DEFAULT_STALE_S,
        clock: Any = None,
        install: bool = True,
        start_poster: bool = True,
    ) -> Dispatcher:
        given = scene.module() if module is None else module
        pulse = marshal.Pulse(stale_s=stale_s, **({} if clock is None else {"clock": clock}))
        runner = marshal.MainThreadRunner(given, pulse=pulse)
        made.append((runner, pulse))
        if install:
            pulse.install(given)
        if start_poster:
            runner.start()
        return Dispatcher(
            tools if tools is not None else default_registry(selfcheck=True),
            lock=threading.Lock(),
            kind="gui",
            session_id="session-1",
            hou=given,
            wait_s=wait_s,
            timeout_s=timeout_s,
            main_thread=runner,
            pulse=pulse,
        )

    try:
        yield build
    finally:
        for runner, pulse in made:
            runner.stop()
            pulse.uninstall()


def call(tool: str, **fields: Any) -> Envelope:
    return Envelope(tool=tool, **fields)


class Blocker:
    """A tool that holds the session until it is let go."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.order: list[Any] = []

    def __call__(self, arguments: Mapping[str, Any]) -> Any:
        self.started.set()
        assert self.release.wait(10.0)
        return {"held": True}


# Section: order


def test_waiting_calls_are_served_in_the_order_they_arrived() -> None:
    blocker = Blocker()
    served: list[str] = []
    tools = ToolRegistry()
    tools.add("bridge.block", blocker)
    tools.add("bridge.mark", lambda arguments: served.append(arguments["who"]) or {"ok": True})
    running = dispatcher(tools, wait_s=20.0)

    holder = threading.Thread(target=lambda: running.dispatch(call("bridge.block")))
    holder.start()
    assert blocker.started.wait(10.0)

    waiters = []
    for who in ("first", "second", "third"):
        waiter = threading.Thread(
            target=lambda who=who: running.dispatch(
                call("bridge.mark", arguments={"who": who}, wait_s=20.0)
            )
        )
        waiters.append(waiter)
        waiter.start()
        # Each one is queued before the next arrives, so arrival order is known.
        queued = len(waiters)
        _until(lambda queued=queued: running.state()["queued"] == queued)

    blocker.release.set()
    for waiter in waiters:
        waiter.join(10.0)
    holder.join(10.0)

    assert served == ["first", "second", "third"]


def test_a_call_that_gives_up_takes_its_place_in_the_queue_with_it() -> None:
    gate = Gate(threading.Lock())
    assert gate.enter(wait_s=0.0) is True
    assert gate.enter(wait_s=0.05) is False
    assert gate.waiting() == 0
    gate.leave()
    assert gate.enter(wait_s=0.0) is True
    gate.leave()


# Section: busy


def test_skip_if_busy_is_answered_at_once() -> None:
    blocker = Blocker()
    tools = ToolRegistry()
    tools.add("bridge.block", blocker)
    running = dispatcher(tools, wait_s=20.0)

    holder = threading.Thread(target=lambda: running.dispatch(call("bridge.block")))
    holder.start()
    try:
        assert blocker.started.wait(10.0)
        began = time.monotonic()
        reply = running.dispatch(call("bridge.block", wait_s=20.0, skip_if_busy=True))
        assert time.monotonic() - began < 1.0
        error = reply.payload["error"]
        assert error["code"] == "SESSION_BUSY"
        assert error["details"]["current_op"] == "bridge.block"
        assert error["details"]["cause"] == "session busy"
    finally:
        blocker.release.set()
        holder.join(10.0)


def test_a_call_that_will_not_wait_is_told_the_session_is_busy() -> None:
    blocker = Blocker()
    tools = ToolRegistry()
    tools.add("bridge.block", blocker)
    running = dispatcher(tools, wait_s=20.0)
    holder = threading.Thread(target=lambda: running.dispatch(call("bridge.block")))
    holder.start()
    try:
        assert blocker.started.wait(10.0)
        reply = running.dispatch(call("bridge.block", wait_s=0.0))
        assert reply.payload["error"]["code"] == "SESSION_BUSY"
    finally:
        blocker.release.set()
        holder.join(10.0)


# Section: the two budgets


def test_work_that_outlives_its_timeout_answers_and_keeps_running() -> None:
    blocker = Blocker()
    tools = ToolRegistry()
    tools.add("bridge.block", blocker)
    tools.add("bridge.after", lambda arguments: {"after": True})
    running = dispatcher(tools, wait_s=10.0)

    reply = running.dispatch(call("bridge.block", timeout_s=0.3))
    error = reply.payload["error"]
    assert error["code"] == "TIMEOUT"
    assert error["details"]["still_running"] is True
    assert error["details"]["operation_id"] == reply.payload["operation_id"]

    # The session is still held by the work that did not finish.
    state = running.state()
    assert state["busy"] is True
    assert state["current_op"] == "bridge.block"
    assert state["current_op_id"] == error["details"]["operation_id"]
    assert state["current_op_timed_out"] is True

    busy = running.dispatch(call("bridge.after", wait_s=0.0))
    assert busy.payload["error"]["code"] == "SESSION_BUSY"

    blocker.release.set()
    _until(lambda: running.state()["busy"] is False)
    later = running.dispatch(call("bridge.after", wait_s=5.0))
    assert later.payload["ok"] is True
    assert running.state()["last_op"]["tool"] == "bridge.after"


def test_a_run_budget_of_nothing_is_taken_at_its_word() -> None:
    """Zero means answer now, not fall back to the default."""
    blocker = Blocker()
    tools = ToolRegistry()
    tools.add("bridge.block", blocker)
    running = dispatcher(tools, timeout_s=600.0)
    began = time.monotonic()
    try:
        reply = running.dispatch(call("bridge.block", timeout_s=0.0))
        assert time.monotonic() - began < 5.0
        assert reply.payload["error"]["code"] == "TIMEOUT"
    finally:
        blocker.release.set()
        _until(lambda: running.state()["busy"] is False)


def test_work_that_cannot_be_handed_over_gives_the_session_back() -> None:
    """A session held by nothing is worse than a call that failed."""

    class Refusing:
        kind = "gui"

        def submit(self, work: Any) -> Any:
            raise RuntimeError("the interface is going away")

    tools = ToolRegistry()
    tools.add("bridge.edit", lambda arguments: {"edited": True})
    running = dispatcher(tools)
    marshal_choose = marshal.choose_runner
    try:
        marshal.choose_runner = lambda *args, **rest: Refusing()
        reply = running.dispatch(call("bridge.edit", timeout_s=5.0))
    finally:
        marshal.choose_runner = marshal_choose

    error = reply.payload["error"]
    assert error["code"] == "TOOL_FAILED"
    assert error["details"]["exception"] == "RuntimeError"
    assert running.state()["busy"] is False
    # And the session really is free, not just reported free.
    assert running.dispatch(call("bridge.edit", wait_s=1.0)).payload["ok"] is True


def test_a_value_that_will_not_describe_itself_still_comes_back() -> None:
    """A deleted node raises from its own repr, while the reply is being built."""

    class Deleted:
        def __repr__(self) -> str:
            raise RuntimeError("this object was deleted")

    tools = ToolRegistry()
    tools.add("bridge.gone", lambda arguments: {"node": Deleted()})
    reply = dispatcher(tools).dispatch(call("bridge.gone", timeout_s=5.0))
    assert reply.payload["ok"] is True
    assert reply.payload["data"]["node"] == "<Deleted>"
    assert reply.payload["lossy"] is True


# Section: cancelling


def test_a_running_call_can_be_asked_to_stop(scene: Scene) -> None:
    running = dispatcher(hou=scene.module(), wait_s=10.0)
    answers: list[Any] = []
    holder = threading.Thread(
        target=lambda: answers.append(
            running.dispatch(call("bridge.selfcheck", arguments={"sleep_s": 30.0}, timeout_s=60.0))
        )
    )
    holder.start()
    try:
        _until(lambda: running.state()["busy"] is True)
        operation_id = running.state()["current_op_id"]
        asked = running.dispatch(call("bridge.cancel"))
        assert asked.payload["ok"] is True
        assert asked.payload["data"]["asked"] is True
        assert asked.payload["data"]["operation_id"] == operation_id
    finally:
        holder.join(30.0)

    reply = answers[0]
    assert reply.payload["ok"] is True
    assert reply.payload["data"]["stopped_early"] is True
    assert reply.payload["data"]["slept_s"] < 30.0


def test_cancelling_names_what_it_did_not_cancel() -> None:
    running = dispatcher()
    assert running.dispatch(call("bridge.cancel")).payload["data"]["asked"] is False
    assert "bridge.cancel" in running.tools.names()


def test_the_wait_budget_and_the_run_budget_are_separate() -> None:
    tools = ToolRegistry()
    tools.add("bridge.slow", lambda arguments: time.sleep(0.4) or {"slow": True})
    running = dispatcher(tools)
    reply = running.dispatch(call("bridge.slow", wait_s=0.5, timeout_s=10.0))
    assert reply.payload["ok"] is True


# Section: where the work runs


def test_a_mutating_call_in_a_session_with_a_interface_runs_on_the_main_thread(
    scene: Scene, gui: Any
) -> None:
    scene.ui.start()
    running = gui(scene, install=False)
    reply = running.dispatch(
        call("node.create", arguments={"parent": "/obj", "type": "geo"}, timeout_s=10.0)
    )
    assert reply.payload["ok"] is True
    assert reply.payload["data"]["path"] == "/obj/geo1"
    assert reply.payload["picked_by"] == "kick"
    assert scene.ui.ran_on == ["fake-main"]


def test_a_read_in_a_session_with_an_interface_runs_on_the_main_thread(
    scene: Scene, gui: Any
) -> None:
    """A read is marshalled like a mutation, and for the same reasons.

    Off the main thread it waits on the object model lock for as long as a
    cook lasts, and reads ambient state that is not the session's.
    """
    scene.ui.start()
    running = gui(scene, install=False)
    reply = running.dispatch(call("scene.info", timeout_s=10.0))
    assert reply.payload["ok"] is True
    assert reply.payload["picked_by"] == "kick"
    assert scene.ui.ran_on == ["fake-main"]


def test_scene_info_reports_the_frame_the_main_thread_sees(scene: Scene, gui: Any) -> None:
    scene.ui.start()
    running = gui(scene)
    reply = running.dispatch(call("scene.info", timeout_s=10.0))
    assert reply.payload["data"]["frame"] == 72.0


def test_a_main_thread_that_never_picks_the_work_up_says_it_is_busy(scene: Scene, gui: Any) -> None:
    running = gui(scene, wait_s=0.3, install=False)
    reply = running.dispatch(
        call("node.create", arguments={"parent": "/obj", "type": "geo"}, wait_s=0.3)
    )
    error = reply.payload["error"]
    assert error["code"] == "SESSION_BUSY"
    assert error["details"]["cause"] == "main thread busy"
    assert error["details"]["picked_up"] is False
    # The session is free again, and the work that was posted never runs.
    assert running.state()["busy"] is False
    scene.ui.start()
    time.sleep(0.2)
    assert scene.node("/obj").children() == ()


# Section: a busy main thread


def test_a_call_during_a_blocking_cook_is_refused_within_its_wait_and_never_runs_late(
    scene: Scene, gui: Any
) -> None:
    scene.ui.start()
    running = gui(scene, wait_s=0.3)
    scene.ui.cook(1.2)
    _until(lambda: scene.ui.ran_on != [])

    began = time.monotonic()
    reply = running.dispatch(
        call("node.create", arguments={"parent": "/obj", "type": "geo"}, wait_s=0.3)
    )
    assert time.monotonic() - began < 0.6
    error = reply.payload["error"]
    assert error["code"] == "SESSION_BUSY"
    assert error["details"]["cause"] == "main thread busy"
    assert error["details"]["picked_up"] is False
    assert running.state()["busy"] is False

    _until(lambda: scene.ui.ticks > 0, timeout_s=5.0)
    time.sleep(0.3)
    assert scene.node("/obj").children() == ()
    assert scene.undos.undoLabels() == []

    later = running.dispatch(
        call("node.create", arguments={"parent": "/obj", "type": "geo"}, wait_s=5.0)
    )
    assert later.payload["ok"] is True
    assert scene.undos.undoLabels() == ["create node"]


def test_a_read_during_a_blocking_cook_is_refused_the_same_way(scene: Scene, gui: Any) -> None:
    """A read is no worse off than a mutation, and no better."""
    scene.ui.start()
    running = gui(scene, wait_s=1.0, stale_s=0.2)
    scene.ui.cook(2.5)
    _until(lambda: scene.ui.ran_on != [])

    began = time.monotonic()
    reply = running.dispatch(call("scene.info"))
    assert time.monotonic() - began < 1.3
    assert reply.payload["error"]["details"]["cause"] == "main thread busy"

    began = time.monotonic()
    reply = running.dispatch(call("scene.info", wait_s=0.0))
    assert time.monotonic() - began < 0.5
    assert reply.payload["error"]["details"]["cause"] == "main thread busy"

    began = time.monotonic()
    reply = running.dispatch(call("scene.info", skip_if_busy=True))
    assert time.monotonic() - began < 0.05
    assert reply.payload["error"]["details"]["cause"] == "main thread busy"


def test_a_call_that_will_not_wait_is_refused_inside_its_own_promise(
    scene: Scene, gui: Any
) -> None:
    """A cook younger than the stale limit still refuses a skipping caller.

    The stale limit is seconds long, and a caller that asked to be skipped
    wants an answer now, so the short skip threshold is what it is judged by
    instead of a pickup budget it would have to sit through.
    """
    scene.ui.start()
    running = gui(scene, wait_s=1.0, stale_s=5.0)
    scene.ui.cook(2.0)
    _until(lambda: scene.ui.ran_on != [])
    # The main thread has been away for far less than the stale limit.
    _until(lambda: running._pulse.age_s() > running.skip_stale_s)
    assert running._pulse.away() is False

    began = time.monotonic()
    reply = running.dispatch(call("scene.info", skip_if_busy=True))
    took = time.monotonic() - began

    assert took < 0.1, f"a skipping call took {took:.3f} s"
    error = reply.payload["error"]
    assert error["code"] == "SESSION_BUSY"
    assert error["details"]["cause"] == "main thread busy"
    assert error["details"]["picked_up"] is False
    assert scene.node("/obj") is not None


def test_a_call_that_will_not_wait_is_refused_when_work_is_already_queued(
    scene: Scene, gui: Any
) -> None:
    """Queued work means the main thread is spoken for, whatever the pulse says."""
    running = gui(scene, wait_s=1.0, stale_s=5.0, start_poster=False)
    running._main_thread.submit(marshal.Work(lambda: None))
    assert running._main_thread.pending == 1

    began = time.monotonic()
    reply = running.dispatch(call("scene.info", skip_if_busy=True))
    assert time.monotonic() - began < 0.1
    assert reply.payload["error"]["code"] == "SESSION_BUSY"
    assert reply.payload["error"]["details"]["cause"] == "main thread busy"


def test_a_call_that_will_not_wait_still_runs_on_a_free_session(scene: Scene, gui: Any) -> None:
    scene.ui.start()
    running = gui(scene, wait_s=1.0, stale_s=5.0)
    _until(lambda: scene.ui.ticks > 1)
    reply = running.dispatch(call("scene.info", skip_if_busy=True))
    assert reply.payload["ok"] is True


def test_a_stale_main_thread_is_refused_before_anything_is_posted(scene: Scene, gui: Any) -> None:
    now = [100.0]
    running = gui(scene, wait_s=1.0, stale_s=2.0, clock=lambda: now[0])
    now[0] = 103.0

    began = time.monotonic()
    reply = running.dispatch(call("scene.info", wait_s=1.0))
    assert time.monotonic() - began < 0.05
    error = reply.payload["error"]
    assert error["code"] == "SESSION_BUSY"
    assert error["details"]["cause"] == "main thread busy"
    assert error["details"]["main_thread_idle_s"] == 3.0
    assert error["details"]["picked_up"] is False
    # Nothing was handed to the main thread at all.
    assert scene.ui.posted.empty()


def test_a_caller_that_will_wait_longer_than_the_pulse_is_stale_gets_to_queue(
    scene: Scene, gui: Any
) -> None:
    now = [100.0]
    running = gui(scene, wait_s=5.0, stale_s=2.0, clock=lambda: now[0])
    now[0] = 103.0

    waking = threading.Timer(0.3, scene.ui.start)
    waking.start()
    try:
        reply = running.dispatch(call("scene.info", wait_s=5.0, timeout_s=5.0))
    finally:
        waking.cancel()
    assert reply.payload["ok"] is True


def test_busy_is_decided_without_touching_hou(scene: Scene, gui: Any) -> None:
    """The thread that answers a request calls nothing in `hou`, ever."""
    now = [100.0]
    guard = Guard(scene.module())
    running = gui(scene, module=guard, wait_s=1.0, stale_s=2.0, clock=lambda: now[0])
    now[0] = 103.0
    guard.forget()

    here = threading.current_thread().name
    reply = running.dispatch(call("scene.info", wait_s=1.0))
    assert reply.payload["error"]["details"]["cause"] == "main thread busy"
    assert guard.touched_by(here) == []


def test_a_queue_behind_a_busy_main_thread_drains_at_arrival_rate(scene: Scene, gui: Any) -> None:
    scene.ui.start()
    running = gui(scene, wait_s=0.3, stale_s=0.2)
    scene.ui.cook(1.5)
    _until(lambda: scene.ui.ran_on != [])

    answers: list[tuple[float, Any]] = []

    def ask() -> None:
        began = time.monotonic()
        reply = running.dispatch(
            call("node.create", arguments={"parent": "/obj", "type": "geo"}, wait_s=0.3)
        )
        answers.append((time.monotonic() - began, reply))

    askers = [threading.Thread(target=ask) for _ in range(3)]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(10.0)

    assert len(answers) == 3
    causes = []
    for took, reply in answers:
        # Each one is answered inside its own budget: the one that holds the
        # session is refused by the main thread, the ones behind it by the
        # session, and none of them waits for the cook.
        assert took < 1.5
        assert reply.payload["error"]["code"] == "SESSION_BUSY"
        causes.append(reply.payload["error"]["details"]["cause"])
    assert "main thread busy" in causes

    _until(lambda: scene.ui.ticks > 0, timeout_s=5.0)
    time.sleep(0.3)
    assert scene.node("/obj").children() == ()
    assert scene.undos.undoLabels() == []


def test_a_mutation_still_records_one_undo_entry_on_either_pickup_path(
    scene: Scene, gui: Any
) -> None:
    scene.ui.start()
    through_loop = gui(scene, start_poster=False)
    reply = through_loop.dispatch(
        call("node.create", arguments={"parent": "/obj", "type": "geo"}, timeout_s=10.0)
    )
    assert reply.payload["ok"] is True
    assert reply.payload["picked_by"] == "loop"
    assert scene.undos.undoLabels() == ["create node"]

    through_kick = gui(scene, install=False)
    reply = through_kick.dispatch(
        call("node.create", arguments={"parent": "/obj", "type": "geo"}, timeout_s=10.0)
    )
    assert reply.payload["ok"] is True
    assert reply.payload["picked_by"] == "kick"
    assert scene.undos.undoLabels() == ["create node", "create node"]


def test_a_rejected_post_answers_tool_failed_not_busy(scene: Scene, gui: Any) -> None:
    module = scene.module()
    module.ui = _RefusingInterface()
    running = gui(scene, module=module, install=False, wait_s=5.0)

    reply = running.dispatch(call("scene.info", wait_s=5.0, timeout_s=5.0))
    error = reply.payload["error"]
    assert error["code"] == "TOOL_FAILED"
    assert error["details"]["exception"] == "Rejected"
    # And the session is free, not held by work nobody is going to run.
    assert running.state()["busy"] is False


class _RefusingInterface:
    """An interface that will not take a posted callback."""

    def postEventCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        raise RuntimeError("the interface is going away")


def test_a_mutating_call_in_a_headless_session_runs_on_the_main_thread(scene: Scene) -> None:
    loop = marshal.MainLoop()
    stop = threading.Event()
    ran_on: list[str] = []
    main = threading.Thread(target=lambda: loop.run_until(stop), name="fake-process-main")
    main.start()
    _until(lambda: loop.running)
    tools = ToolRegistry()

    def edit(arguments: Mapping[str, Any]) -> Any:
        ran_on.append(threading.current_thread().name)
        scene.node("/obj").createNode("geo")
        return {"edited": True}

    tools.add("bridge.edit", edit, mutating=True, label="edit")
    running = Dispatcher(
        tools,
        lock=threading.Lock(),
        kind="hython",
        hou=scene.module(),
        main_loop=loop,
        wait_s=5.0,
        timeout_s=5.0,
    )
    try:
        reply = running.dispatch(call("bridge.edit", timeout_s=5.0))
    finally:
        stop.set()
        main.join(5.0)
    assert reply.payload["ok"] is True
    assert ran_on == ["fake-process-main"]
    assert reply.payload["undo"]["recorded"] is True


def test_a_mutating_call_with_no_main_thread_loop_says_nothing_was_recorded() -> None:
    """A bridge nobody is pumping still answers, and does not pretend to group.

    There is no `hou` here, so there is no undo group at all. The reply says
    so rather than claiming a step the artist could undo.
    """
    running = dispatcher()
    reply = running.dispatch(call("bridge.selfcheck", timeout_s=5.0))
    assert reply.payload["error"]["code"] == "TOOL_FAILED"
    assert reply.payload["error"]["details"]["undo_recorded"] is False


def test_the_marshal_seam_hands_back_a_result_and_gives_up_on_time(scene: Scene) -> None:
    module = scene.module()
    scene.ui.start()
    assert marshal.run_on_main_thread(lambda: 21 * 2, timeout_s=5.0, hou=module) == 42
    scene.ui.stop()
    with pytest.raises(marshal.MarshalTimeout):
        marshal.run_on_main_thread(lambda: None, timeout_s=1.0, pickup_s=0.2, hou=module)


# Section: the undo group


def test_a_failed_call_that_changed_the_graph_is_rolled_back(scene: Scene) -> None:
    running = dispatcher(hou=scene.module())
    reply = running.dispatch(
        call("bridge.selfcheck", arguments={"creates": 2, "fail_at": 2}, timeout_s=10.0)
    )
    error = reply.payload["error"]
    assert error["code"] == "TOOL_FAILED"
    assert error["details"]["rolled_back"] is True
    assert scene.node("/obj").children() == ()
    assert scene.undos.undoLabels() == []
    assert scene.undos.performed == 1


def test_a_failed_call_that_changed_nothing_is_not_rolled_back(scene: Scene) -> None:
    running = dispatcher(hou=scene.module())
    scene.node("/obj").createNode("geo", "kept")
    before = scene.undos.undoLabels()

    reply = running.dispatch(
        call("bridge.selfcheck", arguments={"creates": 1, "fail_at": 1}, timeout_s=10.0)
    )
    error = reply.payload["error"]
    assert error["code"] == "TOOL_FAILED"
    assert error["details"]["rolled_back"] is False
    # Somebody else's last edit is still there, which is the point.
    assert scene.undos.undoLabels() == before
    assert scene.undos.performed == 0
    assert scene.node("/obj/kept") is not None


def test_a_call_that_worked_leaves_one_undo_entry(scene: Scene) -> None:
    running = dispatcher(hou=scene.module())
    reply = running.dispatch(call("bridge.selfcheck", arguments={"creates": 3}, timeout_s=10.0))
    assert reply.payload["ok"] is True
    assert reply.payload["undo"] == {
        "label": "self check",
        "recorded": True,
        "rolled_back": False,
    }
    assert scene.undos.undoLabels() == ["self check"]
    assert len(scene.node("/obj").children()) == 3


def test_a_group_that_recorded_nothing_is_reported_as_nothing_to_undo(scene: Scene) -> None:
    outcome = run_in_undo_group(lambda: 1 + 1, label="read", hou=scene.module())
    assert outcome.value == 2
    assert outcome.recorded is False
    assert outcome.rolled_back is False


# Section: errors


def test_a_name_that_is_not_a_tool_comes_back_with_the_closest_ones() -> None:
    reply = dispatcher().dispatch(call("scene.inf"))
    error = reply.payload["error"]
    assert error["code"] == "UNKNOWN_TOOL"
    assert error["details"]["did_you_mean"] == ["scene.info"]


def test_an_argument_name_that_is_not_the_tools_comes_back_with_the_closest_ones() -> None:
    reply = dispatcher().dispatch(call("node.create", arguments={"paren": "/obj", "type": "geo"}))
    error = reply.payload["error"]
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["unknown"] == ["paren"]
    assert error["details"]["did_you_mean"][0] == "parent"
    assert "parms" in error["details"]["arguments"]


def test_a_required_argument_that_was_left_out_is_named() -> None:
    reply = dispatcher().dispatch(call("node.create", arguments={"parent": "/obj"}))
    error = reply.payload["error"]
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["missing"] == ["type"]


def test_a_parameter_name_that_is_not_on_the_node_comes_back_with_the_closest_ones(
    scene: Scene,
) -> None:
    running = dispatcher(hou=scene.module())
    reply = running.dispatch(
        call(
            "node.create",
            arguments={"parent": "/obj", "type": "geo", "parms": {"tx1": 1.0}},
            timeout_s=10.0,
        )
    )
    error = reply.payload["error"]
    assert error["code"] == "PARM_NOT_FOUND"
    assert error["details"]["did_you_mean"] == ["tx"]
    assert error["details"]["rolled_back"] is True
    assert scene.node("/obj").children() == ()


def test_a_path_that_is_not_in_the_scene_comes_back_with_the_closest_ones(scene: Scene) -> None:
    running = dispatcher(hou=scene.module())
    reply = running.dispatch(
        call("node.create", arguments={"parent": "/objj", "type": "geo"}, timeout_s=10.0)
    )
    error = reply.payload["error"]
    assert error["code"] == "NODE_NOT_FOUND"
    assert "/obj" in error["details"]["did_you_mean"]


def test_a_node_type_that_does_not_exist_comes_back_as_an_argument_mistake(scene: Scene) -> None:
    running = dispatcher(hou=scene.module())
    reply = running.dispatch(
        call("node.create", arguments={"parent": "/obj", "type": "gep"}, timeout_s=10.0)
    )
    error = reply.payload["error"]
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["did_you_mean"] == ["geo"]


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (ObjectWasDeleted("gone"), "NODE_NOT_FOUND"),
        (InvalidInput("no"), "BAD_ARGUMENTS"),
        (OperationFailed("no"), "TOOL_FAILED"),
        (TypeError("takes 2 positional arguments"), "BAD_ARGUMENTS"),
        (RuntimeError("boom"), "TOOL_FAILED"),
    ],
)
def test_what_houdini_raises_becomes_a_code(raised: Exception, code: str) -> None:
    tools = ToolRegistry()

    def explode(arguments: Mapping[str, Any]) -> Any:
        raise raised

    tools.add("bridge.explode", explode)
    reply = dispatcher(tools).dispatch(call("bridge.explode", timeout_s=5.0))
    error = reply.payload["error"]
    assert error["code"] == code
    assert error["details"]["exception"] == type(raised).__name__
    assert str(raised) not in reply.payload["error"]["message"]


def test_a_coded_error_a_tool_raises_is_carried_with_its_details() -> None:
    tools = ToolRegistry()

    def refuse(arguments: Mapping[str, Any]) -> Any:
        raise BridgeError("PARM_NOT_FOUND", "no parameter named sizex", {"parm": "sizex"})

    tools.add("bridge.refuse", refuse)
    reply = dispatcher(tools).dispatch(call("bridge.refuse", timeout_s=5.0))
    error = reply.payload["error"]
    assert error["code"] == "PARM_NOT_FOUND"
    assert error["details"]["parm"] == "sizex"


@pytest.mark.parametrize(
    ("text", "hidden"),
    [
        ("could not read /Users/somebody/scenes/shot.hip", True),
        (r"could not read C:\Users\somebody\shot.hip", True),
        ('read "/var/log/houdini.log" once', True),
        ("no node at /obj/geo1/box2", False),
        # A node path that happens to hold a folder name is the caller's own
        # subject, and comes back exactly as it went in.
        ("no node at /obj/tmp/thing", False),
        ("no node at /usrdata/thing", False),
    ],
)
def test_a_reply_never_names_a_place_on_disk(text: str, hidden: bool) -> None:
    cleaned = errors.hide_paths(text)
    assert (errors.PATH_MARKER in cleaned) is hidden
    if not hidden:
        assert cleaned == text


def test_every_code_the_bridge_sends_is_in_the_table() -> None:
    for code in ("SESSION_BUSY", "TIMEOUT", "TOOL_FAILED", "UNKNOWN_TOOL", "BAD_ARGUMENTS"):
        assert code in errors.CODES
    assert errors.RESERVED <= set(errors.CODES)


# Section: what the reply can carry


def test_houdini_values_become_something_json_can_hold(scene: Scene) -> None:
    node = scene.node("/obj")
    parm = node.parm("tx")
    converted = encoding.convert(
        {
            "vector": Vector3(1, 2, 3),
            "matrix": Matrix4(1),
            "node": node,
            "parm": parm,
            "nested": [{"node": node}],
        }
    )
    assert converted.lossy is False
    assert converted.value == {
        "vector": [1.0, 2.0, 3.0],
        "matrix": [1.0] * 16,
        "node": "/obj",
        "parm": "/obj/tx",
        "nested": [{"node": "/obj"}],
    }


class Array:
    """An array as the encoder reads one: a shape and a `tolist`."""

    def __init__(self, values: list[Any]) -> None:
        self._values = values
        self.shape = (len(values),)

    def tolist(self) -> list[Any]:
        return list(self._values)


def test_a_long_array_is_cut_and_the_reply_says_so() -> None:
    converted = encoding.convert({"points": Array(list(range(10)))}, max_items=4)
    assert converted.value["points"] == [0, 1, 2, 3]
    assert converted.lossy is True
    assert converted.cut == ["data.points"]


def test_a_real_array_is_read_the_same_way() -> None:
    numpy = pytest.importorskip("numpy")
    converted = encoding.convert({"points": numpy.arange(3)})
    assert converted.value["points"] == [0, 1, 2]
    assert converted.lossy is False


def test_a_mapping_is_capped_in_breadth_and_in_total() -> None:
    wide = {str(key): key for key in range(100)}
    converted = encoding.convert(wide, max_keys=4)
    assert len(converted.value) == 4
    assert converted.lossy is True

    deep = {"rows": [{"value": index} for index in range(100)]}
    budgeted = encoding.convert(deep, max_values=10)
    assert budgeted.lossy is True
    assert len(budgeted.cut) >= 1


def test_a_mapping_that_raises_while_it_is_read_becomes_text() -> None:
    class Awkward(Mapping):
        def items(self):
            raise RuntimeError("gone")

        def __getitem__(self, key: Any) -> Any:
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 0

    converted = encoding.convert({"thing": Awkward()})
    assert converted.lossy is True
    assert isinstance(converted.value["thing"], str)


def test_bytes_come_back_as_text_and_are_capped() -> None:
    converted = encoding.convert({"image": b"abcdef"}, max_bytes=3)
    assert converted.value["image"]["encoding"] == "base64"
    assert converted.value["image"]["bytes"] == 6
    assert converted.lossy is True


def test_something_the_encoder_does_not_know_becomes_short_text() -> None:
    class Odd:
        def __repr__(self) -> str:
            return "x" * 500

    converted = encoding.convert({"odd": Odd()})
    assert len(converted.value["odd"]) == encoding.MAX_REPR
    assert converted.lossy is True


def test_a_reply_that_was_cut_is_marked(scene: Scene) -> None:
    tools = ToolRegistry()
    tools.add("bridge.big", lambda arguments: {"blob": b"x" * (encoding.MAX_BYTES + 1)})
    reply = dispatcher(tools).dispatch(call("bridge.big", timeout_s=5.0))
    assert reply.payload["ok"] is True
    assert reply.payload["lossy"] is True
    assert reply.payload["cut"] == ["data.blob"]


# Section: the two real tools


def test_scene_info_reads_what_is_open(scene: Scene) -> None:
    running = dispatcher(hou=scene.module())
    reply = running.dispatch(call("scene.info", timeout_s=5.0))
    data = reply.payload["data"]
    assert data["houdini_version"] == "22.0.368"
    assert data["frame"] == 1.0
    assert data["fps"] == 24.0
    assert data["nodes"]["/obj"] == 0
    assert data["kind"] == "hython"
    # A headless session cannot tell, and says so rather than always yes.
    assert data["unsaved"] is None


def test_scene_info_says_it_is_unsaved_where_that_is_worth_knowing(scene: Scene) -> None:
    running = dispatcher(kind="gui", hou=scene.module())
    reply = running.dispatch(call("scene.info", timeout_s=5.0))
    assert reply.payload["data"]["unsaved"] is True


def test_node_create_sets_the_parameters_it_was_given(scene: Scene) -> None:
    running = dispatcher(hou=scene.module())
    reply = running.dispatch(
        call(
            "node.create",
            arguments={"parent": "/obj", "type": "geo", "name": "barrel", "parms": {"tx": 3.0}},
            timeout_s=5.0,
        )
    )
    data = reply.payload["data"]
    assert data["path"] == "/obj/barrel"
    assert data["parms_set"] == ["tx"]
    assert scene.node("/obj/barrel").parm("tx").value == 3.0


def _until(ready: Any, timeout_s: float = 10.0) -> None:
    """Wait for something a background thread is about to do."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.01)
    raise AssertionError("waited too long")
