"""Who a session is, and what happens when its scene is replaced.

Against the stand in for Houdini, whose hip file events fire in the order a
real headless session fires them. What a real Houdini has to confirm is in the
integration tests.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from fake_hou import Scene
from nscr_houdini_mcp.bridge import identity as identity_module
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.handlers import ToolRegistry
from nscr_houdini_mcp.bridge.identity import Identity, alias_template, hip_stem


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


def identity(scene: Scene, **overrides: Any) -> Identity:
    settings: dict[str, Any] = {
        "session_id": "session-1",
        "kind": "gui",
        "alias": "example-1",
        "hip_path": scene.hipFile.path(),
        "hou": scene.module(),
    }
    settings.update(overrides)
    made = Identity(**settings)
    made.refresh()
    return made


# Section: names


def test_a_worker_is_named_by_number_and_a_scene_by_its_file() -> None:
    assert alias_template("hython", "/scenes/shot_010.hip") == "w{n}"
    assert alias_template("gui", "/scenes/shot_010.hip") == "shot_010-{n}"
    assert alias_template("gui", None) == "scene-{n}"


def test_a_file_name_an_alias_cannot_hold_is_made_into_one() -> None:
    assert hip_stem("/scenes/shot 010 (final).hip") == "shot-010-final"
    assert hip_stem(None) == ""


# Section: the scene counter


def test_clearing_the_scene_moves_the_epoch_once(scene: Scene) -> None:
    session = identity(scene)
    session.watch()
    assert session.scene_epoch == 0

    scene.hipFile.clear()

    assert session.scene_epoch == 1
    assert session.scene()["changed"] == "cleared"


def test_loading_a_scene_moves_the_epoch_once_although_it_clears_first(scene: Scene) -> None:
    """A load reports a clear inside it. One load is one epoch."""
    session = identity(scene)
    session.watch()

    scene.hipFile.load("/scenes/other.hip")

    assert session.scene_epoch == 1
    assert session.scene()["changed"] == "loaded"
    assert session.scene()["hip_path"] == "/scenes/other.hip"


def test_the_epoch_moves_again_for_every_replacement(scene: Scene) -> None:
    session = identity(scene)
    session.watch()
    scene.hipFile.load("/scenes/other.hip")
    scene.hipFile.clear()
    scene.hipFile.load("/scenes/other.hip")
    assert session.scene_epoch == 3


def test_a_load_that_fails_after_clearing_still_counts(scene: Scene) -> None:
    """The scene really is gone, so the count moves whatever happens next.

    A failed load reports no end of its own, so a session that waited for one
    would go on telling callers their paths were good in an empty scene.
    """
    session = identity(scene)
    session.watch()

    with pytest.raises(Exception, match="cannot read"):
        scene.hipFile.fail_load("/scenes/missing.hip")

    assert session.scene_epoch == 1
    assert session.scene()["nodes"]["/obj"] == 0

    # And the next clear is counted too: nothing was left half set.
    scene.hipFile.clear()
    assert session.scene_epoch == 2


def test_a_load_that_started_too_long_ago_no_longer_holds_a_clear(scene: Scene) -> None:
    session = identity(scene)
    session.watch()
    scene.hipFile._fire("BeforeLoad")
    session._load_began -= identity_module.LOADING_WINDOW_S + 1.0

    scene.hipFile.clear()
    scene.hipFile._fire("AfterLoad")

    # The clear counted, and the late load counted as the replacement it is.
    assert session.scene_epoch == 2


def test_a_merge_or_a_save_leaves_the_epoch_alone(scene: Scene) -> None:
    """A merge adds to the scene, so every path a caller holds still means
    what it meant."""
    session = identity(scene)
    session.watch()
    scene.hipFile.merge("/scenes/other.hip")
    scene.hipFile.save("/scenes/example.hip")
    assert session.scene_epoch == 0


def test_the_watch_can_be_taken_off_again(scene: Scene) -> None:
    session = identity(scene)
    session.watch()
    session.unwatch()
    scene.hipFile.clear()
    assert session.scene_epoch == 0


def test_a_replacement_is_passed_on_with_the_new_epoch_and_file(scene: Scene) -> None:
    seen: list[tuple[int, str | None]] = []
    session = identity(scene, on_change=lambda epoch, path: seen.append((epoch, path)))
    session.watch()
    scene.hipFile.load("/scenes/other.hip")
    # Once when the old scene went, once when the new one had settled.
    assert seen[-1] == (1, "/scenes/other.hip")
    assert [epoch for epoch, _ in seen] == [1, 1]


def test_the_summary_counts_the_top_level_networks(scene: Scene) -> None:
    session = identity(scene)
    scene.node("/obj").createNode("geo")
    session.refresh()
    summary = session.scene()
    assert summary["nodes"]["/obj"] == 1
    assert summary["nodes"]["/out"] == 0
    assert summary["hip_path"] == scene.hipFile.path()
    assert summary["session_id"] == "session-1"


def test_a_session_with_no_houdini_still_answers(scene: Scene) -> None:
    session = Identity(session_id="session-1")
    assert session.watch() is None
    assert session.scene()["nodes"] == {}
    assert session.scene_epoch == 0


# Section: the name going out of date


def test_a_scene_saved_under_another_name_warns_and_never_renames(scene: Scene) -> None:
    session = identity(scene, tracks_hip=True)
    session.watch()
    assert session.drift() is None

    scene.hipFile.load("/scenes/shot_020.hip")

    drift = session.drift()
    assert drift is not None
    assert drift["code"] == "ALIAS_DRIFT"
    assert drift["named_after"] == "example"
    assert drift["hip_stem"] == "shot_020"
    # The name itself does not move under a caller holding it.
    assert session.alias == "example-1"
    assert session.trace()["warnings"] == [drift]


def test_a_worker_is_never_said_to_have_drifted(scene: Scene) -> None:
    session = identity(scene, kind="hython", alias="w1", tracks_hip=False)
    session.watch()
    scene.hipFile.load("/scenes/shot_020.hip")
    assert session.drift() is None
    assert "warnings" not in session.trace()


# Section: what a call carrying an old epoch gets


def dispatcher(scene: Scene, session: Identity, tools: ToolRegistry) -> Dispatcher:
    return Dispatcher(
        tools,
        lock=threading.Lock(),
        kind="hython",
        session_id="session-1",
        identity=session,
        hou=scene.module(),
        wait_s=5.0,
        timeout_s=5.0,
    )


def counting_tools(ran: list[Mapping[str, Any]]) -> ToolRegistry:
    tools = ToolRegistry()
    tools.add(
        "scene.touch",
        lambda arguments: ran.append(dict(arguments)) or {"ran": True},
        mutating=True,
    )
    return tools


def test_a_call_carrying_an_old_epoch_is_refused_before_the_tool_runs(scene: Scene) -> None:
    ran: list[Mapping[str, Any]] = []
    session = identity(scene)
    session.watch()
    running = dispatcher(scene, session, counting_tools(ran))
    scene.hipFile.clear()

    reply = running.dispatch(Envelope(tool="scene.touch", scene_epoch=0))

    error = reply.payload["error"]
    assert error["code"] == "SCENE_REPLACED"
    assert error["details"]["carried_epoch"] == 0
    assert error["details"]["scene_epoch"] == 1
    assert reply.payload["scene"]["nodes"] == {"/obj": 0, "/out": 0, "/stage": 0, "/mat": 0}
    assert ran == []


def test_a_call_carrying_the_epoch_it_was_written_against_runs(scene: Scene) -> None:
    ran: list[Mapping[str, Any]] = []
    session = identity(scene)
    running = dispatcher(scene, session, counting_tools(ran))

    reply = running.dispatch(Envelope(tool="scene.touch", scene_epoch=0))

    assert reply.payload["ok"] is True
    assert len(ran) == 1


def test_a_call_carrying_no_epoch_at_all_runs(scene: Scene) -> None:
    ran: list[Mapping[str, Any]] = []
    session = identity(scene)
    session.watch()
    running = dispatcher(scene, session, counting_tools(ran))
    scene.hipFile.clear()

    reply = running.dispatch(Envelope(tool="scene.touch"))

    assert reply.payload["ok"] is True
    assert reply.payload["scene_epoch"] == 1
    assert len(ran) == 1


def test_a_scene_replaced_while_the_call_waited_is_caught_before_the_tool_runs(
    scene: Scene,
) -> None:
    """The scene can go while a call sits in the queue.

    The call was written against the scene that was open when it was sent, so
    its paths mean nothing in the one that replaced it. The last look happens
    on the thread that is about to touch the scene, not when the call arrived.
    """
    ran: list[Mapping[str, Any]] = []
    session = identity(scene)
    session.watch()
    tools = counting_tools(ran)
    holder = Holder()
    tools.add("scene.hold", holder, mutating=True)
    running = dispatcher(scene, session, tools)

    first = threading.Thread(target=lambda: running.dispatch(Envelope(tool="scene.hold")))
    first.start()
    assert holder.started.wait(10.0)

    answers: list[Any] = []
    waiting = threading.Thread(
        target=lambda: answers.append(
            running.dispatch(Envelope(tool="scene.touch", scene_epoch=0, wait_s=20.0))
        )
    )
    waiting.start()
    _until(lambda: running.state()["queued"] == 1)

    # The scene goes while that call is still waiting for its turn.
    scene.hipFile.load("/scenes/other.hip")
    holder.release.set()
    first.join(20.0)
    waiting.join(20.0)

    error = answers[0].payload["error"]
    assert error["code"] == "SCENE_REPLACED"
    assert error["details"]["carried_epoch"] == 0
    assert error["details"]["scene_epoch"] == 1
    assert answers[0].payload["scene"]["hip_path"] == "/scenes/other.hip"
    assert ran == []


class Holder:
    """A tool that holds the session until it is let go."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.started.set()
        assert self.release.wait(20.0)
        return {"held": True}


def _until(ready: Any, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.02)
    raise AssertionError("waited too long")


def test_every_reply_says_who_answered_and_which_scene_it_was(scene: Scene) -> None:
    session = identity(scene, tracks_hip=True)
    session.watch()
    running = dispatcher(scene, session, counting_tools([]))
    scene.hipFile.load("/scenes/shot_020.hip")

    payload = running.dispatch(Envelope(tool="scene.touch")).payload

    assert payload["session_id"] == "session-1"
    assert payload["alias"] == "example-1"
    assert payload["scene_epoch"] == 1
    assert payload["warnings"][0]["code"] == "ALIAS_DRIFT"
