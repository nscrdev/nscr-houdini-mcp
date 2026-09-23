"""The bridge's own mark of unsaved changes, for a session that cannot say.

A headless Houdini says it has unsaved changes whatever it has, so the bridge
keeps a mark from what it sees: its own saves and loads, the calls that change
the scene, and the scene's own events. `scene.info` reports it with its
source, and a session with a user interface reports Houdini's answer instead.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from fake_hou import Scene
from nscr_houdini_mcp.bridge import dirty
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.handlers import default_registry
from nscr_houdini_mcp.bridge.identity import Identity
from nscr_houdini_mcp.bridge.tools import ToolContext, scene_info


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


def session(scene: Scene, *, kind: str = "hython") -> Dispatcher:
    module = scene.module()
    identity = Identity(session_id="s-1", kind=kind, alias="w1", hou=module)
    identity.watch()
    return Dispatcher(
        default_registry(),
        lock=threading.Lock(),
        kind=kind,
        session_id="s-1",
        identity=identity,
        hou=module,
    )


def run(dispatcher: Dispatcher, tool: str, **arguments: Any) -> dict[str, Any]:
    reply = dispatcher.dispatch(Envelope(tool=tool, arguments=arguments))
    assert reply.payload["ok"] is True, reply.payload
    return reply.payload


def unsaved(dispatcher: Dispatcher) -> tuple[Any, Any]:
    data = run(dispatcher, "scene.info")["data"]
    return data["unsaved"], data["unsaved_source"]


# Section: the mark on its own


def test_the_mark_starts_unknown_and_follows_what_it_sees() -> None:
    mark = dirty.DirtyMarker()
    assert (mark.state_name, mark.unsaved) == (dirty.UNKNOWN, None)
    mark.began("node.create")
    mark.ended("node.create", ok=True)
    assert (mark.state_name, mark.unsaved) == (dirty.DIRTY, True)
    mark.began("scene.save")
    mark.event(dirty.SAVED)
    mark.ended("scene.save", ok=True)
    assert (mark.state_name, mark.unsaved) == (dirty.CLEAN, False)
    mark.event(dirty.MERGED)
    assert mark.state_name == dirty.DIRTY
    mark.event(dirty.SAVED)
    assert mark.state_name == dirty.CLEAN
    mark.event(dirty.CLEARED)
    assert mark.state_name == dirty.CLEAN
    mark.event(dirty.LOADED)
    assert mark.state()["why"] == "loaded"


def test_code_that_saved_on_its_own_leaves_the_mark_unknown() -> None:
    mark = dirty.DirtyMarker()
    mark.began("python.run")
    mark.event(dirty.SAVED)
    mark.ended("python.run", ok=True)
    assert mark.state_name == dirty.UNKNOWN
    assert "saved" in mark.state()["why"]


def test_a_failed_change_or_load_is_unknown_and_a_failed_save_changes_nothing() -> None:
    mark = dirty.DirtyMarker()
    mark.began("node.create")
    mark.ended("node.create", ok=False)
    assert mark.state_name == dirty.UNKNOWN
    mark.began("scene.open")
    mark.ended("scene.open", ok=True)
    assert mark.state_name == dirty.CLEAN
    mark.began("scene.save")
    mark.ended("scene.save", ok=False)
    assert mark.state_name == dirty.CLEAN
    mark.began("scene.open")
    mark.ended("scene.open", ok=False)
    assert mark.state_name == dirty.UNKNOWN


# Section: through a session


def test_a_headless_session_reports_the_bridges_mark(scene: Scene) -> None:
    worker = session(scene)
    assert unsaved(worker) == (None, "bridge")
    run(worker, "python.run", code="hou.node('/obj').createNode('geo')", namespace="n")
    assert unsaved(worker) == (True, "bridge")
    run(worker, "scene.save")
    assert unsaved(worker) == (False, "bridge")
    # Reads leave it alone.
    run(worker, "node.inspect", mode="tree", path="/obj")
    assert unsaved(worker) == (False, "bridge")
    run(worker, "python.run", code="hou.hipFile.save()", namespace="n")
    assert unsaved(worker) == (None, "bridge")
    run(worker, "node.create", parent="/obj", type="null")
    assert unsaved(worker) == (True, "bridge")
    # A save made from outside any call, which a scene event reports.
    scene.hipFile.save()
    assert unsaved(worker) == (False, "bridge")
    scene.hipFile.merge("/elsewhere/other.hip")
    assert unsaved(worker) == (True, "bridge")


def test_a_session_with_a_user_interface_reports_houdinis_own_answer(scene: Scene) -> None:
    mark = dirty.DirtyMarker()
    context = ToolContext(hou=scene.module(), kind="gui", dirty=mark)
    scene.hipFile.unsaved = False
    info = scene_info({}, context)
    assert (info["unsaved"], info["unsaved_source"]) == (False, "houdini")
    scene.hipFile.unsaved = True
    assert scene_info({}, context)["unsaved"] is True
    # Whatever the bridge's own mark says.
    assert mark.unsaved is None


def test_code_that_only_reads_leaves_the_mark_as_it_was(scene: Scene) -> None:
    worker = session(scene)
    run(worker, "scene.save")
    assert unsaved(worker) == (False, "bridge")
    run(worker, "python.run", code="result = len(hou.node('/obj').children())", namespace="n")
    assert unsaved(worker) == (False, "bridge")
    run(worker, "python.run", code="hou.node('/obj').createNode('geo')", namespace="n")
    assert unsaved(worker) == (True, "bridge")


def test_a_change_is_seen_when_the_undo_stack_is_at_its_limit(scene: Scene) -> None:
    worker = session(scene)
    scene.undos.limit = 3
    for index in range(3):
        code = "hou.node('/obj').createNode('null')"
        run(worker, "python.run", code=code, namespace="n", undo_label=f"step {index}")
    run(worker, "scene.save")
    assert unsaved(worker) == (False, "bridge")
    code = "hou.node('/obj').createNode('geo')"
    reply = run(worker, "python.run", code=code, namespace="n", undo_label="step 3")
    # The stack kept its length, and the change was still seen as one.
    assert len(scene.undos.labels) == 3
    assert reply["undo"]["recorded"] is True
    assert unsaved(worker) == (True, "bridge")
