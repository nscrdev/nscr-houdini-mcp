"""A stand in for `hou`, small enough to read and honest about what it models.

The build machines have no Houdini, so the rules around a tool call are tested
against this: a scene of nodes, an undo stack that collapses a group into one
entry, a main thread that only runs what is posted to it, and the four
exception classes the bridge maps. It is not a model of Houdini. It is the
handful of behaviours the dispatch layer depends on, each one checked against
a real headless session before it was written here.
"""

from __future__ import annotations

import queue
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any


def _as_hou(kind: type) -> type:
    """Make a class look as though it came from `hou`, which is how it is read."""
    kind.__module__ = "hou"
    return kind


class Error(Exception):
    """The base of the Houdini exceptions."""


class OperationFailed(Error):
    pass


class ObjectWasDeleted(Error):
    pass


class InvalidInput(Error):
    pass


class PermissionError(Error):  # noqa: A001 - the name is Houdini's
    pass


for _kind in (Error, OperationFailed, ObjectWasDeleted, InvalidInput, PermissionError):
    _as_hou(_kind)


class Vector3:
    def __init__(self, *values: float) -> None:
        self._values = tuple(float(value) for value in values)

    def asTuple(self) -> tuple[float, ...]:  # noqa: N802 - the name is Houdini's
        return self._values


class Matrix4:
    def __init__(self, fill: float = 0.0) -> None:
        self._values = tuple(float(fill) for _ in range(16))

    def asTuple(self) -> tuple[float, ...]:  # noqa: N802 - the name is Houdini's
        return self._values


_as_hou(Vector3)
_as_hou(Matrix4)


class Parm:
    def __init__(self, node: Node, name: str) -> None:
        self._node = node
        self._name = name
        self.value: Any = None

    def name(self) -> str:
        return self._name

    def path(self) -> str:
        return f"{self._node.path()}/{self._name}"

    def set(self, value: Any) -> None:
        self.value = value


class NodeType:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class Node:
    """One node, with the few readers and writers the tools use."""

    PARMS = ("tx", "ty", "tz", "scale")

    def __init__(self, scene: Scene, name: str, type_name: str, parent: Node | None) -> None:
        self._scene = scene
        self._name = name
        self._type = NodeType(type_name)
        self._parent = parent
        self._children: list[Node] = []
        self._parms = {name: Parm(self, name) for name in self.PARMS}

    def name(self) -> str:
        return self._name

    def type(self) -> NodeType:
        return self._type

    def path(self) -> str:
        if self._parent is None:
            return f"/{self._name}" if self._name else ""
        return f"{self._parent.path()}/{self._name}"

    def children(self) -> tuple[Node, ...]:
        return tuple(self._children)

    def parms(self) -> tuple[Parm, ...]:
        return tuple(self._parms.values())

    def parm(self, name: str) -> Parm | None:
        return self._parms.get(name)

    def parmTuple(self, name: str) -> Parm | None:  # noqa: N802 - the name is Houdini's
        return None

    def childTypeCategory(self) -> Any:  # noqa: N802 - the name is Houdini's
        return SimpleNamespace(nodeTypes=lambda: dict.fromkeys(self._scene.types, None))

    def createNode(self, type_name: str, name: str | None = None) -> Node:  # noqa: N802
        if type_name not in self._scene.types:
            raise OperationFailed(f"invalid node type {type_name}")
        chosen = name or self._scene.next_name(type_name)
        if any(child.name() == chosen for child in self._children):
            raise OperationFailed("a node of that name is already there")
        node = Node(self._scene, chosen, type_name, self)
        self._children.append(node)
        self._scene.undos.record(lambda: self._children.remove(node))
        return node


_as_hou(Parm)
_as_hou(NodeType)
_as_hou(Node)


class Undos:
    """An undo stack that collapses a group into one entry, as Houdini does."""

    def __init__(self) -> None:
        self.labels: list[tuple[str, list[Any]]] = []
        self._pending: list[Any] | None = None
        self.performed = 0

    @contextmanager
    def group(self, label: str):
        outer = self._pending
        self._pending = []
        try:
            yield
        finally:
            done, self._pending = self._pending, outer
            if done:
                self.labels.append((label, done))

    def record(self, undo: Any) -> None:
        if self._pending is None:
            self.labels.append(("edit", [undo]))
        else:
            self._pending.append(undo)

    def undoLabels(self) -> list[str]:  # noqa: N802 - the name is Houdini's
        return [label for label, _ in self.labels]

    def performUndo(self) -> None:  # noqa: N802 - the name is Houdini's
        if not self.labels:
            raise OperationFailed("nothing to undo")
        _, actions = self.labels.pop()
        for undo in reversed(actions):
            undo()
        self.performed += 1


# The longest a held main thread stays held when nothing lets it go.
HOLD_CAP_S = 30.0


class MainThread:
    """A main thread that runs only what is posted to it.

    `start` runs the test's event loop. Nothing posted here runs unless that
    loop or a test drains it, which is what makes the marshal visible: work
    that reaches the main thread has a thread name to prove it.

    Two things here are modelled on the real build. The object model lock is
    held by this thread for the whole of every callback it runs, and posting
    takes that lock, so a post during a cook blocks the thread that posts
    exactly as it does in Houdini. And a posted callback fires once and there
    is no call to take it off again, so `removeEventCallback` is absent here
    the way it is absent there.
    """

    def __init__(self) -> None:
        self.posted: queue.Queue[Any] = queue.Queue()
        self.ran_on: list[str] = []
        # Recursive, because Houdini's object model lock is: the thread that
        # holds it can call back into `hou`, which is what lets a callback
        # take itself off while the main thread is running it.
        self.hom_lock = threading.RLock()
        # Set to make the loop skip its ticks, which is what playback does to
        # the event loop callback while posted callbacks still land.
        self.starve_loop = False
        self.ticks = 0
        self._loop_callbacks: list[Any] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def postEventCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        with self.hom_lock:
            self.posted.put(callback)

    def addEventLoopCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        with self.hom_lock:
            self._loop_callbacks.append(callback)

    def removeEventLoopCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        with self.hom_lock:
            if callback in self._loop_callbacks:
                self._loop_callbacks.remove(callback)

    def eventLoopCallbacks(self) -> tuple[Any, ...]:  # noqa: N802 - the name is Houdini's
        return tuple(self._loop_callbacks)

    def cook(self, seconds: float) -> None:
        """Hold the main thread, and the lock with it, for that long.

        Returns at once: the cook is posted, the way a cook is kicked off from
        somewhere else and then owns the main thread until it ends.
        """
        self.postEventCallback(lambda: time.sleep(seconds))

    def hold(self) -> tuple[threading.Event, threading.Event]:
        """Hold the main thread until it is let go, and say when it started.

        The same thing `cook` does, without a duration to outrun: a test that
        has work to do while the main thread is busy waits for the first event
        before it starts and sets the second when it is done, so a slow
        machine cannot let the hold end underneath it.
        """
        begun = threading.Event()
        release = threading.Event()

        def held() -> None:
            begun.set()
            # Capped, so a test that never lets go still ends.
            release.wait(HOLD_CAP_S)

        self.postEventCallback(held)
        return begun, release

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="fake-main", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                callback = self.posted.get(timeout=0.01)
            except queue.Empty:
                callback = None
            if callback is not None:
                with self.hom_lock:
                    self.ran_on.append(threading.current_thread().name)
                    callback()
            if self.starve_loop:
                continue
            with self.hom_lock:
                self.ticks += 1
                for loop_callback in list(self._loop_callbacks):
                    loop_callback()


class HipFileEventType:
    """The event names this Houdini build reports, as read from it."""

    BeforeClear = "BeforeClear"
    AfterClear = "AfterClear"
    BeforeLoad = "BeforeLoad"
    AfterLoad = "AfterLoad"
    BeforeMerge = "BeforeMerge"
    AfterMerge = "AfterMerge"
    BeforeSave = "BeforeSave"
    AfterSave = "AfterSave"
    BeforeQuit = "BeforeQuit"


class HipFile:
    """The scene file, and the events it reports in the order it reports them.

    The orders here were read off a real headless session: a load reports a
    clear of its own inside it, which is why one load is one scene epoch and
    not two.
    """

    def __init__(self, scene: Scene, path: str) -> None:
        self._scene = scene
        self._path = path
        self._callbacks: list[Any] = []

    def path(self) -> str:
        return self._path

    def hasUnsavedChanges(self) -> bool:  # noqa: N802 - the name is Houdini's
        return True

    def addEventCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        self._callbacks.append(callback)

    def removeEventCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _fire(self, *events: str) -> None:
        for event in events:
            for callback in list(self._callbacks):
                callback(event)

    def clear(self, suppress_save_prompt: bool = False) -> None:
        self._scene.empty()
        self._fire(HipFileEventType.BeforeClear, HipFileEventType.AfterClear)

    def load(self, path: str, suppress_save_prompt: bool = False) -> None:
        self._fire(HipFileEventType.BeforeLoad, HipFileEventType.BeforeClear)
        self._scene.empty()
        self._path = str(path)
        self._fire(HipFileEventType.AfterClear, HipFileEventType.AfterLoad)

    def fail_load(self, path: str) -> None:
        """A load that clears the old scene and then gives up.

        Houdini reports nothing to say the load is over, so the session is
        left with an empty scene and no `AfterLoad` ever arrives.
        """
        self._fire(HipFileEventType.BeforeLoad, HipFileEventType.BeforeClear)
        self._scene.empty()
        self._fire(HipFileEventType.AfterClear)
        raise OperationFailed(f"cannot read {path}")

    def merge(self, path: str) -> None:
        self._fire(HipFileEventType.BeforeMerge, HipFileEventType.AfterMerge)

    def save(self, path: str | None = None) -> None:
        if path:
            self._path = str(path)
        self._fire(HipFileEventType.BeforeSave, HipFileEventType.AfterSave)


class Scene:
    """One fake session: a scene, an undo stack and a main thread."""

    def __init__(self, *, types: tuple[str, ...] = ("geo", "null", "cam")) -> None:
        self.types = types
        self.undos = Undos()
        self.ui = MainThread()
        self.root = Node(self, "", "root", None)
        self._counts: dict[str, int] = {}
        self.empty()
        self.undos.labels.clear()
        self.hipFile = HipFile(self, "/Users/somebody/scenes/example.hip")

    def empty(self) -> None:
        """Throw the scene away and put the empty networks back."""
        self.root._children.clear()
        for name in ("obj", "out", "mat", "stage"):
            self.root._children.append(Node(self, name, "network", self.root))

    def next_name(self, type_name: str) -> str:
        self._counts[type_name] = self._counts.get(type_name, 0) + 1
        return f"{type_name}{self._counts[type_name]}"

    def node(self, path: str) -> Node | None:
        found = self.root
        for part in [part for part in str(path).split("/") if part]:
            children = {child.name(): child for child in found.children()}
            if part not in children:
                return None
            found = children[part]
        return found

    def module(self) -> Any:
        """The scene as something that answers like the `hou` module."""
        return SimpleNamespace(
            node=self.node,
            undos=self.undos,
            ui=self.ui,
            hipFile=self.hipFile,
            hipFileEventType=HipFileEventType,
            playbar=SimpleNamespace(frameRange=lambda: (1.0, 240.0)),
            applicationVersionString=lambda: "22.0.368",
            # The frame a thread that is not the main thread reads is not the
            # frame the session is on, which is why ambient state is only ever
            # read on the main thread.
            frame=lambda: 72.0 if threading.current_thread().name == "fake-main" else 1.0,
            fps=lambda: 24.0,
            isUIAvailable=lambda: True,
            Vector3=Vector3,
            Matrix4=Matrix4,
            OperationFailed=OperationFailed,
            ObjectWasDeleted=ObjectWasDeleted,
            InvalidInput=InvalidInput,
            PermissionError=PermissionError,
        )
