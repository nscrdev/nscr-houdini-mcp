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


class MainThread:
    """A main thread that runs only what is posted to it.

    `run_until` is the test's event loop. Nothing posted here runs unless a
    test drains it, which is what makes the marshal visible: work that reaches
    the main thread has a thread name to prove it.
    """

    def __init__(self) -> None:
        self.posted: queue.Queue[Any] = queue.Queue()
        self.ran_on: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def postEventCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        self.posted.put(callback)

    def removeEventCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        return None

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
                callback = self.posted.get(timeout=0.02)
            except queue.Empty:
                continue
            self.ran_on.append(threading.current_thread().name)
            callback()


class Scene:
    """One fake session: a scene, an undo stack and a main thread."""

    def __init__(self, *, types: tuple[str, ...] = ("geo", "null", "cam")) -> None:
        self.types = types
        self.undos = Undos()
        self.ui = MainThread()
        self.root = Node(self, "", "root", None)
        self._counts: dict[str, int] = {}
        for name in ("obj", "out", "mat", "stage"):
            self.root._children.append(Node(self, name, "network", self.root))
        self.undos.labels.clear()

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
            hipFile=SimpleNamespace(
                path=lambda: "/Users/somebody/scenes/example.hip",
                hasUnsavedChanges=lambda: True,
            ),
            playbar=SimpleNamespace(frameRange=lambda: (1.0, 240.0)),
            applicationVersionString=lambda: "22.0.368",
            frame=lambda: 1.0,
            fps=lambda: 24.0,
            isUIAvailable=lambda: True,
            Vector3=Vector3,
            Matrix4=Matrix4,
            OperationFailed=OperationFailed,
            ObjectWasDeleted=ObjectWasDeleted,
            InvalidInput=InvalidInput,
            PermissionError=PermissionError,
        )
