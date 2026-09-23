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
import re
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any


def _as_hou(kind: type, name: str | None = None) -> type:
    """Make a class look as though it came from `hou`, which is how it is read.

    `name` is the class name a real Houdini 22 gives the same thing, where the
    stand in's own name is not it.
    """
    kind.__module__ = "hou"
    if name is not None:
        kind.__name__ = name
        kind.__qualname__ = name
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


class LoadWarning(Error):
    """What a load raises when it loaded, with things it could not resolve."""

    def instanceMessage(self) -> str:  # noqa: N802 - the name is Houdini's
        return str(self)


for _kind in (Error, OperationFailed, ObjectWasDeleted, InvalidInput, PermissionError, LoadWarning):
    _as_hou(_kind)


class Vector3:
    """A vector reads as a sequence, and has no `asTuple`, as in Houdini 22."""

    def __init__(self, *values: float) -> None:
        self._values = tuple(float(value) for value in values)

    def __iter__(self) -> Any:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]


class Matrix4:
    def __init__(self, fill: float = 0.0, values: tuple[float, ...] | None = None) -> None:
        if values is not None:
            self._values = tuple(float(value) for value in values)
        else:
            self._values = tuple(float(fill) for _ in range(16))

    @classmethod
    def translation(cls, x: float, y: float, z: float) -> Matrix4:
        """A row major transform that moves a point, as Houdini writes one."""
        values = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, x, y, z, 1.0]
        return cls(values=tuple(values))

    def asTuple(self) -> tuple[float, ...]:  # noqa: N802 - the name is Houdini's
        return self._values


_as_hou(Vector3)
_as_hou(Matrix4)


class TemplateType:
    """A parameter kind, named the way the build names it: Float, String, Ramp."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class ParmTemplate:
    """What a parameter is: its kind, label, default, tags and menu."""

    def __init__(
        self,
        kind: str,
        label: str = "",
        default: tuple[Any, ...] = (0.0,),
        *,
        tags: dict[str, str] | None = None,
        folder: str = "",
        menu: tuple[str, ...] = (),
    ) -> None:
        self._kind = TemplateType(kind)
        self._label = label
        self.default = default
        self._tags = dict(tags or {})
        self._folder = folder
        self._menu = menu

    def type(self) -> TemplateType:
        return self._kind

    def label(self) -> str:
        return self._label

    def tags(self) -> dict[str, str]:
        return dict(self._tags)

    def folderType(self) -> str:  # noqa: N802 - the name is Houdini's
        return f"folderType.{self._folder or 'Tabs'}"

    def menuItems(self) -> tuple[str, ...]:  # noqa: N802 - the name is Houdini's
        return self._menu

    def defaultValue(self) -> tuple[Any, ...]:  # noqa: N802 - the name is Houdini's
        return self.default


# What an expression may call in this stand in, and what reading a node from
# one does: `npoints` cooks the node it names, as it does in Houdini.
_EXPRESSION_CALL = re.compile(r'(npoints|ch)\("([^"]+)"\)')


class Parm:
    """One parameter component, with a value or an expression."""

    def __init__(
        self,
        node: Node,
        name: str,
        template: ParmTemplate,
        default: Any = None,
        *,
        instance_of: Parm | None = None,
        index: int = 0,
        spare: bool = False,
    ) -> None:
        self._node = node
        self._name = name
        self._template = template
        self.default = default
        self.value: Any = default
        self._expression: str | None = None
        self._language = "hscript"
        self.locked = False
        self._instance_of = instance_of
        self._index = index
        self._spare = spare
        self._tuple: ParmTuple | None = None
        # How many times an evaluation ran, so a check can see nothing did.
        self.evaluations = 0
        # A channel operator that drives this parameter, as an export does.
        # Its value then comes from that node, which a read of it cooks.
        self.override: Node | None = None
        # Keyframes as (frame, expression, language). With more than one, the
        # build will not hand back an expression, as Houdini will not.
        self.keys: list[tuple[float, str, str]] = []

    def name(self) -> str:
        return self._name

    def path(self) -> str:
        return f"{self._node.path()}/{self._name}"

    def node(self) -> Node:
        return self._node

    def tuple(self) -> ParmTuple | None:
        return self._tuple

    def parmTemplate(self) -> ParmTemplate:  # noqa: N802 - the name is Houdini's
        return self._template

    def set(self, value: Any) -> None:
        self.value = value
        self._node.dirty = True
        if self._template.type().name() == "Folder":
            self._node.grow(self)

    def setExpression(self, text: str, language: str = "hscript") -> None:  # noqa: N802
        self._expression = text
        self._language = language
        self._node.dirty = True

    def setKeyframes(self, keys: list[tuple[float, str, str]]) -> None:  # noqa: N802
        self.keys = list(keys)
        self._expression = None
        self._node.dirty = True

    def expression(self) -> str:
        if len(self.keys) > 1:
            raise OperationFailed("Parameter must have exactly one keyframe")
        if self._expression is None:
            raise OperationFailed("Parameter is not animated")
        return self._expression

    def expressionLanguage(self) -> str:  # noqa: N802 - the name is Houdini's
        self.expression()
        return "exprLanguage.Python" if self._language == "python" else "exprLanguage.Hscript"

    def isOverrideTrackActive(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self.override is not None

    def keyframes(self) -> list[Any]:
        # Reading keyframes evaluates them, as it does in Houdini.
        for _, text, language in self.keys:
            self._node.scene.run(self._node, text, language)
        return list(self.keys)

    def eval(self) -> Any:
        self.evaluations += 1
        if self.override is not None:
            self.override.cook()
            return 0.5
        if self.keys:
            _, text, language = self.keys[0]
            return self._node.scene.run(self._node, text, language)
        if self._expression is None:
            return self.value
        return self._node.scene.evaluate(self)

    def evalAsString(self) -> str:  # noqa: N802 - the name is Houdini's
        if self._expression is not None or self.keys or self.override is not None:
            return str(self.eval())
        self.evaluations += 1
        return self._node.scene.expand(str(self.value), self._node)

    def unexpandedString(self) -> str:  # noqa: N802 - the name is Houdini's
        if self._template.type().name() != "String":
            raise OperationFailed("Only string parms have unexpanded strings")
        return str(self.value)

    def isLocked(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self.locked

    def isSpare(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self._spare

    def isAtDefault(  # noqa: N802 - the name is Houdini's
        self, compare_temporary_defaults: bool = True, compare_expressions: bool = True
    ) -> bool:
        # An exported channel leaves the parameter at its default, as it does
        # in Houdini: the value the export writes is not the parameter's own.
        return self._expression is None and not self.keys and self.value == self.default

    def isMultiParmInstance(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self._instance_of is not None

    def parentMultiParm(self) -> Parm | None:  # noqa: N802 - the name is Houdini's
        return self._instance_of

    def multiParmInstanceIndices(self) -> tuple[int, ...]:  # noqa: N802 - the name is Houdini's
        return (self._index,) if self._instance_of is not None else ()


class ParmTuple:
    """A parameter as the pane shows it: one name, one or more components."""

    def __init__(self, name: str, parms: list[Parm]) -> None:
        self._name = name
        self._parms = parms
        for parm in parms:
            parm._tuple = self

    def name(self) -> str:
        return self._name

    def node(self) -> Node:
        return self._parms[0].node()

    def parmTemplate(self) -> ParmTemplate:  # noqa: N802 - the name is Houdini's
        return self._parms[0].parmTemplate()

    def __iter__(self) -> Any:
        return iter(self._parms)

    def __len__(self) -> int:
        return len(self._parms)

    def __getitem__(self, index: int) -> Parm:
        return self._parms[index]

    def set(self, values: Any) -> None:
        for parm, value in zip(self._parms, values, strict=False):
            parm.set(value)

    def isAtDefault(  # noqa: N802 - the name is Houdini's
        self, compare_temporary_defaults: bool = True, compare_expressions: bool = True
    ) -> bool:
        return all(parm.isAtDefault() for parm in self._parms)


# The parameters each stand in node type has, as (name, kind, components,
# default, extras). A type not named here has a translate and a scale.
_STANDARD = (("t", "Float", ("tx", "ty", "tz"), 0.0, {}), ("scale", "Float", ("scale",), 1.0, {}))
TYPE_PARMS: dict[str, tuple[tuple[str, str, tuple[str, ...], Any, dict[str, Any]], ...]] = {
    "box": (("size", "Float", ("sizex", "sizey", "sizez"), 1.0, {}), *_STANDARD),
    "xform": _STANDARD,
    "attribwrangle": (
        (
            "snippet",
            "String",
            ("snippet",),
            "",
            {"tags": {"editor": "1", "editorlang": "VEX"}},
        ),
        ("class", "Menu", ("class",), 2, {"menu": ("detail", "primitive", "point", "vertex")}),
        ("bindings", "Toggle", ("bindings",), 0, {}),
        ("go", "Button", ("go",), 0, {}),
    ),
    "attribcreate": (
        ("group", "String", ("group",), "", {}),
        ("numattr", "Folder", ("numattr",), 1, {"folder": "MultiparmBlock"}),
    ),
    "file": (("file", "String", ("file",), "default.bgeo", {}),),
    # The flipbook render node, with the parameter names a Houdini 22 build has.
    "flipbook": (
        ("camera", "String", ("camera",), "/obj/cam1", {}),
        ("picture", "String", ("picture",), "$HIP/render/$HIPNAME.$OS.$F4.exr", {}),
        ("mkpath", "Toggle", ("mkpath",), 1, {}),
        ("tres", "Toggle", ("tres",), 0, {}),
        ("res", "Int", ("res1", "res2"), 1280, {}),
        ("trange", "Menu", ("trange",), "off", {"menu": ("off", "normal", "on")}),
        ("f", "Float", ("f1", "f2", "f3"), 1.0, {}),
        ("sopsource", "Menu", ("sopsource",), "render", {"menu": ("display", "render")}),
        (
            "shadingmode",
            "Menu",
            ("shadingmode",),
            "smooth",
            {"menu": ("wire", "matcap", "smooth", "smoothwire")},
        ),
        ("vobjects", "String", ("vobjects",), "*", {}),
        ("forceobjects", "String", ("forceobjects",), "", {}),
    ),
    # A camera, with the names a Houdini 22 build has.
    "cam": (
        *_STANDARD[:1],
        ("r", "Float", ("rx", "ry", "rz"), 0.0, {}),
        ("resx", "Int", ("resx",), 1920, {}),
        ("resy", "Int", ("resy",), 1080, {}),
        ("focal", "Float", ("focal",), 50.0, {}),
        ("aperture", "Float", ("aperture",), 41.4214, {}),
        ("near", "Float", ("near",), 0.001, {}),
        ("far", "Float", ("far",), 10000.0, {}),
        (
            "projection",
            "Menu",
            ("projection",),
            "perspective",
            {"menu": ("perspective", "ortho")},
        ),
        ("orthowidth", "Float", ("orthowidth",), 2.0, {}),
        ("win", "Float", ("winx", "winy"), 0.0, {}),
        ("winsize", "Float", ("winsizex", "winsizey"), 1.0, {}),
    ),
}

# What a box draws, in its own object's space.
UNIT_BOX = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))

# What each instance of a multiparm holds, by the multiparm's name.
MULTIPARM_INSTANCE = {
    "numattr": (("name#", "String", ""), ("value#", "Float", 0.0)),
}

# Descriptions for the types whose default name comes from their description.
DESCRIPTIONS = {"xform": "Transform", "attribwrangle": "Attribute Wrangle"}


class NodeType:
    def __init__(self, name: str, category: str = "Sop") -> None:
        self._name = name
        self._category = category

    def name(self) -> str:
        return self._name

    def nameComponents(self) -> tuple[str, str, str, str]:  # noqa: N802 - the name is Houdini's
        return ("", "", self._name, "")

    def nameWithCategory(self) -> str:  # noqa: N802 - the name is Houdini's
        return f"{self._category}/{self._name}"

    def description(self) -> str:
        return DESCRIPTIONS.get(self._name, self._name.title())

    def category(self) -> Any:
        return SimpleNamespace(name=lambda: self._category)

    def defaultColor(self) -> Color:  # noqa: N802 - the name is Houdini's
        return Color(0.8, 0.8, 0.8)


class Color:
    def __init__(self, *rgb: float) -> None:
        self._rgb = tuple(rgb)

    def rgb(self) -> tuple[float, ...]:
        return self._rgb


class Connection:
    """One wire into a node."""

    def __init__(self, index: int, source: Node, output: int) -> None:
        self._index = index
        self._source = source
        self._output = output

    def inputIndex(self) -> int:  # noqa: N802 - the name is Houdini's
        return self._index

    def outputIndex(self) -> int:  # noqa: N802 - the name is Houdini's
        return self._output

    def inputNode(self) -> Node:  # noqa: N802 - the name is Houdini's
        return self._source

    def inputItem(self) -> Node:  # noqa: N802 - the name is Houdini's
        return self._source


class Item:
    """A network box or a sticky note."""

    def __init__(self, parent: Node, name: str, *, text: str = "", nodes: tuple = ()) -> None:
        self._parent = parent
        self._name = name
        self._text = text
        self._nodes = nodes

    def path(self) -> str:
        return f"{self._parent.path()}/{self._name}"

    def comment(self) -> str:
        return self._text

    def text(self) -> str:
        return self._text

    def nodes(self) -> tuple:
        return self._nodes

    def position(self) -> Vector3:
        return Vector3(0.0, 0.0)

    def size(self) -> Vector3:
        return Vector3(2.5, 2.5)


class Node:
    """One node, with the few readers and writers the tools use."""

    def __init__(self, scene: Scene, name: str, type_name: str, parent: Node | None) -> None:
        self.scene = scene
        self._scene = scene
        self._name = name
        self._type = NodeType(type_name, _category(parent))
        self._parent = parent
        self._children: list[Node] = []
        self._tuples: list[ParmTuple] = []
        self._instances: dict[str, list[list[ParmTuple]]] = {}
        for tuple_name, kind, parts, default, extra in TYPE_PARMS.get(type_name, _STANDARD):
            template = ParmTemplate(kind, tuple_name.title(), (default,) * len(parts), **extra)
            parms = [Parm(self, part, template, default) for part in parts]
            self._tuples.append(ParmTuple(tuple_name, parms))
            if kind == "Folder":
                self.grow(parms[0])
        # What the node reports about its cooks, and what its next cook says.
        self.cooks = 0
        self.dirty = True
        self.errors_now: list[str] = []
        self.warnings_now: list[str] = []
        self.fails_with: list[str] = []
        self.flags: set[str] = set()
        self.inputs_now: dict[int, tuple[Node, int]] = {}
        self.note = ""
        self.tint: tuple[float, ...] = (0.8, 0.8, 0.8)
        self.user: dict[str, Any] = {}
        self.boxes: list[Item] = []
        self.stickies: list[Item] = []
        self.locked_asset = False
        # Called after each cook, for a check that needs something to happen
        # in the middle of a read.
        self.after_cook: Any = None
        # The nodes that depend on this one, and whether this one exports
        # channels, which only a channel operator does.
        self.dependents_now: list[Node] = []
        self.export_flag = False
        # What a capture reads: whether an object is hidden, and the box a
        # geometry node draws when it is not a box.
        self.hidden = False
        self.bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def name(self) -> str:
        return self._name

    def type(self) -> NodeType:
        return self._type

    def path(self) -> str:
        if self._parent is None:
            return f"/{self._name}" if self._name else ""
        return f"{self._parent.path()}/{self._name}"

    def children(self) -> tuple[Node, ...]:
        self._scene.listed += 1
        return tuple(self._children)

    def allSubChildren(  # noqa: N802 - the name is Houdini's
        self, top_down: bool = True, recurse_in_locked_nodes: bool = True
    ) -> tuple[Node, ...]:
        found: list[Node] = []
        for child in self._children:
            found.append(child)
            if recurse_in_locked_nodes or not child.locked_asset:
                found.extend(child.allSubChildren(top_down, recurse_in_locked_nodes))
        return tuple(found)

    def grow(self, counter: Parm) -> None:
        """Make or drop the instances of a multiparm to match its count."""
        wanted = int(counter.value or 0)
        groups = self._instances.setdefault(counter.name(), [])
        while len(groups) < wanted:
            index = len(groups) + 1
            group = []
            for pattern, kind, default in MULTIPARM_INSTANCE.get(counter.name(), ()):
                name = pattern.replace("#", str(index))
                template = ParmTemplate(kind, name, (default,))
                parm = Parm(self, name, template, default, instance_of=counter, index=index)
                group.append(ParmTuple(name, [parm]))
            groups.append(group)
        del groups[wanted:]

    def parmTuples(self) -> tuple[ParmTuple, ...]:  # noqa: N802 - the name is Houdini's
        found: list[ParmTuple] = []
        for tuple_ in self._tuples:
            found.append(tuple_)
            for group in self._instances.get(tuple_.name(), ()):
                found.extend(group)
        return tuple(found)

    def parms(self) -> tuple[Parm, ...]:
        return tuple(parm for tuple_ in self.parmTuples() for parm in tuple_)

    def parm(self, name: str) -> Parm | None:
        if "/" in name:
            holder, _, leaf = name.rpartition("/")
            node = self.relative(holder)
            return None if node is None else node.parm(leaf)
        return next((parm for parm in self.parms() if parm.name() == name), None)

    def parmTuple(self, name: str) -> ParmTuple | None:  # noqa: N802 - the name is Houdini's
        return next((tuple_ for tuple_ in self.parmTuples() if tuple_.name() == name), None)

    def addSpareParm(self, name: str, kind: str = "Float", default: Any = 0.0) -> Parm:  # noqa: N802
        template = ParmTemplate(kind, name.title(), (default,))
        parm = Parm(self, name, template, default, spare=True)
        self._tuples.append(ParmTuple(name, [parm]))
        return parm

    def relative(self, path: str) -> Node | None:
        node: Node | None = self
        for part in path.split("/"):
            if node is None:
                return None
            if part in ("", "."):
                continue
            if part == "..":
                node = node._parent
                continue
            node = next((child for child in node._children if child.name() == part), None)
        return node

    # Section: cooking, as far as a read can see it

    def cookCount(self) -> int:  # noqa: N802 - the name is Houdini's
        return self.cooks

    def needsToCook(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self.dirty

    def cook(self, force: bool = False) -> None:
        for source, _ in self.inputs_now.values():
            source.cook()
        if self.dirty or force:
            self.cooks += 1
            self.dirty = False
            self.errors_now = list(self.fails_with)
            if self.after_cook is not None:
                self.after_cook()
        if self.errors_now:
            raise OperationFailed("the node has errors")

    def errors(self) -> tuple[str, ...]:
        return tuple(self.errors_now)

    def warnings(self) -> tuple[str, ...]:
        return tuple(self.warnings_now)

    def lastCookTime(self) -> float:  # noqa: N802 - the name is Houdini's
        return 1.5 if self.cooks else 0.0

    # Section: what the network editor shows

    def isGenericFlagSet(self, flag: str) -> bool:  # noqa: N802 - the name is Houdini's
        return flag in self.flags

    def isLockedHDA(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self.locked_asset

    def dependents(self, include_children: bool = True) -> tuple[Node, ...]:
        return tuple(self.dependents_now)

    def isExportFlagSet(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self.export_flag

    def setInput(self, index: int, source: Node | None, output: int = 0) -> None:  # noqa: N802
        if source is None:
            self.inputs_now.pop(index, None)
        else:
            self.inputs_now[index] = (source, output)
        self.dirty = True

    def inputConnections(self) -> tuple[Connection, ...]:  # noqa: N802 - the name is Houdini's
        return tuple(
            Connection(index, source, output)
            for index, (source, output) in sorted(self.inputs_now.items())
        )

    def inputLabels(self) -> tuple[str, ...]:  # noqa: N802 - the name is Houdini's
        return ("First Input", "Second Input")

    def outputs(self) -> tuple[Node, ...]:
        return tuple(
            node
            for node in self._scene.everything()
            if any(source is self for source, _ in node.inputs_now.values())
        )

    def comment(self) -> str:
        return self.note

    def color(self) -> Color:
        return Color(*self.tint)

    def position(self) -> Vector3:
        return Vector3(1.0, -2.0)

    def userDataDict(self) -> dict[str, Any]:  # noqa: N802 - the name is Houdini's
        return dict(self.user)

    def networkBoxes(self) -> tuple[Item, ...]:  # noqa: N802 - the name is Houdini's
        return tuple(self.boxes)

    def stickyNotes(self) -> tuple[Item, ...]:  # noqa: N802 - the name is Houdini's
        return tuple(self.stickies)

    def childTypeCategory(self) -> Any:  # noqa: N802 - the name is Houdini's
        return SimpleNamespace(nodeTypes=lambda: dict.fromkeys(self._scene.types, None))

    # Section: what a capture reads and changes

    def parent(self) -> Node | None:
        return self._parent

    def destroy(self) -> None:
        parent = self._parent
        if parent is None or self not in parent._children:
            raise ObjectWasDeleted("the node is gone")
        index = parent._children.index(self)
        parent._children.remove(self)
        self._scene.undos.record(lambda: parent._children.insert(index, self))

    def setDisplayFlag(self, on: bool) -> None:  # noqa: N802 - the name is Houdini's
        """The display flag: one geometry node in an object carries it at a time."""
        if self._parent is None:
            return
        before = [node for node in self._parent._children if "Display" in node.flags]
        if on:
            for sibling in self._parent._children:
                sibling.flags.discard("Display")
            self.flags.add("Display")
        else:
            self.flags.discard("Display")

        def undo() -> None:
            for sibling in self._parent._children:
                sibling.flags.discard("Display")
            for node in before:
                node.flags.add("Display")

        self._scene.undos.record(undo)

    def isDisplayFlagSet(self) -> bool:  # noqa: N802 - the name is Houdini's
        return "Display" in self.flags

    def displayNode(self) -> Node | None:  # noqa: N802 - the name is Houdini's
        return next((child for child in self._children if "Display" in child.flags), None)

    def isObjectDisplayed(self) -> bool:  # noqa: N802 - the name is Houdini's
        return not self.hidden

    def worldTransform(self) -> Matrix4:  # noqa: N802 - the name is Houdini's
        t = self.parmTuple("t")
        return Matrix4.translation(*(float(parm.eval()) for parm in t)) if t else Matrix4(0.0)

    def geometry(self) -> Geometry:
        bounds = self.bounds
        if bounds is None and self._type.name() == "box":
            bounds = UNIT_BOX
        return Geometry(bounds)

    def render(self, frame_range: Any = None, **rest: Any) -> None:
        """What a flipbook render node does: one picture per frame, at the picture's path."""
        if self._type.name() != "flipbook":
            raise OperationFailed("only a render node renders")
        self._scene.capture.render_rop(self, frame_range)

    def layer(self) -> ImageLayer:
        return self._scene.capture.layer_of(self, None)

    def layerAtFrame(self, frame: float) -> ImageLayer:  # noqa: N802 - the name is Houdini's
        return self._scene.capture.layer_of(self, frame)

    def saveImage(self, path: str, frame_range: Any = ()) -> None:  # noqa: N802
        self._scene.capture.save_cop2(self, path, frame_range)

    def xRes(self) -> int:  # noqa: N802 - the name is Houdini's
        return self._scene.capture.cop_size[0]

    def yRes(self) -> int:  # noqa: N802 - the name is Houdini's
        return self._scene.capture.cop_size[1]

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


def _category(parent: Node | None) -> str:
    """Which network a node lives in, which is what decides its category."""
    if parent is None:
        return "Manager"
    if parent._parent is None:
        return "Manager"
    kind = parent._type.name()
    if kind == "copnet":
        return "Cop"
    if kind in ("img", "cop2net"):
        return "Cop2"
    return {"/obj": "Object", "/out": "Driver", "/stage": "Lop"}.get(parent.path(), "Sop")


_as_hou(Parm)
_as_hou(ParmTuple)
# The names a real session gives them: a node of any context is a subclass
# whose name ends in `Node`, and its type one whose name ends in `NodeType`.
_as_hou(NodeType, "OpNodeType")
_as_hou(Node, "OpNode")


class Undos:
    """An undo stack that collapses a group into one entry, as Houdini does.

    As in Houdini, the labels come newest first, and an undo asked for inside
    an open group is refused.
    """

    def __init__(self) -> None:
        self.labels: list[tuple[str, list[Any]]] = []
        self._pending: list[Any] | None = None
        self.performed = 0
        self.disabled = 0
        # How many entries the stack keeps, the oldest going first past it,
        # as Houdini's undo levels do. Nothing means no limit.
        self.limit: int | None = None

    def _keep(self, entry: tuple[str, list[Any]]) -> None:
        self.labels.append(entry)
        if self.limit is not None:
            del self.labels[: max(0, len(self.labels) - self.limit)]

    @contextmanager
    def group(self, label: str):
        outer = self._pending
        self._pending = []
        try:
            yield
        finally:
            done, self._pending = self._pending, outer
            if done:
                self._keep((label, done))

    @contextmanager
    def disabler(self):
        """Nothing done inside is kept, as with undos turned off in Houdini."""
        self.disabled += 1
        try:
            yield
        finally:
            self.disabled -= 1

    def areEnabled(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self.disabled == 0

    def record(self, undo: Any) -> None:
        if self.disabled:
            return
        if self._pending is None:
            self._keep(("edit", [undo]))
        else:
            self._pending.append(undo)

    def undoLabels(self) -> list[str]:  # noqa: N802 - the name is Houdini's
        return [label for label, _ in reversed(self.labels)]

    def performUndo(self) -> None:  # noqa: N802 - the name is Houdini's
        if self._pending is not None:
            raise OperationFailed("Cannot undo within an undo group")
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

    # The pane tabs a desktop shows, for the captures that need a user
    # interface. A scene with no desktop has none.
    desktop: Desktop | None = None
    # The main window's device pixel ratio, as the screen it is on has it.
    ratio = 1.0

    def mainQtWindow(self) -> Any:  # noqa: N802 - the name is Houdini's
        return SimpleNamespace(devicePixelRatioF=lambda: self.ratio)

    def paneTabs(self) -> tuple[Any, ...]:  # noqa: N802 - the name is Houdini's
        return tuple(self.desktop.tabs) if self.desktop is not None else ()

    def findPaneTab(self, name: str) -> Any:  # noqa: N802 - the name is Houdini's
        return next((tab for tab in self.paneTabs() if tab.name() == name), None)

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
        # What the scene says about itself. A headless session says it has
        # unsaved changes whatever it has, so that is the default here.
        self.unsaved = True
        self.new = False
        # The text a load warns with, when a test wants one.
        self.load_warning: str | None = None
        self.saved: list[str] = []
        # Whether a save writes a file, for the checks that look at the disk.
        self.writes_files = False
        # What a save raises, when a test wants it to fail part way.
        self.save_error: BaseException | None = None
        self.recent: list[bool] = []

    def path(self) -> str:
        return self._path

    def basename(self) -> str:
        return self._path.replace("\\", "/").rsplit("/", 1)[-1]

    def isNewFile(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self.new

    def hasUnsavedChanges(self) -> bool:  # noqa: N802 - the name is Houdini's
        if self.unsaved is None:
            raise OperationFailed("this session will not say")
        return self.unsaved

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

    def load(
        self, path: str, suppress_save_prompt: bool = False, ignore_load_warnings: bool = False
    ) -> None:
        self._fire(HipFileEventType.BeforeLoad, HipFileEventType.BeforeClear)
        self._scene.empty()
        self._path = str(path)
        self.new = False
        self._fire(HipFileEventType.AfterClear, HipFileEventType.AfterLoad)
        if self.load_warning and not ignore_load_warnings:
            raise LoadWarning(self.load_warning)

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

    def save(self, path: str | None = None, save_to_recent_files: bool = True) -> None:
        if self.save_error is not None:
            if path and self.writes_files:
                with open(path, "wb") as partial:
                    partial.write(b"half a sce")
            raise self.save_error
        self.recent.append(save_to_recent_files)
        if path:
            self._path = str(path)
        self.new = False
        self.saved.append(self._path)
        if self.writes_files:
            with open(self._path, "wb") as written:
                written.write(b"a scene")
        self._fire(HipFileEventType.BeforeSave, HipFileEventType.AfterSave)

    def setName(self, path: str) -> None:  # noqa: N802 - the name is Houdini's
        # A bare untitled name makes the scene untitled again, as it does in
        # Houdini; anything else names a file.
        self.new = str(path) == "untitled.hip"
        self._path = str(path)


class Scene:
    """One fake session: a scene, an undo stack and a main thread."""

    def __init__(self, *, types: tuple[str, ...] = ("geo", "null", "cam")) -> None:
        self.types = types
        self.undos = Undos()
        self.ui = MainThread()
        # How many times any node was asked for its children.
        self.listed = 0
        self.root = Node(self, "", "root", None)
        self._counts: dict[str, int] = {}
        # What a person has selected in the interface.
        self.selected: list[Node] = []
        self.empty()
        self.undos.labels.clear()
        self.hipFile = HipFile(self, "/Users/somebody/scenes/example.hip")
        self.capture = CaptureStandIn(self)

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

    def parm(self, path: str) -> Parm | None:
        holder, _, name = str(path).rpartition("/")
        node = self.node(holder) if holder else None
        return None if node is None else node.parm(name)

    def parm_tuple(self, path: str) -> ParmTuple | None:
        holder, _, name = str(path).rpartition("/")
        node = self.node(holder) if holder else None
        return None if node is None else node.parmTuple(name)

    def everything(self) -> tuple[Node, ...]:
        return self.root.allSubChildren()

    def evaluate(self, parm: Parm) -> Any:
        """Evaluate an expression, cooking what it reads the way Houdini does."""
        return self.run(parm.node(), str(parm._expression), parm._language)

    def run(self, node: Node, text: str, language: str = "hscript") -> Any:
        if language == "python":
            # Any node a Python expression names is read, and so cooked.
            for named in re.findall(r"hou\.node\('([^']+)'\)", text):
                found = node.relative(named)
                if found is not None:
                    found.cook()
            return 0.0

        def call(match: re.Match[str]) -> str:
            kind, target = match.groups()
            if kind == "npoints":
                found = node.relative(target)
                if found is None:
                    raise OperationFailed("no such node")
                found.cook()
                return "8"
            referenced = node.parm(target)
            if referenced is None:
                raise OperationFailed("no such parameter")
            return repr(float(referenced.eval()))

        body = _EXPRESSION_CALL.sub(call, text).replace("$F", repr(self.frame()))
        return eval(body, {"__builtins__": {}}, {})  # noqa: S307 - arithmetic the test wrote

    def expand(self, text: str, node: Node) -> str:
        """A string as Houdini expands it: variables, and backticks evaluated."""
        text = re.sub(r"`([^`]*)`", lambda match: str(self.run(node, match.group(1))), text)
        folder = self.hipFile.path().replace("\\", "/").rsplit("/", 1)[0]
        text = text.replace("$HIP", folder)
        # A variable that only has a value inside a cook reads as nothing.
        return re.sub(r"\$\{?[A-Za-z_]\w*\}?", "", text)

    def frame(self) -> float:
        return 72.0 if threading.current_thread().name == "fake-main" else 1.0

    def module(self) -> Any:
        """The scene as something that answers like the `hou` module."""
        return SimpleNamespace(
            node=self.node,
            parm=self.parm,
            parmTuple=self.parm_tuple,
            selectedNodes=lambda: tuple(self.selected),
            nodeFlag=SimpleNamespace(
                Display="Display",
                Render="Render",
                Bypass="Bypass",
                Template="Template",
                Lock="Lock",
                SoftLock="SoftLock",
            ),
            undos=self.undos,
            ui=self.ui,
            hipFile=self.hipFile,
            hipFileEventType=HipFileEventType,
            playbar=SimpleNamespace(frameRange=lambda: (1.0, 240.0)),
            applicationVersionString=lambda: "22.0.368",
            # The frame a thread that is not the main thread reads is not the
            # frame the session is on, which is why ambient state is only ever
            # read on the main thread.
            frame=self.frame,
            fps=lambda: 24.0,
            isUIAvailable=lambda: True,
            ropNodeTypeCategory=lambda: "Driver",
            nodeType=lambda category, name: name if name in self.types else None,
            nodeTypeCategories=lambda: dict.fromkeys(("Object", "Sop", "Driver", "Cop", "Cop2")),
            paneTabType=PaneTabType,
            geometryViewportType=SimpleNamespace(
                Perspective="Perspective", Top="Top", Front="Front", Right="Right"
            ),
            glShadingType=SimpleNamespace(
                Smooth="Smooth", Wire="Wire", SmoothWire="SmoothWire", MatCap="MatCap"
            ),
            displaySetType=SimpleNamespace(SceneObject="SceneObject", DisplayModel="DisplayModel"),
            imageLayerStorageType=SimpleNamespace(Fixed8="Fixed8", Float32="Float32"),
            ImageLayer=ImageLayer,
            BoundingBox=BoundingBox,
            hmath=SimpleNamespace(buildRotate=build_rotate),
            Vector3=Vector3,
            Matrix4=Matrix4,
            OperationFailed=OperationFailed,
            ObjectWasDeleted=ObjectWasDeleted,
            InvalidInput=InvalidInput,
            PermissionError=PermissionError,
            LoadWarning=LoadWarning,
        )


# Section: what a capture needs


class BoundingBox:
    def __init__(self, *values: float) -> None:
        values = values or (0.0,) * 6
        self._low = tuple(float(value) for value in values[:3])
        self._high = tuple(float(value) for value in values[3:6])
        self.valid = bool(values)

    def minvec(self) -> Vector3:
        return Vector3(*self._low)

    def maxvec(self) -> Vector3:
        return Vector3(*self._high)

    def isValid(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self.valid


class Geometry:
    def __init__(self, bounds: Any) -> None:
        self._bounds = bounds

    def boundingBox(self) -> BoundingBox:  # noqa: N802 - the name is Houdini's
        if self._bounds is None:
            box = BoundingBox()
            box.valid = False
            return box
        low, high = self._bounds
        return BoundingBox(*low, *high)


class Matrix3:
    """A rotation, kept as the angles it was built from so a check can read them."""

    def __init__(self, euler: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.euler = tuple(float(value) for value in euler)

    def asTuple(self) -> tuple[float, ...]:  # noqa: N802 - the name is Houdini's
        return self.euler


class Rotation:
    def __init__(self, euler: Any) -> None:
        self.euler = tuple(float(value) for value in euler)

    def extractRotationMatrix3(self) -> Matrix3:  # noqa: N802 - the name is Houdini's
        return Matrix3(self.euler)


def build_rotate(values: Any, *rest: Any) -> Rotation:
    return Rotation(values if not rest else (values, *rest[:2]))


class PaneTabType:
    SceneViewer = "SceneViewer"
    NetworkEditor = "NetworkEditor"
    Parm = "Parm"


class ViewCamera:
    """What a viewport's own camera holds: where it is, how it turns, its pivot and width."""

    def __init__(
        self,
        translation: tuple[float, ...] = (0.0, 0.0, 10.0),
        rotation: Matrix3 | None = None,
        pivot: tuple[float, ...] = (0.0, 0.0, 0.0),
        ortho_width: float = 4.0,
    ) -> None:
        self._translation = tuple(translation)
        self._rotation = rotation or Matrix3()
        self._pivot = tuple(pivot)
        self._ortho_width = float(ortho_width)

    def translation(self) -> tuple[float, ...]:
        return self._translation

    def setTranslation(self, value: Any) -> None:  # noqa: N802 - the name is Houdini's
        self._translation = tuple(value)

    def rotation(self) -> Matrix3:
        return self._rotation

    def setRotation(self, value: Matrix3) -> None:  # noqa: N802 - the name is Houdini's
        self._rotation = value

    def pivot(self) -> tuple[float, ...]:
        return self._pivot

    def setPivot(self, value: Any) -> None:  # noqa: N802 - the name is Houdini's
        self._pivot = tuple(value)

    def orthoWidth(self) -> float:  # noqa: N802 - the name is Houdini's
        return self._ortho_width

    def setOrthoWidth(self, value: float) -> None:  # noqa: N802 - the name is Houdini's
        self._ortho_width = float(value)

    def stash(self) -> ViewCamera:
        return ViewCamera(self._translation, self._rotation, self._pivot, self._ortho_width)

    def state(self) -> tuple[Any, ...]:
        return (self._translation, self._rotation.euler, self._pivot, self._ortho_width)


class DisplaySet:
    def __init__(self) -> None:
        self._mode = "Smooth"

    def shadedMode(self) -> str:  # noqa: N802 - the name is Houdini's
        return self._mode

    def setShadedMode(self, mode: str) -> None:  # noqa: N802 - the name is Houdini's
        self._mode = mode


class ViewportSettings:
    def __init__(self) -> None:
        self.sets = {"SceneObject": DisplaySet(), "DisplayModel": DisplaySet()}

    def displaySet(self, kind: str) -> DisplaySet:  # noqa: N802 - the name is Houdini's
        return self.sets[kind]


class Viewport:
    """One viewport: its type, the camera it looks through and its own camera."""

    def __init__(self) -> None:
        self._type = "Perspective"
        self._camera: Node | None = None
        self._default = ViewCamera()
        self._settings = ViewportSettings()
        self.framed: list[Any] = []

    def type(self) -> str:
        return self._type

    def changeType(self, kind: str) -> None:  # noqa: N802 - the name is Houdini's
        # Houdini keeps a camera per view type; changing type moves the view.
        self._type = kind
        self._default = ViewCamera((0.0, 20.0, 0.0), Matrix3((-90.0, 0.0, 0.0)), (1, 2, 3), 9.0)

    def camera(self) -> Node | None:
        return self._camera

    def setCamera(self, node: Node) -> None:  # noqa: N802 - the name is Houdini's
        self._camera = node

    def useDefaultCamera(self) -> None:  # noqa: N802 - the name is Houdini's
        self._camera = None

    def defaultCamera(self) -> ViewCamera:  # noqa: N802 - the name is Houdini's
        return self._default.stash()

    def setDefaultCamera(self, view: ViewCamera) -> None:  # noqa: N802 - the name is Houdini's
        self._default = view.stash()

    def settings(self) -> ViewportSettings:
        return self._settings

    def frameAll(self) -> None:  # noqa: N802 - the name is Houdini's
        self.framed.append("all")
        self._default.setTranslation((5.0, 5.0, 5.0))
        self._default.setPivot((0.5, 0.5, 0.5))

    def frameSelected(self) -> None:  # noqa: N802 - the name is Houdini's
        self.framed.append("selection")
        self._default.setTranslation((6.0, 6.0, 6.0))

    def frameBoundingBox(self, box: BoundingBox) -> None:  # noqa: N802 - the name is Houdini's
        self.framed.append((tuple(box.minvec()), tuple(box.maxvec())))
        self._default.setTranslation((7.0, 7.0, 7.0))

    def viewTransform(self) -> Matrix4:  # noqa: N802 - the name is Houdini's
        translation = self._default.translation()
        return Matrix4.translation(*translation)

    def shading(self) -> dict[str, str]:
        return {name: shown.shadedMode() for name, shown in self._settings.sets.items()}

    def tearOffCopy(self) -> None:  # noqa: N802 - the name is Houdini's
        raise AssertionError("a torn off viewport draws nothing and must never be made")

    def createFloatingViewport(self) -> None:  # noqa: N802 - the name is Houdini's
        raise AssertionError("a floating viewport draws nothing and must never be made")


class FlipbookSettings:
    """Settings a flipbook takes, each a getter with no argument and a setter with one.

    The defaults are what an artist might have left in the flipbook dialog,
    so a check can see that a capture sets each one rather than carrying it.
    """

    ARTIST = {
        "outputToMPlay": True,
        "output": "",
        "frameRange": (100.0, 200.0),
        "frameIncrement": 2.0,
        "useResolution": False,
        "resolution": (640, 480),
        "beautyPassOnly": False,
        "visibleObjects": "geo1",
        "visibleTypes": "GeoOnly",
        "useSheetSize": True,
        "useMotionBlur": True,
        "useDepthOfField": True,
        "leaveFrameAtEnd": True,
        "appendFramesToCurrent": True,
        "backgroundImage": "plate.jpg",
        "overrideGamma": True,
        "overrideLUT": True,
        "initializeSimulations": True,
        "renderAllViewports": True,
        "scopeChannelKeyframesOnly": True,
        "audioFilename": "take.wav",
        "outputZoom": 50,
        "cropOutMaskOverlay": False,
        "antialias": "HighQuality",
        "setUseFrameTimeLimit": True,
        "setUseFrameProgressLimit": True,
    }
    FIELDS = tuple(ARTIST)

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = dict(values or FlipbookSettings.ARTIST)

    def stash(self) -> FlipbookSettings:
        return FlipbookSettings(self.values)

    def __getattr__(self, name: str) -> Any:
        if name not in FlipbookSettings.FIELDS:
            raise AttributeError(name)

        def field(*value: Any) -> Any:
            if value:
                self.values[name] = value[0]
                return None
            return self.values[name]

        return field


class Pane:
    def __init__(self) -> None:
        self.tabs: list[Any] = []
        self.current: Any = None

    def currentTab(self) -> Any:  # noqa: N802 - the name is Houdini's
        return self.current

    def add(self, tab: Any) -> Any:
        self.tabs.append(tab)
        tab._pane = self
        if self.current is None:
            self.current = tab
        return tab


class Tab:
    """One pane tab: its kind, its name, and whether it is the one its pane shows."""

    def __init__(self, scene: Scene, kind: str, name: str) -> None:
        self._scene = scene
        self._kind = kind
        self._name = name
        self._pane: Pane | None = None
        self.window: Window | None = None
        self.geometry = Rect(0, 0, 0, 0)

    def type(self) -> str:
        return self._kind

    def name(self) -> str:
        return self._name

    def pane(self) -> Pane | None:
        return self._pane

    def isCurrentTab(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self._pane is not None and self._pane.current is self

    def setIsCurrentTab(self) -> None:  # noqa: N802 - the name is Houdini's
        if self._pane is not None:
            self._pane.current = self

    def qtParentWindow(self) -> Window | None:  # noqa: N802 - the name is Houdini's
        return self.window

    def qtScreenGeometry(self) -> Rect:  # noqa: N802 - the name is Houdini's
        return self.geometry


class SceneViewerTab(Tab):
    def __init__(self, scene: Scene, name: str = "panetab1") -> None:
        super().__init__(scene, PaneTabType.SceneViewer, name)
        self.viewport = Viewport()
        self.settings = FlipbookSettings()

    def curViewport(self) -> Viewport:  # noqa: N802 - the name is Houdini's
        return self.viewport

    def flipbookSettings(self) -> FlipbookSettings:  # noqa: N802 - the name is Houdini's
        return self.settings

    def flipbook(
        self, viewport: Viewport | None = None, settings: Any = None, open_dialog: bool = False
    ) -> None:
        self._scene.capture.flipbook(self, viewport or self.viewport, settings or self.settings)

    def createFloatingViewport(self) -> None:  # noqa: N802 - the name is Houdini's
        raise AssertionError("a floating viewport draws nothing and must never be made")


class NetworkEditorTab(Tab):
    def __init__(self, scene: Scene, name: str = "panetab2") -> None:
        super().__init__(scene, PaneTabType.NetworkEditor, name)
        self._pwd = scene.node("/obj")

    def pwd(self) -> Node | None:
        return self._pwd

    def setPwd(self, node: Node) -> None:  # noqa: N802 - the name is Houdini's
        self._pwd = node


class Desktop:
    def __init__(self) -> None:
        self.tabs: list[Any] = []


class Point:
    def __init__(self, x: float, y: float) -> None:
        self._x, self._y = x, y

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y


class Rect:
    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self._x, self._y, self._w, self._h = x, y, width, height

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y

    def width(self) -> float:
        return self._w

    def height(self) -> float:
        return self._h

    def topLeft(self) -> Point:  # noqa: N802 - the name is Qt's
        return Point(self._x, self._y)


class Pixmap:
    """A grab: device pixels in a Pillow image, and the ratio they were taken at."""

    def __init__(self, image: Any, ratio: float) -> None:
        self.image = image
        self.ratio = ratio

    def width(self) -> int:
        return self.image.size[0]

    def height(self) -> int:
        return self.image.size[1]

    def devicePixelRatio(self) -> float:  # noqa: N802 - the name is Qt's
        return self.ratio

    def copy(self, x: int, y: int, width: int, height: int) -> Pixmap:
        return Pixmap(self.image.crop((x, y, x + width, y + height)), self.ratio)

    def save(self, path: str, kind: str = "PNG") -> bool:
        self.image.save(path, format=kind)
        return True


class Window:
    """A window on screen: where it is in points, and what its panes look like."""

    def __init__(self, x: float, y: float, width: int, height: int, ratio: float = 1.0) -> None:
        self.origin = (x, y)
        self.size = (width, height)
        self.ratio = ratio
        # Screen rectangles painted a colour of their own, for the crop checks.
        self.painted: list[tuple[Rect, tuple[int, int, int, int]]] = []
        self.grabs = 0

    def grab(self) -> Pixmap:
        from PIL import Image, ImageDraw

        self.grabs += 1
        width, height = (round(value * self.ratio) for value in self.size)
        image = Image.new("RGBA", (width, height), (0, 0, 255, 255))
        draw = ImageDraw.Draw(image)
        for rect, colour in self.painted:
            left = round((rect.x() - self.origin[0]) * self.ratio)
            top = round((rect.y() - self.origin[1]) * self.ratio)
            right = round((rect.x() - self.origin[0] + rect.width()) * self.ratio) - 1
            bottom = round((rect.y() - self.origin[1] + rect.height()) * self.ratio) - 1
            draw.rectangle((left, top, right, bottom), fill=colour)
        return Pixmap(image, self.ratio)

    def mapToGlobal(self, point: Point) -> Point:  # noqa: N802 - the name is Qt's
        return Point(point.x() + self.origin[0], point.y() + self.origin[1])

    def rect(self) -> Rect:
        return Rect(0, 0, *self.size)

    def children(self) -> None:
        raise AssertionError("a capture never walks the widget tree")


class ImageLayer:
    """A Copernicus image: its size and its pixels, bottom row first."""

    def __init__(self, width: int, height: int, rgba_top_down: bytes) -> None:
        self._size = (width, height)
        stride = width * 4
        rows = [rgba_top_down[index * stride : (index + 1) * stride] for index in range(height)]
        self._bottom_up = b"".join(reversed(rows))

    def bufferResolution(self) -> tuple[int, int]:  # noqa: N802 - the name is Houdini's
        return self._size

    def allBufferElements(self, storage: str, channels: int) -> bytes:  # noqa: N802
        if storage != "Fixed8" or channels != 4:
            raise OperationFailed("the stand in only reads 8 bit RGBA")
        return self._bottom_up


class CaptureStandIn:
    """What renders, flipbooks and image reads do in the stand in, and what each saw.

    A picture is drawn with Pillow: a grey square on a clear background when
    something is shown, and nothing but the clear background when not, so an
    empty capture is one the checks can make on purpose.
    """

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        # Every render and flipbook, with what the scene looked like at the time.
        self.seen: list[dict[str, Any]] = []
        # Set to make a render or a flipbook finish and write nothing.
        self.writes = True
        # Set to make every picture empty, or one flat colour.
        self.blank = False
        self.flat: tuple[int, int, int, int] | None = None
        # Set to make a flipbook raise the way a failed one does.
        self.flipbook_error: BaseException | None = None
        self.cop_size = (64, 32)
        self.cop_frames: list[float | None] = []
        # How long each rendered frame takes, for the checks on a sequence run as a job.
        self.delay_s = 0.0
        # How much larger than asked a render node draws, and the renders
        # that worked it out, kept apart from the ones a check looks at.
        self.backing = 1.0
        # A frame at which a render raises after writing it, as a failed cook does.
        self.fail_at_frame: float | None = None
        self.probes: list[dict[str, Any]] = []

    # Section: pictures

    def draw(
        self, path: str, size: tuple[int, int], shown: bool, camera: Node | None = None
    ) -> None:
        """The middle half of the frame covered, as a render node's picture would be.

        With `backing` above one, the render node's drawing is modelled as
        it is on a dense display: the frame drawn that many times larger and
        only the bottom left corner kept, unless the camera's window makes up
        for it.
        """
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", size, self.flat or (0, 0, 0, 0))
        if shown and not self.blank and not self.flat:
            width, height = size
            window = (0.0, 0.0, 1.0, 1.0)
            if camera is not None and camera.parm("winx") is not None:
                window = tuple(
                    float(camera.parm(name).eval())
                    for name in ("winx", "winy", "winsizex", "winsizey")
                )
            scale = self.backing if camera is not None else 1.0

            def placed(value: float, offset: float, span: float) -> float:
                return ((value - 0.5 - offset) / span + 0.5) * scale

            left = placed(0.25, window[0], window[2]) * width
            right = placed(0.75, window[0], window[2]) * width
            low = placed(0.25, window[1], window[3])
            high = placed(0.75, window[1], window[3])
            top, bottom = (1.0 - high) * height, (1.0 - low) * height
            ImageDraw.Draw(image).rectangle(
                (round(left), round(top), round(right) - 1, round(bottom) - 1),
                fill=(128, 128, 128, 255),
            )
        image.save(path, format="PNG")

    def shown_objects(self, vobjects: str = "*", forced: str = "") -> list[Node]:
        root = self.scene.node("/obj")
        objects = [] if root is None else list(root._children)
        wanted = set(vobjects.split())
        picked = [
            node for node in objects if ("*" in wanted and not node.hidden) or node.path() in wanted
        ]
        picked += [node for node in objects if node.path() in forced.split()]
        return [node for node in picked if node._type.name() != "cam"]

    def anything_shown(self, objects: list[Node]) -> bool:
        for node in objects:
            shown = node.displayNode()
            if shown is not None and shown.geometry().boundingBox().isValid():
                return True
        return False

    # Section: the flipbook render node

    def render_rop(self, rop: Node, frame_range: Any) -> None:
        camera_path = str(rop.parm("camera").eval())
        camera = self.scene.node(camera_path) if camera_path else None
        if camera is None:
            raise OperationFailed("No camera specified for render.")
        start, end = float(frame_range[0]), float(frame_range[1])
        step = float(frame_range[2]) if len(frame_range) > 2 else 1.0
        size = (
            (int(rop.parm("res1").eval()), int(rop.parm("res2").eval()))
            if rop.parm("tres").eval()
            else (1280, 720)
        )
        objects = self.shown_objects(
            str(rop.parm("vobjects").eval()), str(rop.parm("forceobjects").eval())
        )
        frame = start
        while frame <= end:
            if self.delay_s:
                time.sleep(self.delay_s)
            picture = str(rop.parm("picture").eval()).replace("$F4", f"{int(frame):04d}")
            kept = self.probes if picture.endswith(".probe.png") else self.seen
            kept.append(
                {
                    "route": "rop",
                    "frame": frame,
                    "picture": picture,
                    "size": size,
                    "camera": camera_path,
                    "t": tuple(parm.eval() for parm in camera.parmTuple("t")),
                    "r": tuple(parm.eval() for parm in camera.parmTuple("r")),
                    "projection": camera.parm("projection").eval()
                    if camera.parm("projection")
                    else None,
                    "orthowidth": camera.parm("orthowidth").eval()
                    if camera.parm("orthowidth")
                    else None,
                    "vobjects": rop.parm("vobjects").eval(),
                    "forceobjects": rop.parm("forceobjects").eval(),
                    "sopsource": rop.parm("sopsource").eval(),
                    "shadingmode": rop.parm("shadingmode").eval(),
                    "trange": rop.parm("trange").eval(),
                    "displayed": {
                        node.path(): (node.displayNode().path() if node.displayNode() else None)
                        for node in objects
                    },
                    "undo_enabled": self.scene.undos.disabled == 0,
                    "window": tuple(parm.eval() for parm in camera.parmTuple("win"))
                    + tuple(parm.eval() for parm in camera.parmTuple("winsize")),
                    "follows": camera.inputs_now[0][0].path() if 0 in camera.inputs_now else None,
                    "focal": camera.parm("focal").eval(),
                }
            )
            if self.writes:
                self.draw(picture, size, self.anything_shown(objects), camera)
            if self.fail_at_frame is not None and frame >= self.fail_at_frame:
                raise OperationFailed("the render stopped with an error")
            frame += step

    # Section: the viewport flipbook

    def flipbook(self, tab: SceneViewerTab, viewport: Viewport, settings: FlipbookSettings) -> None:
        if self.flipbook_error is not None:
            raise self.flipbook_error
        values = dict(settings.values)
        start, end = values["frameRange"]
        step = float(values["frameIncrement"] or 1.0)
        size = tuple(values["resolution"]) if values["useResolution"] else (640, 480)
        self.seen.append(
            {
                "route": "viewer",
                "tab": tab.name(),
                "current": tab.isCurrentTab(),
                "settings": values,
                "camera": viewport.camera().path() if viewport.camera() else None,
                "type": viewport.type(),
                "view": viewport._default.state(),
                "shading": viewport.shading(),
            }
        )
        if not self.writes:
            return
        shown = self.anything_shown(self.shown_objects())
        frame = float(start)
        while frame <= float(end):
            path = str(values["output"]).replace("$F4", f"{int(frame):04d}")
            self.draw(path, size, shown)
            frame += step

    # Section: COP images

    def layer_of(self, node: Node, frame: float | None) -> ImageLayer:
        self.cop_frames.append(frame)
        width, height = self.cop_size
        top = bytes((255, 255, 255, 255)) * (width * (height // 2))
        bottom = bytes((0, 0, 0, 255)) * (width * (height - height // 2))
        return ImageLayer(width, height, top + bottom)

    def save_cop2(self, node: Node, path: str, frame_range: Any) -> None:
        self.cop_frames.append(frame_range[0] if frame_range else None)
        if self.writes:
            self.draw(path, self.cop_size, True)
