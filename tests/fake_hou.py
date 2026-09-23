"""A stand in for `hou`, small enough to read and honest about what it models.

The build machines have no Houdini, so the rules around a tool call are tested
against this: a scene of nodes, an undo stack that collapses a group into one
entry, a main thread that only runs what is posted to it, and the four
exception classes the bridge maps. It is not a model of Houdini. It is the
handful of behaviours the dispatch layer depends on, each one checked against
a real headless session before it was written here.
"""

from __future__ import annotations

import os
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
    def __init__(self, fill: float = 0.0) -> None:
        self._values = tuple(float(fill) for _ in range(16))

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


class HoudiniCrashed(BaseException):  # noqa: N818 - it stands for a crash, not an error
    """A call that brings a real Houdini down. It is not an `Exception`, so no
    quiet read can swallow it and a check that makes one fails loudly."""


class ParmTemplate:
    """What a parameter is: its kind, label, default, tags and menu.

    The keywords after `menu` are for a type's own template group, read with
    no node at all: a name, how many components, a menu a script fills in, a
    range, whether it is hidden, and the templates inside a folder.
    """

    def __init__(
        self,
        kind: str,
        label: str = "",
        default: Any = (0.0,),
        *,
        tags: dict[str, str] | None = None,
        folder: str = "",
        menu: tuple[str, ...] = (),
        name: str = "",
        size: int = 1,
        labels: tuple[str, ...] = (),
        script: str = "",
        span: tuple[float, float, bool, bool] | None = None,
        hidden: bool = False,
        children: tuple[ParmTemplate, ...] = (),
        expression: tuple[str, ...] | str | None = None,
        ramp: str = "",
        menu_type: str = "Normal",
    ) -> None:
        self._kind = TemplateType(kind)
        self._label = label
        self.default = default
        self._tags = dict(tags or {})
        self._folder = folder
        self._menu = menu
        self._name = name
        self._size = size
        self._labels = labels
        self._script = script
        self._span = span
        self._hidden = hidden
        self.children = list(children)
        self._expression = expression
        self._ramp = ramp
        self._menu_type = menu_type

    def type(self) -> TemplateType:
        return self._kind

    def name(self) -> str:
        return self._name

    def label(self) -> str:
        return self._label

    def tags(self) -> dict[str, str]:
        return dict(self._tags)

    def folderType(self) -> str:  # noqa: N802 - the name is Houdini's
        return f"folderType.{self._folder or 'Tabs'}"

    def menuItems(self) -> tuple[str, ...]:  # noqa: N802 - the name is Houdini's
        return self._menu

    def menuLabels(self) -> tuple[str, ...]:  # noqa: N802 - the name is Houdini's
        return self._labels or self._menu

    def itemGeneratorScript(self) -> str:  # noqa: N802 - the name is Houdini's
        return self._script

    def defaultValue(self) -> Any:  # noqa: N802 - the name is Houdini's
        if self._kind.name() in ("Separator", "Label", "Button"):
            raise AttributeError("this kind of template has no default")
        return self.default

    def defaultValueAsString(self) -> str:  # noqa: N802 - the name is Houdini's
        # On a menu whose items toggle, Houdini 22 reads the mask of items
        # that are on as an index and the process dies. Nothing may call it.
        raise HoudiniCrashed("defaultValueAsString on a menu template")

    def menuType(self) -> str:  # noqa: N802 - the name is Houdini's
        return f"menuType.{self._menu_type}"

    def defaultExpression(self) -> Any:  # noqa: N802 - the name is Houdini's
        if self._expression is None:
            raise AttributeError("this kind of template has no default expression")
        return self._expression

    def numComponents(self) -> int:  # noqa: N802 - the name is Houdini's
        return self._size

    def minValue(self) -> float:  # noqa: N802 - the name is Houdini's
        return self._range()[0]

    def maxValue(self) -> float:  # noqa: N802 - the name is Houdini's
        return self._range()[1]

    def minIsStrict(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self._range()[2]

    def maxIsStrict(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self._range()[3]

    def _range(self) -> tuple[float, float, bool, bool]:
        if self._span is None:
            raise AttributeError("only a number has a range")
        return self._span

    def isHidden(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self._hidden

    def parmTemplates(self) -> tuple[ParmTemplate, ...]:  # noqa: N802 - the name is Houdini's
        if self._kind.name() != "Folder":
            raise AttributeError("only a folder holds templates")
        return tuple(self.children)

    def parmType(self) -> str:  # noqa: N802 - the name is Houdini's
        if self._kind.name() != "Ramp":
            raise AttributeError("only a ramp has a ramp type")
        return f"rampParmType.{self._ramp}"


class ParmTemplateGroup:
    """A type's parameter templates, top level first, as the pane lays them out."""

    def __init__(self, entries: list[ParmTemplate]) -> None:
        self._entries = entries

    def entries(self) -> tuple[ParmTemplate, ...]:
        return tuple(self._entries)


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
}

# What each instance of a multiparm holds, by the multiparm's name.
MULTIPARM_INSTANCE = {
    "numattr": (("name#", "String", ""), ("value#", "Float", 0.0)),
}

# Descriptions for the types whose default name comes from their description.
DESCRIPTIONS = {"xform": "Transform", "attribwrangle": "Attribute Wrangle"}


class NodeType:
    """A node type: what a node of it is called and, for a type in the
    library, everything a type says about itself without a node."""

    def __init__(
        self,
        name: str,
        category: str = "Sop",
        *,
        label: str | None = None,
        inputs: tuple[int, int] = (1, 1),
        outputs: int = 1,
        unordered: bool = False,
        templates: list[ParmTemplate] | None = None,
        dialog: str = "",
        help_text: str = "",
        library: str | None = None,
        hidden: bool = False,
        deprecated: dict[str, Any] | None = None,
        order: tuple[str, ...] = (),
    ) -> None:
        self._name = name
        self._category = category
        self._label = label
        self._inputs = inputs
        self._outputs = outputs
        self._unordered = unordered
        self.templates = list(templates or [])
        self._dialog = dialog
        self.help_text = help_text
        self._library = library
        self._hidden = hidden
        self._deprecated = deprecated
        self._order = order or (name,)
        # How many times the template group was asked for.
        self.read = 0

    def name(self) -> str:
        return self._name

    def nameComponents(self) -> tuple[str, str, str, str]:  # noqa: N802 - the name is Houdini's
        # A version is the last part when it is a number; the base name is the
        # part before it, and everything in front is the namespace.
        parts = self._name.split("::")
        version = parts.pop() if len(parts) > 1 and parts[-1].replace(".", "").isdigit() else ""
        return ("", "::".join(parts[:-1]), parts[-1], version)

    def nameWithCategory(self) -> str:  # noqa: N802 - the name is Houdini's
        return f"{self._category}/{self._name}"

    def description(self) -> str:
        if self._label is not None:
            return self._label
        return DESCRIPTIONS.get(self._name, self._name.title())

    def category(self) -> Any:
        return SimpleNamespace(name=lambda: self._category)

    def defaultColor(self) -> Color:  # noqa: N802 - the name is Houdini's
        return Color(0.8, 0.8, 0.8)

    def minNumInputs(self) -> int:  # noqa: N802 - the name is Houdini's
        return self._inputs[0]

    def maxNumInputs(self) -> int:  # noqa: N802 - the name is Houdini's
        return self._inputs[1]

    def maxNumOutputs(self) -> int:  # noqa: N802 - the name is Houdini's
        return self._outputs

    def hasUnorderedInputs(self) -> bool:  # noqa: N802 - the name is Houdini's
        return self._unordered

    def parmTemplateGroup(self) -> ParmTemplateGroup:  # noqa: N802 - the name is Houdini's
        self.read += 1
        return ParmTemplateGroup(self.templates)

    def hasSectionData(self, name: str) -> bool:  # noqa: N802 - the name is Houdini's
        return name == "DialogScript" and bool(self._dialog)

    def sectionData(self, name: str) -> str:  # noqa: N802 - the name is Houdini's
        if not self.hasSectionData(name):
            raise OperationFailed("no such section")
        return self._dialog

    def embeddedHelp(self) -> str:  # noqa: N802 - the name is Houdini's
        return self.help_text

    def defaultHelpUrl(self) -> str:  # noqa: N802 - the name is Houdini's
        return f"operator:{self._category}/{self._name}"

    def definition(self) -> Any:
        if self._library is None:
            return None
        return SimpleNamespace(libraryFilePath=lambda: self._library)

    def hidden(self) -> bool:
        return self._hidden

    def deprecated(self) -> bool:
        return self._deprecated is not None

    def deprecationInfo(self) -> dict[str, Any]:  # noqa: N802 - the name is Houdini's
        return dict(self._deprecated or {})

    def namespaceOrder(self) -> tuple[str, ...]:  # noqa: N802 - the name is Houdini's
        return self._order


class NodeTypeCategory:
    """One context's types, by name."""

    def __init__(self, name: str, types: list[NodeType]) -> None:
        self._name = name
        self.types = {kind.name(): kind for kind in types}

    def name(self) -> str:
        return self._name

    def nodeTypes(self) -> dict[str, NodeType]:  # noqa: N802 - the name is Houdini's
        return dict(self.types)


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
    return {"/obj": "Object", "/out": "Driver", "/stage": "Lop"}.get(parent.path(), "Sop")


# Section: the node types a session knows, read without a node

# Where the stand in's own assets say their library is. A check sets `HFS` to
# the folder above it to see it written as `$HFS`.
HFS_LIBRARY = "/opt/hfs/houdini/otls/OPlibSop.hda"

WRANGLE_DIALOG = """{
    name\tattribwrangle
    inputlabel\t1\t"Geometry to Process with Wrangle"
    inputlabel\t2\t"Ancillary Input, point(1, ...) to Access"
    inputlabel\t3\t"Ancillary Input, point(2, ...) to Access"
    inputlabel\t4\t"Ancillary Input, point(3, ...) to Access"
}"""

TOOL_DIALOG = """{
    name\ttool
    inputlabel\t1\t"Mesh to \\"Fix\\""
    outputlabel\t1\t"Kept"
    outputlabel\t2\t"Discarded"
}"""

RAGDOLL_DIALOG = """{
    name\tragdollsolver
    inputlabel\t1\tSkeleton
    inputlabel\t2\t"Constraint Geometry"
    inputlabel\t3\t""
    outputlabel\t1\tSkeleton
}"""

TOOL_HELP = """= Example Tool =

#type: node
#context: sop

\"\"\"Tidies a mesh and splits off the [pieces|Node:sop/split] it cannot fix.\"\"\"

@inputs

Mesh:
    The mesh to tidy.
"""


def _wrangle_templates() -> list[ParmTemplate]:
    """A wrangle's parameters as its type defines them, in folders, with a
    multiparm, a static and a dynamic menu, code, a range, a ramp and one
    hidden parameter."""
    code = ParmTemplate(
        "Folder",
        "Code",
        0,
        name="folder0",
        children=(
            ParmTemplate("String", "Group", ("",), name="group", script="opmenu -l . group"),
            ParmTemplate(
                "Menu",
                "Run Over",
                2,
                name="class",
                menu=("detail", "primitive", "point", "vertex"),
                labels=("Detail (only once)", "Primitives", "Points", "Vertices"),
                expression="",
            ),
            ParmTemplate(
                "Int",
                "Number Count",
                (10,),
                name="vex_numcount",
                span=(0, 10000, True, False),
                expression=("",),
            ),
            ParmTemplate(
                "String",
                "VEXpression",
                ("",),
                name="snippet",
                tags={"editor": "1", "editorlang": "VEX"},
                expression=("",),
            ),
            ParmTemplate("Separator", "", name="sepparm"),
            ParmTemplate(
                "Toggle", "Enforce Prototypes", False, name="vex_strict", expression="off"
            ),
            ParmTemplate(
                "Menu",
                "Channels",
                511,
                name="channels",
                menu=("tx", "ty", "tz", "rx", "ry", "rz"),
                menu_type="StringToggle",
                hidden=True,
            ),
        ),
    )
    bindings = ParmTemplate(
        "Folder",
        "Bindings",
        0,
        name="folder1",
        children=(
            ParmTemplate(
                "Folder",
                "Number of Bindings",
                0,
                name="bindings",
                folder="MultiparmBlock",
                children=(
                    ParmTemplate("String", "Attribute Name", ("",), name="bindname#"),
                    ParmTemplate("String", "VEX Parameter", ("",), name="bindparm#"),
                ),
            ),
            ParmTemplate(
                "Float",
                "Offset",
                (0.0, 1.0, 0.0),
                name="offset",
                size=3,
                span=(-1.0, 1.0, False, False),
                expression=("", "$F", ""),
            ),
        ),
    )
    return [
        code,
        bindings,
        ParmTemplate("Ramp", "Remap", 2, name="remap", ramp="Float"),
        ParmTemplate("Button", "Compile", name="compile"),
        ParmTemplate("String", "Label", ("",), name="descriptiveparm", hidden=True),
    ]


def type_library() -> dict[str, NodeTypeCategory]:
    """The contexts a session has, each with the types a check reads."""
    transform = [
        ParmTemplate("String", "Group", ("",), name="group"),
        ParmTemplate("Float", "Translate", (0.0, 0.0, 0.0), name="t", size=3),
    ]
    sops = [
        NodeType(
            "attribwrangle",
            "Sop",
            inputs=(0, 4),
            templates=_wrangle_templates(),
            dialog=WRANGLE_DIALOG,
            library=HFS_LIBRARY,
        ),
        NodeType("volumewrangle", "Sop", label="Volume Wrangle", inputs=(0, 4)),
        NodeType("wranglehelper", "Sop", label="Wrangle Helper"),
        NodeType("attribwranglecore", "Sop", label="Attribute Wrangle Core", hidden=True),
        NodeType("xform", "Sop", templates=transform),
        NodeType("merge", "Sop", label="Merge", inputs=(0, 9999), unordered=True),
        NodeType(
            "copytopoints",
            "Sop",
            label="Copy to Points",
            inputs=(2, 2),
            order=("copytopoints::2.0", "copytopoints"),
        ),
        NodeType(
            "copytopoints::2.0",
            "Sop",
            label="Copy to Points",
            inputs=(2, 2),
            templates=[ParmTemplate("String", "Source Group", ("",), name="sourcegroup")],
            order=("copytopoints::2.0", "copytopoints"),
        ),
        NodeType(
            "com.example::tool::1.0",
            "Sop",
            label="Example Tool",
            outputs=2,
            templates=[ParmTemplate("Float", "Tolerance", (0.01,), name="tolerance")],
            dialog=TOOL_DIALOG,
            help_text=TOOL_HELP,
            library="/shared/assets/tool.hda",
        ),
        NodeType("null", "Sop", label="Null"),
        # A namespaced name with no version makes the newest version of it.
        NodeType(
            "kinefx::ragdollsolver",
            "Sop",
            label="Ragdoll Solver",
            inputs=(1, 4),
            order=("kinefx::ragdollsolver::2.0", "kinefx::ragdollsolver"),
        ),
        NodeType(
            "kinefx::ragdollsolver::2.0",
            "Sop",
            label="Ragdoll Solver",
            inputs=(1, 3),
            dialog=RAGDOLL_DIALOG,
            order=("kinefx::ragdollsolver::2.0", "kinefx::ragdollsolver"),
        ),
        # One whose namespace order puts a type of another namespace first,
        # which making a node of it never goes to.
        NodeType(
            "apex::invokegraph",
            "Sop",
            label="Invoke Graph",
            order=("invokegraph", "apex::invokegraph"),
        ),
        NodeType("invokegraph", "Sop", label="Invoke Graph (old)"),
        NodeType("splitter", "Sop", label="Splitter", outputs=2),
        NodeType("labs::thing::1.0", "Sop", label="Labs Thing"),
        NodeType(
            "oldsmooth",
            "Sop",
            label="Old Smooth",
            deprecated={"new_type": NodeType("smooth::2.0", "Sop"), "version": "20.0"},
        ),
    ]
    objects = [
        NodeType("geo", "Object", label="Geometry", inputs=(0, 1)),
        NodeType("null", "Object", label="Null", inputs=(0, 1), templates=transform[1:]),
        NodeType("cam", "Object", label="Camera", inputs=(0, 1)),
    ]
    lops = [
        NodeType("attribwrangle", "Lop", inputs=(0, 4)),
        NodeType("null", "Lop"),
        NodeType("wrangler", "Lop", label="Wrangler"),
    ]
    # Recipes, and a network that holds another context: a search with no
    # context leaves both kinds of category out.
    data = [NodeType("sidefx::recipe::lop::testscene_wrangle", "Data", label="Test Scene")]
    vopnets = [NodeType("wranglenet", "VopNet", label="Wrangle Network")]
    return {
        "Sop": NodeTypeCategory("Sop", sops),
        "Object": NodeTypeCategory("Object", objects),
        "Lop": NodeTypeCategory("Lop", lops),
        "Driver": NodeTypeCategory("Driver", [NodeType("null", "Driver", inputs=(0, 9999))]),
        "Cop2": NodeTypeCategory("Cop2", [NodeType("vexfilter", "Cop2", label="VEX Filter")]),
        "Data": NodeTypeCategory("Data", data),
        "VopNet": NodeTypeCategory("VopNet", vopnets),
    }


def preferred_type(library: dict[str, NodeTypeCategory], name: str) -> NodeType | None:
    """The type a bare name makes in a context written as `Sop/name`, as Houdini picks it."""
    category, _, wanted = name.partition("/")
    found = library.get(category)
    if found is None:
        return None
    types = found.nodeTypes()
    exact = types.get(wanted)
    if exact is not None:
        return types.get(exact.namespaceOrder()[0], exact)
    same = [kind for kind in types.values() if kind.nameComponents()[2] == wanted]
    return same[0] if same else None


_as_hou(Parm)
_as_hou(ParmTuple)
# The names a real session gives them: a node of any context is a subclass
# whose name ends in `Node`, and its type one whose name ends in `NodeType`.
_as_hou(NodeType, "OpNodeType")
_as_hou(NodeTypeCategory, "NodeTypeCategory")
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

    def record(self, undo: Any) -> None:
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
        # Every type the session knows, by context, as a type reads with no node.
        self.library = type_library()
        # The folders on the search path, for `findDirectories`.
        self.search_path: list[str] = []
        self.empty()
        self.undos.labels.clear()
        self.hipFile = HipFile(self, "/Users/somebody/scenes/example.hip")

    def empty(self) -> None:
        """Throw the scene away and put the empty networks back."""
        self.root._children.clear()
        for name in ("obj", "out", "mat", "stage"):
            self.root._children.append(Node(self, name, "network", self.root))

    def find_directories(self, relative: str) -> tuple[str, ...]:
        """Every folder on the search path that holds `relative`, in path order."""
        found = [os.path.join(root, relative) for root in self.search_path]
        return tuple(folder for folder in found if os.path.isdir(folder))

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
            nodeTypeCategories=lambda: dict(self.library),
            preferredNodeType=lambda name, parent=None: preferred_type(self.library, name),
            findDirectories=self.find_directories,
            Vector3=Vector3,
            Matrix4=Matrix4,
            OperationFailed=OperationFailed,
            ObjectWasDeleted=ObjectWasDeleted,
            InvalidInput=InvalidInput,
            PermissionError=PermissionError,
            LoadWarning=LoadWarning,
        )
