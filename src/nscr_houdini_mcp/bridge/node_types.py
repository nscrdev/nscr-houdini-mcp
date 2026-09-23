"""`node.type`: what a node type is, read from the type and never from a node.

Two ways in.

- A type by name, in a context. Its inputs and outputs, its parameters as the
  type defines them, with their defaults, menus and ranges, and the one line
  its help opens with. Everything comes from `hou.NodeType`: its parameter
  template group, the dialog script section it carries when it is an asset,
  and its embedded help. So an installed third party asset reads as rightly
  as a built in type, and nothing is made, cooked or loaded to find out.
- A search by keyword over every type's name, label and help line, ranked by
  how closely it matches: the exact name, then a name that starts with the
  words, then one that holds them, then the label, then the help.

A name without a namespace or a version is the type Houdini would make for
it, which for a type with several versions is the newest, and the answer says
which name it was asked for. A name that is not there is `TYPE_NOT_FOUND`
with the closest names in the context; one that is there in another context
says which.

Input and output labels come from the asset's dialog script where there is
one. A type compiled into Houdini carries none, so its labels come from the
headings of its help page, which only approximate what the node shows, and
are left out where that has none either. `labels_from` says which it was.

Help pages are read from the `nodes.zip` Houdini ships and from every
`help/nodes` folder on Houdini's search path, where packages keep theirs. The
first line of each page, its tags and which type it documents are gathered
once per process, the first time a search or a help line needs them, from
the head of each page.

A search with no context leaves out the categories that hold no node a
person places: data recipes, the managers and the networks that only hold
other contexts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import zipfile
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from typing import Any

from nscr_houdini_mcp.bridge.errors import BridgeError, did_you_mean
from nscr_houdini_mcp.bridge.tools import (
    DEFAULT_LIMIT,
    DETAIL_LEVELS,
    MAX_LIMIT,
    ToolContext,
    _ask,
    _choice,
    _houdini,
    _number,
    _quiet,
)

# The short names a caller uses for a context, and the category each means.
# A material network holds shader nodes, so `mat` reads the same types `vop`
# does. Houdini's own category names are taken as they are, in any case.
CONTEXTS: dict[str, str] = {
    "obj": "Object",
    "sop": "Sop",
    "lop": "Lop",
    "cop": "Cop",
    "dop": "Dop",
    "top": "Top",
    "chop": "Chop",
    "out": "Driver",
    "mat": "Vop",
    "vop": "Vop",
}

# The short name for each category, where there is one, for a caller to pass
# back as `context`.
SHORT_NAMES: dict[str, str] = {
    "Object": "obj",
    "Sop": "sop",
    "Lop": "lop",
    "Cop": "cop",
    "Dop": "dop",
    "Top": "top",
    "Chop": "chop",
    "Driver": "out",
    "Vop": "vop",
}

# The folder of the help pages for each category.
HELP_FOLDERS: dict[str, str] = {**SHORT_NAMES, "Cop2": "cop2", "Shop": "shop", "Manager": "manager"}

# The order a search with no context goes through the categories in, and the
# order equally close matches from different ones come back in.
CATEGORY_ORDER = ("Sop", "Object", "Lop", "Dop", "Cop", "Chop", "Top", "Driver", "Vop")

# Categories a search with no context leaves out: they hold no node a person
# places in a network of their own, such as the recipes and the networks
# that only hold other contexts.
UNSEARCHED = frozenset({"Data", "Manager", "Director"})

TYPE_INCLUDES = ("help", "hidden")
NODE_TYPE_ARGUMENTS = (
    "context",
    "type",
    "query",
    "parm_filter",
    "detail",
    "include",
    "limit",
    "offset",
)

# How many near names a type that is not there comes back with.
MAX_SUGGESTIONS = 5

# A type can take thousands of inputs, as a merge does. Past this many only
# the ones with something to say are listed, and the answer says there are more.
MAX_LISTED_INPUTS = 16

# The longest static menu a row carries in full.
MAX_MENU_ITEMS = 500

# Parameter kinds that hold no value and are left out of the table.
NO_VALUE = frozenset({"Separator", "Label", "FolderSet"})

# How much of each help page is read to find what it documents and its first
# line. The header and the summary sit at the top.
HELP_HEAD_BYTES = 3000

# How close a search match is, closest first.
EXACT, PREFIX, IN_NAME, IN_LABEL, IN_HELP, ALL_WORDS = range(6)

# A label is quoted when it has a space in it and bare when it is one word.
_LABEL_LINE = re.compile(
    r'^[ \t]*(input|output)label[ \t]+(\d+)[ \t]+(?:"((?:[^"\\\n]|\\.)*)"|(\S.*?))[ \t]*$',
    re.MULTILINE,
)
_SUMMARY = re.compile(r'"""(.*?)"""', re.DOTALL)
_HEADER = re.compile(r"^#(\w+):\s*(.*?)\s*$", re.MULTILINE)
_LINK = re.compile(r"\[([^\]|]*)(?:\|[^\]]*)?\]")
_SECTION = re.compile(r"^@(\w+)\s*$", re.MULTILINE)
_HEADING = re.compile(r"^(\S[^\n]*?):\s*$", re.MULTILINE)

# The longest search, in characters and in words, and how many types a
# search looks at between two looks at whether it has been asked to stop.
MAX_QUERY_CHARS = 200
MAX_QUERY_WORDS = 16
STOP_EVERY = 256

# How long a help index that could not be read is kept before it is tried
# again, so a Houdini whose help was missing or broken is not asked every call.
HELP_RETRY_S = 60.0

# Headings in a help page's inputs that are not inputs: a note to the reader.
NOTE_WORDS = frozenset({"NOTE", "NOTES", "TIP", "TIPS", "WARNING", "IMPORTANT", "CAUTION"})


def node_type(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Describe one node type, or search the types by keyword."""
    hou = _houdini(context)
    level = _choice(arguments.get("detail"), DETAIL_LEVELS, "detail", "summary")
    included = arguments.get("include") or ()
    if isinstance(included, str) or not all(item in TYPE_INCLUDES for item in included):
        raise BridgeError(
            "BAD_ARGUMENTS",
            "include takes a list of: " + ", ".join(TYPE_INCLUDES),
            {"include": included if isinstance(included, str) else list(included)},
        )
    limit = int(_number(arguments.get("limit") or DEFAULT_LIMIT, "limit", MAX_LIMIT))
    if limit < 1:
        raise BridgeError("BAD_ARGUMENTS", "limit must be at least 1")
    offset = arguments.get("offset") or 0
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise BridgeError("BAD_ARGUMENTS", "offset must be a whole number from 0")
    categories = _categories(hou)
    chosen = _category(categories, arguments.get("context"))
    page = (offset, limit)
    if arguments.get("query") is not None:
        return _search(
            hou, categories, chosen, str(arguments["query"]), set(included), page, context
        )
    if arguments.get("type") is None:
        raise BridgeError("BAD_ARGUMENTS", "node.type needs type or query")
    reader = _TypeReader(level, set(included), arguments.get("parm_filter"))
    return reader.describe(hou, categories, chosen, str(arguments["type"]), page)


# Section: categories and types


def _categories(hou: Any) -> dict[str, Any]:
    """Every category. A failure to list them is the call's failure, not an empty list."""
    found = hou.nodeTypeCategories() or {}
    return {str(name): category for name, category in dict(found).items()}


def _category(categories: Mapping[str, Any], given: Any) -> Any:
    """The category a context names, or nothing when none was given."""
    if given is None:
        return None
    text = str(given).strip()
    name = CONTEXTS.get(text.lower())
    if name is None:
        name = next((known for known in categories if known.lower() == text.lower()), None)
    if name == "Cop" and "Cop" not in categories and "Cop2" in categories:
        name = "Cop2"
    if name is None or name not in categories:
        accepted = sorted(set(CONTEXTS) | set(categories))
        raise BridgeError(
            "BAD_ARGUMENTS",
            f"no context named {text}",
            {"context": text, "did_you_mean": did_you_mean(text, accepted), "contexts": accepted},
            hint="use a short name such as sop or obj, or a category name such as Sop",
        )
    return categories[name]


def _types_of(category: Any) -> dict[str, Any]:
    """A category's types. A failure to list them is not a type that is missing."""
    return {str(name): kind for name, kind in dict(category.nodeTypes() or {}).items()}


def _category_name(category: Any) -> str:
    return str(_ask(category, "name") or "")


def _where(category: Any) -> dict[str, Any]:
    name = _category_name(category)
    said: dict[str, Any] = {"category": name}
    if name in SHORT_NAMES:
        said["context"] = SHORT_NAMES[name]
    return said


def _ordered(categories: Mapping[str, Any]) -> list[Any]:
    first = [categories[name] for name in CATEGORY_ORDER if name in categories]
    rest = [categories[name] for name in sorted(categories) if name not in CATEGORY_ORDER]
    return first + rest


def _find(hou: Any, categories: Mapping[str, Any], chosen: Any, asked: str) -> tuple[Any, Any, str]:
    """The type a name means in a context, as Houdini would resolve it to make one.

    With no context the name has to belong to one category, or be written
    with its category in front, as `Sop/attribwrangle`. The name looked up,
    without that front, comes back too.
    """
    name = asked.strip()
    if chosen is None and "/" in name:
        front, _, rest = name.partition("/")
        known = front in categories or front.lower() in CONTEXTS
        anywhere = any(
            _resolve(hou, category, name) is not None for category in categories.values()
        )
        if known and not anywhere:
            chosen, name = _category(categories, front), rest
    if chosen is not None:
        found = _resolve(hou, chosen, name)
        if found is not None:
            return found, chosen, name
        elsewhere = [
            _where(category)
            for category in _ordered(categories)
            if category is not chosen and _resolve(hou, category, name) is not None
        ]
        place = _category_name(chosen)
        if elsewhere:
            where = ", ".join(item.get("context") or item["category"] for item in elsewhere)
            raise BridgeError(
                "TYPE_NOT_FOUND",
                f"{name} is not in {place}; it is in {where}",
                {"type": name, "category": place, "found_in": elsewhere},
                hint="pass context as one named in found_in",
            )
        raise BridgeError(
            "TYPE_NOT_FOUND",
            f"no {place} type named {name}",
            {
                "type": name,
                "category": place,
                "did_you_mean": near_types(name, _types_of(chosen)),
            },
        )
    having = [
        (category, found)
        for category in _ordered(categories)
        for found in [_resolve(hou, category, name)]
        if found is not None
    ]
    if len(having) == 1:
        category, found = having[0]
        return found, category, name
    if having:
        raise BridgeError(
            "BAD_ARGUMENTS",
            f"{name} is a type in {len(having)} contexts; pass context to say which",
            {"argument": "context", "type": name, "found_in": [_where(c) for c, _ in having]},
            hint="pass context as one named in found_in",
        )
    every: dict[str, Any] = {}
    for category in _ordered(categories):
        for type_name, kind in _types_of(category).items():
            every.setdefault(f"{_category_name(category)}/{type_name}", kind)
    raise BridgeError(
        "TYPE_NOT_FOUND",
        f"no type named {name} in any context",
        {"type": name, "did_you_mean": near_types(name, every)},
    )


def _resolve(hou: Any, category: Any, name: str) -> Any:
    """The type Houdini makes for a name in one category, or nothing.

    A name with no namespace and no version is the one Houdini prefers for
    it, which is the newest version where there are several. A namespaced
    name with no version is the newest version of that same namespace and
    name, as making a node of it gives: the type's namespace order can put a
    type of another namespace or name first, such as `invokegraph` for
    `apex::invokegraph`, and making the node never goes there.
    """
    types = _types_of(category)
    exact = types.get(name)
    if "::" in name:
        namespace, base, version = _split_name(name)
        if exact is None or version:
            return exact
        for candidate in _ask(exact, "namespaceOrder") or ():
            kind = types.get(str(candidate))
            if kind is not None and _components(kind)[1:3] == (namespace, base):
                return kind
        return exact
    preferred = _quiet(lambda: hou.preferredNodeType(f"{_category_name(category)}/{name}"))
    if preferred is not None and _category_name(_ask(preferred, "category")) in (
        "",
        _category_name(category),
    ):
        return preferred
    if exact is not None:
        order = _ask(exact, "namespaceOrder") or ()
        first = types.get(str(order[0])) if order else None
        return first or exact
    return None


def near_types(wanted: str, types: Mapping[str, Any], *, limit: int = MAX_SUGGESTIONS) -> list[str]:
    """Type names close to one that is not there, closest first.

    A name is compared whole and by its base name, without the namespace and
    version, so a misspelled base name still finds a namespaced type.
    """
    names = list(types)
    found = did_you_mean(wanted, names, limit=limit)
    by_base: dict[str, list[str]] = {}
    for name, kind in types.items():
        base = (_components(kind)[2] if kind is not None else "") or _base_name(name)
        by_base.setdefault(base, []).append(name)
    for base in did_you_mean(_base_name(wanted), list(by_base), limit=limit):
        for name in sorted(by_base[base], key=len):
            if name not in found:
                found.append(name)
    return found[:limit]


def _base_name(name: str) -> str:
    """`attribwrangle` for `Sop/attribwrangle`, `tool` for `com.example::tool::1.0`."""
    return _split_name(name.rsplit("/", 1)[-1])[1]


def _split_name(name: str) -> tuple[str, str, str]:
    """The namespace, base name and version written into a full type name.

    Only for a name with no type to ask: a type's own `nameComponents` is
    what says this where there is one.
    """
    parts = name.split("::")
    version = ""
    if len(parts) > 1 and re.fullmatch(r"\d+(\.\d+)*", parts[-1]):
        version = parts.pop()
    return ("::".join(parts[:-1]), parts[-1], version)


def _components(node_type: Any) -> tuple[str, str, str, str]:
    parts = _ask(node_type, "nameComponents")
    if parts and len(parts) == 4:
        return tuple(str(part or "") for part in parts)  # type: ignore[return-value]
    name = str(_ask(node_type, "name") or "")
    return ("", "", name, "")


# Section: one type


class _TypeReader:
    """One description, with the level and the parameter filter the call chose."""

    def __init__(self, level: str, included: set[str], parm_filter: Any) -> None:
        self.standard = level in ("standard", "full")
        self.full = level == "full"
        self.help = self.full or "help" in included
        self.hidden = self.full or "hidden" in included
        wanted = str(parm_filter or "all")
        if wanted == "non_default":
            raise BridgeError(
                "BAD_ARGUMENTS",
                "a type has only defaults; parm_filter takes all or a glob",
                {"argument": "parm_filter"},
            )
        self.glob = None if wanted == "all" else wanted

    def describe(
        self,
        hou: Any,
        categories: Mapping[str, Any],
        chosen: Any,
        asked: str,
        page: tuple[int, int],
    ) -> dict[str, Any]:
        try:
            node_type, category, looked_up = _find(hou, categories, chosen, asked)
            name, least, most, outputs, label, rows = self.read(node_type)
        except BridgeError:
            raise
        except Exception:  # noqa: BLE001 - one more try, then the error is the answer
            # A definition loaded again while it was read leaves the type
            # object behind it stale. It is looked up once more, and what
            # that raises goes to the caller as a coded error.
            categories = _categories(hou)
            if chosen is not None:
                chosen = categories.get(_category_name(chosen), chosen)
            node_type, category, looked_up = _find(hou, categories, chosen, asked)
            name, least, most, outputs, label, rows = self.read(node_type)
        result: dict[str, Any] = {
            "type": name,
            "label": label,
            "category": _category_name(category),
            "min_inputs": least,
            "max_inputs": most,
            "max_outputs": outputs,
        }
        if looked_up != name:
            result["resolved_from"] = looked_up
        if not self.standard:
            result["parm_count"] = len(rows)
        kept: list[Any] = []

        def help_page() -> dict[str, Any] | None:
            if not kept:
                kept.append(_HELP.page_for(hou, node_type, category))
            return kept[0]

        if self.standard:
            result.update(_identity(node_type))
            labels, source = _labels(
                node_type,
                lambda: _HELP.whole(help_page()),
                {"input": most, "output": outputs},
            )
            result["inputs"], more = _inputs(least, most, labels["input"])
            if more:
                result["more_inputs"] = True
            if _ask(node_type, "hasUnorderedInputs"):
                result["unordered_inputs"] = True
            result["outputs"] = [
                {"index": index, "label": labels["output"].get(index)}
                for index in range(min(outputs, MAX_LISTED_INPUTS))
            ]
            result["labels_from"] = source
            offset, limit = page
            chosen_rows = rows[offset : offset + limit]
            result["parms"] = chosen_rows
            result["total"] = len(rows)
            if offset + len(chosen_rows) < len(rows):
                result["more"] = True
                result["next_offset"] = offset + len(chosen_rows)
        if self.help:
            page_help = help_page()
            result["help_summary"] = page_help["summary"] if page_help else None
            result["help_path"] = page_help["path"] if page_help else None
        if kept and not _HELP.available:
            result["help_available"] = False
        result["mark"] = _mark(rows, identity=_definition_identity(node_type))
        return result

    def read(self, node_type: Any) -> tuple[str, int, int, int, str, list[dict[str, Any]]]:
        """What a card cannot do without, read strictly: a failure here is not an empty card."""
        name = str(node_type.name())
        least = int(node_type.minNumInputs() or 0)
        most = int(node_type.maxNumInputs() or 0)
        outputs = int(node_type.maxNumOutputs() or 0)
        label = str(node_type.description() or "")
        rows = self.rows(_entries(node_type), (), hidden=False, multi=False)
        return name, least, most, outputs, label, rows

    def rows(
        self, templates: Any, folders: tuple[str, ...], *, hidden: bool, multi: bool
    ) -> list[dict[str, Any]]:
        """The parameters under some templates, in the order the pane shows them.

        A plain folder is not a row: its parameters are, each saying which
        folders it sits in. A multiparm is a row with its instance template
        nested under it. What sits in a hidden folder is hidden too.
        """
        found: list[dict[str, Any]] = []
        for template in templates or ():
            kind = _kind(template)
            if kind in NO_VALUE:
                continue
            concealed = hidden or bool(_ask(template, "isHidden"))
            if kind == "Folder" and not _is_multiparm(template):
                label = str(_ask(template, "label") or _ask(template, "name") or "")
                found.extend(
                    self.rows(
                        template.parmTemplates(),
                        (*folders, label),
                        hidden=concealed,
                        multi=multi,
                    )
                )
                continue
            if concealed and not self.hidden:
                continue
            row = self.row(template, kind, folders, hidden=concealed, multi=multi)
            if kind == "Folder":
                inner = self.rows(template.parmTemplates(), (), hidden=concealed, multi=True)
                matched = multi or self.keeps(row["name"])
                if not matched:
                    inner = [item for item in inner if self.keeps(item["name"])]
                    if not inner:
                        continue
                row["instances"] = {"parms": inner}
            elif not multi and not self.keeps(row["name"]):
                continue
            found.append(row)
        return found

    def keeps(self, name: str) -> bool:
        return self.glob is None or fnmatchcase(name, self.glob)

    def row(
        self, template: Any, kind: str, folders: tuple[str, ...], *, hidden: bool, multi: bool
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": str(_ask(template, "name") or ""),
            "label": str(_ask(template, "label") or ""),
            "type": kind,
        }
        size = _ask(template, "numComponents")
        row["size"] = int(size) if isinstance(size, int) else 1
        if kind == "Menu" and _toggles(template):
            row["menu_toggles"] = True
        if kind == "Ramp":
            points = _ask(template, "defaultValue")
            if isinstance(points, int):
                row["default_points"] = points
        elif kind != "Button":
            default = _default(template, kind)
            if default is not None:
                row["default"] = default
            expression = _default_expression(template)
            if expression is not None:
                row["default_expr"] = expression
        language = _code_language(template)
        if language:
            row["code"] = language
        if kind == "Ramp":
            ramp = str(_ask(template, "parmType") or "").rsplit(".", 1)[-1].lower()
            if ramp:
                row["ramp"] = ramp
        if multi:
            row["is_multiparm_template"] = True
        if hidden:
            row["hidden"] = True
        if self.full:
            menu = _menu(template, kind)
            if menu is not None:
                row["menu"] = menu
                if isinstance(menu, list) and len(menu) >= MAX_MENU_ITEMS:
                    row["menu_truncated"] = True
            span = _range(template, kind)
            if span is not None:
                row["range"] = span
            if folders:
                row["folder"] = list(folders)
        return row


def _entries(node_type: Any) -> list[Any]:
    """A type's top level templates. Houdini raising here is the call's failure."""
    group = node_type.parmTemplateGroup()
    reader = getattr(group, "entries", None) or group.parmTemplates
    return list(reader() or ())


def _kind(template: Any) -> str:
    return str(template.type().name() or "")


def _is_multiparm(template: Any) -> bool:
    return "multiparm" in str(_ask(template, "folderType") or "").lower()


def _single(values: Any) -> Any:
    if isinstance(values, (tuple, list)):
        values = list(values)
        return values[0] if len(values) == 1 else values
    return values


def _default(template: Any, kind: str) -> Any:
    """The default as the pane shows it. A menu's is the token it selects.

    The token is looked up here from the index and the menu's items, never
    asked of the template: `defaultValueAsString` takes a menu whose items
    toggle on and off for a plain one, reads its default, which is a mask of
    the items that are on, as an index, and brings the whole of Houdini down.
    Such a menu's default stays the mask.
    """
    value = _ask(template, "defaultValue")
    if value is None:
        return None
    if kind == "Menu" and not _toggles(template) and isinstance(value, int):
        items = list(_ask(template, "menuItems") or ())
        if 0 <= value < len(items):
            return str(items[value])
    return _single(value)


def _toggles(template: Any) -> bool:
    """Whether a menu's items are turned on and off each, rather than one chosen."""
    return "toggle" in str(_ask(template, "menuType") or "").lower()


def _default_expression(template: Any) -> Any:
    """The expressions a parameter starts with, when it starts with any."""
    written = _ask(template, "defaultExpression")
    if written is None:
        return None
    texts = [str(text) for text in (written if isinstance(written, (tuple, list)) else [written])]
    if not any(texts):
        return None
    # A toggle's default written as a word is its default, not an expression.
    if len(texts) == 1 and texts[0] in ("on", "off"):
        return None
    return texts[0] if len(texts) == 1 else texts


def _code_language(template: Any) -> str | None:
    tags = _ask(template, "tags") or {}
    language = tags.get("editorlang") if isinstance(tags, Mapping) else None
    return str(language).lower() if language else None


def _menu(template: Any, kind: str) -> Any:
    """The menu's tokens and labels, `dynamic` when a script makes it, or nothing."""
    if kind not in ("Menu", "String", "Int"):
        return None
    script = _ask(template, "itemGeneratorScript")
    if script:
        return "dynamic"
    items = list(_ask(template, "menuItems") or ())
    if not items:
        return None
    labels = list(_ask(template, "menuLabels") or ())
    return [
        {"token": str(token), "label": str(labels[index]) if index < len(labels) else str(token)}
        for index, token in enumerate(items[:MAX_MENU_ITEMS])
    ]


def _range(template: Any, kind: str) -> dict[str, Any] | None:
    if kind not in ("Float", "Int"):
        return None
    low, high = _ask(template, "minValue"), _ask(template, "maxValue")
    if low is None or high is None:
        return None
    span: dict[str, Any] = {"min": low, "max": high}
    if _ask(template, "minIsStrict"):
        span["min_strict"] = True
    if _ask(template, "maxIsStrict"):
        span["max_strict"] = True
    return span


def _identity(node_type: Any) -> dict[str, Any]:
    """Where a type comes from: its namespace, version, asset library, standing."""
    _, namespace, _, version = _components(node_type)
    definition = _ask(node_type, "definition")
    said: dict[str, Any] = {
        "namespace": namespace or None,
        "version": version or None,
        "is_asset": definition is not None,
        "asset_library": _library(definition) if definition is not None else None,
        "deprecated": bool(_ask(node_type, "deprecated")),
    }
    if said["deprecated"]:
        info = _ask(node_type, "deprecationInfo") or {}
        replaced = info.get("new_type") if isinstance(info, Mapping) else None
        if replaced is not None:
            said["replaced_by"] = str(_ask(replaced, "name") or replaced)
        reason = info.get("reason") if isinstance(info, Mapping) else None
        if reason:
            said["deprecation_reason"] = str(reason)
    if _ask(node_type, "hidden"):
        said["hidden"] = True
    return said


def _library(definition: Any) -> str | None:
    """The asset's library file, with Houdini's own folder written as `$HFS`."""
    path = _ask(definition, "libraryFilePath")
    if not path:
        return None
    text = str(path)
    root = os.environ.get("HFS")
    if root and text.startswith(root.rstrip("/\\")):
        return "$HFS" + text[len(root.rstrip("/\\")) :].replace("\\", "/")
    return text


def _labels(
    node_type: Any,
    page: Callable[[], Mapping[str, Any] | None],
    counts: Mapping[str, int],
) -> tuple[dict[str, dict[int, str]], str | None]:
    """Input and output labels by index, and where they came from.

    An asset's dialog script says what the node shows, so when it names any
    label the labels are its alone. Otherwise the headings of the help page's
    inputs and outputs stand in, which is an approximation: the page is
    written by hand and can name an input differently from the node, or lag
    behind it. The whole page is read only then, and no more headings are
    taken than the type has inputs or outputs.
    """
    found: dict[str, dict[int, str]] = {"input": {}, "output": {}}
    if _ask(node_type, "hasSectionData", "DialogScript"):
        script = str(_ask(node_type, "sectionData", "DialogScript") or "")
        for match in _LABEL_LINE.finditer(script):
            side, number, quoted, bare = match.groups()
            text = quoted.replace('\\"', '"') if quoted is not None else bare
            found[side][int(number) - 1] = text
    if found["input"] or found["output"]:
        return found, "dialog_script"
    whole = page() or {}
    for side in ("input", "output"):
        headings = list(whole.get(f"{side}s") or ())[: counts[side]]
        found[side] = dict(enumerate(headings))
    if found["input"] or found["output"]:
        return found, "help"
    return found, None


def _inputs(least: int, most: int, labels: Mapping[int, str]) -> tuple[list[dict[str, Any]], bool]:
    count = most if most <= MAX_LISTED_INPUTS else max(least, max(labels, default=-1) + 1, 1)
    count = min(count, most)
    listed = [
        {"index": index, "label": labels.get(index), "optional": index >= least}
        for index in range(count)
    ]
    return listed, count < most


def _mark(rows: list[Any], *, identity: Any = None) -> str:
    """A fingerprint of every row a lookup pages through, and where they came from.

    Any change to a row, a default or a menu as much as a name, changes it,
    and so does a definition loaded again from its library.
    """
    text = json.dumps(
        {"rows": rows, "from": identity}, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _definition_identity(node_type: Any) -> list[Any] | None:
    """The library an asset was loaded from and when that file last changed."""
    definition = _ask(node_type, "definition")
    if definition is None:
        return None
    path = _ask(definition, "libraryFilePath")
    changed = None
    if path:
        try:
            changed = os.stat(str(path)).st_mtime_ns
        except (OSError, ValueError):
            changed = None
    return [None if path is None else str(path), changed]


# Section: searching


def _search(
    hou: Any,
    categories: Mapping[str, Any],
    chosen: Any,
    query: str,
    included: set[str],
    page: tuple[int, int],
    context: ToolContext,
) -> dict[str, Any]:
    """Every type that matches, closest first, a page at a time.

    A search that is asked to stop, because its caller gave up or the
    session is going down, hands back what it had ranked so far and says
    `stopped`, with no next page, since the order of a partial search is not
    the order of the whole one.
    """
    if len(query) > MAX_QUERY_CHARS:
        raise BridgeError(
            "BAD_ARGUMENTS",
            f"query is at most {MAX_QUERY_CHARS} characters",
            {"argument": "query", "given": len(query)},
        )
    words = list(dict.fromkeys(query.lower().split()))
    if not words:
        raise BridgeError("BAD_ARGUMENTS", "query needs a word to look for", {"argument": "query"})
    if len(words) > MAX_QUERY_WORDS:
        raise BridgeError(
            "BAD_ARGUMENTS",
            f"query is at most {MAX_QUERY_WORDS} words",
            {"argument": "query", "given": len(words)},
        )
    wanted = " ".join(words)
    if chosen is not None:
        looked = [chosen]
    else:
        looked = [
            category
            for category in _ordered(categories)
            if _category_name(category) not in UNSEARCHED
            and not _category_name(category).endswith("Net")
        ]
    pages = _HELP.pages(hou)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    stopped = False
    seen = 0
    for order, category in enumerate(looked):
        if stopped:
            break
        place = _category_name(category)
        for name, node_type in _types_of(category).items():
            seen += 1
            if seen % STOP_EVERY == 0 and context.should_stop():
                stopped = True
                break
            hidden = bool(_ask(node_type, "hidden"))
            if hidden and "hidden" not in included:
                continue
            label = str(_ask(node_type, "description") or "")
            page_help = _HELP.page_for(hou, node_type, category, embedded=False, pages=pages)
            base = _components(node_type)[2] or _base_name(name)
            closeness = _closeness(wanted, words, name, base, label, page_help)
            if closeness is None:
                continue
            deprecated = bool(_ask(node_type, "deprecated"))
            row: dict[str, Any] = {
                "type": name,
                "category": place,
                "label": label,
                "one_line": page_help["summary"] if page_help else None,
            }
            if hidden:
                row["hidden"] = True
            if deprecated:
                row["deprecated"] = True
            # Among equally close matches: more of the words in the name or
            # label, then a type still in use, then the version Houdini makes
            # for the bare name, then the context, then the shorter name.
            named = f"{name} {label}".lower()
            words_named = sum(1 for word in words if word in named)
            order_of = _ask(node_type, "namespaceOrder") or ()
            superseded = bool(order_of) and str(order_of[0]) != name
            key = (
                closeness,
                -words_named,
                hidden,
                deprecated,
                superseded,
                order,
                len(base),
                name,
            )
            ranked.append((key, row))
    if not stopped and context.should_stop():
        stopped = True
    ranked.sort(key=lambda pair: pair[0])
    rows = [row for _, row in ranked]
    mark = _mark(rows)
    offset, limit = page
    chosen_rows = rows[offset : offset + limit]
    for row in chosen_rows:
        if row["one_line"] is None:
            # An asset's own help, read only for the rows that are sent.
            category = categories.get(row["category"])
            node_type = _types_of(category).get(row["type"]) if category is not None else None
            own = (
                _HELP.page_for(hou, node_type, category, pages=pages)
                if node_type is not None
                else None
            )
            row["one_line"] = own["summary"] if own else None
    result: dict[str, Any] = {
        "query": query,
        "rows": chosen_rows,
        "total": len(rows),
        "mark": mark,
    }
    if chosen is not None:
        result["category"] = _category_name(chosen)
    if not _HELP.available:
        result["help_available"] = False
    if stopped:
        result["stopped"] = True
    elif offset + len(chosen_rows) < len(rows):
        result["more"] = True
        result["next_offset"] = offset + len(chosen_rows)
    return result


def _closeness(
    wanted: str,
    words: list[str],
    name: str,
    base: str,
    label: str,
    page: Mapping[str, Any] | None,
) -> int | None:
    lowered = name.lower()
    base = base.lower()
    if wanted in (lowered, base):
        return EXACT
    if lowered.startswith(wanted) or base.startswith(wanted):
        return PREFIX
    if wanted in lowered:
        return IN_NAME
    if wanted in label.lower():
        return IN_LABEL
    said = ""
    if page:
        said = f"{page.get('summary') or ''} {' '.join(page.get('tags') or ())}".lower()
        if wanted in said:
            return IN_HELP
    text = f"{lowered} {label.lower()} {said}"
    if len(words) > 1 and all(word in text for word in words):
        return ALL_WORDS
    return None


# Section: help pages


class _HelpIndex:
    """What the help pages say about each type, gathered once per process.

    Two places are read: the `nodes.zip` Houdini ships, and every `help/nodes`
    folder on Houdini's search path, which is where a package such as a set
    of add on tools keeps the pages for its own types. Pages are keyed by the
    help folder, namespace, type name and version each one names in its
    header, and a shipped page wins over a package page for the same type.

    The index is made again when the archive or a folder changes, which is
    rare. The archive stays open with it, so reading a whole page for its
    input headings does not open the file again.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source: tuple[Any, ...] | None = None
        self._pages: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._archive: zipfile.ZipFile | None = None
        # Whether the shipped help could be read, and when it was last tried.
        self.available = True
        self._tried = 0.0

    def pages(self, hou: Any) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        """The index, made again first when anything it was made from has changed.

        When the shipped help is missing or cannot be read, that is kept apart
        from an index that is merely empty: `available` says so, and it is
        tried again once `HELP_RETRY_S` has gone by, whether or not anything
        seems to have changed.
        """
        archive = _help_archive(hou)
        folders = _help_folders(hou)
        source = (archive, _stamp(archive), tuple(_folder_stamp(folder) for folder in folders))
        now = time.monotonic()
        with self._lock:
            stale = not self.available and now - self._tried >= HELP_RETRY_S
            if self._source != source or stale:
                self._close()
                opened = _open_archive(archive)
                shipped = _read_archive(opened) if opened is not None else None
                pages = dict(shipped or {})
                for folder in folders:
                    for key, page in _read_folder(folder).items():
                        pages.setdefault(key, page)
                if shipped is None and opened is not None:
                    opened.close()
                self._archive = opened if shipped is not None else None
                self._pages = pages
                self._source = source
                self.available = shipped is not None
                self._tried = now
            return self._pages

    def page_for(
        self,
        hou: Any,
        node_type: Any,
        category: Any,
        *,
        embedded: bool = True,
        pages: Mapping[tuple[str, str, str, str], dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """The help for one type: its own embedded help first, then a page.

        A search leaves the embedded help out, since reading it from every
        asset installed would be the slow part, and hands in the index it
        read once rather than having it looked up again for every type.
        """
        own = str(_ask(node_type, "embeddedHelp") or "") if embedded else ""
        if own.strip():
            parsed = _parse_page(own, whole=True)
            parsed["path"] = str(_ask(node_type, "defaultHelpUrl") or "") or None
            return parsed
        index = self.pages(hou) if pages is None else pages
        if not index:
            return None
        folder = HELP_FOLDERS.get(_category_name(category), _category_name(category).lower())
        _, namespace, base, version = _components(node_type)
        found = index.get((folder, namespace, base, version))
        if found is None and version:
            found = index.get((folder, namespace, base, ""))
        return found

    def whole(self, page: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """The whole of one page, for its input and output headings."""
        if page is None or "inputs" in page:
            return None if page is None else dict(page)
        text = None
        try:
            if page.get("file"):
                with open(page["file"], encoding="utf-8", errors="replace") as opened:
                    text = opened.read()
            elif page.get("member"):
                with self._lock:
                    if self._archive is not None:
                        text = self._archive.read(page["member"]).decode("utf-8", "replace")
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            text = None
        if text is None:
            return dict(page)
        whole = _parse_page(text, whole=True)
        return {**page, "inputs": whole.get("inputs"), "outputs": whole.get("outputs")}

    def _close(self) -> None:
        if self._archive is not None:
            try:
                self._archive.close()
            except OSError:
                pass
            self._archive = None


def _help_archive(hou: Any) -> str | None:
    root = os.environ.get("HFS") or _quiet(lambda: hou.getenv("HFS"))
    if not root:
        return None
    path = os.path.join(str(root), "houdini", "help", "nodes.zip")
    return path if os.path.isfile(path) else None


def _help_folders(hou: Any) -> list[str]:
    """Every `help/nodes` folder on Houdini's search path, in its order."""
    found = _quiet(lambda: hou.findDirectories("help/nodes")) or ()
    return [str(folder) for folder in found if os.path.isdir(str(folder))]


def _stamp(path: str | None) -> tuple[str, int, int] | None:
    if path is None:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (path, stat.st_mtime_ns, stat.st_size)


def _folder_stamp(folder: str) -> tuple[str, int]:
    """A folder and the latest change to it or to a context folder in it."""
    latest = 0
    for place in [folder, *_subfolders(folder)]:
        try:
            latest = max(latest, os.stat(place).st_mtime_ns)
        except OSError:
            continue
    return (folder, latest)


def _subfolders(folder: str) -> list[str]:
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [
        os.path.join(folder, name) for name in names if os.path.isdir(os.path.join(folder, name))
    ]


def _open_archive(path: str | None) -> zipfile.ZipFile | None:
    if path is None:
        return None
    try:
        return zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return None


def _page_key(folder: str, parsed: dict[str, Any]) -> tuple[str, str, str, str] | None:
    """Where one page goes in the index, from what its header says it documents.

    A page names its type in `#internal`, with the namespace and version in
    headers of their own or, as package pages do, written into the name.
    """
    internal = parsed.pop("internal", None)
    namespace = parsed.pop("namespace", "")
    version = parsed.pop("version", "")
    if not internal:
        return None
    if "::" in internal and not namespace and not version:
        namespace, internal, version = _split_name(internal)
    return (folder, namespace, internal, version)


def _read_archive(
    archive: zipfile.ZipFile,
) -> dict[tuple[str, str, str, str], dict[str, Any]] | None:
    """The pages in the shipped archive, or nothing when it cannot be read."""
    pages: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    try:
        for member in archive.namelist():
            folder, _, leaf = member.partition("/")
            if not leaf.endswith(".txt") or "/" in leaf:
                continue
            with archive.open(member) as opened:
                head = opened.read(HELP_HEAD_BYTES).decode("utf-8", "replace")
            parsed = _parse_page(head, whole=False)
            key = _page_key(folder, parsed)
            if key is None:
                continue
            parsed["path"] = f"/nodes/{member[: -len('.txt')]}"
            parsed["member"] = member
            pages.setdefault(key, parsed)
    except (OSError, EOFError, zipfile.BadZipFile):
        return None
    return pages


def _read_folder(root: str) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """The pages under one `help/nodes` folder, one folder per context."""
    pages: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for place in _subfolders(root):
        folder = os.path.basename(place)
        try:
            leaves = sorted(os.listdir(place))
        except OSError:
            continue
        for leaf in leaves:
            path = os.path.join(place, leaf)
            if not leaf.endswith(".txt") or not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as opened:
                    head = opened.read(HELP_HEAD_BYTES)
            except OSError:
                continue
            parsed = _parse_page(head, whole=False)
            key = _page_key(folder, parsed)
            if key is None:
                continue
            parsed["path"] = f"/nodes/{folder}/{leaf[: -len('.txt')]}"
            parsed["file"] = path
            pages.setdefault(key, parsed)
    return pages


def _parse_page(text: str, *, whole: bool) -> dict[str, Any]:
    """The header, the first line and, for a whole page, the input headings."""
    header = {key: value for key, value in _HEADER.findall(text[:HELP_HEAD_BYTES])}
    summary = _SUMMARY.search(text)
    said: dict[str, Any] = {
        "summary": _plain(summary.group(1)) if summary else None,
        "tags": [tag.strip() for tag in header.get("tags", "").split(",") if tag.strip()],
        "internal": header.get("internal", ""),
        "namespace": header.get("namespace", ""),
        "version": header.get("version", ""),
    }
    if whole:
        said["inputs"] = _headings(text, "inputs")
        said["outputs"] = _headings(text, "outputs")
    return said


def _plain(text: str) -> str:
    return " ".join(_LINK.sub(r"\1", text).split())


def _headings(text: str, section: str) -> list[str]:
    """The headings of one `@section` of a help page, in order."""
    bounds = list(_SECTION.finditer(text))
    for index, found in enumerate(bounds):
        if found.group(1) != section:
            continue
        end = bounds[index + 1].start() if index + 1 < len(bounds) else len(text)
        headings: list[str] = []
        for match in _HEADING.finditer(text[found.end() : end]):
            heading = match.group(1).strip()
            # A directive such as `:include ...:` and a note to the reader
            # sit among the headings without being inputs.
            if heading.startswith(":") or heading.upper() in NOTE_WORDS:
                continue
            headings.append(_plain(heading))
        return headings
    return []


_HELP = _HelpIndex()


def reset_help() -> None:
    """Forget the help index, so the next read builds it again."""
    global _HELP
    _HELP = _HelpIndex()
