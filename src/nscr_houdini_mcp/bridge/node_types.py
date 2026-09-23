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
headings of its help page, and are left out where that has none either.

Help pages are read from the `nodes.zip` Houdini ships. The first line of each
page, its tags and which type it documents are gathered once per process, the
first time a search or a help line needs them, from the head of each page.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
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
CATEGORY_ORDER = ("Sop", "Object", "Lop", "Cop", "Dop", "Top", "Chop", "Driver", "Vop")

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

_LABEL_LINE = re.compile(r'^\s*(input|output)label\s+(\d+)\s+"((?:[^"\\]|\\.)*)"', re.MULTILINE)
_SUMMARY = re.compile(r'"""(.*?)"""', re.DOTALL)
_HEADER = re.compile(r"^#(\w+):\s*(.*?)\s*$", re.MULTILINE)
_LINK = re.compile(r"\[([^\]|]*)(?:\|[^\]]*)?\]")
_SECTION = re.compile(r"^@(\w+)\s*$", re.MULTILINE)
_HEADING = re.compile(r"^(\S[^\n]*?):\s*$", re.MULTILINE)


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
        return _search(hou, categories, chosen, str(arguments["query"]), set(included), page)
    if arguments.get("type") is None:
        raise BridgeError("BAD_ARGUMENTS", "node.type needs type or query")
    reader = _TypeReader(level, set(included), arguments.get("parm_filter"))
    return reader.describe(hou, categories, chosen, str(arguments["type"]), page)


# Section: categories and types


def _categories(hou: Any) -> dict[str, Any]:
    found = _quiet(hou.nodeTypeCategories) or {}
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
    return {str(name): kind for name, kind in dict(_quiet(category.nodeTypes) or {}).items()}


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
    it, which is the newest version where there are several.
    """
    types = _types_of(category)
    exact = types.get(name)
    if "::" in name:
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
    for name in names:
        base = _base_name(name)
        by_base.setdefault(base, []).append(name)
    for base in did_you_mean(_base_name(wanted), list(by_base), limit=limit):
        for name in sorted(by_base[base], key=len):
            if name not in found:
                found.append(name)
    return found[:limit]


def _base_name(name: str) -> str:
    """`attribwrangle` for `Sop/attribwrangle`, `tool` for `com.example::tool::1.0`."""
    parts = name.rsplit("/", 1)[-1].split("::")
    if len(parts) >= 3:
        return parts[-2]
    if len(parts) == 2:
        return parts[0] if re.fullmatch(r"\d+(\.\d+)*", parts[1]) else parts[1]
    return parts[0]


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
        node_type, category, looked_up = _find(hou, categories, chosen, asked)
        name = str(_ask(node_type, "name") or "")
        least = int(_ask(node_type, "minNumInputs") or 0)
        most = int(_ask(node_type, "maxNumInputs") or 0)
        outputs = int(_ask(node_type, "maxNumOutputs") or 0)
        result: dict[str, Any] = {
            "type": name,
            "label": str(_ask(node_type, "description") or ""),
            "category": _category_name(category),
            "min_inputs": least,
            "max_inputs": most,
            "max_outputs": outputs,
        }
        if looked_up != name:
            result["resolved_from"] = looked_up
        rows = self.rows(_entries(node_type), (), hidden=False, multi=False)
        if not self.standard:
            result["parm_count"] = len(rows)
        kept: list[Any] = []

        def help_page() -> dict[str, Any] | None:
            if not kept:
                kept.append(_HELP.page_for(hou, node_type, category))
            return kept[0]

        if self.standard:
            result.update(_identity(node_type))
            labels = _labels(node_type, lambda: _HELP.whole(help_page()))
            result["inputs"], more = _inputs(least, most, labels["input"])
            if more:
                result["more_inputs"] = True
            if _ask(node_type, "hasUnorderedInputs"):
                result["unordered_inputs"] = True
            result["outputs"] = [
                {"index": index, "label": labels["output"].get(index)}
                for index in range(min(outputs, MAX_LISTED_INPUTS))
            ]
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
        result["mark"] = _mark([row["name"] for row in rows])
        return result

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
                        _ask(template, "parmTemplates"),
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
                inner = self.rows(_ask(template, "parmTemplates"), (), hidden=concealed, multi=True)
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
        if kind != "Button":
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
    group = _ask(node_type, "parmTemplateGroup")
    if group is None:
        return list(_ask(node_type, "parmTemplates") or ())
    entries = _ask(group, "entries")
    if entries is None:
        entries = _ask(group, "parmTemplates")
    return list(entries or ())


def _kind(template: Any) -> str:
    kind = _quiet(lambda: template.type().name())
    return str(kind) if kind else ""


def _is_multiparm(template: Any) -> bool:
    return "multiparm" in str(_ask(template, "folderType") or "").lower()


def _single(values: Any) -> Any:
    if isinstance(values, (tuple, list)):
        values = list(values)
        return values[0] if len(values) == 1 else values
    return values


def _default(template: Any, kind: str) -> Any:
    """The default as the pane shows it. A menu's is the token it selects."""
    if kind == "Menu":
        token = _ask(template, "defaultValueAsString")
        if token is not None:
            return str(token)
    value = _ask(template, "defaultValue")
    if value is None:
        return None
    return _single(value)


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


def _labels(node_type: Any, page: Callable[[], Mapping[str, Any] | None]) -> dict[str, Any]:
    """Input and output labels by index, from the dialog script or else the help.

    The whole help page is read only when the dialog script has no labels.
    """
    found: dict[str, dict[int, str]] = {"input": {}, "output": {}}
    if _ask(node_type, "hasSectionData", "DialogScript"):
        script = str(_ask(node_type, "sectionData", "DialogScript") or "")
        for side, number, text in _LABEL_LINE.findall(script):
            found[side][int(number) - 1] = text.replace('\\"', '"')
    if not found["input"] or not found["output"]:
        whole = page() or {}
        for side in ("input", "output"):
            if not found[side]:
                found[side] = dict(enumerate(whole.get(f"{side}s") or ()))
    return found


def _inputs(least: int, most: int, labels: Mapping[int, str]) -> tuple[list[dict[str, Any]], bool]:
    count = most if most <= MAX_LISTED_INPUTS else max(least, max(labels, default=-1) + 1, 1)
    count = min(count, most)
    listed = [
        {"index": index, "label": labels.get(index), "optional": index >= least}
        for index in range(count)
    ]
    return listed, count < most


def _mark(names: list[str]) -> str:
    text = json.dumps(names, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Section: searching


def _search(
    hou: Any,
    categories: Mapping[str, Any],
    chosen: Any,
    query: str,
    included: set[str],
    page: tuple[int, int],
) -> dict[str, Any]:
    wanted = " ".join(query.lower().split())
    if not wanted:
        raise BridgeError("BAD_ARGUMENTS", "query needs a word to look for", {"argument": "query"})
    words = wanted.split()
    looked = [chosen] if chosen is not None else _ordered(categories)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for order, category in enumerate(looked):
        place = _category_name(category)
        for name, node_type in _types_of(category).items():
            hidden = bool(_ask(node_type, "hidden"))
            if hidden and "hidden" not in included:
                continue
            label = str(_ask(node_type, "description") or "")
            page_help = _HELP.page_for(hou, node_type, category, embedded=False)
            closeness = _closeness(wanted, words, name, label, page_help)
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
            # for the bare name, then the shorter name.
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
                len(_base_name(name)),
                order,
                name,
            )
            ranked.append((key, row))
    ranked.sort(key=lambda pair: pair[0])
    rows = [row for _, row in ranked]
    offset, limit = page
    chosen_rows = rows[offset : offset + limit]
    for row in chosen_rows:
        if row["one_line"] is None:
            # An asset's own help, read only for the rows that are sent.
            category = categories.get(row["category"])
            node_type = _types_of(category).get(row["type"]) if category is not None else None
            own = _HELP.page_for(hou, node_type, category) if node_type is not None else None
            row["one_line"] = own["summary"] if own else None
    result: dict[str, Any] = {
        "query": query,
        "rows": chosen_rows,
        "total": len(rows),
        "mark": _mark([f"{row['category']}/{row['type']}" for row in rows]),
    }
    if chosen is not None:
        result["category"] = _category_name(chosen)
    if offset + len(chosen_rows) < len(rows):
        result["more"] = True
        result["next_offset"] = offset + len(chosen_rows)
    return result


def _closeness(
    wanted: str, words: list[str], name: str, label: str, page: Mapping[str, Any] | None
) -> int | None:
    lowered = name.lower()
    base = _base_name(lowered)
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
    """What the help pages Houdini ships say about each type, gathered once.

    Keyed by the help folder, namespace, type name and version each page
    names in its header. The index is made again when the file changes, which
    it does only when Houdini is updated.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source: tuple[str, int, int] | None = None
        self._pages: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def page_for(
        self, hou: Any, node_type: Any, category: Any, *, embedded: bool = True
    ) -> dict[str, Any] | None:
        """The help for one type: its own embedded help first, then the shipped page.

        A search leaves the embedded help out: reading it from every asset
        installed would be the slow part of the search.
        """
        own = str(_ask(node_type, "embeddedHelp") or "") if embedded else ""
        if own.strip():
            parsed = _parse_page(own, whole=True)
            parsed["path"] = str(_ask(node_type, "defaultHelpUrl") or "") or None
            return parsed
        pages = self._index(hou)
        if not pages:
            return None
        folder = HELP_FOLDERS.get(_category_name(category), _category_name(category).lower())
        _, namespace, base, version = _components(node_type)
        found = pages.get((folder, namespace, base, version))
        if found is None and version:
            found = pages.get((folder, namespace, base, ""))
        return found

    def whole(self, page: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """The whole of one page, for its input and output headings."""
        if page is None or "inputs" in page:
            return None if page is None else dict(page)
        member = page.get("member")
        source = self._source
        if not member or source is None:
            return dict(page)
        try:
            with zipfile.ZipFile(source[0]) as archive:
                text = archive.read(member).decode("utf-8", "replace")
        except (OSError, KeyError, zipfile.BadZipFile):
            return dict(page)
        whole = _parse_page(text, whole=True)
        return {**page, "inputs": whole.get("inputs"), "outputs": whole.get("outputs")}

    def _index(self, hou: Any) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        path = _help_archive(hou)
        if path is None:
            return {}
        try:
            stat = os.stat(path)
        except OSError:
            return {}
        source = (path, int(stat.st_mtime), int(stat.st_size))
        with self._lock:
            if self._source != source:
                self._pages = _read_archive(path)
                self._source = source
            return self._pages


def _help_archive(hou: Any) -> str | None:
    root = os.environ.get("HFS") or _quiet(lambda: hou.getenv("HFS"))
    if not root:
        return None
    path = os.path.join(str(root), "houdini", "help", "nodes.zip")
    return path if os.path.isfile(path) else None


def _read_archive(path: str) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    pages: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                folder, _, leaf = member.partition("/")
                if not leaf.endswith(".txt") or "/" in leaf:
                    continue
                with archive.open(member) as opened:
                    head = opened.read(HELP_HEAD_BYTES).decode("utf-8", "replace")
                parsed = _parse_page(head, whole=False)
                internal = parsed.pop("internal", None)
                if not internal:
                    continue
                parsed["path"] = f"/nodes/{member[: -len('.txt')]}"
                parsed["member"] = member
                key = (folder, parsed.pop("namespace", ""), internal, parsed.pop("version", ""))
                pages.setdefault(key, parsed)
    except (OSError, zipfile.BadZipFile):
        return {}
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
        return [_plain(match.group(1)) for match in _HEADING.finditer(text[found.end() : end])]
    return []


_HELP = _HelpIndex()


def reset_help() -> None:
    """Forget the help index, so the next read builds it again."""
    global _HELP
    _HELP = _HelpIndex()
