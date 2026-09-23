"""`hou_node_type` through the server, down to a bridge reading the stand in's types.

Every call goes the whole way: the server checks the arguments and the page
token, the router sends the call, and a real dispatcher runs the bridge's
`node.type` against the stand in for `hou`, whose types carry parameter
templates, dialog scripts, embedded help and namespaces the way a real
build's do. The help pages come from a small `nodes.zip` made for each check
under a folder named as `HFS`.
"""

from __future__ import annotations

import base64
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fake_hou import NodeType, ParmTemplate, Scene
from nscr_houdini_mcp.bridge import node_types
from nscr_houdini_mcp.tools import node_type as node_type_tool
from nscr_houdini_mcp.tools.node_type import make_token, query_of
from test_server import talk, text_of
from test_tools_inspect import Through
from test_tools_sessions import Bench

PAGES = {
    "sop/attribwrangle.txt": (
        "= Attribute Wrangle =\n\n#type: node\n#context: sop\n#internal: attribwrangle\n"
        '#tags: attrs, vex\n\n"""Runs a VEX snippet to modify attribute values."""\n\n'
        "@inputs\n\nGeometry to Process:\n    The geometry.\n"
    ),
    "sop/volumewrangle.txt": (
        "#type: node\n#context: sop\n#internal: volumewrangle\n\n"
        '"""Runs a VEX snippet to modify voxel values in a volume."""\n'
    ),
    "sop/xform.txt": (
        "#type: node\n#context: sop\n#internal: xform\n\n= Transform =\n\n"
        '"""Transforms the input geometry\nin object space."""\n\n'
        "@parameters\n\nGroup:\n    Which primitives.\n\n@inputs\n\nInput Geometry:\n"
        "    What is moved.\n\n@related\n\n- [Node:sop/copy]\n"
    ),
    "sop/copytopoints.txt": (
        "#type: node\n#context: sop\n#internal: copytopoints\n#version: 2.0\n\n"
        '"""Copies geometry in the first input onto the points of the second input."""\n\n'
        "@inputs\n\nGeometry to Copy:\n    The geometry.\n\n"
        "Target Points to Copy to:\n    The points.\n"
    ),
    "sop/copytopoints-.txt": (
        '#type: node\n#context: sop\n#internal: copytopoints\n\n"""The first version."""\n'
    ),
    "obj/null.txt": (
        '#type: node\n#context: obj\n#internal: null\n\n"""Serves as a place-holder."""\n'
    ),
    "sop/splitter.txt": (
        "#type: node\n#context: sop\n#internal: splitter\n\n"
        '"""Splits its input in two."""\n\n'
        "@inputs\n\n:include standard_inputs:\n\nNOTE:\n    Read this first.\n\n"
        "Geometry:\n    What is split.\n\nStale Second Input:\n    Gone from the node.\n\n"
        "@outputs\n\nKept:\n    One half.\n\nTip:\n    x\n\nDropped:\n    The other half.\n"
    ),
    "sop/notanode.txt": "No header here.\n",
}

# A package's own help folder: a page for its own type, with the namespace
# and version written into the name, and a page for a type the shipped help
# already documents, which the shipped page wins over.
PACKAGE_PAGES = {
    "sop/labs--thing-1.0.txt": (
        "= Labs Thing =\n\n#type: node\n#context: sop\n#internal: labs::thing::1.0\n\n"
        '""" Makes a thing from its input. """\n\n@inputs\n\nSource:\n    Anything.\n'
    ),
    "sop/attribwrangle.txt": (
        '#type: node\n#context: sop\n#internal: attribwrangle\n\n"""Not the shipped page."""\n'
    ),
    "sop/readme.md": "not a page",
}


@pytest.fixture
def hfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Houdini folder holding only the help pages above."""
    root = tmp_path / "hfs"
    help_folder = root / "houdini" / "help"
    help_folder.mkdir(parents=True)
    with zipfile.ZipFile(help_folder / "nodes.zip", "w") as archive:
        for name, text in PAGES.items():
            archive.writestr(name, text)
    monkeypatch.setenv("HFS", str(root))
    return root


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """A package folder on the search path, holding its own help pages."""
    root = tmp_path / "package"
    for name, text in PACKAGE_PAGES.items():
        page = root / "help" / "nodes" / name
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def scene(hfs: Path, package: Path) -> Any:
    made = Scene(types=("geo", "null", "cam"))
    made.search_path = [str(package), str(package.parent / "nothing-here")]
    wrangle = made.library["Sop"].types["attribwrangle"]
    wrangle._library = str(hfs / "houdini" / "otls" / "OPlibSop.hda")
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture
def bench(tmp_path: Path, scene: Scene) -> Bench:
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1")
    made.sent = Through(scene)  # type: ignore[assignment]
    return made


def lookup(bench: Bench, **arguments: Any) -> Any:
    _, [result] = talk(bench.serve(), ("hou_node_type", arguments))
    return result


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, text_of(result)
    return result.structured_content


def error(result: Any) -> dict[str, Any]:
    assert result.is_error is True, text_of(result)
    return result.structured_content["error"]


def by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in rows}


def names(body: dict[str, Any]) -> list[str]:
    return [row["type"] for row in body["rows"]]


# Section: one type


def test_a_summary_is_who_the_type_is_and_how_much_it_has(bench: Bench) -> None:
    result = lookup(bench, context="sop", type="attribwrangle")
    body = ok(result)
    assert {key: value for key, value in body.items() if key != "trace"} == {
        "type": "attribwrangle",
        "label": "Attribute Wrangle",
        "category": "Sop",
        "min_inputs": 0,
        "max_inputs": 4,
        "max_outputs": 1,
        "parm_count": 9,
    }
    assert body["trace"]["session_id"] == "s-1"
    # A small result is mirrored whole in the text block.
    assert json.loads(text_of(result))["type"] == "attribwrangle"
    [sent] = bench.sent.calls
    assert sent["tool"] == "node.type"
    assert sent["operation_id"] is None
    assert sent["arguments"] == {"context": "sop", "type": "attribwrangle", "limit": 200}


def test_a_standard_read_has_the_inputs_outputs_and_visible_parameters(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="attribwrangle", detail="standard"))
    assert body["inputs"] == [
        {"index": 0, "label": "Geometry to Process with Wrangle", "optional": True},
        {"index": 1, "label": "Ancillary Input, point(1, ...) to Access", "optional": True},
        {"index": 2, "label": "Ancillary Input, point(2, ...) to Access", "optional": True},
        {"index": 3, "label": "Ancillary Input, point(3, ...) to Access", "optional": True},
    ]
    assert body["outputs"] == [{"index": 0, "label": None}]
    # The dialog script names labels, so none come from the help page.
    assert body["labels_from"] == "dialog_script"
    assert body["namespace"] is None and body["version"] is None
    assert body["is_asset"] is True
    assert body["asset_library"] == "$HFS/houdini/otls/OPlibSop.hda"
    assert body["deprecated"] is False
    rows = by_name(body["parms"])
    assert [row["name"] for row in body["parms"]] == [
        "group",
        "class",
        "vex_numcount",
        "snippet",
        "vex_strict",
        "bindings",
        "offset",
        "remap",
        "compile",
    ]
    assert body["total"] == 9
    # A menu's default is the token it selects; a toggle's written default is no expression.
    assert rows["class"] == {
        "name": "class",
        "label": "Run Over",
        "type": "Menu",
        "size": 1,
        "default": "point",
    }
    assert rows["vex_strict"]["default"] is False and "default_expr" not in rows["vex_strict"]
    assert rows["snippet"]["code"] == "vex"
    assert rows["offset"]["default"] == [0.0, 1.0, 0.0]
    assert rows["offset"]["default_expr"] == ["", "$F", ""]
    assert rows["remap"] == {
        "name": "remap",
        "label": "Remap",
        "type": "Ramp",
        "size": 1,
        "default_points": 2,
        "ramp": "float",
    }
    assert "default" not in rows["compile"]
    # A multiparm is one row with its instance template nested under it.
    assert rows["bindings"]["default"] == 0
    assert rows["bindings"]["instances"] == {
        "parms": [
            {
                "name": "bindname#",
                "label": "Attribute Name",
                "type": "String",
                "size": 1,
                "default": "",
                "is_multiparm_template": True,
            },
            {
                "name": "bindparm#",
                "label": "VEX Parameter",
                "type": "String",
                "size": 1,
                "default": "",
                "is_multiparm_template": True,
            },
        ]
    }
    # Menus, ranges, folders, help and hidden parameters wait for full.
    for row in body["parms"]:
        assert not {"menu", "range", "folder", "hidden"} & set(row)
    assert "help_summary" not in body and "descriptiveparm" not in rows


def test_a_full_read_adds_hidden_parameters_menus_ranges_folders_and_help(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="attribwrangle", detail="full"))
    rows = by_name(body["parms"])
    assert rows["class"]["menu"] == [
        {"token": "detail", "label": "Detail (only once)"},
        {"token": "primitive", "label": "Primitives"},
        {"token": "point", "label": "Points"},
        {"token": "vertex", "label": "Vertices"},
    ]
    # A menu a script fills in is named as one, and the script is never run.
    assert rows["group"]["menu"] == "dynamic"
    assert rows["vex_numcount"]["range"] == {"min": 0, "max": 10000, "min_strict": True}
    assert rows["offset"]["range"] == {"min": -1.0, "max": 1.0}
    assert rows["group"]["folder"] == ["Code"]
    assert rows["bindings"]["folder"] == ["Bindings"]
    assert "folder" not in rows["remap"]
    assert rows["descriptiveparm"]["hidden"] is True
    assert body["total"] == 11
    assert body["help_summary"] == "Runs a VEX snippet to modify attribute values."
    assert body["help_path"] == "/nodes/sop/attribwrangle"


def test_include_brings_help_or_hidden_parameters_into_a_lower_level(bench: Bench) -> None:
    summary = ok(lookup(bench, context="sop", type="attribwrangle", include=["help"]))
    assert summary["help_summary"] == "Runs a VEX snippet to modify attribute values."
    assert "parms" not in summary
    standard = ok(
        lookup(bench, context="sop", type="attribwrangle", detail="standard", include=["hidden"])
    )
    assert by_name(standard["parms"])["descriptiveparm"]["hidden"] is True
    assert "menu" not in by_name(standard["parms"])["class"]
    counted = ok(lookup(bench, context="sop", type="attribwrangle", include=["hidden"]))
    assert counted["parm_count"] == 11


def test_a_parm_filter_glob_keeps_matching_names_and_multiparm_instances(bench: Bench) -> None:
    body = ok(
        lookup(bench, context="sop", type="attribwrangle", detail="standard", parm_filter="bind*")
    )
    # The multiparm matched by its own name keeps all of its instance template.
    [row] = body["parms"]
    assert row["name"] == "bindings"
    assert len(row["instances"]["parms"]) == 2
    inner = ok(
        lookup(bench, context="sop", type="attribwrangle", detail="standard", parm_filter="*parm#")
    )
    [row] = inner["parms"]
    assert [item["name"] for item in row["instances"]["parms"]] == ["bindparm#"]
    assert inner["total"] == 1
    none = ok(lookup(bench, context="sop", type="attribwrangle", detail="full", parm_filter="zz*"))
    assert none["parms"] == [] and none["total"] == 0


def test_the_parameters_page_and_the_pages_make_the_whole_table(bench: Bench) -> None:
    whole = ok(lookup(bench, context="sop", type="attribwrangle", detail="full"))["parms"]
    seen: list[dict[str, Any]] = []
    page = None
    for _ in range(10):
        arguments: dict[str, Any] = {"context": "sop", "type": "attribwrangle", "detail": "full"}
        arguments["limit"] = 3
        if page:
            arguments["page"] = page
        body = ok(lookup(bench, **arguments))
        assert body["total"] == 11
        assert "changed" not in body
        seen.extend(body["parms"])
        page = body.get("next_page")
        if not page:
            break
    assert seen == whole
    assert "offset" not in bench.sent.calls[1]["arguments"]
    assert bench.sent.calls[2]["arguments"]["offset"] == 3


def test_a_page_after_the_type_changed_still_comes_back_and_says_so(
    bench: Bench, scene: Scene
) -> None:
    arguments = {"context": "sop", "type": "attribwrangle", "detail": "standard", "limit": 4}
    first = ok(lookup(bench, **arguments))
    # A default changes and no name does, as when an asset is loaded again.
    wrangle = scene.library["Sop"].types["attribwrangle"]
    run_over = wrangle.templates[0].children[1]
    assert run_over.name() == "class"
    run_over.default = 3
    second = ok(lookup(bench, **arguments, page=first["next_page"]))
    assert second["changed"] is True
    assert second["parms"][0]["name"] == "vex_strict"
    third = ok(lookup(bench, **arguments))
    assert by_name(third["parms"])["class"]["default"] == "vertex"


def raw_token(body: Any) -> str:
    text = body if isinstance(body, str) else json.dumps(body, separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


LOOKUP = {"context": "sop", "type": "attribwrangle", "detail": "standard"}
QUERY = query_of("type", LOOKUP)


def fields(**changed: Any) -> dict[str, Any]:
    body = {"v": 1, "m": "type", "s": "s-1", "o": 2, "d": "", "q": QUERY}
    body.update(changed)
    return {key: value for key, value in body.items() if value is not ...}


@pytest.mark.parametrize(
    "page",
    [
        "not a token",
        make_token(mode="type", session_id="s-other", offset=2, mark="", query=QUERY),
        make_token(mode="query", session_id="s-1", offset=2, mark="", query=QUERY),
        make_token(mode="type", session_id="s-1", offset=2, mark="", query="0" * 16),
        raw_token(fields(o=-1)),
        raw_token(fields(o=True)),
        raw_token(fields(o="2")),
        raw_token(fields(m="tree")),
        raw_token(fields(d=...)),
        raw_token(fields(extra=1)),
        raw_token(fields(s="x" * 200)),
        raw_token("[" * 20000),
        "A" * 601,
    ],
)
def test_a_page_token_that_is_not_for_this_lookup_is_refused(bench: Bench, page: str) -> None:
    result = lookup(bench, **LOOKUP, page=page)
    refused = error(result)
    assert refused["code"] == "BAD_CURSOR", text_of(result)
    assert bench.sent.calls == []


def test_a_page_token_for_another_type_or_level_is_refused(bench: Bench) -> None:
    first = ok(lookup(bench, **LOOKUP, limit=2))
    for other in ({"type": "xform"}, {"detail": "full"}, {"parm_filter": "*"}):
        refused = error(lookup(bench, **{**LOOKUP, **other}, limit=2, page=first["next_page"]))
        assert refused["code"] == "BAD_CURSOR"
    searched = error(lookup(bench, context="sop", query="wrangle", page=first["next_page"]))
    assert searched["details"]["token_mode"] == "type"


def test_a_well_formed_token_of_ours_is_taken(bench: Bench) -> None:
    body = ok(lookup(bench, **LOOKUP, page=raw_token(fields())))
    assert body["parms"][0]["name"] == "vex_numcount"
    assert body["changed"] is True


def test_the_longest_token_this_tool_writes_is_read_back() -> None:
    longest = make_token(
        mode="query",
        session_id="s" * node_type_tool.TOKEN_LENGTHS["s"],
        offset=10**9,
        mark="d" * node_type_tool.TOKEN_LENGTHS["d"],
        query="q" * node_type_tool.TOKEN_LENGTHS["q"],
    )
    assert len(longest) <= node_type_tool.MAX_TOKEN_CHARS
    assert node_type_tool.decode_token(longest)["o"] == 10**9


def test_a_bare_name_is_the_version_houdini_makes_and_says_so(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="copytopoints", detail="standard"))
    assert body["type"] == "copytopoints::2.0"
    assert body["resolved_from"] == "copytopoints"
    assert body["version"] == "2.0" and body["namespace"] is None
    assert body["is_asset"] is False and body["asset_library"] is None
    # No dialog script, so the labels come from the help page for this version.
    assert body["inputs"] == [
        {"index": 0, "label": "Geometry to Copy", "optional": False},
        {"index": 1, "label": "Target Points to Copy to", "optional": False},
    ]
    assert body["labels_from"] == "help"
    assert [row["name"] for row in body["parms"]] == ["sourcegroup"]
    helped = ok(lookup(bench, context="sop", type="copytopoints::2.0", include=["help"]))
    assert "resolved_from" not in helped
    assert helped["help_summary"].startswith("Copies geometry in the first input")


def test_a_namespaced_asset_reads_its_own_labels_and_help(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="com.example::tool::1.0", detail="full"))
    assert body["namespace"] == "com.example"
    assert body["version"] == "1.0"
    assert body["is_asset"] is True
    assert body["asset_library"] == "/shared/assets/tool.hda"
    assert body["inputs"] == [{"index": 0, "label": 'Mesh to "Fix"', "optional": False}]
    assert body["outputs"] == [{"index": 0, "label": "Kept"}, {"index": 1, "label": "Discarded"}]
    assert body["help_summary"] == "Tidies a mesh and splits off the pieces it cannot fix."
    assert body["help_path"] == "operator:Sop/com.example::tool::1.0"
    assert by_name(body["parms"])["tolerance"]["default"] == 0.01


def test_labels_come_from_the_help_page_when_the_type_carries_none(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="xform", detail="full"))
    assert body["inputs"] == [{"index": 0, "label": "Input Geometry", "optional": False}]
    # Links and line breaks in the first line are made plain.
    assert body["help_summary"] == "Transforms the input geometry in object space."


def test_a_type_with_no_help_page_says_so_with_nothing(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="wranglehelper", detail="full"))
    assert body["help_summary"] is None and body["help_path"] is None
    assert body["inputs"] == [{"index": 0, "label": None, "optional": False}]
    assert body["labels_from"] is None


def test_a_type_that_takes_any_number_of_inputs_lists_what_it_can_say(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="merge", detail="standard"))
    assert body["max_inputs"] == 9999
    assert body["inputs"] == [{"index": 0, "label": None, "optional": True}]
    assert body["more_inputs"] is True
    assert body["unordered_inputs"] is True


def test_a_deprecated_type_says_what_replaced_it(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="oldsmooth", detail="standard"))
    assert body["deprecated"] is True
    assert body["replaced_by"] == "smooth::2.0"


def test_a_hidden_type_can_still_be_read_and_says_it_is_hidden(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="attribwranglecore", detail="standard"))
    assert body["hidden"] is True


def test_reading_a_type_makes_no_node_cooks_nothing_and_leaves_no_undo(
    bench: Bench, scene: Scene
) -> None:
    before = [node.path() for node in scene.everything()]
    for detail in ("summary", "standard", "full"):
        ok(lookup(bench, context="sop", type="attribwrangle", detail=detail, include=["help"]))
    ok(lookup(bench, query="wrangle"))
    assert [node.path() for node in scene.everything()] == before
    assert all(node.cooks == 0 for node in scene.everything())
    assert scene.undos.labels == []
    assert scene.library["Sop"].types["attribwrangle"].read == 3


# Section: names that are not there


def test_a_misspelled_type_comes_back_with_the_closest_names(bench: Bench) -> None:
    result = lookup(bench, context="sop", type="atribwrangle")
    refused = error(result)
    assert refused["code"] == "TYPE_NOT_FOUND"
    assert refused["message"] == "no Sop type named atribwrangle"
    near = refused["details"]["did_you_mean"]
    assert near[0] == "attribwrangle"
    assert len(near) <= 5
    assert "hint" in refused
    text = text_of(result)
    assert text.startswith("TYPE_NOT_FOUND: no Sop type named atribwrangle")
    assert "attribwrangle" in text and "hint:" in text


def test_a_misspelled_base_name_finds_a_namespaced_type(bench: Bench) -> None:
    refused = error(lookup(bench, context="sop", type="tol"))
    assert "com.example::tool::1.0" in refused["details"]["did_you_mean"]


def test_a_type_in_another_context_says_which_contexts_have_it(bench: Bench) -> None:
    refused = error(lookup(bench, context="obj", type="attribwrangle"))
    assert refused["code"] == "TYPE_NOT_FOUND"
    assert refused["message"] == "attribwrangle is not in Object; it is in sop, lop"
    assert refused["details"]["found_in"] == [
        {"category": "Sop", "context": "sop"},
        {"category": "Lop", "context": "lop"},
    ]
    assert "found_in" in refused["hint"]


def test_without_a_context_a_type_must_belong_to_one(bench: Bench) -> None:
    assert ok(lookup(bench, type="xform"))["category"] == "Sop"
    assert ok(lookup(bench, type="Object/null"))["category"] == "Object"
    assert ok(lookup(bench, type="obj/null"))["category"] == "Object"
    refused = error(lookup(bench, type="null"))
    assert refused["code"] == "BAD_ARGUMENTS"
    assert [item["category"] for item in refused["details"]["found_in"]] == [
        "Sop",
        "Object",
        "Lop",
        "Driver",
    ]
    missing = error(lookup(bench, type="xfrom"))
    assert missing["code"] == "TYPE_NOT_FOUND"
    assert "Sop/xform" in missing["details"]["did_you_mean"]


@pytest.mark.parametrize(
    ("given", "category"),
    [("SOP", "Sop"), ("Object", "Object"), ("out", "Driver"), ("cop", "Cop2")],
)
def test_a_context_is_a_short_name_or_a_category_in_any_case(
    bench: Bench, given: str, category: str
) -> None:
    # This build has only the older compositing category, so cop reads that.
    body = ok(lookup(bench, context=given, query="vex" if category == "Cop2" else "null"))
    assert body["category"] == category
    assert {row["category"] for row in body["rows"]} == {category}


def test_an_unknown_context_is_refused_with_the_names_there_are(bench: Bench) -> None:
    refused = error(lookup(bench, context="sops", type="xform"))
    assert refused["code"] == "BAD_ARGUMENTS"
    assert "sop" in refused["details"]["did_you_mean"]
    assert "Sop" in refused["details"]["contexts"]


# Section: search


def test_a_search_ranks_exact_then_prefix_then_name_then_label_then_help(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", query="wrangle"))
    assert names(body) == ["wranglehelper", "attribwrangle", "volumewrangle"]
    assert body["rows"][1] == {
        "type": "attribwrangle",
        "category": "Sop",
        "label": "Attribute Wrangle",
        "one_line": "Runs a VEX snippet to modify attribute values.",
    }
    assert body["total"] == 3
    assert names(ok(lookup(bench, context="sop", query="Transform"))) == ["xform"]
    assert names(ok(lookup(bench, context="sop", query="voxel"))) == ["volumewrangle"]
    assert names(ok(lookup(bench, context="sop", query="vex"))) == [
        "attribwrangle",
        "volumewrangle",
    ]
    assert names(ok(lookup(bench, context="sop", query="copy points"))) == [
        "copytopoints::2.0",
        "copytopoints",
    ]
    exact = ok(lookup(bench, query="null"))
    assert [(row["category"], row["type"]) for row in exact["rows"]][:2] == [
        ("Sop", "null"),
        ("Object", "null"),
    ]
    assert "category" not in exact


def test_a_search_leaves_hidden_types_out_unless_asked(bench: Bench) -> None:
    assert "attribwranglecore" not in names(ok(lookup(bench, context="sop", query="wrangle")))
    shown = ok(lookup(bench, context="sop", query="wrangle", include=["hidden"]))
    row = next(row for row in shown["rows"] if row["type"] == "attribwranglecore")
    assert row["hidden"] is True
    assert names(shown)[-1] == "attribwranglecore"


def test_a_search_marks_deprecated_types_and_reads_an_assets_own_help(bench: Bench) -> None:
    [old] = ok(lookup(bench, context="sop", query="old smooth"))["rows"]
    assert old["deprecated"] is True
    [tool] = ok(lookup(bench, context="sop", query="example"))["rows"]
    assert tool["one_line"] == "Tidies a mesh and splits off the pieces it cannot fix."


def test_a_search_pages(bench: Bench) -> None:
    whole = names(ok(lookup(bench, query="null")))
    first = ok(lookup(bench, query="null", limit=2))
    assert names(first) == whole[:2]
    assert first["total"] == len(whole)
    second = ok(lookup(bench, query="null", limit=2, page=first["next_page"]))
    assert names(first) + names(second) == whole
    assert "next_page" not in second
    refused = error(lookup(bench, query="nul", limit=2, page=first["next_page"]))
    assert refused["code"] == "BAD_CURSOR"


def test_a_search_needs_a_word(bench: Bench) -> None:
    refused = error(lookup(bench, query="   "))
    assert refused["code"] == "BAD_ARGUMENTS"


# Section: arguments, listing and size


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        ({}, "type"),
        ({"type": "xform", "query": "x"}, "query"),
        ({"query": "x", "parm_filter": "*"}, "parm_filter"),
        ({"type": "xform", "parm_filter": "non_default"}, "parm_filter"),
        ({"type": "xform", "limit": 0}, "limit"),
        ({"type": "xform", "limit": 2001}, "limit"),
        ({"type": "xform", "include": ["menus"]}, "include.0"),
        ({"type": "xform", "detail": "everything"}, "detail"),
        ({"type": "xform", "path": "/obj"}, "path"),
    ],
)
def test_arguments_that_cannot_work_are_refused_before_anything_is_sent(
    bench: Bench, arguments: dict, argument: str
) -> None:
    refused = error(lookup(bench, **arguments))
    assert refused["code"] == "BAD_ARGUMENTS"
    assert refused["details"]["argument"] == argument
    assert bench.sent.calls == []


def test_the_tool_is_listed_last_as_read_only(bench: Bench) -> None:
    listed, _ = talk(bench.serve())
    assert listed.tools[-1].name == "hou_node_type"
    tool = listed.tools[-1]
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.open_world_hint is False


def test_a_large_table_is_summed_up_in_the_text_and_spilled_past_the_cap(
    bench: Bench, scene: Scene
) -> None:
    wrangle = scene.library["Sop"].types["attribwrangle"]
    for index in range(60):
        wrangle.templates.append(ParmTemplate("Float", f"Extra {index}", (0.0,), name=f"x{index}"))
    result = lookup(bench, context="sop", type="attribwrangle", detail="full")
    first = text_of(result).splitlines()[0]
    assert first == "hou_node_type Sop/attribwrangle: 4 inputs at most, 71 of 71 parameters"
    bench.config = replace(bench.config, spill_over_bytes=1024)
    body = ok(lookup(bench, context="sop", type="attribwrangle", detail="full"))
    assert set(body) == {"spilled", "trace"}
    assert Path(body["spilled"]["path"]).is_file()


def test_the_bridge_refuses_its_own_bad_arguments(scene: Scene) -> None:
    from nscr_houdini_mcp.bridge.errors import BridgeError
    from nscr_houdini_mcp.bridge.tools import ToolContext

    context = ToolContext(hou=scene.module())
    for arguments in (
        {"type": "xform", "offset": -1},
        {"type": "xform", "include": "help"},
        {"context": "sop"},
        {"type": "xform", "parm_filter": "non_default"},
    ):
        with pytest.raises(BridgeError) as raised:
            node_types.node_type(arguments, context)
        assert raised.value.code == "BAD_ARGUMENTS"


def test_near_types_compares_whole_names_and_base_names() -> None:
    types = dict.fromkeys(["attribwrangle", "com.example::tool::1.0", "copytopoints::2.0", "box"])
    assert node_types.near_types("atribwrangle", types)[0] == "attribwrangle"
    assert node_types.near_types("tool", types) == ["com.example::tool::1.0"]
    assert node_types.near_types("copytopoint", types) == ["copytopoints::2.0"]
    many = {f"attrib{index}": None for index in range(20)}
    assert len(node_types.near_types("attrib", many)) == 5


def test_the_help_index_is_made_once_and_again_when_the_pages_change(
    bench: Bench, hfs: Path, scene: Scene
) -> None:
    ok(lookup(bench, context="sop", query="wrangle"))
    index = node_types._HELP
    kept = index._pages
    ok(lookup(bench, context="sop", query="voxel"))
    assert index._pages is kept
    archive = hfs / "houdini" / "help" / "nodes.zip"
    with zipfile.ZipFile(archive, "a") as more:
        more.writestr("sop/merge.txt", '#internal: merge\n\n"""Joins its inputs."""\n')
    [row] = ok(lookup(bench, context="sop", query="joins"))["rows"]
    assert row["type"] == "merge"


def test_an_embedded_help_page_is_read_whole_for_its_labels() -> None:
    kind = NodeType("thing", "Sop", help_text='"""One line."""\n\n@inputs\n\nFirst:\n    x\n')
    page = node_types._HELP.page_for(None, kind, kind.category())
    assert page["summary"] == "One line."
    assert page["inputs"] == ["First"]


# Section: labels, ramps, search order, package help and page tokens


def test_a_label_written_without_quotes_is_read(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="kinefx::ragdollsolver", detail="standard"))
    assert body["inputs"] == [
        {"index": 0, "label": "Skeleton", "optional": False},
        {"index": 1, "label": "Constraint Geometry", "optional": True},
        {"index": 2, "label": "", "optional": True},
    ]
    assert body["outputs"] == [{"index": 0, "label": "Skeleton"}]
    assert body["labels_from"] == "dialog_script"


def test_help_labels_skip_directives_and_notes_and_stop_at_the_count(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="splitter", detail="standard"))
    # One input: the directive and the note are not inputs, and the stale
    # second heading is past what the type takes.
    assert body["inputs"] == [{"index": 0, "label": "Geometry", "optional": False}]
    assert body["outputs"] == [
        {"index": 0, "label": "Kept"},
        {"index": 1, "label": "Dropped"},
    ]
    assert body["labels_from"] == "help"


def test_the_description_says_help_labels_are_approximate(bench: Bench) -> None:
    listed, _ = talk(bench.serve())
    tool = listed.tools[-1]
    assert "help" in tool.description and "approximate" in tool.description


def test_a_ramp_says_how_many_points_it_starts_with_and_has_no_default(bench: Bench) -> None:
    body = ok(
        lookup(bench, context="sop", type="attribwrangle", detail="full", parm_filter="remap")
    )
    [row] = body["parms"]
    assert row["default_points"] == 2
    assert "default" not in row


def test_a_search_with_no_context_leaves_out_recipes_and_networks(bench: Bench) -> None:
    found = ok(lookup(bench, query="wrangle"))
    places = {row["category"] for row in found["rows"]}
    assert not places & {"Data", "VopNet"}
    # Named, the context is searched like any other, by the base name.
    [recipe] = ok(lookup(bench, context="Data", query="testscene_wrangle"))["rows"]
    assert recipe["type"] == "sidefx::recipe::lop::testscene_wrangle"
    assert ok(lookup(bench, context="VopNet", query="wranglenet"))["rows"][0]["type"] == (
        "wranglenet"
    )


def test_a_recipes_base_name_comes_from_the_type_not_its_name() -> None:
    kind = NodeType("sidefx::recipe::lop::testscene_wrangle", "Data")
    assert node_types._components(kind)[2] == "testscene_wrangle"
    assert node_types._base_name("sidefx::recipe::lop::testscene_wrangle") == "testscene_wrangle"
    assert node_types._split_name("labs::thing::1.0") == ("labs", "thing", "1.0")
    assert node_types._split_name("copytopoints::2.0") == ("", "copytopoints", "2.0")


def test_equally_close_matches_go_by_context_before_length(bench: Bench) -> None:
    rows = ok(lookup(bench, query="wrangle"))["rows"]
    prefixed = [(row["category"], row["type"]) for row in rows[:2]]
    # A shorter name in a later context does not jump ahead of a Sop.
    assert prefixed == [("Sop", "wranglehelper"), ("Lop", "wrangler")]


def test_a_packages_help_folder_gives_the_help_and_the_labels(bench: Bench) -> None:
    body = ok(lookup(bench, context="sop", type="labs::thing::1.0", detail="full"))
    assert body["help_summary"] == "Makes a thing from its input."
    assert body["help_path"] == "/nodes/sop/labs--thing-1.0"
    assert body["inputs"] == [{"index": 0, "label": "Source", "optional": False}]
    assert body["labels_from"] == "help"
    [row] = ok(lookup(bench, context="sop", query="labs thing"))["rows"]
    assert row["one_line"] == "Makes a thing from its input."
    # The shipped page wins over a package page for the same type.
    shipped = ok(lookup(bench, context="sop", type="attribwrangle", include=["help"]))
    assert shipped["help_summary"] == "Runs a VEX snippet to modify attribute values."


def test_a_package_page_added_later_is_found(bench: Bench, package: Path) -> None:
    ok(lookup(bench, context="sop", query="wrangle"))
    page = package / "help" / "nodes" / "sop" / "splitter2.txt"
    page.write_text('#internal: merge\n\n"""Joins in a package page."""\n', encoding="utf-8")
    [row] = ok(lookup(bench, context="sop", query="package page"))["rows"]
    assert row["type"] == "merge"


def test_include_order_and_a_searchs_detail_do_not_change_its_pages(bench: Bench) -> None:
    arguments = {"context": "sop", "type": "attribwrangle", "detail": "full", "limit": 3}
    first = ok(lookup(bench, **arguments, include=["help", "hidden"]))
    second = ok(lookup(bench, **arguments, include=["hidden", "help"], page=first["next_page"]))
    assert second["parms"][0]["name"] == "snippet"
    searched = ok(lookup(bench, query="null", limit=2))
    more = ok(lookup(bench, query="null", limit=2, detail="full", page=searched["next_page"]))
    assert more["rows"]
    assert query_of("query", {"query": "x", "detail": "full"}) == query_of("query", {"query": "x"})
    assert query_of("type", {"type": "x", "detail": "full"}) != query_of("type", {"type": "x"})


def test_a_search_reads_the_help_index_once_and_a_card_reuses_the_open_archive(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamped: list[Any] = []
    opened: list[Any] = []
    real_stamp, real_open = node_types._stamp, node_types._open_archive

    def stamp(path: Any) -> Any:
        stamped.append(path)
        return real_stamp(path)

    def open_archive(path: Any) -> Any:
        opened.append(path)
        return real_open(path)

    monkeypatch.setattr(node_types, "_stamp", stamp)
    monkeypatch.setattr(node_types, "_open_archive", open_archive)
    ok(lookup(bench, query="wrangle"))
    assert len(stamped) == 1
    for _ in range(2):
        body = ok(lookup(bench, context="sop", type="copytopoints", detail="standard"))
        assert body["inputs"][0]["label"] == "Geometry to Copy"
    # Opened once, when the index was made, and read from since.
    assert len(opened) == 1


# Section: bounds, stale definitions and help that cannot be read


def test_a_query_over_the_bounds_is_refused_before_anything_is_sent(bench: Bench) -> None:
    long = error(lookup(bench, query="w" * 201))
    assert long["code"] == "BAD_ARGUMENTS" and long["details"]["argument"] == "query"
    many = error(lookup(bench, query=" ".join(f"word{index}" for index in range(17))))
    assert many["code"] == "BAD_ARGUMENTS" and many["details"]["argument"] == "query"
    assert bench.sent.calls == []
    # The same word many times is one word.
    assert names(ok(lookup(bench, context="sop", query=" ".join(["wrangle"] * 20))))


def test_the_bridge_holds_the_same_bounds(scene: Scene) -> None:
    from nscr_houdini_mcp.bridge.errors import BridgeError
    from nscr_houdini_mcp.bridge.tools import ToolContext

    context = ToolContext(hou=scene.module())
    for query in ("w" * 201, " ".join(f"word{index}" for index in range(17))):
        with pytest.raises(BridgeError) as raised:
            node_types.node_type({"query": query}, context)
        assert raised.value.code == "BAD_ARGUMENTS"


def test_a_search_asked_to_stop_hands_back_what_it_has(scene: Scene) -> None:
    import threading

    from nscr_houdini_mcp.bridge.tools import ToolContext

    for index in range(600):
        kind = NodeType(f"filler{index:03d}", "Sop")
        scene.library["Sop"].types[kind.name()] = kind
    cancel = threading.Event()
    cancel.set()
    context = ToolContext(hou=scene.module(), cancel=cancel)
    result = node_types.node_type({"query": "filler", "limit": 5}, context)
    assert result["stopped"] is True
    assert "more" not in result and "next_offset" not in result
    assert 0 < result["total"] < 600
    whole = node_types.node_type({"query": "filler", "limit": 5}, ToolContext(hou=scene.module()))
    assert whole["total"] == 600 and "stopped" not in whole


def test_a_page_after_the_asset_library_changed_says_so(
    bench: Bench, scene: Scene, tmp_path: Path
) -> None:
    import os

    library = tmp_path / "tool.hda"
    library.write_bytes(b"an asset library")
    tool = scene.library["Sop"].types["com.example::tool::1.0"]
    tool._library = str(library)
    tool.templates.extend(
        ParmTemplate("Float", f"Extra {index}", (0.0,), name=f"x{index}") for index in range(3)
    )
    arguments = {"context": "sop", "type": "com.example::tool::1.0", "detail": "standard"}
    first = ok(lookup(bench, **arguments, limit=2))
    stamp = library.stat().st_mtime_ns + 5_000_000_000
    os.utime(library, ns=(stamp, stamp))
    second = ok(lookup(bench, **arguments, limit=2, page=first["next_page"]))
    assert second["changed"] is True


def test_a_definition_that_went_stale_is_read_again_once(bench: Bench, scene: Scene) -> None:
    from fake_hou import ObjectWasDeleted

    wrangle = scene.library["Sop"].types["attribwrangle"]
    real = wrangle.parmTemplateGroup
    failures = [ObjectWasDeleted("the definition was reloaded")]

    def once() -> Any:
        if failures:
            raise failures.pop()
        return real()

    wrangle.parmTemplateGroup = once  # type: ignore[method-assign]
    body = ok(lookup(bench, context="sop", type="attribwrangle", detail="standard"))
    assert body["total"] == 9


def test_a_definition_that_stays_broken_is_a_coded_error_not_an_empty_card(
    bench: Bench, scene: Scene
) -> None:
    from fake_hou import ObjectWasDeleted

    wrangle = scene.library["Sop"].types["attribwrangle"]

    def broken() -> Any:
        raise ObjectWasDeleted("the definition is gone")

    wrangle.parmTemplateGroup = broken  # type: ignore[method-assign]
    refused = error(lookup(bench, context="sop", type="attribwrangle", detail="standard"))
    assert refused["code"] == "NODE_NOT_FOUND"
    assert refused["details"]["exception"] == "ObjectWasDeleted"


def test_a_category_that_cannot_list_its_types_is_not_a_missing_type(
    bench: Bench, scene: Scene
) -> None:
    from fake_hou import OperationFailed

    def broken() -> Any:
        raise OperationFailed("the category could not be read")

    scene.library["Sop"].nodeTypes = broken  # type: ignore[method-assign]
    refused = error(lookup(bench, context="sop", type="attribwrangle"))
    assert refused["code"] == "TOOL_FAILED"
    assert refused["details"]["exception"] == "OperationFailed"


def test_help_that_cannot_be_read_says_so_and_is_tried_again_later(
    bench: Bench, hfs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = hfs / "houdini" / "help" / "nodes.zip"
    archive.write_bytes(b"not an archive")
    card = ok(lookup(bench, context="sop", type="xform", detail="full"))
    assert card["help_available"] is False
    assert card["help_summary"] is None
    searched = ok(lookup(bench, context="sop", query="wrangle"))
    assert searched["help_available"] is False
    # A package's pages are still read.
    labs = ok(lookup(bench, context="sop", type="labs::thing::1.0", include=["help"]))
    assert labs["help_summary"] == "Makes a thing from its input."
    made: list[Any] = []
    real = node_types._open_archive

    def counted(path: Any) -> Any:
        made.append(path)
        return real(path)

    monkeypatch.setattr(node_types, "_open_archive", counted)
    ok(lookup(bench, context="sop", query="wrangle"))
    assert made == []
    monkeypatch.setattr(node_types, "HELP_RETRY_S", 0.0)
    ok(lookup(bench, context="sop", query="wrangle"))
    assert len(made) == 1


def test_a_menu_whose_items_toggle_keeps_its_mask_and_no_token_is_asked_for(
    bench: Bench,
) -> None:
    # The stand in's `defaultValueAsString` crashes the way Houdini does, so
    # reading every menu at full proves it is never asked.
    body = ok(lookup(bench, context="sop", type="attribwrangle", detail="full"))
    rows = by_name(body["parms"])
    assert rows["channels"]["default"] == 511
    assert rows["channels"]["menu_toggles"] is True
    assert [item["token"] for item in rows["channels"]["menu"]][:2] == ["tx", "ty"]
    assert rows["class"]["default"] == "point" and "menu_toggles" not in rows["class"]


def test_help_that_is_there_is_not_marked(bench: Bench) -> None:
    card = ok(lookup(bench, context="sop", type="xform", detail="full"))
    assert "help_available" not in card
    summary = ok(lookup(bench, context="sop", type="xform"))
    assert "help_available" not in summary


def test_a_missing_help_archive_is_not_available(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "no-help"
    empty.mkdir()
    monkeypatch.setenv("HFS", str(empty))
    card = ok(lookup(bench, context="sop", type="xform", include=["help"]))
    assert card["help_available"] is False
