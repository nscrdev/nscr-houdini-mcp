"""`hou_inspect` through the server, down to a bridge reading a stand in scene.

Every call goes the whole way: the server checks the arguments and the page
token, the router sends the call, and a real dispatcher runs the bridge's
`node.inspect` against the stand in for `hou`. So what is checked here is the
path a session takes, short of a real Houdini, which the integration tests
cover.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fake_hou import Item, Node, ObjectWasDeleted, Scene
from nscr_houdini_mcp.bridge import client, tools
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.handlers import default_registry
from nscr_houdini_mcp.bridge.identity import Identity
from nscr_houdini_mcp.tools import inspect as inspect_tool
from nscr_houdini_mcp.tools.inspect import make_token, query_of
from test_server import talk, text_of
from test_tools_sessions import Bench

TYPES = ("geo", "null", "cam", "box", "xform", "attribwrangle", "attribcreate", "file")


class Through:
    """Sends each call to a real dispatcher over the stand in scene."""

    def __init__(self, scene: Scene) -> None:
        self.identity = Identity(session_id="s-1", kind="hython", alias="w1", hou=scene.module())
        self.dispatcher = Dispatcher(
            default_registry(),
            lock=threading.Lock(),
            kind="hython",
            session_id="s-1",
            identity=self.identity,
            hou=scene.module(),
            wait_s=5.0,
            timeout_s=10.0,
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(self, session: client.Session, tool: str, **rest: Any) -> client.Answer:
        self.calls.append({"tool": tool, **rest})
        envelope = Envelope(
            tool=tool,
            arguments=rest.get("arguments") or {},
            session_id=rest.get("session_id"),
            scene_epoch=rest.get("scene_epoch"),
            operation_id=rest.get("operation_id"),
            wait_s=rest.get("wait_s"),
            timeout_s=rest.get("timeout_s"),
            skip_if_busy=bool(rest.get("skip_if_busy")),
        )
        return client.Answer(200, dict(self.dispatcher.dispatch(envelope).payload), {})


def build(scene: Scene) -> dict[str, Node]:
    """A small network with something in it for every kind of read."""
    obj = scene.node("/obj")
    geo = obj.createNode("geo", "geo1")
    obj.createNode("cam", "cam1")
    box = geo.createNode("box", "box1")
    box.parm("sizex").set(2.0)
    box.note = "base shape"
    box.tint = (0.2, 0.4, 0.6)
    move = geo.createNode("xform", "transform1")
    move.setInput(0, box)
    move.parm("tx").setExpression("$F*2")
    move.parm("ty").setExpression('npoints("../box1")')
    out = geo.createNode("null", "OUT")
    out.setInput(0, move)
    out.flags = {"Display", "Render"}
    wrangle = geo.createNode("attribwrangle", "attribwrangle1")
    wrangle.setInput(0, box)
    wrangle.parm("snippet").set("@P.y += 1;\n@Cd = 1;")
    make = geo.createNode("attribcreate", "attribcreate1")
    make.parm("numattr").set(2)
    make.parm("name1").set("foo")
    reader = geo.createNode("file", "file1")
    reader.parm("file").set("$HIP/geo/x.bgeo")
    geo.stickies.append(Item(geo, "__stickynote1", text="build the base here"))
    geo.boxes.append(Item(geo, "__netbox1", text="inputs", nodes=(box,)))
    for node in scene.everything():
        node.dirty = True
        node.cooks = 0
    return {
        "geo": geo,
        "box": box,
        "move": move,
        "out": out,
        "wrangle": wrangle,
        "make": make,
        "reader": reader,
    }


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene(types=TYPES)
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture
def nodes(scene: Scene) -> dict[str, Node]:
    return build(scene)


@pytest.fixture
def bench(tmp_path: Path, scene: Scene) -> Bench:
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1")
    made.sent = Through(scene)  # type: ignore[assignment]
    return made


def inspect(bench: Bench, **arguments: Any) -> Any:
    _, [result] = talk(bench.serve(), ("hou_inspect", arguments))
    return result


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, text_of(result)
    return result.structured_content


def error(result: Any) -> dict[str, Any]:
    assert result.is_error is True, text_of(result)
    return result.structured_content["error"]


def paths(body: dict[str, Any]) -> list[str]:
    return [row["path"] for row in body["rows"]]


def cooks(scene: Scene) -> dict[str, int]:
    return {node.path(): node.cooks for node in scene.everything()}


def by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["n"]: row for row in rows}


# Section: tree


def test_a_tree_is_a_summary_of_the_top_by_default(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench))
    assert body["mode"] == "tree"
    assert paths(body) == ["/mat", "/obj", "/out", "/stage"]
    obj = body["rows"][1]
    assert obj == {"path": "/obj", "type": "network", "kids": 2, "not_cooked": True}
    assert body["total"] == 4
    assert "next_page" not in body
    [sent] = bench.sent.calls
    assert sent["tool"] == "node.inspect"
    assert sent["operation_id"] is None


def test_a_deeper_tree_sorts_by_path_and_says_which_names_are_the_defaults(
    bench: Bench, nodes: dict
) -> None:
    body = ok(inspect(bench, path="/obj", depth=2))
    assert paths(body) == [
        "/obj/cam1",
        "/obj/geo1",
        "/obj/geo1/OUT",
        "/obj/geo1/attribcreate1",
        "/obj/geo1/attribwrangle1",
        "/obj/geo1/box1",
        "/obj/geo1/file1",
        "/obj/geo1/transform1",
    ]
    rows = {row["path"]: row for row in body["rows"]}
    assert rows["/obj/geo1/box1"]["auto"] is True
    # Named after its type's description rather than its type name.
    assert rows["/obj/geo1/transform1"]["auto"] is True
    assert "auto" not in rows["/obj/geo1/OUT"]
    assert rows["/obj/geo1"]["kids"] == 6
    # A summary is identity and counts: no wires, flags or positions.
    assert "flags" not in rows["/obj/geo1/OUT"] and "in" not in rows["/obj/geo1/OUT"]
    assert "boxes" not in body and "notes" not in body


def test_a_standard_tree_adds_flags_wires_boxes_and_notes(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench, path="/obj/geo1", detail="standard"))
    rows = {row["path"]: row for row in body["rows"]}
    assert rows["/obj/geo1/OUT"]["flags"] == ["display", "render"]
    assert rows["/obj/geo1/OUT"]["in"] == ["/obj/geo1/transform1"]
    assert rows["/obj/geo1/box1"]["comment"] == "base shape"
    assert body["boxes"] == [
        {"path": "/obj/geo1/__netbox1", "comment": "inputs", "nodes": ["/obj/geo1/box1"]}
    ]
    assert body["notes"] == [{"path": "/obj/geo1/__stickynote1", "text": "build the base here"}]
    assert "changed" not in rows["/obj/geo1/box1"]


def test_a_full_tree_adds_changed_parameter_counts_and_positions(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench, path="/obj/geo1", detail="full"))
    rows = {row["path"]: row for row in body["rows"]}
    assert rows["/obj/geo1/box1"]["changed"] == 1
    assert rows["/obj/geo1/transform1"]["changed"] == 1
    assert "changed" not in rows["/obj/geo1/OUT"]
    assert rows["/obj/geo1/box1"]["pos"] == [1.0, -2.0]
    assert rows["/obj/geo1/box1"]["cooks"] == 0
    assert "cook_ms" not in rows["/obj/geo1/box1"]
    assert body["boxes"][0]["size"] == [2.5, 2.5]


def test_include_brings_one_item_into_a_summary(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench, path="/obj/geo1", include=["wires"]))
    rows = {row["path"]: row for row in body["rows"]}
    assert rows["/obj/geo1/transform1"]["in"] == ["/obj/geo1/box1"]
    assert "flags" not in rows["/obj/geo1/OUT"]
    assert "boxes" not in body


# Section: pages


def test_a_tree_pages_after_the_last_path_it_returned(bench: Bench, nodes: dict) -> None:
    whole = paths(ok(inspect(bench, path="/obj", depth=2)))
    first = ok(inspect(bench, path="/obj", depth=2, limit=3))
    assert paths(first) == whole[:3]
    # A page that stopped early does not know how many rows there are in all.
    assert "total" not in first
    second = ok(inspect(bench, path="/obj", depth=2, limit=3, page=first["next_page"]))
    third = ok(inspect(bench, path="/obj", depth=2, limit=3, page=second["next_page"]))
    assert paths(first) + paths(second) + paths(third) == whole
    assert "next_page" not in third
    assert "scene_changed" not in second and "scene_changed" not in third


def test_a_page_after_an_edit_still_comes_back_and_says_the_scene_changed(
    bench: Bench, scene: Scene, nodes: dict
) -> None:
    first = ok(inspect(bench, path="/obj", depth=2, limit=3))
    nodes["geo"].createNode("null", "added")
    second = ok(inspect(bench, path="/obj", depth=2, limit=3, page=first["next_page"]))
    assert second["scene_changed"] is True
    assert paths(second)[0] > paths(first)[-1]
    assert "/obj/geo1/added" in paths(second)


def test_a_page_after_the_scene_was_replaced_says_so(bench: Bench, nodes: dict) -> None:
    first = ok(inspect(bench, path="/obj", depth=2, limit=3))
    bench.sent.identity.bump("loaded")
    second = ok(inspect(bench, path="/obj", depth=2, limit=3, page=first["next_page"]))
    assert second["scene_changed"] is True
    assert second["trace"]["scene_epoch"] == 1


def test_a_page_token_for_another_mode_is_refused(bench: Bench, nodes: dict) -> None:
    first = ok(inspect(bench, path="/obj", depth=2, limit=3))
    refused = error(inspect(bench, mode="find", pattern="*", page=first["next_page"]))
    assert refused["code"] == "BAD_CURSOR"
    assert refused["details"]["token_mode"] == "tree"
    assert "next_page" in refused["hint"]


def test_a_page_token_for_another_query_is_refused(bench: Bench, nodes: dict) -> None:
    first = ok(inspect(bench, path="/obj/geo1", limit=2))
    for other in ({"path": "/obj"}, {"path": "/obj/geo1", "detail": "full"}):
        refused = error(inspect(bench, limit=2, page=first["next_page"], **other))
        assert refused["code"] == "BAD_CURSOR"
    # The same query spelled with a trailing slash is the same query.
    assert ok(inspect(bench, path="/obj/geo1/", limit=2, page=first["next_page"]))["rows"]


def raw_token(body: Any) -> str:
    text = body if isinstance(body, str) else json.dumps(body, separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def fields(**changed: Any) -> dict[str, Any]:
    body = {"v": 1, "m": "tree", "s": "s-1", "e": 0, "k": "/obj", "d": "", "q": QUERY}
    body.update(changed)
    return {key: value for key, value in body.items() if value is not ...}


QUERY = query_of("tree", {}, None)


@pytest.mark.parametrize(
    "page",
    [
        "not a token",
        "@@@@",
        make_token(mode="tree", session_id="s-other", epoch=0, last="/obj", mark="", query=QUERY),
        raw_token(fields(k="obj")),
        raw_token(fields(v=True)),
        raw_token(fields(e=True)),
        raw_token(fields(e="0")),
        raw_token(fields(e=...)),
        raw_token(fields(d=...)),
        raw_token(fields(extra=1)),
        raw_token(fields(s="x" * 200)),
        raw_token("[" * 700 + "]" * 700),
        raw_token("[" * 20000),
        "A" * 1025,
        raw_token([1, 2]),
    ],
)
def test_a_page_token_that_is_not_ours_is_refused(bench: Bench, nodes: dict, page: str) -> None:
    result = inspect(bench, page=page)
    refused = error(result)
    assert refused["code"] == "BAD_CURSOR", text_of(result)
    assert bench.sent.calls == []


def test_a_well_formed_token_of_ours_is_taken(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench, page=raw_token(fields(k="/mat"))))
    assert paths(body) == ["/obj", "/out", "/stage"]


def test_the_longest_token_this_server_writes_is_read_back() -> None:
    longest = inspect_tool.make_token(
        mode="parms",
        session_id="s" * inspect_tool.TOKEN_LENGTHS["s"],
        epoch=10**9,
        last="/" + "k" * (inspect_tool.TOKEN_LENGTHS["k"] - 1),
        mark="d" * inspect_tool.TOKEN_LENGTHS["d"],
        query="q" * inspect_tool.TOKEN_LENGTHS["q"],
    )
    assert len(longest) <= inspect_tool.MAX_TOKEN_CHARS
    body = inspect_tool.decode_token(longest)
    assert body["k"] == "/" + "k" * (inspect_tool.TOKEN_LENGTHS["k"] - 1)


def test_a_big_page_is_carried_whole(bench: Bench, scene: Scene, nodes: dict) -> None:
    for index in range(1100):
        nodes["geo"].createNode("null", f"n{index:04d}")
    body = ok(inspect(bench, path="/obj/geo1", limit=2000))
    assert len(body["rows"]) == 1106
    assert "lossy" not in body


def test_a_very_large_read_is_spilled_to_a_file(bench: Bench, nodes: dict) -> None:
    bench.config = replace(bench.config, spill_over_bytes=1024)
    body = ok(inspect(bench, path="/obj", depth=2, detail="full"))
    assert set(body) == {"spilled", "trace"}
    assert Path(body["spilled"]["path"]).is_file()


# Section: node


def test_a_node_summary_is_identity_flags_inputs_and_errors(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench, mode="node", path="/obj/geo1/OUT"))
    assert body["nodes"] == [
        {
            "path": "/obj/geo1/OUT",
            "type": "null",
            "flags": ["display", "render"],
            "in": ["/obj/geo1/transform1"],
            "not_cooked": True,
        }
    ]


def test_a_standard_node_read_adds_the_parameter_pane_and_the_wiring(
    bench: Bench, nodes: dict
) -> None:
    nodes["box"].warnings_now = ["a warning from before"]
    [entry] = ok(inspect(bench, mode="node", path="/obj/geo1/box1", detail="standard"))["nodes"]
    assert entry["comment"] == "base shape"
    assert entry["color"] == [0.2, 0.4, 0.6]
    assert entry["out"] == ["/obj/geo1/transform1", "/obj/geo1/attribwrangle1"]
    assert entry["warnings"] == ["a warning from before"]
    # Only what differs from its default, and no positions or user data.
    assert entry["parms"] == [{"n": "size", "v": [2.0, 1.0, 1.0]}]
    assert "pos" not in entry and "user" not in entry
    [moved] = ok(inspect(bench, mode="node", path="/obj/geo1/transform1", detail="standard"))[
        "nodes"
    ]
    assert moved["inputs"] == [{"i": 0, "from": "/obj/geo1/box1", "label": "First Input"}]


def test_a_full_node_read_has_every_parameter_expressions_code_and_user_data(
    bench: Bench, nodes: dict
) -> None:
    nodes["wrangle"].user = {"note": "x" * 3000}
    nodes["wrangle"].addSpareParm("amount", default=0.5)
    [entry] = ok(inspect(bench, mode="node", path="/obj/geo1/attribwrangle1", detail="full"))[
        "nodes"
    ]
    rows = by_name(entry["parms"])
    assert rows["snippet"] == {"n": "snippet", "code": "vex", "v": "@P.y += 1;\n@Cd = 1;"}
    assert rows["class"] == {"n": "class", "v": "point"}
    assert rows["bindings"] == {"n": "bindings", "v": False}
    assert "go" not in rows
    assert rows["amount"]["spare"] == {"label": "Amount", "type": "Float"}
    assert entry["auto"] is True
    assert entry["user"]["note"].endswith("...") and len(entry["user"]["note"]) == 2003
    [moved] = ok(inspect(bench, mode="node", path="/obj/geo1/transform1", detail="full"))["nodes"]
    translate = by_name(moved["parms"])["t"]
    assert translate["expr"] == {"tx": "$F*2", "ty": 'npoints("../box1")'}
    assert translate["lang"] == "hscript"
    assert translate["not_cooked"] == "calls npoints()"
    assert "v" not in translate


def test_a_standard_read_shows_code_as_a_line_count(bench: Bench, nodes: dict) -> None:
    [entry] = ok(inspect(bench, mode="node", path="/obj/geo1/attribwrangle1", detail="standard"))[
        "nodes"
    ]
    assert by_name(entry["parms"])["snippet"] == {"n": "snippet", "code": "vex", "lines": 2}


def test_include_expressions_or_code_at_a_summary_brings_only_those_parameters(
    bench: Bench, nodes: dict
) -> None:
    [moved] = ok(inspect(bench, mode="node", path="/obj/geo1/transform1", include=["expressions"]))[
        "nodes"
    ]
    assert [row["n"] for row in moved["parms"]] == ["t"]
    [wrangle] = ok(inspect(bench, mode="node", path="/obj/geo1/attribwrangle1", include=["code"]))[
        "nodes"
    ]
    assert wrangle["parms"] == [{"n": "snippet", "code": "vex", "v": "@P.y += 1;\n@Cd = 1;"}]


def test_a_batch_answers_every_path_and_puts_each_error_in_its_own_entry(
    bench: Bench, nodes: dict
) -> None:
    body = ok(
        inspect(
            bench,
            mode="node",
            paths=["/obj/geo1/box1", "/obj/geo1/boxx", "/obj/geo1/transform1/tx"],
        )
    )
    good, missing, parm = body["nodes"]
    assert good["path"] == "/obj/geo1/box1" and "error" not in good
    assert missing["error"]["code"] == "NODE_NOT_FOUND"
    assert missing["error"]["did_you_mean"][0] == "/obj/geo1/box1"
    assert parm["error"]["code"] == "PATH_NOT_A_NODE"


def test_a_single_missing_path_fails_the_call_with_the_closest_paths(
    bench: Bench, nodes: dict
) -> None:
    refused = error(inspect(bench, mode="node", path="/obj/box1"))
    assert refused["code"] == "NODE_NOT_FOUND"
    # One level too high still finds it, by its name further down.
    assert "/obj/geo1/box1" in refused["details"]["did_you_mean"]
    assert len(refused["details"]["did_you_mean"]) <= 5


def test_a_parameter_path_where_a_node_is_wanted_says_which_node_holds_it(
    bench: Bench, nodes: dict
) -> None:
    refused = error(inspect(bench, mode="node", path="/obj/geo1/box1/size"))
    assert refused["code"] == "PATH_NOT_A_NODE"
    assert refused["details"]["node"] == "/obj/geo1/box1"
    assert "mode parms" in refused["hint"]
    tree = error(inspect(bench, path="/obj/geo1/box1/sizex"))
    assert tree["code"] == "PATH_NOT_A_NODE"


# Section: parms


def test_parms_keep_the_ones_that_differ_from_their_defaults(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench, mode="parms", path="/obj/geo1/transform1"))
    [entry] = body["nodes"]
    assert entry["path"] == "/obj/geo1/transform1"
    assert entry["not_cooked"] is True
    [translate] = entry["parms"]
    # Expressions always come with a parameter read, safe or not.
    assert translate["expr"] == {"tx": "$F*2", "ty": 'npoints("../box1")'}
    assert translate["not_cooked"] == "calls npoints()"
    assert body["total"] == 1


def test_parms_all_and_a_glob(bench: Bench, nodes: dict) -> None:
    every = ok(inspect(bench, mode="parms", path="/obj/geo1/box1", parm_filter="all"))
    assert [row["n"] for row in every["nodes"][0]["parms"]] == ["scale", "size", "t"]
    globbed = ok(inspect(bench, mode="parms", path="/obj/geo1/box1", parm_filter="size*"))
    assert [row["n"] for row in globbed["nodes"][0]["parms"]] == ["size"]
    by_component = ok(inspect(bench, mode="parms", path="/obj/geo1/box1", parm_filter="tz"))
    assert [row["n"] for row in by_component["nodes"][0]["parms"]] == ["t"]


def test_a_parameter_path_reads_that_one_parameter(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench, mode="parms", path="/obj/geo1/box1/sizex"))
    assert body["nodes"] == [{"path": "/obj/geo1/box1", "not_cooked": True, "parms": [
        {"n": "size", "v": [2.0, 1.0, 1.0]}
    ]}]  # fmt: skip


def test_a_multiparm_keeps_its_instances_nested(bench: Bench, nodes: dict) -> None:
    body = ok(inspect(bench, mode="parms", path="/obj/geo1/attribcreate1"))
    [row] = body["nodes"][0]["parms"]
    assert row == {"n": "numattr", "v": 2, "inst": [[{"n": "name1", "v": "foo"}], []]}
    every = ok(inspect(bench, mode="parms", path="/obj/geo1/attribcreate1", parm_filter="all"))
    rows = by_name(every["nodes"][0]["parms"])
    assert rows["numattr"]["inst"] == [
        [{"n": "name1", "v": "foo"}, {"n": "value1", "v": 0.0}],
        [{"n": "name2", "v": ""}, {"n": "value2", "v": 0.0}],
    ]
    named = ok(inspect(bench, mode="parms", path="/obj/geo1/attribcreate1", parm_filter="num*"))
    assert len(named["nodes"][0]["parms"][0]["inst"][1]) == 2


def test_parms_page_across_a_batch(bench: Bench, nodes: dict) -> None:
    arguments = {
        "mode": "parms",
        "paths": ["/obj/geo1/box1", "/obj/geo1/missing", "/obj/geo1/attribcreate1"],
        "parm_filter": "all",
        "limit": 2,
    }
    first = ok(inspect(bench, **arguments))
    assert first["total"] == 6
    assert [entry["path"] for entry in first["nodes"]] == ["/obj/geo1/attribcreate1"]
    second = ok(inspect(bench, **arguments, page=first["next_page"]))
    third = ok(inspect(bench, **arguments, page=second["next_page"]))
    # The path that was not there is an entry of its own, in its place by path.
    assert [entry["path"] for entry in third["nodes"]] == ["/obj/geo1/box1", "/obj/geo1/missing"]
    assert third["nodes"][1]["error"]["code"] == "NODE_NOT_FOUND"
    assert "next_page" not in third
    names = [
        (entry["path"], row["n"])
        for page in (first, second, third)
        for entry in page["nodes"]
        for row in entry.get("parms", [])
    ]
    assert names == [
        ("/obj/geo1/attribcreate1", "group"),
        ("/obj/geo1/attribcreate1", "numattr"),
        ("/obj/geo1/box1", "scale"),
        ("/obj/geo1/box1", "size"),
        ("/obj/geo1/box1", "t"),
    ]
    assert all("error" not in entry for entry in first["nodes"] + second["nodes"])


def test_a_string_carries_its_text_and_what_it_expands_to(
    bench: Bench, scene: Scene, nodes: dict
) -> None:
    [row] = ok(inspect(bench, mode="parms", path="/obj/geo1/file1"))["nodes"][0]["parms"]
    assert row == {"n": "file", "v": "$HIP/geo/x.bgeo", "ev": "/Users/somebody/scenes/geo/x.bgeo"}
    nodes["reader"].parm("file").set('$HIP/`npoints("../box1")`.bgeo')
    [held] = ok(inspect(bench, mode="parms", path="/obj/geo1/file1"))["nodes"][0]["parms"]
    assert held == {"n": "file", "v": '$HIP/`npoints("../box1")`.bgeo', "not_cooked": "backtick"}
    assert nodes["box"].cooks == 0
    [cooked] = ok(inspect(bench, mode="parms", path="/obj/geo1/file1", evaluate=True))["nodes"][0][
        "parms"
    ]
    assert cooked["ev"] == "/Users/somebody/scenes/8.bgeo"
    assert nodes["box"].cooks == 1


def test_a_reference_to_a_safe_parameter_is_read_and_one_to_a_cook_is_not(
    bench: Bench, nodes: dict
) -> None:
    safe = nodes["out"].addSpareParm("follow")
    safe.setExpression('ch("../transform1/tx") + 1')
    cooking = nodes["out"].addSpareParm("count")
    cooking.setExpression('ch("../transform1/ty")')
    body = ok(inspect(bench, mode="parms", path="/obj/geo1/OUT"))
    rows = by_name(body["nodes"][0]["parms"])
    assert rows["follow"]["v"] == 3.0
    assert rows["follow"]["expr"] == {"follow": 'ch("../transform1/tx") + 1'}
    assert rows["count"]["not_cooked"] == 'ch("../transform1/ty"): calls npoints()'
    assert nodes["box"].cooks == 0


def test_locked_and_keyed_parameters_say_so(bench: Bench, nodes: dict) -> None:
    nodes["box"].parm("scale").locked = True
    nodes["box"].parm("tx").setExpression("bezier()")
    body = ok(inspect(bench, mode="parms", path="/obj/geo1/box1", parm_filter="all"))
    rows = by_name(body["nodes"][0]["parms"])
    assert rows["t"]["keys"] is True
    assert rows["scale"]["lock"] is True


# Section: cooking


def test_nothing_is_cooked_or_evaluated_unsafely_by_any_read_without_evaluate(
    bench: Bench, scene: Scene, nodes: dict
) -> None:
    before = cooks(scene)
    for detail in ("summary", "standard", "full"):
        ok(inspect(bench, path="/", depth=8, detail=detail))
        ok(inspect(bench, mode="node", paths=[node.path() for node in nodes.values()]))
        ok(inspect(bench, mode="node", path="/obj/geo1/transform1", detail=detail))
        ok(inspect(bench, mode="parms", path="/obj/geo1/transform1", parm_filter="all"))
    assert cooks(scene) == before
    assert nodes["move"].parm("ty").evaluations == 0


def test_errors_from_an_earlier_cook_are_marked_stale_once_the_node_changes(
    bench: Bench, nodes: dict
) -> None:
    nodes["out"].fails_with = ["cannot find the input"]
    with pytest.raises(Exception, match="errors"):
        nodes["out"].cook()
    [fresh] = ok(inspect(bench, mode="node", path="/obj/geo1/OUT", detail="standard"))["nodes"]
    assert fresh["err"] == 1
    assert fresh["errors"] == ["cannot find the input"]
    assert "stale" not in fresh and "not_cooked" not in fresh
    nodes["out"].dirty = True
    [stale] = ok(inspect(bench, mode="node", path="/obj/geo1/OUT", detail="standard"))["nodes"]
    assert stale["errors"] == ["cannot find the input"]
    assert "changed after its last cook" in stale["stale"]


def test_evaluate_cooks_what_it_reads_and_clears_the_marks(bench: Bench, nodes: dict) -> None:
    body = ok(
        inspect(bench, mode="node", path="/obj/geo1/transform1", detail="full", evaluate=True)
    )
    [entry] = body["nodes"]
    assert "not_cooked" not in entry and "stale" not in entry
    translate = by_name(entry["parms"])["t"]
    assert translate["v"] == [2.0, 8.0, 0.0]
    assert "not_cooked" not in translate
    assert nodes["move"].cooks == 1
    assert nodes["box"].cooks == 1
    assert entry["cooks"] == 1 and entry["cook_ms"] == 1.5


def test_evaluate_never_cooks_a_render_driver(bench: Bench, scene: Scene, nodes: dict) -> None:
    driver = scene.node("/out").createNode("null", "rop1")
    driver.dirty = True
    [entry] = ok(inspect(bench, mode="node", path="/out/rop1", evaluate=True))["nodes"]
    assert entry["not_cooked"] is True
    assert driver.cooks == 0


# Section: find and selection


def test_find_by_name_by_path_and_by_type(bench: Bench, nodes: dict) -> None:
    assert paths(ok(inspect(bench, mode="find", pattern="box*"))) == ["/obj/geo1/box1"]
    assert paths(ok(inspect(bench, mode="find", pattern="/obj/*/OUT"))) == ["/obj/geo1/OUT"]
    assert paths(ok(inspect(bench, mode="find", type="null"))) == ["/obj/geo1/OUT"]
    assert paths(ok(inspect(bench, mode="find", type="Sop/attrib*", path="/obj/geo1"))) == [
        "/obj/geo1/attribcreate1",
        "/obj/geo1/attribwrangle1",
    ]
    first = ok(inspect(bench, mode="find", pattern="*1", limit=2))
    second = ok(inspect(bench, mode="find", pattern="*1", limit=2, page=first["next_page"]))
    assert paths(first) == ["/obj/cam1", "/obj/geo1"]
    assert paths(second) == ["/obj/geo1/attribcreate1", "/obj/geo1/attribwrangle1"]


def test_find_needs_something_to_look_for(bench: Bench, nodes: dict) -> None:
    refused = error(inspect(bench, mode="find"))
    assert refused["code"] == "BAD_ARGUMENTS"
    assert bench.sent.calls == []


def test_a_worker_has_no_selection_and_says_so(bench: Bench, scene: Scene, nodes: dict) -> None:
    scene.selected = [nodes["box"]]
    body = ok(inspect(bench, mode="selection"))
    assert body["rows"] == []
    assert body["note"] == "no selection outside a GUI"


def test_a_gui_reads_its_selection_sorted_by_path(scene: Scene, nodes: dict) -> None:
    scene.selected = [nodes["out"], nodes["box"]]
    context = tools.ToolContext(hou=scene.module(), kind="gui", session_id="s-1")
    body = tools.inspect({"mode": "selection"}, context)
    assert [row["path"] for row in body["rows"]] == ["/obj/geo1/OUT", "/obj/geo1/box1"]


def test_a_read_asked_to_stop_hands_back_what_it_has_and_where_to_go_on(
    scene: Scene, nodes: dict
) -> None:
    stop = threading.Event()
    stop.set()
    context = tools.ToolContext(hou=scene.module(), session_id="s-1", cancel=stop)
    body = tools.inspect({"mode": "tree", "path": "/obj", "depth": 2}, context)
    assert body["rows"] == []
    assert body["stopped"] is True
    assert body["more"] is True and body["last"] == "/"


# Section: arguments


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        ({"path": "/obj", "paths": ["/obj"]}, "paths"),
        ({"mode": "node", "path": "obj/geo1"}, "path"),
        ({"mode": "node"}, "path"),
        ({"mode": "tree", "paths": ["/obj"]}, "paths"),
        ({"mode": "selection", "path": "/obj"}, "path"),
        ({"depth": 9}, "depth"),
        ({"limit": 0}, "limit"),
        ({"limit": 2001}, "limit"),
        ({"mode": "node", "paths": [f"/obj/n{index}" for index in range(51)]}, "paths"),
    ],
)
def test_arguments_that_cannot_work_are_refused_before_anything_is_sent(
    bench: Bench, nodes: dict, arguments: dict, argument: str
) -> None:
    refused = error(inspect(bench, **arguments))
    assert refused["code"] == "BAD_ARGUMENTS"
    assert refused["details"]["argument"] == argument
    assert bench.sent.calls == []


def test_the_tool_is_listed_after_hou_scene_without_a_read_only_hint(bench: Bench) -> None:
    listed, _ = talk(bench.serve())
    names = [tool.name for tool in listed.tools]
    assert names[names.index("hou_scene") + 1] == "hou_inspect"
    [tool] = [tool for tool in listed.tools if tool.name == "hou_inspect"]
    # A read that evaluates may cook, so it is not promised to be read only.
    assert tool.annotations is None or tool.annotations.read_only_hint is None


# Section: values that could only be had by cooking


def risky(nodes: dict[str, Node], scene: Scene) -> dict[str, Any]:
    """Every way a value can pull on a cook, set up on the stand in."""
    source = scene.node("/obj").createNode("null", "chopper")
    source.dirty = True
    source.cooks = 0
    nodes["box"].parm("tx").override = source
    nodes["box"].parm("ty").setKeyframes(
        [(1.0, "len(hou.node('../OUT').geometry().points())", "python"), (10.0, "2", "python")]
    )
    out = nodes["out"]
    out.addSpareParm("safe", default=1.0)
    out.addSpareParm("trick").setExpression('ch(strcat("un", "safe")) + 0 * strlen("ch(\'safe\')")')
    out.addSpareParm("py").setExpression("hou.node('../box1').cook()", "python")
    out.addSpareParm("chain").setExpression('ch("py") + 1')
    nodes["reader"].parm("file").set("$NPT/x.bgeo")
    out.addSpareParm("via", kind="String", default="").setExpression('chs("../file1/file")')
    return {"source": source}


def test_every_value_that_could_cook_is_withheld_and_says_why(
    bench: Bench, scene: Scene, nodes: dict
) -> None:
    source = risky(nodes, scene)["source"]
    before = cooks(scene)
    for detail in ("summary", "standard", "full"):
        ok(inspect(bench, mode="node", paths=["/obj/geo1/box1", "/obj/geo1/OUT"], detail=detail))
        ok(inspect(bench, path="/obj", depth=3, detail=detail))
    box = by_name(
        ok(inspect(bench, mode="parms", path="/obj/geo1/box1", parm_filter="all"))["nodes"][0][
            "parms"
        ]
    )
    out = by_name(
        ok(inspect(bench, mode="parms", path="/obj/geo1/OUT", parm_filter="all"))["nodes"][0][
            "parms"
        ]
    )
    reader = ok(inspect(bench, mode="parms", path="/obj/geo1/file1"))["nodes"][0]["parms"][0]
    assert cooks(scene) == before
    risky_parms = ("trick", "py", "chain", "via")
    assert [nodes["out"].parm(name).evaluations for name in risky_parms] == [0, 0, 0, 0]
    # A channel operator's export: withheld, and never taken for a default.
    assert box["t"]["not_cooked"] == "override"
    assert box["t"]["keys"] is True
    assert "v" not in box["t"]
    assert out["trick"]["not_cooked"] == "ch() of a worked out name"
    assert out["py"]["not_cooked"] == "python"
    assert out["chain"]["not_cooked"] == 'ch("py"): python'
    assert out["safe"] == {"n": "safe", "v": 1.0}
    assert reader == {"n": "file", "v": "$NPT/x.bgeo", "not_cooked": "variable $NPT"}
    assert out["via"]["not_cooked"] == 'chs("../file1/file"): variable $NPT'
    assert source.cooks == 0


def test_a_parameter_with_keyframes_is_withheld_for_its_keyframes(
    bench: Bench, scene: Scene, nodes: dict
) -> None:
    nodes["box"].parm("ty").setKeyframes([(1.0, "1", "hscript"), (10.0, "5", "hscript")])
    [row] = [
        row
        for row in ok(inspect(bench, mode="parms", path="/obj/geo1/box1"))["nodes"][0]["parms"]
        if row["n"] == "t"
    ]
    assert row == {"n": "t", "not_cooked": "keyframes", "keys": True}


def test_with_evaluate_the_same_values_are_read(bench: Bench, scene: Scene, nodes: dict) -> None:
    source = risky(nodes, scene)["source"]
    box = by_name(
        ok(inspect(bench, mode="parms", path="/obj/geo1/box1", evaluate=True, parm_filter="all"))[
            "nodes"
        ][0]["parms"]
    )
    assert "not_cooked" not in box["t"]
    assert box["t"]["v"][0] == 0.5
    assert source.cooks == 1


# Section: paging every mode


def test_node_entries_are_sorted_and_paged(bench: Bench, nodes: dict) -> None:
    asked = ["/obj/geo1/file1", "/obj/geo1/box1", "/obj/geo1/OUT", "/obj/geo1/box1/", "/obj/cam1"]
    first = ok(inspect(bench, mode="node", paths=asked, limit=2))
    assert [entry["path"] for entry in first["nodes"]] == ["/obj/cam1", "/obj/geo1/OUT"]
    assert first["total"] == 4
    second = ok(inspect(bench, mode="node", paths=asked, limit=2, page=first["next_page"]))
    assert [entry["path"] for entry in second["nodes"]] == ["/obj/geo1/box1", "/obj/geo1/file1"]
    assert "next_page" not in second


def test_a_find_with_a_small_limit_stops_walking_once_the_page_is_full(
    bench: Bench, scene: Scene, nodes: dict
) -> None:
    for index in range(300):
        nodes["geo"].createNode("null", f"z{index:03d}")
    scene.listed = 0
    body = ok(inspect(bench, mode="find", pattern="box*", limit=1))
    assert paths(body) == ["/obj/geo1/box1"]
    assert "next_page" in body
    assert scene.listed < 20


def test_a_walk_cut_short_by_the_scan_bound_goes_on_from_where_it_stopped(
    bench: Bench, nodes: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = paths(ok(inspect(bench, mode="find", pattern="*", limit=100)))
    monkeypatch.setattr(tools, "MAX_NODES_SCANNED", 3)
    seen: list[str] = []
    page = None
    for _ in range(20):
        body = ok(
            inspect(bench, mode="find", pattern="*", limit=100, **({"page": page} if page else {}))
        )
        assert body.get("truncated") is True or "next_page" not in body
        seen.extend(paths(body))
        page = body.get("next_page")
        if not page:
            break
    assert seen == whole


def test_near_misses_for_a_batch_are_gathered_once(
    bench: Bench, nodes: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    gathered: list[str] = []
    below = tools._NearIndex.below

    def counted(self: Any, above: str, root: Any) -> Any:
        if above not in self._below:
            gathered.append(above)
        return below(self, above, root)

    monkeypatch.setattr(tools._NearIndex, "below", counted)
    body = ok(inspect(bench, mode="node", paths=[f"/obj/geo1/boxx{index}" for index in range(10)]))
    assert all(entry["error"]["code"] == "NODE_NOT_FOUND" for entry in body["nodes"])
    assert gathered == ["/obj/geo1"]


def test_near_misses_rank_by_the_last_part_and_look_further_down(
    bench: Bench, scene: Scene, nodes: dict
) -> None:
    for name in ("a1", "a2", "a3", "a4", "a5", "a6"):
        scene.node("/obj").createNode("null", name)
    scene.node("/obj").createNode("geo", "mysub").createNode("null", "inner")
    refused = error(inspect(bench, mode="node", path="/obj/inner"))
    assert refused["details"]["did_you_mean"] == ["/obj/mysub/inner"]


def test_a_multiparm_with_many_instances_is_cut_and_says_so(bench: Bench, nodes: dict) -> None:
    nodes["make"].parm("numattr").set(250)
    body = ok(inspect(bench, mode="parms", path="/obj/geo1/attribcreate1", parm_filter="all"))
    row = by_name(body["nodes"][0]["parms"])["numattr"]
    assert row["v"] == 250 and row["inst_total"] == 250 and row["truncated"] is True
    assert len(row["inst"]) == 200


def test_a_multiparm_named_on_its_own_pages_through_its_instances(
    bench: Bench, nodes: dict
) -> None:
    nodes["make"].parm("numattr").set(250)
    arguments = {"mode": "parms", "path": "/obj/geo1/attribcreate1/numattr", "limit": 100}
    names: list[str] = []
    starts: list[int] = []
    page = None
    for _ in range(5):
        body = ok(inspect(bench, **arguments, **({"page": page} if page else {})))
        [row] = body["nodes"][0]["parms"]
        assert row["v"] == 250 and body["total"] == 250
        starts.append(row["inst_from"])
        names.extend(group[0]["n"] for group in row["inst"])
        page = body.get("next_page")
        if not page:
            break
    assert starts == [1, 101, 201]
    assert names == [f"name{index}" for index in range(1, 251)]


def test_boxes_and_notes_come_whole_up_to_their_own_bound(
    bench: Bench, nodes: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    geo = nodes["geo"]
    geo.stickies.extend(Item(geo, f"__stickynote{index}", text="n") for index in range(2, 5))
    body = ok(inspect(bench, path="/obj/geo1", detail="standard", limit=1))
    assert len(body["notes"]) == 4 and "notes_truncated" not in body
    monkeypatch.setattr(tools, "MAX_NETWORK_ITEMS", 2)
    body = ok(inspect(bench, path="/obj/geo1", detail="standard", limit=1))
    assert len(body["notes"]) == 2 and body["notes_truncated"] is True
    assert "boxes_truncated" not in body
    later = ok(inspect(bench, path="/obj/geo1", detail="standard", limit=1, page=body["next_page"]))
    assert "notes" not in later


# Section: batches


def test_a_node_gone_part_way_through_a_parameter_batch_is_one_entry(
    bench: Bench, nodes: dict
) -> None:
    box = nodes["box"]
    real = box.path
    calls = {"count": 0}

    def path() -> str:
        calls["count"] += 1
        if calls["count"] > 1:
            raise ObjectWasDeleted("gone")
        return real()

    box.path = path  # type: ignore[method-assign]
    body = ok(inspect(bench, mode="parms", paths=["/obj/geo1/box1", "/obj/geo1/transform1"]))
    entries = {entry["path"]: entry for entry in body["nodes"]}
    assert entries["/obj/geo1/box1"]["error"]["code"] == "NODE_NOT_FOUND"
    assert entries["/obj/geo1/transform1"]["parms"][0]["n"] == "t"


def test_every_node_asked_for_has_one_entry_and_one_row_per_parameter(
    bench: Bench, nodes: dict
) -> None:
    body = ok(
        inspect(
            bench,
            mode="parms",
            paths=[
                "/obj/cam1",
                "/obj/geo1/box1/tx",
                "/obj/geo1/box1/ty",
                "/obj/geo1/box1/t",
                "/obj/cam1/",
            ],
        )
    )
    assert body["nodes"] == [
        {"path": "/obj/cam1", "not_cooked": True, "parms": []},
        {"path": "/obj/geo1/box1", "not_cooked": True, "parms": [{"n": "t", "v": [0.0, 0.0, 0.0]}]},
    ]


# Section: stopping


def test_an_evaluating_tree_stops_between_rows_when_asked(scene: Scene, nodes: dict) -> None:
    stop = threading.Event()
    nodes["make"].after_cook = stop.set
    context = tools.ToolContext(hou=scene.module(), session_id="s-1", cancel=stop)
    body = tools.inspect({"path": "/obj/geo1", "evaluate": True}, context)
    assert [row["path"] for row in body["rows"]] == ["/obj/geo1/OUT", "/obj/geo1/attribcreate1"]
    assert body["stopped"] is True and body["more"] is True
    assert body["last"] == "/obj/geo1/attribcreate1"
    assert nodes["wrangle"].cooks == 0 and nodes["reader"].cooks == 0


def test_a_read_that_times_out_is_asked_to_stop(scene: Scene, nodes: dict) -> None:
    through = Through(scene)
    nodes["box"].after_cook = lambda: time.sleep(0.5)
    reply = through.dispatcher.dispatch(
        Envelope(
            tool="node.inspect",
            arguments={"mode": "tree", "path": "/obj/geo1", "evaluate": True},
            timeout_s=0.1,
        )
    )
    assert reply.payload["error"]["code"] == "TIMEOUT"
    deadline = time.monotonic() + 5.0
    while through.dispatcher.running is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert through.dispatcher.running is None
    assert nodes["reader"].cooks == 0


def test_numbers_an_export_that_has_not_cooked_may_drive_are_withheld(
    bench: Bench, scene: Scene, nodes: dict
) -> None:
    source = scene.node("/obj").createNode("null", "exporter")
    source.export_flag = True
    source.cooks = 0
    nodes["box"].dependents_now = [source]
    nodes["box"].parm("tx").override = source
    body = ok(inspect(bench, mode="parms", path="/obj/geo1/box1", parm_filter="all"))
    rows = by_name(body["nodes"][0]["parms"])
    # Until the channel operator has cooked, nothing says which numbers it drives.
    assert rows["size"]["not_cooked"] == "override"
    assert rows["scale"]["not_cooked"] == "override"
    assert source.cooks == 0
    source.cooks = 1
    nodes["box"].parm("tx").override = None
    rows = by_name(
        ok(inspect(bench, mode="parms", path="/obj/geo1/box1", parm_filter="all"))["nodes"][0][
            "parms"
        ]
    )
    assert rows["size"] == {"n": "size", "v": [2.0, 1.0, 1.0]}
