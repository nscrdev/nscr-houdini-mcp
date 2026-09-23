"""Objects and arrays sent as JSON text are read into them before the schema check.

Some clients send a nested argument as a string holding its JSON. Every tool
argument the schema wants as an object or an array takes that form too. Where
the schema takes a string as well as an object, only text holding a JSON
object is read, and a plain string is left as it came.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp.tools import base
from nscr_houdini_mcp.tools.base import ToolSpec, inputs, structured_kinds
from nscr_houdini_mcp.tools.registry import TOOLS
from test_server import Stage, serve, talk

# One well formed value for every argument that takes an object or an array.
SAMPLES: dict[tuple[str, str], Any] = {
    ("hou_inspect", "paths"): ["/obj/geo1"],
    ("hou_inspect", "include"): ["wires", "flags"],
    ("hou_node_type", "include"): ["help"],
    ("hou_compare", "candidate"): {"source": "file", "path": "/abs/a.png"},
    ("hou_compare", "adjust"): {"dx": 0.1, "dy": 0.0, "scale": 1.0},
    ("hou_compare", "region"): [0, 0, 0.5, 0.5],
    ("hou_compare", "regions"): {"face": [0.2, 0.2, 0.4, 0.4]},
    ("hou_outputs", "filter"): {"kind": "capture"},
    ("hou_capture", "resolution"): [640, 480],
    ("hou_capture", "frames"): [1, 12],
    ("hou_capture", "region"): [0, 0, 1, 1],
    ("hou_capture", "camera"): {"position": [0, 1, 5], "look_at": [0, 0, 0]},
    ("hou_compare", "detail_crops"): ["keys", "face"],
}


def takes_structure(schema: Any) -> bool:
    """Whether a property schema takes an object or an array, read here on its own terms."""
    if not isinstance(schema, dict):
        return False
    declared = schema.get("type") or []
    kinds = {declared} if isinstance(declared, str) else set(declared)
    if "properties" in schema:
        kinds.add("object")
    if "items" in schema:
        kinds.add("array")
    branches = [*schema.get("anyOf", []), *schema.get("oneOf", [])]
    return bool(kinds & {"object", "array"}) or any(takes_structure(b) for b in branches)


def echoing(spec: ToolSpec) -> ToolSpec:
    """The tool with its real schema and a body that hands back what it was given."""

    def echo(call: Any) -> Mapping[str, Any]:
        return {"given": call.arguments}

    return ToolSpec(
        name=spec.name, description=spec.description, input_schema=spec.input_schema, handler=echo
    )


def given(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content["given"]


def test_every_object_or_array_argument_is_known() -> None:
    """A new object or array argument is added to the samples, and so to the tests."""
    expected = {
        (spec.name, name)
        for spec in TOOLS
        for name, schema in spec.input_schema["properties"].items()
        if takes_structure(schema)
    }
    found = {(spec.name, name) for spec in TOOLS for name in spec.structured}
    assert expected == set(SAMPLES)
    assert found == expected


def test_an_any_of_or_one_of_argument_is_read_too() -> None:
    spec = ToolSpec(
        name="test_branches",
        description="test only",
        input_schema=inputs(
            {
                "either": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                "one": {"oneOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [1]}]},
            }
        ),
        handler=lambda call: {"given": call.arguments},
    )
    assert spec.structured == {"either": (dict,), "one": (list,)}
    _, [result] = talk(
        serve(Stage([]), tools=(spec,)),
        ("test_branches", {"either": '{"a": 1}', "one": "[1, 2]"}),
    )
    assert given(result) == {"either": {"a": 1}, "one": [1, 2]}


def test_a_string_or_array_argument_reads_only_a_list_of_strings() -> None:
    [spec] = [spec for spec in TOOLS if spec.name == "hou_compare"]
    assert spec.structured["detail_crops"] == (list,)
    for sent, arrived in (
        ('["keys"]', ["keys"]),
        (' ["a", "b"]', ["a", "b"]),
        ("auto", "auto"),
        ("[1, 2]", "[1, 2]"),
        ("[not json", "[not json"),
    ):
        assert spec.decode({"detail_crops": sent})["detail_crops"] == arrived


@pytest.mark.parametrize(
    "text",
    ["[NaN, 1]", "[Infinity, 1]", "[-Infinity, 1]", "[" * 100000 + "]" * 100000],
    ids=["nan", "infinity", "minus infinity", "nested deep"],
)
def test_text_json_cannot_read_safely_is_refused_by_the_schema(text: str) -> None:
    [spec] = [spec for spec in TOOLS if spec.name == "hou_capture"]
    _, [result] = talk(
        serve(Stage([]), tools=(echoing(spec),)), ("hou_capture", {"resolution": text})
    )
    assert result.is_error
    error = result.structured_content["error"]
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["argument"] == "resolution"


def test_a_failure_while_reading_arguments_is_a_coded_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    [spec] = [spec for spec in TOOLS if spec.name == "hou_capture"]

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("broken")

    monkeypatch.setattr(base, "decoded", broken)
    _, [result] = talk(
        serve(Stage([]), tools=(echoing(spec),)), ("hou_capture", {"resolution": "[1, 2]"})
    )
    assert result.is_error
    assert result.structured_content["error"]["code"] == "TOOL_FAILED"


@pytest.mark.parametrize(("tool", "argument"), sorted(SAMPLES))
def test_each_object_or_array_argument_takes_its_json_text(tool: str, argument: str) -> None:
    [spec] = [spec for spec in TOOLS if spec.name == tool]
    value = SAMPLES[(tool, argument)]
    _, [as_text, as_value] = talk(
        serve(Stage([]), tools=(echoing(spec),)),
        (tool, {argument: json.dumps(value)}),
        (tool, {argument: value}),
    )
    assert given(as_text)[argument] == value
    assert given(as_value)[argument] == value


def test_text_that_is_not_the_wanted_json_is_refused_by_the_schema() -> None:
    [spec] = [spec for spec in TOOLS if spec.name == "hou_capture"]
    _, [broken, wrong_kind] = talk(
        serve(Stage([]), tools=(echoing(spec),)),
        ("hou_capture", {"resolution": "[640, 480"}),
        ("hou_capture", {"resolution": '{"w": 640}'}),
    )
    for result in (broken, wrong_kind):
        assert result.is_error
        error = result.structured_content["error"]
        assert error["code"] == "BAD_ARGUMENTS"
        assert error["details"]["argument"] == "resolution"


def test_a_string_or_object_argument_reads_only_an_object_from_text() -> None:
    [spec] = [spec for spec in TOOLS if spec.name == "hou_capture"]
    assert spec.structured["camera"] == (dict,)
    camera = {"position": [0, 1, 5], "look_at": [0, 0, 0]}
    _, results = talk(
        serve(Stage([]), tools=(echoing(spec),)),
        ("hou_capture", {"camera": json.dumps(camera)}),
        ("hou_capture", {"camera": "  " + json.dumps(camera)}),
        ("hou_capture", {"camera": camera}),
        ("hou_capture", {"camera": "/obj/cam1"}),
        ("hou_capture", {"camera": "[1, 2]"}),
        ("hou_capture", {"camera": "{not json"}),
    )
    as_text, padded, as_value, plain, bracketed, broken = (given(r)["camera"] for r in results)
    assert as_text == padded == as_value == camera
    # Plain strings, and text that is not a JSON object, stay as they came.
    assert plain == "/obj/cam1"
    assert bracketed == "[1, 2]"
    assert broken == "{not json"


@pytest.mark.parametrize(
    ("schema", "kinds"),
    [
        ({"type": "object"}, (dict,)),
        ({"type": "array"}, (list,)),
        ({"type": ["array", "object"]}, (dict, list)),
        ({"properties": {"a": {}}}, (dict,)),
        ({"items": {"type": "number"}}, (list,)),
        ({"type": ["string", "object"]}, (dict,)),
        ({"type": ["string", "array"]}, (list,)),
        ({"anyOf": [{"type": "string"}, {"items": {}}]}, (list,)),
        ({"oneOf": [{"enum": ["a"]}, {"properties": {}}]}, (dict,)),
        ({"type": "string"}, ()),
        ({}, ()),
        ({"enum": ["a", "b"]}, ()),
    ],
)
def test_which_schemas_take_json_text(schema: dict[str, Any], kinds: tuple[type, ...]) -> None:
    assert set(structured_kinds(schema)) == set(kinds)


# Section: over stdio, as a client sends it

ECHO_SERVER = """
from nscr_houdini_mcp.server import build_server
from nscr_houdini_mcp.tools.base import ToolSpec, inputs

def echo(call):
    return {"given": call.arguments}

spec = ToolSpec(
    name="test_echo",
    description="test only",
    input_schema=inputs({"box": {"type": "array"}, "pick": {"type": "object"}}),
    handler=echo,
)
build_server((spec,)).run(transport="stdio")
"""


def test_json_text_arguments_over_stdio(tmp_path: Path) -> None:
    env = {**os.environ, "NSCR_MCP_HOME": str(tmp_path / "home")}
    child = subprocess.Popen(
        [sys.executable, "-c", ECHO_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    assert child.stdin is not None and child.stdout is not None

    def send(message: dict) -> None:
        child.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        child.stdin.flush()

    def answer(wanted: int) -> dict:
        while True:
            message = json.loads(child.stdout.readline())
            if message.get("id") == wanted:
                return message

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            }
        )
        answer(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "test_echo",
                    "arguments": {"box": "[0, 0, 1, 1]", "pick": '{"source": "file"}'},
                },
            }
        )
        result = answer(2)["result"]
        assert not result.get("isError"), result
        assert result["structuredContent"]["given"] == {
            "box": [0, 0, 1, 1],
            "pick": {"source": "file"},
        }
    finally:
        child.stdin.close()
        child.wait(timeout=10)
