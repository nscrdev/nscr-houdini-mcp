from __future__ import annotations

import pytest

from nscr_houdini_mcp.bridge.envelope import (
    EnvelopeError,
    error_payload,
    ok_payload,
    parse_envelope,
)


def test_a_full_envelope_is_read_field_for_field() -> None:
    envelope = parse_envelope(
        {
            "token": "abc",
            "session_id": "s1",
            "scene_epoch": 3,
            "operation_id": "op1",
            "tool": "bridge.ping",
            "arguments": {"echo": "hello"},
        }
    )
    assert envelope.tool == "bridge.ping"
    assert envelope.arguments == {"echo": "hello"}
    assert envelope.token == "abc"
    assert envelope.session_id == "s1"
    assert envelope.scene_epoch == 3
    assert envelope.operation_id == "op1"


def test_only_the_tool_name_is_required() -> None:
    envelope = parse_envelope({"tool": "bridge.ping"})
    assert envelope.arguments == {}
    assert envelope.token is None
    assert envelope.scene_epoch is None


def test_the_arguments_are_copied_out_of_the_request() -> None:
    given = {"echo": 1}
    envelope = parse_envelope({"tool": "bridge.ping", "arguments": given})
    given["echo"] = 2
    assert envelope.arguments == {"echo": 1}


@pytest.mark.parametrize(
    "payload",
    [
        "not an object",
        {},
        {"tool": ""},
        {"tool": 7},
        {"tool": "Bridge.Ping"},
        {"tool": "bridge ping"},
        {"tool": "bridge.ping", "arguments": ["echo"]},
        {"tool": "bridge.ping", "scene_epoch": "3"},
        {"tool": "bridge.ping", "scene_epoch": True},
        {"tool": "bridge.ping", "session_id": 7},
        {"tool": "bridge.ping", "operaton_id": "op1"},
    ],
)
def test_a_request_that_cannot_be_read_is_refused(payload: object) -> None:
    with pytest.raises(EnvelopeError):
        parse_envelope(payload)


def test_a_misspelled_field_says_what_the_fields_are() -> None:
    with pytest.raises(EnvelopeError) as raised:
        parse_envelope({"tool": "bridge.ping", "operaton_id": "op1"})
    assert raised.value.code == "BAD_ENVELOPE"
    assert raised.value.details["unknown"] == ["operaton_id"]
    assert "operation_id" in raised.value.details["known"]


def test_both_payload_shapes_say_whether_the_call_ran() -> None:
    assert ok_payload({"pong": True}, timing_ms=1.23456) == {
        "ok": True,
        "data": {"pong": True},
        "timing_ms": 1.235,
    }
    assert error_payload("TOOL_UNKNOWN", "no tool", details={"tools": []}) == {
        "ok": False,
        "error": {"code": "TOOL_UNKNOWN", "message": "no tool", "details": {"tools": []}},
    }
