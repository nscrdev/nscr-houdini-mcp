"""Result shapes: errors a client can act on from the text alone, and spill."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from nscr_houdini_mcp.bridge.errors import CODES as BRIDGE_CODES
from nscr_houdini_mcp.results import (
    CODES,
    HINTS,
    MIRROR_CHARS,
    PREVIEW_CHARS,
    SERVER_CODES,
    CallError,
    Spill,
    error_result,
    ok_result,
    reap_spill,
)

TRACE = {"session_id": "s-1", "alias": "w1", "scene_epoch": 2}


def text_of(result) -> str:
    [block] = result.content
    return block.text


def test_every_code_has_a_hint() -> None:
    assert set(CODES) == set(BRIDGE_CODES) | set(SERVER_CODES)
    assert sorted(set(CODES) - set(HINTS)) == []


@pytest.mark.parametrize("code", sorted(CODES))
def test_every_code_is_an_error_whose_text_stands_alone(code: str) -> None:
    error = CallError(code, CODES[code], details={"session_id": "s-1"})
    result = error_result(error, TRACE)
    assert result.is_error is True
    text = text_of(result)
    first = text.splitlines()[0]
    assert first == f"{code}: {CODES[code]}"
    assert f"hint: {HINTS[code]}" in text
    assert 'details: {"session_id":"s-1"}' in text
    body = result.structured_content
    assert body["error"]["code"] == code
    assert body["error"]["message"] == CODES[code]
    assert body["error"]["hint"] == HINTS[code]
    assert body["error"]["details"] == {"session_id": "s-1"}
    assert body["trace"] == TRACE


@pytest.mark.parametrize("code", sorted(BRIDGE_CODES))
def test_every_bridge_refusal_becomes_the_same_code(code: str) -> None:
    payload = {
        "ok": False,
        "error": {"code": code, "message": "said by the bridge"},
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": 5,
        "operation_id": "op-1",
    }
    error = CallError.from_reply(payload)
    assert error.code == code
    assert error.message == "said by the bridge"
    assert error.hint == HINTS[code]
    assert error.trace == {
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": 5,
        "operation_id": "op-1",
    }
    result = error_result(error, error.trace)
    assert result.is_error is True
    assert text_of(result).startswith(f"{code}: said by the bridge")


def test_a_hint_the_bridge_wrote_wins_over_the_default() -> None:
    error = CallError.from_reply(
        {"ok": False, "error": {"code": "SESSION_BUSY", "message": "m", "hint": "wait 3 s"}}
    )
    assert error.hint == "wait 3 s"


def test_a_reply_with_no_error_object_is_a_bad_reply() -> None:
    error = CallError.from_reply({"ok": False})
    assert error.code == "BAD_REPLY"
    assert error.hint == HINTS["BAD_REPLY"]


def test_long_details_are_cut_in_the_text_and_kept_whole_in_the_structure() -> None:
    details = {"candidates": ["x" * 50] * 100}
    result = error_result(CallError("SESSION_AMBIGUOUS", "m", details=details))
    text = text_of(result)
    assert "more characters in structuredContent" in text
    assert len(text) < 2000
    assert result.structured_content["error"]["details"] == details


def test_an_error_before_any_session_still_carries_an_empty_trace() -> None:
    result = error_result(CallError("NO_SESSION", "no Houdini session is live"))
    assert result.structured_content["trace"] == {
        "session_id": None,
        "alias": None,
        "scene_epoch": None,
    }


def test_a_small_result_is_mirrored_whole_in_the_text() -> None:
    result = ok_result({"pong": True}, TRACE)
    assert result.is_error in (None, False)
    body = result.structured_content
    assert body == {"pong": True, "trace": TRACE}
    assert json.loads(text_of(result)) == body


def test_a_large_result_gets_a_summary_line_instead_of_a_second_copy() -> None:
    data = {"rows": ["r" * 40] * 100}
    result = ok_result(data, TRACE, tool="hou_inspect")
    text = text_of(result)
    assert len(json.dumps(data)) > MIRROR_CHARS
    assert text.startswith("hou_inspect: ")
    assert "rows" in text
    assert '"scene_epoch":2' in text
    assert result.structured_content["rows"] == data["rows"]


# Section: spill


class Moment:
    def __call__(self) -> datetime:
        return datetime(2026, 9, 22, 14, 3, 5)


def test_a_result_over_the_cap_is_written_to_a_dated_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    spill = Spill(home / "spill", 1024, clock=Moment())
    data = {"rows": [{"path": f"/obj/geo{i}", "points": i} for i in range(200)]}
    result = ok_result(data, TRACE, spill=spill, tool="hou_inspect")

    body = result.structured_content
    assert set(body) == {"spilled", "trace"}
    assert body["trace"] == TRACE
    spilled = body["spilled"]
    path = Path(spilled["path"])
    assert path.parent == home / "spill" / "2026-09-22"
    assert path.name.startswith("140305-hou_inspect-")
    assert path.suffix == ".json"
    assert path.is_relative_to(home)

    written = path.read_bytes()
    assert spilled["bytes"] == len(written) > 1024
    assert spilled["sha256"] == hashlib.sha256(written).hexdigest()
    assert json.loads(written.decode("utf-8")) == {**data, "trace": TRACE}
    assert set(spilled) == {"path", "bytes", "sha256"}

    # The preview is for a reader of the text alone, so it is there only.
    text = text_of(result)
    assert str(path) in text
    assert "over the cap of 1024" in text
    assert text.endswith(written.decode("utf-8")[:PREVIEW_CHARS])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_the_spill_folder_and_file_are_private(tmp_path: Path) -> None:
    folder = tmp_path / "spill"
    spill = Spill(folder, 1024, clock=Moment())
    path = Path(
        ok_result({"rows": ["z" * 100] * 20}, TRACE, spill=spill).structured_content["spilled"][
            "path"
        ]
    )
    for item in (folder, path.parent):
        assert item.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o077 == 0


def test_bytes_in_a_result_are_refused_not_turned_into_text() -> None:
    with pytest.raises(CallError) as caught:
        ok_result({"image": b"\x89PNG"}, TRACE, tool="hou_capture")
    assert caught.value.code == "RESULT_NOT_JSON"
    assert "hou_capture" in caught.value.message


def test_a_path_in_a_result_reads_as_text() -> None:
    result = ok_result({"where": Path("a") / "b"}, TRACE)
    assert result.structured_content["where"] == Path("a") / "b"
    assert json.loads(text_of(result))["where"] == str(Path("a") / "b")


def test_old_spilled_results_are_reaped_and_new_ones_kept(tmp_path: Path) -> None:
    folder = tmp_path / "spill"
    old_day = folder / "2026-09-01"
    new_day = folder / "2026-09-22"
    old_day.mkdir(parents=True)
    new_day.mkdir()
    old = old_day / "old.json"
    new = new_day / "new.json"
    other = new_day / "notes.txt"
    for item in (old, new, other):
        item.write_text("{}", encoding="utf-8")
    now = time.time()
    os.utime(old, (now - 8 * 86400, now - 8 * 86400))
    assert reap_spill(folder, 7, now=now) == 1
    assert not old_day.exists()
    assert new.is_file() and other.is_file()
    assert reap_spill(tmp_path / "never-made", 7) == 0


def test_a_result_at_the_cap_is_returned_as_it_is(tmp_path: Path) -> None:
    spill = Spill(tmp_path / "spill", 1024)
    data = {"x": "y" * 100}
    result = ok_result(data, TRACE, spill=spill)
    assert "spilled" not in result.structured_content
    assert not (tmp_path / "spill").exists()


def test_two_spills_in_one_second_are_two_files(tmp_path: Path) -> None:
    spill = Spill(tmp_path / "spill", 1024, clock=Moment())
    data = {"rows": ["z" * 100] * 20}
    first = ok_result(data, TRACE, spill=spill).structured_content["spilled"]["path"]
    second = ok_result(data, TRACE, spill=spill).structured_content["spilled"]["path"]
    assert first != second


def test_a_spill_that_cannot_be_written_says_so(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the folder should be", encoding="utf-8")
    spill = Spill(blocked, 1024)
    with pytest.raises(CallError) as caught:
        ok_result({"rows": ["z" * 100] * 20}, TRACE, spill=spill)
    assert caught.value.code == "SPILL_FAILED"
