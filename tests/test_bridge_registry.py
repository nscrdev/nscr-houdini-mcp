from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from nscr_houdini_mcp.bridge import liveness, registry


def entry(session_id: str, **fields: object) -> dict:
    row = {
        "session_id": session_id,
        "alias": session_id,
        "pid": os.getpid(),
        "pid_start": liveness.process_start_stamp(),
        "port": 18100,
        "token": "secret",
        "started_at": 1.0,
    }
    row.update(fields)
    return row


def test_this_process_says_when_it_started() -> None:
    stamp = liveness.process_start_stamp()
    if sys.platform not in ("linux", "darwin", "win32"):
        pytest.skip("this system does not report a start time")
    assert stamp
    assert liveness.process_start_stamp() == stamp


def test_a_pid_that_is_not_running_is_not_this_process() -> None:
    assert liveness.same_process(None, "x") is False
    assert liveness.same_process(-1, "x") is False


def test_a_process_with_no_recorded_start_cannot_be_settled() -> None:
    assert liveness.same_process(os.getpid(), None) is None


def test_this_process_matches_its_own_stamp() -> None:
    stamp = liveness.process_start_stamp()
    if not stamp:
        pytest.skip("this system does not report a start time")
    assert liveness.same_process(os.getpid(), stamp) is True
    assert liveness.same_process(os.getpid(), "a different stamp") is False


def test_a_written_entry_reads_back(tmp_path: Path) -> None:
    registry.write_entry(tmp_path, entry("one"))
    assert [row["session_id"] for row in registry.list_entries(tmp_path)] == ["one"]
    assert registry.read_entry(registry.entry_path(tmp_path, "one"))["token"] == "secret"


def test_an_entry_from_a_process_that_is_gone_is_cleared_on_read(tmp_path: Path) -> None:
    registry.write_entry(tmp_path, entry("live"))
    registry.write_entry(tmp_path, entry("dead", pid=-1, pid_start="gone"))
    assert [row["session_id"] for row in registry.live_entries(tmp_path)] == ["live"]
    assert registry.entry_path(tmp_path, "dead").exists() is False
    assert registry.entry_path(tmp_path, "live").exists() is True


def test_an_entry_whose_pid_now_belongs_to_something_else_is_cleared(tmp_path: Path) -> None:
    if not liveness.process_start_stamp():
        pytest.skip("this system does not report a start time")
    registry.write_entry(tmp_path, entry("recycled", pid_start="an older process"))
    assert registry.live_entries(tmp_path) == []
    assert registry.entry_path(tmp_path, "recycled").exists() is False


def test_an_entry_can_be_found_by_id_or_by_alias(tmp_path: Path) -> None:
    registry.write_entry(tmp_path, entry("one", alias="w1"))
    assert registry.find_entry(tmp_path, "one")["session_id"] == "one"
    assert registry.find_entry(tmp_path, "w1")["session_id"] == "one"
    assert registry.find_entry(tmp_path, "w2") is None


def test_an_unreadable_file_is_passed_over_rather_than_raised_about(tmp_path: Path) -> None:
    registry.write_entry(tmp_path, entry("one"))
    (registry.registry_dir(tmp_path) / "half.json").write_text("{", encoding="utf-8")
    assert [row["session_id"] for row in registry.list_entries(tmp_path)] == ["one"]


def test_removing_an_entry_that_is_not_there_is_not_a_failure(tmp_path: Path) -> None:
    registry.remove_entry(tmp_path, "never-existed")
    assert registry.list_entries(tmp_path) == []


@pytest.mark.parametrize("winerror,timeout", [(5, 1), (32, 0), (33, 0)])
def test_failed_registry_removal_is_not_hidden_or_retried_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winerror: int, timeout: float
) -> None:
    path = registry.write_entry(tmp_path, entry("held"))
    attempts = 0
    failure = PermissionError("file cannot be removed")
    failure.winerror = winerror

    def refuse(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise failure

    monkeypatch.setattr(Path, "unlink", refuse)
    monkeypatch.setattr(registry, "REMOVE_TIMEOUT_S", timeout)
    with pytest.raises(PermissionError) as caught:
        registry.remove_entry(tmp_path, "held")
    assert caught.value is failure
    assert attempts == 1
    assert path.exists()


def test_a_session_is_opened_only_while_its_process_is_there(tmp_path: Path) -> None:
    from nscr_houdini_mcp.bridge import client

    registry.write_entry(tmp_path, entry("live", port=18111, token="a-secret"))
    session = client.Session.open(tmp_path, "live")
    assert session.token == "a-secret"
    assert session.port == 18111

    registry.write_entry(tmp_path, entry("dead", pid=-1, pid_start="gone"))
    with pytest.raises(client.SessionGone):
        client.Session.open(tmp_path, "dead")
    with pytest.raises(client.SessionGone):
        client.Session.open(tmp_path, "never-existed")
