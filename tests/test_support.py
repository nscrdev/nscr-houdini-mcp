import asyncio
import multiprocessing as mp
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import support
from nscr_houdini_mcp.store import process_is_alive


def report_and_wait(release, index, barrier, results) -> None:
    results.put({"index": index, "pid": os.getpid(), "error": None})
    release.wait(10)


def test_child_reports_can_be_checked_before_the_launcher_exits() -> None:
    release = mp.get_context("spawn").Event()

    def check(reports):
        try:
            assert not release.is_set()
            assert process_is_alive(reports[0]["pid"])
        finally:
            release.set()

    reports = support.run_children(report_and_wait, 1, (release,), before_join=check)
    assert release.is_set()
    assert not process_is_alive(reports[0]["pid"])


@pytest.mark.parametrize("session", ["worker-id", None, {}, {"lifetime": "independent"}])
def test_only_server_bound_session_mappings_are_stopped(session) -> None:
    connected = SimpleNamespace(call_tool=AsyncMock())
    result = SimpleNamespace(structured_content={"session": session})
    asyncio.run(support.stop_server_bound_workers(connected, [result]))
    connected.call_tool.assert_not_awaited()


@pytest.mark.parametrize("lifetime", [None, "independent", "server"])
def test_persistence_skip_only_applies_to_server_bound_workers(lifetime) -> None:
    if lifetime == "server":
        with pytest.raises(pytest.skip.Exception, match="needs it to outlive its server"):
            support.require_independent_worker({"lifetime": lifetime})
    else:
        support.require_independent_worker({"lifetime": lifetime})


def test_a_server_bound_session_mapping_is_stopped() -> None:
    connected = SimpleNamespace(call_tool=AsyncMock())
    result = SimpleNamespace(
        structured_content={"session": {"session_id": "worker-id", "lifetime": "server"}}
    )
    asyncio.run(support.stop_server_bound_workers(connected, [result]))
    connected.call_tool.assert_awaited_once_with(
        "hou_sessions", {"action": "stop", "session": "worker-id"}
    )
