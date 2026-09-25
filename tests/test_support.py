import asyncio
import importlib
import multiprocessing as mp
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import support
from nscr_houdini_mcp.store import process_is_alive


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_fixtures_do_not_open_a_persistent_client(monkeypatch, platform) -> None:
    monkeypatch.setattr(support, "sys", SimpleNamespace(platform=platform))
    with support.persistent_client(None, timeout_s=1) as send:
        assert send is None


@pytest.mark.parametrize(
    "module",
    [
        "capture",
        "jobs",
        "docs",
        "hython",
        "compare_capture",
        "compare",
        "node_type",
        "outputs",
        "inspect",
        "python",
    ],
)
def test_tool_fixtures_use_per_batch_clients_without_a_persistent_sender(monkeypatch, module):
    name = "test_tools_hython" if module == "hython" else f"test_tools_{module}_hython"
    tools = importlib.import_module(name)
    result = [object()]
    batch = AsyncMock(return_value=result)
    monkeypatch.setattr(tools, "_run", batch)
    place = {"send": None}
    call = ("hou_ping", {})
    assert tools.run(place, call) is result
    batch.assert_awaited_once_with(place, [call])


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
