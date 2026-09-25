import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import support


@pytest.mark.parametrize("session", ["worker-id", None, {}, {"lifetime": "independent"}])
def test_only_server_bound_session_mappings_are_stopped(session) -> None:
    connected = SimpleNamespace(call_tool=AsyncMock())
    result = SimpleNamespace(structured_content={"session": session})
    asyncio.run(support.stop_server_bound_workers(connected, [result]))
    connected.call_tool.assert_not_awaited()


def test_a_server_bound_session_mapping_is_stopped() -> None:
    connected = SimpleNamespace(call_tool=AsyncMock())
    result = SimpleNamespace(
        structured_content={"session": {"session_id": "worker-id", "lifetime": "server"}}
    )
    asyncio.run(support.stop_server_bound_workers(connected, [result]))
    connected.call_tool.assert_awaited_once_with(
        "hou_sessions", {"action": "stop", "session": "worker-id"}
    )
