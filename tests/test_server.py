from __future__ import annotations

import asyncio

from nscr_houdini_mcp.server import SERVER_NAME, build_server


def test_server_builds_with_a_deterministic_tool_list() -> None:
    server = build_server()
    assert server.name == SERVER_NAME
    assert server.instructions

    first = asyncio.run(server.list_tools())
    second = asyncio.run(build_server().list_tools())
    assert [tool.name for tool in first] == [tool.name for tool in second]
