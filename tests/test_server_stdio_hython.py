"""The real server over stdio, driven by the SDK's own client, against a real worker.

This is the whole path in one piece: a client starts the server as a child
process and speaks the protocol over its stdin and stdout, the server resolves
the session from the store, signs a call, and a hython worker the pool started
answers it.

Protocol revisions. The client is run twice. `auto` probes and settles on the
modern revision, 2026-07-28; `legacy` forces the initialize handshake, which
settles on 2025-11-25. The test names carry the revision each one checks.

Skipped, not failed, when there is no Houdini on this machine, which is the
case on the build machines. House rules as in the pool checks: one real worker
at most, in a state folder of this file's own, on this file's own ports, and
stopped again whatever happened, with a check that nothing is left. Nothing
here opens, saves or looks at a scene file.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

import support
from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge import registry
from nscr_houdini_mcp.store import WorkerRecord, process_is_alive


def hython_available() -> bool:
    try:
        pool.hython_path()
    except pool.HythonNotFound:
        return False
    return True


pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
]

PORT_RANGE = support.POOL_PORTS

# The server is this interpreter running the package's own entry point with
# no arguments, which is how a client starts it.
SERVER_CODE = "from nscr_houdini_mcp.cli import main; raise SystemExit(main([]))"

# Each protocol revision the client can settle on, by the mode that asks for it.
ERAS = {
    "modern-2026-07-28": ("auto", "2026-07-28"),
    "legacy-2025-11-25": ("legacy", "2025-11-25"),
}

READ_TIMEOUT_S = 120.0


@pytest.fixture(scope="module")
def worker(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, WorkerRecord]]:
    """One warm worker in a state folder of its own, stopped at the end."""
    home = tmp_path_factory.mktemp("stdio") / "home"
    home.mkdir()
    config = pool.PoolConfig(
        home=home, cap=1, max_idle_s=600.0, port_range=PORT_RANGE, start_timeout_s=240.0
    )
    with pool.open_store(home) as store:
        record = pool.start_worker(config, store)
    try:
        yield home, record
    finally:
        left = support.stop_everything(home, config)
        assert left == [], f"workers were left running: {left}"
        assert not pool.worker_is_alive(record)
        assert record.pid is None or not process_is_alive(record.pid)
        assert registry.live_entries(home) == []


def server_params(home: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", SERVER_CODE],
        env={"NSCR_MCP_HOME": str(home), "PYTHONIOENCODING": "utf-8"},
    )


async def _session_run(home: Path, mode: str, calls: list[tuple[str, dict]]) -> dict[str, Any]:
    async with Client(
        server_params(home), mode=mode, read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        listed = await connected.list_tools()
        results = [await connected.call_tool(name, args) for name, args in calls]
        return {
            "protocol": connected.protocol_version,
            "tools": listed.tools,
            "results": results,
        }


def run_session(home: Path, mode: str, *calls: tuple[str, dict]) -> dict[str, Any]:
    return asyncio.run(_session_run(home, mode, list(calls)))


@pytest.mark.parametrize("era", sorted(ERAS))
def test_a_protocol_client_lists_the_tools_and_pings_a_worker(
    worker: tuple[Path, WorkerRecord], era: str
) -> None:
    home, record = worker
    mode, revision = ERAS[era]
    with pool.open_store(home) as store:
        leased_before = store.get_worker(record.token).leased_at

    seen = run_session(
        home,
        mode,
        ("hou_ping", {}),
        ("hou_ping", {"session": record.alias}),
        ("hou_ping", {"session": "nobody"}),
    )

    assert seen["protocol"] == revision
    assert [tool.name for tool in seen["tools"]] == ["hou_ping"]
    unnamed, by_alias, unknown = seen["results"]

    for result in (unnamed, by_alias):
        assert not result.is_error, result.content[0].text
        body = result.structured_content
        assert body["session_id"] == record.session_id
        assert body["alias"] == record.alias
        assert body["kind"] == "hython"
        assert str(body["build"]).startswith("22.")
        assert body["health"]["status"] == "ok"
        assert body["call"]["ok"] is True
        assert body["transport"]["server"] == "stdio"
        assert body["transport"]["port"] in range(PORT_RANGE[0], PORT_RANGE[1] + 1)
        assert isinstance(body["scene_epoch"], int)
        assert body["trace"]["session_id"] == record.session_id
        assert body["trace"]["scene_epoch"] == body["scene_epoch"]

    assert unknown.is_error is True
    text = unknown.content[0].text
    assert text.startswith("SESSION_UNKNOWN: no session answers to nobody")
    assert "hint: " in text
    assert record.session_id in text

    # Routing a call to a worker renews its idle lease.
    with pool.open_store(home) as store:
        assert store.get_worker(record.token).leased_at > leased_before
