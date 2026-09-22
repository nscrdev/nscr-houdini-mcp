"""MCP server process for Houdini.

This module never imports `hou`. It builds the server object and runs it over
stdio. Tools come from `tools.registry` in a fixed order, so the tool list is
the same for every client and can be cached.

The server keeps nothing in memory that matters. Sessions, workers and
receipts live in the coordination store, and each call reads what it needs
there. What is kept here is a convenience: the signed client for each session
already reached, and the config once it has loaded. A config that fails to
load is read again on the next call, so fixing the file needs no restart.

A tool call runs on a worker thread, because a call can wait on a busy session
for up to its `wait_s` and `timeout_s`, and the protocol loop must stay free to
answer the client in the meantime.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult
from mcp_types import Tool as MCPTool

from nscr_houdini_mcp.bridge.errors import did_you_mean
from nscr_houdini_mcp.config import Config, ConfigError, load_config
from nscr_houdini_mcp.results import CallError, Spill, error_result, ok_result
from nscr_houdini_mcp.router import Router
from nscr_houdini_mcp.tools.base import Call, ToolSpec
from nscr_houdini_mcp.tools.registry import TOOLS

SERVER_NAME = "nscr-houdini-mcp"

INSTRUCTIONS = """
Drives SideFX Houdini 22: GUI sessions running the bridge, and hython workers.
Tools take an optional `session` id or alias; with one live session it is used.
Every result has a trace (session_id, alias, scene_epoch); pass scene_epoch back on edits.
SESSION_BUSY means another call is running: pass wait_s (up to 50) rather than retry loops.
Slow work returns a job id to wait on, not a sleep; resend a lost edit with its operation_id.
""".strip()

log = logging.getLogger(__name__)


def package_version() -> str:
    """Installed version, or a placeholder when running from a source tree."""
    try:
        return _pkg_version("nscr-houdini-mcp")
    except PackageNotFoundError:
        return "0.0.0+local"


class Runtime:
    """What every call shares: the config, the router and the spill writer."""

    def __init__(
        self,
        tools: Sequence[ToolSpec],
        *,
        config_loader: Callable[[], Config] = load_config,
        router_factory: Callable[[Config], Router] | None = None,
    ) -> None:
        self.tools = {spec.name: spec for spec in tools}
        if len(self.tools) != len(tools):
            raise ValueError("two tools share a name")
        self._config_loader = config_loader
        self._router_factory = router_factory or _router_for
        self._config: Config | None = None
        self._router: Router | None = None
        self._lock = threading.Lock()

    def settings(self) -> tuple[Config, Router]:
        """The config and the router, loading them on first use."""
        with self._lock:
            if self._config is None:
                try:
                    config = self._config_loader()
                except ConfigError as error:
                    raise CallError(
                        "CONFIG_INVALID",
                        error.message,
                        details=error.details(),
                    ) from None
                self._router = self._router_factory(config)
                self._config = config
            assert self._router is not None
            return self._config, self._router

    def run(self, name: str, arguments: Mapping[str, Any] | None) -> CallToolResult:
        """Run one tool call to a finished result. Never raises."""
        arguments = dict(arguments or {})
        spec = self.tools.get(name)
        if spec is None:
            return error_result(
                CallError(
                    "UNKNOWN_TOOL",
                    f"no tool named {name}",
                    details={
                        "did_you_mean": did_you_mean(name, self.tools),
                        "tools": list(self.tools),
                    },
                )
            )
        refused = spec.check(arguments)
        if refused is not None:
            return error_result(refused)
        call: Call | None = None
        try:
            config, router = self.settings()
            call = Call(spec, arguments, router, transport=config.transport)
            data = spec.handler(call)
            summary = spec.summary(data) if spec.summary else None
            spill = Spill(config.spill_folder, config.spill_over_bytes)
            return ok_result(data, call.trace, spill=spill, tool=name, summary=summary)
        except CallError as error:
            return error_result(error, call.trace if call else None)
        except Exception as error:  # noqa: BLE001 - a crash becomes a coded answer
            log.exception("tool %s raised", name)
            return error_result(
                CallError(
                    "TOOL_FAILED",
                    f"the server raised {type(error).__name__} running {name}",
                    details={"exception": type(error).__name__, "tool": name},
                ),
                call.trace if call else None,
            )


def _router_for(config: Config) -> Router:
    return Router(config.state_home, default_session=config.default_session)


class HoudiniServer(MCPServer):
    """An `MCPServer` whose tools are plain specs rather than typed functions.

    The schemas are written by hand as dicts, so they say exactly what the
    client is sent, and every call goes through one runner that shapes errors
    and spills large results the same way for every tool.
    """

    def __init__(self, runtime: Runtime, **rest: Any) -> None:
        super().__init__(**rest)
        self.runtime = runtime

    async def list_tools(self) -> list[MCPTool]:
        return [spec.as_tool() for spec in self.runtime.tools.values()]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> CallToolResult:
        return await anyio.to_thread.run_sync(self.runtime.run, name, arguments)


def build_server(
    tools: Sequence[ToolSpec] = TOOLS,
    *,
    config_loader: Callable[[], Config] = load_config,
    router_factory: Callable[[Config], Router] | None = None,
) -> HoudiniServer:
    """Build the server object with its tools in their fixed order."""
    runtime = Runtime(tools, config_loader=config_loader, router_factory=router_factory)
    return HoudiniServer(
        runtime,
        name=SERVER_NAME,
        version=package_version(),
        instructions=INSTRUCTIONS,
    )


def run() -> None:
    """Run the server on the configured transport. Only stdio exists today."""
    build_server().run(transport="stdio")
