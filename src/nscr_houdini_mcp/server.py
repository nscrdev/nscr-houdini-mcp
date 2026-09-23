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
import anyio.from_thread
import anyio.lowlevel
from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult
from mcp_types import Tool as MCPTool

from nscr_houdini_mcp.bridge.errors import did_you_mean
from nscr_houdini_mcp.config import Config, ConfigError, load_config
from nscr_houdini_mcp.results import CallError, Spill, error_result, ok_result, reap_spill
from nscr_houdini_mcp.router import Router
from nscr_houdini_mcp.tools.base import Call, ToolSpec
from nscr_houdini_mcp.tools.registry import TOOLS

SERVER_NAME = "nscr-houdini-mcp"

INSTRUCTIONS = """
Drives SideFX Houdini 22: GUI sessions running the bridge, and hython workers.
Tools take an optional `session` id or alias; with one live session it is used.
Every result has a trace (session_id, alias, scene_epoch); pass scene_epoch back on edits.
SESSION_BUSY means another call is running: pass wait_s (up to 50) rather than retry loops.
""".strip()

# Methods the SDK answers by default that this server has nothing behind. Left
# in, they would be advertised as capabilities. The tool list never changes
# while the server runs, so there is nothing to subscribe to either.
UNSERVED_METHODS = (
    "prompts/list",
    "prompts/get",
    "resources/list",
    "resources/read",
    "resources/templates/list",
    "subscriptions/listen",
)

log = logging.getLogger(__name__)


def package_version() -> str:
    """Installed version, or a placeholder when running from a source tree."""
    try:
        return _pkg_version("nscr-houdini-mcp")
    except PackageNotFoundError:
        return "0.0.0+local"


class Runtime:
    """What every call shares: the config and the router."""

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

    def run(
        self,
        name: str,
        arguments: Mapping[str, Any] | None,
        progress: Callable[[float, float | None, str | None], None] | None = None,
    ) -> CallToolResult:
        """Run one tool call to a finished result. Never raises.

        `progress` sends a progress note to the client, for a call that holds
        on purpose; it does nothing when the client asked for none.
        """
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
            call = Call(
                spec,
                arguments,
                router,
                transport=config.transport,
                config=config,
                progress=progress,
            )
            data = spec.handler(call)
            summary = spec.summary(data) if spec.summary else None
            spill = Spill(config.spill_folder, config.spill_over_bytes)
            failed = bool(spec.failed(data)) if spec.failed else False
            return ok_result(
                data,
                call.trace,
                spill=spill,
                tool=name,
                summary=summary,
                is_error=failed,
                extra=call.content,
            )
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


async def in_daemon_thread(work: Callable[..., Any], *args: Any) -> Any:
    """Run blocking work on a thread of its own and wait for it, cancellably.

    A call can sit on a busy session for as long as its budgets allow. When the
    client goes away or cancels, the wait here ends at once and the thread is
    left to finish on its own. It is a daemon thread, so it never holds the
    process open: a worker thread that is not one keeps the interpreter alive
    at exit until the socket gives up, which is the delay this avoids. The work
    it was doing is safe to abandon: a change carries an operation id, and the
    bridge keeps the receipt.
    """
    done = anyio.Event()
    token = anyio.lowlevel.current_token()
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = work(*args)
        except BaseException as error:  # noqa: BLE001 - handed to the waiting task
            box["error"] = error
        finally:
            try:
                anyio.from_thread.run_sync(done.set, token=token)
            except RuntimeError:
                # The loop has closed: nobody is waiting for the answer.
                pass

    threading.Thread(target=run, name="nscr-mcp-call", daemon=True).start()
    await done.wait()
    if "error" in box:
        raise box["error"]
    return box["value"]


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
        handlers = self._lowlevel_server._request_handlers
        for method in UNSERVED_METHODS:
            handlers.pop(method, None)

    async def list_tools(self) -> list[MCPTool]:
        return [spec.as_tool() for spec in self.runtime.tools.values()]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> CallToolResult:
        return await in_daemon_thread(self.runtime.run, name, arguments, notifier(context))


def notifier(context: Any) -> Callable[[float, float | None, str | None], None] | None:
    """A way for a call's thread to send the client a progress note.

    The note goes out on the protocol loop and is waited for here, briefly. A
    client that sent no progress token gets nothing, which the SDK decides.
    """
    report = getattr(context, "report_progress", None)
    if context is None or report is None:
        return None
    token = anyio.lowlevel.current_token()

    def send(done: float, total: float | None, message: str | None) -> None:
        try:
            anyio.from_thread.run(report, done, total, message, token=token)
        except Exception:  # noqa: BLE001 - a note nobody can take is dropped
            pass

    return send


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


def reap_at_start(config_loader: Callable[[], Config] = load_config) -> int:
    """Clear old spilled results. A config that will not load is left for the
    first call to report, so a start never fails over housekeeping."""
    try:
        config = config_loader()
        return reap_spill(config.spill_folder, config.spill_keep_days)
    except (ConfigError, OSError):
        return 0


def run() -> None:
    """Run the server on the configured transport. Only stdio exists today."""
    reap_at_start()
    build_server().run(transport="stdio")
