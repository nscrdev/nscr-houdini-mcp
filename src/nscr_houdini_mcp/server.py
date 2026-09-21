"""MCP server process for Houdini.

This module never imports `hou`. It builds the server object and runs it over
stdio. Tools are registered in a fixed order so the tool list is deterministic
and cacheable by a client.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from mcp.server.mcpserver import MCPServer

SERVER_NAME = "nscr-houdini-mcp"

INSTRUCTIONS = """
Drive SideFX Houdini 22 from a small tool set.

Nothing is wired up yet: this build registers no tools. Later builds address
named Houdini sessions, return a short summary before any bulk payload, and
hand back a job id for work that takes a while.
""".strip()


def package_version() -> str:
    """Installed version, or a placeholder when running from a source tree."""
    try:
        return _pkg_version("nscr-houdini-mcp")
    except PackageNotFoundError:
        return "0.0.0+local"


def build_server() -> MCPServer:
    """Build the server object with its tools registered in a fixed order."""
    server = MCPServer(
        name=SERVER_NAME,
        version=package_version(),
        instructions=INSTRUCTIONS,
    )
    for register in TOOL_REGISTRARS:
        register(server)
    return server


# One entry per tool module, in the order the tool list should present them.
TOOL_REGISTRARS: list = []


def run() -> None:
    """Run the server on stdio."""
    build_server().run(transport="stdio")
