"""Command line entry point."""

from __future__ import annotations

import argparse

from nscr_houdini_mcp.server import SERVER_NAME, package_version, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=SERVER_NAME,
        description="Run the Houdini MCP server on stdio.",
    )
    parser.add_argument("--version", action="version", version=package_version())
    parser.parse_args(argv)
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
