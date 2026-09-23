"""Which version of this package is running, on either side of the bridge.

The server and the bridge inside Houdini are the same package, but they need
not be the same copy: Houdini imports the copy `bridge install` made, and an
upgrade reaches it only when the install is run again and Houdini restarted.
Each bridge reports the version it runs, and the server warns when that is
not its own. This module imports nothing, so Houdini can read it too.
"""

from __future__ import annotations

from typing import Any

# Kept equal to the version in pyproject.toml; a test holds them together.
VERSION = "0.1.0"

# Moves when the server and the bridge stop understanding each other's calls.
PROTOCOL = 1

MISMATCH_CODE = "BRIDGE_VERSION_MISMATCH"
MISMATCH_HINT = "run nscr-houdini-mcp bridge install again, then restart Houdini"


def mismatch(package_version: Any, protocol: Any) -> dict[str, Any] | None:
    """A warning when a bridge runs another version than this server, else nothing.

    A bridge that reports no version is older than the report itself, so it
    is another version too.
    """
    if package_version == VERSION and protocol == PROTOCOL:
        return None
    said = str(package_version) if package_version else "no version"
    return {
        "code": MISMATCH_CODE,
        "message": f"the bridge runs {said}, this server {VERSION}",
        "bridge_version": package_version or None,
        "server_version": VERSION,
        "hint": MISMATCH_HINT,
    }
