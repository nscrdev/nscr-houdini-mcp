"""The bridge that runs inside a Houdini process.

Importable by Houdini's own Python with nothing but the standard library and
what Houdini ships. It may use the coordination store from this package, and
nothing else from it: the server process and the bridge are separate programs
that meet over loopback HTTP and the store file.
"""

from __future__ import annotations

from nscr_houdini_mcp.bridge.app import Bridge, BridgeConfig, BridgeError
from nscr_houdini_mcp.bridge.envelope import Envelope, EnvelopeError, Reply
from nscr_houdini_mcp.bridge.handlers import ToolRegistry, default_registry

__all__ = [
    "Bridge",
    "BridgeConfig",
    "BridgeError",
    "Envelope",
    "EnvelopeError",
    "Reply",
    "ToolRegistry",
    "default_registry",
]
