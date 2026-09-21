"""The few things the bridge asks Houdini itself.

All of it is read once at start and then cached, so no request has to touch
`hou` to be answered. Every call here works when `hou` is missing: the same
package is imported by tests and by the server process, and neither has it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

GUI = "gui"
HYTHON = "hython"


def houdini() -> Any | None:
    """The `hou` module when this process is a Houdini, otherwise nothing."""
    try:
        import hou
    except ImportError:
        return None
    return hou


def session_kind() -> str:
    """Whether this process has a user interface."""
    hou = houdini()
    if hou is None:
        return HYTHON
    try:
        return GUI if hou.isUIAvailable() else HYTHON
    except Exception:  # noqa: BLE001 - a broken build must not stop the bridge
        return HYTHON


def describe() -> dict[str, Any]:
    """Build facts worth recording once, with whatever this process can read."""
    hou = houdini()
    facts: dict[str, Any] = {"houdini_version": None, "hfs": None, "hip_path": None}
    if hou is None:
        return facts
    for key, read in (
        ("houdini_version", lambda: hou.applicationVersionString()),
        ("hfs", lambda: hou.expandString("$HFS")),
        ("hip_path", lambda: hou.hipFile.path()),
    ):
        try:
            facts[key] = read()
        except Exception:  # noqa: BLE001 - a missing fact is not a failed start
            facts[key] = None
    return facts


def install_quit_hook(callback: Callable[[], None]) -> Callable[[], None] | None:
    """Run a callback when Houdini is about to quit.

    Returns the way to take the hook off again, or nothing when this Houdini
    has no such event. A session that quits some other way is still cleaned up
    by the interpreter exit hook.
    """
    hou = houdini()
    if hou is None:
        return None
    try:
        before_quit = hou.hipFileEventType.BeforeQuit
    except AttributeError:
        return None

    def on_event(event_type, *_rest):
        if event_type == before_quit:
            callback()

    try:
        hou.hipFile.addEventCallback(on_event)
    except Exception:  # noqa: BLE001 - no hook is better than no bridge
        return None

    def remove() -> None:
        try:
            hou.hipFile.removeEventCallback(on_event)
        except Exception:  # noqa: BLE001 - the session may already be tearing down
            pass

    return remove
