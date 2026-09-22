"""The few things the bridge asks Houdini itself.

All of it is read once at start and then cached, so no request has to touch
`hou` to be answered. Every call here works when `hou` is missing: the same
package is imported by tests and by the server process, and neither has it.
"""

from __future__ import annotations

import threading
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


# Section: callbacks told to go

# Callbacks of ours that were told to stop but are still registered with
# Houdini, each with the call that takes it off. Anything may add to this, and
# nothing that adds calls into `hou`. Only the main thread empties it, or the
# thread stopping a session with no user interface, where the call is free.
_LEFTOVERS: list[tuple[Any, Any, Callable[[], None]]] = []
_LEFTOVERS_LOCK = threading.Lock()


def leave(hou: Any, key: Any, take_off: Callable[[], None]) -> None:
    """Put a callback on the list to come off. No `hou` call, so any thread."""
    with _LEFTOVERS_LOCK:
        if not any(held is key for _, held, _ in _LEFTOVERS):
            _LEFTOVERS.append((hou, key, take_off))


def keep(key: Any) -> bool:
    """Take a callback back off the list. False when it is off Houdini already."""
    with _LEFTOVERS_LOCK:
        for index, (_, held, _) in enumerate(_LEFTOVERS):
            if held is key:
                del _LEFTOVERS[index]
                return True
    return False


def leftovers(hou: Any) -> int:
    """How many callbacks of ours in this Houdini are waiting to come off."""
    with _LEFTOVERS_LOCK:
        return sum(1 for owner, _, _ in _LEFTOVERS if owner is hou)


def clear_leftovers(hou: Any, *, log: Callable[[str], None] | None = None) -> int:
    """Take every callback on the list off this Houdini, here and now.

    Calls into `hou`, so it runs on the main thread, on the thread starting a
    bridge while the session is idle, or anywhere in a session with no user
    interface. Safe to call from more than one of those: each callback is
    taken off the list before it is taken off Houdini, so it comes off once.
    """
    with _LEFTOVERS_LOCK:
        mine = [entry for entry in _LEFTOVERS if entry[0] is hou]
        _LEFTOVERS[:] = [entry for entry in _LEFTOVERS if entry[0] is not hou]
    for _, _, take_off in mine:
        try:
            take_off()
        except Exception as error:  # noqa: BLE001 - the session may already be tearing down
            if log is not None:
                log(f"could not take a callback off: {type(error).__name__}: {error}")
    return len(mine)


def take_off_now(key: Any) -> None:
    """Take one callback off, here, if it is still on the list. Main thread only."""
    with _LEFTOVERS_LOCK:
        found = [entry for entry in _LEFTOVERS if entry[1] is key]
        _LEFTOVERS[:] = [entry for entry in _LEFTOVERS if entry[1] is not key]
    for _, _, take_off in found:
        try:
            take_off()
        except Exception:  # noqa: BLE001 - the session may already be tearing down
            pass


# Section: the quit hook


class QuitHook:
    """Run a callback when Houdini is about to quit.

    A session that quits some other way is still cleaned up by the interpreter
    exit hook. Installing calls into `hou`, so it happens where the bridge
    starts. Removing never does: see `remove`.
    """

    def __init__(self, callback: Callable[[], None], *, hou: Any | None = None) -> None:
        self._callback = callback
        self._hou = hou
        self._on_event: Callable[..., None] | None = None
        self._live = False

    @property
    def installed(self) -> bool:
        return self._live

    def install(self) -> bool:
        """Start listening. False when this Houdini has no such event.

        A hook that was told to go but is still registered is taken back
        rather than registered a second time.
        """
        if self._live:
            return True
        hou = self._hou if self._hou is not None else houdini()
        if hou is None:
            return False
        if self._on_event is not None and keep(self._on_event):
            self._live = True
            return True
        try:
            before_quit = hou.hipFileEventType.BeforeQuit
        except AttributeError:
            return False

        def on_event(event_type: Any = None, *_rest: Any) -> None:
            if not self._live:
                # Told to go. This is the main thread, which is the only place
                # taking the callback off does not wait for a cook to end.
                take_off_now(on_event)
                return
            if event_type == before_quit:
                self._callback()

        try:
            hou.hipFile.addEventCallback(on_event)
        except Exception:  # noqa: BLE001 - no hook is better than no bridge
            return False
        self._hou = hou
        self._on_event = on_event
        self._live = True
        return True

    def remove(self) -> None:
        """Set the flag and return. Never calls into `hou` from this thread.

        The callback goes on the list of callbacks to take off, and the main
        thread takes it off on its next visit.
        """
        if not self._live:
            return
        self._live = False
        on_event, hou = self._on_event, self._hou
        if on_event is None:
            return

        def take_off() -> None:
            hou.hipFile.removeEventCallback(on_event)

        leave(hou, on_event, take_off)
