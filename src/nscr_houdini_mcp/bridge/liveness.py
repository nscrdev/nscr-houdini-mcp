"""Telling one process from another that happens to have the same number.

A pid on its own is not an identity. Numbers are handed out again, so a
session file left behind by a process that crashed can name a pid that now
belongs to something else entirely. Each system can say when a process
started, and a pid plus a start time is an identity that holds.

Every answer here is a string, so it goes into a session file as it is and
comes back comparable. `None` means this system would not say, and a caller
that gets `None` has learned nothing and must not pretend otherwise.

The answers themselves live with the coordination store, which asks the same
question of its own rows and may not import anything from the bridge. This is
the name the bridge knows them by.
"""

from __future__ import annotations

from nscr_houdini_mcp.store import process_is_alive, process_start_stamp, same_process

__all__ = ["process_is_alive", "process_start_stamp", "same_process"]
