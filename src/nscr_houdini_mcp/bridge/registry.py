"""The file a bridge writes so other processes can find and address it.

The coordination store is what servers read to list and route. This file is
the same facts in a form a person can open, plus the one fact the store never
holds: the token. That is why the file is written so only its owner can read
it, and why nothing else in the project prints its contents.

There is no clean exit marker. A clean exit deletes the file, so a file whose
process is gone means a crash. A file left by a crash is worse than useless:
the port it names is free again and anything could be answering on it, so a
reader clears those before handing any of them back.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge import liveness
from nscr_houdini_mcp.bridge.security import private_dir, write_private

REGISTRY_DIR_NAME = "sessions"
REMOVE_TIMEOUT_S = 1.0
ERROR_SHARING_VIOLATION = 32
ERROR_LOCK_VIOLATION = 33


def registry_dir(home: Path) -> Path:
    """The folder of session files under the per user state folder."""
    return home / REGISTRY_DIR_NAME


def entry_path(home: Path, session_id: str) -> Path:
    """The file for one session. Named by id, because ids are never reused."""
    return registry_dir(home) / f"{session_id}.json"


def write_entry(home: Path, entry: Mapping[str, Any]) -> Path:
    """Write one session file, readable by its owner alone."""
    session_id = entry["session_id"]
    text = json.dumps(dict(entry), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return write_private(entry_path(home, session_id), text)


def read_entry(path: Path) -> dict[str, Any]:
    """Read one session file."""
    return json.loads(path.read_text(encoding="utf-8"))


def remove_entry(home: Path, session_id: str) -> None:
    """Delete a session file, briefly waiting for Windows readers to release it."""
    path = entry_path(home, session_id)
    deadline = time.monotonic() + REMOVE_TIMEOUT_S
    delay = 0.01
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as error:
            if (
                sys.platform != "win32"
                or getattr(error, "winerror", None)
                not in (ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION)
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.1)


def list_entries(home: Path) -> list[dict[str, Any]]:
    """Every readable session file, oldest start first.

    A file being written or removed right now is skipped rather than raised
    over: the caller wants the sessions that are there.
    """
    folder = registry_dir(home)
    if not folder.is_dir():
        return []
    entries = []
    for path in sorted(folder.glob("*.json")):
        try:
            entries.append(read_entry(path))
        except (OSError, ValueError):
            continue
    return sorted(entries, key=lambda entry: entry.get("started_at") or 0)


def ensure_registry_dir(home: Path) -> Path:
    """Make the session folder, private to its owner."""
    return private_dir(registry_dir(home))


def entry_is_live(entry: Mapping[str, Any]) -> bool | None:
    """Whether the process named in an entry is still the one that wrote it.

    `False` means the process is gone, so the entry is rubbish and the port it
    names may belong to something else now. `None` means this system would not
    say which process a pid is, so the entry cannot be ruled out or in.
    """
    return liveness.same_process(entry.get("pid"), entry.get("pid_start"))


def live_entries(home: Path, *, remove_stale: bool = True) -> list[dict[str, Any]]:
    """Session files whose process is still there, dropping the rest.

    A Houdini that crashes leaves its file behind, holding a token and a port
    that anything could be sitting on by now. Clearing those on read keeps a
    caller from ever reaching for one.
    """
    live = []
    for entry in list_entries(home):
        if entry_is_live(entry) is False:
            if remove_stale:
                remove_entry(home, entry.get("session_id", ""))
            continue
        live.append(entry)
    return live


def find_entry(home: Path, handle: str, *, remove_stale: bool = True) -> dict[str, Any] | None:
    """One live session by id or by alias, or nothing."""
    for entry in live_entries(home, remove_stale=remove_stale):
        if handle in (entry.get("session_id"), entry.get("alias")):
            return entry
    return None
