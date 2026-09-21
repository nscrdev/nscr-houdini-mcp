"""The file a bridge writes so other processes can find and address it.

The coordination store is what servers read to list and route. This file is
the same facts in a form a person can open, plus the one fact the store never
holds: the token. That is why the file is written so only its owner can read
it, and why nothing else in the project prints its contents.

There is no clean exit marker. A clean exit deletes the file, so a file whose
process is gone means a crash.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge.security import private_dir, write_private

REGISTRY_DIR_NAME = "sessions"


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
    """Delete one session file. A missing file is already the wanted state."""
    entry_path(home, session_id).unlink(missing_ok=True)


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
