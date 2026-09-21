"""Token and header checks, and the private files the token lives in.

The bridge endpoint runs Houdini work, so the only thing between it and every
other process on the machine is this module plus a loopback bind. Three rules,
all of them measured rather than assumed:

- The transport authenticates nobody. A registered function runs for whoever
  posts to it, so every handler checks the token itself.
- The transport's cross origin whitelist does not stop a request reaching the
  handler, only a browser reading the answer. So a request that carries an
  `Origin` or `Referer` header is refused in the handler.
- The token is a process secret. It goes in a file only its owner can read and
  is never written to a log or an error body.
"""

from __future__ import annotations

import hmac
import os
import secrets
import sys
from collections.abc import Mapping
from pathlib import Path

from nscr_houdini_mcp.bridge.envelope import BROWSER_HEADERS

TOKEN_BYTES = 32

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700


def mint_token() -> str:
    """A fresh token for one Houdini process. Never rotated, never reused."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_matches(presented: object, expected: str) -> bool:
    """Constant time compare that survives a missing or odd shaped token."""
    if not isinstance(presented, str) or not presented:
        return False
    try:
        return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
    except (TypeError, ValueError):
        return False


def normalise_headers(raw: Mapping[str, object] | None) -> dict[str, str]:
    """Header names lowercased, so a check cannot be dodged by casing."""
    if not raw:
        return {}
    return {str(name).lower(): str(value) for name, value in raw.items()}


def browser_header(headers: Mapping[str, str]) -> str | None:
    """The first browser only header present, or nothing.

    An empty value still counts: a page can send `Origin: null`, and the point
    is that our own client sends neither header at all.
    """
    for name in BROWSER_HEADERS:
        if name in headers:
            return name
    return None


def private_dir(path: Path) -> Path:
    """Make a folder only its owner can enter, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        try:
            path.chmod(PRIVATE_DIR_MODE)
        except OSError:
            pass
    return path


def write_private(path: Path, text: str) -> Path:
    """Write text only the owner can read.

    On POSIX the mode is set as the file is created, so there is no window
    where the content is readable by anyone else. On Windows the per user
    profile folder carries the restriction and there is no mode to set.
    """
    private_dir(path.parent)
    data = text.encode("utf-8")
    if sys.platform == "win32":
        path.write_bytes(data)
        return path
    # Replace rather than truncate, so a reader never sees a half written file.
    temporary = path.with_name(path.name + ".part")
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    return path


def is_private(path: Path) -> bool:
    """Whether a file is readable by its owner alone. Always true on Windows."""
    if sys.platform == "win32":
        return True
    return (path.stat().st_mode & 0o077) == 0
