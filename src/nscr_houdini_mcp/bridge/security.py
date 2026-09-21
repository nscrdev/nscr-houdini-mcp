"""Header rules, and the files the token lives in.

The bridge endpoint runs Houdini work, so what stands between it and the rest
of the machine is this module, the signing module and a loopback bind.

Header rules, in the order a request meets them:

- A request carrying `Origin` or `Referer` is refused. The bridge's own client
  sends neither. A page in a browser always sends one.
- A request whose `Host` is not this bridge's own loopback name and port is
  refused, which stops a name pointed at 127.0.0.1 from being used to dress a
  request up as same site.

File rules:

- The token lives in one file that only its owner can read, created with the
  mode already set, so there is no moment when it is readable by anyone else.
- The folder holding it is checked after it is made: not a link, owned by this
  user, nothing granted to group or other. A folder that fails is an error,
  not a warning.
- On Windows there are no mode bits. What protects the file is the per user
  profile folder it sits in, so a state folder in a shared or synced place is
  refused outright.
"""

from __future__ import annotations

import os
import secrets
import sys
from collections.abc import Mapping
from pathlib import Path

from nscr_houdini_mcp.bridge.envelope import BROWSER_HEADERS
from nscr_houdini_mcp.bridge.net import LOOPBACK_NAMES
from nscr_houdini_mcp.store import shared_location_warning

TOKEN_BYTES = 32

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700

HOST_HEADER = "host"


class InsecureLocation(Exception):
    """A folder that must be private to this user is not."""


def mint_token() -> str:
    """A fresh token for one Houdini process. Never rotated, never reused."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def normalise_headers(raw: Mapping[str, object] | None) -> dict[str, str]:
    """Header names lowercased, so a check cannot be dodged by casing."""
    if not raw:
        return {}
    return {str(name).lower(): str(value) for name, value in raw.items()}


def browser_header(headers: Mapping[str, str]) -> str | None:
    """The first browser only header present, or nothing.

    An empty value still counts: a page can send `Origin: null`, and the point
    is that the bridge's own client sends neither header at all.
    """
    for name in BROWSER_HEADERS:
        if name in headers:
            return name
    return None


def host_allowed(headers: Mapping[str, str], port: int) -> bool:
    """Whether the `Host` a request carried is this bridge's own.

    A missing host is refused too. Every client this bridge answers sends one.
    """
    raw = headers.get(HOST_HEADER)
    if not raw:
        return False
    host = raw.strip()
    if host.startswith("["):
        name, _, tail = host.partition("]")
        name = name[1:]
        port_text = tail[1:] if tail.startswith(":") else ""
    else:
        name, _, port_text = host.partition(":")
    if name.lower() not in LOOPBACK_NAMES:
        return False
    return port_text == str(port)


def check_home(home: Path) -> None:
    """Refuse a state folder that cannot keep a secret.

    On Windows the mode bits do not exist, so the only protection is the per
    user profile folder. A folder that is shared, synced or on a network path
    is not that, and a token has no business in one.
    """
    warning = shared_location_warning(home)
    if warning is not None:
        raise InsecureLocation(
            f"{home} is not a private per user folder, so it cannot hold a token"
        )


def private_dir(path: Path) -> Path:
    """Make a folder only its owner can enter, then prove that it is one."""
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        path.chmod(PRIVATE_DIR_MODE)
    check_private_dir(path)
    return path


def check_private_dir(path: Path) -> None:
    """Raise unless this folder is a real folder, owned here and closed off."""
    stat = path.lstat()
    if not os.path.isdir(path) or os.path.islink(path):
        raise InsecureLocation(f"{path} is a link, not a folder")
    if sys.platform == "win32":
        return
    if stat.st_uid != os.getuid():
        raise InsecureLocation(f"{path} belongs to another user")
    if stat.st_mode & 0o077:
        raise InsecureLocation(f"{path} is open to other users")


def write_private(path: Path, text: str) -> Path:
    """Write text only the owner can read.

    The mode is set as the file is created, and the name is created
    exclusively, so nothing can be waiting in place of it. The finished file
    is moved over the old one, so a reader never sees half of it.
    """
    private_dir(path.parent)
    data = text.encode("utf-8")
    if sys.platform == "win32":
        # No mode bits. The profile folder is the protection, and `check_home`
        # has already refused a folder that is not one.
        temporary = path.with_name(path.name + ".part")
        temporary.unlink(missing_ok=True)
        temporary.write_bytes(data)
        os.replace(temporary, path)
        return path
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        if not is_private(temporary):
            raise InsecureLocation(f"{temporary} was created readable by others")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    return path


def is_private(path: Path) -> bool:
    """Whether a file is readable by its owner alone.

    Always true on Windows, where the folder rather than the file carries the
    restriction.
    """
    if sys.platform == "win32":
        return True
    stat = path.lstat()
    return not os.path.islink(path) and stat.st_uid == os.getuid() and not stat.st_mode & 0o077
