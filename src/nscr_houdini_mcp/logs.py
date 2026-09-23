"""Where the server writes its log, and how much it writes.

The server writes to standard error, which a client that starts it over stdio
usually keeps, and to `logs/server.log` in the state folder, beside the
bridge and worker logs. Standard output is the protocol and is never written.
The folder is made private to this user and the file is readable by this user
only, like the bridge and worker logs beside it.

`NSCR_MCP_LOG_LEVEL` sets how much: `debug`, `info`, `warning` (the default)
or `error`. A value it does not know is read as the default, and the first
line written says so.

The file is started again at `server.log.1` when it has grown past a few
megabytes, when a server starts. Several servers may share the file, and
each line carries the process id so they can be told apart. The roll over is
best effort while other servers run: it happens only for the server that
takes the roll over lock, and not at all where the file cannot be moved,
such as on Windows while another server has it open.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from nscr_houdini_mcp.bridge import security

LEVEL_ENV_VAR = "NSCR_MCP_LOG_LEVEL"
DEFAULT_LEVEL = "warning"
LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

# How much of a value the level setting does not know is quoted back.
QUOTED_CHARS = 20

LOG_DIR_NAME = "logs"
SERVER_LOG_NAME = "server.log"
ROLL_LOCK_NAME = "server.log.lock"
ROLL_OVER_BYTES = 5 * 1024 * 1024

# Every logger in the package hangs off this one.
ROOT_LOGGER = "nscr_houdini_mcp"

FORMAT = "%(asctime)s %(levelname)s pid=%(process)d %(name)s: %(message)s"


def level_from_env() -> tuple[int, str | None]:
    """The level asked for, and a complaint when the value was not one."""
    wanted = os.environ.get(LEVEL_ENV_VAR, "").strip().lower()
    if not wanted:
        return LEVELS[DEFAULT_LEVEL], None
    if wanted in LEVELS:
        return LEVELS[wanted], None
    return LEVELS[DEFAULT_LEVEL], (
        f"{LEVEL_ENV_VAR}={wanted[:QUOTED_CHARS]!r} is not one of {', '.join(LEVELS)};"
        f" using {DEFAULT_LEVEL}"
    )


def server_log_path(home: Path | str) -> Path:
    return Path(home) / LOG_DIR_NAME / SERVER_LOG_NAME


class _FileHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """A handler on a stream this module opened, which it closes with it."""

    def close(self) -> None:
        try:
            self.acquire()
            try:
                if self.stream is not None:
                    self.stream.close()
            finally:
                self.release()
        finally:
            super().close()


def setup(home: Path | str) -> logging.Logger:
    """Send the package's log to standard error and the server log file.

    Safe to call more than once: the handlers it added before are replaced.
    A file that cannot be written, or a folder that is not private, leaves
    standard error alone and says why.
    """
    level, complaint = level_from_env()
    logger = logging.getLogger(ROOT_LOGGER)
    for handler in [h for h in logger.handlers if getattr(h, "_nscr_mcp", False)]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level)
    # A host that configured the root logger must not get every line twice.
    logger.propagate = False
    formatter = logging.Formatter(FORMAT)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    path = server_log_path(home)
    problem = None
    try:
        security.private_dir(path.parent)
        _roll_over(path)
        handlers.append(_FileHandler(_open_private(path)))
    except (OSError, security.InsecureLocation) as error:
        problem = f"could not open {path.name} in the state folder: {error}"
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._nscr_mcp = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    for line in (complaint, problem):
        if line:
            logger.warning(line)
    return logger


def _open_private(path: Path) -> IO[str]:
    """The log file for appending, made readable by this user only.

    A link where the file should be is refused rather than followed.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, security.PRIVATE_FILE_MODE)
    try:
        if sys.platform != "win32":
            # A file an earlier build made with wider permissions is closed off.
            os.fchmod(fd, security.PRIVATE_FILE_MODE)
        return os.fdopen(fd, "a", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise


def _roll_over(path: Path) -> None:
    """Start the file again when it has grown too big. Best effort.

    Only the server holding the roll over lock looks, so two servers starting
    together never both move the file, and the second finds the new small one
    and leaves it. A file that cannot be moved, as on Windows while another
    server has it open, is left to grow until a later start can.
    """
    try:
        with _roll_lock(path.with_name(ROLL_LOCK_NAME)) as held:
            if not held or not path.is_file() or path.stat().st_size <= ROLL_OVER_BYTES:
                return
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


@contextmanager
def _roll_lock(lock_path: Path) -> Iterator[bool]:
    """Whether this process took the roll over lock, without waiting for it."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, security.PRIVATE_FILE_MODE)
    try:
        try:
            _lock(fd)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def _lock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
