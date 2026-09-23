"""Where the server writes its log, and how much it writes.

The server writes to standard error, which a client that starts it over stdio
usually keeps, and to `logs/server.log` in the state folder, beside the
bridge and worker logs. Standard output is the protocol and is never written.

`NSCR_MCP_LOG_LEVEL` sets how much: `debug`, `info`, `warning` (the default)
or `error`. A value it does not know is read as the default, and the first
line written says so.

The file is started again at `server.log.1` when it has grown past a few
megabytes, once, when the server starts. Several servers may share the file;
each line carries the process id so they can be told apart.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LEVEL_ENV_VAR = "NSCR_MCP_LOG_LEVEL"
DEFAULT_LEVEL = "warning"
LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

LOG_DIR_NAME = "logs"
SERVER_LOG_NAME = "server.log"
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
        f"{LEVEL_ENV_VAR}={wanted!r} is not one of {', '.join(LEVELS)}; using {DEFAULT_LEVEL}"
    )


def server_log_path(home: Path | str) -> Path:
    return Path(home) / LOG_DIR_NAME / SERVER_LOG_NAME


def setup(home: Path | str) -> logging.Logger:
    """Send the package's log to standard error and the server log file.

    Safe to call more than once: the handlers it added before are replaced.
    A file that cannot be written leaves standard error alone.
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

    stream = logging.StreamHandler(sys.stderr)
    handlers: list[logging.Handler] = [stream]
    path = server_log_path(home)
    problem = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _roll_over(path)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    except OSError as error:
        problem = f"could not open {path.name} in the state folder: {error}"
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._nscr_mcp = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    for line in (complaint, problem):
        if line:
            logger.warning(line)
    return logger


def _roll_over(path: Path) -> None:
    """Start the file again when it has grown too big. Best effort."""
    try:
        if path.stat().st_size > ROLL_OVER_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass
