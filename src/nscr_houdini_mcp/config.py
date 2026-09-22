"""The server's own settings, read from one TOML file in the state folder.

Every key is optional. A missing file in the state folder is the same as an
empty one: the server runs on the defaults below. A file named outright, by
`NSCR_MCP_CONFIG` or on the command line, must be there. A file that is there
and wrong is refused whole, with the key, what was wrong with it and, for a
misspelled key, the nearest name, because a setting that is quietly ignored is
worse than one that stops the server from guessing.

Where the file is. `NSCR_MCP_CONFIG` names it when set. Otherwise it is
`config.toml` in the state folder, which `NSCR_MCP_HOME` moves. The file's own
`state_home` then decides where sessions, the store and spilled results are
read and written, so a bridge started with the same `NSCR_MCP_HOME` and a
server reading a file that names another folder do not see each other. Keep
them pointing at the same place.

Which hython. `hython` names one outright and wins. `houdini_build` picks an
install by version, `22.0` for the newest 22.0 build or `22.0.368` for that
one. With neither, `NSCR_MCP_HYTHON` is read, then the newest install this
machine has is used.

The same file may hold `[outputs]` and `[conventions]` tables. Those belong to
the output paths, which check them when they read them, so they are passed
over here.

This module never imports `hou`.
"""

from __future__ import annotations

import difflib
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import pool
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge.security import InsecureLocation, check_home

CONFIG_ENV_VAR = "NSCR_MCP_CONFIG"
CONFIG_FILE_NAME = "config.toml"
SPILL_DIR_NAME = "spill"

# The only transport this build serves. The key exists so a file written now
# still reads the same when a second one arrives.
TRANSPORTS = ("stdio",)

# Tables other parts of the project read from the same file.
OTHER_TABLES = ("outputs", "conventions")

DEFAULT_SPILL_OVER_BYTES = 64 * 1024
MIN_SPILL_OVER_BYTES = 1024
MAX_SPILL_OVER_BYTES = 64 * 1024 * 1024

MAX_POOL_CAP = 16

# Ports a worker may be given. Below this are the ports a system keeps.
MIN_PORT = 1024
MAX_PORT = 65535

DEFAULT_SPILL_KEEP_DAYS = 7
MAX_SPILL_KEEP_DAYS = 365

_BUILD = re.compile(r"^\d+\.\d+(\.\d+)*$")

# Every key the file may hold, in the order `config show` prints them.
KEYS = (
    "hython",
    "houdini_build",
    "default_session",
    "pool_cap",
    "worker_ports",
    "state_home",
    "spill_dir",
    "spill_over_bytes",
    "spill_keep_days",
    "transport",
)

TEMPLATE = f"""\
# Settings for the nscr-houdini-mcp server. Every key is optional; an empty
# string or a missing key means the default. Paths go in single quotes, which
# TOML reads as written, so a Windows path needs no doubled backslashes.

# The hython that workers start with. Wins over houdini_build.
hython = ''

# Pick an install by version instead: "22.0" for the newest 22.0 build, or a
# full build such as "22.0.368". With neither set, NSCR_MCP_HYTHON is read,
# then the newest install on this machine is used.
houdini_build = ""

# The session a call reaches when it names none and several are live.
# An alias such as "w1" or a session id.
default_session = ""

# How many hython workers may run at once, beside any Houdini you have open.
pool_cap = {pool.DEFAULT_CAP}

# The first and last port a worker started from here may listen on.
worker_ports = [{pool.DEFAULT_PORT_RANGE[0]}, {pool.DEFAULT_PORT_RANGE[1]}]

# Where sessions, the coordination store and logs live. Bridges must use the
# same folder (NSCR_MCP_HOME) or the server will not see them.
state_home = ''

# Where results too large to return are written. Default: <state_home>/{SPILL_DIR_NAME}
spill_dir = ''

# Results larger than this many bytes are written to a file and returned as
# the path and a preview.
spill_over_bytes = {DEFAULT_SPILL_OVER_BYTES}

# Spilled results older than this many days are removed when the server starts.
spill_keep_days = {DEFAULT_SPILL_KEEP_DAYS}

# How clients reach the server. Only "stdio" is served by this build.
transport = "stdio"
"""


class ConfigError(Exception):
    """The config file is there and cannot be used as it is."""

    def __init__(self, message: str, *, path: Path | None = None, key: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.key = key

    def details(self) -> dict[str, Any]:
        """The key and where the file is, told relative to the state folder."""
        return {"path": shown_path(self.path), "key": self.key}


def shown_path(path: Path | None) -> str | None:
    """A config path as a caller is told it: under the state folder when it is
    there, otherwise its name alone, so no other place on disk is given away."""
    if path is None:
        return None
    home = store_module.default_home()
    try:
        return Path(path).relative_to(home).as_posix()
    except ValueError:
        return Path(path).name


@dataclass(frozen=True)
class Config:
    """The settings as the server uses them, every default filled in."""

    path: Path
    exists: bool = False
    hython: Path | None = None
    houdini_build: str | None = None
    default_session: str | None = None
    pool_cap: int = pool.DEFAULT_CAP
    worker_ports: tuple[int, int] = pool.DEFAULT_PORT_RANGE
    state_home: Path = field(default_factory=store_module.default_home)
    spill_dir: Path | None = None
    spill_over_bytes: int = DEFAULT_SPILL_OVER_BYTES
    spill_keep_days: int = DEFAULT_SPILL_KEEP_DAYS
    transport: str = TRANSPORTS[0]
    # Which keys the file set. The rest are defaults.
    from_file: frozenset[str] = frozenset()

    @property
    def spill_folder(self) -> Path:
        return self.spill_dir if self.spill_dir is not None else self.state_home / SPILL_DIR_NAME

    @property
    def store_path(self) -> Path:
        return self.state_home / store_module.STORE_FILE_NAME

    def shown(self) -> dict[str, Any]:
        """Every key with the value in use, for a person to read."""
        values = {
            "hython": self.hython,
            "houdini_build": self.houdini_build,
            "default_session": self.default_session,
            "pool_cap": self.pool_cap,
            "worker_ports": list(self.worker_ports),
            "state_home": self.state_home,
            "spill_dir": self.spill_folder,
            "spill_over_bytes": self.spill_over_bytes,
            "spill_keep_days": self.spill_keep_days,
            "transport": self.transport,
        }
        return {key: values[key] for key in KEYS}


def config_path() -> Path:
    """Where the config file is read from."""
    named = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if named:
        return Path(named).expanduser()
    return store_module.default_home() / CONFIG_FILE_NAME


def load_config(path: Path | str | None = None) -> Config:
    """Read the config file, or the defaults when there is none.

    Raises `ConfigError` for a file that cannot be read or holds a wrong value,
    and for a file named outright that is not there: whoever named it meant
    those settings, and running on the defaults instead would hide the mistake.
    """
    named = path is not None or bool(os.environ.get(CONFIG_ENV_VAR, "").strip())
    where = Path(path).expanduser() if path is not None else config_path()
    if not where.is_file():
        if named:
            raise ConfigError(f"the config file named is not there: {where.name}", path=where)
        return Config(path=where)
    try:
        text = where.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(
            f"could not read the config file ({type(error).__name__})", path=where
        ) from None
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(_toml_message(error), path=where) from None
    return parse_config(raw, path=where)


def _toml_message(error: tomllib.TOMLDecodeError) -> str:
    message = f"the config file is not valid TOML: {error}"
    text = str(error).lower()
    if "hex" in text or "escape" in text:
        # A Windows path in double quotes reads its backslashes as escapes.
        message += (
            "; a backslash in double quotes starts an escape, so put paths in single"
            " quotes or write them with forward slashes"
        )
    return message


def parse_config(raw: Mapping[str, Any], *, path: Path) -> Config:
    """Check every key of a decoded file and build the settings from it."""
    unknown = [key for key in raw if key not in KEYS and key not in OTHER_TABLES]
    if unknown:
        key = str(unknown[0])
        near = difflib.get_close_matches(key, KEYS, n=1, cutoff=0.5)
        tail = f"; did you mean {near[0]}?" if near else f"; known keys: {', '.join(KEYS)}"
        raise ConfigError(f"unknown key {key!r}{tail}", path=path, key=key)

    values: dict[str, Any] = {}
    for key in KEYS:
        if key not in raw:
            continue
        value = _CHECKS[key](raw[key], key, path)
        if value is not None:
            values[key] = value

    if "hython" in values and "houdini_build" in values:
        raise ConfigError(
            "set hython or houdini_build, not both: hython names the program outright",
            path=path,
            key="houdini_build",
        )
    return Config(path=path, exists=True, from_file=frozenset(values), **values)


# Section: one check per key. Each returns the value to use, or nothing for an
# empty string, which reads as "use the default".


def _text(value: Any, key: str, path: Path) -> str | None:
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string, got {_kind(value)}", path=path, key=key)
    return value.strip() or None


def _folder(value: Any, key: str, path: Path) -> Path | None:
    text = _text(value, key, path)
    if text is None:
        return None
    folder = Path(text).expanduser()
    if not folder.is_absolute():
        raise ConfigError(f"{key} must be an absolute path, got {text!r}", path=path, key=key)
    return folder


def _private_folder(value: Any, key: str, path: Path) -> Path | None:
    """A folder results are written to, which must be this user's own."""
    folder = _folder(value, key, path)
    if folder is not None:
        try:
            check_home(folder)
        except InsecureLocation as error:
            raise ConfigError(
                f"{key} cannot hold private results: {error}", path=path, key=key
            ) from None
    return folder


def _hython(value: Any, key: str, path: Path) -> Path | None:
    # Whether the file is there is checked when a worker starts, so a config
    # written before Houdini is installed still loads.
    return _folder(value, key, path)


def _build(value: Any, key: str, path: Path) -> str | None:
    text = _text(value, key, path)
    if text is not None and not _BUILD.match(text):
        raise ConfigError(
            f"{key} must be a version such as 22.0 or 22.0.368, got {text!r}", path=path, key=key
        )
    return text


def _whole(value: Any, key: str, path: Path, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be a whole number, got {_kind(value)}", path=path, key=key)
    if not low <= value <= high:
        raise ConfigError(f"{key} must be from {low} to {high}, got {value}", path=path, key=key)
    return value


def _cap(value: Any, key: str, path: Path) -> int:
    return _whole(value, key, path, 1, MAX_POOL_CAP)


def _ports(value: Any, key: str, path: Path) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(
            f"{key} must be two port numbers, the first and the last, got {_kind(value)}",
            path=path,
            key=key,
        )
    first, last = (_whole(item, key, path, MIN_PORT, MAX_PORT) for item in value)
    if first > last:
        raise ConfigError(f"{key} must list the lower port first", path=path, key=key)
    return (first, last)


def _spill_over(value: Any, key: str, path: Path) -> int:
    return _whole(value, key, path, MIN_SPILL_OVER_BYTES, MAX_SPILL_OVER_BYTES)


def _keep_days(value: Any, key: str, path: Path) -> int:
    return _whole(value, key, path, 1, MAX_SPILL_KEEP_DAYS)


def _transport(value: Any, key: str, path: Path) -> str | None:
    text = _text(value, key, path)
    if text is not None and text not in TRANSPORTS:
        raise ConfigError(
            f"{key} must be one of {', '.join(TRANSPORTS)}, got {text!r}", path=path, key=key
        )
    return text


def _kind(value: Any) -> str:
    return {bool: "true or false", int: "a number", float: "a decimal", str: "a string"}.get(
        type(value), type(value).__name__
    )


_CHECKS = {
    "hython": _hython,
    "houdini_build": _build,
    "default_session": _text,
    "pool_cap": _cap,
    "worker_ports": _ports,
    "state_home": _folder,
    "spill_dir": _private_folder,
    "spill_over_bytes": _spill_over,
    "spill_keep_days": _keep_days,
    "transport": _transport,
}


# Section: which hython


def resolve_hython(config: Config) -> Path | None:
    """The hython the settings name, or nothing to let the pool look for one.

    A configured path is taken as given. A build is matched against the
    installs this machine has and raises `ConfigError` when none matches, with
    the builds that are there.
    """
    if config.hython is not None:
        return config.hython
    if config.houdini_build is None:
        return None
    wanted = config.houdini_build
    installs = install_module.find_installs()
    for found in installs:
        if found.version == wanted or found.version.startswith(wanted + "."):
            return found.hfs / "bin" / ("hython.exe" if os.name == "nt" else "hython")
    have = ", ".join(found.version for found in installs if found.version) or "none"
    raise ConfigError(
        f"no Houdini {wanted} install found; installs here: {have}",
        path=config.path,
        key="houdini_build",
    )


def describe_hython(config: Config) -> str:
    """One line on which hython workers would start with, and why."""
    try:
        chosen = resolve_hython(config)
    except ConfigError as error:
        return f"none: {error.message}"
    if chosen is not None:
        why = "hython" if config.hython is not None else f"houdini_build {config.houdini_build}"
        return f"{chosen} (from {why})"
    try:
        found = pool.hython_path()
    except pool.HythonNotFound as error:
        return f"none: {error}"
    why = pool.HYTHON_ENV_VAR if os.environ.get(pool.HYTHON_ENV_VAR) else "the newest install"
    return f"{found} (from {why})"


def write_template(path: Path, *, force: bool = False) -> Path:
    """Write a commented file with every key at its default."""
    if path.exists() and not force:
        raise FileExistsError(f"{path} is already there; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8", newline="\n")
    return path
