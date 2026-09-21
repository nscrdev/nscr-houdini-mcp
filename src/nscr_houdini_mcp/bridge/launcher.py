"""Find hython, and run one bridge inside a fresh one.

Used by the tests that need a real Houdini, and by anything else that wants a
bridge it can start and stop on purpose. The started process is tied to its
input stream, so it goes away when the launcher does.

Lookup order for the binary: the configured path, the `HFS` environment
variable, then the platform's usual install folder, newest build first.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.bridge.main import STOP_WORD
from nscr_houdini_mcp.bridge.net import DEFAULT_PORT_RANGE
from nscr_houdini_mcp.store import HOME_ENV_VAR

HYTHON_ENV_VAR = "NSCR_MCP_HYTHON"
HFS_ENV_VAR = "HFS"

START_TIMEOUT_S = 180.0
STOP_TIMEOUT_S = 60.0

MODULE = "nscr_houdini_mcp.bridge.main"


class HythonNotFound(Exception):
    """No hython on this machine, or none where it was said to be."""


class BridgeStartFailed(Exception):
    """The process started but no bridge announced itself."""


def _executable(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def install_roots() -> list[Path]:
    """Where Houdini usually lives on this system."""
    if sys.platform == "darwin":
        return [Path("/Applications/Houdini")]
    if sys.platform == "win32":
        roots = []
        for variable in ("ProgramFiles", "ProgramW6432"):
            base = os.environ.get(variable)
            if base:
                roots.append(Path(base) / "Side Effects Software")
        return roots
    return [Path("/opt")]


def _from_hfs(hfs: Path) -> Path:
    return hfs / "bin" / _executable("hython")


def candidates() -> list[Path]:
    """Every hython this machine seems to have, newest looking first."""
    found: list[Path] = []
    hfs = os.environ.get(HFS_ENV_VAR)
    if hfs:
        found.append(_from_hfs(Path(hfs)))
    for root in install_roots():
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            if sys.platform == "darwin":
                found.append(
                    _from_hfs(
                        child
                        / "Frameworks"
                        / "Houdini.framework"
                        / "Versions"
                        / "Current"
                        / "Resources"
                    )
                )
            else:
                found.append(_from_hfs(child))
    seen: list[Path] = []
    for path in found:
        if path not in seen:
            seen.append(path)
    return seen


def find_hython(configured: Path | str | None = None) -> Path:
    """The hython to use, or `HythonNotFound`.

    A configured path is used as given, and a wrong one is an error rather
    than a reason to go looking somewhere else.
    """
    named = configured or os.environ.get(HYTHON_ENV_VAR)
    if named:
        path = Path(named).expanduser()
        if not path.is_file():
            raise HythonNotFound(f"{path} is not a file")
        return path
    for path in candidates():
        if path.is_file():
            return path
    raise HythonNotFound("no hython found, set the path in config")


def hython_available(configured: Path | str | None = None) -> bool:
    """Whether a Houdini is there to test against."""
    try:
        find_hython(configured)
    except HythonNotFound:
        return False
    return True


class HythonBridge:
    """A bridge in a hython of its own. Start it, use it, stop it."""

    def __init__(
        self,
        *,
        home: Path,
        hython: Path | str | None = None,
        port_range: tuple[int, int] = DEFAULT_PORT_RANGE,
        alias: str | None = None,
        extra_args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.home = Path(home)
        self.hython = find_hython(hython)
        self.port_range = port_range
        self.alias = alias
        self.extra_args = list(extra_args)
        self.env = dict(env) if env is not None else None
        self.process: subprocess.Popen[str] | None = None
        self.log_path = self.home / "hython.log"
        self.entry: dict[str, Any] | None = None
        self._log = None

    @property
    def port(self) -> int:
        return int(self._entry()["port"])

    @property
    def session_id(self) -> str:
        return str(self._entry()["session_id"])

    @property
    def session(self) -> client.Session:
        """What a caller needs to sign a request to this bridge."""
        return client.Session.from_entry(self._entry())

    def start(self, *, timeout_s: float = START_TIMEOUT_S) -> dict[str, Any]:
        """Start hython and wait until its session file is on disk."""
        if self.process is not None:
            raise RuntimeError("this bridge is already started")
        registry.ensure_registry_dir(self.home)
        command = [
            str(self.hython),
            "-m",
            MODULE,
            "--home",
            str(self.home),
            "--port",
            str(self.port_range[0]),
            "--max-port",
            str(self.port_range[1]),
            *self.extra_args,
        ]
        if self.alias:
            command += ["--alias", self.alias]
        # Output goes to a file, not a pipe: a pipe nobody reads fills up and
        # stops the process that is writing it.
        self._log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(  # noqa: S603 - the binary is ours to name
            command,
            stdin=subprocess.PIPE,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=self._child_env(),
        )
        self.entry = self._wait_for_entry(timeout_s)
        return self.entry

    def stop(self, *, timeout_s: float = STOP_TIMEOUT_S) -> int | None:
        """Ask the process to stop, then make sure it has."""
        process = self.process
        if process is None or process.returncode is not None:
            return None if process is None else process.returncode
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write(f"{STOP_WORD}\n")
                process.stdin.flush()
                process.stdin.close()
        except OSError:
            pass
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                code = process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=30.0)
        self._close_log()
        return code

    def output(self) -> str:
        """Whatever the process printed, for a failure that needs explaining."""
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> HythonBridge:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _entry(self) -> dict[str, Any]:
        if self.entry is None:
            raise RuntimeError("the bridge has not started")
        return self.entry

    def _child_env(self) -> dict[str, str]:
        environment = dict(os.environ if self.env is None else self.env)
        # The package has to be importable inside Houdini's own interpreter.
        source_root = str(Path(__file__).resolve().parents[2])
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root if not existing else os.pathsep.join([source_root, existing])
        )
        environment[HOME_ENV_VAR] = str(self.home)
        return environment

    def _wait_for_entry(self, timeout_s: float) -> dict[str, Any]:
        """Watch for the session file this process writes when it is ready."""
        process = self.process
        assert process is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            code = process.poll()
            for entry in registry.list_entries(self.home):
                if entry.get("pid") == process.pid:
                    return entry
            if code is not None:
                raise BridgeStartFailed(
                    f"hython exited with {code} before the bridge started:\n{self.output()}"
                )
            time.sleep(0.25)
        self.stop()
        raise BridgeStartFailed(f"no bridge after {timeout_s} seconds")

    def health(self, **rest: Any) -> client.Answer:
        return client.health(self.session, **rest)

    def call(self, tool: str, **rest: Any) -> client.Answer:
        return client.call(self.session, tool, **rest)
