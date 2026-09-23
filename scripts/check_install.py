#!/usr/bin/env python3
"""Check that a clean install from a built wheel works, with no checkout in reach.

What it does, all inside one temporary folder:

1. Builds the wheel with `uv build`.
2. Makes a throwaway virtual environment and installs the wheel into it.
3. Runs the installed `nscr-houdini-mcp` from that folder, never from the
   checkout, with the state folder, the Houdini preference folder and the
   packages folder all pointed inside it:
   - `bridge install --dry-run --packages-dir <tmp>`, then a real install
     into the same temporary folder, whose package file must point
     `NSCR_MCP_SRC` at the environment's site-packages and `NSCR_MCP_PAYLOAD`
     at the `houdini` folder inside the installed package;
   - `bridge status`;
   - `skills path`, which must name a folder inside the installed package;
   - `config init`, which must write the config file into the temporary home;
   - an MCP `initialize` and `tools/list` over stdio, which must list the
     eleven tools.
4. Checks that the installed package, not the checkout, is what Python imports.

Needs `uv` on the path. Nothing outside the temporary folder is written, and
the folder is removed at the end unless `--keep` is given or a check failed.
Exit code is 0 when every check passed and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PACKAGE = "nscr_houdini_mcp"
ENTRY_POINT = "nscr-houdini-mcp"
TOOLS = (
    "hou_ping",
    "hou_sessions",
    "hou_scene",
    "hou_inspect",
    "hou_python",
    "hou_jobs",
    "hou_node_type",
    "hou_docs",
    "hou_compare",
    "hou_outputs",
    "hou_capture",
)
PROTOCOL_VERSION = "2025-06-18"
STEP_TIMEOUT_S = 180.0
REPLY_TIMEOUT_S = 30.0


class Checks:
    """What passed and what did not, printed as it goes."""

    def __init__(self) -> None:
        self.failed: list[str] = []

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        print(f"{'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail else ""))
        if not ok:
            self.failed.append(what)
        return ok


def run(
    command: Sequence[str | Path], *, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    shown = " ".join(str(part) for part in command)
    print(f"$ {shown}")
    done = subprocess.run(
        [str(part) for part in command],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=STEP_TIMEOUT_S,
        check=False,
    )
    for line in (done.stdout + done.stderr).splitlines():
        print(f"    {line}")
    return done


def clean_env(tmp: Path) -> dict[str, str]:
    """This environment with nothing that could reach the checkout or real state."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "HOUDINI_PACKAGE_DIR")
        and not key.startswith("NSCR_MCP_")
    }
    env["NSCR_MCP_HOME"] = str(tmp / "home")
    env["HOUDINI_USER_PREF_DIR"] = str(tmp / "prefs" / "houdini__HVER__")
    env["PYTHONNOUSERSITE"] = "1"
    return env


def venv_bin(venv: Path, name: str) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def inside(path: str | Path, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def field(output: str, name: str) -> str | None:
    """The value on the line of `bridge install` output that starts with `name`."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(name + " "):
            return stripped[len(name) :].strip()
    return None


def talk_stdio(entry: Path, *, env: dict[str, str], cwd: Path) -> list[dict[str, Any]]:
    """`initialize`, `notifications/initialized` and `tools/list`, one line each."""
    child = subprocess.Popen(
        [str(entry)],
        env=env,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read() -> None:
        assert child.stdout is not None
        for line in child.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()

    def send(message: dict[str, Any]) -> None:
        assert child.stdin is not None
        child.stdin.write(json.dumps(message) + "\n")
        child.stdin.flush()

    def answer(request_id: int) -> dict[str, Any]:
        while True:
            line = lines.get(timeout=REPLY_TIMEOUT_S)
            if line is None:
                raise RuntimeError("the server closed its output before answering")
            message = json.loads(line)
            if message.get("id") == request_id:
                return message

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "check-install", "version": "0"},
                },
            }
        )
        first = answer(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        second = answer(2)
        return [first, second]
    finally:
        if child.stdin is not None:
            child.stdin.close()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        if child.stderr is not None:
            for line in child.stderr.read().splitlines():
                print(f"    server: {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default="3.11", help="the Python the environment is made with")
    parser.add_argument("--keep", action="store_true", help="leave the temporary folder behind")
    args = parser.parse_args(argv)

    if shutil.which("uv") is None:
        print("uv is not on the path; it builds the wheel and makes the environment")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="nscr-mcp-install-"))
    print(f"working in {tmp}")
    checks = Checks()
    try:
        _check(tmp, args.python, checks)
    except Exception as error:  # noqa: BLE001 - reported as the failure it is
        checks.check(False, "the check ran to the end", f"{type(error).__name__}: {error}")
    if checks.failed:
        print(f"\n{len(checks.failed)} checks failed; the folder is kept: {tmp}")
        return 1
    print("\nevery check passed")
    if args.keep:
        print(f"kept {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def _check(tmp: Path, python: str, checks: Checks) -> None:
    env = clean_env(tmp)
    work = tmp / "work"
    work.mkdir()

    built = run(["uv", "build", "--wheel", "--out-dir", tmp / "dist", REPO], env=env, cwd=work)
    wheels = sorted((tmp / "dist").glob("*.whl"))
    if not checks.check(built.returncode == 0 and len(wheels) == 1, "the wheel builds"):
        return
    wheel = wheels[0]

    venv = tmp / "venv"
    made = run(["uv", "venv", "--python", python, venv], env=env, cwd=work)
    if not checks.check(made.returncode == 0, "the environment is made"):
        return
    python_exe = venv_bin(venv, "python")
    added = run(["uv", "pip", "install", "--python", python_exe, wheel], env=env, cwd=work)
    if not checks.check(added.returncode == 0, "the wheel installs"):
        return

    where = run(
        [
            python_exe,
            "-c",
            "import json, sysconfig, nscr_houdini_mcp as m;"
            " print(json.dumps({'file': m.__file__, 'purelib': sysconfig.get_paths()['purelib']}))",
        ],
        env=env,
        cwd=work,
    )
    found = json.loads(where.stdout.strip().splitlines()[-1]) if where.returncode == 0 else {}
    site = Path(found.get("purelib", tmp / "missing"))
    checks.check(
        bool(found) and inside(found["file"], site) and not inside(found["file"], REPO),
        "the installed package is what Python imports",
        str(found.get("file")),
    )
    package_dir = site / PACKAGE
    checks.check(
        (package_dir / "houdini" / "packages").is_dir(), "the wheel carries the houdini folder"
    )
    checks.check(
        any((package_dir / "skills").glob("*/SKILL.md")), "the wheel carries the skills folder"
    )

    entry = venv_bin(venv, ENTRY_POINT)
    packages = tmp / "packages"

    dry = run(
        [entry, "bridge", "install", "--dry-run", "--packages-dir", packages], env=env, cwd=work
    )
    pythonpath = field(dry.stdout, "pythonpath")
    houdini_path = field(dry.stdout, "houdini path")
    checks.check(dry.returncode == 0, "bridge install --dry-run runs")
    checks.check(
        pythonpath is not None and inside(pythonpath, site) and not inside(pythonpath, REPO),
        "the dry run points the python path at site-packages",
        str(pythonpath),
    )
    checks.check(
        houdini_path is not None and inside(houdini_path, package_dir),
        "the dry run points the houdini path inside the installed package",
        str(houdini_path),
    )
    checks.check(not packages.exists() or not any(packages.iterdir()), "the dry run wrote nothing")

    real = run([entry, "bridge", "install", "--packages-dir", packages], env=env, cwd=work)
    package_file = packages / "nscr_houdini_mcp.json"
    variables: dict[str, str] = {}
    if real.returncode == 0 and package_file.is_file():
        for item in json.loads(package_file.read_text(encoding="utf-8")).get("env", []):
            if isinstance(item, dict):
                variables.update({k: v for k, v in item.items() if isinstance(v, str)})
    source = variables.get("NSCR_MCP_SRC")
    checks.check(
        source is not None
        and Path(source).resolve() == site.resolve()
        and not inside(source, REPO),
        "the package file points NSCR_MCP_SRC at site-packages",
        str(source),
    )
    payload = variables.get("NSCR_MCP_PAYLOAD")
    checks.check(
        payload is not None and Path(payload).resolve() == (package_dir / "houdini").resolve(),
        "the package file points NSCR_MCP_PAYLOAD at the installed houdini folder",
        str(payload),
    )

    status = run([entry, "bridge", "status", "--packages-dir", packages], env=env, cwd=work)
    checks.check(status.returncode == 0, "bridge status runs")

    removed = run([entry, "bridge", "uninstall", "--packages-dir", packages], env=env, cwd=work)
    checks.check(
        removed.returncode == 0 and not package_file.exists(), "bridge uninstall takes it away"
    )

    skills = run([entry, "skills", "path"], env=env, cwd=work)
    skills_dir = skills.stdout.strip().splitlines()[-1] if skills.stdout.strip() else ""
    checks.check(
        skills.returncode == 0 and bool(skills_dir) and inside(skills_dir, package_dir),
        "skills path names the installed skills",
        skills_dir,
    )

    config = run([entry, "config", "init"], env=env, cwd=work)
    config_file = tmp / "home" / "config.toml"
    checks.check(
        config.returncode == 0 and config_file.is_file(),
        "config init writes into the temporary home",
        str(config_file),
    )

    print(f"$ {entry} (stdio: initialize, tools/list)")
    first, second = talk_stdio(entry, env=env, cwd=work)
    server = first.get("result", {}).get("serverInfo", {})
    checks.check(
        server.get("name") == "nscr-houdini-mcp",
        "initialize answers as this server",
        f"{server.get('name')} {server.get('version')}",
    )
    names = tuple(tool.get("name") for tool in second.get("result", {}).get("tools", []))
    checks.check(names == TOOLS, "tools/list lists the eleven tools", ", ".join(map(str, names)))


if __name__ == "__main__":
    sys.exit(main())
