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
     `NSCR_MCP_PAYLOAD` at the `houdini` folder inside the installed package
     and `NSCR_MCP_SRC` at a folder that holds this package and nothing else
     of the environment (no numpy, no PIL, no mcp), never at site-packages,
     since whatever else is there would load in place of Houdini's own;
   - that folder imported by a Python that is not the environment's, with no
     site-packages at all, which must load the bridge and nothing from
     outside the folder and that Python's own library (and, with `--hython`,
     the same in a real Houdini, whose numpy must be its own);
   - `bridge status`, then `bridge uninstall`, which must take the folder
     away with the package file;
   - `skills path`, which must name a folder inside the installed package;
   - `config init`, which must write the config file into the temporary home;
   - an MCP `initialize` and `tools/list` over stdio, which must list the
     eleven tools.
4. Checks that the installed package, not the checkout, is what Python imports.

Needs `uv` on the path. This script writes nothing outside the temporary
folder, but uv keeps its usual cache as it builds and installs, and may
download a Python to make the environment with. The folder is removed at the
end unless `--keep` is given or a check failed.
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

# Libraries of the environment that must never reach Houdini's path.
NEVER_BESIDE = ("numpy", "PIL", "mcp", "pydantic", "anyio")

# What the bridge side imports inside Houdini: the autostart, the bridge and
# the hython worker's own entry point.
BRIDGE_MODULES = (
    "nscr_houdini_mcp",
    "nscr_houdini_mcp.bridge",
    "nscr_houdini_mcp.bridge.app",
    "nscr_houdini_mcp.bridge.net",
    "nscr_houdini_mcp.bridge.main",
)

IMPORT_PROBE = f"""
import json
import os
import sys
import sysconfig

import importlib

for name in {BRIDGE_MODULES!r}:
    importlib.import_module(name)

folder = os.path.realpath(sys.argv[1])
allowed = [folder] + [
    os.path.realpath(sysconfig.get_paths()[key]) for key in ("stdlib", "platstdlib")
] + ([os.path.realpath(os.environ["HFS"])] if os.environ.get("HFS") else [])
outside = sorted(
    name
    for name, module in list(sys.modules.items())
    if name != "__main__"
    and getattr(module, "__file__", None)
    and not any(
        os.path.realpath(module.__file__).startswith(root + os.sep) for root in allowed
    )
)
numpy_file = None
try:
    import numpy

    numpy_file = numpy.__file__
except ImportError:
    pass
print(
    "probe "
    + json.dumps(
        {{
            "package": sys.modules["nscr_houdini_mcp"].__file__,
            "outside": outside,
            "numpy": numpy_file,
            "hfs": os.environ.get("HFS"),
            "prefix": sys.prefix,
            "python": sys.version.split()[0],
        }}
    )
)
"""


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


def probe(
    python: Sequence[str | Path], folder: Path, *, env: dict[str, str], cwd: Path
) -> dict[str, Any]:
    """What a Python loads when `folder` is all it has besides its own library."""
    probe_env = dict(env)
    probe_env["PYTHONPATH"] = str(folder)
    script = cwd / "import_probe.py"
    script.write_text(IMPORT_PROBE, encoding="utf-8")
    done = run([*python, script, folder], env=probe_env, cwd=cwd)
    for line in done.stdout.splitlines():
        if line.startswith("probe "):
            return json.loads(line[len("probe ") :])
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default="3.11", help="the Python the environment is made with")
    parser.add_argument("--keep", action="store_true", help="leave the temporary folder behind")
    parser.add_argument(
        "--hython", type=Path, default=None, help="also import the folder in this Houdini"
    )
    args = parser.parse_args(argv)

    if shutil.which("uv") is None:
        print("uv is not on the path; it builds the wheel and makes the environment")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="nscr-mcp-install-"))
    print(f"working in {tmp}")
    checks = Checks()
    try:
        _check(tmp, args.python, checks, args.hython)
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


def _check(tmp: Path, python: str, checks: Checks, hython: Path | None) -> None:
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
        all((site / name).is_dir() for name in NEVER_BESIDE),
        "site-packages holds the libraries Houdini must never see",
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
    copies = tmp / "home" / "houdini-python"
    checks.check(
        pythonpath is not None
        and inside(pythonpath, copies)
        and not inside(pythonpath, site)
        and not inside(pythonpath, REPO),
        "the dry run points the python path at a folder of its own, not site-packages",
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
        and Path(source).resolve() != site.resolve()
        and inside(source, copies)
        and not inside(source, REPO),
        "the package file points NSCR_MCP_SRC at a folder of its own, not site-packages",
        str(source),
    )
    source_dir = Path(source) if source else tmp / "missing"
    held = sorted(child.name for child in source_dir.iterdir()) if source_dir.is_dir() else []
    importable = [name for name in held if not name.startswith(".")]
    checks.check(
        importable == [PACKAGE],
        "that folder holds this package and nothing else",
        ", ".join(held),
    )
    checks.check(
        all((site / name).is_dir() for name in NEVER_BESIDE)
        and not any((source_dir / name).exists() for name in NEVER_BESIDE),
        "the environment's numpy, PIL, mcp, pydantic and anyio stay out of it",
    )
    checks.check(
        (source_dir / PACKAGE / "bridge" / "app.py").is_file(),
        "the copy there holds the bridge",
    )

    # A Python that is not the environment's, with no site-packages of any
    # kind, stands in for Houdini's: the bridge has to import from the folder
    # alone and bring nothing in from anywhere but it and the standard library.
    found_probe = probe([sys.executable, "-S", "-s"], source_dir, env=env, cwd=work)
    checks.check(
        bool(found_probe)
        and inside(found_probe.get("package", ""), source_dir)
        and not found_probe.get("outside"),
        f"the bridge imports under Python {found_probe.get('python')} from that folder alone",
        f"outside: {found_probe.get('outside')}" if found_probe else "no answer",
    )
    checks.check(
        bool(found_probe) and found_probe.get("numpy") is None,
        "no numpy is reachable through that folder",
        str(found_probe.get("numpy")),
    )
    if hython is not None:
        in_houdini = probe([hython], source_dir, env=env, cwd=work)
        # Houdini's own libraries live under $HFS, or under the Python it ships
        # with, which on macOS sits beside $HFS inside the same install.
        own = [Path(root) for root in (in_houdini.get("hfs"), in_houdini.get("prefix")) if root]
        checks.check(
            bool(in_houdini) and inside(in_houdini.get("package", ""), source_dir),
            "hython imports the bridge from that folder",
            str(in_houdini.get("package")),
        )
        checks.check(
            any(inside(in_houdini.get("numpy") or "", root) for root in own),
            "hython's numpy is Houdini's own",
            str(in_houdini.get("numpy")),
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
    checks.check(not source_dir.exists(), "bridge uninstall takes the folder away too")

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
