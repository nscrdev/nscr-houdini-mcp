"""The Houdini package file that puts the bridge on a session's path.

A Houdini package is one JSON file in a folder Houdini reads at startup. This
module writes that file, takes it away again, and finds the Houdini installs
on this machine. It imports no `hou` and touches nothing outside the packages
folder of the Houdini version it was asked about.

Two rules stand behind everything here:

- Every file this writes carries a marker key. A file without that marker was
  put there by somebody else, so it is reported and left exactly as it is.
  Nothing is merged into another package and nothing is overwritten blind.
- Paths are read from this module's own location rather than written down, so
  a package file names the source folder this copy is actually running from,
  whether that is a checkout or an installed one.

The packages folder is `HOUDINI_USER_PREF_DIR` when that is set, and otherwise
the usual per user folder for the system: `~/Library/Preferences/houdini/<v>`
on macOS, `Documents\\houdini<v>` under the profile on Windows, `~/houdini<v>`
on Linux.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_FILE_NAME = "nscr_houdini_mcp.json"
PACKAGES_DIR_NAME = "packages"

# Houdini treats a key starting with two slashes as a comment, so the marker
# rides along without meaning anything to it.
MARKER_KEY = "//nscr-houdini-mcp"
MARKER_VALUE = "written by nscr-houdini-mcp"

DEFAULT_HOUDINI_VERSION = "22.0"

PREF_DIR_ENV_VAR = "HOUDINI_USER_PREF_DIR"
HFS_ENV_VAR = "HFS"

PAYLOAD_ENV_VAR = "NSCR_MCP_PAYLOAD"
SOURCE_ENV_VAR = "NSCR_MCP_SRC"
AUTOSTART_ENV_VAR = "NSCR_MCP_AUTOSTART"

# What Houdini itself puts in place of the version in a pref dir setting.
VERSION_TOKEN = "__HVER__"

_VERSION_IN_NAME = re.compile(r"(\d+\.\d+(?:\.\d+)*)")


class InstallError(Exception):
    """Something the caller has to fix before a package can be written."""


class PayloadMissing(InstallError):
    """The Houdini side files are not next to this copy of the package."""


class NotOurs(InstallError):
    """A package file of that name is there and this tool did not write it."""


# Section: where things are


def source_root() -> Path:
    """The folder that has to be on `PYTHONPATH` for `import nscr_houdini_mcp`."""
    return Path(__file__).resolve().parent.parent


def payload_root() -> Path:
    """The shipped `houdini` folder, the one that goes on `HOUDINI_PATH`."""
    here = Path(__file__).resolve()
    for candidate in (here.parent / "houdini", here.parents[2] / "houdini"):
        if (candidate / PACKAGES_DIR_NAME).is_dir():
            return candidate
    raise PayloadMissing(
        "no houdini payload folder next to this package, so there is nothing to point Houdini at"
    )


def user_pref_dir(version: str = DEFAULT_HOUDINI_VERSION) -> Path:
    """Houdini's per user folder for one version.

    `HOUDINI_USER_PREF_DIR` wins when it is set, which is how a test, or an
    artist with a moved preference folder, keeps this away from the real one.
    """
    override = os.environ.get(PREF_DIR_ENV_VAR)
    if override:
        return Path(override.replace(VERSION_TOKEN, version)).expanduser()
    if sys.platform == "win32":
        profile = os.environ.get("USERPROFILE")
        home = Path(profile) if profile else Path.home()
        return home / "Documents" / f"houdini{version}"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Preferences" / "houdini" / version
    return Path.home() / f"houdini{version}"


def packages_dir(version: str = DEFAULT_HOUDINI_VERSION) -> Path:
    """The folder Houdini reads package files from."""
    return user_pref_dir(version) / PACKAGES_DIR_NAME


def package_path(version: str = DEFAULT_HOUDINI_VERSION) -> Path:
    """The one file this tool ever writes for a version."""
    return packages_dir(version) / PACKAGE_FILE_NAME


def pref_dirs() -> list[tuple[str, Path]]:
    """Every Houdini preference folder on this machine, version and path.

    With `HOUDINI_USER_PREF_DIR` set there is exactly one, because that is
    what Houdini itself would use for any version.
    """
    override = os.environ.get(PREF_DIR_ENV_VAR)
    if override:
        path = Path(override.replace(VERSION_TOKEN, DEFAULT_HOUDINI_VERSION)).expanduser()
        return [(_version_in(path.name) or DEFAULT_HOUDINI_VERSION, path)]
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Preferences" / "houdini"
        pattern = "*"
    elif sys.platform == "win32":
        profile = os.environ.get("USERPROFILE")
        root = (Path(profile) if profile else Path.home()) / "Documents"
        pattern = "houdini*"
    else:
        root = Path.home()
        pattern = "houdini*"
    found = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not child.match(pattern):
                continue
            version = _version_in(child.name)
            if version:
                found.append((version, child))
    return found


def _version_in(name: str) -> str | None:
    match = _VERSION_IN_NAME.search(name)
    return match.group(1) if match else None


# Section: the package file


def document(
    *,
    autostart: bool = False,
    source: Path | None = None,
    payload: Path | None = None,
) -> dict[str, Any]:
    """The package Houdini reads, as data.

    `HOUDINI_PATH` gets the payload folder, so Houdini runs the startup script
    in it. `PYTHONPATH` gets the source folder, so the bridge is importable in
    Houdini's own interpreter. The startup script opens no port unless the
    auto start variable is on.
    """
    source = Path(source) if source is not None else source_root()
    payload = Path(payload) if payload is not None else payload_root()
    return {
        MARKER_KEY: MARKER_VALUE,
        "//note": "Written by the bridge install command. Edits here are lost on the next one.",
        "enable": True,
        "env": [
            {PAYLOAD_ENV_VAR: str(payload)},
            {SOURCE_ENV_VAR: str(source)},
            {AUTOSTART_ENV_VAR: "1" if autostart else "0"},
            {"PYTHONPATH": {"value": f"${SOURCE_ENV_VAR}", "method": "prepend"}},
        ],
        "path": f"${PAYLOAD_ENV_VAR}",
    }


def read_document(path: Path) -> dict[str, Any] | None:
    """One package file as data, or nothing when it cannot be read as one."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def is_ours(path: Path) -> bool:
    """Whether this tool wrote the file at that path.

    A file that is not there is not ours either. Anything unreadable, or
    readable and without the marker, belongs to somebody else and is left be.
    """
    loaded = read_document(path)
    return bool(loaded) and loaded.get(MARKER_KEY) == MARKER_VALUE


@dataclass(frozen=True)
class InstallResult:
    """What one install did, or would have done."""

    path: Path
    version: str
    autostart: bool
    source: Path
    payload: Path
    written: bool
    replaced: bool
    dry_run: bool
    lines: list[str] = field(default_factory=list)


def install(
    version: str = DEFAULT_HOUDINI_VERSION,
    *,
    autostart: bool = False,
    dry_run: bool = False,
) -> InstallResult:
    """Write the package file for one Houdini version.

    A file already there and carrying the marker is replaced. One that is not
    ours raises, and nothing on disk is touched.
    """
    source = source_root()
    payload = payload_root()
    path = package_path(version)
    exists = path.exists()
    if exists and not is_ours(path):
        raise NotOurs(f"{path} was not written by this tool, so it is left alone")
    body = document(autostart=autostart, source=source, payload=payload)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(body, indent=4, sort_keys=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return InstallResult(
        path=path,
        version=version,
        autostart=autostart,
        source=source,
        payload=payload,
        written=not dry_run,
        replaced=exists,
        dry_run=dry_run,
        lines=[
            f"package       {path}",
            f"houdini        {version}",
            f"houdini path   {payload}",
            f"pythonpath     {source}",
            f"{AUTOSTART_ENV_VAR}   {'1' if autostart else '0'}",
        ],
    )


@dataclass(frozen=True)
class RemovedPackage:
    """One package file an uninstall looked at."""

    path: Path
    version: str
    removed: bool
    reason: str


def uninstall(version: str | None = None) -> list[RemovedPackage]:
    """Take away the package files this tool wrote.

    With no version, every preference folder on this machine is looked at. A
    file of the same name that this tool did not write is reported and kept.
    """
    targets = [(version, packages_dir(version))] if version else _every_packages_dir()
    seen: list[Path] = []
    results = []
    for found_version, folder in targets:
        path = folder / PACKAGE_FILE_NAME
        if path in seen:
            continue
        seen.append(path)
        if not path.exists():
            continue
        if not is_ours(path):
            results.append(
                RemovedPackage(path, found_version, False, "not written by this tool, kept")
            )
            continue
        path.unlink()
        results.append(RemovedPackage(path, found_version, True, "removed"))
    return results


def _every_packages_dir() -> list[tuple[str, Path]]:
    return [(version, path / PACKAGES_DIR_NAME) for version, path in pref_dirs()]


@dataclass(frozen=True)
class InstalledPackage:
    """Whether one Houdini version has this package, and which one."""

    version: str
    path: Path
    present: bool
    ours: bool
    autostart: bool | None


def installed(version: str | None = None) -> list[InstalledPackage]:
    """The package state of every Houdini preference folder found."""
    targets = [(version, packages_dir(version))] if version else _every_packages_dir()
    states = []
    for found_version, folder in targets:
        path = folder / PACKAGE_FILE_NAME
        loaded = read_document(path) if path.exists() else None
        ours = bool(loaded) and loaded.get(MARKER_KEY) == MARKER_VALUE
        states.append(
            InstalledPackage(
                version=found_version,
                path=path,
                present=path.exists(),
                ours=ours,
                autostart=_autostart_of(loaded) if ours else None,
            )
        )
    return states


def _autostart_of(loaded: dict[str, Any] | None) -> bool | None:
    for item in (loaded or {}).get("env") or []:
        if isinstance(item, dict) and AUTOSTART_ENV_VAR in item:
            return str(item[AUTOSTART_ENV_VAR]).strip().lower() in ("1", "true", "yes", "on")
    return None


# Section: the Houdini installs on this machine


@dataclass(frozen=True)
class HoudiniInstall:
    """One Houdini on this machine."""

    version: str
    root: Path
    hfs: Path

    @property
    def short_version(self) -> str:
        """The two part version, which is what names a preference folder."""
        parts = self.version.split(".")
        return ".".join(parts[:2]) if len(parts) > 1 else self.version


def install_roots() -> list[Path]:
    """Where Houdini installs usually sit on this system."""
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


def hfs_of(root: Path) -> Path:
    """The folder inside an install that holds `bin` and the version file."""
    if sys.platform == "darwin":
        return root / "Frameworks" / "Houdini.framework" / "Versions" / "Current" / "Resources"
    return root


def find_installs(configured: Path | str | None = None) -> list[HoudiniInstall]:
    """Every Houdini this machine seems to have, newest first.

    A configured path, or `HFS`, is taken as given and comes first. The rest
    is the platform's usual install folder: `Houdini*` under
    `/Applications/Houdini`, `Houdini *` under Side Effects Software, `hfs*`
    under `/opt`.
    """
    found: list[HoudiniInstall] = []
    named = configured or os.environ.get(HFS_ENV_VAR)
    if named:
        hfs = Path(named).expanduser()
        found.append(HoudiniInstall(_version_in(hfs.name) or "", hfs.parent, hfs))
    loose: list[HoudiniInstall] = []
    for root in install_roots():
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name.lower()
            if not (name.startswith("houdini") or name.startswith("hfs")):
                continue
            version = _version_in(child.name)
            if not version:
                continue
            loose.append(HoudiniInstall(version, child, hfs_of(child)))
    loose.sort(key=lambda item: _sort_key(item.version), reverse=True)
    for item in loose:
        if all(item.hfs != seen.hfs for seen in found):
            found.append(item)
    return found


def _sort_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


# Section: starting a bridge by hand


def snippet(source: Path | None = None) -> str:
    """Python to paste into the shell of a Houdini that is already open.

    The path is worked out when this prints, so the snippet names the source
    folder of the copy the artist is running.
    """
    root = Path(source) if source is not None else source_root()
    return "\n".join(
        [
            "import sys",
            f"root = {str(root)!r}",
            "if root not in sys.path:",
            "    sys.path.insert(0, root)",
            "import hou",
            "from nscr_houdini_mcp.bridge import Bridge",
            "bridge = Bridge()",
            "record = bridge.start()",
            "hou.session.nscr_mcp_bridge = bridge",
            "print(record.alias, record.session_id, bridge.port)",
        ]
    )
