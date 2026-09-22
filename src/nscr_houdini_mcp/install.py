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

Finding the packages folder is its own job, because a preference folder is
often moved by something this command cannot see: a line in `houdini.env`, a
launcher, another package, a redirected documents folder. The order is: the
folder the caller named, `HOUDINI_PACKAGE_DIR` in this shell, what a real
Houdini says when one is asked, `HOUDINI_USER_PREF_DIR` in this shell, then
the usual folder for the system (`~/Library/Preferences/houdini/<v>` on macOS,
`Documents\\houdini<v>` under the profile on Windows, `~/houdini<v>` on Linux).
Every command reports which of those decided, and status lists them all.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_FILE_NAME = "nscr_houdini_mcp.json"
PACKAGES_DIR_NAME = "packages"

# Houdini treats a key starting with two slashes as a comment, so the marker
# rides along without meaning anything to it.
MARKER_KEY = "//nscr-houdini-mcp"
MARKER_VALUE = "written by nscr-houdini-mcp"

# The folders an install had to make, written into the file it made them for,
# so an uninstall can take away its own leftovers and nothing else.
CREATED_KEY = "//folders-this-made"

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
    """The shipped `houdini` folder, the one that goes on `HOUDINI_PATH`.

    An installed copy carries the folder inside the package, and that is the
    one preferred. A checkout has it at the top of the tree instead.
    """
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


def default_packages_dir(version: str = DEFAULT_HOUDINI_VERSION) -> Path:
    """The packages folder of the preference folder for this system."""
    return user_pref_dir(version) / PACKAGES_DIR_NAME


def packages_dir(
    version: str = DEFAULT_HOUDINI_VERSION,
    *,
    override: Path | str | None = None,
    ask_houdini: bool = True,
) -> Path:
    """The folder Houdini reads package files from, worked out in full."""
    return resolve(version, override=override, ask_houdini=ask_houdini).path


def package_path(
    version: str = DEFAULT_HOUDINI_VERSION,
    *,
    override: Path | str | None = None,
    ask_houdini: bool = True,
) -> Path:
    """The one file this tool ever writes for a version."""
    return packages_dir(version, override=override, ask_houdini=ask_houdini) / PACKAGE_FILE_NAME


def pref_dirs() -> list[tuple[str, Path]]:
    """Every Houdini preference folder on this machine, version and path.

    With `HOUDINI_USER_PREF_DIR` set there is exactly one, because that is
    what Houdini itself would use for any version.
    """
    override = os.environ.get(PREF_DIR_ENV_VAR)
    if override:
        return _pref_dirs_from(override)
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


def _pref_dirs_from(setting: str) -> list[tuple[str, Path]]:
    """Every folder the preference folder setting names on this machine.

    The setting carries the version token, so one setting stands for as many
    folders as there are Houdini versions here. Looking only at the default
    version would miss a package left behind by another one.
    """
    template = Path(setting).expanduser()
    if VERSION_TOKEN not in str(template):
        version = _version_in(template.name) or DEFAULT_HOUDINI_VERSION
        return [(version, template)]
    parent = Path(str(template.parent).replace(VERSION_TOKEN, DEFAULT_HOUDINI_VERSION))
    found = []
    if parent.is_dir():
        pattern = template.name.replace(VERSION_TOKEN, "*")
        for child in sorted(parent.glob(pattern)):
            version = _version_in(child.name)
            if child.is_dir() and version:
                found.append((version, child))
    if not found:
        default = Path(str(template).replace(VERSION_TOKEN, DEFAULT_HOUDINI_VERSION))
        return [(DEFAULT_HOUDINI_VERSION, default)]
    return found


def _version_in(name: str) -> str | None:
    match = _VERSION_IN_NAME.search(name)
    return match.group(1) if match else None


# Section: which packages folder this machine's Houdini really reads
#
# A preference folder in the usual place is the easy case. The variable that
# moves it is often set inside `houdini.env`, or by a launcher, or by another
# package, and a shell that runs this command sees none of that. So the
# lookup goes, in order: what the caller named, the package folder variable in
# this shell, what a real Houdini says when asked, the preference folder
# variable in this shell, and only then the usual place for the system.
# Nothing is remembered between runs: each command asks again.


PACKAGE_DIR_ENV_VAR = "HOUDINI_PACKAGE_DIR"
HSITE_ENV_VAR = "HSITE"

SOURCE_GIVEN = "--packages-dir"
SOURCE_PACKAGE_ENV = f"{PACKAGE_DIR_ENV_VAR} in this shell"
SOURCE_HOUDINI_PACKAGE = "HOUDINI_PACKAGE_DIR as Houdini reads it"
SOURCE_HOUDINI_HOME = "the home folder Houdini reports"
SOURCE_HSITE = "HSITE, which Houdini also scans"
SOURCE_PREF_ENV = f"{PREF_DIR_ENV_VAR} in this shell"
SOURCE_DEFAULT = "the usual folder for this system"

# How long one question to a Houdini may take. A cold hython is a few seconds,
# and a machine where it takes longer than this is one where the answer is not
# worth the wait: the lookup carries on without it and says so.
ASK_TIMEOUT_S = 10.0

# What the asking script prints its answer behind, so warnings and licence
# lines on the same stream cannot be mistaken for it.
ANSWER_MARKER = "nscr-houdini-mcp-answer "

ASK_SCRIPT = f"""
import json

import hou


def expand(text):
    try:
        return hou.text.expandString(text)
    except AttributeError:
        return hou.expandString(text)


print(
    {ANSWER_MARKER!r}
    + json.dumps(
        {{
            "home": hou.homeHoudiniDirectory(),
            "package_dir": expand("$HOUDINI_PACKAGE_DIR"),
            "user_pref_dir": expand("$HOUDINI_USER_PREF_DIR"),
            "hsite": expand("$HSITE"),
            "version": hou.applicationVersionString(),
        }}
    )
)
"""


@dataclass(frozen=True)
class HoudiniAnswer:
    """What a real Houdini said about its own folders."""

    home: Path
    package_dirs: list[Path]
    user_pref_dir: Path | None
    hsite: Path | None
    version: str
    hython: Path


@dataclass(frozen=True)
class Candidate:
    """One folder the lookup considered, and where the idea came from."""

    source: str
    path: Path
    used: bool = False
    note: str = ""


@dataclass(frozen=True)
class Lookup:
    """The folder to write into, and everything that led to it."""

    path: Path
    source: str
    candidates: list[Candidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def split_paths(value: str | None) -> list[Path]:
    """A path list variable as paths, keeping the order Houdini reads it in."""
    if not value:
        return []
    found = []
    for part in value.split(os.pathsep):
        text = part.strip()
        if text and text != "&":
            found.append(Path(text).expanduser())
    return found


def ask_houdini(version: str = DEFAULT_HOUDINI_VERSION) -> tuple[HoudiniAnswer | None, str]:
    """Ask a Houdini on this machine where it reads packages from.

    Returns the answer and a line saying what happened, because a lookup that
    could not ask has to say so rather than quietly move on. A Houdini that is
    not there, will not start, or takes too long is not an error: the lookup
    carries on with what it can work out by itself.
    """
    installs = find_installs()
    if not installs:
        return None, "no Houdini found to ask, so its own folders could not be read"
    hython = installs[0].hfs / "bin" / ("hython.exe" if sys.platform == "win32" else "hython")
    if not hython.is_file():
        return None, f"no hython at {hython}, so Houdini's own folders could not be read"
    try:
        finished = subprocess.run(  # noqa: S603 - the binary is this machine's Houdini
            [str(hython), "-c", ASK_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=ASK_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"{hython} did not answer within {ASK_TIMEOUT_S:.0f} seconds"
    except OSError as error:
        return None, f"{hython} could not be run: {error}"
    answer = _read_answer(finished.stdout, hython)
    if answer is None:
        first = (finished.stderr or finished.stdout or "").strip().splitlines()
        reason = first[-1] if first else f"exit code {finished.returncode}"
        return None, f"{hython} gave no answer: {reason}"
    return answer, f"asked {hython}"


def _read_answer(printed: str, hython: Path) -> HoudiniAnswer | None:
    for line in (printed or "").splitlines():
        if not line.startswith(ANSWER_MARKER):
            continue
        try:
            loaded = json.loads(line[len(ANSWER_MARKER) :])
        except ValueError:
            return None
        if not isinstance(loaded, dict) or not loaded.get("home"):
            return None
        pref = loaded.get("user_pref_dir")
        site = loaded.get("hsite")
        return HoudiniAnswer(
            home=Path(str(loaded["home"])),
            package_dirs=split_paths(loaded.get("package_dir")),
            user_pref_dir=Path(str(pref)) if pref else None,
            hsite=Path(str(site)) if site else None,
            version=str(loaded.get("version") or ""),
            hython=hython,
        )
    return None


def resolve(
    version: str = DEFAULT_HOUDINI_VERSION,
    *,
    override: Path | str | None = None,
    ask_houdini: bool = True,
) -> Lookup:
    """Work out which packages folder to write into, and show the working.

    Houdini reads every folder on its package path, so when a variable names
    several the first is written into and the rest are reported.
    """
    candidates: list[Candidate] = []
    notes: list[str] = []

    if override is not None:
        chosen = Path(override).expanduser()
        candidates.append(Candidate(SOURCE_GIVEN, chosen, used=True))
        return Lookup(chosen, SOURCE_GIVEN, candidates, notes)

    from_shell = split_paths(os.environ.get(PACKAGE_DIR_ENV_VAR))
    if from_shell:
        _add(candidates, SOURCE_PACKAGE_ENV, from_shell)
        return Lookup(from_shell[0], SOURCE_PACKAGE_ENV, candidates, notes)

    if ask_houdini:
        answer, note = _ask(version)
        notes.append(note)
        if answer is not None:
            if answer.package_dirs:
                _add(candidates, SOURCE_HOUDINI_PACKAGE, answer.package_dirs)
                source = SOURCE_HOUDINI_PACKAGE
                chosen = answer.package_dirs[0]
            else:
                chosen = answer.home / PACKAGES_DIR_NAME
                source = SOURCE_HOUDINI_HOME
                candidates.append(Candidate(SOURCE_HOUDINI_HOME, chosen, used=True))
            _add_hsite(candidates, answer, version)
            return Lookup(chosen, source, candidates, notes)

    from_pref = os.environ.get(PREF_DIR_ENV_VAR)
    if from_pref:
        chosen = Path(from_pref.replace(VERSION_TOKEN, version)).expanduser() / PACKAGES_DIR_NAME
        candidates.append(Candidate(SOURCE_PREF_ENV, chosen, used=True))
        return Lookup(chosen, SOURCE_PREF_ENV, candidates, notes)

    chosen = default_packages_dir(version)
    candidates.append(Candidate(SOURCE_DEFAULT, chosen, used=True))
    return Lookup(chosen, SOURCE_DEFAULT, candidates, notes)


def _ask(version: str) -> tuple[HoudiniAnswer | None, str]:
    """The question, as one call, so a test can answer it without a Houdini."""
    return ask_houdini(version)


def _add_hsite(candidates: list[Candidate], answer: HoudiniAnswer, version: str) -> None:
    """Report the site folder Houdini also scans, and never write into it."""
    if not answer.hsite:
        return
    short = _short(answer.version, version)
    candidates.append(
        Candidate(
            SOURCE_HSITE,
            answer.hsite / f"houdini{short}" / PACKAGES_DIR_NAME,
            note="shared with other people, so nothing is written here",
        )
    )


def _add(candidates: list[Candidate], source: str, paths: list[Path]) -> None:
    """The first of a path list is written into, the rest are reported."""
    for index, path in enumerate(paths):
        candidates.append(
            Candidate(
                source,
                path,
                used=index == 0,
                note="" if index == 0 else "also scanned by Houdini",
            )
        )


def _short(reported: str, fallback: str) -> str:
    version = _version_in(reported) or fallback
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 else version


# Section: the package file


# Characters Houdini reads as something else inside a package value. A path
# holding one cannot be written down as it stands, and writing it anyway would
# put Houdini on a folder nobody named.
UNWRITABLE = ("$", "`")


def check_writable(path: Path, what: str) -> None:
    """Refuse a path Houdini would read as an expression rather than a path."""
    text = str(path)
    for character in UNWRITABLE:
        if character in text:
            raise InstallError(
                f"the {what} folder has a {character} in its name, which Houdini reads as"
                f" something to expand rather than as part of the path: {text}"
            )


def document(
    *,
    autostart: bool = False,
    source: Path | None = None,
    payload: Path | None = None,
    created: list[Path] | None = None,
) -> dict[str, Any]:
    """The package Houdini reads, as data.

    `HOUDINI_PATH` gets the payload folder through `hpath`, so Houdini runs
    the startup files in it. `PYTHONPATH` gets the source folder, so the
    bridge is importable in Houdini's own interpreter. Nothing opens a port
    unless the auto start variable is on.

    The folders this install had to make are written down, so an uninstall can
    take away what it made and nothing else.
    """
    source = Path(source) if source is not None else source_root()
    payload = Path(payload) if payload is not None else payload_root()
    check_writable(source, "source")
    check_writable(payload, "payload")
    body: dict[str, Any] = {
        MARKER_KEY: MARKER_VALUE,
        "//note": "Written by the bridge install command. Edits here are lost on the next one.",
        "enable": True,
        "env": [
            {PAYLOAD_ENV_VAR: str(payload)},
            {SOURCE_ENV_VAR: str(source)},
            {AUTOSTART_ENV_VAR: "1" if autostart else "0"},
            {"PYTHONPATH": {"value": f"${SOURCE_ENV_VAR}", "method": "prepend"}},
        ],
        "hpath": f"${PAYLOAD_ENV_VAR}",
    }
    if created:
        body[CREATED_KEY] = [str(folder) for folder in created]
    return body


def read_document(path: Path) -> dict[str, Any] | None:
    """One package file as data, or nothing when it cannot be read as one."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def is_ours(path: Path) -> bool:
    """Whether this tool wrote the file at that path.

    It has to be a plain file, not a link to one somewhere else, and it has to
    be shaped like the package this writes, not merely carry the marker. A
    file that is not there is not ours either. Anything else belongs to
    somebody else and is left exactly as it is.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return False
    loaded = read_document(path)
    if not loaded or loaded.get(MARKER_KEY) != MARKER_VALUE:
        return False
    return isinstance(loaded.get("env"), list) and "enable" in loaded


def created_folders(loaded: dict[str, Any] | None) -> list[Path]:
    """The folders the install that wrote this file had to make."""
    listed = (loaded or {}).get(CREATED_KEY)
    if not isinstance(listed, list):
        return []
    return [Path(str(item)) for item in listed]


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
    lookup: Lookup | None = None
    lines: list[str] = field(default_factory=list)


def install(
    version: str = DEFAULT_HOUDINI_VERSION,
    *,
    autostart: bool = False,
    dry_run: bool = False,
    packages: Path | str | None = None,
    lookup: Lookup | None = None,
) -> InstallResult:
    """Write the package file for one Houdini version.

    A file already there and carrying the marker is replaced. One that is not
    ours raises, and nothing on disk is touched. `packages` names the folder
    outright; without it the folder is worked out, and a caller that has
    already worked it out passes that `lookup` rather than asking again.
    """
    source = source_root()
    payload = payload_root()
    found = lookup or resolve(version, override=packages)
    path = found.path / PACKAGE_FILE_NAME
    # A link is never written through. Following one would write the package
    # wherever it points, which is outside the packages folder, and a link
    # that points nowhere would pass for a file that is not there at all.
    if path.is_symlink():
        raise NotOurs(f"{path} is a link, so it is left alone")
    exists = path.exists()
    if exists and not is_ours(path):
        raise NotOurs(f"{path} was not written by this tool, so it is left alone")
    would_make = _missing_folders(path.parent)
    body = document(autostart=autostart, source=source, payload=payload, created=would_make)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(path, json.dumps(body, indent=4, ensure_ascii=False) + "\n")
    return InstallResult(
        path=path,
        version=version,
        autostart=autostart,
        source=source,
        payload=payload,
        written=not dry_run,
        replaced=exists,
        dry_run=dry_run,
        lookup=found,
        lines=[
            f"package       {path}",
            f"folder from   {found.source}",
            f"houdini        {version}",
            f"houdini path   {payload}",
            f"pythonpath     {source}",
            f"{AUTOSTART_ENV_VAR}   {'1' if autostart else '0'}",
        ],
    )


def _missing_folders(folder: Path) -> list[Path]:
    """The folders that would have to be made to hold a file in this one.

    Deepest first, which is the order they can be taken away again in.
    """
    missing = []
    walk = folder
    while not walk.exists() and walk != walk.parent:
        missing.append(walk)
        walk = walk.parent
    return missing


def _write_atomically(path: Path, text: str) -> None:
    """Write beside the file, then move it into place in one step.

    A half written package is one Houdini would read and refuse, and moving
    over a name never follows a link that appears in the meantime.
    """
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as opened:
            opened.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class RemovedPackage:
    """One package file an uninstall looked at."""

    path: Path
    version: str
    removed: bool
    reason: str


def uninstall(
    version: str | None = None,
    *,
    packages: Path | str | None = None,
    lookup: Lookup | None = None,
) -> list[RemovedPackage]:
    """Take away the package files this tool wrote.

    Every folder this machine could be reading packages from is looked at, so
    a file left in the old place is found after the folder has moved. A file
    of the same name that this tool did not write is reported and kept.
    """
    targets = _targets(version, packages=packages, lookup=lookup)
    seen: list[Path] = []
    results = []
    for found_version, folder in targets:
        path = folder / PACKAGE_FILE_NAME
        if path in seen:
            continue
        seen.append(path)
        if path.is_symlink():
            results.append(RemovedPackage(path, found_version, False, "a link, kept"))
            continue
        if not path.exists():
            continue
        if not is_ours(path):
            results.append(
                RemovedPackage(path, found_version, False, "not written by this tool, kept")
            )
            continue
        made = created_folders(read_document(path))
        path.unlink()
        results.append(RemovedPackage(path, found_version, True, "removed"))
        _remove_empty(made)
    return results


def _remove_empty(folders: list[Path]) -> None:
    """Take away the folders that install made, while they are empty.

    Only folders the package file itself named, so a packages folder that was
    already there when this arrived is never touched.
    """
    for folder in folders:
        try:
            if folder.is_dir() and not folder.is_symlink() and not any(folder.iterdir()):
                folder.rmdir()
        except OSError:
            return


def _targets(
    version: str | None,
    *,
    packages: Path | str | None = None,
    lookup: Lookup | None = None,
) -> list[tuple[str, Path]]:
    """Every folder to look in, the one that would be written into first.

    Uninstall and status both work over this list. A folder Houdini names is
    where the file is now; the per version preference folders are where an
    earlier install may have left one.
    """
    wanted = version or DEFAULT_HOUDINI_VERSION
    found = lookup or resolve(wanted, override=packages)
    targets = [(wanted, candidate.path) for candidate in found.candidates]
    if version is None and packages is None:
        targets += [(known, path / PACKAGES_DIR_NAME) for known, path in pref_dirs()]
    seen: list[Path] = []
    unique = []
    for known, path in targets:
        if path in seen:
            continue
        seen.append(path)
        unique.append((known, path))
    return unique


@dataclass(frozen=True)
class InstalledPackage:
    """Whether one Houdini version has this package, and which one."""

    version: str
    path: Path
    present: bool
    ours: bool
    autostart: bool | None


def installed(
    version: str | None = None,
    *,
    packages: Path | str | None = None,
    lookup: Lookup | None = None,
) -> list[InstalledPackage]:
    """The package state of every folder this machine may read packages from."""
    targets = _targets(version, packages=packages, lookup=lookup)
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
