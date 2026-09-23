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
- Houdini's Python is given this package and nothing else. An installed copy
  lives in a site-packages folder next to every other library of its
  environment, and putting that folder in front of Houdini's own would load
  those libraries in place of Houdini's. So a source folder that holds anything
  importable besides this package is never named: the package is copied into a
  folder of its own under the state folder, and that copy is what Houdini gets.

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

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import store as store_module

PACKAGE_NAME = "nscr_houdini_mcp"
PACKAGE_FILE_NAME = "nscr_houdini_mcp.json"
PACKAGES_DIR_NAME = "packages"

# Houdini treats a key starting with two slashes as a comment, so the marker
# rides along without meaning anything to it.
MARKER_KEY = "//nscr-houdini-mcp"
MARKER_VALUE = "written by nscr-houdini-mcp"

# The folders an install had to make, written into the file it made them for,
# so an uninstall can take away its own leftovers and nothing else.
CREATED_KEY = "//folders-this-made"

# The copy of the package this install made for Houdini to import, written
# into the package file so an uninstall takes it away with the file.
COPY_KEY = "//python-copy-this-made"

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


# Section: the folder Houdini's Python imports this package from
#
# Whatever folder goes on Houdini's `PYTHONPATH` is searched before Houdini's
# own libraries. It must hold this package and nothing else, or a library of
# this tool's environment (numpy built for another Python, say) is loaded in
# place of the one Houdini ships. A checkout's `src` holds only this package
# and is named as it is, so an edit there reaches the next Houdini. Any other
# source folder is copied: into a folder owned by one package file for
# `bridge install`, refreshed on every install and removed on uninstall, or
# into one owned by the source folder for the sessions this tool starts
# itself, refreshed whenever the source has changed. A copy rather than a
# link, because a link on Windows needs rights an artist may not have.

COPIES_DIR_NAME = "houdini-python"
STAMP_FILE_NAME = ".nscr-houdini-mcp.json"

# File endings Python imports a module from, on any system, and `.pth`, which
# a site folder reads to add yet more paths.
_MODULE_SUFFIXES = (".py", ".pyc", ".pyw", ".pyd", ".so", ".pth")

# Never imported as a module: it only caches the files next to it.
_NOT_A_MODULE = ("__pycache__",)


def package_root() -> Path:
    """The folder of this package itself."""
    return Path(__file__).resolve().parent


def strays(folder: Path) -> list[str]:
    """Everything a Python could import from this folder besides this package.

    A folder is importable when its name is a Python name, with or without an
    `__init__.py`, because a folder without one is still a namespace package.
    A file is when its name starts with a Python name and ends like a module.
    """
    found = []
    for child in sorted(Path(folder).iterdir(), key=lambda item: item.name):
        name = child.name
        if name == PACKAGE_NAME or name in _NOT_A_MODULE:
            continue
        if child.is_dir():
            if name.isidentifier():
                found.append(name)
        elif name.endswith(".pth"):
            found.append(name)
        elif name.split(".", 1)[0].isidentifier() and name.endswith(_MODULE_SUFFIXES):
            found.append(name)
    return found


def fingerprint(package: Path) -> str:
    """What the package's files are right now, as one short text.

    Names, sizes and change times, so a copy made from an older version, or
    before an edit, is told apart from a current one without reading a file.
    """
    package = Path(package)
    digest = hashlib.sha256(str(package).encode("utf-8"))
    for path in sorted(_package_files(package)):
        stat = path.stat()
        relative = path.relative_to(package).as_posix()
        digest.update(f"\0{relative}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:32]


def _package_files(package: Path) -> list[Path]:
    found = []
    for folder, children, files in os.walk(package):
        children[:] = [name for name in children if name not in _NOT_A_MODULE]
        found.extend(Path(folder) / name for name in files if not name.endswith(".pyc"))
    return found


def copies_dir(home: Path) -> Path:
    """The folder every copy of the package is made in."""
    return Path(home) / COPIES_DIR_NAME


def copy_for_package_file(package_file: Path, home: Path) -> Path:
    """The folder holding the copy one package file points Houdini at.

    Named after the package file, so two package files never share a copy and
    taking one away can never pull the ground from under another.
    """
    return copies_dir(home) / ("package-" + _short_hash(str(package_file)))


def copy_for_source(package: Path, home: Path) -> Path:
    """The folder holding the copy that sessions this tool starts import."""
    return copies_dir(home) / ("run-" + _short_hash(str(package)))


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def python_path_for_run(home: Path | None = None) -> Path:
    """The folder to put on the path of a Houdini this tool starts itself.

    The source folder when it holds only this package, otherwise a copy that
    is refreshed here whenever the source has changed. Several servers may
    start sessions at once, so a current copy another one has just finished is
    taken as it is rather than fought over.
    """
    source = source_root()
    if not strays(source):
        return source
    home = Path(home) if home is not None else store_module.default_home()
    package = package_root()
    folder = copy_for_source(package, home)
    stamp = fingerprint(package)
    if copy_stamp(folder) == stamp:
        return folder
    try:
        refresh_copy(folder, package, stamp)
    except OSError:
        if copy_stamp(folder) != stamp:
            raise
    return folder


def refresh_copy(folder: Path, package: Path | None = None, stamp: str | None = None) -> Path:
    """Make `folder` hold a fresh copy of the package and nothing else.

    The copy is built beside the folder and moved into place in one step, so
    a Houdini starting meanwhile sees the old copy or the new one, never half
    of either. A folder there that this tool did not make is left alone.
    """
    package = Path(package) if package is not None else package_root()
    stamp = stamp or fingerprint(package)
    folder = Path(folder)
    if (folder.exists() or folder.is_symlink()) and not is_our_copy(folder):
        raise InstallError(f"{folder} was not made by this tool, so it is left alone")
    folder.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(dir=str(folder.parent), prefix=folder.name + ".part-"))
    try:
        shutil.copytree(
            package,
            building / PACKAGE_NAME,
            ignore=shutil.ignore_patterns(*_NOT_A_MODULE, "*.pyc"),
        )
        stamp_text = json.dumps(
            {MARKER_KEY: MARKER_VALUE, "source": str(package), "fingerprint": stamp}
        )
        (building / STAMP_FILE_NAME).write_text(stamp_text + "\n", encoding="utf-8")
        _swap_in(building, folder)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return folder


def _swap_in(new: Path, folder: Path) -> None:
    """Put `new` where `folder` is, and take the old one away."""
    old = None
    if folder.exists():
        old = folder.with_name(f"{folder.name}.old-{secrets.token_hex(4)}")
        os.replace(folder, old)
    try:
        os.replace(new, folder)
    except OSError:
        if old is not None and not folder.exists():
            os.replace(old, folder)
            old = None
        raise
    finally:
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)


def read_stamp(folder: Path) -> dict[str, Any] | None:
    """What a copy says about itself, or nothing when it is not a copy of ours."""
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        return None
    loaded = read_document(folder / STAMP_FILE_NAME)
    if not loaded or loaded.get(MARKER_KEY) != MARKER_VALUE:
        return None
    return loaded


def is_our_copy(folder: Path) -> bool:
    """Whether this tool made the copy in that folder."""
    return read_stamp(folder) is not None


def copy_stamp(folder: Path) -> str | None:
    """The fingerprint of the source a copy was made from."""
    loaded = read_stamp(folder)
    value = loaded.get("fingerprint") if loaded else None
    return value if isinstance(value, str) else None


def copy_is_current(folder: Path) -> bool | None:
    """Whether a copy still matches the source it was made from.

    Nothing when that source is gone, since then there is nothing to match.
    """
    loaded = read_stamp(folder)
    if not loaded:
        return False
    named = loaded.get("source")
    if not isinstance(named, str) or not named or not Path(named).is_dir():
        return None
    return fingerprint(Path(named)) == loaded.get("fingerprint")


def remove_copy(folder: Path) -> bool:
    """Take away a copy this tool made, and the folder of copies once empty."""
    folder = Path(folder)
    if not is_our_copy(folder):
        return False
    shutil.rmtree(folder)
    try:
        if not any(folder.parent.iterdir()):
            folder.parent.rmdir()
    except OSError:
        pass
    return True


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
    copy: Path | None = None,
) -> dict[str, Any]:
    """The package Houdini reads, as data.

    `HOUDINI_PATH` gets the payload folder through `hpath`, so Houdini runs
    the startup files in it. `PYTHONPATH` gets the source folder, which holds
    this package and nothing else, so the bridge is importable in Houdini's
    own interpreter and nothing of this tool's environment is. Nothing opens
    a port unless the auto start variable is on.

    The folders this install had to make, and the copy of the package it made
    when there was one, are written down, so an uninstall can take away what
    it made and nothing else.
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
    if copy is not None:
        body[COPY_KEY] = str(copy)
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


def copy_of(loaded: dict[str, Any] | None) -> Path | None:
    """The copy of the package the install that wrote this file made."""
    named = (loaded or {}).get(COPY_KEY)
    return Path(named) if isinstance(named, str) and named else None


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
    # The package folder the copy on `source` was made from, when there is one.
    copied_from: Path | None = None


def install(
    version: str = DEFAULT_HOUDINI_VERSION,
    *,
    autostart: bool = False,
    dry_run: bool = False,
    packages: Path | str | None = None,
    lookup: Lookup | None = None,
    home: Path | str | None = None,
) -> InstallResult:
    """Write the package file for one Houdini version.

    A file already there and carrying the marker is replaced. One that is not
    ours raises, and nothing on disk is touched. `packages` names the folder
    outright; without it the folder is worked out, and a caller that has
    already worked it out passes that `lookup` rather than asking again.

    When the source folder holds anything besides this package, the package
    is copied into a folder of its own under `home` (the state folder when
    none is given) and that copy is what the file names. It is made again on
    every install, so running this after an upgrade brings Houdini up to date.
    """
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
    earlier_copy = copy_of(read_document(path)) if exists else None

    payload = payload_root()
    source = source_root()
    copied_from = None
    copy = None
    if strays(source):
        copied_from = package_root()
        home = Path(home) if home is not None else store_module.default_home()
        copy = copy_for_package_file(path, home)
        source = copy
    check_writable(source, "source")

    would_make = _missing_folders(path.parent)
    body = document(
        autostart=autostart, source=source, payload=payload, created=would_make, copy=copy
    )
    if not dry_run:
        if copy is not None:
            refresh_copy(copy, copied_from)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(path, json.dumps(body, indent=4, ensure_ascii=False) + "\n")
        if earlier_copy is not None and earlier_copy != copy:
            remove_copy(earlier_copy)
    lines = [
        f"package       {path}",
        f"folder from   {found.source}",
        f"houdini        {version}",
        f"houdini path   {payload}",
        f"pythonpath     {source}",
    ]
    if copied_from is not None:
        lines.append(f"copied from    {copied_from}")
    lines.append(f"{AUTOSTART_ENV_VAR}   {'1' if autostart else '0'}")
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
        lines=lines,
        copied_from=copied_from,
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
        loaded = read_document(path)
        made = created_folders(loaded)
        copy = copy_of(loaded)
        path.unlink()
        results.append(RemovedPackage(path, found_version, True, "removed"))
        _remove_empty(made)
        if copy is not None and remove_copy(copy):
            results.append(RemovedPackage(copy, found_version, True, "removed its copy"))
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
    # The copy of the package Houdini imports, when the install made one, and
    # whether it still matches its source (nothing when the source is gone).
    copy: Path | None = None
    copy_current: bool | None = None


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
        copy = copy_of(loaded) if ours else None
        states.append(
            InstalledPackage(
                version=found_version,
                path=path,
                present=path.exists(),
                ours=ours,
                autostart=_autostart_of(loaded) if ours else None,
                copy=copy,
                copy_current=copy_is_current(copy) if copy is not None else None,
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


def snippet(source: Path | None = None, *, home: Path | None = None) -> str:
    """Python to paste into the shell of a Houdini that is already open.

    The path is worked out when this prints, so the snippet names the source
    folder of the copy the artist is running, or, when that folder holds
    other libraries too, a copy of the package alone made under `home`.
    """
    root = Path(source) if source is not None else python_path_for_run(home)
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
