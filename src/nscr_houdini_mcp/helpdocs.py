"""Houdini's own documentation: the help folder every install ships, and the
help server a session runs.

Two places a page can come from:

1. The corpus under `$HFS/houdini/help`: one zip file per book, holding one
   markup file per page (`nodes.zip` holds `sop/attribwrangle.txt`, which is
   the page `nodes/sop/attribwrangle`), and a few books kept as plain folders
   of the same files. It needs no session at all and answers in a few
   milliseconds, so it is read first whenever it is the same build as the
   session.
2. The help server of a running Houdini. It runs inside the session's own
   Python, so it answers slowly, and not at all while the session cooks or
   runs code. It is read only for a build with no help folder here, or for a
   page the folder does not have. The server process asks the session once
   for `hou.helpServerUrl()` and keeps the answer per session. Each request is
   bounded as a whole by a short timeout and follows no redirect; a session
   whose help server timed out is left alone for a minute.

Pages from either place are kept in a small cache under the state folder,
keyed by build, source and path, and bounded in size. Search runs over an
index of every page's title and first paragraph, built once per build and
kept beside the cache. A build folder whose install has gone is removed.

A node's versions and namespaces are in its file name: the type
`kinefx::rigattribwrangle` is `kinefx--rigattribwrangle.txt`, the type
`copytopoints::2.0` is `copytopoints-2.0.txt` or, when that is the current
version, `copytopoints.txt` with `#version: 2.0` in it, and an older version
with no number of its own is `copytopoints-.txt`.

This module never imports `hou`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import threading
import time
import urllib.parse
import zipfile
import zlib
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import helptext
from nscr_houdini_mcp.bridge.security import InsecureLocation, private_dir, write_private

DOCS_DIR_NAME = "docs"
INDEX_FILE_NAME = "index.json"
BUILD_FILE_NAME = "build.json"
INDEX_VERSION = 2
PAGE_CACHE_VERSION = 2

# How long one request to a help server may take, from connecting to the
# last byte.
HELP_TIMEOUT_S = 2.0

# How long a connection to a help server may take to be made. On loopback a
# listening port takes one at once.
CONNECT_TIMEOUT_S = 0.5

# How long a session whose help server timed out is left alone.
QUIET_AFTER_TIMEOUT_S = 60.0

# The most a help server page is read. A node page with its navigation is
# under a megabyte.
MAX_HTML_BYTES = 8 * 1024 * 1024

# How much of each page's top the index reads for its title and summary.
HEAD_BYTES = 4096

# The page cache of one build is trimmed back to three quarters of this when
# it is over.
CACHE_MAX_BYTES = 16 * 1024 * 1024

# Everything kept under the docs folder, for every build: past this, the
# builds used least recently go first, whole.
DOCS_MAX_BYTES = 64 * 1024 * 1024

# How many indexes are kept in memory, the most recently used.
MAX_KEPT_INDEXES = 2

# The largest page file read, packed or not. The largest page shipped is
# well under a megabyte.
MAX_PAGE_BYTES = 8 * 1024 * 1024

# How many zip files are kept open at once, across every install.
MAX_OPEN_ARCHIVES = 32

# How far down the help folder a book kept as a folder is walked.
MAX_FOLDER_DEPTH = 8

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Books whose pages are left out of the index: the pages about example
# files, which would crowd every search, and the licenses. They still read.
UNINDEXED_BOOKS = frozenset({"examples", "files", "licenses"})

# Where the last part of a path is a name a person types: a node's internal
# name, a function's, a class's.
NAMED_BOOKS = ("nodes/", "vex/", "hom/")

# What a damaged member of a zip file can raise when it is read.
DAMAGED = (OSError, EOFError, RuntimeError, ValueError, zipfile.BadZipFile, zlib.error)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_VERSION_LINE = re.compile(r'#define\s+SYS_VERSION_FULL\s+"([^"]+)"')
_HELP_PATH = re.compile(r"^[A-Za-z0-9_./:+-]+$")
_LINK_KIND = re.compile(r"^([A-Za-z]+):(?!:)(.*)$")
_VERSION = re.compile(r"^\d+(?:\.\d+)*$")
_VERSIONED_FILE = re.compile(r"^(.+)-(\d+(?:\.\d+)*)?$")


class HelpServerError(Exception):
    """The help server could not be reached or did not answer in time."""

    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


class PageMissing(Exception):
    """The help server answered that it has no such page."""


# Section: help paths


def tidy_path(path: str) -> str | None:
    """A help path as the corpus names it, or nothing for one that cannot be.

    Takes `nodes/sop/attribwrangle`, a leading slash, a trailing `.html`,
    `.txt` or `/index`, the link forms `Node:sop/attribwrangle` and
    `Vex:noise`, and node type names with a namespace or a version, such as
    `nodes/sop/copytopoints::2.0`.
    """
    text = path.strip().split("#", 1)[0].split("?", 1)[0]
    if not text or not _HELP_PATH.match(text):
        return None
    kind = _LINK_KIND.match(text)
    if kind and kind.group(1) in helptext.LINK_ROOTS:
        text = helptext.LINK_ROOTS[kind.group(1)] + kind.group(2)
    text = text.strip("/")
    for suffix in (".html", ".txt"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    parts = [part for part in text.split("/") if part]
    if not parts:
        return None
    if "::" in parts[-1]:
        parts[-1] = file_stem(parts[-1])
    if any(part in (".", "..") or ":" in part for part in parts):
        return None
    if len(parts) > 1 and parts[-1] == "index":
        parts.pop()
    return "/".join(parts)


def file_stem(name: str) -> str:
    """The file name a node type name is kept under.

    `ns::name::1.0` is `ns--name-1.0`: namespaces joined with `--`, the
    version after a `-`.
    """
    pieces = [piece for piece in name.split("::")]
    version = None
    if len(pieces) > 1 and _VERSION.match(pieces[-1]):
        version = pieces.pop()
    stem = "--".join(pieces)
    return f"{stem}-{version}" if version else stem


def _below(version: str, other: str) -> bool:
    """Whether one version number is lower than another, part by part."""
    return tuple(int(part) for part in version.split(".")) < tuple(
        int(part) for part in other.split(".")
    )


def version_label(path: str, version: str | None) -> str | None:
    """The version to show for a page: what it says, else what its file name
    says, else `older` for an older page whose file carries no number."""
    if version:
        return version
    versioned = _VERSIONED_FILE.match(path.rsplit("/", 1)[-1])
    if versioned is None:
        return None
    return versioned.group(2) or "older"


def is_old(path: str) -> bool:
    """Whether a path names a version of a page other than the current one."""
    return bool(_VERSIONED_FILE.match(path.rsplit("/", 1)[-1]))


def is_news(path: str) -> bool:
    return path.startswith("news/") or "whatsnew" in path


# Section: open zip files


class _Archive:
    """One zip file kept open, with the names in it."""

    def __init__(self, path: Path, stamp: tuple[int, int]) -> None:
        self.stamp = stamp
        self.zip = zipfile.ZipFile(path)
        self.members = {info.filename: info for info in self.zip.infolist()}
        self.lock = threading.Lock()

    def read(self, member: str) -> bytes | None:
        """One member, or nothing when it is missing, damaged or too large.

        The size the zip claims is checked first, and the read stops past the
        cap whatever it claimed.
        """
        info = self.members.get(member)
        if info is None or info.file_size > MAX_PAGE_BYTES:
            return None
        with self.lock:
            try:
                with self.zip.open(info) as stream:
                    data = stream.read(MAX_PAGE_BYTES + 1)
            except (*DAMAGED, KeyError):
                return None
        return None if len(data) > MAX_PAGE_BYTES else data

    def close(self) -> None:
        with self.lock:
            try:
                self.zip.close()
            except OSError:
                pass


_ARCHIVES: OrderedDict[str, _Archive] = OrderedDict()
_ARCHIVES_LOCK = threading.Lock()


def _archive(path: Path) -> _Archive | None:
    """The zip file at `path`, opened once and kept while it is unchanged."""
    try:
        stat = path.stat()
    except OSError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    key = str(path)
    with _ARCHIVES_LOCK:
        kept = _ARCHIVES.get(key)
        if kept is not None and kept.stamp == stamp:
            _ARCHIVES.move_to_end(key)
            return kept
        if kept is not None:
            del _ARCHIVES[key]
            kept.close()
        try:
            opened = _Archive(path, stamp)
        except (OSError, zipfile.BadZipFile, ValueError):
            return None
        _ARCHIVES[key] = opened
        while len(_ARCHIVES) > MAX_OPEN_ARCHIVES:
            _, oldest = _ARCHIVES.popitem(last=False)
            oldest.close()
        return opened


def close_archives() -> None:
    """Close every zip file kept open. For tests, and before a folder goes."""
    with _ARCHIVES_LOCK:
        while _ARCHIVES:
            _, kept = _ARCHIVES.popitem()
            kept.close()


# Section: the corpus


@dataclass(frozen=True)
class Corpus:
    """The help folder of one Houdini install."""

    hfs: Path
    build: str

    @property
    def root(self) -> Path:
        return self.hfs / "houdini" / "help"

    def exists(self) -> bool:
        return self.root.is_dir()

    def fingerprint(self) -> str:
        """Changes when any page file in the folder changes.

        Every zip file and every file under a book kept as a folder, by name,
        size and modification time to the nanosecond, with the install's own
        version header. Worked out at most once every few seconds.
        """
        key = str(self.root)
        now = time.monotonic()
        with _FINGERPRINTS_LOCK:
            kept = _FINGERPRINTS.get(key)
            if kept is not None and now - kept[0] < FINGERPRINT_KEEP_S:
                return kept[1]
        digest = hashlib.sha256(self.build.encode("utf-8"))
        header = self.hfs / "toolkit" / "include" / "SYS" / "SYS_Version.h"
        for path in (header, *_help_files(self.root)):
            try:
                stat = path.stat()
            except OSError:
                continue
            digest.update(f"\n{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        said = digest.hexdigest()[:16]
        with _FINGERPRINTS_LOCK:
            _FINGERPRINTS[key] = (now, said)
        return said

    def read(self, path: str) -> str | None:
        """One page's markup, or nothing when the corpus has no such page.

        A version asked for by number that has no file of its own, such as
        `copytopoints-1.0`, is found where Houdini keeps it instead:

        - the current page, `copytopoints`, when it says it is that version;
        - the older page with no number, `copytopoints-`, when it says it is
          that version;
        - that older page when it says no version and the version asked for
          is below the current page's, since it holds the version before.
        """
        clean = tidy_path(path)
        if clean is None:
            return None
        text = self._find(clean)
        if text is not None:
            return text
        folder, _, name = clean.rpartition("/")
        versioned = _VERSIONED_FILE.match(name)
        if not (versioned and versioned.group(2)):
            return None
        wanted = versioned.group(2)
        stem = f"{folder}/{versioned.group(1)}" if folder else versioned.group(1)
        current = self._find(stem)
        current_version = version_of(current) if current is not None else None
        if current is not None and current_version == wanted:
            return current
        older = self._find(stem + "-")
        if older is None:
            return None
        older_version = version_of(older)
        if older_version == wanted:
            return older
        if older_version is None and current_version and _below(wanted, current_version):
            return older
        return None

    def _find(self, path: str) -> str | None:
        for folder_file, book, member in self._candidates(path):
            if folder_file is not None:
                text = _read_file(folder_file)
            else:
                text = _read_member(self.root / f"{book}.zip", member)
            if text is not None:
                return text
        return None

    def _candidates(self, path: str) -> Iterator[tuple[Path | None, str, str]]:
        """Where a page may be: a file under the folder, then a zip member."""
        parts = path.split("/")
        for tail in (parts, [*parts, "index"]):
            yield self.root.joinpath(*tail[:-1], tail[-1] + ".txt"), "", ""
            if len(tail) > 1:
                yield None, tail[0], "/".join(tail[1:]) + ".txt"

    def pages(self, damaged: list[str] | None = None) -> Iterator[tuple[str, str]]:
        """Every page's path and the top of its markup, for the index.

        A member of a zip file that cannot be read is left out, and its name
        added to `damaged`.
        """
        try:
            entries = sorted(self.root.iterdir(), key=lambda entry: entry.name)
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if name.startswith((".", "_")):
                continue
            if entry.is_file() and name.endswith(".zip"):
                book = name[: -len(".zip")]
                if book in UNINDEXED_BOOKS:
                    continue
                yield from _zip_pages(entry, book, damaged)
            elif entry.is_file() and name.endswith(".txt"):
                head = _read_file(entry, limit=HEAD_BYTES)
                if head is not None:
                    yield name[: -len(".txt")], head
            elif entry.is_dir() and name not in UNINDEXED_BOOKS:
                yield from _folder_pages(entry, name)


# How long a worked out fingerprint is used before the folder is looked at again.
FINGERPRINT_KEEP_S = 5.0

# The most files a fingerprint looks at, so a folder that is not a help
# folder cannot keep it walking.
MAX_FINGERPRINT_FILES = 20_000

_FINGERPRINTS: dict[str, tuple[float, str]] = {}
_FINGERPRINTS_LOCK = threading.Lock()


def _help_files(root: Path) -> Iterator[Path]:
    """The zip files at the top of a help folder, then every file below it."""
    count = 0
    for current, folders, files in os.walk(root):
        here = Path(current)
        depth = len(here.relative_to(root).parts)
        folders[:] = sorted(name for name in folders if not name.startswith("."))
        if depth >= MAX_FOLDER_DEPTH:
            folders[:] = []
        for name in sorted(files):
            if depth == 0 and not name.endswith((".zip", ".txt")):
                continue
            count += 1
            if count > MAX_FINGERPRINT_FILES:
                return
            yield here / name


def version_of(markup: str) -> str | None:
    return helptext.markup_head(markup[:HEAD_BYTES]).properties.get("version") or None


def _zip_pages(archive: Path, book: str, damaged: list[str] | None) -> Iterator[tuple[str, str]]:
    try:
        with zipfile.ZipFile(archive) as opened:
            for info in opened.infolist():
                member = info.filename
                if info.is_dir() or not member.endswith(".txt") or _hidden(member):
                    continue
                if info.file_size > MAX_PAGE_BYTES:
                    if damaged is not None:
                        damaged.append(f"{archive.name}:{member}")
                    continue
                try:
                    with opened.open(info) as stream:
                        head = stream.read(HEAD_BYTES)
                except DAMAGED:
                    if damaged is not None:
                        damaged.append(f"{archive.name}:{member}")
                    continue
                yield _page_path(book, member), head.decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile, ValueError):
        if damaged is not None:
            damaged.append(archive.name)
        return


def _folder_pages(folder: Path, book: str) -> Iterator[tuple[str, str]]:
    for current, folders, files in os.walk(folder):
        here = Path(current)
        depth = len(here.relative_to(folder).parts)
        folders[:] = sorted(name for name in folders if not name.startswith((".", "_")))
        if depth >= MAX_FOLDER_DEPTH:
            folders[:] = []
        for name in sorted(files):
            if not name.endswith(".txt") or name.startswith((".", "_")):
                continue
            head = _read_file(here / name, limit=HEAD_BYTES)
            if head is None:
                continue
            member = (here / name).relative_to(folder).as_posix()
            yield _page_path(book, member), head


def _page_path(book: str, member: str) -> str:
    stem = member[: -len(".txt")]
    if stem == "index":
        return book
    if stem.endswith("/index"):
        stem = stem[: -len("/index")]
    return f"{book}/{stem}"


def _hidden(member: str) -> bool:
    return any(part.startswith((".", "_")) for part in member.split("/"))


def _read_file(path: Path, *, limit: int | None = None) -> str | None:
    """A file's text, or nothing when it is missing or over the page cap."""
    cap = MAX_PAGE_BYTES if limit is None else limit
    try:
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            data = stream.read(cap + 1)
    except OSError:
        return None
    if len(data) > cap:
        if limit is None:
            return None
        data = data[:cap]
    return data.decode("utf-8", "replace")


def _read_member(archive: Path, member: str) -> str | None:
    opened = _archive(archive)
    if opened is None:
        return None
    data = opened.read(member)
    return None if data is None else data.decode("utf-8", "replace")


# Section: which install


def build_of(hfs: Path) -> str | None:
    """The build an install says it is, from the header every install ships."""
    header = hfs / "toolkit" / "include" / "SYS" / "SYS_Version.h"
    text = _read_file(header, limit=64 * 1024)
    if text:
        found = _VERSION_LINE.search(text)
        if found:
            return found.group(1)
    return None


def hfs_of_hython(hython: Path) -> Path:
    """The install a hython belongs to: it sits in `$HFS/bin`."""
    try:
        hython = hython.resolve()
    except OSError:
        pass
    return hython.parent.parent


# Section: the state folder

_SWEPT: set[str] = set()
_SWEPT_LOCK = threading.Lock()


def docs_home(state_home: Path, build: str | None, hfs: Path | None = None) -> Path:
    """The private folder the index and the page cache of one build live in.

    The folder says which install it is for, so the first use in a process
    can remove the folders of builds whose install has gone.
    """
    root = Path(state_home) / DOCS_DIR_NAME
    folder = root / _SAFE.sub("_", build or "unknown")
    try:
        private_dir(root)
        sweep(root)
        private_dir(folder)
        said = folder / BUILD_FILE_NAME
        body = json.dumps({"build": build, "hfs": str(hfs) if hfs else None}, ensure_ascii=False)
        if _read_file(said) != body:
            write_private(said, body)
        else:
            # Used now: the quota takes the builds used least recently first.
            os.utime(said)
    except (OSError, InsecureLocation):
        pass
    return folder


def keep_to_quota(folder: Path, cap: int = DOCS_MAX_BYTES) -> list[str]:
    """Remove other builds' folders, least recently used first, until
    everything under the docs folder fits in `cap`. The build in use stays."""
    root = folder.parent
    sizes: list[tuple[float, int, Path]] = []
    total = 0
    try:
        for entry in root.iterdir():
            if not entry.is_dir() or entry.is_symlink():
                continue
            size = _folder_size(entry)
            total += size
            try:
                used = (entry / BUILD_FILE_NAME).stat().st_mtime
            except OSError:
                used = 0.0
            if entry != folder:
                sizes.append((used, size, entry))
    except OSError:
        return []
    removed = []
    for _, size, entry in sorted(sizes, key=lambda item: item[0]):
        if total <= cap:
            break
        shutil.rmtree(entry, ignore_errors=True)
        total -= size
        removed.append(entry.name)
    return removed


def _folder_size(folder: Path) -> int:
    total = 0
    for current, _, files in os.walk(folder):
        for name in files:
            try:
                total += (Path(current) / name).stat().st_size
            except OSError:
                continue
    return total


def sweep(root: Path) -> list[str]:
    """Remove the folders of builds whose help folder is no longer on disk.

    Only a folder this module made, which says its install, is removed; one
    for a build known only from a help server is kept. Once per process.
    """
    with _SWEPT_LOCK:
        if str(root) in _SWEPT:
            return []
        _SWEPT.add(str(root))
    removed = []
    try:
        folders = [entry for entry in root.iterdir() if entry.is_dir()]
    except OSError:
        return removed
    for folder in folders:
        if folder.is_symlink():
            continue
        try:
            said = json.loads((folder / BUILD_FILE_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hfs = said.get("hfs") if isinstance(said, dict) else None
        if not isinstance(hfs, str) or not hfs:
            continue
        if (Path(hfs) / "houdini" / "help").is_dir():
            continue
        shutil.rmtree(folder, ignore_errors=True)
        removed.append(folder.name)
    return removed


def forget_sweeps() -> None:
    """Let the next use sweep again, and look at every folder afresh. For tests."""
    with _SWEPT_LOCK:
        _SWEPT.clear()
    with _FINGERPRINTS_LOCK:
        _FINGERPRINTS.clear()


# Section: the page cache


def cache_key(build: str | None, source: str, path: str, form: str, fingerprint: str) -> str:
    text = f"{PAGE_CACHE_VERSION}\n{build}\n{source}\n{fingerprint}\n{path}\n{form}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def cache_get(folder: Path, key: str) -> dict[str, Any] | None:
    path = folder / "pages" / f"{key}.json"
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        return None
    try:
        # Read recently: the trim takes the oldest first.
        os.utime(path)
    except OSError:
        pass
    return body


def cache_put(folder: Path, key: str, body: dict[str, Any], *, cap: int | None = None) -> None:
    """Keep one rendered page, then trim the cache if it has grown past its cap."""
    pages = folder / "pages"
    try:
        write_private(pages / f"{key}.json", json.dumps(body, ensure_ascii=False))
    except (OSError, InsecureLocation):
        return
    trim(pages, CACHE_MAX_BYTES if cap is None else cap)
    keep_to_quota(folder)


def trim(pages: Path, cap: int) -> None:
    """Take the least recently read pages away until the cache is well under its cap."""
    try:
        entries = []
        for entry in pages.iterdir():
            if entry.suffix != ".json":
                continue
            stat = entry.stat()
            entries.append((stat.st_mtime, stat.st_size, entry))
    except OSError:
        return
    total = sum(size for _, size, _ in entries)
    if total <= cap:
        return
    goal = cap * 3 // 4
    for _, size, entry in sorted(entries, key=lambda item: item[0]):
        if total <= goal:
            break
        try:
            entry.unlink()
            total -= size
        except OSError:
            continue


# Section: the index

# One page as the index keeps it: path, title, first paragraph, version, and
# whether the page says it is deprecated.
Row = tuple[str, str, str, str, bool]


@dataclass(frozen=True)
class Index:
    """Every page's path, title and first paragraph, for one build."""

    build: str
    fingerprint: str
    pages: tuple[Row, ...]
    built_s: float
    built_at: float
    # Members of the help folder that could not be read and were left out.
    damaged: tuple[str, ...] = field(default=())
    # Whether this call built it, rather than reading it back.
    fresh: bool = False

    def note(self) -> str:
        how = "built now" if self.fresh else "read from the cache"
        said = (
            f"index of {len(self.pages)} pages for Houdini {self.build}, {how};"
            f" building it took {self.built_s:.1f} s"
        )
        if self.damaged:
            said += f"; {len(self.damaged)} damaged pages in the help folder were left out"
        return said


_INDEXES: OrderedDict[str, Index] = OrderedDict()
_INDEX_LOCK = threading.Lock()


def load_index(state_home: Path, corpus: Corpus) -> Index:
    """The index for this build: kept in memory, then on disk, then built.

    A build is indexed once. The fingerprint of the help folder is part of the
    key, so a folder that changed under the same build is indexed again.
    """
    fingerprint = corpus.fingerprint()
    key = f"{corpus.root}\n{fingerprint}"
    with _INDEX_LOCK:
        kept = _INDEXES.get(key)
        if kept is not None:
            _INDEXES.move_to_end(key)
            return kept
        folder = docs_home(state_home, corpus.build, corpus.hfs)
        index = _read_index(folder / INDEX_FILE_NAME, corpus, fingerprint)
        if index is None:
            index = build_index(corpus, fingerprint)
            _write_index(folder, index)
        # What is kept was built by an earlier call, as far as the next one knows.
        _INDEXES[key] = replace(index, fresh=False)
        while len(_INDEXES) > MAX_KEPT_INDEXES:
            _INDEXES.popitem(last=False)
        return index


def forget_indexes() -> None:
    """Drop the indexes kept in memory. For tests."""
    with _INDEX_LOCK:
        _INDEXES.clear()


def build_index(corpus: Corpus, fingerprint: str | None = None) -> Index:
    started = time.monotonic()
    pages: list[Row] = []
    damaged: list[str] = []
    for path, head in corpus.pages(damaged):
        page = helptext.markup_head(head)
        properties = page.properties
        if properties.get("type") == "include" or properties.get("index") == "no":
            continue
        title = page.title or path.rsplit("/", 1)[-1]
        version = properties.get("version", "")
        pages.append((path, title, helptext.excerpt(page.summary), version, page.deprecated))
    return Index(
        build=corpus.build,
        fingerprint=fingerprint or corpus.fingerprint(),
        pages=tuple(pages),
        built_s=round(time.monotonic() - started, 3),
        built_at=time.time(),
        damaged=tuple(damaged),
        fresh=True,
    )


def _read_index(path: Path, corpus: Corpus, fingerprint: str) -> Index | None:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(body, dict)
        or body.get("version") != INDEX_VERSION
        or body.get("build") != corpus.build
        or body.get("fingerprint") != fingerprint
        or not isinstance(body.get("pages"), list)
    ):
        return None
    pages = tuple(
        (str(row[0]), str(row[1]), str(row[2]), str(row[3]), bool(row[4]))
        for row in body["pages"]
        if isinstance(row, list) and len(row) == 5
    )
    damaged = body.get("damaged")
    return Index(
        build=corpus.build,
        fingerprint=fingerprint,
        pages=pages,
        built_s=float(body.get("built_s") or 0.0),
        built_at=float(body.get("built_at") or 0.0),
        damaged=tuple(str(item) for item in damaged) if isinstance(damaged, list) else (),
    )


def _write_index(folder: Path, index: Index) -> None:
    body = {
        "version": INDEX_VERSION,
        "build": index.build,
        "fingerprint": index.fingerprint,
        "built_s": index.built_s,
        "built_at": index.built_at,
        "damaged": list(index.damaged),
        "pages": [list(row) for row in index.pages],
    }
    try:
        write_private(folder / INDEX_FILE_NAME, json.dumps(body, ensure_ascii=False))
    except (OSError, InsecureLocation):
        # An index that cannot be kept is built again next time; it still answers now.
        return
    keep_to_quota(folder)


# Section: ranking


def rank(query: str, title: str, path: str, excerpt: str) -> int | None:
    """How well a page matches: 0 exact title, 1 prefix, 2 substring, 3 body.

    Under `nodes/`, `vex/` and `hom/` the last part of the path counts as a
    second title, without its version, so the internal name of a node
    (`attribwrangle`) finds it as well as its label does. Nothing when the
    page does not match at all.
    """
    wanted = " ".join(query.lower().split())
    if not wanted:
        return None
    names = [" ".join(title.lower().split())]
    if path.startswith(NAMED_BOOKS):
        last = path.rsplit("/", 1)[-1].lower()
        versioned = _VERSIONED_FILE.match(last)
        names.append(versioned.group(1) if versioned else last)
    if wanted in names:
        return 0
    if any(name.startswith(wanted) for name in names):
        return 1
    if any(wanted in name for name in names):
        return 2
    body = f"{title} {path} {excerpt}".lower()
    if all(word in body for word in wanted.split()):
        return 3
    return None


def order(tier: int, path: str, title: str, deprecated: bool = False) -> tuple[Any, ...]:
    """Where a match goes in the results.

    Release notes last of all, then by how well it matched; among equals the
    current version of a page before older ones and anything deprecated, a
    geometry node before other nodes, a node before a function or a class
    and those before the guides, then the shorter title.
    """
    if path.startswith("nodes/sop/"):
        book = 0
    elif path.startswith("nodes/"):
        book = 1
    elif path.startswith(NAMED_BOOKS):
        book = 2
    else:
        book = 3
    return (is_news(path), tier, is_old(path) or deprecated, book, len(title), path)


def search_index(index: Index, query: str, limit: int) -> list[tuple[int, Row]]:
    found = []
    for row in index.pages:
        tier = rank(query, row[1], row[0], row[2])
        if tier is not None:
            found.append((tier, row))
    found.sort(key=lambda item: order(item[0], item[1][0], item[1][1], item[1][4]))
    return found[: max(limit, 0)]


# Section: the help server


def usable_url(url: Any) -> str | None:
    """A help server address to use, or nothing.

    Only plain HTTP on this machine's loopback, with no name or password in
    it: help configured to come from a website is left to the corpus, and
    nothing is fetched anywhere else.
    """
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        parsed = urllib.parse.urlsplit(url.strip())
        parsed.port  # noqa: B018 - raises for a port that is not a number
    except ValueError:
        return None
    if parsed.scheme != "http" or (parsed.hostname or "") not in LOOPBACK_HOSTS:
        return None
    if "@" in parsed.netloc:
        # A name and password in the address: not what a help server hands out.
        return None
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))


def fetch(base: str, path: str, *, query: dict[str, str] | None = None) -> str:
    """One page from a help server, as text.

    Raises `PageMissing` for a 404 and `HelpServerError` for anything else
    that is not an answer, a redirect included, since no redirect is followed.
    `HELP_TIMEOUT_S` bounds the whole request: it runs on a thread of its own,
    which is left behind if it has not finished by then.

    The two ways it can fail are told apart, because they call for different
    things. A connection that is refused, reset or not made within
    `CONNECT_TIMEOUT_S` means nothing is listening there any more
    (`timed_out` false): the address may have moved. A connection that was
    made and then not answered in time means the help server is there and
    busy (`timed_out` true). On loopback a listening port takes a connection
    at once, even from a process that is busy, so a connect that does not
    complete quickly is a port nobody listens on, whatever the system does
    with it.
    """
    parsed = urllib.parse.urlsplit(base)
    target = (
        (parsed.path.rstrip("/") or "") + "/" + urllib.parse.quote(path.strip("/"), safe="/_.-+")
    )
    if query:
        target += "?" + urllib.parse.urlencode(query)
    deadline = time.monotonic() + HELP_TIMEOUT_S
    box: dict[str, Any] = {"phase": "connect"}

    def work() -> None:
        try:
            box["text"] = _fetch(parsed, target, path, deadline, box)
        except BaseException as error:  # noqa: BLE001 - handed to the waiting thread
            box["error"] = error

    worker = threading.Thread(target=work, name="nscr-mcp-help", daemon=True)
    worker.start()
    worker.join(max(deadline - time.monotonic(), 0.0))
    if worker.is_alive():
        if box["phase"] == "connect":
            raise HelpServerError("the help server did not take a connection")
        raise HelpServerError(
            f"the help server did not answer within {HELP_TIMEOUT_S:g} s", timed_out=True
        )
    if "error" in box:
        raise box["error"]
    return box["text"]


def _fetch(
    parsed: urllib.parse.SplitResult, target: str, path: str, deadline: float, box: dict
) -> str:
    """The request itself, with no proxy and no redirect, noting its phase in `box`."""
    try:
        port = parsed.port or 80
    except ValueError:
        raise HelpServerError("the help server address has no usable port") from None
    connection = http.client.HTTPConnection(
        parsed.hostname or "127.0.0.1", port, timeout=min(CONNECT_TIMEOUT_S, HELP_TIMEOUT_S)
    )
    try:
        try:
            connection.connect()
        except OSError as error:
            # Refused, reset, unreachable, or not made in time.
            raise HelpServerError(
                f"the help server did not take a connection ({type(error).__name__})"
            ) from None
        box["phase"] = "read"
        if connection.sock is not None:
            connection.sock.settimeout(max(deadline - time.monotonic(), 0.01))
        try:
            connection.request("GET", target, headers={"Accept": "text/html"})
            answer = connection.getresponse()
            if answer.status == 404:
                raise PageMissing(path)
            if 300 <= answer.status < 400:
                raise HelpServerError(f"the help server redirected ({answer.status}), refused")
            if answer.status != 200:
                raise HelpServerError(f"the help server answered {answer.status}")
            chunks: list[bytes] = []
            size = 0
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise HelpServerError("the help server answered too slowly", timed_out=True)
                if connection.sock is not None:
                    connection.sock.settimeout(left)
                chunk = answer.read1(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_HTML_BYTES:
                    raise HelpServerError("the help server page is too large")
                chunks.append(chunk)
            charset = answer.headers.get_content_charset() or "utf-8"
        except TimeoutError:
            raise HelpServerError(
                f"the help server did not answer within {HELP_TIMEOUT_S:g} s", timed_out=True
            ) from None
        except (ConnectionError, http.client.RemoteDisconnected) as error:
            # It hung up: the process behind the port may have gone.
            raise HelpServerError(
                f"the help server dropped the connection ({type(error).__name__})"
            ) from None
        except (OSError, http.client.HTTPException) as error:
            raise HelpServerError(
                f"the help server did not answer ({type(error).__name__})"
            ) from None
    finally:
        connection.close()
    return b"".join(chunks).decode(charset, "replace")


class HelpUrls:
    """The help server address of each session, asked for once and kept, and
    the sessions whose help server timed out lately."""

    def __init__(self) -> None:
        self._urls: dict[str, str | None] = {}
        self._quiet: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> tuple[bool, str | None]:
        with self._lock:
            if session_id in self._urls:
                return True, self._urls[session_id]
            return False, None

    def put(self, session_id: str, url: str | None) -> None:
        with self._lock:
            self._urls[session_id] = url

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._urls.pop(session_id, None)

    def timed_out(self, session_id: str, *, now: float | None = None) -> None:
        """Leave this session's help server alone for a while."""
        with self._lock:
            moment = time.monotonic() if now is None else now
            self._quiet[session_id] = moment + QUIET_AFTER_TIMEOUT_S

    def quiet_for(self, session_id: str, *, now: float | None = None) -> float:
        """How many seconds this session's help server is still left alone."""
        with self._lock:
            until = self._quiet.get(session_id)
            left = 0.0 if until is None else until - (time.monotonic() if now is None else now)
            if left <= 0 and until is not None:
                del self._quiet[session_id]
            return max(left, 0.0)

    def clear(self) -> None:
        with self._lock:
            self._urls.clear()
            self._quiet.clear()


URLS = HelpUrls()
