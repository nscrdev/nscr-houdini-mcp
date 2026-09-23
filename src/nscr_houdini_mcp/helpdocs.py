"""Houdini's own documentation: the help server a session runs, and the help
folder every install ships.

Two places a page can come from, in this order:

1. The help server of a running Houdini. The server process asks the session
   once for `hou.helpServerUrl()` and keeps the answer per session; every read
   after that goes straight to the help server over HTTP on loopback, which is
   a thread of its own inside Houdini, so a session busy cooking still answers.
   Each request has a short timeout. On any failure other than a page that is
   not there, the address is asked for once more and the request tried again.
2. The corpus under `$HFS/houdini/help`: one zip file per book, holding one
   markup file per page (`nodes.zip` holds `sop/attribwrangle.txt`, which
   is the page `nodes/sop/attribwrangle`), and a few books kept as plain
   folders of the same files. It needs no session at all.

Corpus reads are kept in a small cache under the state folder, keyed by build
and path, and bounded in size. Search runs over an index of every page's title
and first paragraph, built once per build and kept beside the cache.

This module never imports `hou`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import helptext
from nscr_houdini_mcp.bridge.security import InsecureLocation, private_dir, write_private

DOCS_DIR_NAME = "docs"
INDEX_FILE_NAME = "index.json"
INDEX_VERSION = 1
PAGE_CACHE_VERSION = 1

# How long one request to a help server may take, to connect and to answer.
HELP_TIMEOUT_S = 2.0

# The most a help server page is read. A node page with its navigation is
# under a megabyte.
MAX_HTML_BYTES = 8 * 1024 * 1024

# How much of each page's top the index reads for its title and summary.
HEAD_BYTES = 4096

# The page cache is trimmed back to three quarters of this when it is over.
CACHE_MAX_BYTES = 16 * 1024 * 1024

# How far down the help folder a book kept as a folder is walked.
MAX_FOLDER_DEPTH = 8

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Books whose pages are left out of the index: the pages about example
# files, which would crowd every search, and the licenses. They still read.
UNINDEXED_BOOKS = frozenset({"examples", "files", "licenses"})

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_VERSION_LINE = re.compile(r'#define\s+SYS_VERSION_FULL\s+"([^"]+)"')
_HELP_PATH = re.compile(r"^[A-Za-z0-9_./:+-]+$")


class HelpServerError(Exception):
    """The help server could not be reached or did not answer in time."""


class PageMissing(Exception):
    """The help server answered that it has no such page."""


# Section: help paths


def tidy_path(path: str) -> str | None:
    """A help path as the corpus names it, or nothing for one that cannot be.

    Takes `nodes/sop/attribwrangle`, a leading slash, a trailing `.html` or
    `.txt`, and the link forms `Node:sop/attribwrangle` and `Vex:noise`.
    """
    text = path.strip().split("#", 1)[0].split("?", 1)[0]
    if not text or not _HELP_PATH.match(text):
        return None
    text = helptext.resolve_link(text, "") if ":" in text else text
    text = text.strip("/")
    for suffix in (".html", ".txt"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    parts = [part for part in text.split("/") if part]
    if not parts or any(part in (".", "..") or ":" in part for part in parts):
        return None
    return "/".join(parts)


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
        """Changes when any book in the folder is replaced."""
        parts = [self.build]
        try:
            entries = sorted(self.root.iterdir(), key=lambda entry: entry.name)
        except OSError:
            entries = []
        for entry in entries:
            try:
                stat = entry.stat()
            except OSError:
                continue
            parts.append(f"{entry.name}:{stat.st_size}:{int(stat.st_mtime)}")
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]

    def read(self, path: str) -> str | None:
        """One page's markup, or nothing when the corpus has no such page."""
        clean = tidy_path(path)
        if clean is None:
            return None
        for folder_file, book, member in self._candidates(clean):
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

    def pages(self) -> Iterator[tuple[str, str]]:
        """Every page's path and the top of its markup, for the index."""
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
                yield from _zip_pages(entry, book)
            elif entry.is_file() and name.endswith(".txt"):
                head = _read_file(entry, limit=HEAD_BYTES)
                if head is not None:
                    yield name[: -len(".txt")], head
            elif entry.is_dir() and name not in UNINDEXED_BOOKS:
                yield from _folder_pages(entry, name)


def _zip_pages(archive: Path, book: str) -> Iterator[tuple[str, str]]:
    try:
        with zipfile.ZipFile(archive) as opened:
            for info in opened.infolist():
                member = info.filename
                if info.is_dir() or not member.endswith(".txt") or _hidden(member):
                    continue
                try:
                    with opened.open(info) as stream:
                        head = stream.read(HEAD_BYTES)
                except (OSError, zipfile.BadZipFile, RuntimeError):
                    continue
                yield _page_path(book, member), head.decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile):
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
    try:
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            data = stream.read(limit if limit is not None else -1)
    except OSError:
        return None
    return data.decode("utf-8", "replace")


def _read_member(archive: Path, member: str) -> str | None:
    try:
        with zipfile.ZipFile(archive) as opened:
            try:
                data = opened.read(member)
            except KeyError:
                return None
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return None
    return data.decode("utf-8", "replace")


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


def docs_home(state_home: Path, corpus: Corpus) -> Path:
    """The folder the index and the page cache of one build live in."""
    return Path(state_home) / DOCS_DIR_NAME / _SAFE.sub("_", corpus.build or "unknown")


# Section: the page cache


def cache_key(corpus: Corpus, path: str, form: str, fingerprint: str) -> str:
    text = f"{PAGE_CACHE_VERSION}\n{corpus.build}\n{fingerprint}\n{path}\n{form}"
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


@dataclass(frozen=True)
class Index:
    """Every page's path, title and first paragraph, for one build."""

    build: str
    fingerprint: str
    pages: tuple[tuple[str, str, str], ...]
    built_s: float
    built_at: float
    # Whether this call built it, rather than reading it back.
    fresh: bool = False

    def note(self) -> str:
        how = "built now" if self.fresh else "read from the cache"
        return (
            f"index of {len(self.pages)} pages for Houdini {self.build}, {how};"
            f" building it took {self.built_s:.1f} s"
        )


_INDEXES: dict[str, Index] = {}
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
            return kept
        folder = docs_home(state_home, corpus)
        index = _read_index(folder / INDEX_FILE_NAME, corpus, fingerprint)
        if index is None:
            index = build_index(corpus, fingerprint)
            _write_index(folder, index)
        _INDEXES[key] = index
        return index


def forget_indexes() -> None:
    """Drop the indexes kept in memory. For tests."""
    with _INDEX_LOCK:
        _INDEXES.clear()


def build_index(corpus: Corpus, fingerprint: str | None = None) -> Index:
    started = time.monotonic()
    pages: list[tuple[str, str, str]] = []
    for path, head in corpus.pages():
        page = helptext.markup_head(head)
        properties = page.properties
        if properties.get("type") == "include" or properties.get("index") == "no":
            continue
        title = page.title or path.rsplit("/", 1)[-1]
        pages.append((path, title, helptext.excerpt(page.summary)))
    return Index(
        build=corpus.build,
        fingerprint=fingerprint or corpus.fingerprint(),
        pages=tuple(pages),
        built_s=round(time.monotonic() - started, 3),
        built_at=time.time(),
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
        (str(row[0]), str(row[1]), str(row[2]))
        for row in body["pages"]
        if isinstance(row, list) and len(row) == 3
    )
    return Index(
        build=corpus.build,
        fingerprint=fingerprint,
        pages=pages,
        built_s=float(body.get("built_s") or 0.0),
        built_at=float(body.get("built_at") or 0.0),
    )


def _write_index(folder: Path, index: Index) -> None:
    body = {
        "version": INDEX_VERSION,
        "build": index.build,
        "fingerprint": index.fingerprint,
        "built_s": index.built_s,
        "built_at": index.built_at,
        "pages": [list(row) for row in index.pages],
    }
    try:
        private_dir(folder)
        write_private(folder / INDEX_FILE_NAME, json.dumps(body, ensure_ascii=False))
    except (OSError, InsecureLocation):
        # An index that cannot be kept is built again next time; it still answers now.
        pass


# Section: ranking


def rank(query: str, title: str, path: str, excerpt: str) -> int | None:
    """How well a page matches: 0 exact title, 1 prefix, 2 substring, 3 body.

    The last part of the path counts as a second title, so the internal name
    of a node (`attribwrangle`) finds it as well as its label does. Nothing
    when the page does not match at all.
    """
    wanted = query.strip().lower()
    if not wanted:
        return None
    names = (title.strip().lower(), path.rsplit("/", 1)[-1].lower())
    if wanted in names:
        return 0
    if any(name.startswith(wanted) for name in names):
        return 1
    if any(wanted in name for name in names):
        return 2
    words = wanted.split()
    body = f"{title} {path} {excerpt}".lower()
    if all(word in body for word in words):
        return 3
    return None


def search_index(index: Index, query: str, limit: int) -> list[tuple[int, str, str, str]]:
    found = []
    for path, title, excerpt in index.pages:
        tier = rank(query, title, path, excerpt)
        if tier is not None:
            found.append((tier, path, title, excerpt))
    found.sort(key=lambda row: (row[0], len(row[2]), row[1]))
    return found[: max(limit, 0)]


# Section: the help server


def usable_url(url: Any) -> str | None:
    """A help server address to use, or nothing.

    Only plain HTTP on this machine's loopback: help configured to come from
    a website is left to the corpus, and nothing is fetched anywhere else.
    """
    if not isinstance(url, str) or not url.strip():
        return None
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme != "http" or (parsed.hostname or "") not in LOOPBACK_HOSTS:
        return None
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))


def fetch(base: str, path: str, *, query: dict[str, str] | None = None) -> str:
    """One page from a help server, as text.

    Raises `PageMissing` for a 404 and `HelpServerError` for anything else
    that is not an answer within `HELP_TIMEOUT_S`, which bounds the whole
    request, not only each wait on the socket.
    """
    url = base.rstrip("/") + "/" + urllib.parse.quote(path.strip("/"), safe="/_.-+")
    if query:
        url += "?" + urllib.parse.urlencode(query)
    deadline = time.monotonic() + HELP_TIMEOUT_S
    # No proxy: a proxy set for the machine must not carry a loopback request.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=HELP_TIMEOUT_S) as answer:
            chunks: list[bytes] = []
            size = 0
            while True:
                if time.monotonic() > deadline:
                    raise HelpServerError("the help server took too long to answer")
                chunk = answer.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_HTML_BYTES:
                    raise HelpServerError("the help server page is too large")
                chunks.append(chunk)
            charset = answer.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise PageMissing(path) from None
        raise HelpServerError(f"the help server answered {error.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise HelpServerError(f"the help server did not answer: {type(error).__name__}") from None
    return b"".join(chunks).decode(charset, "replace")


class HelpUrls:
    """The help server address of each session, asked for once and kept."""

    def __init__(self) -> None:
        self._urls: dict[str, str | None] = {}
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

    def clear(self) -> None:
        with self._lock:
            self._urls.clear()


URLS = HelpUrls()
