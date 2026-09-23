"""`hou_docs`: search or read Houdini's own documentation for the build in use.

Three modes, all read only.

- `search` looks for `query` in page titles and first paragraphs. Results are
  ranked: a title that is the query, then one that starts with it, then one
  that holds it, then a page whose first paragraph or path holds every word.
  The last part of a page's path counts as a title too, so a node's internal
  name finds it. The first result's `note` says how big the index is and how
  long it took to build.
- `page` reads one page by its help path, such as `nodes/sop/attribwrangle`.
- `vex` reads the page of one VEX function, such as `noise`.

Where the pages come from, in order: the help server of the session the call
reaches, then the help folder of the install. `session` only decides which
build and which help server: a read never queues behind a busy session, and
with no session at all the install named in config (`hython` or
`houdini_build`) is read, or the newest one on this machine. `source` says
which answered. With neither, the call is `HELP_UNAVAILABLE`.

A page's text is cut at `max_chars` (20,000 unless you say), with `truncated`
and the whole page written to the spill folder, named in `spill_path`.

This module never imports `hou`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import helpdocs, helptext, pool
from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge.errors import did_you_mean
from nscr_houdini_mcp.config import ConfigError, resolve_hython
from nscr_houdini_mcp.results import CallError, Spill
from nscr_houdini_mcp.router import Target
from nscr_houdini_mcp.tools.base import SESSION, Call, ToolSpec, inputs, outputs

MODES = ("search", "page", "vex")

DEFAULT_LIMIT = 10
MAX_LIMIT = 50

DEFAULT_MAX_CHARS = 20_000
MIN_MAX_CHARS = 100
MAX_MAX_CHARS = 1_000_000

# Asking a session where its help server is never queues: a busy session is
# read from the help folder instead, and asked again on the next call.
ASK_WAIT_S = 0.0
ASK_TIMEOUT_S = 10.0

# Where a cut may move back to the end of a line, at most.
CUT_SLACK = 500

VEX_FOLDER = "vex/functions/"

# Why a call that named no session goes on without one.
NO_SESSION_CODES = frozenset({"NO_SESSION", "SESSION_AMBIGUOUS", "STORE_UNAVAILABLE"})

_FUNCTION = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Place:
    """Where this call's pages can come from."""

    target: Target | None = None
    corpus: helpdocs.Corpus | None = None
    notes: list[str] = field(default_factory=list)
    # Whether a help server answered this call, with a page or with none.
    served: bool = False

    @property
    def build(self) -> str | None:
        if self.corpus is not None and self.corpus.build:
            return self.corpus.build
        if self.target is not None:
            return self.target.houdini_version
        return None


def docs(call: Call) -> Mapping[str, Any]:
    arguments = call.arguments
    mode = arguments.get("mode") or _mode_of(arguments)
    markdown = arguments.get("format") == "markdown"
    limit = _within("limit", arguments.get("limit"), 1, MAX_LIMIT) or DEFAULT_LIMIT
    budget = (
        _within("max_chars", arguments.get("max_chars"), MIN_MAX_CHARS, MAX_MAX_CHARS)
        or DEFAULT_MAX_CHARS
    )
    if mode == "search":
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise _bad("query", "search needs a query")
        return search(call, locate(call), query, limit)
    path = _path_of(mode, arguments)
    return read_page(call, locate(call), path, markdown=markdown, budget=budget, mode=mode)


def _mode_of(arguments: Mapping[str, Any]) -> str:
    for name, mode in (("function", "vex"), ("path", "page"), ("query", "search")):
        if arguments.get(name):
            return mode
    raise _bad("mode", "pass mode, or one of query, path or function")


def _path_of(mode: str, arguments: Mapping[str, Any]) -> str:
    if mode == "vex":
        name = str(arguments.get("function") or "").strip()
        if not name:
            raise _bad("function", "vex needs a function name such as noise")
        if not _FUNCTION.match(name):
            raise _bad("function", f"{name} is not a VEX function name")
        return VEX_FOLDER + name
    given = str(arguments.get("path") or "")
    if not given.strip():
        raise _bad("path", "page needs a help path such as nodes/sop/attribwrangle")
    path = helpdocs.tidy_path(given)
    if path is None:
        raise _bad("path", f"{given} is not a help path such as nodes/sop/attribwrangle")
    return path


def _within(name: str, value: Any, low: int, high: int) -> int | None:
    """Refuse a number outside its range. Checked here to keep the schema small."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise CallError(
            "BAD_ARGUMENTS",
            f"{name} must be a whole number from {low} to {high}",
            details={"argument": name, "given": value},
        )
    return value


def _bad(argument: str, message: str) -> CallError:
    return CallError("BAD_ARGUMENTS", message, details={"argument": argument})


# Section: where the pages are


def locate(call: Call) -> Place:
    """The session to ask, if any, and the install whose help folder to read."""
    place = Place()
    try:
        place.target = call.target()
    except CallError as error:
        if call.arguments.get("session") or error.code not in NO_SESSION_CODES:
            raise
        place.notes.append(f"no session to ask ({error.code}), so the help folder was read")
    hfs, build = _from_session(place.target)
    if hfs is None:
        hfs = _configured_hfs(call, build, place)
    if hfs is not None:
        place.corpus = helpdocs.Corpus(hfs, build or _build_of(hfs))
    return place


def _from_session(target: Target | None) -> tuple[Path | None, str | None]:
    if target is None:
        return None, None
    facts = target.record.capabilities if isinstance(target.record.capabilities, dict) else {}
    build = target.houdini_version or facts.get("houdini_version")
    hfs = facts.get("hfs")
    return (Path(hfs) if isinstance(hfs, str) and hfs else None), build


def _configured_hfs(call: Call, build: str | None, place: Place) -> Path | None:
    """The install to read: the session's build, then config, then this machine.

    A session whose own folder is not known is matched to an install by its
    build. Without a session, the config's `hython` or `houdini_build` names
    the install, then `NSCR_MCP_HYTHON`, then the newest install found.
    """
    installs = install_module.find_installs()
    if build:
        for found in installs:
            if found.version == build:
                return found.hfs
    if place.target is None and call.config is not None:
        try:
            hython = resolve_hython(call.config)
        except ConfigError as error:
            place.notes.append(error.message)
            hython = None
        if hython is not None:
            return helpdocs.hfs_of_hython(Path(hython))
    named = os.environ.get(pool.HYTHON_ENV_VAR, "").strip()
    if named:
        return helpdocs.hfs_of_hython(Path(named).expanduser())
    return installs[0].hfs if installs else None


def _build_of(hfs: Path) -> str:
    said = helpdocs.build_of(hfs)
    if said:
        return said
    for found in install_module.find_installs():
        if found.hfs == hfs and found.version:
            return found.version
    return ""


def _state_home(call: Call) -> Path:
    return call.config.state_home if call.config is not None else store_module.default_home()


# Section: the help server


def help_base(call: Call, place: Place, *, refresh: bool = False) -> str | None:
    """The session's help server address: kept from before, or asked for now."""
    target = place.target
    if target is None:
        return None
    if not refresh:
        known, url = helpdocs.URLS.get(target.session_id)
        if known:
            return url
    try:
        reply = call.bridge("help.server", {}, wait_s=ASK_WAIT_S, timeout_s=ASK_TIMEOUT_S)
    except CallError as error:
        # Not kept: a session busy now can say on the next call.
        place.notes.append(f"the session did not say where its help server is ({error.code})")
        return None
    data = reply.get("data") if isinstance(reply.get("data"), dict) else {}
    url = helpdocs.usable_url(data.get("url"))
    helpdocs.URLS.put(target.session_id, url)
    if place.corpus is None and isinstance(data.get("hfs"), str) and data["hfs"]:
        hfs = Path(data["hfs"])
        place.corpus = helpdocs.Corpus(hfs, data.get("houdini_version") or _build_of(hfs))
    return url


def from_server(
    call: Call, place: Place, path: str, *, query: dict[str, str] | None = None
) -> str | None:
    """A page from the help server, or nothing when there is none to ask.

    Raises `helpdocs.PageMissing` when the help server says it has no such
    page. Any other failure asks the session for the address once more and
    tries again, then gives up for the help folder.
    """
    base = help_base(call, place)
    for attempt in range(2):
        if base is None:
            return None
        try:
            text = helpdocs.fetch(base, path, query=query)
        except helpdocs.PageMissing:
            place.served = True
            raise
        except helpdocs.HelpServerError as error:
            if attempt:
                place.notes.append(f"{error}, so the help folder was read")
                return None
            base = help_base(call, place, refresh=True)
            continue
        place.served = True
        return text
    return None


# Section: page


def read_page(
    call: Call, place: Place, path: str, *, markdown: bool, budget: int, mode: str
) -> dict[str, Any]:
    found: tuple[str | None, str, str] | None = None
    try:
        page = from_server(call, place, path)
    except helpdocs.PageMissing:
        page = None
    if page is not None:
        title, text = helptext.html_to_text(page, markdown=markdown)
        if text:
            found = (title, text, "help_server")
    shipped = place.corpus is not None and place.corpus.exists()
    if found is None and shipped:
        read = read_corpus(call, place.corpus, path, markdown=markdown)
        if read is not None:
            found = (read[0], read[1], "corpus")
    if found is None:
        if not place.served and not shipped:
            raise unavailable(place)
        raise not_found(call, place, path, mode)
    title, text, source = found
    return finish(call, place, path, title, text, source, budget=budget, mode=mode)


def read_corpus(
    call: Call, corpus: helpdocs.Corpus, path: str, *, markdown: bool
) -> tuple[str | None, str] | None:
    """One page from the help folder, through the cache."""
    form = "markdown" if markdown else "plain"
    fingerprint = corpus.fingerprint()
    folder = helpdocs.docs_home(_state_home(call), corpus)
    key = helpdocs.cache_key(corpus, path, form, fingerprint)
    kept = helpdocs.cache_get(folder, key)
    if kept is not None:
        return kept.get("title"), kept["text"]
    source = corpus.read(path)
    if source is None:
        return None
    page = helptext.markup_to_text(source, markdown=markdown, read=corpus.read, where=path)
    title = page.title or path.rsplit("/", 1)[-1]
    helpdocs.cache_put(folder, key, {"path": path, "title": title, "text": page.text})
    return title, page.text


def finish(
    call: Call,
    place: Place,
    path: str,
    title: str | None,
    text: str,
    source: str,
    *,
    budget: int,
    mode: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": mode,
        "path": path,
        "title": title,
        "source": source,
        "build": place.build,
        "truncated": len(text) > budget,
    }
    if len(text) > budget:
        cut = text[:budget]
        line_end = cut.rfind("\n")
        if line_end >= budget - CUT_SLACK:
            cut = cut[:line_end]
        result["text"] = cut.rstrip()
        result["total_chars"] = len(text)
        result["spill_path"] = spill(call, path, title, source, text)
    else:
        result["text"] = text
    if place.notes:
        result["note"] = "; ".join(place.notes)
    return result


def spill(call: Call, path: str, title: str | None, source: str, text: str) -> str | None:
    """Write the whole page to the spill folder and say where it went."""
    if call.config is None:
        return None
    body = json.dumps(
        {"path": path, "title": title, "source": source, "text": text}, ensure_ascii=False
    )
    try:
        written = Spill(call.config.spill_folder, call.config.spill_over_bytes).write(
            body, tool="hou_docs"
        )
    except CallError:
        return None
    return written["path"]


def unavailable(place: Place) -> CallError:
    details: dict[str, Any] = {"build": place.build}
    if place.notes:
        details["notes"] = place.notes
    return CallError(
        "HELP_UNAVAILABLE",
        "no help server answered and no help folder was found for this Houdini",
        details=details,
    )


def not_found(call: Call, place: Place, path: str, mode: str) -> CallError:
    near: list[str] = []
    if place.corpus is not None and place.corpus.exists():
        paths = [row[0] for row in helpdocs.load_index(_state_home(call), place.corpus).pages]
        if mode == "vex":
            names = [each[len(VEX_FOLDER) :] for each in paths if each.startswith(VEX_FOLDER)]
            near = did_you_mean(path[len(VEX_FOLDER) :], names)
        else:
            near = did_you_mean(path, paths)
    what = f"function {path[len(VEX_FOLDER) :]}" if mode == "vex" else f"page at {path}"
    return CallError(
        "DOC_NOT_FOUND",
        f"the help for Houdini {place.build or 'this build'} has no {what}",
        details={"path": path, "did_you_mean": near, "build": place.build},
    )


# Section: search


def search(call: Call, place: Place, query: str, limit: int) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    tiers: dict[str, int] = {}
    try:
        page = from_server(call, place, "_search", query={"q": query})
    except helpdocs.PageMissing:
        page = None
    if page is not None:
        for hit in helptext.search_hits(page):
            tier = helpdocs.rank(query, hit["title"], hit["path"], hit["excerpt"])
            # The help server found it, in the body if not in the title.
            tiers[hit["path"]] = 3 if tier is None else tier
            rows[hit["path"]] = {**hit, "source": "help_server"}
    index = None
    if place.corpus is not None and place.corpus.exists():
        index = helpdocs.load_index(_state_home(call), place.corpus)
        for tier, path, title, excerpt in helpdocs.search_index(index, query, max(limit, 50)):
            if path in rows:
                if not rows[path]["excerpt"]:
                    rows[path]["excerpt"] = excerpt
                continue
            tiers[path] = tier
            rows[path] = {"path": path, "title": title, "excerpt": excerpt, "source": "corpus"}
    if not place.served and index is None:
        raise unavailable(place)
    ordered = sorted(
        rows.values(), key=lambda row: (tiers[row["path"]], len(row["title"]), row["path"])
    )
    results = ordered[:limit]
    notes = ([index.note()] if index is not None else ["help server search only"]) + place.notes
    result: dict[str, Any] = {"mode": "search", "query": query, "build": place.build}
    if results:
        results[0] = {**results[0], "note": "; ".join(notes)}
    else:
        result["note"] = "; ".join(notes)
    result["results"] = results
    return result


# Section: the one line a long result is summed up in


def summary_line(data: Mapping[str, Any]) -> str:
    if data.get("mode") == "search":
        shown = data.get("results") or []
        first = f"; first {shown[0]['path']}" if shown else ""
        return f"hou_docs search {data.get('query')!r}: {len(shown)} results{first}"
    line = (
        f"hou_docs {data.get('mode')} {data.get('path')}: {data.get('title')},"
        f" {len(data.get('text') or '')} characters from {data.get('source')}"
    )
    if data.get("truncated"):
        line += f", cut from {data.get('total_chars')}; all of it in {data.get('spill_path')}"
    return line


HOU_DOCS = ToolSpec(
    name="hou_docs",
    description=(
        "Search or read Houdini's own documentation for the installed build. "
        "Modes: search, page, vex."
    ),
    input_schema=inputs(
        {
            "mode": {"type": "string", "enum": list(MODES)},
            "session": SESSION,
            "query": {"type": "string"},
            "path": {"type": "string", "description": "e.g. nodes/sop/attribwrangle"},
            "function": {"type": "string"},
            "limit": {"type": "integer"},
            "format": {"type": "string", "enum": list(helptext.FORMATS)},
            "max_chars": {"type": "integer"},
        }
    ),
    output_schema=outputs({}),
    handler=docs,
    read_only=True,
    idempotent=True,
    open_world=False,
    summary=summary_line,
)
