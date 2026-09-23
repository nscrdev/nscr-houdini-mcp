"""`hou_docs`: search or read Houdini's own documentation for the build in use.

Three modes, all read only.

- `search` looks for `query` in page titles and first paragraphs. Results are
  ranked: a title that is the query, then one that starts with it, then one
  that holds it, then a page whose path or first paragraph holds every word.
  Under `nodes/`, `vex/` and `hom/` the last part of a page's path counts as a
  title too, so a node's internal name finds it. Among equals the current
  version of a page comes before older ones and deprecated ones, and release
  notes come last of all. The first result's `note` says how big the index is
  and how long it took to build.
- `page` reads one page by its help path, such as `nodes/sop/attribwrangle`,
  or a node type with its namespace or version, such as
  `nodes/sop/copytopoints::2.0`.
- `vex` reads the page of one VEX function, such as `noise`.

Where the pages come from. The help folder of the install comes first: with a
session, only the folder of exactly the session's build, and with no session,
the install named in config (`hython` or `houdini_build`), or the newest one on
this machine. The session's help server is read only when there is no folder
for its build here, or for a page that folder does not have. The help server
runs inside the session, so it is slow and does not answer while the session
is busy: asking where it is never queues, and a help server that timed out is
left alone for a minute. `source` says which answered and `build` which build
was read. With neither, the call is `HELP_UNAVAILABLE`.

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
# told apart at once, and asked again on the next call.
ASK_WAIT_S = 0.0
ASK_TIMEOUT_S = 2.0

# Where a cut may move back to the end of a line, at most.
CUT_SLACK = 500

VEX_FOLDER = "vex/functions/"

HELP_SERVER = "help_server"
CORPUS = "corpus"

# Why a call that named no session goes on without one.
NO_SESSION_CODES = frozenset({"NO_SESSION", "SESSION_AMBIGUOUS", "STORE_UNAVAILABLE"})

_FUNCTION = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Place:
    """Where this call's pages can come from."""

    target: Target | None = None
    # The session's build, when there is a session.
    session_build: str | None = None
    # A help folder of exactly the session's build, or with no session the
    # configured or newest install's.
    corpus: helpdocs.Corpus | None = None
    notes: list[str] = field(default_factory=list)
    # Whether a help server answered this call, with a page or with none.
    served: bool = False

    def build_of(self, source: str) -> str | None:
        """The build a page from `source` was read for."""
        if source == CORPUS and self.corpus is not None:
            return self.corpus.build or None
        return self.session_build


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
    """The session to ask, if any, and the help folder to read."""
    place = Place()
    try:
        place.target = call.target()
    except CallError as error:
        if call.arguments.get("session") or error.code not in NO_SESSION_CODES:
            raise
        place.notes.append(f"no session to ask ({error.code}), so the help folder was read")
    if place.target is not None:
        place.corpus = _session_corpus(place)
    else:
        hfs = _configured_hfs(call, place)
        if hfs is not None:
            corpus = helpdocs.Corpus(hfs, _build_of(hfs))
            place.corpus = corpus if corpus.exists() else None
    return place


def _session_corpus(place: Place) -> helpdocs.Corpus | None:
    """The help folder of exactly the session's build, or nothing.

    The session's own install first, then any install of the same build. A
    folder of another build is never read for a session: its pages would be
    another Houdini's.
    """
    target = place.target
    assert target is not None
    facts = target.record.capabilities if isinstance(target.record.capabilities, dict) else {}
    build = target.houdini_version or facts.get("houdini_version") or None
    place.session_build = build
    candidates: list[Path] = []
    hfs = facts.get("hfs")
    if isinstance(hfs, str) and hfs:
        candidates.append(Path(hfs))
    if build:
        candidates.extend(found.hfs for found in install_module.find_installs())
    for candidate in candidates:
        said = helpdocs.build_of(candidate)
        if said is None and build and str(candidate) == hfs:
            # The session's own install, whose header is missing: its build
            # is the session's.
            said = build
        if build is not None and said != build:
            continue
        corpus = helpdocs.Corpus(candidate, said or "")
        if corpus.exists():
            return corpus
    place.notes.append(
        f"no help folder for Houdini {build or 'of this session'} on this machine,"
        " so the session's help server was read"
    )
    return None


def _configured_hfs(call: Call, place: Place) -> Path | None:
    """The install to read with no session: config, then this machine.

    The config's `hython` or `houdini_build` names the install, then
    `NSCR_MCP_HYTHON`, then the newest install found.
    """
    if call.config is not None:
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
    installs = install_module.find_installs()
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


def _cache_folder(call: Call, place: Place, source: str) -> Path:
    build = place.build_of(source)
    hfs = place.corpus.hfs if place.corpus is not None and place.corpus.build == build else None
    return helpdocs.docs_home(_state_home(call), build, hfs)


# Section: the help server


def help_base(
    call: Call,
    place: Place,
    *,
    refresh: bool = False,
    after: helpdocs.HelpServerError | None = None,
) -> str | None:
    """The session's help server address: kept from before, or asked for now.

    The ask never waits behind other work: a session that is busy says so at
    once, and nothing is kept, so the next call asks again.
    """
    target = place.target
    if target is None:
        return None
    if not refresh:
        known, url = helpdocs.URLS.get(target.session_id)
        if known:
            return url
    try:
        reply = call.bridge(
            "help.server", {}, wait_s=ASK_WAIT_S, timeout_s=ASK_TIMEOUT_S, skip_if_busy=True
        )
    except CallError as error:
        said = f"the session did not say where its help server is ({error.code})"
        place.notes.append(f"{after}, and {said}" if after is not None else said)
        return None
    data = reply.get("data") if isinstance(reply.get("data"), dict) else {}
    url = helpdocs.usable_url(data.get("url"))
    helpdocs.URLS.put(target.session_id, url)
    if url is None:
        place.notes.append("the session serves no help on this machine's loopback")
    return url


def from_server(
    call: Call, place: Place, path: str, *, query: dict[str, str] | None = None
) -> str | None:
    """A page from the help server, or nothing when there is none to ask.

    Raises `helpdocs.PageMissing` when the help server says it has no such
    page. A timeout leaves the help server alone for a minute. Any other
    failure asks the session for the address once more and tries again.
    """
    target = place.target
    if target is None:
        return None
    resting = helpdocs.URLS.quiet_for(target.session_id)
    if resting > 0:
        place.notes.append(
            f"the help server timed out lately and is left alone for {resting:.0f} s more"
        )
        return None
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
            if error.timed_out:
                helpdocs.URLS.timed_out(target.session_id)
                place.notes.append(
                    f"{error}; it is left alone for {helpdocs.QUIET_AFTER_TIMEOUT_S:.0f} s"
                )
                return None
            if attempt:
                place.notes.append(str(error))
                return None
            base = help_base(call, place, refresh=True, after=error)
            continue
        place.served = True
        return text
    return None


# Section: page


@dataclass
class Found:
    title: str | None
    text: str
    source: str
    version: str | None = None


def read_page(
    call: Call, place: Place, path: str, *, markdown: bool, budget: int, mode: str
) -> dict[str, Any]:
    found = read_corpus(call, place, path, markdown=markdown) if place.corpus else None
    if found is None:
        found = read_server(call, place, path, markdown=markdown)
    if found is None:
        if not place.served and place.corpus is None:
            raise unavailable(place)
        raise not_found(call, place, path, mode)
    return finish(call, place, path, found, budget=budget, mode=mode)


def read_corpus(call: Call, place: Place, path: str, *, markdown: bool) -> Found | None:
    """One page from the help folder, through the cache."""
    corpus = place.corpus
    assert corpus is not None
    form = "markdown" if markdown else "plain"
    folder = _cache_folder(call, place, CORPUS)
    key = helpdocs.cache_key(corpus.build, CORPUS, path, form, corpus.fingerprint())
    kept = helpdocs.cache_get(folder, key)
    if kept is not None:
        return Found(kept.get("title"), kept["text"], CORPUS, kept.get("version"))
    source = corpus.read(path)
    if source is None:
        return None
    page = helptext.markup_to_text(source, markdown=markdown, read=corpus.read, where=path)
    title = page.title or path.rsplit("/", 1)[-1]
    version = page.properties.get("version") or None
    body = {"path": path, "title": title, "text": page.text, "version": version}
    helpdocs.cache_put(folder, key, body)
    return Found(title, page.text, CORPUS, version)


def read_server(call: Call, place: Place, path: str, *, markdown: bool) -> Found | None:
    """One page from the session's help server, through the cache."""
    if place.target is None:
        return None
    form = "markdown" if markdown else "plain"
    folder = _cache_folder(call, place, HELP_SERVER)
    key = helpdocs.cache_key(place.session_build, HELP_SERVER, path, form, "")
    kept = helpdocs.cache_get(folder, key)
    if kept is not None:
        return Found(kept.get("title"), kept["text"], HELP_SERVER)
    try:
        page = from_server(call, place, path)
    except helpdocs.PageMissing:
        return None
    if page is None:
        return None
    title, text = helptext.html_to_text(page, markdown=markdown)
    if not text:
        return None
    if place.session_build:
        # A page is only kept under a build that is known.
        helpdocs.cache_put(folder, key, {"path": path, "title": title, "text": text})
    return Found(title, text, HELP_SERVER)


def finish(
    call: Call, place: Place, path: str, found: Found, *, budget: int, mode: str
) -> dict[str, Any]:
    text = found.text
    result: dict[str, Any] = {
        "mode": mode,
        "path": path,
        "title": found.title,
        "source": found.source,
        "build": place.build_of(found.source),
        "truncated": len(text) > budget,
    }
    version = helpdocs.version_label(path, found.version)
    if version:
        result["version"] = version
    if len(text) > budget:
        cut = text[:budget]
        line_end = cut.rfind("\n")
        if line_end >= budget - CUT_SLACK:
            cut = cut[:line_end]
        result["text"] = cut.rstrip()
        result["total_chars"] = len(text)
        result["spill_path"] = spill(call, path, found)
    else:
        result["text"] = text
    if place.notes:
        result["note"] = "; ".join(place.notes)
    return result


def spill(call: Call, path: str, found: Found) -> str:
    """Write the whole page to the spill folder and say where it went.

    Raises `SPILL_FAILED` when it cannot be written: a cut page with nowhere
    to find the rest is not an answer.
    """
    if call.config is None:
        raise CallError(
            "SPILL_FAILED", "the page was too long to return and no spill folder is set"
        )
    body = json.dumps(
        {"path": path, "title": found.title, "source": found.source, "text": found.text},
        ensure_ascii=False,
    )
    written = Spill(call.config.spill_folder, call.config.spill_over_bytes).write(
        body, tool="hou_docs"
    )
    return written["path"]


def unavailable(place: Place) -> CallError:
    details: dict[str, Any] = {"build": place.session_build}
    if place.notes:
        details["notes"] = place.notes
    return CallError(
        "HELP_UNAVAILABLE",
        "no help server answered and no help folder was found for this Houdini",
        details=details,
    )


def not_found(call: Call, place: Place, path: str, mode: str) -> CallError:
    near: list[str] = []
    if place.corpus is not None:
        paths = [row[0] for row in helpdocs.load_index(_state_home(call), place.corpus).pages]
        if mode == "vex":
            names = [each[len(VEX_FOLDER) :] for each in paths if each.startswith(VEX_FOLDER)]
            near = did_you_mean(path[len(VEX_FOLDER) :], names)
        else:
            near = did_you_mean(path, paths)
    build = place.build_of(CORPUS if place.corpus is not None else HELP_SERVER)
    what = f"function {path[len(VEX_FOLDER) :]}" if mode == "vex" else f"page at {path}"
    details: dict[str, Any] = {"path": path, "did_you_mean": near, "build": build}
    if place.notes:
        details["notes"] = place.notes
    return CallError(
        "DOC_NOT_FOUND",
        f"the help for Houdini {build or 'this build'} has no {what}",
        details=details,
    )


# Section: search


def search(call: Call, place: Place, query: str, limit: int) -> dict[str, Any]:
    if place.corpus is not None:
        index = helpdocs.load_index(_state_home(call), place.corpus)
        results = [
            _row(path, title, excerpt, CORPUS, version)
            for _, (path, title, excerpt, version, _) in helpdocs.search_index(index, query, limit)
        ]
        notes = [index.note(), *place.notes]
        source = CORPUS
    else:
        results = search_server(call, place, query, limit)
        if not place.served:
            raise unavailable(place)
        notes = ["the help server's own search", *place.notes]
        source = HELP_SERVER
    result: dict[str, Any] = {"mode": "search", "query": query, "build": place.build_of(source)}
    if results:
        results[0] = {**results[0], "note": "; ".join(notes)}
    else:
        result["note"] = "; ".join(notes)
    result["results"] = results
    return result


def search_server(call: Call, place: Place, query: str, limit: int) -> list[dict[str, Any]]:
    """The help server's own search, ranked the way the index is."""
    try:
        page = from_server(call, place, "_search", query={"q": query})
    except helpdocs.PageMissing:
        return []
    if page is None:
        return []
    ranked = []
    for hit in helptext.search_hits(page, helpdocs.tidy_path):
        tier = helpdocs.rank(query, hit["title"], hit["path"], hit["excerpt"])
        # The help server found it, in the body if not in the title.
        tier = 3 if tier is None else tier
        ranked.append((helpdocs.order(tier, hit["path"], hit["title"]), hit))
    ranked.sort(key=lambda item: item[0])
    return [
        _row(hit["path"], hit["title"], hit["excerpt"], HELP_SERVER, None)
        for _, hit in ranked[:limit]
    ]


def _row(path: str, title: str, excerpt: str, source: str, version: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {"path": path, "title": title, "excerpt": excerpt, "source": source}
    label = helpdocs.version_label(path, version)
    if label:
        row["version"] = label
    return row


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
