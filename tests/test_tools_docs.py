"""`hou_docs` through the server, against a stand in help folder and help server.

The help folder is a tiny tree made here in the layout a real install ships:
one zip per book holding help markup, a book kept as a plain folder, and a
version header. The help server is a small HTTP server of the test's own,
serving HTML shaped like the real one's and a search page. The session behind
it is a real dispatcher over the stand in `hou`, whose `helpServerUrl` points
at that server, so the address comes the way it does in use: one bridge call,
kept per session, that never queues.

Three places a call can stand: no session at all, a session whose build has
this help folder (`matched`), and a session of a build with no help folder
here (`remote`), which only its help server can answer for.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import zipfile
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from fake_hou import Scene
from nscr_houdini_mcp import helpdocs, helptext, pool
from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp.bridge import tools
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.handlers import default_registry
from nscr_houdini_mcp.bridge.tools import ToolContext
from nscr_houdini_mcp.tools.registry import TOOLS
from test_server import talk, text_of
from test_tools_inspect import Through
from test_tools_sessions import Bench

BUILD = "22.0.999"
OTHER_BUILD = "22.0.998"

# Section: a help folder

PAGES = {
    "nodes.zip": {
        "sop/attribwrangle.txt": """= Attribute Wrangle =

#type: node
#context: sop
#internal: attribwrangle

\"\"\"Runs a VEX snippet to modify attribute values.\"\"\"

== Overview ==

This node _runs the snippet_ on every point. See [Point Wrangle|Node:sop/pointwrangle].

WARNING:
    This node requires that you understand the [vex language|/vex/].

* Press ((MMB)) on the node to see errors.

== Syntax ==

:include wrangle_syntax:

@parameters

Group:
    A subset of points to run the program on.

:include _run_over:

:include pointwrangle#snippet:

{{{
#!vex
@P.y += 1;
}}}

@related

- [Node:sop/pointwrangle]
""",
        "sop/wrangle_syntax.txt": """#type: include

The __VEX snippet__ parameter holds the code to run.
""",
        "sop/_run_over.txt": """#type: include

Run Over:
    #id: class

    Apply the VEX code to each component of this type.
""",
        "sop/pointwrangle.txt": """= Point Wrangle =

#type: node
#context: sop

:warning:Deprecated:
    The Point Wrangle SOP is now deprecated.

\"\"\"Runs a VEX snippet to modify point attributes.\"\"\"

@parameters

VEXpression:
    #id: snippet

    A snippet of VEX code that will manipulate the point attributes.
""",
        "sop/wranglehelper.txt": """= Wrangle Helper =

#type: node

\"\"\"Helps with wrangles.\"\"\"
""",
        "cop/wrangle.txt": """= Wrangle =

#type: node
#context: cop

\"\"\"Runs a VEX snippet to modify layer values.\"\"\"
""",
        "sop/box.txt": """= Box =

\"\"\"Creates a cube.\"\"\"
""",
        "sop/index.txt": """= Geometry nodes =

\"\"\"Every geometry node.\"\"\"
""",
        "sop/copytopoints.txt": """= Copy to Points =

#type: node
#version: 2.0

\"\"\"Copies geometry onto points.\"\"\"
""",
        "sop/copytopoints-.txt": """= Copy to Points =

#type: node

\"\"\"The older way to copy geometry onto points.\"\"\"
""",
        "sop/guide.txt": """= Guide =

#type: node
#version: 3.0

\"\"\"The current guide.\"\"\"
""",
        "sop/guide-.txt": """= Guide =

#type: node
#version: 1.5

\"\"\"The guide before.\"\"\"
""",
        "sop/labs--grid-1.1.txt": """= Labs Grid =

#type: node
#version: 1.1

\"\"\"A grid from the labs.\"\"\"
""",
        "sop/table.txt": """= Table Page =

\"\"\"Has a table.\"\"\"

table>>
    tr>>
        th>> Name
        th>> Type
    tr>>
        td>> `P`
        td>> vector
    tr>>
        td>>
        `orient`
        td>>
            quaternion
            of the shape

After the table.
""",
    },
    "vex.zip": {
        "index.txt": """= VEX =

\"\"\"VEX is a high-performance expression language.\"\"\"
""",
        "functions/noise.txt": """= noise =

#type: vex
#group: noise

\"\"\"Perlin-style noise.\"\"\"

:usage: `float noise(vector pos)`

    Sample 3D noise.

NOTE:
    This function generates non-periodic noise.

@related

:include _common#noiselinks/:
""",
        "functions/pnoise.txt": """= pnoise =

\"\"\"Periodic noise.\"\"\"
""",
        "functions/_common.txt": """#type: include

:null:
    #id: noiselinks

    - [Vex:pnoise]
    - [Vex:snoise]
""",
    },
    "news.zip": {
        "20/wrangle.txt": """= Wrangle =

\"\"\"What is new in wrangles.\"\"\"
""",
    },
}

FOLDER_PAGES = {
    "copernicus/intro.txt": """= Copernicus intro =

Layers you can wrangle and more.
""",
}


def make_install(root: Path, *, with_help: bool = True, build: str = BUILD) -> Path:
    """An install folder with a hython in it and, unless told, its help."""
    hfs = root / "hfs"
    (hfs / "bin").mkdir(parents=True)
    (hfs / "bin" / "hython").write_text("", encoding="utf-8")
    header = hfs / "toolkit" / "include" / "SYS"
    header.mkdir(parents=True)
    (header / "SYS_Version.h").write_text(f'#define SYS_VERSION_FULL "{build}"\n', encoding="utf-8")
    if not with_help:
        return hfs
    help_root = hfs / "houdini" / "help"
    help_root.mkdir(parents=True)
    for book, members in PAGES.items():
        with zipfile.ZipFile(help_root / book, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, text in members.items():
                archive.writestr(name, text)
    for name, text in FOLDER_PAGES.items():
        path = help_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (help_root / "index.txt").write_text("= Houdini =\n\nThe start page.\n", encoding="utf-8")
    return hfs


# Section: a help server

PAGE_HTML = """<!DOCTYPE html>
<html><head><title>Attribute Wrangle</title><script>var nav = "script text";</script></head>
<body>
<nav role="navigation"><a href="/">Navigation bar text</a></nav>
<main>
<header><div id="title">
<p class="ancestors"><a class="ancestor">Houdini 22.0</a></p>
<div class="pageicon"><img src="icon.svg"/></div>
<h1 class="title">Attribute
Wrangle <span class="subtitle">geometry node</span></h1>
<p class="summary">Runs a VEX snippet to modify attribute values.</p>
</div></header>
<div id="content">
<table id="premeta" class="metatable"><tr><td class="label">On this page</td></tr></table>
<section class="heading"><h2 class="label heading" id="overview">Overview
<span class="headerlink"><a href="#overview">&#182;</a></span></h2>
<div class="content"><p>Served overview text.</p>
<ul class="bullets"><li class="bullet"><p class="label">Press
<span class="keys"><img class="keyicon" title="MMB" src="mmb.svg"/></span> on the node.</p></li>
</ul></div></section>
<div id="group" class="parameter sbs-item"><p class="label">Group</p>
<div class="content"><p>A subset of <code>points</code>.</p></div></div>
<table><tr><th>Name</th><th>Type</th></tr><tr><td><p>P</p></td><td>vector</td></tr></table>
<pre>@P.y += 1;
@Cd = 1;</pre>
</div>
</main>
<div id="toc">Table of contents text</div>
</body></html>
"""

NOISE_HTML = """<html><body><main><header><h1 class="title">noise</h1></header>
<div id="content"><div class="usage item"><p class="label"><code class="vexsignature">
float&nbsp;noise(vector pos)</code></p></div><p>Served noise text.</p></div></main></body></html>
"""

SEARCH_HTML = """<div class="results-inner"><p class="stats">3 hits</p>
<div class="hit-block"><div class="hit findpage"><p class="label">
<a class="label" href="/find?q=wrangle">Open on separate page</a></p></div>
<div class="instants"><div class="hit instant"><p class="label">
<a class="label" href="/nodes/cop/wrangle">Wrangle</a>
<small class="desc">Copernicus node</small></p>
<div class="content"><p class="summary">Runs a VEX snippet to modify layer values.</p></div>
</div></div>
<section class="search-category"><div class="hits" data-name="node/sop">
<div class="hit"><p class="label"><a class="label" href="/nodes/sop/attribwrangle.html">
Attribute
Wrangle</a><small class="desc">geometry node</small></p></div>
<div class="hit"><p class="label"><a class="label" href="/nodes/sop/attribwrangle">
Attribute Wrangle</a><small class="desc">geometry node</small></p></div>
<div class="hit"><p class="label"><a class="label" href="/nodes/dop/popwrangle">
POP Wrangle</a><small class="desc">dynamics node</small></p></div>
<div class="hit"><p class="label"><a class="label" href="/news/20/wrangle">
Wrangle</a><small class="desc">what is new</small></p></div>
<div class="hit"><p class="label"><a class="label" href="/nodes/sop/snippetsop">
Snippet SOP</a><small class="desc">geometry node</small></p></div>
<div class="hit"><p class="label"><a class="label" href="http://192.0.2.1:9/x">
Elsewhere</a></p></div>
<div class="more hit"><p class="label">11 more in Geometry nodes</p></div>
</div></section></div></div>
"""


class HelpServer:
    """A help server of the test's own, on loopback."""

    def __init__(self) -> None:
        self.pages: dict[str, str] = {
            "/nodes/sop/attribwrangle": PAGE_HTML,
            "/vex/functions/noise": NOISE_HTML,
            "/nodes/dop/popwrangle": PAGE_HTML.replace("Attribute", "POP"),
        }
        # Where a path answers with a redirect.
        self.redirects: dict[str, str] = {}
        self.delay = 0.0
        # Seconds between the pieces of a body sent slowly, when set.
        self.drip = 0.0
        self.asked: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the name is the standard library's
                owner.asked.append(self.path)
                if owner.delay:
                    time.sleep(owner.delay)
                where = urlsplit(self.path)
                if where.path in owner.redirects:
                    self.send_response(302)
                    self.send_header("Location", owner.redirects[where.path])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if where.path == "/_search":
                    body = SEARCH_HTML if parse_qs(where.query).get("q") else ""
                else:
                    body = owner.pages.get(where.path)
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = body.encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    if owner.drip:
                        for start in range(0, len(data), 16):
                            self.wfile.write(data[start : start + 16])
                            self.wfile.flush()
                            time.sleep(owner.drip)
                    else:
                        self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def dead_url() -> str:
    """The address of a help server that has gone: it listened, then closed.

    Whether this system refuses a connection there at once or lets it hang,
    the help server has gone either way, and must be read as that.
    """
    gone = HelpServer()
    url = gone.url
    gone.close()
    return url


class Silent:
    """A help server that takes a connection and never answers, as a busy one does."""

    def __init__(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.held: list[socket.socket] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._take, daemon=True)
        self.thread.start()

    def _take(self) -> None:
        self.listener.settimeout(0.05)
        while not self.stop.is_set():
            try:
                taken, _ = self.listener.accept()
            except OSError:
                continue
            self.held.append(taken)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.listener.getsockname()[1]}/"

    def close(self) -> None:
        self.stop.set()
        self.thread.join(2.0)
        for taken in self.held:
            taken.close()
        self.listener.close()


# Section: fixtures


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Nothing from another test, and no Houdini of this machine's own."""
    monkeypatch.setattr(install_module, "find_installs", lambda configured=None: [])
    monkeypatch.delenv(pool.HYTHON_ENV_VAR, raising=False)
    monkeypatch.delenv("HFS", raising=False)

    def forget() -> None:
        helpdocs.URLS.clear()
        helpdocs.forget_indexes()
        helpdocs.forget_sweeps()
        helpdocs.close_archives()

    forget()
    yield
    forget()


@pytest.fixture
def hfs(tmp_path: Path) -> Path:
    return make_install(tmp_path / "install")


@pytest.fixture
def served() -> Iterator[HelpServer]:
    made = HelpServer()
    try:
        yield made
    finally:
        made.close()


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


def bench_for(tmp_path: Path, hython: Path) -> Bench:
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.config = replace(made.config, hython=hython)
    return made


@pytest.fixture
def lone(tmp_path: Path, hfs: Path) -> Bench:
    """No session at all; config names the install by its hython."""
    return bench_for(tmp_path, hfs / "bin" / "hython")


@pytest.fixture
def matched(tmp_path: Path, hfs: Path, scene: Scene, served: HelpServer) -> Bench:
    """A live session on this install: its help folder is the session's build."""
    made = bench_for(tmp_path, hfs / "bin" / "hython")
    add_session(made, hfs, build=BUILD)
    scene.help_url = served.url
    made.sent = Through(scene)  # type: ignore[assignment]
    return made


@pytest.fixture
def remote(tmp_path: Path, hfs: Path, scene: Scene, served: HelpServer) -> Bench:
    """A live session of a build with no help folder here.

    Config names this machine's install, of another build, which a session of
    that build must never be answered from.
    """
    made = bench_for(tmp_path, hfs / "bin" / "hython")
    add_session(made, None, build=OTHER_BUILD)
    scene.help_url = served.url
    made.sent = Through(scene)  # type: ignore[assignment]
    return made


def add_session(bench: Bench, hfs: Path | None, *, build: str, session_id: str = "s-1") -> None:
    facts: dict[str, Any] = {"houdini_version": build}
    if hfs is not None:
        facts["hfs"] = str(hfs)
    with bench.store() as store:
        store.register_session(
            session_id,
            kind="hython",
            pid=os.getpid(),
            pid_start=bench.stamp,
            alias="w1",
            port=18000,
            hip_path=None,
            capabilities=facts,
        )
    bench.reachable.add(session_id)


def docs(bench: Bench, **arguments: Any) -> Any:
    _, [result] = talk(bench.serve(), ("hou_docs", arguments))
    return result


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, text_of(result)
    return result.structured_content


def failed(result: Any, code: str) -> dict[str, Any]:
    assert result.is_error, text_of(result)
    error = result.structured_content["error"]
    assert error["code"] == code, error
    return error


def asked_bridge(bench: Bench) -> int:
    return sum(1 for call in bench.sent.calls if call["tool"] == "help.server")  # type: ignore[attr-defined]


def pages_asked(served: HelpServer) -> list[str]:
    return [path for path in served.asked if not path.startswith("/_search")]


# Section: the tool list


def test_hou_docs_is_listed_last_and_read_only() -> None:
    assert TOOLS[-1].name == "hou_docs"
    tool = TOOLS[-1].as_tool()
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.input_schema["additionalProperties"] is False
    assert set(tool.input_schema["properties"]) == {
        "mode",
        "session",
        "query",
        "path",
        "function",
        "limit",
        "format",
        "max_chars",
    }


# Section: the help folder first


def test_a_session_of_this_build_is_answered_from_the_folder_without_asking_it(
    matched: Bench, served: HelpServer
) -> None:
    body = ok(docs(matched, path="nodes/sop/attribwrangle"))
    assert body["source"] == "corpus"
    assert body["build"] == BUILD
    assert body["trace"]["session_id"] == "s-1"
    assert asked_bridge(matched) == 0
    assert served.asked == []
    found = ok(docs(matched, query="wrangle"))
    assert {row["source"] for row in found["results"]} == {"corpus"}
    assert served.asked == []


def test_a_page_the_folder_does_not_have_comes_from_the_help_server(
    matched: Bench, served: HelpServer
) -> None:
    body = ok(docs(matched, path="nodes/dop/popwrangle"))
    assert body["source"] == "help_server"
    assert body["build"] == BUILD
    assert body["title"] == "POP Wrangle"
    assert asked_bridge(matched) == 1


def test_a_busy_session_of_this_build_is_answered_from_the_folder(
    matched: Bench, scene: Scene
) -> None:
    with busy(matched, scene):
        started = time.monotonic()
        body = ok(docs(matched, mode="vex", function="noise"))
        took = time.monotonic() - started
    assert body["source"] == "corpus"
    assert took < 0.5
    assert asked_bridge(matched) == 0


def test_the_folder_of_another_build_is_never_read_for_a_session(
    tmp_path: Path, scene: Scene, served: HelpServer
) -> None:
    # The session says its install is here, but the install is another build.
    other = make_install(tmp_path / "other", build=BUILD)
    made = bench_for(tmp_path, other / "bin" / "hython")
    add_session(made, other, build=OTHER_BUILD)
    scene.help_url = served.url
    made.sent = Through(scene)  # type: ignore[assignment]
    body = ok(docs(made, path="nodes/sop/attribwrangle"))
    assert body["source"] == "help_server"
    assert body["build"] == OTHER_BUILD
    assert "no help folder for Houdini 22.0.998" in body["note"]
    # Kept under the session's build, not the other install's.
    assert (made.home / "docs" / OTHER_BUILD / "pages").is_dir()
    assert not (made.home / "docs" / BUILD).exists()


def test_with_no_session_the_configured_install_is_read(lone: Bench) -> None:
    body = ok(docs(lone, path="nodes/sop/attribwrangle"))
    assert body["source"] == "corpus"
    assert body["title"] == "Attribute Wrangle"
    assert body["build"] == BUILD
    assert "NO_SESSION" in body["note"]
    text = body["text"]
    assert text.startswith("Runs a VEX snippet to modify attribute values.")
    assert "This node runs the snippet on every point. See Point Wrangle." in text
    assert "Warning:\nThis node requires that you understand the vex language." in text
    assert "- Press MMB on the node to see errors." in text
    # Includes, whole and by id, with the id lines gone.
    assert "The VEX snippet parameter holds the code to run." in text
    assert "Run Over:\nApply the VEX code to each component of this type." in text
    assert "VEXpression:\nA snippet of VEX code that will manipulate the point attributes." in text
    assert "    @P.y += 1;" in text
    for markup in ("#type", "#id", "@parameters", ":include", "{{{", "__", "[Node:"):
        assert markup not in text
    assert body["trace"]["session_id"] is None


def test_markdown_keeps_headings_lists_and_code(lone: Bench) -> None:
    text = ok(docs(lone, path="nodes/sop/attribwrangle", format="markdown"))["text"]
    assert "## Overview" in text
    assert "## Parameters" in text
    assert "**Group**" in text
    assert "- Press MMB" in text
    assert "```vex\n@P.y += 1;\n```" in text
    assert "The **VEX snippet** parameter" in text
    plain = ok(docs(lone, path="nodes/sop/attribwrangle"))["text"]
    assert "##" not in plain and "```" not in plain and "**" not in plain


def test_a_table_keeps_each_row_on_one_line(lone: Bench) -> None:
    text = ok(docs(lone, path="nodes/sop/table"))["text"]
    assert "Name | Type\nP | vector\norient | quaternion of the shape" in text
    assert "After the table." in text


def test_a_vex_function_from_the_folder_pulls_in_its_shared_links(lone: Bench) -> None:
    body = ok(docs(lone, mode="vex", function="noise"))
    assert body["source"] == "corpus"
    text = body["text"]
    assert "float noise(vector pos)" in text
    assert "Note:\nThis function generates non-periodic noise." in text
    assert "- pnoise\n- snoise" in text


def test_paths_come_in_several_spellings(lone: Bench) -> None:
    for given in (
        "/nodes/sop/attribwrangle",
        "nodes/sop/attribwrangle.html",
        "Node:sop/attribwrangle",
        "nodes/sop/attribwrangle#parameters",
    ):
        assert ok(docs(lone, path=given))["path"] == "nodes/sop/attribwrangle"
    # A folder's page is its index, in a zip or not, however it is named.
    assert ok(docs(lone, path="vex"))["title"] == "VEX"
    assert ok(docs(lone, path="vex/index"))["path"] == "vex"
    assert ok(docs(lone, path="nodes/sop"))["title"] == "Geometry nodes"
    assert ok(docs(lone, path="copernicus/intro"))["title"] == "Copernicus intro"


def test_versioned_and_namespaced_node_types(lone: Bench) -> None:
    current = ok(docs(lone, path="nodes/sop/copytopoints::2.0"))
    assert current["path"] == "nodes/sop/copytopoints-2.0"
    assert current["version"] == "2.0"
    assert "Copies geometry onto points." in current["text"]
    plain = ok(docs(lone, path="Node:sop/copytopoints"))
    assert plain["version"] == "2.0"
    older = ok(docs(lone, path="nodes/sop/copytopoints-"))
    assert "The older way" in older["text"]
    assert older["version"] == "older"
    labs = ok(docs(lone, path="nodes/sop/labs::grid::1.1"))
    assert labs["title"] == "Labs Grid"
    assert labs["version"] == "1.1"
    failed(docs(lone, path="nodes/sop/copytopoints::3.0"), "DOC_NOT_FOUND")
    assert helpdocs.tidy_path("nodes/sop/kinefx::rigattribwrangle") == (
        "nodes/sop/kinefx--rigattribwrangle"
    )


def test_an_older_version_is_found_in_the_file_with_no_number(lone: Bench) -> None:
    # The older page says no version: it stands for any version below the
    # current page's.
    older = ok(docs(lone, path="nodes/sop/copytopoints::1.0"))
    assert "The older way" in older["text"]
    assert older["path"] == "nodes/sop/copytopoints-1.0"
    assert older["version"] == "1.0"
    # The older page says its version: only that one finds it.
    guide = ok(docs(lone, path="nodes/sop/guide::1.5"))
    assert "The guide before." in guide["text"]
    assert guide["version"] == "1.5"
    assert "The current guide." in ok(docs(lone, path="nodes/sop/guide::3.0"))["text"]
    failed(docs(lone, path="nodes/sop/guide::1.0"), "DOC_NOT_FOUND")
    failed(docs(lone, path="nodes/sop/box::1.0"), "DOC_NOT_FOUND")


def test_a_page_read_twice_comes_from_the_cache(
    lone: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[str] = []
    real = helpdocs.Corpus.read

    def counted(self: helpdocs.Corpus, path: str) -> str | None:
        reads.append(path)
        return real(self, path)

    monkeypatch.setattr(helpdocs.Corpus, "read", counted)
    first = ok(docs(lone, path="nodes/sop/attribwrangle"))
    assert reads
    reads.clear()
    second = ok(docs(lone, path="nodes/sop/attribwrangle"))
    assert reads == []
    assert second["text"] == first["text"]
    pages = lone.home / "docs" / BUILD / "pages"
    assert len(list(pages.glob("*.json"))) == 1
    # Each form is kept on its own.
    ok(docs(lone, path="nodes/sop/attribwrangle", format="markdown"))
    assert reads
    assert len(list(pages.glob("*.json"))) == 2


def test_the_cache_folders_are_private(lone: Bench) -> None:
    ok(docs(lone, path="vex"))
    if os.name != "nt":
        for folder in (lone.home / "docs", lone.home / "docs" / BUILD):
            assert folder.stat().st_mode & 0o077 == 0, folder


def test_an_edited_page_in_a_plain_folder_is_read_again(
    lone: Bench, hfs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpdocs, "FINGERPRINT_KEEP_S", 0.0)
    assert "Layers you can wrangle" in ok(docs(lone, path="copernicus/intro"))["text"]
    page = hfs / "houdini" / "help" / "copernicus" / "intro.txt"
    stat = page.stat()
    # The same size, a moment later.
    page.write_text("= Copernicus intro =\n\nLayers you can paint and more.\n", encoding="utf-8")
    os.utime(page, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert "Layers you can paint" in ok(docs(lone, path="copernicus/intro"))["text"]


def test_the_page_cache_keeps_to_its_size(tmp_path: Path) -> None:
    folder = tmp_path / "docs" / "b"
    for number in range(40):
        helpdocs.cache_put(folder, f"k{number:02d}", {"text": "x" * 1000}, cap=10_000)
    kept = list((folder / "pages").glob("*.json"))
    assert sum(path.stat().st_size for path in kept) <= 10_000
    assert kept
    # The newest are the ones left.
    assert (folder / "pages" / "k39.json").is_file()
    assert helpdocs.cache_get(folder, "k39") == {"text": "x" * 1000}
    assert helpdocs.cache_get(folder, "k00") is None


def test_the_builds_share_one_quota_and_the_least_used_go_first(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    for number, name in enumerate(("old", "middle", "now")):
        folder = helpdocs.docs_home(tmp_path, name)
        (folder / "pages").mkdir()
        (folder / "pages" / "a.json").write_text("x" * 4000, encoding="utf-8")
        os.utime(folder / helpdocs.BUILD_FILE_NAME, (1000 + number, 1000 + number))
    removed = helpdocs.keep_to_quota(root / "now", cap=9000)
    assert removed == ["old"]
    assert sorted(entry.name for entry in root.iterdir()) == ["middle", "now"]
    # The build in use stays whatever it weighs.
    assert helpdocs.keep_to_quota(root / "now", cap=10) == ["middle"]
    assert [entry.name for entry in root.iterdir()] == ["now"]


def test_the_folder_of_a_build_whose_install_went_is_removed(tmp_path: Path, hfs: Path) -> None:
    gone = tmp_path / "gone-install"
    helpdocs.docs_home(tmp_path, "9.9.9", gone)
    helpdocs.docs_home(tmp_path, BUILD, hfs)
    helpdocs.docs_home(tmp_path, "server-only")
    helpdocs.forget_sweeps()
    helpdocs.docs_home(tmp_path, BUILD, hfs)
    left = sorted(entry.name for entry in (tmp_path / "docs").iterdir())
    assert left == [BUILD, "server-only"]


def test_a_long_page_is_cut_and_spilled_whole(lone: Bench) -> None:
    whole = ok(docs(lone, path="nodes/sop/attribwrangle"))["text"]
    body = ok(docs(lone, path="nodes/sop/attribwrangle", max_chars=120))
    assert body["truncated"] is True
    assert len(body["text"]) <= 120
    assert whole.startswith(body["text"])
    assert body["total_chars"] == len(whole)
    spilled = json.loads(Path(body["spill_path"]).read_text(encoding="utf-8"))
    assert spilled["text"] == whole
    assert spilled["path"] == "nodes/sop/attribwrangle"


def test_a_spill_that_fails_is_an_error_not_a_lost_page(lone: Bench, tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("", encoding="utf-8")
    lone.config = replace(lone.config, spill_dir=blocked)
    failed(docs(lone, path="nodes/sop/attribwrangle", max_chars=120), "SPILL_FAILED")


def test_a_damaged_member_is_left_out_and_counted(lone: Bench, hfs: Path) -> None:
    archive = hfs / "houdini" / "help" / "nodes.zip"
    with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as opened:
        opened.writestr("sop/broken.txt", "= Broken =\n\n" + "words " * 4000)
    with zipfile.ZipFile(archive) as opened:
        info = opened.getinfo("sop/broken.txt")
    data = bytearray(archive.read_bytes())
    start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    data[start : start + 64] = b"\xff" * 64
    archive.write_bytes(bytes(data))
    body = ok(docs(lone, query="wrangle"))
    assert "1 damaged pages in the help folder were left out" in body["results"][0]["note"]
    failed(docs(lone, path="nodes/sop/broken"), "DOC_NOT_FOUND")
    assert ok(docs(lone, path="nodes/sop/box"))["title"] == "Box"


def test_a_page_file_over_the_cap_is_not_read(lone: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpdocs, "MAX_PAGE_BYTES", 100)
    failed(docs(lone, path="nodes/sop/attribwrangle"), "DOC_NOT_FOUND")
    assert ok(docs(lone, path="nodes/sop/box"))["title"] == "Box"


def test_includes_share_one_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = {f"a/p{number}": f"Part {number}.\n" for number in range(10)}
    top = "\n".join(f":include p{number}:" for number in range(10))
    monkeypatch.setattr(helptext, "MAX_INCLUDES", 3)
    text = helptext.markup_to_text(top, read=pages.get, where="a/top").text
    assert text.count("Part") == 3
    monkeypatch.setattr(helptext, "MAX_INCLUDES", 64)
    monkeypatch.setattr(helptext, "MAX_INCLUDED_CHARS", 20)
    text = helptext.markup_to_text(top, read=pages.get, where="a/top").text
    assert text.count("Part") == 2


# Section: search


def test_search_ranks_exact_then_prefix_then_substring_then_body(lone: Bench) -> None:
    body = ok(docs(lone, query="wrangle"))
    paths = [row["path"] for row in body["results"]]
    assert paths == [
        "nodes/cop/wrangle",
        "nodes/sop/wranglehelper",
        "nodes/sop/attribwrangle",
        # Deprecated, so after the current node of the same standing.
        "nodes/sop/pointwrangle",
        "copernicus/intro",
        # Release notes last, whatever their title.
        "news/20/wrangle",
    ]
    first = body["results"][0]
    assert first["title"] == "Wrangle"
    assert first["excerpt"] == "Runs a VEX snippet to modify layer values."
    assert first["source"] == "corpus"
    assert "index of" in first["note"] and "built now" in first["note"]
    assert "took" in first["note"]
    assert all("note" not in row for row in body["results"][1:])


def test_search_finds_a_node_by_its_internal_name(lone: Bench) -> None:
    body = ok(docs(lone, query="attribwrangle", limit=1))
    assert [row["path"] for row in body["results"]] == ["nodes/sop/attribwrangle"]


def test_search_prefers_the_current_version(lone: Bench) -> None:
    rows = ok(docs(lone, query="copy to points"))["results"]
    assert [row["path"] for row in rows] == ["nodes/sop/copytopoints", "nodes/sop/copytopoints-"]
    assert rows[0]["version"] == "2.0"
    # The older page says no version of its own, and still reads as older.
    assert rows[1]["version"] == "older"


def test_the_index_is_built_once_and_kept(lone: Bench) -> None:
    first = ok(docs(lone, query="noise"))
    assert "built now" in first["results"][0]["note"]
    # The same process: the copy kept in memory is not built now.
    again = ok(docs(lone, query="noise"))
    assert "read from the cache" in again["results"][0]["note"]
    kept = lone.home / "docs" / BUILD / "index.json"
    assert kept.is_file()
    helpdocs.forget_indexes()
    later = ok(docs(lone, query="noise"))
    assert "read from the cache" in later["results"][0]["note"]
    assert [row["path"] for row in later["results"]][:2] == [
        "vex/functions/noise",
        "vex/functions/pnoise",
    ]


def test_at_most_two_indexes_are_kept_in_memory(tmp_path: Path) -> None:
    for number in range(3):
        install = make_install(tmp_path / f"i{number}", build=f"22.0.{number}")
        helpdocs.load_index(tmp_path / "home", helpdocs.Corpus(install, f"22.0.{number}"))
    assert len(helpdocs._INDEXES) == helpdocs.MAX_KEPT_INDEXES


def test_search_takes_a_limit(lone: Bench) -> None:
    assert len(ok(docs(lone, query="wrangle", limit=2))["results"]) == 2


def test_a_search_that_finds_nothing_still_says_what_it_searched(lone: Bench) -> None:
    body = ok(docs(lone, query="zzzz"))
    assert body["results"] == []
    assert "index of" in body["note"]


# Section: the help server, for a build with no folder here


def test_a_page_comes_from_the_sessions_help_server(remote: Bench, served: HelpServer) -> None:
    body = ok(docs(remote, mode="page", path="nodes/sop/attribwrangle"))
    assert body["source"] == "help_server"
    assert body["title"] == "Attribute Wrangle"
    assert body["build"] == OTHER_BUILD
    assert body["truncated"] is False
    text = body["text"]
    assert text.startswith("Runs a VEX snippet to modify attribute values.")
    assert "Served overview text." in text
    assert "- Press MMB on the node." in text
    assert "Group:\nA subset of points." in text
    assert "Name | Type\nP | vector" in text
    assert "    @P.y += 1;" in text
    for furniture in (
        "Navigation bar",
        "script text",
        "Table of contents",
        "On this page",
        "Houdini 22.0",
    ):
        assert furniture not in text
    assert body["trace"]["session_id"] == "s-1"
    assert served.asked == ["/nodes/sop/attribwrangle"]


def test_a_vex_function_comes_from_the_help_server(remote: Bench) -> None:
    body = ok(docs(remote, mode="vex", function="noise", format="markdown"))
    assert body["source"] == "help_server"
    assert body["path"] == "vex/functions/noise"
    assert body["title"] == "noise"
    assert "`float noise(vector pos)`" in body["text"]
    assert "Served noise text." in body["text"]


def test_the_address_is_asked_for_once_and_without_queueing(remote: Bench, scene: Scene) -> None:
    for path in ("nodes/sop/attribwrangle", "vex/functions/noise", "nodes/dop/popwrangle"):
        assert ok(docs(remote, path=path))["source"] == "help_server"
    assert asked_bridge(remote) == 1
    assert scene.help_asked == 1
    [ask] = [call for call in remote.sent.calls if call["tool"] == "help.server"]  # type: ignore[attr-defined]
    assert ask["skip_if_busy"] is True
    assert ask["wait_s"] == 0
    assert ask["timeout_s"] == 2.0


def test_help_server_pages_are_cached_too(remote: Bench, served: HelpServer) -> None:
    first = ok(docs(remote, path="nodes/sop/attribwrangle"))
    second = ok(docs(remote, path="nodes/sop/attribwrangle"))
    assert second["text"] == first["text"]
    assert second["source"] == "help_server"
    assert pages_asked(served) == ["/nodes/sop/attribwrangle"]


def test_a_help_server_that_went_away_is_asked_for_again_once(remote: Bench, scene: Scene) -> None:
    scene.help_url = dead_url()
    error = failed(docs(remote, path="nodes/sop/attribwrangle"), "HELP_UNAVAILABLE")
    assert "did not take a connection" in " ".join(error["details"]["notes"])
    assert "left alone" not in " ".join(error["details"]["notes"])
    assert asked_bridge(remote) == 2


def test_a_new_address_after_a_failure_is_used(remote: Bench, scene: Scene) -> None:
    live = scene.help_url
    scene.help_url = dead_url()
    failed(docs(remote, path="nodes/sop/attribwrangle"), "HELP_UNAVAILABLE")
    # The session moved its help server; the kept address fails, so it is
    # asked again, and the new one answers.
    scene.help_url = live
    assert ok(docs(remote, path="nodes/sop/attribwrangle"))["source"] == "help_server"
    assert asked_bridge(remote) == 3


def test_a_slow_help_server_is_given_up_on_and_left_alone(
    remote: Bench, served: HelpServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpdocs, "HELP_TIMEOUT_S", 0.3)
    served.delay = 3.0
    started = time.monotonic()
    error = failed(docs(remote, path="nodes/sop/attribwrangle"), "HELP_UNAVAILABLE")
    assert time.monotonic() - started < 1.5
    assert "left alone for 60 s" in " ".join(error["details"]["notes"])
    # A timeout is not a reason to ask for the address again.
    assert asked_bridge(remote) == 1
    # Within the minute, the help server is not asked at all.
    served.delay = 0.0
    error = failed(docs(remote, path="vex/functions/noise"), "HELP_UNAVAILABLE")
    assert "left alone for" in " ".join(error["details"]["notes"])
    assert served.asked == ["/nodes/sop/attribwrangle"]


def test_a_help_server_that_takes_the_connection_and_never_answers_is_left_alone(
    remote: Bench, scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpdocs, "HELP_TIMEOUT_S", 0.3)
    silent = Silent()
    try:
        scene.help_url = silent.url
        started = time.monotonic()
        error = failed(docs(remote, path="nodes/sop/attribwrangle"), "HELP_UNAVAILABLE")
        assert time.monotonic() - started < 1.5
        assert "left alone for 60 s" in " ".join(error["details"]["notes"])
        # It is there, only busy: the address is not asked for again.
        assert asked_bridge(remote) == 1
        assert helpdocs.URLS.quiet_for("s-1") > 0
    finally:
        silent.close()


def test_the_two_failures_are_told_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpdocs, "HELP_TIMEOUT_S", 0.3)
    with pytest.raises(helpdocs.HelpServerError) as gone:
        helpdocs.fetch(dead_url(), "nodes/sop/attribwrangle")
    assert gone.value.timed_out is False
    silent = Silent()
    try:
        with pytest.raises(helpdocs.HelpServerError) as busy_one:
            helpdocs.fetch(silent.url, "nodes/sop/attribwrangle")
        assert busy_one.value.timed_out is True
    finally:
        silent.close()


def test_a_connect_that_does_not_complete_is_a_help_server_that_went(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A system that lets a connection to a closed port hang rather than
    # refuse it: the connect is what does not finish.
    def hangs(self: Any) -> None:
        time.sleep(1.0)

    monkeypatch.setattr(helpdocs.http.client.HTTPConnection, "connect", hangs)
    monkeypatch.setattr(helpdocs, "HELP_TIMEOUT_S", 0.3)
    started = time.monotonic()
    with pytest.raises(helpdocs.HelpServerError) as raised:
        helpdocs.fetch("http://127.0.0.1:9/", "nodes/sop/attribwrangle")
    assert raised.value.timed_out is False
    assert time.monotonic() - started < 0.9


def test_a_busy_session_is_not_waited_for(remote: Bench, scene: Scene) -> None:
    with busy(remote, scene):
        started = time.monotonic()
        error = failed(docs(remote, path="nodes/sop/attribwrangle"), "HELP_UNAVAILABLE")
        took = time.monotonic() - started
    assert took < 0.5
    assert "SESSION_BUSY" in " ".join(error["details"]["notes"])
    # Not kept: the next call asks again, and the session is free by then.
    assert helpdocs.URLS.get("s-1") == (False, None)
    assert ok(docs(remote, path="nodes/sop/attribwrangle"))["source"] == "help_server"


def test_a_failure_and_a_busy_refresh_are_both_named(remote: Bench, scene: Scene) -> None:
    ok(docs(remote, path="vex/functions/noise"))
    scene.help_url = dead_url()
    helpdocs.URLS.put("s-1", dead_url())
    with busy(remote, scene):
        error = failed(docs(remote, path="nodes/sop/attribwrangle"), "HELP_UNAVAILABLE")
    [note] = error["details"]["notes"][-1:]
    assert "the help server did not take a connection" in note
    assert "SESSION_BUSY" in note


def test_a_redirect_is_refused(remote: Bench, served: HelpServer) -> None:
    other = HelpServer()
    try:
        served.redirects["/nodes/sop/attribwrangle"] = other.url + "nodes/sop/attribwrangle"
        error = failed(docs(remote, path="nodes/sop/attribwrangle"), "HELP_UNAVAILABLE")
        assert "redirected" in " ".join(error["details"]["notes"])
        with pytest.raises(helpdocs.HelpServerError):
            helpdocs.fetch(served.url, "nodes/sop/attribwrangle")
        assert other.asked == []
    finally:
        other.close()


def test_a_slow_body_is_cut_off_at_the_limit(
    served: HelpServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpdocs, "HELP_TIMEOUT_S", 0.5)
    # Sixteen bytes every tenth of a second: the whole page would take minutes.
    served.drip = 0.1
    started = time.monotonic()
    with pytest.raises(helpdocs.HelpServerError) as raised:
        helpdocs.fetch(served.url, "nodes/sop/attribwrangle")
    assert raised.value.timed_out
    assert time.monotonic() - started < 1.0


def test_slow_headers_are_cut_off_at_the_limit(
    served: HelpServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpdocs, "HELP_TIMEOUT_S", 0.5)
    served.delay = 3.0
    started = time.monotonic()
    with pytest.raises(helpdocs.HelpServerError) as raised:
        helpdocs.fetch(served.url, "nodes/sop/attribwrangle")
    assert raised.value.timed_out
    assert time.monotonic() - started < 1.0


def test_help_served_from_elsewhere_is_not_fetched(
    remote: Bench, scene: Scene, served: HelpServer
) -> None:
    scene.help_url = "https://www.example.com/docs/houdini/"
    error = failed(docs(remote, path="nodes/sop/attribwrangle"), "HELP_UNAVAILABLE")
    assert "no help on this machine's loopback" in " ".join(error["details"]["notes"])
    assert served.asked == []


def test_search_with_only_a_help_server(remote: Bench, served: HelpServer) -> None:
    body = ok(docs(remote, query="wrangle", limit=10))
    paths = [row["path"] for row in body["results"]]
    assert paths == [
        "nodes/cop/wrangle",
        "nodes/sop/attribwrangle",
        "nodes/dop/popwrangle",
        "nodes/sop/snippetsop",
        "news/20/wrangle",
    ]
    assert {row["source"] for row in body["results"]} == {"help_server"}
    assert body["results"][1]["title"] == "Attribute Wrangle"
    assert "help server's own search" in body["results"][0]["note"]
    assert body["build"] == OTHER_BUILD


def test_a_page_neither_route_has_is_not_found(matched: Bench) -> None:
    failed(docs(matched, path="nodes/sop/nothere"), "DOC_NOT_FOUND")


# Section: refusals


def test_no_help_anywhere_is_help_unavailable(tmp_path: Path) -> None:
    bare = make_install(tmp_path / "bare", with_help=False)
    lone = bench_for(tmp_path, bare / "bin" / "hython")
    result = docs(lone, path="nodes/sop/attribwrangle")
    error = failed(result, "HELP_UNAVAILABLE")
    assert "houdini_build" in error["hint"]
    text = text_of(result)
    assert text.startswith("HELP_UNAVAILABLE: ")
    assert "hint: " in text
    failed(docs(lone, query="wrangle"), "HELP_UNAVAILABLE")


def test_a_page_that_is_not_there_names_the_nearest(lone: Bench) -> None:
    error = failed(docs(lone, path="nodes/sop/attribwrangl"), "DOC_NOT_FOUND")
    assert error["details"]["did_you_mean"][0] == "nodes/sop/attribwrangle"
    assert "hint" in error
    error = failed(docs(lone, mode="vex", function="nois"), "DOC_NOT_FOUND")
    assert "noise" in error["details"]["did_you_mean"]


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        ({"mode": "search"}, "query"),
        ({"mode": "page"}, "path"),
        ({"mode": "vex"}, "function"),
        ({}, "mode"),
        ({"mode": "vex", "function": "no-ise"}, "function"),
        ({"path": "../../etc/passwd"}, "path"),
        ({"path": "nodes/../../secret"}, "path"),
        ({"path": "C:\\\\help\\\\x"}, "path"),
        ({"path": "nodes/sop:x/box"}, "path"),
        ({"query": "x", "limit": 0}, "limit"),
        ({"query": "x", "limit": 51}, "limit"),
        ({"path": "vex", "max_chars": 5}, "max_chars"),
    ],
)
def test_arguments_are_checked_before_anything_is_read(
    lone: Bench, arguments: dict[str, Any], argument: str
) -> None:
    error = failed(docs(lone, **arguments), "BAD_ARGUMENTS")
    assert error["details"]["argument"] == argument


def test_a_named_session_that_is_not_there_is_an_error(lone: Bench) -> None:
    failed(docs(lone, path="vex", session="nobody"), "SESSION_UNKNOWN")


# Section: the pieces


class busy:  # noqa: N801 - reads as the state it holds the session in
    """Hold the session with a real call on its dispatcher, as a long cook does."""

    def __init__(self, bench: Bench, scene: Scene) -> None:
        through = bench.sent
        self.dispatcher = Dispatcher(
            default_registry(selfcheck=True),
            lock=threading.Lock(),
            kind="hython",
            session_id="s-1",
            identity=through.identity,  # type: ignore[attr-defined]
            hou=scene.module(),
            wait_s=5.0,
            timeout_s=10.0,
        )
        through.dispatcher = self.dispatcher  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        self.dispatcher.dispatch(
            Envelope(
                tool="bridge.selfcheck",
                arguments={"sleep_s": 30.0},
                session_id="s-1",
                timeout_s=60.0,
            )
        )

    def __enter__(self) -> busy:
        self.thread.start()
        deadline = time.monotonic() + 5.0
        while self.dispatcher.running is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert self.dispatcher.running is not None
        return self

    def __exit__(self, *exc: Any) -> None:
        self.dispatcher.dispatch(
            Envelope(
                tool="bridge.cancel",
                arguments={"operation_id": self.dispatcher.running.operation_id}
                if self.dispatcher.running
                else {},
                session_id="s-1",
            )
        )
        self.thread.join(15.0)


def test_the_bridge_says_where_its_help_is(scene: Scene) -> None:
    scene.help_url = "http://127.0.0.1:48000/"
    said = tools.help_server({}, ToolContext(hou=scene.module()))
    assert said["url"] == "http://127.0.0.1:48000/"
    assert said["houdini_version"] == "22.0.368"
    scene.help_url = None
    assert tools.help_server({}, ToolContext(hou=scene.module()))["url"] is None


def test_only_a_loopback_http_address_is_used() -> None:
    assert helpdocs.usable_url("http://127.0.0.1:4000/") == "http://127.0.0.1:4000/"
    assert helpdocs.usable_url("http://localhost:4000") == "http://localhost:4000/"
    assert helpdocs.usable_url("http://[::1]:4000/") == "http://[::1]:4000/"
    assert helpdocs.usable_url("https://127.0.0.1:4000/") is None
    assert helpdocs.usable_url("http://10.0.0.2:4000/") is None
    assert helpdocs.usable_url("http://127.0.0.1.example.com/") is None
    assert helpdocs.usable_url("http://user:secret@127.0.0.1:4000/") is None
    assert helpdocs.usable_url("http://user@127.0.0.1:4000/") is None
    assert helpdocs.usable_url("http://127.0.0.1:port/") is None
    assert helpdocs.usable_url("") is None
    assert helpdocs.usable_url(None) is None


def test_ranks() -> None:
    assert helpdocs.rank("box", "Box", "nodes/sop/box", "") == 0
    assert helpdocs.rank("attribwrangle", "Attribute Wrangle", "nodes/sop/attribwrangle", "") == 0
    assert helpdocs.rank("copytopoints", "Copy", "nodes/sop/copytopoints-2.0", "") == 0
    assert helpdocs.rank("bo", "Box", "nodes/sop/box", "") == 1
    assert helpdocs.rank("wrangle", "Attribute Wrangle", "nodes/sop/attribwrangle", "") == 2
    assert helpdocs.rank("snippet vex", "Wrangle", "nodes/cop/wrangle", "Runs a VEX snippet") == 3
    assert helpdocs.rank("fluid", "Box", "nodes/sop/box", "Creates a cube") is None
    # Outside nodes, functions and classes a path is not a title.
    assert helpdocs.rank("intro", "Getting going", "copernicus/intro", "") == 3


def test_order_puts_news_last_and_old_versions_after_current() -> None:
    rows = [
        (0, "news/20/vellum", "Vellum"),
        (2, "nodes/sop/vellumsolver", "Vellum Solver"),
        (0, "vellum", "Vellum"),
        (0, "nodes/sop/copytopoints-", "Copy to Points"),
        (0, "nodes/sop/copytopoints", "Copy to Points"),
    ]
    ordered = [row[1] for row in sorted(rows, key=lambda row: helpdocs.order(*row))]
    assert ordered == [
        "nodes/sop/copytopoints",
        "vellum",
        "nodes/sop/copytopoints-",
        "nodes/sop/vellumsolver",
        "news/20/vellum",
    ]


def test_a_block_is_found_by_its_id() -> None:
    source = PAGES["nodes.zip"]["sop/pointwrangle.txt"]
    assert helptext.block_of(source, "snippet", inner=True) == (
        "A snippet of VEX code that will manipulate the point attributes."
    )
    whole = helptext.block_of(source, "snippet")
    assert whole is not None and whole.startswith("VEXpression:\n")
    assert helptext.block_of(source, "nothing") is None


def test_a_summary_after_a_warning_is_still_the_summary() -> None:
    head = helptext.markup_head(PAGES["nodes.zip"]["sop/pointwrangle.txt"])
    assert head.summary == "Runs a VEX snippet to modify point attributes."
    assert head.deprecated is True


def test_includes_stop_when_they_go_round() -> None:
    pages = {
        "a/one": "= One =\n\nFirst.\n\n:include two:\n",
        "a/two": "Second.\n\n:include one:\n",
    }
    page = helptext.markup_to_text(pages["a/one"], read=pages.get, where="a/one")
    assert page.text.count("First.") == 1
    assert page.text.count("Second.") == 1


def test_help_server_search_hits_are_read_from_its_page() -> None:
    hits = helptext.search_hits(SEARCH_HTML, helpdocs.tidy_path)
    assert [hit["path"] for hit in hits] == [
        "nodes/cop/wrangle",
        "nodes/sop/attribwrangle",
        "nodes/dop/popwrangle",
        "news/20/wrangle",
        "nodes/sop/snippetsop",
    ]
    assert hits[0]["excerpt"] == "Runs a VEX snippet to modify layer values."
    # The title a line break ran through is one line.
    assert hits[1] == {
        "path": "nodes/sop/attribwrangle",
        "title": "Attribute Wrangle",
        "excerpt": "geometry node",
    }


def test_a_folder_page_named_two_ways_is_one_hit() -> None:
    page = (
        '<div class="hit"><a class="label" href="/vellum/index.html">Vellum</a></div>'
        '<div class="hit"><a class="label" href="/vellum">Vellum</a></div>'
        '<div class="hit"><a class="label" href="/vellum/index">Vellum</a></div>'
    )
    assert [hit["path"] for hit in helptext.search_hits(page, helpdocs.tidy_path)] == ["vellum"]


def test_a_search_page_gives_at_most_a_few_hundred_hits() -> None:
    page = "".join(
        f'<div class="hit"><a class="label" href="/nodes/sop/n{number}">N{number}</a></div>'
        for number in range(1000)
    )
    assert len(helptext.search_hits(page, helpdocs.tidy_path)) == helptext.MAX_HITS


def test_a_deep_or_long_page_stays_within_its_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    deep = "<main>" + "<div>" * 5000 + "deep text" + "</div>" * 5000 + "</main>"
    started = time.monotonic()
    title, text = helptext.html_to_text(deep)
    assert time.monotonic() - started < 2.0
    assert "deep text" in text
    monkeypatch.setattr(helptext, "MAX_HTML_TOKENS", 50)
    _, text = helptext.html_to_text("<main>" + "<p>word</p>" * 1000 + "</main>")
    assert text.count("word") < 50


def test_pictures_and_clips_on_a_page_leave_no_text() -> None:
    assert helptext.inline("[Image:/images/shelf/copy.jpg] Copy", markdown=False) == "Copy"
    assert helptext.inline("See [Anim:/anim/copy.mp4].", markdown=False) == "See ."
    assert helptext.inline("A [Node:sop/box] and [/vex/random]", markdown=False) == (
        "A sop/box and /vex/random"
    )
