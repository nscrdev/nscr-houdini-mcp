"""`hou_docs` through the server, against a stand in help server and help folder.

The help folder is a tiny tree made here in the layout a real install ships:
one zip per book holding help markup, a book kept as a plain folder, and a
version header. The help server is a small HTTP server of the test's own,
serving HTML shaped like the real one's and a search page. The session behind
it is a real dispatcher over the stand in `hou`, whose `helpServerUrl` points
at that server, so the address comes the way it does in use: one bridge call,
kept per session.
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
from nscr_houdini_mcp.bridge import client, tools
from nscr_houdini_mcp.bridge.tools import ToolContext
from nscr_houdini_mcp.tools.registry import TOOLS
from test_server import talk, text_of
from test_tools_inspect import Through
from test_tools_sessions import Bench

BUILD = "22.0.999"

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
}

FOLDER_PAGES = {
    "copernicus/intro.txt": """= Copernicus intro =

Layers you can wrangle and more.
""",
}


def make_install(root: Path, *, with_help: bool = True) -> Path:
    """An install folder with a hython in it and, unless told, its help."""
    hfs = root / "hfs"
    (hfs / "bin").mkdir(parents=True)
    (hfs / "bin" / "hython").write_text("", encoding="utf-8")
    header = hfs / "toolkit" / "include" / "SYS"
    header.mkdir(parents=True)
    (header / "SYS_Version.h").write_text(f'#define SYS_VERSION_FULL "{BUILD}"\n', encoding="utf-8")
    if not with_help:
        return hfs
    help_root = hfs / "houdini" / "help"
    help_root.mkdir(parents=True)
    for book, members in PAGES.items():
        with zipfile.ZipFile(help_root / book, "w") as archive:
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
<h1 class="title">Attribute Wrangle <span class="subtitle">geometry node</span></h1>
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
<div class="hit"><p class="label"><a class="label" href="/nodes/sop/attribwrangle">
Attribute Wrangle</a><small class="desc">geometry node</small></p></div>
<div class="hit"><p class="label"><a class="label" href="/nodes/dop/popwrangle">
POP Wrangle</a><small class="desc">dynamics node</small></p></div>
<div class="hit"><p class="label"><a class="label" href="/nodes/sop/snippetsop">
Snippet SOP</a><small class="desc">geometry node</small></p></div>
<div class="more hit"><p class="label">11 more in Geometry nodes</p></div>
</div></section></div></div>
"""


class HelpServer:
    """A help server of the test's own, on loopback."""

    def __init__(self) -> None:
        self.pages = {
            "/nodes/sop/attribwrangle": PAGE_HTML,
            "/vex/functions/noise": NOISE_HTML,
        }
        self.delay = 0.0
        self.asked: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the name is the standard library's
                owner.asked.append(self.path)
                if owner.delay:
                    time.sleep(owner.delay)
                where = urlsplit(self.path)
                if where.path == "/_search":
                    body = SEARCH_HTML if parse_qs(where.query).get("q") else ""
                else:
                    body = owner.pages.get(where.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = body.encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
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
    """An address on loopback where nothing listens."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/"


# Section: fixtures


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Nothing from another test, and no Houdini of this machine's own."""
    monkeypatch.setattr(install_module, "find_installs", lambda configured=None: [])
    monkeypatch.delenv(pool.HYTHON_ENV_VAR, raising=False)
    monkeypatch.delenv("HFS", raising=False)
    helpdocs.URLS.clear()
    helpdocs.forget_indexes()
    yield
    helpdocs.URLS.clear()
    helpdocs.forget_indexes()


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
def bench(tmp_path: Path, hfs: Path, scene: Scene, served: HelpServer) -> Bench:
    """One live session on the install, whose help server is the stand in."""
    made = bench_for(tmp_path, hfs / "bin" / "hython")
    add_session(made, hfs)
    scene.help_url = served.url
    made.sent = Through(scene)  # type: ignore[assignment]
    return made


def add_session(bench: Bench, hfs: Path | None, session_id: str = "s-1", alias: str = "w1") -> None:
    facts: dict[str, Any] = {"houdini_version": BUILD}
    if hfs is not None:
        facts["hfs"] = str(hfs)
    with bench.store() as store:
        store.register_session(
            session_id,
            kind="hython",
            pid=os.getpid(),
            pid_start=bench.stamp,
            alias=alias,
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


# Section: the help server first


def test_a_page_comes_from_the_sessions_help_server(bench: Bench, served: HelpServer) -> None:
    body = ok(docs(bench, mode="page", path="nodes/sop/attribwrangle"))
    assert body["source"] == "help_server"
    assert body["title"] == "Attribute Wrangle"
    assert body["build"] == BUILD
    assert body["truncated"] is False
    text = body["text"]
    assert text.startswith("Runs a VEX snippet to modify attribute values.")
    assert "Served overview text." in text
    assert "- Press MMB on the node." in text
    assert "Group:\nA subset of points." in text
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


def test_a_vex_function_comes_from_the_help_server(bench: Bench) -> None:
    body = ok(docs(bench, mode="vex", function="noise", format="markdown"))
    assert body["source"] == "help_server"
    assert body["path"] == "vex/functions/noise"
    assert body["title"] == "noise"
    assert "`float noise(vector pos)`" in body["text"]
    assert "Served noise text." in body["text"]


def test_the_address_is_asked_for_once_per_session(bench: Bench, scene: Scene) -> None:
    for _ in range(3):
        assert ok(docs(bench, path="nodes/sop/attribwrangle"))["source"] == "help_server"
    assert asked_bridge(bench) == 1
    assert scene.help_asked == 1


def test_a_help_server_that_went_away_is_asked_for_again_once_then_the_folder_is_read(
    bench: Bench, scene: Scene
) -> None:
    scene.help_url = dead_url()
    body = ok(docs(bench, path="nodes/sop/attribwrangle"))
    assert body["source"] == "corpus"
    assert "help server did not answer" in body["note"]
    assert asked_bridge(bench) == 2


def test_a_new_address_after_a_failure_is_used(
    bench: Bench, scene: Scene, served: HelpServer
) -> None:
    live = scene.help_url
    scene.help_url = dead_url()
    assert ok(docs(bench, path="nodes/sop/attribwrangle"))["source"] == "corpus"
    # The session moved its help server; the kept address fails, so it is
    # asked again, and the new one answers.
    scene.help_url = live
    body = ok(docs(bench, path="nodes/sop/attribwrangle"))
    assert body["source"] == "help_server"
    assert asked_bridge(bench) == 3


def test_a_slow_help_server_is_given_up_on_within_the_timeout(
    bench: Bench, served: HelpServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpdocs, "HELP_TIMEOUT_S", 0.3)
    served.delay = 3.0
    started = time.monotonic()
    body = ok(docs(bench, path="nodes/sop/attribwrangle"))
    took = time.monotonic() - started
    assert body["source"] == "corpus"
    # Two tries of 0.3 s each, well short of the server's 3 s.
    assert took < 2.0
    assert len(served.asked) == 2


def test_a_page_the_help_server_does_not_have_is_read_from_the_folder(
    bench: Bench, served: HelpServer
) -> None:
    body = ok(docs(bench, path="nodes/sop/pointwrangle"))
    assert body["source"] == "corpus"
    assert body["title"] == "Point Wrangle"
    # A missing page is an answer, not a failure: the address is not asked again.
    assert asked_bridge(bench) == 1


def test_a_busy_session_is_not_waited_for(bench: Bench) -> None:
    inner = bench.sent

    def busy(session: client.Session, tool: str, **rest: Any) -> client.Answer:
        if tool == "help.server":
            assert rest["wait_s"] == 0
            payload = {"ok": False, "error": {"code": "SESSION_BUSY", "message": "busy"}}
            return client.Answer(200, payload, {})
        return inner(session, tool, **rest)

    bench.sent = busy  # type: ignore[assignment]
    body = ok(docs(bench, path="nodes/sop/attribwrangle"))
    assert body["source"] == "corpus"
    assert "SESSION_BUSY" in body["note"]
    # Not kept: the next call asks again.
    assert helpdocs.URLS.get("s-1") == (False, None)


def test_help_served_from_elsewhere_is_left_for_the_folder(
    bench: Bench, scene: Scene, served: HelpServer
) -> None:
    scene.help_url = "https://www.example.com/docs/houdini/"
    body = ok(docs(bench, path="nodes/sop/attribwrangle"))
    assert body["source"] == "corpus"
    assert served.asked == []


# Section: the help folder


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
    # A folder's page is its index, in a zip or not.
    assert ok(docs(lone, path="vex"))["title"] == "VEX"
    assert ok(docs(lone, path="nodes/sop"))["title"] == "Geometry nodes"
    assert ok(docs(lone, path="copernicus/intro"))["title"] == "Copernicus intro"


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
    kept = list((lone.home / "docs" / BUILD / "pages").glob("*.json"))
    assert len(kept) == 1
    # Each form is kept on its own.
    ok(docs(lone, path="nodes/sop/attribwrangle", format="markdown"))
    assert reads
    assert len(list((lone.home / "docs" / BUILD / "pages").glob("*.json"))) == 2


def test_the_page_cache_keeps_to_its_size(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    for number in range(40):
        helpdocs.cache_put(folder, f"k{number:02d}", {"text": "x" * 1000}, cap=10_000)
    kept = list((folder / "pages").glob("*.json"))
    assert sum(path.stat().st_size for path in kept) <= 10_000
    assert kept
    # The newest are the ones left.
    assert (folder / "pages" / "k39.json").is_file()
    assert helpdocs.cache_get(folder, "k39") == {"text": "x" * 1000}
    assert helpdocs.cache_get(folder, "k00") is None


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


# Section: search


def test_search_ranks_exact_then_prefix_then_substring_then_body(lone: Bench) -> None:
    body = ok(docs(lone, query="wrangle"))
    paths = [row["path"] for row in body["results"]]
    assert paths == [
        "nodes/cop/wrangle",
        "nodes/sop/wranglehelper",
        "nodes/sop/pointwrangle",
        "nodes/sop/attribwrangle",
        "copernicus/intro",
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


def test_the_index_is_built_once_and_kept(lone: Bench) -> None:
    ok(docs(lone, query="noise"))
    kept = lone.home / "docs" / BUILD / "index.json"
    assert kept.is_file()
    helpdocs.forget_indexes()
    again = ok(docs(lone, query="noise"))
    assert "read from the cache" in again["results"][0]["note"]
    assert [row["path"] for row in again["results"]][:2] == [
        "vex/functions/noise",
        "vex/functions/pnoise",
    ]


def test_search_puts_the_help_servers_hits_with_the_folders(
    bench: Bench, served: HelpServer
) -> None:
    body = ok(docs(bench, query="wrangle", limit=10))
    rows = {row["path"]: row for row in body["results"]}
    assert [row["path"] for row in body["results"]][0] == "nodes/cop/wrangle"
    assert rows["nodes/dop/popwrangle"]["source"] == "help_server"
    assert rows["nodes/sop/attribwrangle"]["source"] == "help_server"
    assert rows["nodes/sop/pointwrangle"]["source"] == "corpus"
    # A hit the server found in the body ranks after every title match.
    assert list(rows).index("nodes/sop/snippetsop") > list(rows).index("nodes/sop/attribwrangle")
    assert "find" not in " ".join(rows)
    assert [path for path in served.asked if path.startswith("/_search")] == ["/_search?q=wrangle"]


def test_search_takes_a_limit(lone: Bench) -> None:
    assert len(ok(docs(lone, query="wrangle", limit=2))["results"]) == 2


def test_a_search_that_finds_nothing_still_says_what_it_searched(lone: Bench) -> None:
    body = ok(docs(lone, query="zzzz"))
    assert body["results"] == []
    assert "index of" in body["note"]


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


def test_a_page_neither_route_has_is_not_found(bench: Bench) -> None:
    failed(docs(bench, path="nodes/sop/nothere"), "DOC_NOT_FOUND")


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
    assert helpdocs.usable_url("https://127.0.0.1:4000/") is None
    assert helpdocs.usable_url("http://10.0.0.2:4000/") is None
    assert helpdocs.usable_url("http://127.0.0.1.example.com/") is None
    assert helpdocs.usable_url("") is None
    assert helpdocs.usable_url(None) is None


def test_ranks() -> None:
    assert helpdocs.rank("box", "Box", "nodes/sop/box", "") == 0
    assert helpdocs.rank("attribwrangle", "Attribute Wrangle", "nodes/sop/attribwrangle", "") == 0
    assert helpdocs.rank("bo", "Box", "nodes/sop/box", "") == 1
    assert helpdocs.rank("wrangle", "Attribute Wrangle", "nodes/sop/attribwrangle", "") == 2
    assert helpdocs.rank("snippet vex", "Wrangle", "nodes/cop/wrangle", "Runs a VEX snippet") == 3
    assert helpdocs.rank("fluid", "Box", "nodes/sop/box", "Creates a cube") is None


def test_a_block_is_found_by_its_id() -> None:
    source = PAGES["nodes.zip"]["sop/pointwrangle.txt"]
    assert helptext.block_of(source, "snippet", inner=True) == (
        "A snippet of VEX code that will manipulate the point attributes."
    )
    whole = helptext.block_of(source, "snippet")
    assert whole is not None and whole.startswith("VEXpression:\n")
    assert helptext.block_of(source, "nothing") is None


def test_includes_stop_when_they_go_round() -> None:
    pages = {
        "a/one": "= One =\n\nFirst.\n\n:include two:\n",
        "a/two": "Second.\n\n:include one:\n",
    }
    page = helptext.markup_to_text(pages["a/one"], read=pages.get, where="a/one")
    assert page.text.count("First.") == 1
    assert page.text.count("Second.") == 1


def test_help_server_search_hits_are_read_from_its_page() -> None:
    hits = helptext.search_hits(SEARCH_HTML)
    assert [hit["path"] for hit in hits] == [
        "nodes/cop/wrangle",
        "nodes/sop/attribwrangle",
        "nodes/dop/popwrangle",
        "nodes/sop/snippetsop",
    ]
    assert hits[0]["excerpt"] == "Runs a VEX snippet to modify layer values."
    assert hits[1] == {
        "path": "nodes/sop/attribwrangle",
        "title": "Attribute Wrangle",
        "excerpt": "geometry node",
    }


def test_pictures_and_clips_on_a_page_leave_no_text() -> None:
    assert helptext.inline("[Image:/images/shelf/copy.jpg] Copy", markdown=False) == "Copy"
    assert helptext.inline("See [Anim:/anim/copy.mp4].", markdown=False) == "See ."
    assert helptext.inline("A [Node:sop/box] and [/vex/random]", markdown=False) == (
        "A sop/box and /vex/random"
    )
