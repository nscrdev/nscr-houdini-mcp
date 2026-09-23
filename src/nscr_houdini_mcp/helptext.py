"""Readable text from Houdini's help, in either of the forms it comes in.

The installed help is plain text markup, one `.txt` file a page, and the help server
Houdini runs turns the same pages into HTML. Both come out here as the text a
person reads: the title on its own, then the summary, the headings, the
parameter names with what each one does, lists, notes and code. Navigation,
icons, page furniture and property lines go.

Two forms of text:

- `plain`, the default: headings and parameter names as lines of their own,
  list items as `- `, code indented by four spaces, no other markup.
- `markdown`: headings as `#` lines, parameter names in bold, code fenced
  with its language, inline code in backticks.

A page can pull part of another page in with `:include path#id:`. The
caller hands in a function that reads another page's markup, and the included
part is rendered where it stands, a few levels deep at most.

This module never imports `hou` and reads no files of its own.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser

FORMATS = ("plain", "markdown")

# How deep one include may pull in another.
MAX_INCLUDE_DEPTH = 4

# What all the includes of one page may pull in together, by count and by
# the size of the markup read.
MAX_INCLUDES = 64
MAX_INCLUDED_CHARS = 2 * 1024 * 1024

# What one help server page may make the reader do: how deep its tags may
# nest before deeper ones are read as plain text, how many tags and pieces
# of text it looks at, and how much text it keeps.
MAX_HTML_DEPTH = 200
MAX_HTML_TOKENS = 400_000
MAX_HTML_TEXT = 4 * 1024 * 1024

# How many hits a search page gives, and how long any one field of a hit is.
MAX_HITS = 300
MAX_HIT_FIELD = 1000

# The longest excerpt a search result carries.
EXCERPT_CHARS = 240

# What an `@name` line opens, in the words a page shows for it.
SECTIONS = {
    "parameters": "Parameters",
    "related": "Related",
    "inputs": "Inputs",
    "outputs": "Outputs",
    "locals": "Local variables",
    "examples": "Examples",
    "subtopics": "Subtopics",
    "usage": "Usage",
    "attributes": "Attributes",
    "methods": "Methods",
    "functions": "Functions",
}

NOTICES = ("NOTE", "TIP", "WARNING", "IMPORTANT", "DEPRECATED", "CAUTION", "TODO")

_TITLE = re.compile(r"^=\s*(.+?)\s*=\s*$")
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*(?:\([\w./-]*\))?\s*$")
_PROPERTY = re.compile(r"^#[A-Za-z_][\w-]*\s*:")
_SECTION = re.compile(r"^@([A-Za-z_]\w*)\s*$")
_INCLUDE = re.compile(r"^:include\s+(.+?)\s*:\s*$")
_USAGE = re.compile(r"^:usage:\s*(.*)$")
_DIRECTIVE = re.compile(r"^:([\w-]*):\s*(.*)$")
_NOTICE = re.compile(r"^(" + "|".join(NOTICES) + r"):\s*(.*)$")
_TABLE = re.compile(r"^(table|tr|td|th|thead|tbody)\b[^>]*>>\s*(.*)$")
_TAG_LINE = re.compile(r"^</?[A-Za-z][^>]*>$")
_BULLET = re.compile(r"^(?:[*-]|#|::)\s+(.*)$")
_LABEL = re.compile(r"^(.+?):\s*$")
_ID = re.compile(r"^\s*#id\s*:\s*(\S+)\s*$")
_DEPRECATED = re.compile(r"^:\w+:\s*deprecated\b", re.IGNORECASE)

_CODE_SPAN = re.compile(r"(`[^`\n]*`)")
_LINK = re.compile(r"\[([^\[\]]+)\]")
_BOLD = re.compile(r"__(.+?)__")
_KEYS = re.compile(r"\(\((.+?)\)\)")
_ITALIC_QUOTES = re.compile(r"''(.+?)''")
_ITALIC = re.compile(r"(?<![\w/\\])_([^_\s][^_\n]*?)_(?![\w])")
_STRONG = re.compile(r"(?<![\w*])\*([^*\s][^*\n]*?)\*(?![\w*])")
_PLACEHOLDER = re.compile(r"<<(.+?)>>")
_MEDIA = re.compile(r"\.(?:png|jpe?g|gif|svg|webp|mp4|webm|mov|m4v)$", re.IGNORECASE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BLANKS = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\r\f\v\xa0]+")


def check_format(value: str | None) -> str:
    return value if value in FORMATS else "plain"


# Section: help markup


@dataclass
class MarkupPage:
    """One page read from its markup."""

    title: str | None = None
    summary: str | None = None
    properties: dict[str, str] = field(default_factory=dict)
    # Whether the page says at its top that what it describes is deprecated.
    deprecated: bool = False
    text: str = ""


def markup_head(source: str) -> MarkupPage:
    """The title, the properties and the first paragraph, without the rest.

    What the search index keeps of every page, read from the top of the file
    only, so an index of thousands of pages stays quick to build.
    """
    page = MarkupPage()
    lines = source.splitlines()
    index = 0
    first: str | None = None
    while index < len(lines):
        stripped = lines[index].strip()
        index += 1
        if not stripped:
            continue
        if page.title is None and _TITLE.match(stripped) and not _HEADING.match(stripped):
            page.title = inline(_TITLE.match(stripped).group(1), markdown=False)
            continue
        if _PROPERTY.match(stripped):
            key, _, value = stripped[1:].partition(":")
            page.properties.setdefault(key.strip().lower(), value.strip())
            continue
        if stripped.startswith('"""'):
            body, index = _quoted(lines, index - 1)
            page.summary = _squash(inline(body, markdown=False))
            break
        if _DEPRECATED.match(stripped):
            page.deprecated = True
        if _HEADING.match(stripped) or _SECTION.match(stripped):
            # The body has begun: no summary line is coming.
            break
        if _not_a_paragraph(stripped):
            continue
        # A paragraph before any summary line stands in for one, unless a
        # summary line follows it.
        paragraph = [stripped]
        while index < len(lines) and lines[index].strip():
            paragraph.append(lines[index].strip())
            index += 1
        if first is None:
            first = _squash(inline(" ".join(paragraph), markdown=False))
    if page.summary is None:
        page.summary = first
    if "deprecated" in page.properties.get("status", "").lower():
        page.deprecated = True
    return page


def _not_a_paragraph(stripped: str) -> bool:
    return bool(
        _HEADING.match(stripped)
        or _SECTION.match(stripped)
        or _DIRECTIVE.match(stripped)
        or _INCLUDE.match(stripped)
        or _TABLE.match(stripped)
        or _TAG_LINE.match(stripped)
        or stripped.startswith(("<!--", "{{{", "[Image:", "[Icon:"))
    )


def _quoted(lines: list[str], start: int) -> tuple[str, int]:
    """A `\"\"\"` block from the line it opens on, and the line after it closes."""
    first = lines[start].strip()[3:]
    if '"""' in first:
        return first.split('"""', 1)[0], start + 1
    parts = [first]
    index = start + 1
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if '"""' in line:
            parts.append(line.split('"""', 1)[0])
            break
        parts.append(line)
    return " ".join(part for part in parts if part), index


def markup_to_text(
    source: str,
    *,
    markdown: bool = False,
    read: Callable[[str], str | None] | None = None,
    where: str = "",
) -> MarkupPage:
    """A whole page as text, with its includes pulled in through `read`.

    `where` is the page's own help path, such as `nodes/sop/attribwrangle`,
    which an include names its target relative to.
    """
    page = markup_head(source)
    renderer = _Markup(markdown=markdown, read=read)
    lines = renderer.render(source, where=where, depth=0, seen=frozenset({f"{where}#"}))
    page.text = _finish(lines)
    return page


class _Markup:
    def __init__(self, *, markdown: bool, read: Callable[[str], str | None] | None) -> None:
        self.markdown = markdown
        self.read = read
        # Shared by every include of the page, however deep.
        self.includes_left = MAX_INCLUDES
        self.chars_left = MAX_INCLUDED_CHARS

    def render(self, source: str, *, where: str, depth: int, seen: frozenset[str]) -> list[str]:
        out: list[str] = []
        lines = _COMMENT.sub("", source.expandtabs(4)).splitlines()
        index = 0
        titled = False
        # Where the last label ended, so the blank line under it is dropped
        # and the label sits on its text.
        labelled = -1
        rows = _Rows(out)
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            index += 1
            if not stripped:
                if len(out) != labelled and not rows.open:
                    out.append("")
                continue
            if stripped.startswith("{{{"):
                index = self._code(lines, index - 1, out)
                continue
            if rows.take(stripped, _indent(line), self.inline):
                continue
            if not titled and depth == 0 and _TITLE.match(stripped):
                if not _HEADING.match(stripped):
                    titled = True
                    continue
            if _PROPERTY.match(stripped):
                continue
            if stripped.startswith('"""'):
                body, index = _quoted(lines, index - 1)
                out.extend(["", self.inline(body), ""])
                continue
            heading = _HEADING.match(stripped)
            if heading:
                self._heading(out, len(heading.group(1)), heading.group(2))
                continue
            section = _SECTION.match(stripped)
            if section:
                name = section.group(1).lower()
                self._heading(out, 2, SECTIONS.get(name, name.replace("_", " ").capitalize()))
                continue
            include = _INCLUDE.match(stripped)
            if include:
                out.extend(self._include(include.group(1), where=where, depth=depth, seen=seen))
                continue
            usage = _USAGE.match(stripped)
            if usage:
                out.append(self.inline(usage.group(1)))
                continue
            notice = _NOTICE.match(stripped)
            if notice:
                word = notice.group(1).capitalize()
                rest = self.inline(notice.group(2)) if notice.group(2) else ""
                label = f"**{word}:**" if self.markdown else f"{word}:"
                out.append(f"{label} {rest}".rstrip())
                labelled = len(out) if not rest else -1
                continue
            directive = _DIRECTIVE.match(stripped)
            if directive:
                if directive.group(2):
                    out.append(self.inline(directive.group(2)))
                continue
            if _TAG_LINE.match(stripped):
                continue
            if stripped.startswith("[") and stripped.endswith("]") and not self.inline(stripped):
                # A line that is only a picture.
                continue
            bullet = _BULLET.match(stripped)
            if bullet:
                out.append("- " + self.inline(bullet.group(1)))
                continue
            label = _LABEL.match(stripped)
            if label and _opens_block(lines, index, _indent(line)):
                text = self.inline(label.group(1))
                out.extend(["", f"**{text}**" if self.markdown else f"{text}:"])
                labelled = len(out)
                continue
            out.append(self.inline(stripped))
        rows.end()
        return out

    def _heading(self, out: list[str], level: int, text: str) -> None:
        text = self.inline(text)
        out.extend(["", ("#" * level + " " + text) if self.markdown else text, ""])

    def _code(self, lines: list[str], start: int, out: list[str]) -> int:
        """A `{{{ }}}` block, kept as it is written. Returns the line after it."""
        first = lines[start].strip()[3:]
        if "}}}" in first:
            code = first.split("}}}", 1)[0].strip()
            out.append(f"`{code}`" if self.markdown else code)
            return start + 1
        body: list[str] = []
        language = ""
        index = start + 1
        while index < len(lines):
            line = lines[index]
            index += 1
            if line.strip().startswith("}}}"):
                break
            if not body and not language and line.strip().startswith("#!"):
                language = line.strip()[2:].strip()
                continue
            body.append(line)
        margin = min((_indent(line) for line in body if line.strip()), default=0)
        body = [line[margin:].rstrip() for line in body]
        while body and not body[-1]:
            body.pop()
        if self.markdown:
            out.extend(["", f"```{language}", *body, "```", ""])
        else:
            out.extend(["", *("    " + line if line else "" for line in body), ""])
        return index

    def _include(self, target: str, *, where: str, depth: int, seen: frozenset[str]) -> list[str]:
        if self.read is None or depth >= MAX_INCLUDE_DEPTH:
            return []
        path, _, anchor = target.partition("#")
        inner = anchor.endswith("/")
        anchor = anchor.rstrip("/")
        path = resolve_link(path, where) if path else where
        key = f"{path}#{anchor}"
        if key in seen:
            return []
        if self.includes_left <= 0:
            return []
        self.includes_left -= 1
        source = self.read(path)
        if source is None or len(source) > self.chars_left:
            return []
        self.chars_left -= len(source)
        if anchor:
            source = block_of(source, anchor, inner=inner)
            if source is None:
                return []
        else:
            source = _without_head(source)
        return self.render(source, where=path, depth=depth + 1, seen=seen | {key})

    def inline(self, text: str) -> str:
        return inline(text, markdown=self.markdown)


class _Rows:
    """A table's rows, each kept on one line with its cells apart."""

    def __init__(self, out: list[str]) -> None:
        self.out = out
        self.cells: list[str] | None = None
        self.indent = 0

    @property
    def open(self) -> bool:
        return self.cells is not None

    def take(self, stripped: str, indent: int, inline: Callable[[str], str]) -> bool:
        """Whether this line belongs to a table, taking it if so."""
        table = _TABLE.match(stripped)
        if table:
            kind, text = table.group(1), table.group(2)
            if kind in ("table", "tr"):
                self.end()
                if kind == "tr":
                    self.cells = []
                    self.indent = indent
            elif kind in ("td", "th"):
                if self.cells is None:
                    self.cells = []
                    self.indent = max(indent - 1, 0)
                self.cells.append(inline(text) if text else "")
            return True
        if self.cells is not None and indent > self.indent:
            # What a cell says on the lines under it joins the cell.
            piece = inline(stripped)
            if self.cells:
                self.cells[-1] = f"{self.cells[-1]} {piece}".strip()
            else:
                self.cells.append(piece)
            return True
        if self.cells is not None:
            self.end()
            self.out.append("")
        return False

    def end(self) -> None:
        if self.cells is not None and any(self.cells):
            self.out.append(" | ".join(self.cells))
        self.cells = None


def _without_head(source: str) -> str:
    """A page without its title, for an include of the whole page."""
    lines = source.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _TITLE.match(stripped) and not _HEADING.match(stripped):
            return "\n".join(lines[index + 1 :])
        break
    return source


def block_of(source: str, anchor: str, *, inner: bool = False) -> str | None:
    """The block of a page that carries `#id: anchor`, or its body alone.

    The block is the line that opens it, such as a parameter's name, and every
    line indented under that one.
    """
    lines = source.expandtabs(4).splitlines()
    for index, line in enumerate(lines):
        found = _ID.match(line)
        if not found or found.group(1) != anchor:
            continue
        depth = _indent(line)
        opener = index - 1
        while opener >= 0 and (not lines[opener].strip() or _indent(lines[opener]) >= depth):
            opener -= 1
        if opener < 0:
            return None
        margin = _indent(lines[opener])
        end = opener + 1
        while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > margin):
            end += 1
        body = [text for text in lines[opener + 1 : end] if not _ID.match(text)]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        inset = min((_indent(text) for text in body if text.strip()), default=0)
        body = [text[inset:] for text in body]
        if inner:
            return "\n".join(body)
        head = lines[opener][margin:]
        return "\n".join([head, *("    " + text if text else "" for text in body)])
    return None


def resolve_link(target: str, where: str) -> str:
    """A help path from a link or include target, against the page it is on.

    `/vex/_strictvariables` is from the top, `wrangle_syntax` is beside the
    page, and `Node:sop/attribvop` or `Vex:noise` name a node or a function.
    """
    target = target.strip()
    kind, colon, rest = target.partition(":")
    if colon and kind in LINK_ROOTS and not rest.startswith("//"):
        return (LINK_ROOTS[kind] + rest.strip().strip("/")).strip("/")
    if target.startswith("/"):
        return target.strip("/")
    folder = where.rsplit("/", 1)[0] if "/" in where else ""
    parts = [part for part in f"{folder}/{target}".split("/") if part and part != "."]
    tidy: list[str] = []
    for part in parts:
        if part == "..":
            if tidy:
                tidy.pop()
        else:
            tidy.append(part)
    return "/".join(tidy)


LINK_ROOTS = {"Node": "nodes/", "Vex": "vex/functions/", "Hom": "hom/hou/", "Cmd": "commands/"}


def inline(text: str, *, markdown: bool) -> str:
    """One line of help markup with its inline markup worked out."""
    pieces = _CODE_SPAN.split(text)
    out: list[str] = []
    for number, piece in enumerate(pieces):
        if number % 2:
            code = _PLACEHOLDER.sub(r"<\1>", piece)
            out.append(code if markdown else code[1:-1])
            continue
        piece = _LINK.sub(_link_text, piece)
        piece = _PLACEHOLDER.sub(r"<\1>", piece)
        piece = _KEYS.sub(r"\1", piece)
        if markdown:
            piece = _BOLD.sub(r"**\1**", piece)
            piece = _STRONG.sub(r"**\1**", piece)
            piece = _ITALIC_QUOTES.sub(r"*\1*", piece)
            piece = _ITALIC.sub(r"*\1*", piece)
        else:
            piece = _BOLD.sub(r"\1", piece)
            piece = _STRONG.sub(r"\1", piece)
            piece = _ITALIC_QUOTES.sub(r"\1", piece)
            piece = _ITALIC.sub(r"\1", piece)
        out.append(piece)
    return "".join(out).strip()


def _link_text(found: re.Match[str]) -> str:
    body = found.group(1)
    if "|" in body:
        return body.split("|", 1)[0].strip()
    kind, colon, rest = body.partition(":")
    if colon and (kind in ("Image", "Icon") or _MEDIA.search(rest.strip())):
        # A picture or a clip shown on the page: nothing to read.
        return ""
    if colon and kind in LINK_ROOTS:
        return rest.strip().rsplit("/", 1)[-1] if kind != "Node" else rest.strip()
    return body.strip()


def _opens_block(lines: list[str], index: int, indent: int) -> bool:
    """Whether the next line with text on it sits deeper than `indent`."""
    while index < len(lines):
        if lines[index].strip():
            return _indent(lines[index]) > indent
        index += 1
    return False


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _squash(text: str) -> str:
    return " ".join(text.split())


def _finish(lines: list[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines)
    return _BLANKS.sub("\n\n", text).strip()


def excerpt(text: str | None, limit: int = EXCERPT_CHARS) -> str:
    text = _squash(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


# Section: help server HTML

# Parts of a page that are furniture, by tag, id or class.
_SKIP_TAGS = frozenset({"script", "style", "nav", "button", "img", "svg", "noscript", "form"})
_SKIP_IDS = frozenset({"toc", "premeta", "navsearch", "qbtn"})
_SKIP_CLASSES = frozenset(
    {"headerlink", "ancestors", "pageicon", "example-buttons", "subtitle", "tagged-list"}
)
_BLOCKS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "table",
        "tr",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "pre",
        "td",
        "th",
    }
)
_VOID = frozenset(
    {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "wbr"}
)


def html_to_text(page: str, *, markdown: bool = False) -> tuple[str | None, str]:
    """The title and the readable text of one help server page."""
    start = page.find("<main")
    end = page.rfind("</main>")
    if start >= 0 and end > start:
        page = page[start : end + len("</main>")]
    reader = _HtmlText(markdown=markdown)
    reader.feed(page)
    reader.close()
    return reader.title, _finish(reader.lines())


class _HtmlText(HTMLParser):
    def __init__(self, *, markdown: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.markdown = markdown
        self.title: str | None = None
        self._lines: list[str] = []
        self._line: list[str] = []
        self._stack: list[tuple[str, str]] = []  # tag and what it opened as
        self._skip = 0
        self._pre = 0
        self._in_title = False
        self._title_parts: list[str] = []
        # The next text starts a line or follows a marker, so it loses its
        # leading space.
        self._fresh = True
        # Whether the line being built holds any text yet, beside its markers.
        self._has_text = False
        # A label that turned out to head a list of tags, which is dropped.
        self._drop_label = False
        # Table cells open, and cells so far in the row: a row is one line.
        self._cell = 0
        self._cells_in_row = 0
        # What the page has made this reader do, against the caps.
        self._tokens = 0
        self._kept = 0
        self.cut = False

    def _spent(self) -> bool:
        """Count one tag or piece of text; say whether the page is over its caps."""
        self._tokens += 1
        if self._tokens > MAX_HTML_TOKENS or self._kept > MAX_HTML_TEXT:
            self.cut = True
        return self.cut

    def lines(self) -> list[str]:
        self._flush()
        return self._lines

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._spent() or (tag not in _VOID and len(self._stack) >= MAX_HTML_DEPTH):
            return
        found = dict(attrs)
        classes = set((found.get("class") or "").split())
        if tag in _VOID:
            if self._skip:
                return
            if tag == "br":
                self._flush()
            elif tag == "img" and "keyicon" in classes and found.get("title"):
                # A key or mouse button drawn as an icon: its name is the text.
                self._text(str(found["title"]))
            return
        parent = self._stack[-1][0] if self._stack else ""
        role = ""
        if (
            self._skip
            or tag in _SKIP_TAGS
            or found.get("id") in _SKIP_IDS
            or classes & _SKIP_CLASSES
        ):
            self._skip += 1
            role = "skip"
        elif tag == "i" and "fa-tag" in classes:
            self._drop_label = True
        elif tag == "h1" and self.title is None:
            self._in_title = True
            role = "title"
        elif tag in ("h2", "h3", "h4", "h5", "h6"):
            self._open_line("#" * int(tag[1]) + " " if self.markdown else "")
            role = "heading"
        elif (tag == "p" or tag == "td") and "label" in classes and parent != "li":
            if parent == "div" and self._stack and self._stack[-1][1] == "usage":
                self._open_line("")
                role = "block"
            else:
                self._open_line("**" if self.markdown else "")
                role = "label"
        elif tag == "div" and "usage" in classes:
            self._flush()
            role = "usage"
        elif tag == "li":
            self._flush()
            self._marker("- ")
            role = "item"
        elif tag == "pre":
            self._open_line("")
            self._pre += 1
            if self.markdown:
                self._lines.append("```")
            role = "pre"
        elif tag == "code" and not self._pre and self.markdown:
            self._marker("`")
            role = "code"
        elif tag in ("strong", "b") and self.markdown:
            self._marker("**")
            role = "strong"
        elif tag == "tr":
            self._flush()
            self._cells_in_row = 0
            role = "row"
        elif tag in ("td", "th"):
            if self._cells_in_row and self._has_text:
                self._trim()
                self._marker(" | ")
            else:
                self._flush()
            self._cells_in_row += 1
            self._cell += 1
            role = "cell"
        elif tag in _BLOCKS and self._cell:
            role = "inline block"
        elif tag in _BLOCKS:
            # A paragraph opening a list item keeps the item's marker.
            if self._has_text:
                self._flush()
            role = "para" if tag == "p" and parent != "li" else "block"
        self._stack.append((tag, role))

    def handle_endtag(self, tag: str) -> None:
        if self._spent() or tag in _VOID:
            return
        # Close back to the matching tag; loose HTML closes what it left open.
        # The stack is never deeper than the cap, so this look is bounded.
        for depth in range(len(self._stack) - 1, -1, -1):
            if self._stack[depth][0] == tag:
                break
        else:
            return
        while len(self._stack) > depth:
            _, role = self._stack.pop()
            self._close(role)

    def _close(self, role: str) -> None:
        if role == "skip":
            self._skip -= 1
        elif role == "title":
            self._in_title = False
            self.title = _squash("".join(self._title_parts)) or None
        elif role == "heading":
            self._flush()
            self._lines.append("")
        elif role == "label":
            if self._drop_label:
                self._drop_label = False
                self._line = []
                self._fresh = True
                return
            self._trim()
            self._line.append("**" if self.markdown else ":")
            self._flush()
        elif role == "pre":
            self._flush()
            self._pre -= 1
            if self.markdown:
                self._lines.append("```")
            self._lines.append("")
        elif role in ("code", "strong"):
            self._trim()
            self._line.append("`" if role == "code" else "**")
        elif role in ("block", "item", "usage"):
            self._flush()
        elif role == "para":
            self._flush()
            self._lines.append("")
        elif role == "row":
            self._flush()
        elif role == "cell":
            self._cell -= 1
        elif role == "inline block" and self._has_text:
            self._marker(" ")

    def handle_data(self, data: str) -> None:
        if self._spent() or self._skip:
            return
        self._kept += len(data)
        if self._in_title:
            if sum(len(part) for part in self._title_parts) < MAX_HIT_FIELD:
                self._title_parts.append(data)
            return
        if self._pre:
            parts = data.split("\n")
            for number, part in enumerate(parts):
                if number:
                    self._lines.append(("" if self.markdown else "    ") + "".join(self._line))
                    self._line = []
                self._line.append(part)
            return
        self._text(data)

    def _text(self, data: str) -> None:
        text = _SPACES.sub(" ", data.replace("\n", " "))
        if self._fresh:
            text = text.lstrip()
            if not text:
                return
        elif self._line and self._line[-1].endswith(" ") and text.startswith(" "):
            text = text.lstrip()
        self._line.append(text)
        self._fresh = False
        self._has_text = True

    def _open_line(self, marker: str) -> None:
        self._flush()
        self._lines.append("")
        self._marker(marker)

    def _marker(self, marker: str) -> None:
        if marker:
            self._line.append(marker)
        self._fresh = True

    def _trim(self) -> None:
        if self._line:
            self._line[-1] = self._line[-1].rstrip()

    def _flush(self) -> None:
        self._fresh = True
        self._has_text = False
        if self._pre:
            if self._line:
                self._lines.append(("" if self.markdown else "    ") + "".join(self._line))
                self._line = []
            return
        text = "".join(self._line).strip()
        self._line = []
        if text and text not in ("-", "**", "****", ":", "``"):
            self._lines.append(text)


def search_hits(page: str, tidy: Callable[[str], str | None] | None = None) -> list[dict[str, str]]:
    """The hits on a help server search page: path, title and what it says.

    `tidy` turns each link into a help path the way every other path is
    named, or refuses it; a hit whose link it refuses is left out. At most
    `MAX_HITS` are read.
    """
    reader = _Hits()
    reader.feed(page)
    reader.close()
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for hit in reader.hits:
        href = hit.get("href", "")
        path = tidy(href) if tidy is not None else href.split("#", 1)[0].strip("/")
        if not path or path.startswith(("find", "_")) or path in seen:
            continue
        seen.add(path)
        hits.append(
            {
                "path": path,
                "title": _squash(html.unescape(hit.get("title", ""))),
                "excerpt": excerpt(hit.get("summary") or hit.get("desc") or ""),
            }
        )
    return hits


class _Hits(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[dict[str, str]] = []
        self._hit: dict[str, str] | None = None
        self._depth = 0
        self._field: str | None = None
        self._field_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = dict(attrs)
        classes = (found.get("class") or "").split()
        if tag == "div" and "hit" in classes and not {"more", "findpage"} & set(classes):
            if len(self.hits) >= MAX_HITS:
                self._hit = None
                return
            self._hit = {}
            self.hits.append(self._hit)
            self._depth = 1
            return
        if self._hit is None:
            return
        if tag == "div":
            self._depth += 1
        if tag == "a" and "label" in classes and "href" not in self._hit:
            self._hit["href"] = (found.get("href") or "")[:MAX_HIT_FIELD]
            self._field, self._field_tag = "title", "a"
        elif tag == "small" and "desc" in classes:
            self._field, self._field_tag = "desc", "small"
        elif tag == "p" and "summary" in classes:
            self._field, self._field_tag = "summary", "p"

    def handle_endtag(self, tag: str) -> None:
        if self._hit is None:
            return
        if tag == self._field_tag:
            self._field = self._field_tag = None
        if tag == "div":
            self._depth -= 1
            if self._depth <= 0:
                self._hit = None

    def handle_data(self, data: str) -> None:
        if self._hit is not None and self._field:
            held = self._hit.get(self._field, "")
            if len(held) < MAX_HIT_FIELD:
                self._hit[self._field] = (held + data)[:MAX_HIT_FIELD]
