#!/usr/bin/env python3
"""Fail when tracked text names a specific MCP client, agent harness or model vendor.

The server, its tool descriptions, its skills and its docs must read the same
whatever is driving them. Naming one product in shipped text makes the project
look tied to it, so the name belongs in nobody's copy of the repo.

This file holds the term list on purpose: it is the one place the names are
allowed to appear.

A second rule keeps the shipped parts apart from any `integrations/` folder:
nothing under `skills/` or `src/` may import from it, link to it, or carry a
line copied out of it. Whatever lives there is written for one client, and the
skills and the server must stay readable without it.

Usage:
    scripts/lint_client_names.py             # every tracked text file, and all of skills/
    scripts/lint_client_names.py FILE ...    # only these files

Escape hatch for the name rule, and only that one: put `lint-allow:
client-names` on the offending line, or on the line right above it when the
file format has no room for a trailing comment.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Terms are matched case insensitively, on word boundaries.
TERMS = [
    "anthropic",
    "openai",
    "chatgpt",
    "copilot",
    "codex",
    "cursor",
    "windsurf",
    "cline",
    "grok",
    "gemini",
    "claude",
    "opus",
    "sonnet",
    "haiku",
    "fable",
]

ALLOW_MARKER = "lint-allow: client-names"

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".bgeo",
    ".hip",
    ".exr",
    ".otl",
    ".hda",
}

PATTERN = re.compile(r"\b(" + "|".join(TERMS) + r")\b", re.IGNORECASE)

SELF = Path(__file__).resolve()

# The folders that ship, and so must stand on their own.
SHIPPED_DIRS = ("skills", "src")

# The folder whose text is for one client only.
INTEGRATIONS_DIR = "integrations"

# An import of the folder as a module, or a path into it, in any file format.
INTEGRATIONS_REFERENCE = re.compile(
    r"(?:^|[^\w])" + INTEGRATIONS_DIR + r"(?:[/\\]|\.\w)"
    r"|\b(?:from|import)\s+[\w.]*\b" + INTEGRATIONS_DIR + r"\b",
    re.IGNORECASE,
)

# A line this long, whitespace collapsed, found in both places is a copy. A
# shorter one could be a heading or a common phrase shared by accident.
COPIED_LINE_MIN = 40


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [root / name for name in out.split("\0") if name]


def files_under(root: Path, folder: str) -> list[Path]:
    """Every file under one top level folder on disk, tracked or not yet.

    A skill is checked from the moment it is written, not only once it is
    committed, so the folder is walked as well as the tracked list.
    """
    base = root / folder
    if not base.is_dir():
        return []
    return sorted(path for path in base.rglob("*") if path.is_file() or path.is_symlink())


def default_paths(root: Path) -> list[Path]:
    """The tracked files, plus everything under `skills/` on disk."""
    seen: dict[Path, None] = {}
    for path in [*tracked_files(root), *files_under(root, "skills")]:
        seen.setdefault(path, None)
    return list(seen)


def read_text(path: Path) -> str | None:
    if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        return None


def check(paths: list[Path], root: Path) -> list[str]:
    problems: list[str] = []
    for path in paths:
        if path.resolve() == SELF:
            continue
        text = read_text(path)
        if text is None:
            continue
        lines = text.splitlines()
        for number, line in enumerate(lines, start=1):
            previous = lines[number - 2] if number > 1 else ""
            if ALLOW_MARKER in line or ALLOW_MARKER in previous:
                continue
            match = PATTERN.search(line)
            if match:
                rel = path.relative_to(root) if path.is_relative_to(root) else path
                problems.append(f"{rel}:{number}: names a client or vendor: {match.group(0)}")
    return problems


def is_under(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
    except ValueError:
        return False
    return True


def shipped(path: Path, root: Path) -> bool:
    """Whether a path is in one of the folders that ship."""
    return any(is_under(path, root / folder) for folder in SHIPPED_DIRS)


def normalised_lines(text: str) -> set[str]:
    """The long lines of a text, with runs of whitespace made one space."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return {line for line in lines if len(line) >= COPIED_LINE_MIN}


def check_integrations(paths: list[Path], root: Path) -> list[str]:
    """Shipped files that import, link to or copy from `integrations/`."""
    folder = root / INTEGRATIONS_DIR
    source_lines: set[str] = set()
    for path in files_under(root, INTEGRATIONS_DIR):
        text = read_text(path)
        if text is not None:
            source_lines |= normalised_lines(text)

    problems: list[str] = []
    for path in paths:
        # The path as written, not resolved, so a link placed in a shipped
        # folder is judged by where it sits and then by where it points.
        where = path if path.is_absolute() else root / path
        if not shipped(where, root):
            continue
        rel = where.relative_to(root)
        if where.is_symlink():
            target = where.resolve()
            if is_under(target, folder.resolve()):
                problems.append(f"{rel}: links into {INTEGRATIONS_DIR}/")
                continue
        text = read_text(where)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if INTEGRATIONS_REFERENCE.search(line):
                problems.append(f"{rel}:{number}: refers to {INTEGRATIONS_DIR}/")
            elif source_lines and " ".join(line.split()) in source_lines:
                problems.append(f"{rel}:{number}: copies a line from {INTEGRATIONS_DIR}/")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)

    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    # The folder part is resolved and the name kept, so a link given on the
    # command line is still seen as a link.
    paths = (
        [p.absolute().parent.resolve() / p.name for p in args.paths]
        if args.paths
        else default_paths(root)
    )
    failed = False
    problems = check(paths, root)
    if problems:
        failed = True
        print("Client or vendor names found in tracked text:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            f"\nRewrite the line so it works for any client, or mark it with `{ALLOW_MARKER}`.",
            file=sys.stderr,
        )
    crossings = check_integrations(paths, root)
    if crossings:
        failed = True
        print(f"Shipped files that lean on {INTEGRATIONS_DIR}/:", file=sys.stderr)
        for problem in crossings:
            print(f"  {problem}", file=sys.stderr)
        print(
            f"\nskills/ and src/ must stand without {INTEGRATIONS_DIR}/: write the text afresh.",
            file=sys.stderr,
        )
    if failed:
        return 1
    print(f"client name lint: clean ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
