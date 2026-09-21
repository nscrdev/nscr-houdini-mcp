#!/usr/bin/env python3
"""Fail when tracked text names a specific MCP client, agent harness or model vendor.

The server, its tool descriptions, its skills and its docs must read the same
whatever is driving them. Naming one product in shipped text makes the project
look tied to it, so the name belongs in nobody's copy of the repo.

This file holds the term list on purpose: it is the one place the names are
allowed to appear.

Usage:
    scripts/lint_client_names.py             # every tracked text file
    scripts/lint_client_names.py FILE ...    # only these files

Escape hatch: put `lint-allow: client-names` on the offending line, or on the
line right above it when the file format has no room for a trailing comment.
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


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [root / name for name in out.split("\0") if name]


def read_text(path: Path) -> str | None:
    if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
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

    paths = [p.resolve() for p in args.paths] if args.paths else tracked_files(root)
    problems = check(paths, root)
    if problems:
        print("Client or vendor names found in tracked text:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            f"\nRewrite the line so it works for any client, or mark it with `{ALLOW_MARKER}`.",
            file=sys.stderr,
        )
        return 1
    print(f"client name lint: clean ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
