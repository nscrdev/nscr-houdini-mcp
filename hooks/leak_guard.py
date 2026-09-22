#!/usr/bin/env python3
"""Block a commit that carries private working material into this public repo.

The guard matches staged file paths, staged file content and the commit message
against a term list. The list itself would give away what it protects, so it is
not tracked: it lives at `.context/leak-terms.txt`, which is ignored by git.

Fail closed: if the list is missing, unreadable or empty, the commit is
rejected. A missing guard must never read as a passing guard.

Usage:
    leak_guard.py --staged            # staged paths and staged content
    leak_guard.py --message FILE      # a commit message file
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TERMS_PATH = Path(".context") / "leak-terms.txt"

# The term list may have been written on any platform, so read it with a
# tolerant encoding and strip the carriage return a CRLF file leaves behind.
TEXT_ENCODING = "utf-8-sig"

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".bgeo",
    ".sc",
    ".hip",
    ".hiplc",
    ".hipnc",
    ".exr",
    ".otl",
    ".hda",
}


# Credit lines, refused whatever the term list says. Each one is written as
# its separate words, so this file does not match its own rule, and the words
# are joined into a pattern that allows any run of spaces, line breaks and
# hyphens between them. A line break counts, so a credit line wrapped by an
# editor is caught as readily as one on a single line.
FIXED_PHRASES = (
    ("Co", "Authored", "By"),
    ("Generated", "with"),
)

# Fixed terms that are one word and take no separator.
FIXED_WORDS = ("nore" + "ply@",)

# What may sit between the words of a credit line and still be one.
BETWEEN_WORDS = r"[-\s]+"


def fail(message: str) -> int:
    print(f"leak guard: {message}", file=sys.stderr)
    return 1


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def repo_root() -> Path:
    return Path(git("rev-parse", "--show-toplevel").strip())


def load_terms(root: Path) -> tuple[list[str], re.Pattern[str]] | None:
    path = root / TERMS_PATH
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding=TEXT_ENCODING)
    except (OSError, UnicodeDecodeError):
        return None

    terms: list[str] = []
    for line in raw.splitlines():
        line = line.split("#", 1)[0].strip().strip("\r")
        if line:
            terms.append(line)
    # The list decides whether this guard is armed at all. A file of nothing
    # but comments is an empty list, and an empty list blocks the commit.
    if not terms:
        return None

    parts = [_word_pattern(term) for term in terms + list(FIXED_WORDS)]
    for phrase in FIXED_PHRASES:
        parts.append(BETWEEN_WORDS.join(_word_pattern(word) for word in phrase))
    return terms, re.compile("|".join(parts), re.IGNORECASE)


def _word_pattern(term: str) -> str:
    """One term, matched on its own rather than inside a longer word."""
    escaped = re.escape(term)
    if term[:1].isalnum():
        escaped = r"(?<!\w)" + escaped
    if term[-1:].isalnum():
        escaped = escaped + r"(?!\w)"
    return escaped


def staged_paths(root: Path) -> list[str]:
    out = git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    return [name for name in out.split("\0") if name]


def staged_content(path: str) -> str | None:
    if Path(path).suffix.lower() in SKIP_SUFFIXES:
        return None
    blob = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=False).stdout
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan(text: str, pattern: re.Pattern[str], label: str) -> list[str]:
    """Every match in a file or a message, with the line each one starts on.

    The whole text is searched at once rather than line by line, because a
    credit line broken across two lines is still a credit line.
    """
    hits = []
    for match in pattern.finditer(text):
        number = text.count("\n", 0, match.start()) + 1
        found = " ".join(match.group(0).split())
        hits.append(f"{label}:{number}: {found}")
    return hits


def check_staged(root: Path, pattern: re.Pattern[str]) -> list[str]:
    hits: list[str] = []
    for path in staged_paths(root):
        if Path(path).as_posix() == TERMS_PATH.as_posix():
            hits.append(f"{path}: the term list must never be committed")
            continue
        match = pattern.search(path)
        if match:
            hits.append(f"{path}: file path contains a private term: {match.group(0)}")
        text = staged_content(path)
        if text is not None:
            hits.extend(scan(text, pattern, path))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true")
    group.add_argument("--message", type=Path)
    args = parser.parse_args(argv)

    try:
        root = repo_root()
    except subprocess.CalledProcessError:
        return fail("not inside a git repository")

    loaded = load_terms(root)
    if loaded is None:
        return fail(
            f"term list {TERMS_PATH} is missing, unreadable or empty.\n"
            "  This guard keeps private working material out of a public repo, so a\n"
            "  missing list blocks the commit. Restore the file from your own backup,\n"
            "  then commit again. Never add it to git."
        )
    _, pattern = loaded

    if args.staged:
        hits = check_staged(root, pattern)
        source = "staged changes"
    else:
        try:
            text = args.message.read_text(encoding=TEXT_ENCODING)
        except (OSError, UnicodeDecodeError) as error:
            return fail(f"cannot read the commit message: {error}")
        hits = scan(text, pattern, "commit message")
        source = "the commit message"

    if hits:
        print(f"leak guard: private terms found in {source}:", file=sys.stderr)
        for hit in hits[:40]:
            print(f"  {hit}", file=sys.stderr)
        if len(hits) > 40:
            print(f"  ... and {len(hits) - 40} more", file=sys.stderr)
        print(
            "\nRewrite in your own words with no reference to the private material.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
