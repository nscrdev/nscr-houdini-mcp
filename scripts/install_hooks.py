#!/usr/bin/env python3
"""Point this clone at the tracked hooks directory.

Works on macOS, Linux and Windows. Run it with any Python 3.11 or newer:

    python scripts/install_hooks.py

The hooks need the term list at `.context/leak-terms.txt`, which is ignored by
git and kept only on your own machine. Without it every commit is refused.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

HOOKS_DIR = "hooks"
HOOK_FILES = ("pre-commit", "commit-msg", "leak_guard.py")
TERMS_PATH = Path(".context") / "leak-terms.txt"


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(out)


def make_executable(path: Path) -> None:
    """Add the execute bit where the platform has one."""
    if os.name == "nt":
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    try:
        root = repo_root()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("not inside a git repository", file=sys.stderr)
        return 1

    for name in HOOK_FILES:
        path = root / HOOKS_DIR / name
        if not path.is_file():
            print(f"missing hook file: {path}", file=sys.stderr)
            return 1
        make_executable(path)

    subprocess.run(["git", "config", "core.hooksPath", HOOKS_DIR], cwd=root, check=True)
    print(f"hooks installed: core.hooksPath = {HOOKS_DIR}")

    if not (root / TERMS_PATH).is_file():
        print()
        print(f"warning: {TERMS_PATH.as_posix()} is missing, so commits will be refused.")
        print("Create it with one term per line ('#' starts a comment). Never track it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
