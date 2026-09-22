"""The guard that keeps private working material out of a public repo.

The terms it protects are not in the repo, so the tests write a list of their
own in a temporary folder. The credit lines are built from their words here,
the way the guard builds them, so this file does not carry one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

GUARD = Path(__file__).resolve().parents[1] / "hooks" / "leak_guard.py"

CREDIT = "Co" + "-Authored" + "-By"
MADE_WITH = "Generated" + " with"


def load() -> Any:
    spec = importlib.util.spec_from_file_location("leak_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard() -> Any:
    return load()


def terms_at(root: Path, text: str) -> Path:
    path = root / ".context" / "leak-terms.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# Section: a list that arms nothing


def test_a_missing_list_blocks(guard: Any, tmp_path: Path) -> None:
    assert guard.load_terms(tmp_path) is None


def test_an_empty_list_blocks(guard: Any, tmp_path: Path) -> None:
    terms_at(tmp_path, "")
    assert guard.load_terms(tmp_path) is None


def test_a_list_of_nothing_but_comments_blocks(guard: Any, tmp_path: Path) -> None:
    """A guard with nothing to match must never read as a guard that passed."""
    terms_at(tmp_path, "# the real list is kept off this machine\n\n   \n# and here\n")
    assert guard.load_terms(tmp_path) is None


# Section: what a list catches


def pattern(guard: Any, tmp_path: Path, text: str = "seeded-term\n") -> Any:
    terms_at(tmp_path, text)
    loaded = guard.load_terms(tmp_path)
    assert loaded is not None
    return loaded[1]


def test_a_term_from_the_list_is_caught(guard: Any, tmp_path: Path) -> None:
    found = pattern(guard, tmp_path)
    assert guard.scan("nothing here\nand a seeded-term there\n", found, "file")
    assert guard.scan("seeded-termite", found, "file") == []


def test_a_term_is_caught_whatever_its_case(guard: Any, tmp_path: Path) -> None:
    found = pattern(guard, tmp_path)
    assert guard.scan("A SEEDED-TERM", found, "file")


@pytest.mark.parametrize(
    "credit",
    [
        CREDIT,
        CREDIT.replace("-", " - "),
        CREDIT.replace("-A", " -A"),
        CREDIT.upper(),
        CREDIT.replace("-", "\n", 1),
        MADE_WITH,
        MADE_WITH.replace(" ", "  "),
        MADE_WITH.replace(" ", "\n"),
        MADE_WITH.replace(" ", "-"),
        "nore" + "ply@example.test",
    ],
)
def test_a_credit_line_is_caught_however_it_is_spaced(
    guard: Any, tmp_path: Path, credit: str
) -> None:
    found = pattern(guard, tmp_path)
    hits = guard.scan(f"a message\n\n{credit}: somebody\n", found, "commit message")
    assert hits, credit


def test_an_ordinary_message_passes(guard: Any, tmp_path: Path) -> None:
    found = pattern(guard, tmp_path)
    message = "Finish the receipt of work that outlived its call\n\nIt is recorded when it ends.\n"
    assert guard.scan(message, found, "commit message") == []


def test_the_words_of_a_credit_line_are_harmless_apart(guard: Any, tmp_path: Path) -> None:
    found = pattern(guard, tmp_path)
    assert guard.scan("the file was generated, together with its sidecar", found, "file") == []


def test_this_guard_does_not_match_its_own_source(guard: Any, tmp_path: Path) -> None:
    # A list whose own term is in neither file, so only the credit lines and
    # the address are being asked about here.
    found = pattern(guard, tmp_path, "a-word-" + "in-neither-file\n")
    assert guard.scan(GUARD.read_text(encoding="utf-8"), found, "guard") == []
    assert guard.scan(Path(__file__).read_text(encoding="utf-8"), found, "tests") == []
