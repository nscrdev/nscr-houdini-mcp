"""The lint that keeps shipped text free of client names and of `integrations/`.

The integrations rule is checked against a folder built in a temporary
repository, since the real one does not exist yet.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

LINT = Path(__file__).resolve().parents[1] / "scripts" / "lint_client_names.py"

LONG_LINE = "This sentence is long enough that finding it twice cannot be a coincidence."


def load() -> Any:
    spec = importlib.util.spec_from_file_location("lint_client_names", LINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lint() -> Any:
    return load()


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "repo"
    write(root, "integrations/one/notes.md", f"# Notes\n\n{LONG_LINE}\nshort line\n")
    return root


def test_an_import_from_integrations_in_src_is_refused(lint: Any, repo: Path) -> None:
    path = write(repo, "src/pkg/mod.py", "from integrations.one import thing\n")
    problems = lint.check_integrations([path], repo)
    assert problems == [f"{Path('src/pkg/mod.py')}:1: refers to integrations/"]


def test_a_package_relative_import_is_refused(lint: Any, repo: Path) -> None:
    path = write(repo, "src/pkg/mod.py", "import pkg.integrations\nfrom .integrations import x\n")
    assert len(lint.check_integrations([path], repo)) == 2


def test_a_link_in_a_skill_is_refused(lint: Any, repo: Path) -> None:
    path = write(repo, "skills/one/SKILL.md", "See [the notes](../../integrations/one/notes.md).\n")
    problems = lint.check_integrations([path], repo)
    assert problems == [f"{Path('skills/one/SKILL.md')}:1: refers to integrations/"]


def test_a_copied_line_is_refused_even_reflowed(lint: Any, repo: Path) -> None:
    reflowed = "   " + LONG_LINE.replace(" ", "  ") + "\n"
    path = write(repo, "skills/one/SKILL.md", f"# Mine\n\n{reflowed}")
    problems = lint.check_integrations([path], repo)
    assert problems == [f"{Path('skills/one/SKILL.md')}:3: copies a line from integrations/"]


def test_a_short_shared_line_is_not_a_copy(lint: Any, repo: Path) -> None:
    path = write(repo, "skills/one/SKILL.md", "# Notes\n\nshort line\n")
    assert lint.check_integrations([path], repo) == []


def test_a_symlink_into_integrations_is_refused(lint: Any, repo: Path) -> None:
    link = repo / "skills" / "one" / "SKILL.md"
    link.parent.mkdir(parents=True)
    try:
        os.symlink(repo / "integrations" / "one" / "notes.md", link)
    except (OSError, NotImplementedError):
        pytest.skip("this system cannot make a symlink here")
    problems = lint.check_integrations([link], repo)
    assert problems == [f"{Path('skills/one/SKILL.md')}: links into integrations/"]


def test_files_outside_the_shipped_folders_may_mention_it(lint: Any, repo: Path) -> None:
    text = f"The integrations/ folder holds one client each.\n{LONG_LINE}\n"
    doc = write(repo, "docs/layout.md", text)
    assert lint.check_integrations([doc], repo) == []


def test_plain_words_about_integrations_are_fine(lint: Any, repo: Path) -> None:
    path = write(repo, "skills/one/SKILL.md", "Works without any integrations. Nothing else.\n")
    assert lint.check_integrations([path], repo) == []


def test_the_rule_holds_with_no_integrations_folder(lint: Any, tmp_path: Path) -> None:
    root = tmp_path.resolve()
    path = write(root, "src/pkg/mod.py", "from integrations import x\nprint('ok')\n")
    expected = [f"{Path('src/pkg/mod.py')}:1: refers to integrations/"]
    assert lint.check_integrations([path], root) == expected


def test_an_untracked_skill_is_checked_by_default(lint: Any, tmp_path: Path) -> None:
    root = tmp_path.resolve()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    write(root, "README.md", "hello\n")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
    skill = write(root, "skills/new/SKILL.md", "draft\n")
    paths = lint.default_paths(root)
    assert skill in paths
    assert root / "README.md" in paths


def test_a_client_name_in_a_skill_is_refused(lint: Any, tmp_path: Path) -> None:
    root = tmp_path.resolve()
    term = lint.TERMS[0]
    path = write(root, "skills/one/SKILL.md", f"Built for {term} only.\n")
    problems = lint.check([path], root)
    assert len(problems) == 1 and problems[0].endswith(term)
