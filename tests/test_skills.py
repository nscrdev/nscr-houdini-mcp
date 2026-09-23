"""The shipped agent skills: their shape, their lint, and the commands that copy them."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import agent_skills, cli
from nscr_houdini_mcp.tools.registry import TOOLS

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "houdini-artist" / "SKILL.md"
LINT = ROOT / "scripts" / "lint_client_names.py"

PARTS = [
    "House conventions (edit these)",
    "How to think",
    "The revision budget",
    "Before you say done",
    "If a Houdini MCP server is connected",
]

# Tools the skill may name before they are in the registry.
ARRIVING = {"hou_capture", "hou_compare", "hou_outputs", "hou_docs", "hou_node_type"}


def load_lint() -> Any:
    spec = importlib.util.spec_from_file_location("lint_client_names", LINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def skill_text() -> str:
    return SKILL.read_text(encoding="utf-8")


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    """The frontmatter as plain keys and values, and the body after it."""
    lines = text.splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    fields: dict[str, str] = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields, "\n".join(lines[end + 1 :])


def sections(body: str) -> dict[str, list[str]]:
    """The non-blank lines under each second level heading."""
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = found.setdefault(line[3:].strip(), [])
        elif current is not None and line.strip():
            current.append(line)
    return found


# Section: the skill itself


def test_the_skill_is_under_two_hundred_lines() -> None:
    assert len(skill_text().splitlines()) < 200


def test_the_frontmatter_names_the_skill_and_what_it_needs() -> None:
    fields, _ = frontmatter(skill_text())
    assert fields["name"] == "houdini-artist"
    assert fields["compatibility"] == "Works best with a Houdini 22 MCP server connection."
    assert len(fields["description"]) > 100
    assert "allowed-tools" not in fields


def test_the_skill_has_exactly_the_five_parts_in_order() -> None:
    _, body = frontmatter(skill_text())
    headings = [line[3:].strip() for line in body.splitlines() if line.startswith("## ")]
    assert headings == PARTS
    assert not [line for line in body.splitlines() if line.startswith("# ")]


def test_the_short_parts_stay_short() -> None:
    _, body = frontmatter(skill_text())
    parts = sections(body)
    assert len(parts["House conventions (edit these)"]) <= 12
    assert len(parts["If a Houdini MCP server is connected"]) <= 8


def test_the_conventions_match_the_server_defaults() -> None:
    _, body = frontmatter(skill_text())
    conventions = "\n".join(sections(body)["House conventions (edit these)"])
    assert "`null`" in conventions and "`OUT_" in conventions
    assert "[conventions]" in conventions
    assert "$HIP" in conventions


def test_every_tool_named_is_real_or_arriving() -> None:
    named = set(re.findall(r"`(hou_\w+)`", skill_text()))
    registered = {tool.name for tool in TOOLS}
    assert named - registered <= ARRIVING
    assert registered - {"hou_ping"} <= named


def test_no_client_prefix_on_tool_names() -> None:
    assert not re.search(r"\w+__hou_\w+", skill_text())


def test_the_skill_has_no_dashes_links_or_quotes_of_sources() -> None:
    lines = skill_text().splitlines()
    for number, line in enumerate(lines, start=1):
        if line == "---":
            continue
        assert "—" not in line and "–" not in line, number
        assert "--" not in line, number
        assert not re.search(r"https?://|www\.", line), number
        assert not re.search(r"\]\(", line), number


def test_the_skill_passes_the_lint() -> None:
    lint = load_lint()
    assert lint.check([SKILL], ROOT) == []
    assert lint.check_integrations([SKILL], ROOT) == []
    assert lint.main([str(SKILL)]) == 0


# Section: where the skills are


def test_the_skills_travel_inside_the_package() -> None:
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = settings["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["skills"] == "nscr_houdini_mcp/skills"


def test_the_skills_next_to_the_package_are_preferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    beside = tmp_path / "nscr_houdini_mcp"
    (beside / "skills" / "one").mkdir(parents=True)
    (beside / "skills" / "one" / "SKILL.md").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(agent_skills, "__file__", str(beside / "agent_skills.py"))
    assert agent_skills.skills_root() == beside / "skills"


def test_skills_path_prints_the_folder(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["skills", "path"]) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed == agent_skills.skills_root()
    assert (printed / "houdini-artist" / "SKILL.md").is_file()


def test_skills_path_says_when_there_are_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(agent_skills, "__file__", str(tmp_path / "a" / "b" / "agent_skills.py"))
    assert cli.main(["skills", "path"]) == 1
    assert "no skills folder" in capsys.readouterr().out


# Section: installing them


def test_install_copies_every_skill_into_the_named_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "made" / "for" / "me"
    assert cli.main(["skills", "install", str(dest)]) == 0
    copied = dest / "houdini-artist" / "SKILL.md"
    assert copied.read_bytes() == SKILL.read_bytes()
    assert "installed houdini-artist" in capsys.readouterr().out


def test_install_again_changes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["skills", "install", str(tmp_path)]) == 0
    capsys.readouterr()
    assert cli.main(["skills", "install", str(tmp_path)]) == 0
    assert "unchanged houdini-artist" in capsys.readouterr().out


def test_an_edited_skill_is_kept_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["skills", "install", str(tmp_path)]) == 0
    copied = tmp_path / "houdini-artist" / "SKILL.md"
    mine = tmp_path / "houdini-artist" / "notes.md"
    copied.write_text("my own conventions\n", encoding="utf-8")
    mine.write_text("mine\n", encoding="utf-8")
    capsys.readouterr()

    # Keeping an edit is the normal upgrade path, so it is not a failure.
    assert cli.main(["skills", "install", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "kept houdini-artist" in out and "--force" in out
    assert copied.read_text(encoding="utf-8") == "my own conventions\n"

    assert cli.main(["skills", "install", str(tmp_path), "--force"]) == 0
    assert "replaced houdini-artist" in capsys.readouterr().out
    assert copied.read_bytes() == SKILL.read_bytes()
    assert mine.read_text(encoding="utf-8") == "mine\n"


def test_a_link_where_the_skill_goes_is_never_touched(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    try:
        os.symlink(elsewhere, dest / "houdini-artist", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system cannot make a symlink here")
    assert cli.main(["skills", "install", str(dest), "--force"]) == 0
    assert list(elsewhere.iterdir()) == []


def test_a_linked_skill_file_is_never_written_through(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("not yours to change\n", encoding="utf-8")
    dest = tmp_path / "dest"
    (dest / "houdini-artist").mkdir(parents=True)
    try:
        os.symlink(outside, dest / "houdini-artist" / "SKILL.md")
    except (OSError, NotImplementedError):
        pytest.skip("this system cannot make a symlink here")
    assert cli.main(["skills", "install", str(dest), "--force"]) == 0
    out = capsys.readouterr().out
    assert "kept houdini-artist" in out and "holds a link" in out
    assert outside.read_text(encoding="utf-8") == "not yours to change\n"


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are a Windows thing")
def test_a_junction_where_the_skill_goes_is_never_written_through(tmp_path: Path) -> None:
    import _winapi

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    _winapi.CreateJunction(str(elsewhere), str(dest / "houdini-artist"))
    assert agent_skills.is_link(dest / "houdini-artist")
    results = agent_skills.install(dest, force=True)
    assert [item.outcome for item in results] == [agent_skills.KEPT]
    assert list(elsewhere.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are a Windows thing")
def test_a_junction_inside_the_skill_is_never_written_through(tmp_path: Path) -> None:
    import _winapi

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    dest = tmp_path / "dest"
    (dest / "houdini-artist").mkdir(parents=True)
    _winapi.CreateJunction(str(elsewhere), str(dest / "houdini-artist" / "extra"))
    results = agent_skills.install(dest, force=True)
    assert [item.outcome for item in results] == [agent_skills.KEPT]
    assert list(elsewhere.iterdir()) == []


def test_a_first_install_that_fails_part_way_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(agent_skills.shutil, "copyfile", broken)
    assert cli.main(["skills", "install", str(tmp_path)]) == 1
    assert "disk full" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []

    monkeypatch.undo()
    assert cli.main(["skills", "install", str(tmp_path)]) == 0
    assert "installed houdini-artist" in capsys.readouterr().out


def test_install_refuses_a_file_as_the_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "not-a-folder"
    target.write_text("x\n", encoding="utf-8")
    assert cli.main(["skills", "install", str(target)]) == 1
    assert "not a folder" in capsys.readouterr().out
