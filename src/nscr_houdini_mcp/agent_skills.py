"""The agent skills that ship with this package, and copying them out.

A skill is a folder with a `SKILL.md` in it. An installed copy carries the
folders inside the package and a checkout has them at the top of the tree, the
same way the Houdini payload travels. Which folder a client reads skills from
is the client's own business, so nothing here guesses one: `install` copies
into the folder it is given and nowhere else.

The skills are meant to be edited, so a copy already in place that differs
from the shipped one is left alone unless the caller says to write over it.
"""

from __future__ import annotations

import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path

SKILL_FILE = "SKILL.md"
SKILLS_DIR_NAME = "skills"

INSTALLED = "installed"
REPLACED = "replaced"
UNCHANGED = "unchanged"
KEPT = "kept"


class SkillsError(RuntimeError):
    """The skills could not be found or copied."""


@dataclass(frozen=True)
class Copied:
    """What happened to one skill on its way into the named folder."""

    name: str
    path: Path
    outcome: str
    note: str = ""


def skills_root() -> Path:
    """The folder holding the shipped skills, one folder per skill.

    An installed copy has it inside the package, and that one is preferred. A
    checkout has it at the top of the tree instead.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent / SKILLS_DIR_NAME, here.parents[2] / SKILLS_DIR_NAME):
        if candidate.is_dir() and any(candidate.glob(f"*/{SKILL_FILE}")):
            return candidate
    raise SkillsError("no skills folder next to this package, so there is nothing to copy")


def shipped_skills(root: Path | None = None) -> list[Path]:
    """Every skill folder under the root, in name order."""
    base = root if root is not None else skills_root()
    return sorted(path.parent for path in base.glob(f"*/{SKILL_FILE}"))


def _files(folder: Path) -> list[Path]:
    """The files under a folder, relative to it, skipping caches and links."""
    return sorted(
        path.relative_to(folder)
        for path in folder.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
    )


def _same(source: Path, target: Path) -> bool:
    """Whether every shipped file is in the target with the same bytes."""
    for rel in _files(source):
        there = target / rel
        if not there.is_file() or not filecmp.cmp(source / rel, there, shallow=False):
            return False
    return True


def _copy(source: Path, target: Path) -> None:
    """Write the shipped files into the target, leaving anything else there."""
    for rel in _files(source):
        there = target / rel
        there.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, there)


def install(dest: Path, *, force: bool = False, root: Path | None = None) -> list[Copied]:
    """Copy each shipped skill into `dest/<skill name>`.

    A skill that is not there yet is copied. One that is there with the same
    files is left as it is. One that differs, which usually means someone
    edited it, is kept unless `force`, and with `force` only the shipped files
    are written over, so a file of the person's own beside them stays. A link
    or a plain file where a skill folder should be is never touched.
    """
    dest = Path(dest).expanduser().absolute()
    if dest.exists() and not dest.is_dir():
        raise SkillsError(f"{dest} is there and is not a folder")
    skills = shipped_skills(root)
    if not skills:
        raise SkillsError("the skills folder has no skills in it")
    dest.mkdir(parents=True, exist_ok=True)

    results: list[Copied] = []
    for source in skills:
        target = dest / source.name
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            note = "a link or a file of that name is there, left as it is"
            results.append(Copied(source.name, target, KEPT, note))
            continue
        if not target.exists():
            _copy(source, target)
            results.append(Copied(source.name, target, INSTALLED))
            continue
        if _same(source, target):
            results.append(Copied(source.name, target, UNCHANGED))
            continue
        if not force:
            note = "differs from the shipped copy, pass --force to write the shipped files over it"
            results.append(Copied(source.name, target, KEPT, note))
            continue
        _copy(source, target)
        results.append(Copied(source.name, target, REPLACED))
    return results
