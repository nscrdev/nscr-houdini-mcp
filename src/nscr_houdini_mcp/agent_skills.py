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
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

SKILL_FILE = "SKILL.md"
SKILLS_DIR_NAME = "skills"

INSTALLED = "installed"
REPLACED = "replaced"
UNCHANGED = "unchanged"
KEPT = "kept"

# The attribute Windows sets on a junction or any other reparse point. The
# `stat` module has it only on Windows, and the value is fixed.
REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


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


def is_link(path: Path) -> bool:
    """Whether a path is a link of any kind, without following it.

    A symbolic link counts, and so does a Windows junction, which
    `Path.is_symlink` does not see before Python 3.12. Any reparse point
    counts on Windows, since writing through one lands somewhere else.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    if attributes & REPARSE_POINT:
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction is not None and isjunction(path))


def _link_inside(folder: Path) -> Path | None:
    """The first link at or under a folder, never following one, or None."""
    if is_link(folder):
        return folder
    pending = [folder]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                if is_link(path):
                    return path
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
    return None


def _files(folder: Path) -> list[Path]:
    """The files under a folder, relative to it, skipping caches and links."""
    return sorted(
        path.relative_to(folder)
        for path in folder.rglob("*")
        if path.is_file()
        and not is_link(path)
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
    """Write the shipped files into the target, leaving anything else there.

    The caller has made sure there is no link anywhere in the target, so no
    write here can land outside it.
    """
    for rel in _files(source):
        there = target / rel
        there.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, there)


def _umask() -> int:
    """The process umask. Reading it means setting it, so it is put straight back."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def _open_up(folder: Path) -> None:
    """Give a staged folder the permissions a plain copy would have had.

    A temporary folder is made for its owner only, so without this a fresh
    install would leave a skill other accounts cannot read. Folders get 0o777
    and files 0o666, each masked by the umask, as `mkdir` and `open` would.
    """
    mask = _umask()
    os.chmod(folder, 0o777 & ~mask)
    for path in folder.rglob("*"):
        if is_link(path):
            continue
        os.chmod(path, (0o777 if path.is_dir() else 0o666) & ~mask)


def _copy_new(source: Path, target: Path) -> None:
    """Copy a skill that is not there yet, whole or not at all.

    The files go into a temporary folder beside the target, which is renamed
    into place once every file is in it. A copy that fails part way leaves
    nothing at the target, so the next run does not mistake it for an edit.
    """
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        _copy(source, staging)
        _open_up(staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def install(dest: Path, *, force: bool = False, root: Path | None = None) -> list[Copied]:
    """Copy each shipped skill into `dest/<skill name>`.

    A skill that is not there yet is copied whole, through a temporary folder.
    One that is there with the same files is left as it is. One that differs,
    which usually means someone edited it, is kept unless `force`, and with
    `force` only the shipped files are written over, so a file of the person's
    own beside them stays. A plain file where a skill folder should be, or a
    link anywhere at or under the skill folder, is never written through, with
    or without `force`.

    Raises `SkillsError` when there is nothing to copy or the folder is not a
    folder, and `OSError` when a copy fails. Keeping an edited skill is not a
    failure.
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
        if is_link(target) or (target.exists() and not target.is_dir()):
            note = "a link or a file of that name is there, left as it is"
            results.append(Copied(source.name, target, KEPT, note))
            continue
        if not target.exists():
            _copy_new(source, target)
            results.append(Copied(source.name, target, INSTALLED))
            continue
        link = _link_inside(target)
        if link is not None:
            note = f"holds a link, {link}, so nothing is written into it"
            results.append(Copied(source.name, target, KEPT, note))
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
