"""Managed output paths, built from a token table instead of by hand.

No tool takes an output path. The caller names a kind and a name, and this
module answers with two strings for the same file: the template that goes into
a parameter, which keeps `$HIP`, `${OS}` and `$F4` unexpanded so a scene still
works on another machine, and the expanded path for the run, which the server
uses and freezes in the run record. Freezing matters: a node renamed after a
run started must not move that run's files.

The grammar is data. Built in defaults come first, then a per user file, then a
file beside the scene, each one winning key by key over the one before it. A
template is a string of `<token>` pieces and plain text, so a studio changes a
layout by editing one line rather than by changing code.

Version numbers come from the coordination store, inside its transaction, and
the version folder is then created with an exclusive `mkdir`. The number is the
agreement between processes on one machine and the folder is the last guard
when a scene folder is shared. Agent artifacts that are not versioned carry the
run id in the file name, so two captures in the same second are two files.

This module never imports `hou` and never touches Houdini.
"""

from __future__ import annotations

import json
import re
import secrets
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from . import store as store_module
from .store import Store, write_export

# Kinds the table knows. A kind that is not here has no grammar, so the caller
# gets an error rather than a path invented on the spot.
OUTPUT_KINDS = (
    "render",
    "flipbook",
    "comp",
    "cache",
    "usd",
    "hip",
    "capture",
    "compare",
)

# Every token a template may use. Anything else fails on load, so a typo is a
# clear message and not a literal `<nmae>` in a file name.
TOKENS = (
    "output_root",
    "cache_root",
    "producer",
    "kind",
    "name",
    "ver",
    "date",
    "date_iso",
    "time",
    "run_id",
    "session",
    "hipname",
    "hipfamily",
    "frame",
    "ext",
)

# Renders, flipbooks and comps are dated: a person browses them by the day they
# were made. Caches, USD layers and hip files are stable under the name, because
# the scene reads them back and only the version should change between runs.
# Agent artifacts live under `.agent` and carry the run id.
DEFAULT_GRAMMAR = {
    "render": "<output_root>/renders/<producer>/<date>_<name>/v<ver>/<name>_v<ver>.<frame>.<ext>",
    "flipbook": (
        "<output_root>/flipbook/<producer>/<date>_<name>/v<ver>/<name>_v<ver>.<frame>.<ext>"
    ),
    "comp": "<output_root>/comp/<producer>/<date>_<name>/v<ver>/<name>_v<ver>.<frame>.<ext>",
    "cache": "<cache_root>/geo/<producer>/<name>/v<ver>/<name>_v<ver>.<frame>.<ext>",
    "usd": "<output_root>/usd/<producer>/<name>/v<ver>/<name>_v<ver>.<ext>",
    "hip": "<output_root>/<name>_v<ver>.<ext>",
    "capture": "<output_root>/.agent/captures/<date>/<time>_<name>_<run_id>.<ext>",
    "compare": "<output_root>/.agent/compare/<date>_<name>/<ver>_<run_id>/",
}

DEFAULT_EXTENSIONS = {
    "render": "exr",
    "flipbook": "png",
    "comp": "exr",
    "cache": "bgeo.sc",
    "usd": "usd",
    "hip": "hip",
    "capture": "png",
    "compare": "",
}

# Roots are templates too, so a studio can point a kind somewhere else without
# touching the rest of the line. Empty `cache_root` means the output root.
DEFAULT_OUTPUTS = {
    "output_root": "$HIP",
    "cache_root": "",
    "producer": "",
    "version_width": 3,
    "frame_token": "$F4",
}

# Which node marks an output. It is config because a studio that marks outputs
# with something else should not have to change code, and no API names a type.
DEFAULT_CONVENTIONS = {
    "output_marker_type": "null",
    "output_marker_prefix": "OUT_",
    "output_marker_enabled": True,
}

USER_FILE_NAMES = ("config.toml", "config.json")
PROJECT_FILE_NAMES = (".agent/outputs.toml", ".agent/outputs.json")
PROJECT_TABLES = ("outputs", "conventions")

SIDECAR_NAME = "_run.json"
SCRATCH_DIR_NAME = "scratch"
UNTITLED_FAMILY = "untitled"

# A name that already carries something a version tool would read as a version.
VERSION_IN_NAME = re.compile(r"(?<![A-Za-z0-9])v\d+", re.IGNORECASE)
_TOKEN = re.compile(r"<([a-z_]+)>")
_ILLEGAL_IN_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_TRAILING_VERSION = re.compile(r"[._-]v\d+$", re.IGNORECASE)
_DRIVE = re.compile(r"^[A-Za-z]:")

MKDIR_ATTEMPTS = 8
RUN_ID_BYTES = 6

# Stands in for the version while the shape of a line is worked out. It is not
# a legal name character, so it can never come from a name or a date.
_VERSION_MARK = "\x00ver\x00"


class OutputError(Exception):
    """Base class for output path failures."""


class UnknownKind(OutputError):
    """A kind the table has no grammar for."""


class ConventionError(OutputError):
    """A conventions file that cannot be used as written."""


class AllocationFailed(OutputError):
    """A version folder could not be claimed after several tries."""


def new_run_id() -> str:
    """A fresh run id. Short, because it ends up in file names."""
    return f"run-{secrets.token_hex(RUN_ID_BYTES)}"


def sanitize_name(text: str) -> str:
    """Name reduced to letters, digits, underscore and dash.

    Anything else becomes an underscore, because these strings end up in file
    names on three systems and in Houdini parameters.
    """
    cleaned = _ILLEGAL_IN_NAME.sub("_", text.strip())
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_")
    return cleaned or "output"


def split_hip(hip_path: str | Path) -> tuple[str, str]:
    """Scene folder and scene file stem, whatever system wrote the path.

    A path from a Windows machine can be read on macOS or Linux, so the
    separator decides how it is split rather than the system running the code.
    The folder comes back with forward slashes, which Houdini accepts
    everywhere.
    """
    text = str(hip_path).strip()
    if not text:
        raise OutputError("hip path is empty")
    windows = "\\" in text or bool(_DRIVE.match(text))
    pure = PureWindowsPath(text) if windows else PurePosixPath(text)
    parent = pure.parent.as_posix()
    if windows:
        parent = parent.replace("\\", "/")
    stem = pure.name
    for suffix in (".hipnc", ".hiplc", ".hip"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return parent, stem


def hip_family(hip_path: str | Path | None) -> str:
    """Scene name without its version, which is what records are scoped to.

    Two scenes in one folder must not share records, and `shot_v003` and
    `shot_v004` are the same scene at two moments.
    """
    if hip_path is None:
        return UNTITLED_FAMILY
    _, stem = split_hip(hip_path)
    return sanitize_name(_TRAILING_VERSION.sub("", stem)) if stem else UNTITLED_FAMILY


@dataclass(frozen=True)
class Conventions:
    """The token table in force, and where each part of it came from."""

    grammar: Mapping[str, str]
    extensions: Mapping[str, str]
    output_root: str
    cache_root: str
    producer: str
    version_width: int
    frame_token: str
    output_marker_type: str
    output_marker_prefix: str
    output_marker_enabled: bool
    sources: tuple[str, ...] = ()

    def template_for(self, kind: str) -> str:
        """Grammar line for a kind."""
        try:
            return self.grammar[kind]
        except KeyError:
            raise UnknownKind(f"no grammar for kind {kind!r}") from None

    def extension_for(self, kind: str) -> str:
        """Default extension for a kind, without its dot."""
        return self.extensions.get(kind, "")

    def root_for(self, kind: str) -> str:
        """Root template a kind writes under."""
        if kind == "cache" and self.cache_root:
            return self.cache_root
        return self.output_root

    def marker_name(self, name: str) -> str:
        """Readable name for an output marker node."""
        return f"{self.output_marker_prefix}{sanitize_name(name)}"

    def is_versioned(self, kind: str) -> bool:
        """Whether this kind's grammar asks for a version number."""
        return "<ver>" in self.template_for(kind)


DEFAULT_CONVENTIONS_TABLE = Conventions(
    grammar=dict(DEFAULT_GRAMMAR),
    extensions=dict(DEFAULT_EXTENSIONS),
    output_root=str(DEFAULT_OUTPUTS["output_root"]),
    cache_root=str(DEFAULT_OUTPUTS["cache_root"]),
    producer=str(DEFAULT_OUTPUTS["producer"]),
    version_width=int(DEFAULT_OUTPUTS["version_width"]),
    frame_token=str(DEFAULT_OUTPUTS["frame_token"]),
    output_marker_type=str(DEFAULT_CONVENTIONS["output_marker_type"]),
    output_marker_prefix=str(DEFAULT_CONVENTIONS["output_marker_prefix"]),
    output_marker_enabled=bool(DEFAULT_CONVENTIONS["output_marker_enabled"]),
)


@dataclass(frozen=True)
class OutputPlan:
    """One managed output, as a template and as a frozen path."""

    kind: str
    name: str
    run_id: str
    template: str
    path: str
    directory: str
    sidecar: str
    hip_family: str
    version: int | None = None
    version_dir: str | None = None
    session_id: str | None = None
    source_node: str | None = None
    unsaved_hip: bool = False
    is_directory: bool = False
    warnings: tuple[str, ...] = ()
    tokens: Mapping[str, str] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """The path part of a run record, readable on its own."""
        return {
            "template": self.template,
            "path": self.path,
            "directory": self.directory,
            "sidecar": self.sidecar,
            "version_dir": self.version_dir,
            "is_directory": self.is_directory,
            "unsaved_hip": self.unsaved_hip,
            "warnings": list(self.warnings),
        }


# -- conventions files ----------------------------------------------------


def load_conventions(
    *,
    home: Path | str | None = None,
    hip_path: str | Path | None = None,
) -> Conventions:
    """Built in defaults, then the per user file, then the file by the scene.

    Later wins, key by key, so a project file that sets one line leaves the
    rest of the table alone.
    """
    root = Path(home) if home is not None else store_module.default_home()
    layers: list[tuple[Path, dict[str, Any]]] = []
    user = _first_existing(root, USER_FILE_NAMES)
    if user is not None:
        layers.append((user, _read_table(user)))
    if hip_path is not None:
        hip_dir, _ = split_hip(hip_path)
        project = _first_existing(Path(hip_dir), PROJECT_FILE_NAMES)
        if project is not None:
            table = _read_table(project)
            extra = sorted(set(table) - set(PROJECT_TABLES))
            if extra:
                raise ConventionError(
                    f"{project}: a project file may only hold "
                    f"{' and '.join(PROJECT_TABLES)}, found {', '.join(extra)}"
                )
            layers.append((project, table))
    return _merge(layers)


def _first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root.joinpath(*name.split("/"))
        if candidate.is_file():
            return candidate
    return None


def _read_table(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if path.suffix == ".json":
            loaded = json.loads(raw.decode("utf-8"))
        else:
            loaded = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConventionError(f"{path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConventionError(f"{path}: the file must hold a table")
    return loaded


def _merge(layers: list[tuple[Path, dict[str, Any]]]) -> Conventions:
    grammar = dict(DEFAULT_GRAMMAR)
    extensions = dict(DEFAULT_EXTENSIONS)
    outputs = dict(DEFAULT_OUTPUTS)
    conventions = dict(DEFAULT_CONVENTIONS)
    sources: list[str] = []

    for path, table in layers:
        outputs_table = _sub_table(path, table, "outputs")
        conventions_table = _sub_table(path, table, "conventions")
        grammar.update(_kind_map(path, outputs_table.pop("grammar", {}), "grammar"))
        extensions.update(_kind_map(path, outputs_table.pop("extensions", {}), "extensions"))
        _unknown_keys(path, "outputs", outputs_table, DEFAULT_OUTPUTS)
        _unknown_keys(path, "conventions", conventions_table, DEFAULT_CONVENTIONS)
        outputs.update(outputs_table)
        conventions.update(conventions_table)
        sources.append(str(path))

    width = outputs["version_width"]
    if not isinstance(width, int) or isinstance(width, bool) or not 1 <= width <= 8:
        raise ConventionError("outputs.version_width must be a whole number from 1 to 8")
    enabled = conventions["output_marker_enabled"]
    if not isinstance(enabled, bool):
        raise ConventionError("conventions.output_marker_enabled must be true or false")
    for key in ("output_root", "cache_root", "producer", "frame_token"):
        _needs_text(f"outputs.{key}", outputs[key])
    for key in ("output_marker_type", "output_marker_prefix"):
        _needs_text(f"conventions.{key}", conventions[key])
    if not str(conventions["output_marker_type"]).strip():
        raise ConventionError("conventions.output_marker_type must name a node type")
    for kind, template in grammar.items():
        _check_template(kind, template)
    for key in ("output_root", "cache_root"):
        _check_root(key, str(outputs[key]))

    return Conventions(
        grammar=grammar,
        extensions={kind: str(value).lstrip(".") for kind, value in extensions.items()},
        output_root=str(outputs["output_root"]) or "$HIP",
        cache_root=str(outputs["cache_root"]),
        producer=str(outputs["producer"]).strip("/"),
        version_width=width,
        frame_token=str(outputs["frame_token"]),
        output_marker_type=str(conventions["output_marker_type"]),
        output_marker_prefix=str(conventions["output_marker_prefix"]),
        output_marker_enabled=enabled,
        sources=tuple(sources),
    )


def _sub_table(path: Path, table: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = table.get(name, {})
    if not isinstance(value, dict):
        raise ConventionError(f"{path}: [{name}] must be a table")
    return dict(value)


def _kind_map(path: Path, value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConventionError(f"{path}: [outputs.{name}] must be a table")
    unknown = sorted(set(value) - set(OUTPUT_KINDS))
    if unknown:
        raise ConventionError(f"{path}: outputs.{name} names unknown kinds: {', '.join(unknown)}")
    for kind, entry in value.items():
        if not isinstance(entry, str):
            raise ConventionError(f"{path}: outputs.{name}.{kind} must be text")
    return {kind: str(entry) for kind, entry in value.items()}


def _unknown_keys(
    path: Path, name: str, table: Mapping[str, Any], known: Mapping[str, Any]
) -> None:
    unknown = sorted(set(table) - set(known))
    if unknown:
        raise ConventionError(f"{path}: [{name}] has unknown keys: {', '.join(unknown)}")


def _needs_text(label: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ConventionError(f"{label} must be text")


def _check_template(kind: str, template: str) -> None:
    if kind not in OUTPUT_KINDS:
        raise ConventionError(f"no such kind: {kind}")
    if not isinstance(template, str) or not template.strip():
        raise ConventionError(f"the {kind} template is empty")
    if "\\" in template:
        raise ConventionError(f"the {kind} template must use forward slashes")
    if template.startswith("/") or _DRIVE.match(template):
        raise ConventionError(f"the {kind} template must not start at a drive or a root")
    unknown = sorted({name for name in _TOKEN.findall(template) if name not in TOKENS})
    if unknown:
        raise ConventionError(f"the {kind} template uses unknown tokens: {', '.join(unknown)}")


def _check_root(label: str, root: str) -> None:
    if "\\" in root:
        raise ConventionError(f"outputs.{label} must use forward slashes")
    if _DRIVE.match(root):
        raise ConventionError(f"outputs.{label} must not name a drive, so scenes stay portable")


# -- building paths -------------------------------------------------------


def plan_path(
    kind: str,
    *,
    run_id: str,
    name: str | None = None,
    node_name: str | None = None,
    hip_path: str | Path | None = None,
    session_id: str | None = None,
    version: int | None = None,
    ext: str | None = None,
    conventions: Conventions | None = None,
    when: datetime | None = None,
    scratch_root: str | Path | None = None,
) -> OutputPlan:
    """Work out both strings for one output, without touching the disk.

    The template keeps Houdini variables, and uses `${OS}` for the name when the
    name is the node's own, so a rename carries through to the next run. The
    path is the same line with the scene folder and the name filled in, which is
    what the run record freezes.
    """
    table = conventions or DEFAULT_CONVENTIONS_TABLE
    template = table.template_for(kind)
    versioned = "<ver>" in template
    if versioned and version is None:
        raise OutputError(f"{kind} needs a version number")

    warnings: list[str] = []
    chosen = sanitize_name(name or node_name or kind)
    from_node = node_name is not None and chosen == sanitize_name(node_name)
    if VERSION_IN_NAME.search(chosen):
        warnings.append(
            f"the name {chosen} reads as if it already holds a version, "
            "which confuses tools that step versions by the path"
        )

    unsaved = hip_path is None
    if unsaved:
        root_dir = _scratch_dir(scratch_root, session_id)
        warnings.append("the scene has not been saved, so this run goes to a scratch folder")
        hip_stem = UNTITLED_FAMILY
    else:
        root_dir, hip_stem = split_hip(hip_path)

    family = hip_family(hip_path)
    moment = when or datetime.now()
    common = {
        "kind": kind,
        "producer": table.producer,
        "date": moment.strftime("%Y%m%d"),
        "date_iso": moment.strftime("%Y-%m-%d"),
        "time": moment.strftime("%H%M%S"),
        "ver": "" if version is None else f"{version:0{table.version_width}d}",
        "run_id": run_id,
        "session": session_id or "",
        "hipname": hip_stem,
        "hipfamily": family,
        "frame": table.frame_token,
        "ext": (ext if ext is not None else table.extension_for(kind)).lstrip("."),
    }
    root_template = table.root_for(kind)
    common["output_root"] = root_template
    common["cache_root"] = root_template
    parm_tokens = dict(common, name="${OS}" if from_node else chosen)
    frozen_tokens = dict(common, name=chosen)

    parm_path = _fill(template, parm_tokens)
    frozen = expand(_fill(template, frozen_tokens), hip_dir=root_dir)
    if unsaved:
        # Nothing is stored in a scene that has no folder yet, so the scratch
        # root replaces `$HIP` in both strings rather than only in the frozen one.
        parm_path = expand(parm_path, hip_dir=root_dir)

    is_directory = template.rstrip().endswith("/")
    directory = frozen.rstrip("/") if is_directory else _parent(frozen)
    drop = _version_drop(template, frozen_tokens, is_directory) if versioned else None
    version_dir = None
    if drop is not None:
        parts = frozen.rstrip("/").split("/")
        version_dir = "/".join(parts[: len(parts) - drop])
    sidecar = _sidecar(frozen, directory, version_dir, is_directory, common["ext"])

    return OutputPlan(
        kind=kind,
        name=chosen,
        run_id=run_id,
        template=parm_path,
        path=frozen,
        directory=directory,
        sidecar=sidecar,
        hip_family=family,
        version=version,
        version_dir=version_dir,
        session_id=session_id,
        unsaved_hip=unsaved,
        is_directory=is_directory,
        warnings=tuple(warnings),
        tokens=frozen_tokens,
    )


def expand(text: str, *, hip_dir: str | Path, frame: int | None = None) -> str:
    """Fill in the Houdini variables this server owns, and nothing else.

    Separators are settled here and nowhere earlier, so the stored template is
    the same text on every system. Forward slashes are kept, because Houdini
    and Python both read them on Windows.
    """
    root = str(hip_dir).replace("\\", "/").rstrip("/")
    filled = text.replace("${HIP}", root).replace("$HIP", root)
    if frame is not None:
        filled = re.sub(
            r"\$F(\d*)", lambda match: str(frame).zfill(int(match.group(1) or 1)), filled
        )
    return filled


def _fill(template: str, tokens: Mapping[str, str]) -> str:
    """Replace every token, then drop the segments that came out empty."""

    def one(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in tokens:
            raise OutputError(f"unknown token <{name}>")
        return tokens[name]

    filled = _TOKEN.sub(one, template)
    trailing = filled.endswith("/")
    parts = [part for part in filled.split("/") if part not in ("", ".")]
    joined = "/".join(parts)
    # A name ending in a dot means an empty extension token, which is a kind
    # that writes a folder rather than a file.
    joined = joined.rstrip(".")
    return f"{joined}/" if trailing else joined


def _parent(path: str) -> str:
    head, _, _ = path.rstrip("/").rpartition("/")
    return head or "."


def _version_drop(template: str, tokens: Mapping[str, str], is_dir: bool) -> int | None:
    """How many trailing segments sit below the folder that holds the version.

    The answer is counted from the end, because the scene folder in front of
    the line has a length of its own.
    """
    marked = _fill(template, dict(tokens, ver=_VERSION_MARK))
    parts = marked.rstrip("/").split("/")
    start = len(parts) - 1 if is_dir else len(parts) - 2
    for index in range(start, -1, -1):
        if _VERSION_MARK in parts[index]:
            return len(parts) - 1 - index
    return None


def _sidecar(frozen: str, directory: str, version_dir: str | None, is_dir: bool, ext: str) -> str:
    """Where the readable run record goes for this output."""
    if is_dir:
        return f"{frozen.rstrip('/')}/{SIDECAR_NAME}"
    if version_dir:
        return f"{version_dir}/{SIDECAR_NAME}"
    leaf = frozen.rsplit("/", 1)[-1]
    if ext and leaf.endswith(f".{ext}"):
        leaf = leaf[: -(len(ext) + 1)]
    return f"{directory}/{leaf}{SIDECAR_NAME}"


def _scratch_dir(scratch_root: str | Path | None, session_id: str | None) -> str:
    root = Path(scratch_root) if scratch_root is not None else store_module.default_home()
    folder = root / SCRATCH_DIR_NAME / sanitize_name(session_id or "session")
    return folder.as_posix()


# -- allocation -----------------------------------------------------------


def allocate(
    store: Store,
    kind: str,
    *,
    name: str | None = None,
    hip_path: str | Path | None = None,
    node_path: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
    ext: str | None = None,
    conventions: Conventions | None = None,
    when: datetime | None = None,
    scratch_root: str | Path | None = None,
) -> OutputPlan:
    """Take a version, claim its folder, record the run and write the sidecar.

    The number comes out of the store's transaction and the folder is then
    created with an exclusive `mkdir`. A folder that is already there means
    another machine wrote it, so the next number is taken instead. What comes
    back is frozen: a later rename of the node changes the next run, never this
    one.
    """
    table = conventions or DEFAULT_CONVENTIONS_TABLE
    table.template_for(kind)
    run = run_id or new_run_id()
    node_name = node_path.rstrip("/").rsplit("/", 1)[-1] if node_path else None
    if name is None and node_name is None and kind == "hip":
        name = hip_family(hip_path)

    plan = _claim(
        store,
        kind,
        table=table,
        name=name,
        node_name=node_name,
        hip_path=hip_path,
        session_id=session_id,
        run=run,
        ext=ext,
        when=when,
        scratch_root=scratch_root,
    )

    store.create_run(
        run,
        kind=kind,
        name=plan.name,
        hip_family=plan.hip_family,
        version=plan.version,
        session_id=session_id,
        source_node=node_path,
        job_id=job_id,
        paths=plan.as_record(),
        scene={
            "hip_path": None if hip_path is None else str(hip_path),
            "unsaved_hip": plan.unsaved_hip,
        },
    )
    if plan.version is not None:
        store.attach_version_run(
            kind=kind,
            name=plan.name,
            hip_family=plan.hip_family,
            version=plan.version,
            run_id=run,
        )
    write_export(store.run_export(run), plan.sidecar)
    return replace(plan, source_node=node_path)


def _claim(
    store: Store,
    kind: str,
    *,
    table: Conventions,
    name: str | None,
    node_name: str | None,
    hip_path: str | Path | None,
    session_id: str | None,
    run: str,
    ext: str | None,
    when: datetime | None,
    scratch_root: str | Path | None,
) -> OutputPlan:
    """One plan whose folder on disk is this run's, and nobody else's."""
    versioned = table.is_versioned(kind)
    for _ in range(MKDIR_ATTEMPTS if versioned else 1):
        version = None
        if versioned:
            probe = plan_path(
                kind,
                run_id=run,
                name=name,
                node_name=node_name,
                hip_path=hip_path,
                session_id=session_id,
                version=1,
                ext=ext,
                conventions=table,
                when=when,
                scratch_root=scratch_root,
            )
            version = store.allocate_version(
                kind=kind, name=probe.name, hip_family=probe.hip_family, run_id=run
            )
        plan = plan_path(
            kind,
            run_id=run,
            name=name,
            node_name=node_name,
            hip_path=hip_path,
            session_id=session_id,
            version=version,
            ext=ext,
            conventions=table,
            when=when,
            scratch_root=scratch_root,
        )
        if _make_room(plan):
            return plan
    raise AllocationFailed(f"could not claim a folder for {kind} after {MKDIR_ATTEMPTS} tries")


def _make_room(plan: OutputPlan) -> bool:
    """Create this run's folder, and say whether it was ours to create."""
    exclusive = plan.version_dir or (plan.directory if plan.is_directory else None)
    if exclusive is None:
        Path(plan.directory).mkdir(parents=True, exist_ok=True)
        # A versioned line without a folder of its own, a hip file among them,
        # is guarded by the file instead.
        return plan.version is None or not Path(plan.path).exists()
    try:
        Path(exclusive).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False
    Path(plan.directory).mkdir(parents=True, exist_ok=True)
    return True
