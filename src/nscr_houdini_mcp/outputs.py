"""Managed output paths, built from a token table instead of by hand.

No tool takes an output path. The caller names a kind and a name, and this
module answers with three strings for the same output: the template, which
keeps `$HIP`, `${OS}` and `$F4` and is what a person reads and edits; the
parameter value for this run, which is the expanded path, because the server
owns a run that has started and a node renamed halfway through must not move
its files; and the same path in the run record, so the record and the scene
agree.

The grammar is data. Built in defaults come first, then a per user file, then a
file beside the scene, each one winning key by key over the one before it. A
template is a string of `<token>` pieces and plain text, so a studio changes a
layout by editing one line rather than by changing code.

Nothing a file or a caller says may leave the output root. Templates, roots,
producer levels and extensions are checked when they are read, and the finished
path is checked against the root before anything is created on disk. A root has
to start at a Houdini variable, so no machine path is ever written into a scene.

Version numbers come from the coordination store, inside its transaction, and
the version folder is then created with an exclusive `mkdir`. The number is the
agreement between processes on one machine and the folder is the last guard
when a scene folder is shared. A kind with no folder of its own claims an
exclusive file beside its output instead. Agent artifacts that are not
versioned carry the run id in the file name, so two captures in the same second
are two files.

This module never imports `hou` and never touches Houdini.
"""

from __future__ import annotations

import hashlib
import json
import os
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

# Houdini variables a template may hold, and the only ones this server fills
# in. `$HIP` and `$HOUDINI_TEMP_DIR` come from the session, `$JOB` from the
# environment. Anything else is refused when the table is read, so no run ever
# creates a folder named after a variable nobody expanded.
ALLOWED_VARIABLES = ("HIP", "HOUDINI_TEMP_DIR", "JOB")

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
CLAIM_SUFFIX = ".claim"
SCRATCH_ROOT = f"$HOUDINI_TEMP_DIR/{store_module.APP_DIR_NAME}"
UNTITLED_FAMILY = "untitled"

# A name that already carries something a version tool would read as a version.
VERSION_IN_NAME = re.compile(r"(?<![A-Za-z0-9])v\d+", re.IGNORECASE)

# Names Windows keeps for devices, whatever the extension is.
WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{digit}" for digit in range(1, 10)}
    | {f"LPT{digit}" for digit in range(1, 10)}
)

_TOKEN = re.compile(r"<([a-z_]+)>")
_VARIABLE = re.compile(r"\$\{(\w+)\}|\$(\w+)")
_FRAME_VARIABLE = re.compile(r"^F\d*$")
_ILLEGAL_IN_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_TRAILING_VERSION = re.compile(r"[._-]v\d+$", re.IGNORECASE)
_OWN_VERSION = re.compile(r"[._-]v(\d+)$", re.IGNORECASE)
_DRIVE = re.compile(r"^[A-Za-z]:")
_ROOT_START = re.compile(rf"^\$\{{?({'|'.join(ALLOWED_VARIABLES)})\}}?(/|$)")
_EXTENSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

MKDIR_ATTEMPTS = 16
RUN_ID_BYTES = 6
NAME_HASH_LENGTH = 6

# Stands in for the version while the shape of a line is worked out. It is not
# a legal name character, so it can never come from a name or a date.
_VERSION_MARK = "\x00ver\x00"


class OutputError(Exception):
    """Base class for output path failures."""


class UnknownKind(OutputError):
    """A kind the table has no grammar for."""


class ConventionError(OutputError):
    """A conventions file, or a path built from one, that cannot be used."""


class AllocationFailed(OutputError):
    """A version could not be claimed after several tries."""


def new_run_id() -> str:
    """A fresh run id. Short, because it ends up in file names."""
    return f"run-{secrets.token_hex(RUN_ID_BYTES)}"


def sanitize_name(text: str) -> str:
    """Name reduced to letters, digits, underscore and dash.

    Anything else becomes an underscore, because these strings end up in file
    names on three systems and in Houdini parameters. A name written in another
    script has nothing left after that, so a short hash of the original is
    added and two such names stay two names. Device names Windows keeps for
    itself get a suffix, because a file cannot have one.
    """
    raw = text.strip()
    cleaned = _ILLEGAL_IN_NAME.sub("_", raw)
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_")
    if raw and not cleaned:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:NAME_HASH_LENGTH]
        cleaned = f"output_{digest}"
    cleaned = cleaned or "output"
    if cleaned.split(".")[0].upper() in WINDOWS_RESERVED:
        cleaned = f"{cleaned}_out"
    return cleaned


def clean_extension(text: str) -> str:
    """Extension text an output may end in, without its leading dot.

    An extension arrives from a caller as well as from a file, so it is checked
    rather than trusted: letters, digits, dot, dash and underscore, and nothing
    that could step out of the folder.
    """
    value = str(text).strip()
    if value.startswith("."):
        value = value[1:]
    if not value:
        return ""
    if not _EXTENSION.match(value) or ".." in value:
        raise ConventionError(f"{text!r} is not an extension this can write")
    return value


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


def hip_version_floor(hip_path: str | Path | None) -> int:
    """The version a scene's own name carries, such as 3 for `shot_v003.hip`.

    The files already beside the output are read when the version is claimed,
    from the folder the output goes to, which need not be the scene's own.
    """
    if hip_path is None:
        return 0
    _, stem = split_hip(hip_path)
    own = _OWN_VERSION.search(stem)
    return int(own.group(1)) if own else 0


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
    """One managed output: the line a person reads, and the run's own path."""

    kind: str
    name: str
    run_id: str
    template: str
    parm: str
    path: str
    root: str
    under_root: str
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
        """The path part of a run record, readable on its own.

        `path`, `directory` and `sidecar` are absolute on the machine that made
        the run. `under_root` is the same output written from the root, which is
        what to read when the folder is opened from somewhere else. `template`
        is the editable line, and `parm` is what the node was set to.
        """
        return {
            "template": self.template,
            "parm": self.parm,
            "path": self.path,
            "root": self.root,
            "under_root": self.under_root,
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
    rest of the table alone. The file beside the scene comes with the scene, so
    it is read as a suggestion and checked like one.
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
    _check_producer(str(outputs["producer"]))
    _check_variables("outputs.frame_token", str(outputs["frame_token"]))
    clean = {kind: clean_extension(value) for kind, value in extensions.items()}

    return Conventions(
        grammar=grammar,
        extensions=clean,
        output_root=str(outputs["output_root"]) or str(DEFAULT_OUTPUTS["output_root"]),
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
    if ".." in template.split("/"):
        raise ConventionError(f"the {kind} template must stay under its root, so no .. in it")
    unknown = sorted({name for name in _TOKEN.findall(template) if name not in TOKENS})
    if unknown:
        raise ConventionError(f"the {kind} template uses unknown tokens: {', '.join(unknown)}")
    _check_variables(f"the {kind} template", template)


def _check_root(label: str, root: str) -> None:
    if not root:
        return
    if "\\" in root:
        raise ConventionError(f"outputs.{label} must use forward slashes")
    if root.startswith("/") or _DRIVE.match(root):
        raise ConventionError(
            f"outputs.{label} must not name a drive or start at a root, so scenes stay portable"
        )
    if ".." in root.split("/"):
        raise ConventionError(f"outputs.{label} must not step out of a folder with ..")
    if not _ROOT_START.match(root):
        allowed = ", ".join(f"${name}" for name in ALLOWED_VARIABLES)
        raise ConventionError(f"outputs.{label} must start at one of {allowed}")
    _check_variables(f"outputs.{label}", root)


def _check_producer(producer: str) -> None:
    if "\\" in producer:
        raise ConventionError("outputs.producer must use forward slashes")
    if producer.startswith("/") or _DRIVE.match(producer):
        raise ConventionError("outputs.producer is a level under the root, not a root of its own")
    if ".." in producer.split("/"):
        raise ConventionError("outputs.producer must stay under the root, so no .. in it")
    _check_variables("outputs.producer", producer)


def _check_variables(label: str, text: str) -> None:
    """Every `$` in the line has to be one this server knows how to fill in."""
    for braced, plain in _VARIABLE.findall(text):
        name = braced or plain
        if name in ALLOWED_VARIABLES or name == "OS" or _FRAME_VARIABLE.match(name):
            continue
        raise ConventionError(f"{label} uses ${name}, which this does not expand")


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
    """Work out the line and the path for one output, without touching disk.

    The template keeps the Houdini variables and uses `${OS}` for a name that
    came from the node, which is the line to show a person and the line to put
    on a node that has no run yet. The parameter value for this run is the
    expanded path: the server owns a run once it has started, and a rename
    while a render is going must not send half the frames somewhere else.
    """
    table = conventions or DEFAULT_CONVENTIONS_TABLE
    template = table.template_for(kind)
    versioned = "<ver>" in template
    if versioned and version is None:
        raise OutputError(f"{kind} needs a version number")

    warnings: list[str] = []
    chosen = sanitize_name(name or node_name or kind)
    from_node = name is None and node_name is not None
    if VERSION_IN_NAME.search(chosen):
        warnings.append(
            f"the name {chosen} reads as if it already holds a version, "
            "which confuses tools that step versions by the path"
        )

    unsaved = hip_path is None
    if unsaved:
        root_template = f"{SCRATCH_ROOT}/{sanitize_name(session_id or 'session')}"
        warnings.append("the scene has not been saved, so this run goes to a scratch folder")
        hip_stem = UNTITLED_FAMILY
        hip_dir = None
    else:
        root_template = table.root_for(kind)
        hip_dir, hip_stem = split_hip(hip_path)

    family = hip_family(hip_path)
    moment = when or datetime.now()
    extension = clean_extension(ext if ext is not None else table.extension_for(kind))
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
        "ext": extension,
        "output_root": root_template,
        "cache_root": root_template,
    }
    line = _fill(template, dict(common, name="${OS}" if from_node else chosen))
    literal = _fill(template, dict(common, name=chosen))

    temp_dir = _temp_dir(scratch_root) if unsaved else None
    root = _normalize(expand(root_template, hip_dir=hip_dir, temp_dir=temp_dir))
    frozen = _normalize(expand(literal, hip_dir=hip_dir, temp_dir=temp_dir))
    if not _inside(root, frozen):
        raise ConventionError(f"{frozen} would leave the output root {root}")

    is_directory = template.rstrip().endswith("/")
    if is_directory:
        frozen = f"{frozen}/"
    directory = frozen.rstrip("/") if is_directory else _parent(frozen)
    drop = _version_drop(template, dict(common, name=chosen), is_directory) if versioned else None
    version_dir = None
    if drop is not None:
        parts = frozen.rstrip("/").split("/")
        version_dir = "/".join(parts[: len(parts) - drop])
    sidecar = _sidecar(frozen, directory, version_dir, is_directory, extension)

    return OutputPlan(
        kind=kind,
        name=chosen,
        run_id=run_id,
        template=line,
        parm=frozen,
        path=frozen,
        root=root,
        under_root=frozen[len(root) :].lstrip("/"),
        directory=directory,
        sidecar=sidecar,
        hip_family=family,
        version=version,
        version_dir=version_dir,
        session_id=session_id,
        unsaved_hip=unsaved,
        is_directory=is_directory,
        warnings=tuple(warnings),
        tokens=dict(common, name=chosen),
    )


def expand(
    text: str,
    *,
    hip_dir: str | Path | None = None,
    temp_dir: str | Path | None = None,
    job: str | Path | None = None,
    frame: int | None = None,
) -> str:
    """Fill in the Houdini variables this server owns, and nothing else.

    Only whole names are replaced, so `$HIPNAME` is left for Houdini rather
    than cut in half. A variable this server knows but cannot answer for is an
    error, because a folder named after an unexpanded variable is worse than a
    refusal. Separators are settled here and nowhere earlier, so the stored
    template is the same text on every system.
    """
    values: dict[str, str] = {}
    if hip_dir is not None:
        values["HIP"] = _posix(hip_dir)
    if temp_dir is not None:
        values["HOUDINI_TEMP_DIR"] = _posix(temp_dir)
    from_env = job if job is not None else os.environ.get("JOB")
    if from_env:
        values["JOB"] = _posix(from_env)

    def one(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name in values:
            return values[name]
        if name in ALLOWED_VARIABLES:
            raise OutputError(f"${name} has no value here, so this path cannot be worked out")
        return match.group(0)

    filled = _VARIABLE.sub(one, text)
    if frame is not None:
        filled = re.sub(
            r"\$F(\d*)", lambda match: str(frame).zfill(int(match.group(1) or 1)), filled
        )
    return filled


def _posix(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/")


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


def _normalize(path: str) -> str:
    """Collapse `.` and `..` by reading the text, never by asking the disk.

    A network path keeps its `//server/share` anchor, and nothing climbs above
    it: a `..` that would is kept, so the containment check refuses the path.
    """
    if path.startswith("//") and not path.startswith("///"):
        pieces = [part for part in path[2:].split("/") if part]
        anchor, rest = pieces[:2], pieces[2:]
        return "//" + "/".join(anchor + _collapse(rest))
    lead = "/" if path.startswith("/") else ""
    return lead + "/".join(_collapse(path.split("/")))


def _collapse(pieces: list[str]) -> list[str]:
    parts: list[str] = []
    for part in pieces:
        if part in ("", "."):
            continue
        if part == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            else:
                parts.append("..")
            continue
        parts.append(part)
    return parts


def _inside(root: str, path: str) -> bool:
    """Whether a finished path is the root or sits under it."""
    root_parts = [part for part in root.split("/") if part]
    path_parts = [part for part in path.split("/") if part]
    if ".." in path_parts or path.startswith("/") != root.startswith("/"):
        return False
    return path_parts[: len(root_parts)] == root_parts and len(path_parts) >= len(root_parts)


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


def _temp_dir(scratch_root: str | Path | None) -> str:
    """Where a scene with no folder of its own writes.

    The template keeps `$HOUDINI_TEMP_DIR`, so a scene saved later carries no
    trace of this machine. This is only the value it stands for here.
    """
    if scratch_root is not None:
        return _posix(scratch_root)
    from_env = os.environ.get("HOUDINI_TEMP_DIR")
    if from_env:
        return _posix(from_env)
    return _posix(store_module.default_home() / "temp")


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
    above: int = 0,
) -> OutputPlan:
    """Take a version, claim it on disk, record the run and write the sidecar.

    The number comes out of the store's transaction and the folder is then
    created with an exclusive `mkdir`. A folder that is already there means
    another machine wrote it, so that number loses this run's name and the next
    one is tried. What comes back is frozen: a later rename of the node changes
    the next run, never this one.

    `above` is a number the new version must be higher than, for a sequence
    that already has versions this store never handed out: a scene saved by
    hand as `_v007` goes on at `_v008`, not at `_v001`.
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
        above=above,
    )

    try:
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
        write_export(store.run_export(run), plan.sidecar)
    except BaseException:
        # The place was claimed for a run that could not be recorded, so it
        # goes back rather than staying claimed by nobody.
        release(store, plan)
        raise
    return replace(plan, source_node=node_path)


def release(store: Store, plan: OutputPlan) -> None:
    """Give back a place that was claimed and never written.

    The claim file and the record beside the output go, the run's record goes,
    and the number keeps its place in the sequence with no run on it. A file
    kind whose output is there after all is left alone. Version folders are
    left too: an empty folder costs nothing and is the guard that keeps the
    number from being written twice.
    """
    if not plan.version_dir and not plan.is_directory and Path(plan.path).exists():
        return
    for leftover in (f"{plan.path}{CLAIM_SUFFIX}", plan.sidecar):
        try:
            Path(leftover).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        store.drop_run(plan.run_id)
        if plan.version is not None:
            store.disown_version(
                kind=plan.kind, name=plan.name, hip_family=plan.hip_family, version=plan.version
            )
    except store_module.StoreError:
        pass


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
    above: int = 0,
) -> OutputPlan:
    """One plan whose place on disk is this run's, and nobody else's."""
    options = {
        "run_id": run,
        "name": name,
        "node_name": node_name,
        "hip_path": hip_path,
        "session_id": session_id,
        "ext": ext,
        "conventions": table,
        "when": when,
        "scratch_root": scratch_root,
    }
    if not table.is_versioned(kind):
        plan = plan_path(kind, version=None, **options)
        _check_real_place(plan)
        Path(plan.directory).mkdir(parents=True, exist_ok=True)
        return plan

    probe = plan_path(kind, version=1, **options)
    # Versions already in the folder the output goes to were handed out
    # somewhere this store never saw. The sequence goes on above them rather
    # than spending its tries finding each one taken.
    floor = max(above, _versions_on_disk(kind, probe, table))
    if floor > 0:
        store.skip_versions_to(
            kind=kind, name=probe.name, hip_family=probe.hip_family, version=floor
        )
    for _ in range(MKDIR_ATTEMPTS):
        version = store.allocate_version(
            kind=kind, name=probe.name, hip_family=probe.hip_family, run_id=run
        )
        plan = plan_path(kind, version=version, **options)
        if _make_room(plan):
            return plan
        # The number is taken on disk by whoever won, so it keeps its place in
        # the sequence and only loses this run's name.
        store.disown_version(
            kind=kind, name=probe.name, hip_family=probe.hip_family, version=version
        )
    raise AllocationFailed(f"could not claim a place for {kind} after {MKDIR_ATTEMPTS} tries")


def _versions_on_disk(kind: str, probe: OutputPlan, table: Conventions) -> int:
    """The highest version already in the folder a versioned output goes to.

    For a kind with a version folder, the folders beside it; for a kind whose
    version is in the file name, the files beside it. Only names this line of
    the grammar would write count. A scene file counts under any of the three
    scene suffixes, because a license can change which one is written.
    """
    marked = _fill(table.template_for(kind), dict(probe.tokens, ver=_VERSION_MARK))
    segments = marked.rstrip("/").split("/")
    if probe.version_dir:
        folder = str(Path(probe.version_dir).parent)
        below = len(probe.path.rstrip("/").split("/")) - len(probe.version_dir.split("/"))
        leaf = segments[-1 - below] if below < len(segments) else ""
        tail = ""
    else:
        folder = probe.directory
        leaf = segments[-1]
        extension = probe.tokens.get("ext", "")
        tail = ""
        if extension and leaf.endswith(f".{extension}"):
            leaf = leaf[: -(len(extension) + 1)]
            tail = r"\.(hip|hipnc|hiplc)" if kind == "hip" else re.escape(f".{extension}")
    if _VERSION_MARK not in leaf:
        return 0
    pattern = re.compile(
        "^" + r"(\d+)".join(re.escape(piece) for piece in leaf.split(_VERSION_MARK)) + tail + "$",
        re.IGNORECASE,
    )
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    highest = 0
    for found in (pattern.match(name) for name in names):
        if found:
            highest = max(highest, int(found.group(1)))
    return highest


def _check_real_place(plan: OutputPlan) -> None:
    """Refuse an output that is itself a link: nothing is written through one.

    A linked folder on the way is fine, and common: `$HIP/render` pointing at
    a bigger disk is a normal setup. The path is kept under the root by the
    check on its text, which refuses any `..` that would climb out.
    """
    target = plan.path.rstrip("/")
    if os.path.islink(target):
        raise ConventionError(f"{target} is a link, and an output is never written through one")


def _make_room(plan: OutputPlan) -> bool:
    """Take this run's place on disk, and say whether it was free.

    A kind with a version folder claims the folder. A kind whose version lives
    in the file name claims a small file beside the output, because two
    machines with their own stores can hand out the same number and only one of
    them may write it.
    """
    _check_real_place(plan)
    exclusive = plan.version_dir or (plan.directory if plan.is_directory else None)
    if exclusive is not None:
        try:
            Path(exclusive).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return False
        Path(plan.directory).mkdir(parents=True, exist_ok=True)
        return True

    Path(plan.directory).mkdir(parents=True, exist_ok=True)
    if plan.version is None:
        return True
    if Path(plan.path).exists():
        return False
    return _claim_file(f"{plan.path}{CLAIM_SUFFIX}")


def _claim_file(path: str) -> bool:
    """Create a marker file, and say whether this call is the one that made it."""
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.close(handle)
    return True
