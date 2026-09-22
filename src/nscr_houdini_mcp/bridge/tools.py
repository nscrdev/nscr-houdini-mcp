"""The first tools that touch a scene.

They are bridge level tools, named the way `bridge.ping` is named. The tool
surface a client sees is a separate, smaller set built on top of these.

- `scene.info` reads. It cooks nothing and changes nothing. Asked for, it
  also says which files and assets the scene points at that are not there.
- `scene.open`, `scene.save` and `scene.save_as` change which file the
  session holds or write it. None of them can be undone, so they run without
  an undo group and say so. A load replaces the scene, which moves the scene
  epoch; a save does not.
- `bridge.capabilities` reads facts about the process rather than the scene:
  the build, the license, the renderers that are really installed and the ways
  this session can make a picture. The pool records the answer beside the
  worker, so another process can pick a worker without asking it anything.
- `node.create` mutates, so it runs on the main thread in a graphical session
  and inside one undo group in every session.
- `node.inspect` reads nodes, networks and parameters a page at a time. It
  cooks nothing unless the call asks it to, and says so on every value and
  every error list it could only read from an earlier cook.
- `python.run` runs code a caller sent, with `hou`, in a namespace kept
  between calls. It counts as a change every time, whatever the code does.
- `bridge.selfcheck` mutates on purpose, and can be asked to take its time, to
  fail part way, or to throw the scene away, so the queue, the timeout, the
  rollback, the scene epoch and the receipts can be tried against a real
  Houdini rather than only against a stand in. It is registered in a worker
  this project started to be driven and nowhere else.
"""

from __future__ import annotations

import errno
import hashlib
import linecache
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge.errors import (
    MAX_HINTS,
    BridgeError,
    did_you_mean,
    hide_paths,
    map_exception,
)

# The networks a scene summary counts, in the order a person reads them.
CONTEXTS = ("/obj", "/out", "/stage", "/mat", "/ch", "/shop", "/img", "/tasks")

# The render node types a capability probe looks for. A name that is not
# registered in this build is simply not reported.
RENDER_TYPES = (
    "karma",
    "ifd",
    "usdrender_rop",
    "arnold",
    "Redshift_ROP",
    "vray_renderer",
)

# Bounds on what the self check will do, so a mistyped argument cannot park
# the session for long or fill a scene.
MAX_SLEEP_S = 60.0
MAX_CREATES = 64

# How long the self check sleeps between looks at the cancel flag.
SLICE_S = 0.05

# What a scene file may end in: full, non commercial and limited commercial.
HIP_SUFFIXES = (".hip", ".hipnc", ".hiplc")

# Bounds on the dependency report, so a scene with thousands of references
# answers in bounded time and size. A report that hit one says so.
MAX_REFERENCES = 2000
MAX_REPORTED = 200
MAX_NODES_SCANNED = 50000
MAX_WARNING_LINES = 50

# What a load warning says about a node type this build does not have, and
# about an asset whose library could not be found.
_BAD_TYPE = re.compile(r"Bad node type found:\s*(\S+)\s+in\s+(\S+?)\.?\s*$")
_INCOMPLETE = re.compile(r'"(/[^"]+)"\s+using incomplete asset definition')

# A frame in a file name, which makes one reference a sequence of files.
_FRAME_VARIABLE = re.compile(r"\$\{?(F\d*|FF|SF|T)\}?")

# Values a file parameter can hold that are not places on disk.
_NOT_ON_DISK = ("op:", "opdef:", "oplib:", "temp:", "http:", "https:")

UNDO_NOTE = "a scene file change cannot be undone"

# The scene suffix each license writes, by the license category's own name.
# A license not named here writes any of them.
LICENSE_SUFFIX = {
    "apprentice": ".hipnc",
    "apprenticehd": ".hipnc",
    "education": ".hipnc",
    "indie": ".hiplc",
}

# The name Houdini gives a scene that has never been saved. Setting it gives
# the session back its untitled state.
UNTITLED_NAME = "untitled.hip"

# How old a private save file has to be before it is taken as left behind.
PRIVATE_KEEP_S = 3600.0

# Errors from a hard link that mean the file system has none, rather than
# that something went wrong with this file.
NO_LINKS = frozenset(
    code
    for code in (
        getattr(errno, name, None) for name in ("EPERM", "ENOTSUP", "EOPNOTSUPP", "ENOSYS", "EXDEV")
    )
    if code is not None
)
WINDOWS_NO_LINKS = frozenset({1, 50})

_TRAILING_VERSION = re.compile(r"[._-]v\d+$", re.IGNORECASE)

# The words Houdini puts on a node whose asset library was not found.
STUB_WARNING = "incomplete asset definition"


@dataclass(frozen=True)
class ToolContext:
    """What a tool is told about the call it is running under.

    `cancel` is set when somebody asks this call to stop, `stopping` when the
    session itself is going down. A tool that takes any time at all should
    look at `should_stop` between pieces of work and return what it has. It is
    a request: nothing takes the session off a tool that ignores it.
    """

    hou: Any | None = None
    kind: str = "hython"
    session_id: str = ""
    scene_epoch: int = 0
    operation_id: str = ""
    label: str = ""
    cancel: Any = None
    stopping: Any = None
    # Takes one progress note for health to show, while this call runs.
    progress: Any = None
    # The state folder, and a way to open the store, for managed outputs.
    home: Any = None
    open_store: Any = None

    def should_stop(self) -> bool:
        """Whether this call has been asked to stop, or the session has."""
        return any(flag is not None and flag.is_set() for flag in (self.cancel, self.stopping))


# Section: reads


def scene_info(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """What is open, where it is, and how big it is. Nothing is cooked.

    In a graphical session this runs on the main thread, so `frame` is the
    frame the artist sees.
    """
    hou = _houdini(context)
    info = {
        "hip_path": _quiet(hou.hipFile.path),
        "hip_name": _quiet(lambda: hou.hipFile.basename()),
        "untitled": _quiet(lambda: hou.hipFile.isNewFile()),
        "houdini_version": _quiet(hou.applicationVersionString),
        "frame": _quiet(hou.frame),
        "fps": _quiet(hou.fps),
        "frame_range": _range(hou),
        "nodes": _counts(hou),
        "undo_entries": _undo_entries(hou),
        "unsaved": _unsaved(hou, context),
        "scene_epoch": context.scene_epoch,
        "session_id": context.session_id,
        "kind": context.kind,
    }
    if arguments.get("dependencies"):
        info["dependencies"] = dependencies(hou)
    return info


def _counts(hou: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in CONTEXTS:
        node = _quiet(lambda path=path: hou.node(path))
        if node is not None:
            children = _quiet(node.children)
            if children is not None:
                counts[path] = len(children)
    return counts


def capabilities(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """What this session can do, read once when a worker comes up.

    Everything here is a fact about the process, not about a scene, so it is
    true for as long as the session lives and worth recording where other
    processes can read it. A fact this build will not give is `None` or an
    empty list, never a guess.
    """
    hou = _houdini(context)
    return {
        "houdini_version": _quiet(hou.applicationVersionString),
        "houdini_build": _build(hou),
        "hfs": _quiet(lambda: hou.expandString("$HFS")),
        "gui": context.kind == "gui",
        "license": _license(hou),
        "renderers": _renderers(hou),
        "capture": _capture_routes(hou),
        "cancellation": True,
        "max_threads": _quiet(lambda: hou.expandString("$HOUDINI_MAXTHREADS")) or None,
        "platform": sys.platform,
    }


def _build(hou: Any) -> list[int] | None:
    """The version as numbers, for a comparison that does not parse text."""
    parts = _quiet(hou.applicationVersion)
    return None if parts is None else [int(part) for part in parts]


def _license(hou: Any) -> str | None:
    """Which kind of license this process got, in the build's own words."""
    category = _ask(hou, "licenseCategory")
    if category is None:
        return None
    return str(_quiet(category.name) or category)


def _renderers(hou: Any) -> list[str]:
    """The renderers this install has, by what is really there.

    Two questions: whether the standalone USD renderer sits beside hython, and
    which render node types this build knows. A type that is not registered
    means the renderer is not installed, whatever else is on the machine.
    """
    found: list[str] = []
    hfs = _quiet(lambda: hou.expandString("$HFS"))
    if hfs:
        husk = Path(hfs) / "bin" / ("husk.exe" if sys.platform == "win32" else "husk")
        if husk.is_file():
            found.append("husk")
    category = _quiet(hou.ropNodeTypeCategory)
    if category is None:
        return found
    for name in RENDER_TYPES:
        if _quiet(lambda name=name: hou.nodeType(category, name)) is not None:
            found.append(name)
    return found


def _capture_routes(hou: Any) -> list[str]:
    """The ways this session can produce a picture.

    A viewport flipbook needs a user interface, so a worker never has one. The
    render node route is there whenever the type is registered.
    """
    routes: list[str] = []
    if _quiet(hou.isUIAvailable):
        routes.append("viewport_flipbook")
    category = _quiet(hou.ropNodeTypeCategory)
    if category is not None and _quiet(lambda: hou.nodeType(category, "opengl")) is not None:
        routes.append("opengl_rop")
    return routes


def _unsaved(hou: Any, context: ToolContext) -> bool | None:
    """Whether the scene has edits that are not on disk, where that is known.

    A headless session answers yes to this even straight after a save, so the
    answer is only meaningful with a user interface. Nothing is better than a
    value that is always the same.
    """
    if context.kind != "gui":
        return None
    return _quiet(hou.hipFile.hasUnsavedChanges)


def _undo_entries(hou: Any) -> int | None:
    """How many entries are on the undo stack, where the build will say.

    One call that changes the scene should add one. It is here so a caller can
    check that for itself rather than take the reply's word for it.
    """
    labels = _quiet(hou.undos.undoLabels)
    return None if labels is None else len(labels)


def _range(hou: Any) -> list[float] | None:
    frames = _quiet(lambda: hou.playbar.frameRange())
    return None if frames is None else [float(value) for value in frames]


# Section: what a scene points at


def dependencies(hou: Any, warning: str = "") -> dict[str, Any]:
    """What the scene needs that this machine does not have.

    Three lists. Node types the scene names that this build has no
    definition for, which only a load reports, because such a node is never
    made. Assets whose library was not found, which leaves the node on a
    stub definition that has no parameters. And file references that point
    at nothing on disk.

    A file reference counts only when it is read and was set by somebody: a
    parameter at its default and every output driver are passed over, because
    an output that is not written yet is not missing. A reference with a
    frame in it counts as there when the file for the current frame, or for
    the first or last frame of the range, is there.
    """
    types, named, lines = _read_warning(warning)
    assets, truncated = _incomplete_assets(hou)
    for path in named:
        if path not in {asset["node"] for asset in assets}:
            node = _ask(hou, "node", path)
            node_type = _ask(node, "type") if node is not None else None
            assets.append({"node": path, "type": _ask(node_type, "name") if node_type else None})
    missing, checked, cut = _missing_files(hou)
    return {
        "unresolved_types": types[:MAX_REPORTED],
        "missing_hdas": assets[:MAX_REPORTED],
        "missing_files": missing,
        "load_warnings": lines,
        "references_checked": checked,
        "truncated": truncated or cut or len(types) > MAX_REPORTED,
    }


def _read_warning(text: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """The node types a load warning names, the asset nodes, and its lines."""
    types: list[dict[str, Any]] = []
    assets: list[str] = []
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        # The first line only repeats which file was loading.
        if not line or line.startswith("Error loading:"):
            continue
        if line.startswith("Warning:"):
            line = line[len("Warning:") :].strip()
        bad = _BAD_TYPE.search(line)
        if bad:
            types.append({"type": bad.group(1), "parent": bad.group(2)})
        stub = _INCOMPLETE.search(line)
        if stub:
            assets.append(stub.group(1))
        if len(lines) < MAX_WARNING_LINES:
            lines.append(line)
    return types, assets, lines


def _incomplete_assets(hou: Any) -> tuple[list[dict[str, Any]], bool]:
    """Nodes whose asset definition is a stub, because its library is missing.

    Houdini puts a warning on such a node saying the definition is
    incomplete, and the node has no parameters of its own. Only a node whose
    type has an asset definition is asked, and each type once however many
    nodes use it.
    """
    root = _quiet(lambda: hou.node("/"))
    if root is None:
        return [], False
    nodes = _quiet(lambda: root.allSubChildren(recurse_in_locked_nodes=False)) or ()
    verdicts: dict[str, bool] = {}
    found: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if index >= MAX_NODES_SCANNED or len(found) >= MAX_REPORTED:
            return found, True
        node_type = _ask(node, "type")
        if node_type is None:
            continue
        key = str(_ask(node_type, "nameWithCategory") or _ask(node_type, "name"))
        if key not in verdicts:
            verdicts[key] = _is_stub(node, node_type)
        if verdicts[key]:
            found.append({"node": node.path(), "type": _ask(node_type, "name")})
    return found, False


def _is_stub(node: Any, node_type: Any) -> bool:
    if _ask(node_type, "definition") is None:
        return False
    warnings = _ask(node, "warnings") or ()
    return any(STUB_WARNING in str(warning) for warning in warnings)


def _missing_files(hou: Any) -> tuple[list[dict[str, Any]], int, bool]:
    """File references that point at nothing, how many were read, and whether
    the read stopped at a bound."""
    references = _ask(hou, "fileReferences") or ()
    frames = _frames_to_try(hou)
    missing: list[dict[str, Any]] = []
    checked = 0
    for parm, raw in references:
        if checked >= MAX_REFERENCES or len(missing) >= MAX_REPORTED:
            return missing, checked, True
        checked += 1
        if parm is not None and _not_a_read(parm):
            continue
        values = [value for value in _values(hou, parm, raw, frames) if _on_disk(value)]
        if not values or any(os.path.exists(value) for value in values):
            continue
        missing.append({"parm": None if parm is None else _ask(parm, "path"), "path": values[0]})
    return missing, checked, False


def _not_a_read(parm: Any) -> bool:
    if _ask(parm, "isAtDefault"):
        return True
    category = _quiet(lambda: parm.node().type().category().name())
    return category == "Driver"


def _values(hou: Any, parm: Any, raw: Any, frames: list[float]) -> list[str]:
    """The paths one reference stands for: one, or one per frame tried."""
    if parm is None:
        value = _quiet(lambda: hou.expandString(str(raw)))
        return [str(value)] if value else []
    if _FRAME_VARIABLE.search(str(raw or "")):
        found = [_quiet(lambda frame=frame: parm.evalAsStringAtFrame(frame)) for frame in frames]
        return [str(value) for value in found if value]
    value = _ask(parm, "evalAsString")
    return [str(value)] if value else []


def _frames_to_try(hou: Any) -> list[float]:
    frames = [_ask(hou, "frame")]
    frames.extend(_range(hou) or [])
    return [float(frame) for frame in frames if frame is not None]


def _on_disk(value: str) -> bool:
    """Whether a value names a file on disk that can be looked for."""
    text = value.strip()
    if not text or text.startswith(_NOT_ON_DISK) or "$" in text:
        return False
    return os.path.isabs(text)


# Section: the scene file


def scene_open(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Load a scene file and report what it points at that is not there.

    A graphical session with changes that are not saved is refused unless the
    call says to throw them away, because a load that loses somebody's work
    without asking is worse than one more round trip. A worker's scene is the
    caller's own, and a headless session cannot tell a saved scene from an
    edited one anyway, so a worker is never refused for it.

    A load that finds problems still loads. What it could not resolve comes
    back as data, not as an error.
    """
    hou = _houdini(context)
    path = _hip_path(arguments["path"])
    if not os.path.isfile(path):
        raise BridgeError(
            "FILE_NOT_FOUND",
            "there is no scene file at that path",
            {"suffix": Path(path).suffix},
            hint="check the path, or list the folder, then open a file that is there",
        )
    unsaved = _unsaved(hou, context)
    if unsaved is None and context.kind == "gui":
        # A session with a user interface that will not say is taken to have
        # changes, because the cost of being wrong is somebody's work.
        unsaved = True
    discard = bool(arguments.get("discard_unsaved"))
    if unsaved and not discard:
        raise BridgeError(
            "UNSAVED_CHANGES",
            "the scene open in this session has changes that are not saved",
            {"hip_name": _quiet(lambda: hou.hipFile.basename())},
            hint="save the scene first, or pass discard_unsaved true to throw the changes away",
        )
    warning = ""
    try:
        hou.hipFile.load(path, suppress_save_prompt=True, ignore_load_warnings=False)
    except Exception as error:  # noqa: BLE001 - a warning is data, anything else goes on up
        if type(error).__name__ != "LoadWarning":
            raise
        warning = _warning_text(error)
    return {
        "hip_path": _quiet(hou.hipFile.path),
        "hip_name": _quiet(lambda: hou.hipFile.basename()),
        "nodes": _counts(hou),
        "discarded_unsaved": bool(unsaved) if unsaved is not None else None,
        "dependencies": dependencies(hou, warning),
        "undo": UNDO_NOTE,
    }


def _warning_text(error: BaseException) -> str:
    message = _quiet(error.instanceMessage) if hasattr(error, "instanceMessage") else None
    return str(message if message else error)


def scene_save(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Save the scene over its own file. A scene with no file is refused."""
    hou = _houdini(context)
    if _quiet(lambda: hou.hipFile.isNewFile()):
        raise BridgeError(
            "SCENE_UNTITLED",
            "the scene has never been saved, so it has no file to save over",
            hint="use save_increment, which picks a new versioned file for it",
        )
    hou.hipFile.save()
    path = _quiet(hou.hipFile.path)
    return {"hip_path": path, "bytes": _size(path), "undo": UNDO_NOTE}


def scene_save_as(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Save the scene to a new file. A file that is there is never written over.

    The scene is written to a private name in the same folder first and then
    published under the name asked for with a hard link, which fails when
    anything is at that name, whoever put it there and however late. A file
    system that cannot link falls back to a look before the write, and the
    reply says so.

    The session holds the new file from here on, as it would after a save as
    in the interface. The scene is the same scene, so the epoch stays.
    """
    hou = _houdini(context)
    path = _hip_path(arguments["path"])
    if not os.path.isabs(path):
        raise BridgeError("BAD_ARGUMENTS", "a scene file path has to be absolute")
    if os.path.lexists(path):
        raise _file_exists(path)
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        raise BridgeError(
            "FILE_NOT_FOUND",
            "the folder for that scene file is not there",
            hint="create the folder first, or save somewhere that is there",
        )
    wanted = license_suffix(hou)
    if wanted and not path.lower().endswith(wanted):
        raise BridgeError(
            "BAD_ARGUMENTS",
            f"this license saves scene files as {wanted}, so it would not write that name",
            {"suffix": Path(path).suffix, "license_suffix": wanted},
            hint=f"ask for a path ending in {wanted}",
        )
    stem, suffix = os.path.splitext(path)
    private = f"{stem}.part{os.getpid()}{suffix}"
    _sweep_private(folder, os.path.basename(stem), suffix)
    if os.path.isfile(private) and not os.path.islink(private):
        # Left by an attempt of this process that stopped before it published.
        os.remove(private)
    # What to give the session back if this does not work out: the file it
    # held, or for a scene that never had one, Houdini's own untitled name,
    # which makes it untitled again.
    was_untitled = bool(_ask(hou.hipFile, "isNewFile"))
    before = UNTITLED_NAME if was_untitled else _ask(hou.hipFile, "path")
    try:
        _save_privately(hou, private)
        warnings = _publish(private, path)
    except BaseException:
        _discard(private)
        _ask(hou.hipFile, "setName", before or UNTITLED_NAME)
        raise
    hou.hipFile.setName(path)
    return {
        "hip_path": _ask(hou.hipFile, "path") or path,
        "bytes": _size(path),
        "warnings": warnings,
        "undo": UNDO_NOTE,
    }


def _save_privately(hou: Any, private: str) -> None:
    """Write the scene under its private name, kept out of the recent files."""
    try:
        hou.hipFile.save(private, save_to_recent_files=False)
    except TypeError:
        # A build without the keyword.
        hou.hipFile.save(private)


def _sweep_private(folder: str, stem: str, suffix: str) -> None:
    """Remove private files of this scene family left by an attempt that died.

    Only files older than an hour are taken, so a save running in another
    process right now is never touched.
    """
    family = _TRAILING_VERSION.sub("", stem)
    pattern = re.compile(
        rf"^{re.escape(family)}(?:[._-]v\d+)?\.part\d+{re.escape(suffix)}$", re.IGNORECASE
    )
    cutoff = time.time() - PRIVATE_KEEP_S
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for name in names:
        if not pattern.match(name):
            continue
        found = os.path.join(folder, name)
        try:
            if not os.path.islink(found) and os.path.getmtime(found) < cutoff:
                os.remove(found)
        except OSError:
            continue


def _publish(private: str, path: str) -> list[str]:
    """Give a written file its name, only if nothing has that name yet.

    A hard link does that in one step. Where the file system cannot link, the
    name is taken first with an exclusive create, or on Windows by a rename,
    which refuses a name that is there. Any other failure is the save's own.
    """
    try:
        os.link(private, path)
    except FileExistsError:
        raise _file_exists(path) from None
    except OSError as error:
        if not links_unsupported(error):
            raise BridgeError(
                "TOOL_FAILED",
                "the scene was written but could not be given its name",
                {"errno": errno.errorcode.get(error.errno or 0, error.errno)},
                hint="check the space and the permissions of the folder, then save again",
            ) from None
        _publish_without_links(private, path)
        return ["this folder cannot link files, so the file was published by a rename"]
    _discard(private)
    return []


def _publish_without_links(private: str, path: str) -> None:
    if sys.platform == "win32":
        try:
            os.rename(private, path)
        except FileExistsError:
            raise _file_exists(path) from None
        return
    try:
        placeholder = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise _file_exists(path) from None
    os.close(placeholder)
    try:
        os.replace(private, path)
    except OSError:
        _discard(path)
        raise


def links_unsupported(error: OSError) -> bool:
    """Whether an error from a hard link means this file system has none."""
    if getattr(error, "winerror", None) in WINDOWS_NO_LINKS:
        return True
    return error.errno in NO_LINKS


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _file_exists(path: str) -> BridgeError:
    return BridgeError(
        "FILE_EXISTS",
        "a file is already at that path",
        {"suffix": Path(path).suffix},
        hint="ask for the next version rather than writing over this one",
    )


def license_suffix(hou: Any) -> str | None:
    """The only scene suffix this license writes, or nothing when it writes any.

    A license that writes one kind of file renames anything else on the way
    out, so a path is checked against it before a save rather than after.
    """
    name = str(_license(hou) or "").lower().replace(" ", "")
    return LICENSE_SUFFIX.get(name)


def _size(path: Any) -> int | None:
    try:
        return os.path.getsize(str(path))
    except (OSError, TypeError):
        return None


# Section: mutations


def create_node(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Make one node under a parent, and set the parameters that were given."""
    hou = _houdini(context)
    parent_path = str(arguments["parent"])
    type_name = str(arguments["type"])
    wanted_name = arguments.get("name")
    parms = arguments.get("parms") or {}
    if not isinstance(parms, Mapping):
        raise BridgeError("BAD_ARGUMENTS", "parms must be an object of name to value")

    parent = _node(hou, parent_path)
    try:
        node = (
            parent.createNode(type_name, str(wanted_name))
            if wanted_name
            else parent.createNode(type_name)
        )
    except Exception as error:  # noqa: BLE001 - a wrong type is the caller's mistake
        if _is_hou_error(error):
            raise BridgeError(
                "BAD_ARGUMENTS",
                f"no node type {type_name} can be made in that network",
                {
                    "parent": parent_path,
                    "type": type_name,
                    "did_you_mean": _types(parent, type_name),
                },
                hint="ask for the parent's child types and use one of those names",
            ) from None
        raise

    for name, value in parms.items():
        _set_parm(node, str(name), value)

    return {
        "path": node.path(),
        "name": node.name(),
        "type": node.type().name(),
        "parent": parent_path,
        "parms_set": sorted(str(name) for name in parms),
    }


def _set_parm(node: Any, name: str, value: Any) -> None:
    parm = node.parm(name) or node.parmTuple(name)
    if parm is None:
        raise BridgeError(
            "PARM_NOT_FOUND",
            f"the node has no parameter named {name}",
            {"node": node.path(), "parm": name, "did_you_mean": _parms(node, name)},
            hint="ask for the node's parameters and use one of those names",
        )
    parm.set(value)


def selfcheck(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Take a while, make a few nodes, fail or replace the scene where asked.

    It exists so the rules around a tool can be tried against a real Houdini:
    the queue, the wait, the timeout, one undo entry per call, the rollback
    when a call fails after it has already changed the graph, the scene epoch,
    and what a repeated operation id does.
    """
    hou = _houdini(context)
    sleep_s = _number(arguments.get("sleep_s"), "sleep_s", MAX_SLEEP_S)
    creates = int(_number(arguments.get("creates"), "creates", MAX_CREATES))
    fail_at = arguments.get("fail_at")
    if fail_at is not None:
        fail_at = int(_number(fail_at, "fail_at", MAX_CREATES))
    parent_path = str(arguments.get("parent") or "/obj")

    slept = _sleep(sleep_s, context)

    made: list[str] = []
    parent = _node(hou, parent_path) if creates or fail_at else None
    for index in range(1, creates + 1):
        if context.should_stop():
            break
        if fail_at is not None and fail_at == index:
            raise BridgeError(
                "TOOL_FAILED",
                "the self check was asked to fail here",
                {"failed_at": index, "created_before_failing": list(made)},
            )
        made.append(parent.createNode("geo").path())

    scene = _scene_moves(hou, arguments)
    return {
        "created": made,
        "slept_s": round(slept, 3),
        "stopped_early": context.should_stop(),
        "operation_id": context.operation_id,
        **scene,
    }


def _scene_moves(hou: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Save, clear or load a scene, for the checks about scene identity.

    Each one is what it says: `save_hip` writes the scene where it is told,
    `new_scene` throws the scene away, and `load_hip` reads one back. The last
    two replace the scene, which is what moves the session's scene epoch.
    """
    done: dict[str, Any] = {}
    saved = arguments.get("save_hip")
    if saved:
        hou.hipFile.save(_hip_path(saved))
        done["saved"] = str(hou.hipFile.path())
    if arguments.get("new_scene"):
        hou.hipFile.clear(suppress_save_prompt=True)
        done["cleared"] = True
    loaded = arguments.get("load_hip")
    if loaded:
        hou.hipFile.load(_hip_path(loaded), suppress_save_prompt=True)
        done["loaded"] = str(hou.hipFile.path())
    return done


def _hip_path(value: Any) -> str:
    """One scene file path, refused unless it names a scene file."""
    text = os.path.expanduser(str(value))
    if not text.lower().endswith(HIP_SUFFIXES):
        raise BridgeError(
            "BAD_ARGUMENTS",
            "a scene file path has to end in .hip, .hipnc or .hiplc",
            {"suffix": Path(text).suffix},
        )
    return text


def _sleep(seconds: float, context: ToolContext) -> float:
    """Wait, looking often at whether this call has been asked to stop."""
    began = time.monotonic()
    while time.monotonic() - began < seconds:
        if context.should_stop():
            break
        time.sleep(min(SLICE_S, seconds - (time.monotonic() - began)))
    return time.monotonic() - began


# Section: reading nodes and parameters

INSPECT_MODES = ("tree", "node", "parms", "find", "selection")
DETAIL_LEVELS = ("summary", "standard", "full")
INCLUDES = ("wires", "flags", "errors", "expressions", "code", "notes", "cook_time")

# What each level brings in on its own. An `include` item brings one of these
# in at a lower level.
STANDARD_ITEMS = frozenset({"wires", "flags", "errors", "notes"})
FULL_ITEMS = frozenset(INCLUDES)

MAX_BATCH = 50
MAX_TREE_DEPTH = 8
DEFAULT_LIMIT = 200
MAX_LIMIT = 2000

# How many near misses a path that is not there comes back with, and how many
# nodes are looked through for a name like its last part.
MAX_NEAR = 5
NEAR_SCAN = 5000

# How many instances of one multiparm a row carries, and how many network
# boxes and sticky notes a tree carries. Past either, the answer says so.
MAX_INSTANCES = 200
MAX_NETWORK_ITEMS = 500

# The flags a row reports, by the name it reports them under and the name the
# build gives them. A flag a node does not have reads as not set.
FLAG_NAMES = (
    ("display", "Display"),
    ("render", "Render"),
    ("bypass", "Bypass"),
    ("template", "Template"),
    ("lock", "Lock"),
    ("soft_lock", "SoftLock"),
)

# The node categories a read with `evaluate` cooks. A render driver cooks by
# rendering and a task network by running work, so neither is ever cooked by
# a read, and a network or shader node has no cook of its own to ask for.
COOKABLE = frozenset({"Sop", "Object", "Lop", "Cop2", "Cop", "Chop", "Dop"})

STALE_REASON = "the node, or something it reads, changed after its last cook"
NO_SELECTION = "no selection outside a GUI"

# Why a value was left out of a read that may not cook. A reason about an
# expression names what in it could cook.
OVERRIDE = "override"
KEYFRAMES = "keyframes"
PYTHON = "python"
BACKTICK = "backtick"

# Parameter kinds with no value to read.
_NO_VALUE = frozenset({"Button", "FolderSet", "Separator", "Label"})

# What an expression or a string may use and still be read without a cook.
# A variable that only means something inside a cook, such as `$NPT` or
# `$PT`, reads as nothing outside one, so it is left alone, and so is
# anything else not named here.
SAFE_VARIABLES = frozenset(
    {
        "F",
        "FF",
        "T",
        "FPS",
        "FSTART",
        "FEND",
        "NFRAMES",
        "RFSTART",
        "RFEND",
        "PI",
        "E",
        "HIP",
        "HIPNAME",
        "HIPFILE",
        "JOB",
        "OS",
        "HOME",
        "HFS",
        "HH",
        "ACTIVETAKE",
        "EYE",
        "HOUDINI_TEMP_DIR",
    }
)
# `$F4` and its like are the frame, padded.
_PADDED_FRAME = re.compile(r"F\d+")

CURVE_FUNCTIONS = frozenset(
    {
        "bezier",
        "linear",
        "constant",
        "cubic",
        "ease",
        "easein",
        "easeout",
        "easep",
        "easeinp",
        "easeoutp",
        "spline",
        "qlinear",
        "qcubic",
        "cycle",
        "cyclet",
        "cycleoffset",
        "cycleoffsett",
        "match",
        "matchin",
        "matchout",
        "vmatch",
        "vmatchin",
        "vmatchout",
        "repeat",
        "repeatt",
    }
)
SAFE_FUNCTIONS = CURVE_FUNCTIONS | frozenset(
    {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "clamp",
        "cos",
        "cosh",
        "deg",
        "exp",
        "fit",
        "fit01",
        "fit10",
        "fit11",
        "floor",
        "frac",
        "int",
        "log",
        "log10",
        "max",
        "min",
        "noise",
        "pow",
        "rad",
        "rand",
        "rint",
        "round",
        "sign",
        "sin",
        "sinh",
        "smooth",
        "sqrt",
        "tan",
        "tanh",
        "if",
        "strcat",
        "substr",
        "strlen",
        "padzero",
        "sprintf",
        "atof",
    }
)
# A reference to another parameter is followed, a few steps deep, and is safe
# when what it points at is. Only the form with one quoted name is read.
REFERENCE_FUNCTIONS = frozenset({"ch", "chf", "chs", "chsraw"})
MAX_REFERENCE_DEPTH = 4

# One HScript token: a quoted string, a variable, a name, a number, or any
# other single character. A quote that is never closed is a character of its
# own, which makes the expression unreadable and so not safe.
_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"'
    r"|'(?:[^'\\]|\\.)*'"
    r"|\$\{?[A-Za-z_]\w*\}?"
    r"|[A-Za-z_]\w*"
    r"|\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+"
    r"|\S"
)
_VARIABLE_NAME = re.compile(r"\$\{?([A-Za-z_]\w*)\}?")
_CURVE = re.compile(r"\s*(" + "|".join(sorted(CURVE_FUNCTIONS)) + r")\s*\(\s*\)\s*")
_TRAILING_DIGITS = re.compile(r"\d+$")
_NOT_A_WORD = re.compile(r"[^a-z0-9]")

# How long a text value in user data may be before it is cut.
MAX_TEXT = 2000

# How a multiparm instance sorts in a page key: its number, padded.
_INSTANCE_KEY = "{:08d}"


def inspect(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Read nodes, networks and parameters, one page of rows at a time.

    Nothing is cooked unless `evaluate` is set. Without it every value that
    would pull on a cook is left out and says why, and what a node reports
    about its last cook is marked when it has never cooked or is out of date.
    """
    return _Reader(_houdini(context), context, arguments).run()


class _Reader:
    """One read, with the choices the call made."""

    def __init__(self, hou: Any, context: ToolContext, arguments: Mapping[str, Any]) -> None:
        self.hou = hou
        self.context = context
        self.arguments = arguments
        self.mode = _choice(arguments.get("mode"), INSPECT_MODES, "mode", "tree")
        level = _choice(arguments.get("detail"), DETAIL_LEVELS, "detail", "summary")
        included = arguments.get("include") or ()
        if isinstance(included, str) or not all(item in INCLUDES for item in included):
            raise BridgeError(
                "BAD_ARGUMENTS",
                "include takes a list of: " + ", ".join(INCLUDES),
                {"include": list(included) if not isinstance(included, str) else included},
            )
        self.standard = level in ("standard", "full")
        self.full = level == "full"
        brought = set(included)
        if self.standard:
            brought |= STANDARD_ITEMS
        if self.full:
            brought |= FULL_ITEMS
        self.items = frozenset(brought)
        self.evaluate = bool(arguments.get("evaluate"))
        self.limit = int(_number(arguments.get("limit") or DEFAULT_LIMIT, "limit", MAX_LIMIT))
        if self.limit < 1:
            raise BridgeError("BAD_ARGUMENTS", "limit must be at least 1")
        after = arguments.get("after")
        self.after = _key(str(after)) if after else None
        self.batch = bool(arguments.get("batch"))
        self.cooked: set[str] = set()
        self.stopped = False
        self.near = _NearIndex(hou, MAX_NEAR)
        self.exports = _Exports()

    def wants(self, item: str) -> bool:
        return item in self.items

    def run(self) -> dict[str, Any]:
        reader = {
            "tree": self.tree,
            "node": self.nodes,
            "parms": self.parms,
            "find": self.find,
            "selection": self.selection,
        }[self.mode]
        result = reader()
        if self.stopped:
            result["stopped"] = True
        # What the scene is now, so the next page can tell whether it moved,
        # and whether the row this page carries on after is still there.
        result["mark"] = _scene_mark(self.hou)
        if self.after is not None and self.mode in ("tree", "find", "selection"):
            if _quiet(lambda: self.hou.node(_text_key(self.after))) is None:
                result["resume_gone"] = True
        return result

    def should_stop(self) -> bool:
        if self.context.should_stop():
            self.stopped = True
        return self.stopped

    # Section: which nodes

    def tree(self) -> dict[str, Any]:
        root = self.node_at(str(self.arguments.get("path") or "/"))
        depth = int(_number(self.arguments.get("depth") or 1, "depth", MAX_TREE_DEPTH))
        if depth < 1:
            raise BridgeError("BAD_ARGUMENTS", "depth must be at least 1")
        result = self.page_of_walk(root, depth, None)
        if self.after is None and self.wants("notes") and not self.stopped:
            result.update(self.network_items(root, depth))
        return result

    def find(self) -> dict[str, Any]:
        root = self.node_at(str(self.arguments.get("path") or "/"))
        pattern = self.arguments.get("pattern")
        type_glob = self.arguments.get("type")
        if not pattern and not type_glob:
            raise BridgeError("BAD_ARGUMENTS", "find needs pattern, type or both")

        def keep(node: Any) -> bool:
            if pattern and not _glob_node(node, str(pattern)):
                return False
            return not type_glob or _glob_type(node, str(type_glob))

        return self.page_of_walk(root, None, keep)

    def selection(self) -> dict[str, Any]:
        if self.context.kind != "gui":
            return {"rows": [], "total": 0, "note": NO_SELECTION}
        chosen = list(_quiet(self.hou.selectedNodes) or ())
        keyed = sorted(((_key(node.path()), node) for node in chosen), key=lambda pair: pair[0])
        start = self.start_of([key for key, _ in keyed])
        rows: list[dict[str, Any]] = []
        last = self.after
        for key, node in keyed[start : start + self.limit]:
            if self.should_stop():
                break
            self.cook(node)
            rows.append(self.row(node))
            last = key
        result: dict[str, Any] = {"rows": rows, "total": len(keyed)}
        if start + len(rows) < len(keyed):
            result["more"] = True
            result["last"] = _text_key(last)
        return result

    def page_of_walk(self, root: Any, depth: int | None, keep: Any) -> dict[str, Any]:
        """One page of rows under a node, in path order, from where the last left off.

        The walk goes down in path order and stops as soon as the page is full,
        so a page after the first never looks at the nodes before it and a walk
        cut short by the scan bound leaves a clean point to go on from.
        """
        found, exhausted, last, truncated = self.walk(root, depth, keep)
        rows: list[dict[str, Any]] = []
        for node in found:
            if self.should_stop():
                break
            self.cook(node)
            rows.append(self.row(node))
        result: dict[str, Any] = {"rows": rows}
        if self.stopped:
            # Carry on after the last row built, whatever the walk had reached.
            resume = _key(rows[-1]["path"]) if rows else self.after
            result["more"] = True
            result["last"] = _text_key(resume)
            return result
        if not exhausted:
            result["more"] = True
            result["last"] = _text_key(last)
        elif self.after is None:
            result["total"] = len(rows)
        if truncated:
            result["truncated"] = True
        return result

    def walk(
        self, root: Any, depth: int | None, keep: Any
    ) -> tuple[list[Any], bool, tuple[str, ...] | None, bool]:
        """The next nodes under a root, in path order, a page's worth.

        Siblings are taken in name order and each node before its insides,
        which is the order the path keys sort in. A subtree wholly before the
        point a page carries on from is passed over without being opened. The
        answer is the nodes kept, whether the walk reached the end, the key
        of the last node looked at, and whether the scan bound cut it short.
        """
        found: list[Any] = []
        stack = [(child, 1) for child in reversed(_sorted_children(root))]
        last = self.after
        scanned = 0
        while stack:
            if self.should_stop():
                return found, False, last, False
            if scanned >= MAX_NODES_SCANNED:
                return found, False, last, True
            node, level = stack.pop()
            key = _key(node.path())
            descend = (depth is None or level < depth) and not _ask(node, "isLockedHDA")
            if self.after is not None and key <= self.after:
                if descend and self.after[: len(key)] == key:
                    stack.extend((child, level + 1) for child in reversed(_sorted_children(node)))
                continue
            scanned += 1
            last = key
            if descend:
                stack.extend((child, level + 1) for child in reversed(_sorted_children(node)))
            if keep is not None and not keep(node):
                continue
            found.append(node)
            if len(found) >= self.limit:
                return found, not stack, last, False
        return found, True, last, False

    def start_of(self, keys: list[tuple[str, ...]]) -> int:
        if self.after is None:
            return 0
        for index, key in enumerate(keys):
            if key > self.after:
                return index
        return len(keys)

    # Section: one node

    def node_at(self, path: str) -> Any:
        return _node_at(self.hou, path, self.near)

    def row(self, node: Any) -> dict[str, Any]:
        """One compact row: who the node is, then what the level asks for."""
        row = self.identity(node)
        if _default_name(node):
            row["auto"] = True
        kids = len(_quiet(node.children) or ())
        if kids:
            row["kids"] = kids
        errors = _ask(node, "errors") or ()
        if errors:
            row["err"] = len(errors)
        row.update(self.marks(node))
        if self.wants("flags"):
            flags = self.flags(node)
            if flags:
                row["flags"] = flags
        if self.wants("wires"):
            wired = self.inputs(node)
            if wired:
                row["in"] = wired
        if self.wants("errors"):
            row.update(self.messages(node, errors))
        if self.wants("notes"):
            comment = _ask(node, "comment")
            if comment:
                row["comment"] = comment
        if self.full:
            changed = self.changed(node)
            if changed:
                row["changed"] = changed
            row.update(_position(node))
        if self.wants("cook_time"):
            row.update(self.cook_time(node))
        return row

    def entry(self, node: Any) -> dict[str, Any]:
        """Everything the level asks for about one node."""
        entry = self.identity(node)
        flags = self.flags(node)
        if flags:
            entry["flags"] = flags
        wired = self.inputs(node)
        if wired:
            entry["in"] = wired
        errors = _ask(node, "errors") or ()
        if errors:
            entry["err"] = len(errors)
        entry.update(self.marks(node))
        if self.wants("wires"):
            labelled = self.labelled_inputs(node)
            if labelled:
                entry["inputs"] = labelled
            outputs = [item.path() for item in _quiet(node.outputs) or ()]
            if outputs:
                entry["out"] = outputs
        if self.wants("errors"):
            entry.update(self.messages(node, errors))
        if self.wants("notes"):
            comment = _ask(node, "comment")
            if comment:
                entry["comment"] = comment
        if self.standard:
            color = self.color(node)
            if color:
                entry["color"] = color
        rows = self.parm_rows_of(node)
        if rows is not None:
            entry["parms"] = rows
        if self.wants("cook_time"):
            entry.update(self.cook_time(node))
        if self.full:
            if _default_name(node):
                entry["auto"] = True
            entry.update(_position(node))
            user = _ask(node, "userDataDict")
            if user:
                entry["user"] = {str(key): _cut(value) for key, value in dict(user).items()}
        return entry

    def identity(self, node: Any) -> dict[str, Any]:
        node_type = _ask(node, "type")
        return {"path": node.path(), "type": _ask(node_type, "name") if node_type else None}

    def marks(self, node: Any) -> dict[str, Any]:
        """What a read of the node's last cook has to say about that cook."""
        if node.path() in self.cooked:
            return {}
        count = _ask(node, "cookCount")
        if count == 0:
            return {"not_cooked": True}
        if count and _ask(node, "needsToCook"):
            return {"stale": STALE_REASON}
        return {}

    def cook(self, node: Any) -> None:
        """Cook one node for a read that may, when it is a kind that cooks."""
        if not self.evaluate:
            return
        category = _quiet(lambda: node.type().category().name())
        if category not in COOKABLE:
            return
        if not _ask(node, "needsToCook") and (_ask(node, "cookCount") or 0) > 0:
            self.cooked.add(node.path())
            return
        try:
            node.cook(force=False)
        except Exception:  # noqa: BLE001 - a cook that fails leaves its errors on the node
            pass
        self.cooked.add(node.path())

    def flags(self, node: Any) -> list[str]:
        names = _quiet(lambda: self.hou.nodeFlag)
        if names is None:
            return []
        found: list[str] = []
        for label, attribute in FLAG_NAMES:
            flag = getattr(names, attribute, None)
            if flag is not None and _ask(node, "isGenericFlagSet", flag):
                found.append(label)
        return found

    def inputs(self, node: Any) -> list[str | None]:
        """The paths wired into each input, in input order."""
        wired: dict[int, str] = {}
        for connection in _quiet(node.inputConnections) or ():
            index = _ask(connection, "inputIndex")
            source = _source(connection)
            if isinstance(index, int) and source:
                wired[index] = source
        if not wired:
            return []
        return [wired.get(index) for index in range(max(wired) + 1)]

    def labelled_inputs(self, node: Any) -> list[dict[str, Any]]:
        labels = list(_quiet(node.inputLabels) or ())
        found: list[dict[str, Any]] = []
        for connection in _quiet(node.inputConnections) or ():
            index = _ask(connection, "inputIndex")
            source = _source(connection)
            if not isinstance(index, int) or not source:
                continue
            item: dict[str, Any] = {"i": index, "from": source}
            if index < len(labels) and labels[index]:
                item["label"] = str(labels[index])
            output = _ask(connection, "outputIndex")
            if output:
                item["out"] = output
            found.append(item)
        return sorted(found, key=lambda item: item["i"])

    def messages(self, node: Any, errors: Any) -> dict[str, Any]:
        said: dict[str, Any] = {}
        if errors:
            said["errors"] = [str(text) for text in errors]
        warnings = _ask(node, "warnings") or ()
        if warnings:
            said["warnings"] = [str(text) for text in warnings]
        return said

    def cook_time(self, node: Any) -> dict[str, Any]:
        """How many cooks there have been, and how long the last one took.

        A node that has never cooked has no last cook, so it has no time.
        """
        said: dict[str, Any] = {}
        count = _ask(node, "cookCount")
        if count is not None:
            said["cooks"] = int(count)
        spent = _ask(node, "lastCookTime") if count else None
        if spent is not None:
            said["cook_ms"] = round(float(spent), 3)
        return said

    def color(self, node: Any) -> list[float] | None:
        own = _quiet(lambda: tuple(node.color().rgb()))
        usual = _quiet(lambda: tuple(node.type().defaultColor().rgb()))
        if own is None or own == usual:
            return None
        return [round(float(value), 3) for value in own]

    def changed(self, node: Any) -> int:
        """How many parameters differ from their defaults."""
        top, _ = _parm_tree(node)
        return sum(
            1
            for tuple_ in top
            if _kind(tuple_) not in _NO_VALUE and _has_value(tuple_) and not _at_default(tuple_)
        )

    def network_items(self, root: Any, depth: int) -> dict[str, Any]:
        """The network boxes and sticky notes in the networks a tree reads.

        Complete up to a bound of their own, and on the first page only, with
        a flag when the bound was reached.
        """
        boxes: list[dict[str, Any]] = []
        notes: list[dict[str, Any]] = []
        networks = [(root, 0)]
        scanned = 0
        while networks and scanned < MAX_NODES_SCANNED:
            if len(boxes) > MAX_NETWORK_ITEMS and len(notes) > MAX_NETWORK_ITEMS:
                break
            network, level = networks.pop(0)
            scanned += 1
            for box in _quiet(network.networkBoxes) or ():
                if len(boxes) > MAX_NETWORK_ITEMS:
                    break
                boxes.append(self.box(box))
            for note in _quiet(network.stickyNotes) or ():
                if len(notes) > MAX_NETWORK_ITEMS:
                    break
                notes.append({"path": note.path(), "text": str(_ask(note, "text") or "")})
            if level + 1 < depth:
                networks.extend(
                    (child, level + 1)
                    for child in _sorted_children(network)
                    if not _ask(child, "isLockedHDA")
                )
        found: dict[str, Any] = {}
        if boxes:
            found["boxes"] = boxes[:MAX_NETWORK_ITEMS]
        if notes:
            found["notes"] = notes[:MAX_NETWORK_ITEMS]
        if len(boxes) > MAX_NETWORK_ITEMS:
            found["boxes_truncated"] = True
        if len(notes) > MAX_NETWORK_ITEMS:
            found["notes_truncated"] = True
        return found

    def box(self, box: Any) -> dict[str, Any]:
        item: dict[str, Any] = {"path": box.path()}
        comment = _ask(box, "comment")
        if comment:
            item["comment"] = comment
        inside = [node.path() for node in _quiet(box.nodes) or ()]
        if inside:
            item["nodes"] = inside
        if self.full:
            item.update(_position(box))
            size = _quiet(lambda: list(box.size()))
            if size is not None:
                item["size"] = [round(float(value), 3) for value in size]
        return item

    # Section: several nodes by path

    def paths(self) -> list[str]:
        given = self.arguments.get("paths") or ()
        if isinstance(given, str) or not given:
            raise BridgeError("BAD_ARGUMENTS", "this mode needs paths")
        if len(given) > MAX_BATCH:
            raise BridgeError("BAD_ARGUMENTS", f"at most {MAX_BATCH} paths in one call")
        return list(dict.fromkeys(_tidy(str(path)) for path in given))

    def nodes(self) -> dict[str, Any]:
        """Node entries, one per node however it was named, sorted and paged."""
        items: dict[tuple[str, ...], tuple[str, Any]] = {}
        for path in self.paths():
            try:
                node = self.node_at(path)
                items.setdefault(_key(node.path()), ("node", node))
            except Exception as error:  # noqa: BLE001 - one bad path is one bad entry
                if not self.batch:
                    raise
                items.setdefault(_key(path), ("error", _failed_entry(path, error)))
        keys = sorted(items)
        start = self.start_of(keys)
        entries: list[dict[str, Any]] = []
        last = self.after
        for key in keys[start : start + self.limit]:
            if self.should_stop():
                break
            kind, payload = items[key]
            if kind == "error":
                entries.append(payload)
            else:
                try:
                    self.cook(payload)
                    entries.append(self.entry(payload))
                except Exception as error:  # noqa: BLE001 - one bad node is one bad entry
                    if not self.batch:
                        raise
                    entries.append(_failed_entry(_text_key(key), error))
            last = key
        result: dict[str, Any] = {"nodes": entries, "total": len(keys)}
        if start + len(entries) < len(keys):
            result["more"] = True
            result["last"] = _text_key(last)
        return result

    # Section: parameters

    def parms(self) -> dict[str, Any]:
        """Parameter tables of one or more nodes, sorted and paged across all of them.

        Every node asked for has an entry, even when nothing in it passes the
        filter. A parameter named twice, by its tuple and by a component, is one
        row. A multiparm named on its own pages through its instances.
        """
        wanted: dict[str, dict[str, Any]] = {}
        items: dict[tuple[str, ...], tuple[str, Any]] = {}
        for path in self.paths():
            try:
                node, only = _parm_target(self.hou, path, self.near)
                target = wanted.setdefault(
                    node.path(), {"node": node, "whole": False, "names": set()}
                )
                if only is None:
                    target["whole"] = True
                else:
                    target["names"].add(only)
            except Exception as error:  # noqa: BLE001 - one bad path is one bad entry
                if not self.batch:
                    raise
                items.setdefault(_key(path), ("error", _failed_entry(path, error)))
        for path, target in wanted.items():
            gathered: dict[tuple[str, ...], tuple[str, Any]] = {}
            try:
                self.gather(path, target, gathered)
            except Exception as error:  # noqa: BLE001 - one bad node is one bad entry
                if not self.batch:
                    raise
                gathered = {_key(path): ("error", _failed_entry(path, error))}
            items.update(gathered)
        keys = sorted(items)
        start = self.start_of(keys)
        grouped: dict[str, dict[str, Any]] = {}
        failed: set[str] = set()
        count = 0
        last = self.after
        for key in keys[start : start + self.limit]:
            if self.should_stop():
                break
            count += 1
            last = key
            kind, payload = items[key]
            if kind == "error":
                grouped[payload["path"]] = payload
                continue
            path = payload[1]
            if path in failed:
                continue
            try:
                self.add_to_page(grouped, kind, payload)
            except Exception as error:  # noqa: BLE001 - one bad node is one bad entry
                if not self.batch:
                    raise
                grouped[path] = _failed_entry(path, error)
                failed.add(path)
        result: dict[str, Any] = {"nodes": list(grouped.values()), "total": len(keys)}
        if start + count < len(keys):
            result["more"] = True
            result["last"] = _text_key(last)
        return result

    def gather(
        self, path: str, target: Mapping[str, Any], items: dict[tuple[str, ...], Any]
    ) -> None:
        """The page items one node gives: its rows, its instances, or itself when empty."""
        node = target["node"]
        names: set[str] = set(target["names"])
        node_key = _key(path)
        top, children = _parm_tree(node)
        if not target["whole"] and len(names) == 1:
            only = next(iter(names))
            tuple_ = next((item for item in top if item.name() == only), None)
            if tuple_ is not None and _is_multi(tuple_, _kind(tuple_)):
                groups = children.get(only, {})
                for index in sorted(groups):
                    chosen = self.choose(groups[index], children, lambda _: True, whole=True)
                    key = (*node_key, only, _INSTANCE_KEY.format(index))
                    items[key] = ("inst", (node, path, tuple_, index, chosen, len(groups)))
                if not groups:
                    items[(*node_key, only)] = ("row", (node, path, (tuple_, _kind(tuple_), [])))
                return
        if target["whole"]:
            test, whole = self.parm_test(None)
        else:
            test, whole = (lambda _: False), True

        def keeps(tuple_: Any) -> bool:
            return tuple_.name() in names or bool(test(tuple_))

        chosen = self.choose(top, children, keeps, whole=whole, named=names)
        for item in chosen:
            items[(*node_key, item[0].name())] = ("row", (node, path, item))
        if not chosen:
            items[node_key] = ("empty", (node, path))

    def add_to_page(self, grouped: dict[str, dict[str, Any]], kind: str, payload: Any) -> None:
        node, path = payload[0], payload[1]
        if path not in grouped:
            self.cook(node)
            grouped[path] = {"path": path, **self.marks(node), "parms": []}
        rows = grouped[path]["parms"]
        if kind == "row":
            rows.append(self.parm_row(payload[2], expressions=True))
        elif kind == "inst":
            _, _, tuple_, index, chosen, total = payload
            group = [self.parm_row(inner, expressions=True) for inner in chosen]
            if rows and rows[-1]["n"] == tuple_.name() and "inst_from" in rows[-1]:
                rows[-1]["inst"].append(group)
            else:
                rows.append({"n": tuple_.name(), "v": total, "inst_from": index, "inst": [group]})

    def parm_test(self, only: str | None) -> tuple[Any, bool]:
        """Which parameters a read keeps, and whether a kept multiparm keeps all of itself.

        A node read at `full` keeps every parameter unless a filter was given;
        everything else keeps the ones that differ from their defaults.
        """
        if only is not None:
            return (lambda tuple_: tuple_.name() == only), True
        wanted = str(self.arguments.get("parm_filter") or "")
        if not wanted:
            wanted = "all" if self.full and self.mode == "node" else "non_default"
        if wanted == "all":
            return (lambda tuple_: True), True
        if wanted == "non_default":
            return (lambda tuple_: not _at_default(tuple_)), False
        return (lambda tuple_: _glob_parm(tuple_, wanted)), True

    def parm_rows_of(self, node: Any) -> list[dict[str, Any]] | None:
        """The parameters a node entry carries at this level, or nothing."""
        top, children = _parm_tree(node)
        whole = False
        if self.standard:
            test, whole = self.parm_test(None)
        elif self.wants("expressions") or self.wants("code"):

            def test(tuple_: Any) -> bool:
                return (self.wants("expressions") and _animated(tuple_)) or (
                    self.wants("code") and _code_language(tuple_) is not None
                )
        else:
            return None
        chosen = self.choose(top, children, test, whole=whole)
        chosen.sort(key=lambda item: item[0].name())
        return [self.parm_row(item, expressions=self.wants("expressions")) for item in chosen]

    def choose(
        self,
        tuples: Any,
        children: Mapping[str, Any],
        test: Any,
        *,
        whole: bool = False,
        named: set[str] | None = None,
    ) -> list[Any]:
        """The parameters a test keeps, with the instances of each multiparm.

        A multiparm is kept when it passes, or when any of its instances do.
        Its instances are the ones that pass, or all of them when `whole` is
        set and the multiparm itself passed, which is how a glob or a name
        that picks out a multiparm reads. At most `MAX_INSTANCES` instances
        are looked at; the row says how many there are.
        """
        chosen: list[Any] = []
        for tuple_ in tuples:
            kind = _kind(tuple_)
            multi = _is_multi(tuple_, kind)
            if kind in _NO_VALUE or (kind == "Folder" and not multi):
                continue
            hit = bool(_quiet(lambda tuple_=tuple_: test(tuple_)))
            instances = None
            if multi:
                groups = children.get(tuple_.name(), {})
                picked_whole = hit and (whole or tuple_.name() in (named or ()))
                inner = (lambda _: True) if picked_whole else test
                instances = [
                    self.choose(groups[index], children, inner, whole=picked_whole)
                    for index in sorted(groups)[:MAX_INSTANCES]
                ]
                if not hit and not any(instances):
                    continue
                instances = _Instances(instances, len(groups))
            elif not hit:
                continue
            chosen.append((tuple_, kind, instances))
        return chosen

    def parm_row(self, item: Any, *, expressions: bool) -> dict[str, Any]:
        """One parameter as a row: its value, and where it comes from."""
        tuple_, kind, instances = item
        row: dict[str, Any] = {"n": tuple_.name()}
        parms = list(_quiet(lambda: list(tuple_)) or ())
        if self.full and parms and _ask(parms[0], "isSpare"):
            template = _ask(tuple_, "parmTemplate")
            row["spare"] = {"label": _ask(template, "label"), "type": kind}
        if any(_ask(parm, "isLocked") for parm in parms):
            row["lock"] = True
        if instances is not None:
            total = getattr(instances, "total", len(instances))
            row["v"] = total
            if kind == "Ramp":
                row["ramp"] = True
            row["inst"] = [
                [self.parm_row(inner, expressions=expressions) for inner in group]
                for group in instances
            ]
            if total > len(instances):
                row["inst_total"] = total
                row["truncated"] = True
            return row
        if kind == "Data":
            row["data"] = True
            return row
        code = _code_language(tuple_)
        if code is not None:
            row["code"] = code
            text = str(_ask(parms[0], "unexpandedString") or "") if parms else ""
            if self.wants("code"):
                row["v"] = text
            else:
                row["lines"] = text.count("\n") + 1 if text else 0
            return row
        template = _ask(tuple_, "parmTemplate")
        values: list[Any] = []
        written: dict[str, str] = {}
        languages: set[str] = set()
        withheld: str | None = None
        failed = keyed = False
        expanded: list[tuple[str, Any]] = []
        for parm in parms:
            if not self.evaluate and self.exports.drives(parm):
                withheld = withheld or OVERRIDE
                continue
            text, keys = _expression_of(parm)
            if keys:
                keyed = True
                if not self.evaluate:
                    withheld = withheld or KEYFRAMES
                    continue
            if text is not None:
                written[parm.name()] = text
                language = _language(parm)
                languages.add(language)
                keyed = keyed or bool(_CURVE.fullmatch(text))
                if not self.evaluate:
                    problem = _expression_problem(parm, text, language, 0, self.exports)
                    if problem:
                        withheld = withheld or problem
                        continue
            value = _value(parm, kind, template, evaluated=text is not None or keys)
            if value is _FAILED:
                failed = True
                continue
            values.append(value)
            if kind == "String" and text is None and not keys:
                expanded.append((str(value), self.expanded(parm, str(value))))
        if withheld:
            row["not_cooked"] = withheld
        elif failed:
            row["failed"] = True
        else:
            row["v"] = values[0] if len(values) == 1 else values
        if written and (expressions or withheld):
            row["expr"] = written
            row["lang"] = "python" if "python" in languages else "hscript"
        if keyed:
            row["keys"] = True
        held = next((value.reason for _, value in expanded if isinstance(value, _Held)), None)
        if held and not withheld:
            row["not_cooked"] = held
        elif any(value is not None for _, value in expanded) and not held:
            texts = [raw if value is None else value for raw, value in expanded]
            row["ev"] = texts[0] if len(texts) == 1 else texts
        return row

    def expanded(self, parm: Any, raw: str) -> Any:
        """A string as Houdini expands it, when that differs and is safe to ask."""
        if "$" not in raw and "`" not in raw:
            return None
        if not self.evaluate:
            problem = _string_problem(raw)
            if problem:
                return _Held(problem)
        value = _ask(parm, "evalAsString")
        return None if value is None or value == raw else str(value)


# Section: reading helpers for inspect

_FAILED = object()


class _Held:
    """A value left out, and why."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


class _Instances(list):
    """The instances of a multiparm a row carries, and how many there are."""

    def __init__(self, groups: list[Any], total: int) -> None:
        super().__init__(groups)
        self.total = total


def _choice(value: Any, allowed: tuple[str, ...], name: str, default: str) -> str:
    if value is None:
        return default
    if value not in allowed:
        raise BridgeError(
            "BAD_ARGUMENTS",
            f"{name} must be one of: " + ", ".join(allowed),
            {name: str(value), "did_you_mean": did_you_mean(str(value), allowed)},
        )
    return str(value)


def _key(path: str) -> tuple[str, ...]:
    """The sort key of a path: its parts, so a node sorts just before its insides."""
    return tuple(part for part in str(path).split("/") if part)


def _text_key(key: tuple[str, ...] | None) -> str:
    return "/" + "/".join(key or ())


def _tidy(path: str) -> str:
    """A path without a trailing or doubled slash, so one node has one spelling."""
    return _text_key(_key(path)) if path.startswith("/") else path


def _sorted_children(node: Any) -> list[Any]:
    children = list(_quiet(node.children) or ())
    return sorted(children, key=lambda child: str(_ask(child, "name") or ""))


def _scene_mark(hou: Any) -> str:
    """A short fingerprint of the scene's edit history.

    Every edit that can be undone leaves an entry on the undo stack, and a
    read leaves none, so the entries change between two pages exactly when
    the scene was edited in between. An edit made with undo turned off is not
    seen; the scene epoch and the resume point cover a scene replaced or a
    row taken away.
    """
    labels = _quiet(lambda: list(hou.undos.undoLabels()))
    text = "\n".join(str(label) for label in labels) if labels is not None else "?"
    return hashlib.sha256(f"{len(labels or ())}\n{text}".encode()).hexdigest()[:16]


def _node_at(hou: Any, path: str, near: _NearIndex | None = None) -> Any:
    """The node at a path, or why there is none.

    A path that names a parameter is `PATH_NOT_A_NODE`, which says which node
    holds it. Anything else that is not there is `NODE_NOT_FOUND` with the
    closest paths that are.
    """
    node = _quiet(lambda: hou.node(path))
    if node is not None:
        return node
    parm = _quiet(lambda: hou.parmTuple(path)) or _quiet(lambda: hou.parm(path))
    if parm is not None:
        holder = _quiet(lambda: parm.node().path())
        raise BridgeError(
            "PATH_NOT_A_NODE",
            f"{path} is a parameter, not a node",
            {"path": path, "node": holder, "parm": _ask(parm, "name")},
            hint="read it with mode parms, or ask for the node that holds it",
        )
    finder = near or _NearIndex(hou, MAX_NEAR)
    raise BridgeError(
        "NODE_NOT_FOUND",
        f"no node at {path}",
        {"path": path, "did_you_mean": finder.near(path)},
        hint="read the tree above it and use a path that is there",
    )


def _parm_target(hou: Any, path: str, near: _NearIndex | None = None) -> tuple[Any, str | None]:
    """A node whose table is wanted, or the node and name of one parameter."""
    node = _quiet(lambda: hou.node(path))
    if node is not None:
        return node, None
    found = _quiet(lambda: hou.parmTuple(path))
    if found is None:
        single = _quiet(lambda: hou.parm(path))
        found = _quiet(lambda: single.tuple()) if single is not None else None
    if found is not None:
        return found.node(), found.name()
    return _node_at(hou, path, near), None


def _failed_entry(path: str, error: BaseException) -> dict[str, Any]:
    coded = map_exception(error, tool="node.inspect")
    said: dict[str, Any] = {"code": coded.code, "message": coded.message}
    if coded.hint:
        said["hint"] = coded.hint
    near = coded.details.get("did_you_mean")
    if near:
        said["did_you_mean"] = near
    return {"path": path, "error": said}


def _source(connection: Any) -> str | None:
    """Where one wire comes from: a node's path, or a network's own input."""
    upstream = _ask(connection, "inputNode")
    if upstream is not None:
        return str(upstream.path())
    item = _ask(connection, "inputItem")
    return None if item is None else _ask(item, "path")


def _default_name(node: Any) -> bool:
    """Whether a node still has the name its type gave it, such as `box1`."""
    name = str(_ask(node, "name") or "")
    stem = _TRAILING_DIGITS.sub("", name)
    if not stem or stem == name:
        return False
    node_type = _ask(node, "type")
    parts = _ask(node_type, "nameComponents") if node_type is not None else None
    base = str(parts[2]) if parts and len(parts) > 2 else str(_ask(node_type, "name") or "")
    words = _NOT_A_WORD.sub("", str(_ask(node_type, "description") or "").lower())
    return stem in {base, words}


def _position(item: Any) -> dict[str, list[float]]:
    """Where an item sits in its network, when the build will say."""
    place = _quiet(lambda: list(item.position()))
    return {} if place is None else {"pos": [round(float(value), 3) for value in place]}


def _cut(value: Any) -> Any:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "..."


def _glob_node(node: Any, pattern: str) -> bool:
    """A glob on the full path when it has a slash in it, on the name when not."""
    subject = node.path() if "/" in pattern else _ask(node, "name")
    return bool(subject) and fnmatchcase(str(subject), pattern)


def _glob_type(node: Any, pattern: str) -> bool:
    node_type = _ask(node, "type")
    if node_type is None:
        return False
    names = (_ask(node_type, "name"), _ask(node_type, "nameWithCategory"))
    return any(name and fnmatchcase(str(name), pattern) for name in names)


def _glob_parm(tuple_: Any, pattern: str) -> bool:
    names = [tuple_.name(), *(parm.name() for parm in _quiet(lambda: list(tuple_)) or ())]
    return any(fnmatchcase(str(name), pattern) for name in names)


def _parm_tree(node: Any) -> tuple[list[Any], dict[str, dict[int, list[Any]]]]:
    """A node's own parameters, and the instances under each multiparm.

    Instances are grouped under the name of the multiparm that holds them, by
    their instance number, in the order the node lists them.
    """
    top: list[Any] = []
    children: dict[str, dict[int, list[Any]]] = {}
    for tuple_ in _quiet(node.parmTuples) or ():
        first = _quiet(lambda tuple_=tuple_: tuple_[0])
        if first is None:
            continue
        if _ask(first, "isMultiParmInstance"):
            parent = _ask(first, "parentMultiParm")
            indices = _ask(first, "multiParmInstanceIndices") or ()
            if parent is None or not indices:
                continue
            group = children.setdefault(str(parent.name()), {})
            group.setdefault(int(indices[-1]), []).append(tuple_)
        else:
            top.append(tuple_)
    return top, children


def _kind(tuple_: Any) -> str:
    """The parameter's kind in the build's own words: Float, String, Ramp."""
    kind = _quiet(lambda: tuple_.parmTemplate().type().name())
    return str(kind) if kind else ""


def _is_multi(tuple_: Any, kind: str) -> bool:
    if kind == "Ramp":
        return True
    if kind != "Folder":
        return False
    folder = _quiet(lambda: tuple_.parmTemplate().folderType())
    return "multiparm" in str(folder).lower()


def _has_value(tuple_: Any) -> bool:
    kind = _kind(tuple_)
    return kind != "Folder" or _is_multi(tuple_, kind)


def _components(tuple_: Any) -> list[Any]:
    return list(_quiet(lambda: list(tuple_)) or ())


def _at_default(tuple_: Any) -> bool:
    """Whether a parameter is at its default, comparing expressions as text.

    Comparing values would evaluate an expression, which can cook whatever it
    reads, so the text of an expression is what is compared. A parameter a
    channel operator drives is never at its default, whatever it reads.
    """
    if any(_quiet(parm.isOverrideTrackActive) for parm in _components(tuple_)):
        return False
    return bool(_quiet(lambda: tuple_.isAtDefault(compare_expressions=True)))


def _animated(tuple_: Any) -> bool:
    for parm in _components(tuple_):
        text, keys = _expression_of(parm)
        if text is not None or keys or _quiet(parm.isOverrideTrackActive):
            return True
    return False


def _expression_of(parm: Any) -> tuple[str | None, bool]:
    """A parameter's expression, and whether it has keyframes it will not show.

    The build hands back the expression of a parameter with one keyframe and
    refuses one with more. Reading the keyframes themselves evaluates them,
    which can cook, so a refusal that is not the plain "not animated" is taken
    to mean keyframes and nothing more is asked.
    """
    try:
        return str(parm.expression()), False
    except Exception as error:  # noqa: BLE001 - the refusal is the answer
        return None, "not animated" not in str(error).lower()


def _code_language(tuple_: Any) -> str | None:
    """The language of a parameter that holds code, such as `vex` or `python`."""
    tags = _quiet(lambda: tuple_.parmTemplate().tags()) or {}
    language = tags.get("editorlang") if isinstance(tags, Mapping) else None
    return str(language).lower() if language else None


def _language(parm: Any) -> str:
    said = str(_quiet(parm.expressionLanguage) or "")
    return "python" if "python" in said.lower() else "hscript"


def _value(parm: Any, kind: str, template: Any, *, evaluated: bool) -> Any:
    """One component's value, as a person reads it in the parameter pane."""
    if kind == "String":
        read = parm.evalAsString if evaluated else parm.unexpandedString
        value = _quiet(read)
        return _FAILED if value is None else str(value)
    value = _quiet(parm.eval)
    if value is None:
        return _FAILED
    if kind == "Toggle":
        return bool(value)
    if kind == "Menu":
        items = _quiet(template.menuItems) if template is not None else None
        if items and isinstance(value, int) and 0 <= value < len(items):
            return str(items[value])
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (int, str, bool)):
        return value
    return str(value)


def _variable_allowed(name: str) -> bool:
    return name in SAFE_VARIABLES or bool(_PADDED_FRAME.fullmatch(name))


def _string_problem(raw: str) -> str | None:
    """Why expanding a string outside a cook would be wrong, or nothing."""
    if "`" in raw:
        return BACKTICK
    for name in _VARIABLE_NAME.findall(raw):
        if not _variable_allowed(name):
            return f"variable ${name}"
    return None


def _parm_problem(parm: Any, depth: int, exports: _Exports) -> str | None:
    """Why reading a parameter's value could cook, or nothing when it cannot."""
    if exports.drives(parm):
        return OVERRIDE
    text, keys = _expression_of(parm)
    if keys:
        return KEYFRAMES
    if text is None:
        raw = _quiet(parm.unexpandedString)
        return None if raw is None else _string_problem(str(raw))
    return _expression_problem(parm, text, _language(parm), depth, exports)


def _expression_problem(
    parm: Any, text: str, language: str, depth: int, exports: _Exports
) -> str | None:
    """Why evaluating an expression could cook, or nothing when it cannot.

    Only HScript is read, token by token: numbers, operators, the variables
    and functions that read nothing from the scene, and references to another
    parameter written as one quoted name, which are followed a few steps and
    are as safe as what they reach. A reference is never read out of a
    string, and one whose target is worked out is refused.
    """
    if language != "hscript":
        return PYTHON
    if depth > MAX_REFERENCE_DEPTH:
        return "references too deep"
    tokens = _TOKEN.findall(text)
    for index, token in enumerate(tokens):
        if token == "`":
            return BACKTICK
        if token[0] in "\"'":
            if len(token) < 2 or token[-1] != token[0]:
                return "unreadable expression"
            problem = _string_problem(token[1:-1])
            if problem:
                return problem
            continue
        if token[0] == "$":
            name = token.strip("${}")
            if not _variable_allowed(name):
                return f"variable ${name}"
            continue
        if not (token[0].isalpha() or token[0] == "_"):
            continue
        called = index + 1 < len(tokens) and tokens[index + 1] == "("
        if not called:
            return f"reads {token}"
        if token in REFERENCE_FUNCTIONS:
            problem = _reference_problem(parm, token, tokens[index + 2 : index + 4], depth, exports)
            if problem:
                return problem
        elif token not in SAFE_FUNCTIONS:
            return f"calls {token}()"
    return None


def _reference_problem(
    parm: Any, function: str, rest: list[str], depth: int, exports: _Exports
) -> str | None:
    """Whether one `ch()` and its like reaches only what is safe to read."""
    if len(rest) != 2 or rest[1] != ")" or rest[0][0] not in "\"'" or len(rest[0]) < 2:
        return f"{function}() of a worked out name"
    target = rest[0][1:-1]
    if not target or any(mark in target for mark in ("$", "`", "\\")):
        return f"{function}() of a worked out name"
    node = _quiet(parm.node)
    referenced = _quiet(lambda: node.parm(target)) if node is not None else None
    if referenced is None:
        return f"{function}() of a parameter that is not there"
    problem = _parm_problem(referenced, depth + 1, exports)
    return f'{function}("{target}"): {problem}' if problem else None


class _Exports:
    """Which parameters a channel operator's export may drive, found without a cook.

    A parameter an export drives says so once the channel operator has
    cooked. Before that first cook it does not, yet reading it cooks the
    channel operator. So a node that an exporting channel operator depends on
    and that has never cooked has every number on it taken as driven.
    """

    def __init__(self) -> None:
        self._waiting: dict[str, bool] = {}

    def drives(self, parm: Any) -> bool:
        if _quiet(parm.isOverrideTrackActive):
            return True
        if _quiet(lambda: parm.parmTemplate().type().name()) == "String":
            return False
        node = _quiet(parm.node)
        if node is None:
            return False
        path = str(node.path())
        if path not in self._waiting:
            found = _quiet(lambda: node.dependents(include_children=False)) or ()
            self._waiting[path] = any(
                _ask(other, "isExportFlagSet") is True and not _ask(other, "cookCount")
                for other in found
            )
        return self._waiting[path]


# How alike two names must be to be offered as a near miss.
NEAR_CUTOFF = 0.5


class _NearIndex:
    """The closest existing paths to ones that are not there.

    The nodes under an ancestor are gathered once, level by level up to a
    bound, and shared by every path in a batch that falls back to the same
    ancestor.
    """

    def __init__(self, hou: Any, limit: int) -> None:
        self.hou = hou
        self.limit = limit
        self._below: dict[str, list[tuple[str, str, int]]] = {}

    def near(self, path: str) -> list[str]:
        parts = [part for part in str(path).split("/") if part]
        leaf = parts[-1] if parts else ""
        while parts:
            parts.pop()
            above = "/" + "/".join(parts)
            found = _quiet(lambda above=above: self.hou.node(above))
            if found is not None:
                return self.rank(leaf, self.below(above, found))
        return []

    def below(self, above: str, root: Any) -> list[tuple[str, str, int]]:
        if above not in self._below:
            seen: list[tuple[str, str, int]] = []
            level = [(child, 1) for child in _quiet(root.children) or ()]
            while level and len(seen) < NEAR_SCAN:
                following: list[tuple[Any, int]] = []
                for node, depth in level:
                    if len(seen) >= NEAR_SCAN:
                        break
                    seen.append((str(_ask(node, "name") or ""), str(node.path()), depth))
                    if not _ask(node, "isLockedHDA"):
                        following.extend(
                            (child, depth + 1) for child in _quiet(node.children) or ()
                        )
                level = following
            self._below[above] = seen
        return self._below[above]

    def rank(self, leaf: str, candidates: list[tuple[str, str, int]]) -> list[str]:
        """Candidates whose names are close to the last part, closest first."""
        if not leaf:
            return []
        wanted = leaf.lower()
        scored = []
        for name, path, depth in candidates:
            lowered = name.lower()
            ratio = SequenceMatcher(None, wanted, lowered).ratio()
            if ratio >= NEAR_CUTOFF or (wanted and lowered.startswith(wanted)):
                scored.append((-ratio, depth, path))
        scored.sort()
        return [path for _, _, path in scored[: self.limit]]


# Section: running Python

# How long a namespace nobody has used is kept before it is dropped.
NAMESPACE_IDLE_S = 24 * 60 * 60.0

# How much of what the code prints is kept, the end of it. The caller is
# handed a much shorter tail; this is what a spilled result can hold.
CAPTURE_MAX_CHARS = 512_000

# How many lines of a traceback come back.
TRACEBACK_LINES = 20

# How much of an exception's own message comes back.
MAX_MESSAGE = 2000

# How many recent snippets keep their source for tracebacks to quote. A
# function defined in an older one still runs, its lines just go unquoted.
SOURCES_KEPT = 64

_NAMESPACE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@dataclass
class _Kept:
    values: dict[str, Any]
    helper: Helper
    used_at: float


class Namespaces:
    """The Python namespaces one session keeps between calls, by name.

    Each is one dict, seeded with `hou` and the `mcp` helper and nothing
    else, and kept until a call resets it, the session ends, or nobody has
    used it for a day. Separate dicts keep variables apart and nothing more:
    every namespace works on the same scene.

    The code runs as `python.run`, a change like any other, so it is in one
    undo group, holds the session while it runs and takes a receipt under its
    operation id. What it raises is not a failed call: it comes back as data,
    with the namespace as it was at the raise and the scene as the code left
    it, so one undo takes the whole call back.
    """

    def __init__(self, *, clock: Any = time.monotonic, idle_s: float = NAMESPACE_IDLE_S) -> None:
        self._clock = clock
        self._idle_s = idle_s
        self._kept: dict[str, _Kept] = {}
        self._lock = threading.Lock()
        self._sources: deque[str] = deque()
        self._count = 0

    def run(self, arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
        """Run one snippet and say what it left, printed and raised."""
        hou = _houdini(context)
        code = arguments.get("code")
        name = arguments.get("namespace")
        if not isinstance(code, str):
            raise BridgeError("BAD_ARGUMENTS", "code must be text")
        if not isinstance(name, str) or not _NAMESPACE_NAME.match(name):
            raise BridgeError(
                "BAD_ARGUMENTS",
                "namespace must be 1 to 64 letters, digits, dot, dash or underscore",
                {"argument": "namespace"},
            )
        kept = self._take(name, reset=bool(arguments.get("reset")), hou=hou)
        values = kept.values
        kept.helper._context = context
        # The two names every snippet can count on are put back each time,
        # and a result from an earlier call never passes for this one's.
        values["hou"] = hou
        values["mcp"] = kept.helper
        values.pop("result", None)

        capture = _Capture(CAPTURE_MAX_CHARS)
        error: dict[str, Any] | None = None
        began = time.monotonic()
        try:
            compiled = compile(code, self._source_name(code), "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as raised:
            error = _syntax_error(raised)
        else:
            with capture.installed():
                try:
                    exec(compiled, values)
                except BaseException as raised:  # noqa: BLE001 - the code's own failure is data
                    error = _raised(raised)
        finally:
            with self._lock:
                kept.used_at = self._clock()
        answer: dict[str, Any] = {
            "result": values.get("result"),
            "stdout": capture.text(),
            "stdout_dropped": capture.dropped,
            "duration_ms": round((time.monotonic() - began) * 1000.0, 3),
            "namespace": name,
        }
        if error is not None:
            answer["error"] = error
        return answer

    def _take(self, name: str, *, reset: bool, hou: Any) -> _Kept:
        with self._lock:
            self._drop_idle()
            kept = self._kept.get(name)
            if kept is None or reset:
                helper = Helper()
                kept = _Kept({"hou": hou, "mcp": helper}, helper, self._clock())
                self._kept[name] = kept
            kept.used_at = self._clock()
            return kept

    def _drop_idle(self) -> None:
        now = self._clock()
        for name in [key for key, kept in self._kept.items() if now - kept.used_at > self._idle_s]:
            del self._kept[name]

    def _source_name(self, code: str) -> str:
        """A name for one snippet, with its lines kept for a traceback to quote."""
        with self._lock:
            self._count += 1
            name = f"<hou_python {self._count}>"
            self._sources.append(name)
            while len(self._sources) > SOURCES_KEPT:
                linecache.cache.pop(self._sources.popleft(), None)
        linecache.cache[name] = (len(code), None, code.splitlines(True), name)
        return name


class Helper:
    """The `mcp` object in every namespace.

    `output_path(kind, name, ext)` hands out a managed path for this session
    and scene, from the same table and the same version sequence the server
    uses: render, flipbook, comp, cache, usd, hip, capture or compare.
    `progress(done, total, message)` leaves a note health shows while the
    call runs. `cancelled()` says whether somebody asked this call to stop,
    for a long loop to look at between pieces of work.
    """

    def __init__(self) -> None:
        self._context: ToolContext | None = None

    def __repr__(self) -> str:
        return "<mcp: output_path(kind, name, ext), progress(done, total, message), cancelled()>"

    def output_path(self, kind: str, name: str | None = None, ext: str | None = None) -> str:
        from nscr_houdini_mcp import outputs

        context = self._now()
        if context.home is None or context.open_store is None:
            raise RuntimeError("this session keeps no state folder, so it has no output paths")
        hou = _houdini(context)
        hip = None if _quiet(hou.hipFile.isNewFile) else _quiet(hou.hipFile.path)
        # The same scratch folder the server picks for a scene with no file.
        scratch = None if os.environ.get("HOUDINI_TEMP_DIR") else Path(context.home) / "temp"
        conventions = outputs.load_conventions(home=context.home, hip_path=hip)
        with context.open_store() as store:
            plan = outputs.allocate(
                store,
                str(kind),
                name=None if name is None else str(name),
                hip_path=hip,
                session_id=context.session_id or None,
                ext=None if ext is None else str(ext),
                conventions=conventions,
                scratch_root=scratch,
            )
        return plan.path

    def progress(self, done: float, total: float | None = None, message: str | None = None) -> None:
        for value, label in ((done, "done"), (total, "total")):
            number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not number and (value is not None or label == "done"):
                raise TypeError(f"{label} must be a number")
        note = self._now().progress
        if note is not None:
            note(
                {
                    "done": done,
                    "total": total,
                    "message": None if message is None else str(message)[:200],
                }
            )

    def cancelled(self) -> bool:
        return self._now().should_stop()

    def _now(self) -> ToolContext:
        if self._context is None:
            raise RuntimeError("mcp only works inside a hou_python call")
        return self._context


class _Capture:
    """What one call's code prints, and nothing any other thread prints.

    `sys.stdout` and `sys.stderr` belong to the whole process, so while the
    code runs they are swapped for a router that keeps writes from this
    thread and hands every other thread's straight on to what was there. The
    end is kept, up to a bound, and what was dropped from the front is counted.
    """

    def __init__(self, limit: int) -> None:
        self.owner = threading.get_ident()
        self.limit = limit
        self.dropped = 0
        self._parts: list[str] = []
        self._size = 0

    def add(self, text: str) -> None:
        self._parts.append(text)
        self._size += len(text)
        if self._size > 2 * self.limit:
            self._trim()

    def text(self) -> str:
        self._trim()
        return "".join(self._parts)

    def _trim(self) -> None:
        whole = "".join(self._parts)
        if len(whole) > self.limit:
            self.dropped += len(whole) - self.limit
            whole = whole[-self.limit :]
        self._parts = [whole] if whole else []
        self._size = len(whole)

    @contextmanager
    def installed(self) -> Iterator[None]:
        out, err = sys.stdout, sys.stderr
        routed_out, routed_err = _Routed(out, self), _Routed(err, self)
        sys.stdout, sys.stderr = routed_out, routed_err
        try:
            yield
        finally:
            # Code that set streams of its own keeps them.
            if sys.stdout is routed_out:
                sys.stdout = out
            if sys.stderr is routed_err:
                sys.stderr = err


class _Routed:
    """A stream that keeps one thread's writes and passes on the rest."""

    def __init__(self, fallback: Any, capture: _Capture) -> None:
        self._fallback = fallback
        self._capture = capture

    def write(self, text: Any) -> int:
        text = str(text)
        if threading.get_ident() == self._capture.owner:
            self._capture.add(text)
            return len(text)
        if self._fallback is None:
            return len(text)
        return self._fallback.write(text)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        if threading.get_ident() != self._capture.owner and self._fallback is not None:
            self._fallback.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fallback, name)


def _raised(error: BaseException) -> dict[str, Any]:
    """An exception the code raised, as data with no place on disk in it."""
    frames = error.__traceback__
    # The first frame is the call into the code, which says nothing about it.
    frames = frames.tb_next if frames is not None else None
    lines = "".join(traceback.format_exception(type(error), error, frames)).splitlines()
    return {
        "type": type(error).__name__,
        "message": hide_paths(_message(error))[:MAX_MESSAGE],
        "traceback_tail": hide_paths("\n".join(lines[-TRACEBACK_LINES:])),
    }


def _syntax_error(error: BaseException) -> dict[str, Any]:
    lines = "".join(traceback.format_exception_only(type(error), error)).splitlines()
    said: dict[str, Any] = {
        "type": type(error).__name__,
        "message": hide_paths(str(getattr(error, "msg", None) or _message(error)))[:MAX_MESSAGE],
        "line": getattr(error, "lineno", None),
        "offset": getattr(error, "offset", None),
        "traceback_tail": hide_paths("\n".join(lines[-TRACEBACK_LINES:])),
    }
    return said


def _message(error: BaseException) -> str:
    try:
        return str(error)
    except Exception:  # noqa: BLE001 - an exception that will not describe itself
        return ""


# Section: shared helpers


def _houdini(context: ToolContext) -> Any:
    if context.hou is None:
        raise BridgeError(
            "TOOL_FAILED",
            "this tool needs a Houdini and this process has none",
            hint="send the call to a session, not to a bare bridge",
        )
    return context.hou


def _node(hou: Any, path: str) -> Any:
    """The node at a path, or a not found error with the closest paths."""
    node = _quiet(lambda: hou.node(path))
    if node is None:
        raise BridgeError(
            "NODE_NOT_FOUND",
            f"no node at {path}",
            {"path": path, "did_you_mean": _near(hou, path)},
            hint="read the scene and use a path that is there",
        )
    return node


def _near(hou: Any, path: str, *, limit: int = MAX_HINTS) -> list[str]:
    """Paths close to one that is not there, closest first."""
    return _NearIndex(hou, limit).near(path)


def _types(parent: Any, wanted: str) -> list[str]:
    """Child type names close to one the network does not have."""
    category = _quiet(parent.childTypeCategory)
    if category is None:
        return []
    types = _quiet(category.nodeTypes) or {}
    return did_you_mean(wanted, list(types))


def _parms(node: Any, wanted: str) -> list[str]:
    parms = _quiet(node.parms) or ()
    return did_you_mean(wanted, [parm.name() for parm in parms])


def _number(value: Any, name: str, cap: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeError("BAD_ARGUMENTS", f"{name} must be a number")
    if value < 0 or value > cap:
        raise BridgeError("BAD_ARGUMENTS", f"{name} must be between 0 and {cap:g}")
    return float(value)


def _is_hou_error(error: BaseException) -> bool:
    return type(error).__module__.split(".")[0] == "hou"


def _ask(owner: Any, method: str, *args: Any) -> Any:
    """Call one method of a Houdini object, or nothing when it is not there."""
    return _quiet(lambda: getattr(owner, method)(*args))


def _quiet(read: Any) -> Any:
    """Read one fact, or nothing when this build or session will not say."""
    try:
        return read()
    except Exception:  # noqa: BLE001 - a fact we cannot read is a fact we do not have
        return None
