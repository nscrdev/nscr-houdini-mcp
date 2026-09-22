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
- `bridge.selfcheck` mutates on purpose, and can be asked to take its time, to
  fail part way, or to throw the scene away, so the queue, the timeout, the
  rollback, the scene epoch and the receipts can be tried against a real
  Houdini rather than only against a stand in. It is registered in a worker
  this project started to be driven and nowhere else.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge.errors import MAX_HINTS, BridgeError, did_you_mean, map_exception

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

# Parameter kinds with no value to read, and the ones that hold a list of
# instances rather than a value.
_NO_VALUE = frozenset({"Button", "FolderSet", "Separator", "Label"})

# What an expression may use and still be read without a cook. Anything else,
# a Python expression, a backtick, or a variable that reads geometry, is left
# alone unless the call says it may cook.
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
    }
)
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
# when what it points at is.
REFERENCE_FUNCTIONS = frozenset({"ch", "chf", "chs", "chsraw"})
MAX_REFERENCE_DEPTH = 4

_CALL = re.compile(r"([A-Za-z_]\w*)\s*\(")
_VARIABLE_NAME = re.compile(r"\$\{?([A-Za-z_]\w*)\}?")
_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_REFERENCE = re.compile(r"\b(?:ch|chf|chs|chsraw)\s*\(\s*(\"[^\"]*\"|'[^']*')")
_CURVE = re.compile(r"\s*(" + "|".join(sorted(CURVE_FUNCTIONS)) + r")\s*\(\s*\)\s*")
_TRAILING_DIGITS = re.compile(r"\d+$")
_NOT_A_WORD = re.compile(r"[^a-z0-9]")

# How long a text value in user data may be before it is cut.
MAX_TEXT = 2000


def inspect(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Read nodes, networks and parameters, one page of rows at a time.

    Nothing is cooked unless `evaluate` is set. Without it every value that
    would pull on a cook is left out and marked, and what a node reports about
    its last cook is marked when it has never cooked or is out of date.
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
        return result

    # Section: which nodes

    def tree(self) -> dict[str, Any]:
        root = _node_at(self.hou, str(self.arguments.get("path") or "/"))
        depth = int(_number(self.arguments.get("depth") or 1, "depth", MAX_TREE_DEPTH))
        if depth < 1:
            raise BridgeError("BAD_ARGUMENTS", "depth must be at least 1")
        found: list[Any] = []
        networks: list[Any] = []
        truncated = False
        stack = [(root, 0)]
        while stack and not truncated:
            node, level = stack.pop()
            if node is not root and _ask(node, "isLockedHDA"):
                continue
            children = _quiet(node.children) or ()
            if children:
                networks.append(node)
            for child in children:
                if len(found) >= MAX_NODES_SCANNED:
                    truncated = True
                    break
                found.append(child)
                if level + 1 < depth:
                    stack.append((child, level + 1))
        result = self.page_of_nodes(found)
        if self.after is None and self.wants("notes"):
            result.update(self.network_items(networks))
        if truncated:
            result["truncated"] = True
        return result

    def find(self) -> dict[str, Any]:
        root = _node_at(self.hou, str(self.arguments.get("path") or "/"))
        pattern = self.arguments.get("pattern")
        type_glob = self.arguments.get("type")
        if not pattern and not type_glob:
            raise BridgeError("BAD_ARGUMENTS", "find needs pattern, type or both")
        candidates = (
            _quiet(lambda: root.allSubChildren(top_down=True, recurse_in_locked_nodes=False)) or ()
        )
        found: list[Any] = []
        truncated = False
        for index, node in enumerate(candidates):
            if index >= MAX_NODES_SCANNED:
                truncated = True
                break
            if pattern and not _glob_node(node, str(pattern)):
                continue
            if type_glob and not _glob_type(node, str(type_glob)):
                continue
            found.append(node)
        result = self.page_of_nodes(found)
        if truncated:
            result["truncated"] = True
        return result

    def selection(self) -> dict[str, Any]:
        if self.context.kind != "gui":
            return {"rows": [], "total": 0, "digest": _digest([]), "note": NO_SELECTION}
        return self.page_of_nodes(list(_quiet(self.hou.selectedNodes) or ()))

    def page_of_nodes(self, nodes: list[Any]) -> dict[str, Any]:
        """One page of rows, sorted by path, starting after the last one seen."""
        keyed = sorted(((_key(node.path()), node) for node in nodes), key=lambda pair: pair[0])
        rows: list[dict[str, Any]] = []
        start = self.start_of([key for key, _ in keyed])
        for _key_of, node in keyed[start : start + self.limit]:
            if self.context.should_stop():
                self.stopped = True
                break
            self.cook(node)
            rows.append(self.row(node))
        return {"rows": rows, **self.paged([key for key, _ in keyed], start, len(rows))}

    def start_of(self, keys: list[tuple[str, ...]]) -> int:
        if self.after is None:
            return 0
        for index, key in enumerate(keys):
            if key > self.after:
                return index
        return len(keys)

    def paged(self, keys: list[tuple[str, ...]], start: int, count: int) -> dict[str, Any]:
        """How many rows matched, and where the next page starts when there is one."""
        result: dict[str, Any] = {"total": len(keys), "digest": _digest(keys)}
        if start + count < len(keys):
            result["more"] = True
            result["last"] = _text_key(keys[start + count - 1] if count else self.after)
        return result

    # Section: one node

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
            if _kind(tuple_) not in _NO_VALUE and not _at_default(tuple_) and _has_value(tuple_)
        )

    def network_items(self, networks: list[Any]) -> dict[str, Any]:
        boxes: list[dict[str, Any]] = []
        notes: list[dict[str, Any]] = []
        for network in networks:
            for box in _quiet(network.networkBoxes) or ():
                item: dict[str, Any] = {"path": box.path()}
                comment = _ask(box, "comment")
                if comment:
                    item["comment"] = comment
                inside = [node.path() for node in _quiet(box.nodes) or ()]
                if inside:
                    item["nodes"] = inside
                if self.full:
                    item.update(_position(box))
                    size = _quiet(lambda box=box: list(box.size()))
                    if size is not None:
                        item["size"] = [round(float(value), 3) for value in size]
                boxes.append(item)
            for note in _quiet(network.stickyNotes) or ():
                notes.append({"path": note.path(), "text": str(_ask(note, "text") or "")})
        found: dict[str, Any] = {}
        if boxes:
            found["boxes"] = boxes[: self.limit]
        if notes:
            found["notes"] = notes[: self.limit]
        return found

    # Section: several nodes by path

    def nodes(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for path in self.paths():
            if self.context.should_stop():
                self.stopped = True
                break
            try:
                node = _node_at(self.hou, path)
                self.cook(node)
                entries.append(self.entry(node))
            except Exception as error:  # noqa: BLE001 - one bad path is one bad entry
                if not self.batch:
                    raise
                entries.append(_failed_entry(path, error))
        return {"nodes": entries}

    def paths(self) -> list[str]:
        given = self.arguments.get("paths") or ()
        if isinstance(given, str) or not given:
            raise BridgeError("BAD_ARGUMENTS", "this mode needs paths")
        if len(given) > MAX_BATCH:
            raise BridgeError("BAD_ARGUMENTS", f"at most {MAX_BATCH} paths in one call")
        return list(dict.fromkeys(str(path) for path in given))

    # Section: parameters

    def parms(self) -> dict[str, Any]:
        """The parameter tables of one or more nodes, paged across all of them."""
        failed: list[dict[str, Any]] = []
        chosen: list[tuple[tuple[str, ...], Any, Any]] = []
        for path in self.paths():
            try:
                node, only = _parm_target(self.hou, path)
            except Exception as error:  # noqa: BLE001 - one bad path is one bad entry
                if not self.batch:
                    raise
                failed.append(_failed_entry(path, error))
                continue
            top, children = _parm_tree(node)
            test, whole = self.parm_test(only)
            node_key = _key(node.path())
            for item in self.choose(top, children, test, whole=whole):
                chosen.append(((*node_key, item[0].name()), node, item))
        chosen.sort(key=lambda entry: entry[0])
        keys = [key for key, _, _ in chosen]
        start = self.start_of(keys)
        grouped: dict[str, dict[str, Any]] = {}
        count = 0
        for _, node, item in chosen[start : start + self.limit]:
            if self.context.should_stop():
                self.stopped = True
                break
            path = node.path()
            if path not in grouped:
                self.cook(node)
                grouped[path] = {"path": path, **self.marks(node), "parms": []}
            grouped[path]["parms"].append(self.parm_row(item, expressions=True))
            count += 1
        result = self.paged(keys, start, count)
        entries = list(grouped.values())
        if self.after is None:
            entries.extend(failed)
            entries.sort(key=lambda entry: _key(entry["path"]))
        result["nodes"] = entries
        return result

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
        self, tuples: Any, children: Mapping[str, Any], test: Any, *, whole: bool = False
    ) -> list[Any]:
        """The parameters a test keeps, with the instances of each multiparm.

        A multiparm is kept when it passes, or when any of its instances do.
        Its instances are the ones that pass, or all of them when `whole` is
        set and the multiparm itself passed, which is how a glob or a name
        that picks out a multiparm reads.
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
                inner = (lambda _: True) if hit and whole else test
                instances = [
                    self.choose(groups[index], children, inner, whole=whole)
                    for index in sorted(groups)
                ]
                if not hit and not any(instances):
                    continue
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
            row["v"] = len(instances)
            if kind == "Ramp":
                row["ramp"] = True
            row["inst"] = [
                [self.parm_row(inner, expressions=expressions) for inner in group]
                for group in instances
            ]
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
        withheld = failed = keyed = False
        expanded: list[tuple[str, Any]] = []
        for parm in parms:
            text = _quiet(parm.expression)
            if text is not None:
                written[parm.name()] = str(text)
                language = _language(parm)
                languages.add(language)
                keyed = keyed or bool(_CURVE.fullmatch(str(text)))
                if not self.evaluate and not _safe_expression(parm, str(text), language, 0):
                    withheld = True
                    continue
            value = _value(parm, kind, template, evaluated=text is not None)
            if value is _FAILED:
                failed = True
                continue
            values.append(value)
            if kind == "String" and text is None:
                expanded.append((str(value), self.expanded(parm, str(value))))
        if withheld:
            row["not_cooked"] = True
        elif failed:
            row["failed"] = True
        else:
            row["v"] = values[0] if len(values) == 1 else values
        if written and (expressions or withheld):
            row["expr"] = written
            row["lang"] = "python" if "python" in languages else "hscript"
        if keyed:
            row["keys"] = True
        found = [value for _, value in expanded]
        if any(value is _WITHHELD for value in found):
            row["not_cooked"] = True
        elif any(value is not None for value in found):
            texts = [raw if value is None else value for raw, value in expanded]
            row["ev"] = texts[0] if len(texts) == 1 else texts
        return row

    def expanded(self, parm: Any, raw: str) -> Any:
        """A string as Houdini expands it, when that differs and is safe to ask."""
        if "$" not in raw and "`" not in raw:
            return None
        if "`" in raw and not self.evaluate:
            return _WITHHELD
        value = _ask(parm, "evalAsString")
        return None if value is None or value == raw else str(value)


# Section: reading helpers for inspect

_FAILED = object()
_WITHHELD = object()


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


def _digest(keys: list[tuple[str, ...]]) -> str:
    """A short fingerprint of the rows a read matched, to tell pages apart."""
    text = "\n".join(_text_key(key) for key in keys)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _node_at(hou: Any, path: str) -> Any:
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
    raise BridgeError(
        "NODE_NOT_FOUND",
        f"no node at {path}",
        {"path": path, "did_you_mean": _near(hou, path, limit=MAX_NEAR)},
        hint="read the tree above it and use a path that is there",
    )


def _parm_target(hou: Any, path: str) -> tuple[Any, str | None]:
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
    return _node_at(hou, path), None


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


def _at_default(tuple_: Any) -> bool:
    """Whether a parameter is at its default, comparing expressions as text.

    Comparing values would evaluate an expression, which can cook whatever it
    reads, so the text of an expression is what is compared.
    """
    return bool(_quiet(lambda: tuple_.isAtDefault(compare_expressions=True)))


def _animated(tuple_: Any) -> bool:
    return any(_quiet(parm.expression) is not None for parm in _quiet(lambda: list(tuple_)) or ())


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


def _safe_expression(parm: Any, text: str, language: str, depth: int) -> bool:
    """Whether evaluating an expression can be done without cooking anything.

    Only an HScript expression made of numbers, the global variables and the
    functions that read nothing from the scene is safe, plus references to
    other parameters that are themselves safe, a few steps deep.
    """
    if language != "hscript" or depth > MAX_REFERENCE_DEPTH or "`" in text:
        return False
    bare = _LITERAL.sub('""', text)
    calls = _CALL.findall(bare)
    for name in calls:
        if name not in SAFE_FUNCTIONS and name not in REFERENCE_FUNCTIONS:
            return False
    if any(name not in SAFE_VARIABLES for name in _VARIABLE_NAME.findall(text)):
        return False
    references = _REFERENCE.findall(text)
    if len(references) != sum(1 for name in calls if name in REFERENCE_FUNCTIONS):
        return False
    node = _quiet(parm.node)
    for quoted in references:
        target = quoted[1:-1]
        if not target or "$" in target or "`" in target or node is None:
            return False
        referenced = _quiet(lambda target=target: node.parm(target))
        if referenced is None or not _safe_parm(referenced, depth + 1):
            return False
    return True


def _safe_parm(parm: Any, depth: int) -> bool:
    text = _quiet(parm.expression)
    if text is None:
        raw = _quiet(parm.unexpandedString)
        return raw is None or "`" not in str(raw)
    return _safe_expression(parm, str(text), _language(parm), depth)


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
    """Paths close to one that is not there, from the deepest parent that is.

    The parent's own children come first. When they give fewer than `limit`,
    nodes further down whose names are close to the last part of the path
    follow, so a node asked for one level too high is still found.
    """
    parts = [part for part in str(path).split("/") if part]
    leaf = parts[-1] if parts else ""
    while parts:
        parts.pop()
        above = "/" + "/".join(parts)
        found = _quiet(lambda above=above: hou.node(above))
        if found is None:
            continue
        children = _quiet(found.children) or ()
        # Only names that really are close. Three unrelated siblings under a
        # heading of did you mean is worse than saying nothing.
        close = did_you_mean(path, [child.path() for child in children], limit=limit)
        if len(close) < limit and leaf:
            close.extend(_named_like(found, leaf, limit - len(close), set(close)))
        return close
    return []


def _named_like(root: Any, name: str, limit: int, skip: set[str]) -> list[str]:
    """Paths under a node whose last part is close to a name, closest first."""
    below = _quiet(lambda: root.allSubChildren(recurse_in_locked_nodes=False)) or ()
    by_name: dict[str, list[str]] = {}
    for index, node in enumerate(below):
        if index >= NEAR_SCAN:
            break
        by_name.setdefault(str(_ask(node, "name")), []).append(str(node.path()))
    found: list[str] = []
    for close in did_you_mean(name, list(by_name), limit=limit):
        for path in by_name[close]:
            if path not in skip and len(found) < limit:
                found.append(path)
    return found


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
