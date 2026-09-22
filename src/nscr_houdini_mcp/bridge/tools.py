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
- `bridge.selfcheck` mutates on purpose, and can be asked to take its time, to
  fail part way, or to throw the scene away, so the queue, the timeout, the
  rollback, the scene epoch and the receipts can be tried against a real
  Houdini rather than only against a stand in. It is registered in a worker
  this project started to be driven and nowhere else.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge.errors import BridgeError, did_you_mean

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
    if os.path.isfile(private) and not os.path.islink(private):
        # Left by an attempt that stopped between the write and the publish.
        os.remove(private)
    before = None if _ask(hou.hipFile, "isNewFile") else _ask(hou.hipFile, "path")
    hou.hipFile.save(private)
    try:
        warnings = _publish(private, path)
    except BaseException:
        _discard(private)
        if before:
            # The session goes back to the file it held. A scene that had no
            # file keeps the private name rather than a made up one.
            _ask(hou.hipFile, "setName", before)
        raise
    hou.hipFile.setName(path)
    return {
        "hip_path": _ask(hou.hipFile, "path") or path,
        "bytes": _size(path),
        "warnings": warnings,
        "undo": UNDO_NOTE,
    }


def _publish(private: str, path: str) -> list[str]:
    """Give a written file its name, only if nothing has that name yet."""
    try:
        os.link(private, path)
    except FileExistsError:
        raise _file_exists(path) from None
    except OSError:
        # This file system has no hard links. The look and the write are two
        # steps then, which is what it was before there was a better way.
        if os.path.lexists(path):
            raise _file_exists(path) from None
        os.replace(private, path)
        return ["this folder cannot link files, so the check and the write were two steps"]
    _discard(private)
    return []


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


def _near(hou: Any, path: str) -> list[str]:
    """Paths close to one that is not there, from the deepest parent that is."""
    parts = [part for part in str(path).split("/") if part]
    while parts:
        parts.pop()
        above = "/" + "/".join(parts)
        found = _quiet(lambda above=above: hou.node(above))
        if found is None:
            continue
        children = _quiet(found.children) or ()
        # Only names that really are close. Three unrelated siblings under a
        # heading of did you mean is worse than saying nothing.
        return did_you_mean(path, [child.path() for child in children])
    return []


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
