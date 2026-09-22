"""`hou_scene`: the scene file a session holds, read, opened and saved.

Four actions.

- `info` reads the scene through `scene.info`. `full` adds what the scene
  points at that is not on this machine.
- `open` loads a scene file. The path is checked here before anything is
  sent, so a mistyped path is refused without the session touching its own
  scene. What the load could not resolve comes back as data. The scene is
  replaced, so the trace carries the new scene epoch.
- `save` writes the scene over its own file. A scene with no file yet is
  refused, because saving it means picking a name, which is what
  `save_increment` is for.
- `save_increment` writes the scene to the next `<name>_v###` file. The path
  comes from the `hip` kind of the output conventions: the number from the
  store's transaction, above any version the family already has beside it,
  and the place claimed on disk before anything is written. A file that is
  there is never written over. A scene with no file yet goes under the
  scratch folder, as every output of an unsaved scene does.

None of these can be undone, and every result says so.

A lost reply to `save_increment` is safe to send again with the same
operation id: the version it took is found under that id and the same save
is asked for, which the session answers from its receipt.

This module never imports `hou`.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import outputs as outputs_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge.tools import HIP_SUFFIXES, UNDO_NOTE
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools.base import (
    DETAIL,
    OPERATION_ID,
    SESSION,
    TIMEOUT_S,
    WAIT_S,
    Call,
    ToolSpec,
    inputs,
    outputs,
)

ACTIONS = ("info", "open", "save", "save_increment")

# The scene suffix each license writes. A license that saves only one kind of
# file would rename anything else on the way out, so the path is chosen to
# match. Anything not named here writes `.hip`.
LICENSE_SUFFIX = {
    "apprentice": "hipnc",
    "apprenticehd": "hipnc",
    "education": "hipnc",
    "indie": "hiplc",
}

_VERSION = re.compile(r"[._-]v(\d+)$", re.IGNORECASE)

# The keys of `scene.info` a summary keeps.
SUMMARY_KEYS = (
    "hip_path",
    "hip_name",
    "untitled",
    "unsaved",
    "frame",
    "fps",
    "frame_range",
    "nodes",
)


def scene(call: Call) -> Mapping[str, Any]:
    action = call.arguments.get("action") or "info"
    return ACTION_HANDLERS[action](call)


# Section: info


def scene_info(call: Call) -> dict[str, Any]:
    full = call.arguments.get("detail") == "full"
    reply = call.bridge("scene.info", {"dependencies": True} if full else None)
    data = dict(reply.get("data") or {})
    result = {key: data.get(key) for key in SUMMARY_KEYS}
    result["version"] = version_of(data.get("hip_name"))
    if full:
        for key in ("houdini_version", "undo_entries", "kind", "dependencies"):
            result[key] = data.get(key)
    result["scene_epoch"] = call.trace.get("scene_epoch")
    return result


def version_of(hip_name: Any) -> int | None:
    """The version a scene file name carries, such as 3 for `shot_v003.hip`."""
    if not hip_name:
        return None
    stem = str(hip_name)
    for suffix in HIP_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    found = _VERSION.search(stem)
    return int(found.group(1)) if found else None


# Section: open and save


def open_scene(call: Call) -> dict[str, Any]:
    given = call.arguments.get("path")
    if not given:
        raise CallError(
            "BAD_ARGUMENTS",
            "open needs path: the scene file to load",
            details={"argument": "path"},
        )
    path = Path(os.path.expanduser(str(given)))
    if not path.is_absolute():
        raise CallError(
            "BAD_ARGUMENTS",
            "path has to be absolute, because the session's folder is not this one",
            details={"argument": "path"},
        )
    if not path.name.lower().endswith(HIP_SUFFIXES):
        raise CallError(
            "BAD_ARGUMENTS",
            "path has to name a scene file ending in .hip, .hipnc or .hiplc",
            details={"argument": "path", "suffix": path.suffix},
        )
    if not path.is_file():
        raise CallError(
            "FILE_NOT_FOUND",
            "there is no scene file at the path given",
            details={"argument": "path", "folder_exists": path.parent.is_dir()},
        )
    arguments = {"path": str(path), "discard_unsaved": bool(call.arguments.get("discard_unsaved"))}
    reply = call.bridge("scene.open", arguments, mutating=True)
    data = dict(reply.get("data") or {})
    data["version"] = version_of(data.get("hip_name"))
    data["scene_epoch"] = call.trace.get("scene_epoch")
    return data


def save_scene(call: Call) -> dict[str, Any]:
    reply = call.bridge("scene.save", mutating=True)
    return dict(reply.get("data") or {})


def save_increment(call: Call) -> dict[str, Any]:
    target = call.target()
    operation_id = call.operation_id()
    run_id = f"run-{operation_id}"
    home = call.router.home
    with call.router.store(create=True) as store:
        run = stored(lambda: store.get_run(run_id))
    if run is not None:
        # The same save sent again after a lost reply: the version it took is
        # the one to ask for, and the session answers from its receipt.
        paths = run.paths if isinstance(run.paths, Mapping) else {}
        plan = Replayed(paths, run.version, run_id)
    else:
        info = dict(call.bridge("scene.info").get("data") or {})
        hip = None if info.get("untitled") else info.get("hip_path")
        plan = allocate(call, home, hip, target.session_id, run_id)
    reply = call.bridge("scene.save_as", {"path": plan.path}, mutating=True)
    data = dict(reply.get("data") or {})
    # The file is there now, and it is its own guard against a second writer.
    Path(f"{plan.path}{outputs_module.CLAIM_SUFFIX}").unlink(missing_ok=True)
    return {
        "hip_path": data.get("hip_path") or plan.path,
        "version": plan.version,
        "bytes": data.get("bytes"),
        "template": plan.template,
        "sidecar": plan.sidecar,
        "run_id": run_id,
        "unsaved_hip": plan.unsaved_hip,
        "warnings": list(plan.warnings),
        "undo": UNDO_NOTE,
    }


class Replayed:
    """The plan of a save that was already allocated, read back from its run."""

    def __init__(self, paths: Mapping[str, Any], version: int | None, run_id: str) -> None:
        self.path = str(paths.get("path") or "")
        self.version = version
        self.template = paths.get("template")
        self.sidecar = paths.get("sidecar")
        self.unsaved_hip = bool(paths.get("unsaved_hip"))
        self.warnings = tuple(paths.get("warnings") or ())
        self.run_id = run_id
        if not self.path:
            raise CallError(
                "OUTPUT_REFUSED",
                "the save under this operation id has no path on record",
                details={"run_id": run_id},
            )


def allocate(
    call: Call, home: Path, hip: str | None, session_id: str, run_id: str
) -> outputs_module.OutputPlan:
    """The next versioned scene path, claimed on disk and recorded."""
    suffix = hip_suffix(call, hip)
    scratch = None if os.environ.get("HOUDINI_TEMP_DIR") else Path(home) / "temp"
    try:
        conventions = outputs_module.load_conventions(home=home, hip_path=hip)
        with call.router.store(create=True) as store:
            return outputs_module.allocate(
                store,
                "hip",
                hip_path=hip,
                session_id=session_id,
                run_id=run_id,
                ext=suffix,
                conventions=conventions,
                scratch_root=scratch,
                above=outputs_module.hip_version_floor(hip),
            )
    except outputs_module.OutputError as error:
        raise CallError(
            "OUTPUT_REFUSED", str(error), details={"kind": "hip", "exception": type(error).__name__}
        ) from None
    except (store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None
    except OSError as error:
        raise CallError(
            "OUTPUT_REFUSED",
            "the folder for the new scene file could not be made",
            details={"kind": "hip", "exception": type(error).__name__},
        ) from None


def hip_suffix(call: Call, hip: str | None) -> str:
    """The suffix to save with: the scene's own, or the one its license writes."""
    if hip:
        for suffix in HIP_SUFFIXES:
            if str(hip).lower().endswith(suffix):
                return suffix.lstrip(".")
    reply = call.bridge("bridge.capabilities")
    data = reply.get("data") or {}
    license_name = str(data.get("license") or "").lower().replace(" ", "")
    return LICENSE_SUFFIX.get(license_name, "hip")


def stored(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except (store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None


def unavailable(error: BaseException) -> CallError:
    return CallError(
        "STORE_UNAVAILABLE",
        "the coordination store could not be read",
        details={"exception": type(error).__name__},
    )


ACTION_HANDLERS: dict[str, Callable[[Call], dict[str, Any]]] = {
    "info": scene_info,
    "open": open_scene,
    "save": save_scene,
    "save_increment": save_increment,
}


HOU_SCENE = ToolSpec(
    name="hou_scene",
    description=(
        "The scene file: info, open (absolute path; returns unresolved_types, missing_hdas, "
        "missing_files; refuses a GUI scene with unsaved changes unless discard_unsaved), "
        "save in place, save_increment to the next <name>_v###, never overwriting. No undo."
    ),
    input_schema=inputs(
        {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "session": SESSION,
            "path": {"type": "string"},
            "discard_unsaved": {"type": "boolean"},
            "detail": DETAIL,
            "operation_id": OPERATION_ID,
            "wait_s": WAIT_S,
            "timeout_s": TIMEOUT_S,
        }
    ),
    output_schema=outputs({}),
    handler=scene,
    open_world=False,
)
