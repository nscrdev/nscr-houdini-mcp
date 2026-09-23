"""`hou_scene`: the scene file a session holds, read, opened and saved.

Four actions.

- `info` reads the scene through `scene.info`. `full` adds what the scene
  points at that is not on this machine, and the names of the reference
  images registered for it with `hou_compare`. `unsaved` comes from Houdini in a
  session with a user interface and from the bridge's own mark in a worker,
  which cannot say for itself, and `unsaved_source` says which; it is nothing
  when the mark cannot tell.
- `open` loads a scene file. The path is checked here before anything is
  sent, so a mistyped path is refused without the session touching its own
  scene. What the load could not resolve comes back as data. The scene is
  replaced, so the trace carries the new scene epoch. An output parameter a
  session that has gone left frozen in this scene gets its own value back,
  and `restored_parms` says which; `save` and `save_increment` do the same
  before they write, so a new file never carries that session's path.
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

`save_increment` claims its operation id in the store before it takes a
version, so the same id sent twice, from one server or two, takes one version
at most. A lost reply is safe to send again with the same id: the version it
took is on the receipt, and the same save is asked for, which the session
answers from its own receipt. A save that definitely failed takes its version
back, with the claim and the record it left beside the scene.

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
from nscr_houdini_mcp import references
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge.tools import HIP_SUFFIXES, LICENSE_SUFFIX, UNDO_NOTE
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools.base import (
    DETAIL,
    OPERATION_ID,
    OPERATION_ID_SEPARATOR,
    SESSION,
    TIMEOUT_S,
    WAIT_S,
    Call,
    ToolSpec,
    inputs,
    outputs,
)
from nscr_houdini_mcp.tools.outputs import restore_left_over

ACTIONS = ("info", "open", "save", "save_increment")

# Where the output conventions come from, named the way a person finds them.
CONVENTION_FILES = (
    f"{outputs_module.PROJECT_FILE_NAMES[0]} beside the scene",
    f"{outputs_module.USER_FILE_NAMES[0]} in the state folder",
)

_VERSION = re.compile(r"[._-]v(\d+)$", re.IGNORECASE)

# The keys of `scene.info` a summary keeps.
SUMMARY_KEYS = (
    "hip_path",
    "hip_name",
    "untitled",
    "unsaved",
    "unsaved_source",
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
        result["references"] = reference_names(call, data)
    result["scene_epoch"] = call.trace.get("scene_epoch")
    return result


def reference_names(call: Call, data: Mapping[str, Any]) -> list[str]:
    """The reference images registered for this scene, so a fresh context finds the goal."""
    hip = None if data.get("untitled") else data.get("hip_path")
    home = call.router.home
    scratch = None if os.environ.get("HOUDINI_TEMP_DIR") else Path(home) / "temp"
    try:
        place = references.folder(
            home=home, hip_path=hip, session_id=call.trace.get("session_id"), scratch_root=scratch
        )
    except outputs_module.OutputError:
        return []
    return references.names(place)


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
    # A session that died while a run held one of this scene's output
    # parameters left it frozen; the session that opens the scene gives it
    # its template back.
    restored = restore_left_over(call, data.get("hip_path"))
    if restored:
        data["restored_parms"] = restored
    return data


def save_scene(call: Call) -> dict[str, Any]:
    # What a session that has gone left frozen goes back before the file is
    # written, so the save does not carry its path on.
    restored = restore_left_over(call)
    reply = call.bridge("scene.save", mutating=True)
    data = dict(reply.get("data") or {})
    if restored:
        data["restored_parms"] = restored
    return data


def save_increment(call: Call) -> dict[str, Any]:
    """Save the scene as its next version, once per operation id.

    The id is claimed in the store before any version is taken, so two
    servers sent the same id cannot both take one. The version the claim took
    is written on the receipt before the save is asked for, so a save whose
    reply was lost is asked for again under the same id and the same path,
    which the session answers from its own receipt.
    """
    target = call.target()
    operation_id = call.operation_id()
    key = f"{operation_id}{OPERATION_ID_SEPARATOR}increment"
    digest = store_module.digest_arguments(
        {"action": "save_increment", "session_id": target.session_id}
    )
    restored: list[dict[str, Any]] = []
    with call.router.store(create=True) as store:
        pending = claim_increment(store, key, digest)
        if pending is not None and "result" in pending:
            return {**pending["result"], "replayed": True}
        if pending is None:
            try:
                info = dict(call.bridge("scene.info").get("data") or {})
                hip = None if info.get("untitled") else info.get("hip_path")
                restored = restore_left_over(call, hip)
                plan = planned(allocate(call, store, hip, target.session_id, operation_id))
            except BaseException:
                stored(lambda: store.drop_operation(key))
                raise
            stored(lambda: store.finish_operation(key, state="running", outcome={"plan": plan}))
        else:
            plan = pending["plan"]
        try:
            reply = call.bridge("scene.save_as", {"path": plan["path"]}, mutating=True)
        except CallError as error:
            if error.code in INDEFINITE:
                # The save may have happened. The plan stays on the receipt,
                # so the same id asks for the same file again.
                keep(store, key, plan, error)
            else:
                undo_plan(store, plan)
                stored(lambda: store.drop_operation(key))
            raise
        data = dict(reply.get("data") or {})
        # The file is there now, and it is its own guard against a second writer.
        Path(f"{plan['path']}{outputs_module.CLAIM_SUFFIX}").unlink(missing_ok=True)
        result = {
            "hip_path": data.get("hip_path") or plan["path"],
            "version": plan["version"],
            "bytes": data.get("bytes"),
            "template": plan["template"],
            "sidecar": plan["sidecar"],
            "run_id": plan["run_id"],
            "unsaved_hip": plan["unsaved_hip"],
            "warnings": list(plan["warnings"]) + list(data.get("warnings") or []),
            "undo": UNDO_NOTE,
        }
        if restored:
            result["restored_parms"] = restored
        stored(lambda: store.finish_operation(key, outcome={"result": result}))
    return result


# Codes after which a save may or may not have happened.
INDEFINITE = frozenset(
    {"OUTCOME_UNKNOWN", "SESSION_UNREACHABLE", "TIMEOUT", "REPLY_NOT_AUTHENTIC", "BAD_REPLY"}
)


def claim_increment(store: Any, key: str, digest: str) -> dict[str, Any] | None:
    """Take the id, or say what an earlier attempt under it got to.

    Nothing when this call is the first. The finished result when it already
    ran, and the plan when a version was taken and the save may not have
    happened. `OUTCOME_UNKNOWN` when another call is on it, or when an attempt
    stopped before it recorded which version it took.
    """
    try:
        claim = store.begin_operation(key, digest)
    except store_module.OperationMismatch as error:
        raise CallError("OPERATION_MISMATCH", str(error)) from None
    except (store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None
    outcome = claim.record.outcome if isinstance(claim.record.outcome, Mapping) else {}
    if claim.outcome_unknown and not claim.claimed:
        raise unknown("another call with this operation id is still saving")
    if "result" in outcome or "plan" in outcome:
        return dict(outcome)
    if claim.claimed and not claim.outcome_unknown:
        return None
    if claim.claimed:
        abandon(store, key)
    raise unknown("an earlier save with this operation id stopped before it said which version")


def planned(plan: outputs_module.OutputPlan) -> dict[str, Any]:
    """What a receipt keeps of a version that was taken.

    The path is made once, here, in the form this system writes its own paths
    in, and that one string is what the session is sent, what the receipt
    keeps and what a retry sends again. The session keys its own receipt on
    the exact text of the call, so a retry that spelled the same file another
    way would be a different call.
    """
    return {
        "path": native(plan.path),
        "version": plan.version,
        "name": plan.name,
        "hip_family": plan.hip_family,
        "template": plan.template,
        "sidecar": native(plan.sidecar),
        "run_id": plan.run_id,
        "unsaved_hip": plan.unsaved_hip,
        "warnings": list(plan.warnings),
    }


def native(path: str) -> str:
    """A path in the one form this system writes paths in."""
    return os.fspath(Path(path))


def keep(store: Any, key: str, plan: Mapping[str, Any], error: CallError) -> None:
    try:
        store.finish_operation(
            key, state="failed", outcome={"plan": dict(plan)}, error={"code": error.code}
        )
    except (store_module.StoreError, sqlite3.Error):
        pass


def undo_plan(store: Any, plan: Mapping[str, Any]) -> None:
    """Take back a version whose save definitely did not happen.

    Its claim, its record beside the scene and its run record describe a file
    that is not there. The number keeps its place in the sequence, with no run
    on it. A file that is there after all is left alone, and so is its record.
    """
    path = Path(str(plan["path"]))
    if path.exists():
        return
    Path(f"{path}{outputs_module.CLAIM_SUFFIX}").unlink(missing_ok=True)
    if plan.get("sidecar"):
        Path(str(plan["sidecar"])).unlink(missing_ok=True)
    try:
        store.drop_run(plan["run_id"])
        store.disown_version(
            kind="hip",
            name=plan["name"],
            hip_family=plan["hip_family"],
            version=plan["version"],
        )
    except (store_module.StoreError, sqlite3.Error):
        pass


def abandon(store: Any, key: str) -> None:
    try:
        store.finish_operation(
            key,
            state=store_module.OPERATION_ABANDONED,
            error={"reason": "the attempt stopped without recording what it did"},
        )
    except (store_module.StoreError, sqlite3.Error):
        pass


def unknown(why: str) -> CallError:
    return CallError(
        "OUTCOME_UNKNOWN",
        why,
        hint="read the scene info to see which file the session holds, then use a new operation_id",
    )


def allocate(
    call: Call, store: Any, hip: str | None, session_id: str, operation_id: str
) -> outputs_module.OutputPlan:
    """The next versioned scene path, claimed on disk and recorded."""
    home = call.router.home
    suffix = hip_suffix(call, hip)
    scratch = None if os.environ.get("HOUDINI_TEMP_DIR") else Path(home) / "temp"
    try:
        conventions = outputs_module.load_conventions(home=home, hip_path=hip)
        return outputs_module.allocate(
            store,
            "hip",
            hip_path=hip,
            session_id=session_id,
            run_id=f"run-{operation_id}",
            ext=suffix,
            conventions=conventions,
            scratch_root=scratch,
            above=outputs_module.hip_version_floor(hip),
        )
    except outputs_module.AllocationFailed as error:
        raise CallError("OUTPUT_BUSY", str(error), details={"kind": "hip"}) from None
    except outputs_module.OutputError as error:
        raise CallError(
            "OUTPUT_REFUSED",
            str(error),
            details={
                "kind": "hip",
                "exception": type(error).__name__,
                "conventions": list(CONVENTION_FILES),
            },
        ) from None
    except (store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None
    except OSError as error:
        raise CallError(
            "OUTPUT_UNWRITABLE",
            "the folder for the new scene file could not be made",
            details={"kind": "hip", "exception": type(error).__name__},
        ) from None


def hip_suffix(call: Call, hip: str | None) -> str:
    """The suffix to save with, which is the one Houdini will really write.

    A license that writes one kind of scene file renames anything else, so its
    suffix wins. Otherwise the scene keeps its own, and a scene with no file
    gets `.hip`.
    """
    reply = call.bridge("bridge.capabilities")
    data = reply.get("data") or {}
    name = str(data.get("license") or "").lower().replace(" ", "")
    if name in LICENSE_SUFFIX:
        return LICENSE_SUFFIX[name].lstrip(".")
    if hip:
        for suffix in HIP_SUFFIXES:
            if str(hip).lower().endswith(suffix):
                return suffix.lstrip(".")
    return "hip"


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
