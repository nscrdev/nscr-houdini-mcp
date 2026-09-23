"""`hou_outputs`: managed output paths and the record of what a scene made.

Three actions.

- `resolve` takes a place for one output: a kind from the output table, a
  name, and an extension when the kind's own is not the one wanted. `node`
  is the node the output belongs to, and the name comes from it when none is
  given, staying `${OS}` in the line a person reads. The answer is the same
  output twice: `parm_string`, the unexpanded line with `$HIP` that belongs
  on the node, and `expanded_path`, the absolute path for this run, frozen
  now. With them come the `version`, the `folder` the output goes to, the
  `run_id` and the `sidecar` record written beside it. A version is taken
  every time, so `resolve` changes the store even when it changes no scene.
- `list` reads what this scene has made, newest first: every run of every
  kind for the scene's family in its folder, or for a scene never saved, the
  runs of this session. Each row has the path, the line, the kind, the
  version, the session, the node, when it was made and whether anything is
  on disk for it. `filter` narrows it by `kind`, a `name` glob and `since`, a
  time as ISO text or seconds since 1970.
- `lint` reads the output parameters under `node` (`/` unless named) and
  reports each one that breaks the conventions, one row per problem:
  `absolute_path`, `outside_hip`, `unversioned`, `missing_on_disk`, or
  `frozen_after_run` for a run's own path that nobody gave back.

`list` and `lint` page like every read: a `limit`, and `next_page` to send
back as `page` with the same arguments. A token from another action, session
or query is `BAD_CURSOR`. A `lint` page after the scene was replaced still
comes back and says `scene_changed`.

Every action first puts back what a session that has gone left frozen in this
scene, which is the sweep the parameter ruling asks of the next session to
open a scene. `hou_scene open` does the same, and both say what they put back
in `restored_parms`.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import outputs as outputs_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client
from nscr_houdini_mcp.bridge.errors import did_you_mean
from nscr_houdini_mcp.bridge.security import InsecureLocation, private_dir
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools.base import SESSION, WAIT_S, Call, ToolSpec, inputs, outputs

ACTIONS = ("resolve", "list", "lint")

# Kinds a caller may ask for. Record kinds are the server's own.
KINDS = tuple(
    kind for kind in outputs_module.OUTPUT_KINDS if kind not in outputs_module.RECORD_KINDS
)
FILTER_KEYS = ("kind", "name", "since")

DEFAULT_LIST_LIMIT = 50
DEFAULT_LINT_LIMIT = 200
MAX_LIMIT = 2000

# How many runs a list reads from the store at a time, and how many it looks
# at in one call before it hands back a page token instead.
BATCH = 200
MAX_SCANNED = 5000

# How many parameters one sweep puts back.
MAX_RESTORES = 50

TOKEN_VERSION = 1
MAX_TOKEN_CHARS = 1800
TOKEN_FIELDS = {"v": int, "a": str, "s": str, "e": int, "q": str, "k": list}

# Where the output conventions come from, named the way a person finds them.
CONVENTION_FILES = (
    f"{outputs_module.PROJECT_FILE_NAMES[0]} beside the scene",
    f"{outputs_module.USER_FILE_NAMES[0]} in the state folder",
)


def outputs_tool(call: Call) -> dict[str, Any]:
    action = call.arguments.get("action") or "list"
    limit = call.arguments.get("limit")
    if limit is not None and not 1 <= limit <= MAX_LIMIT:
        raise CallError(
            "BAD_ARGUMENTS",
            f"limit must be from 1 to {MAX_LIMIT}",
            details={"argument": "limit", "given": limit},
        )
    return ACTION_HANDLERS[action](call)


# Section: resolve


def resolve(call: Call) -> dict[str, Any]:
    arguments = call.arguments
    kind = arguments.get("kind")
    if not kind:
        raise CallError(
            "BAD_ARGUMENTS",
            "resolve needs kind: which line of the output table to use",
            details={"argument": "kind", "kinds": list(KINDS)},
        )
    if kind not in KINDS:
        raise CallError(
            "BAD_ARGUMENTS",
            f"no output kind {kind}",
            details={"argument": "kind", "did_you_mean": did_you_mean(kind, KINDS)},
        )
    node = node_argument(arguments.get("node"))
    target = call.target()
    hip = scene_here(call)
    if node is not None:
        # A node that is not there is refused with the closest paths, rather
        # than naming an output after a typo.
        call.bridge("node.inspect", {"mode": "node", "paths": [node], "batch": False})
    restored = restore_left_over(call, hip)
    home = call.router.home
    spill_root = call.config.spill_folder if call.config is not None else None
    with call.router.store(create=True) as store:
        plan = allocate(
            store,
            kind,
            name=arguments.get("name"),
            ext=arguments.get("ext"),
            node=node,
            hip=hip,
            session_id=target.session_id,
            home=home,
            spill_root=spill_root,
        )
    said: dict[str, Any] = {
        "action": "resolve",
        "kind": plan.kind,
        "name": plan.name,
        "parm_string": plan.template,
        "expanded_path": plan.path,
        "version": plan.version,
        "folder": plan.directory,
        "run_id": plan.run_id,
        "sidecar": plan.sidecar,
    }
    if plan.unsaved_hip:
        said["unsaved_hip"] = True
    if plan.warnings:
        said["warnings"] = list(plan.warnings)
    if restored:
        said["restored_parms"] = restored
    return said


def allocate(
    store: Any,
    kind: str,
    *,
    name: Any,
    ext: Any,
    node: str | None,
    hip: str | None,
    session_id: str,
    home: Any,
    spill_root: Path | None,
) -> outputs_module.OutputPlan:
    """A place for one output, claimed on disk and recorded as a run."""
    scratch = None if os.environ.get("HOUDINI_TEMP_DIR") else Path(home) / "temp"
    try:
        if kind == outputs_module.SPILL_KIND and spill_root is not None:
            # The spill folder is kept private, as the server keeps it for
            # results it writes there itself.
            private_dir(Path(spill_root))
        conventions = outputs_module.load_conventions(home=home, hip_path=hip)
        return outputs_module.allocate(
            store,
            kind,
            name=None if name is None else str(name),
            hip_path=hip,
            node_path=node,
            session_id=session_id,
            ext=None if ext is None else str(ext),
            conventions=conventions,
            scratch_root=scratch,
            spill_root=spill_root,
        )
    except outputs_module.AllocationFailed as error:
        raise CallError("OUTPUT_BUSY", str(error), details={"kind": kind}) from None
    except outputs_module.OutputError as error:
        raise CallError(
            "OUTPUT_REFUSED",
            str(error),
            details={
                "kind": kind,
                "exception": type(error).__name__,
                "conventions": list(CONVENTION_FILES),
            },
        ) from None
    except (store_module.StoreError, sqlite3.Error) as error:
        raise unavailable(error) from None
    except (OSError, InsecureLocation) as error:
        raise CallError(
            "OUTPUT_UNWRITABLE",
            "the folder for the output could not be made",
            details={"kind": kind, "exception": type(error).__name__},
        ) from None


# Section: list


def list_outputs(call: Call) -> dict[str, Any]:
    arguments = call.arguments
    wanted = filter_argument(arguments.get("filter"))
    limit = int(arguments.get("limit") or DEFAULT_LIST_LIMIT)
    target = call.target()
    hip = scene_here(call)
    family = outputs_module.hip_family(hip)
    folder = scene_folder(hip)
    query = fingerprint("list", family, folder, wanted)
    token = read_token(arguments.get("page"), "list", target.session_id, query)
    before = None if token is None else (float(token["k"][0]), int(token["k"][1]))
    restored = restore_left_over(call, hip)

    rows: list[dict[str, Any]] = []
    last: tuple[float, int] | None = None
    more = False
    with call.router.store() as store:
        cursor = before
        scanned = 0
        while store is not None and not more:
            batch = stored(
                lambda cursor=cursor: store.find_runs(
                    hip_family=family,
                    kind=wanted.get("kind"),
                    name_glob=wanted.get("name"),
                    since=wanted.get("since"),
                    session_id=target.session_id if hip is None else None,
                    before=cursor,
                    limit=BATCH,
                )
            )
            for record in batch:
                scanned += 1
                cursor = (record.created_at, int(record.seq or 0))
                if hip is not None and not made_here(record, folder):
                    continue
                if len(rows) >= limit:
                    more = True
                    break
                rows.append(run_row(record))
                last = cursor
            if len(batch) < BATCH:
                break
            if not more and scanned >= MAX_SCANNED:
                # Long enough for one call: the next page carries on from here.
                more, last = True, cursor
    said: dict[str, Any] = {"action": "list", "runs": rows, "scene_family": family}
    if more and last is not None:
        said["next_page"] = make_token(
            action="list",
            session_id=target.session_id,
            epoch=current_epoch(call),
            query=query,
            key=[last[0], last[1]],
        )
    if restored:
        said["restored_parms"] = restored
    return said


def run_row(record: store_module.RunRecord) -> dict[str, Any]:
    """One run as a list shows it."""
    paths = record.paths if isinstance(record.paths, Mapping) else {}
    path = str(paths.get("path") or "")
    row: dict[str, Any] = {
        "run_id": record.run_id,
        "kind": record.kind,
        "name": record.name,
        "version": record.version,
        "path": path,
        "template": paths.get("template"),
        "session": record.session_id,
        "created": datetime.fromtimestamp(record.created_at, tz=UTC).isoformat(timespec="seconds"),
        "on_disk": bool(path) and outputs_module.on_disk(path),
    }
    if record.source_node:
        row["node"] = record.source_node
    if record.job_id:
        row["job_id"] = record.job_id
    return row


def made_here(record: store_module.RunRecord, folder: str | None) -> bool:
    """Whether a run was made by a scene in this folder."""
    scene = record.scene if isinstance(record.scene, Mapping) else {}
    return scene_folder(scene.get("hip_path")) == folder


def scene_folder(hip: Any) -> str | None:
    if not hip:
        return None
    folder, _ = outputs_module.split_hip(str(hip))
    return outputs_module.scene_key(folder)


def filter_argument(value: Any) -> dict[str, Any]:
    """The list filter, checked here rather than in the schema every client pays for."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CallError(
            "BAD_ARGUMENTS",
            "filter is an object with kind, name or since",
            details={"argument": "filter"},
        )
    unknown = [key for key in value if key not in FILTER_KEYS]
    if unknown:
        raise CallError(
            "BAD_ARGUMENTS",
            f"filter takes no key named {unknown[0]}",
            details={
                "argument": f"filter.{unknown[0]}",
                "did_you_mean": did_you_mean(str(unknown[0]), FILTER_KEYS),
            },
        )
    wanted: dict[str, Any] = {}
    kind = value.get("kind")
    if kind is not None:
        if kind not in outputs_module.OUTPUT_KINDS:
            raise CallError(
                "BAD_ARGUMENTS",
                f"no output kind {kind}",
                details={"argument": "filter.kind", "did_you_mean": did_you_mean(str(kind), KINDS)},
            )
        wanted["kind"] = kind
    name = value.get("name")
    if name is not None:
        if not isinstance(name, str) or not name:
            raise CallError(
                "BAD_ARGUMENTS",
                "filter.name is a glob such as beauty*",
                details={"argument": "filter.name"},
            )
        wanted["name"] = name
    since = value.get("since")
    if since is not None:
        wanted["since"] = moment(since)
    return wanted


def moment(value: Any) -> float:
    """Seconds since 1970 from a number, or from ISO text read as local time."""
    if isinstance(value, bool):
        raise bad_since()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            raise bad_since() from None
        return parsed.timestamp()
    raise bad_since()


def bad_since() -> CallError:
    return CallError(
        "BAD_ARGUMENTS",
        "filter.since is a time such as 2026-09-21T14:00 or seconds since 1970",
        details={"argument": "filter.since"},
    )


# Section: lint


def lint(call: Call) -> dict[str, Any]:
    arguments = call.arguments
    scope = node_argument(arguments.get("node")) or "/"
    limit = int(arguments.get("limit") or DEFAULT_LINT_LIMIT)
    target = call.target()
    query = fingerprint("lint", tidy(scope))
    token = read_token(arguments.get("page"), "lint", target.session_id, query)
    hip = scene_here(call)
    restored = restore_left_over(call, hip)
    sent: dict[str, Any] = {"scope": scope, "limit": limit}
    if token is not None:
        sent["after"] = token["k"]
    reply = call.bridge("outputs.lint", sent)
    data = dict(reply.get("data") or {})
    epoch = current_epoch(call)
    said: dict[str, Any] = {"action": "lint", "rows": data.get("rows") or []}
    for key in ("roots", "parms_checked", "truncated", "notes"):
        if key in data:
            said[key] = data[key]
    if data.get("more") and data.get("last"):
        said["next_page"] = make_token(
            action="lint",
            session_id=target.session_id,
            epoch=epoch,
            query=query,
            key=list(data["last"]),
        )
    if token is not None and token["e"] != epoch:
        said["scene_changed"] = True
    if restored:
        said["restored_parms"] = restored
    return said


# Section: the sweep for parameters a gone session left frozen


def restore_left_over(call: Call, hip: str | None) -> list[dict[str, Any]]:
    """Put back what a session that has gone left frozen in this scene.

    A session that dies while a run holds a parameter cannot give the
    template back, and the scene may have been saved with the run's path in
    it. The next session to open the scene does it instead, asked here. A
    record from a scene that was never saved can never be given back, so it
    is dropped. Each parameter is its own change, under an id of its own, and
    one that fails says so and leaves the rest to go on.
    """
    key = outputs_module.scene_key(hip)
    with call.router.store() as store:
        if store is None:
            return []
        stored(store.forget_unrestorable_frozen_parms)
        if key is None:
            return []
        rows = stored(lambda: store.orphan_frozen_parms(hip_key=key))
    done: list[dict[str, Any]] = []
    target = call.target()
    for row in rows[:MAX_RESTORES]:
        arguments = {"node": row.node_path, "parm": row.parm_name, "owner": row.session_id}
        try:
            reply = call.router.call(
                target,
                "outputs.restore_parm",
                arguments,
                operation_id=client.new_operation_id(),
                wait_s=call.arguments.get("wait_s"),
            )
        except CallError as error:
            done.append({**arguments, "restored": False, "reason": error.code})
            continue
        answer = dict(reply.get("data") or {})
        answer["owner"] = row.session_id
        done.append(answer)
    return done


# Section: shared pieces


def scene_here(call: Call) -> str | None:
    """The scene file the session holds, or nothing for one never saved."""
    info = dict(call.bridge("scene.info").get("data") or {})
    if info.get("untitled"):
        return None
    hip = info.get("hip_path")
    return str(hip) if hip else None


def node_argument(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text.startswith("/"):
        raise CallError(
            "BAD_ARGUMENTS",
            f"{text} is not an absolute node path such as /out/karma1",
            details={"argument": "node", "path": text},
        )
    return text


def tidy(path: str) -> str:
    return "/" + "/".join(part for part in path.split("/") if part)


def current_epoch(call: Call) -> int:
    epoch = call.trace.get("scene_epoch")
    return epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else -1


def fingerprint(*parts: Any) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_token(*, action: str, session_id: str, epoch: int, query: str, key: list[Any]) -> str:
    body = {"v": TOKEN_VERSION, "a": action, "s": session_id, "e": epoch, "q": query, "k": key}
    text = json.dumps(body, separators=(",", ":"), ensure_ascii=True)
    return base64.urlsafe_b64encode(text.encode("ascii")).decode("ascii").rstrip("=")


def read_token(page: Any, action: str, session_id: str, query: str) -> dict[str, Any] | None:
    """The token a caller sent back, or `BAD_CURSOR` when it is not one of ours."""
    if page is None:
        return None
    unreadable = bad_token("the page token could not be read")
    if not isinstance(page, str) or len(page) > MAX_TOKEN_CHARS:
        raise unreadable
    try:
        padded = page + "=" * (-len(page) % 4)
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        body = json.loads(raw.decode("ascii"))
    except (binascii.Error, UnicodeError, ValueError, RecursionError):
        raise unreadable from None
    if not isinstance(body, dict) or set(body) != set(TOKEN_FIELDS):
        raise unreadable
    for name, kind in TOKEN_FIELDS.items():
        if isinstance(body[name], bool) or not isinstance(body[name], kind):
            raise unreadable
    if body["v"] != TOKEN_VERSION or not _key_fits(body["a"], body["k"]):
        raise unreadable
    if body["a"] != action:
        raise bad_token(f"that page token belongs to {body['a']}, not {action}")
    if body["s"] != session_id:
        raise bad_token("that page token belongs to another session")
    if body["q"] != query:
        raise bad_token(
            "that page token belongs to other arguments; send the same ones as the first page"
        )
    return body


def _key_fits(action: str, key: list[Any]) -> bool:
    if len(key) != 2:
        return False
    first, second = key
    if action == "list":
        return (
            isinstance(first, (int, float))
            and not isinstance(first, bool)
            and isinstance(second, int)
            and not isinstance(second, bool)
        )
    return isinstance(first, str) and first.startswith("/") and isinstance(second, str)


def bad_token(message: str) -> CallError:
    return CallError("BAD_CURSOR", message, details={"argument": "page"})


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


def summary_line(data: Mapping[str, Any]) -> str:
    """What a client that reads only text is shown of a long result."""
    action = data.get("action")
    if action == "resolve":
        return f"hou_outputs resolve: {data.get('parm_string')} -> {data.get('expanded_path')}"
    count = len(data.get("runs") or data.get("rows") or [])
    line = f"hou_outputs {action}: {count} {'runs' if action == 'list' else 'rows'}"
    if data.get("next_page"):
        line += "; more with next_page"
    return line


ACTION_HANDLERS: dict[str, Callable[[Call], dict[str, Any]]] = {
    "resolve": resolve,
    "list": list_outputs,
    "lint": lint,
}


HOU_OUTPUTS = ToolSpec(
    name="hou_outputs",
    description=(
        "Managed output paths and records: resolve a path for a kind and name, list what "
        "this scene has produced, lint output parameters that point outside the managed tree."
    ),
    input_schema=inputs(
        {
            "action": {"enum": list(ACTIONS)},
            "session": SESSION,
            "kind": {"type": "string"},
            "name": {"type": "string"},
            "ext": {"type": "string"},
            "node": {"type": "string"},
            "filter": {"type": "object"},
            "limit": {"type": "integer"},
            "page": {"type": "string"},
            "wait_s": WAIT_S,
        }
    ),
    output_schema=outputs({}),
    handler=outputs_tool,
    open_world=False,
    summary=summary_line,
)
