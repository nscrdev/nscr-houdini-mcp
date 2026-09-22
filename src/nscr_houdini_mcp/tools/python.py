"""`hou_python`: run Python inside a session, with the whole `hou` API.

The code runs in a namespace the session keeps between calls, so a variable
set in one call is there in the next. Each namespace is one dict, seeded with
`hou` and the `mcp` helper and nothing else, and kept until a call passes
`reset`, the session ends, or nobody has used it for a day. A call that names
no namespace gets this server's own, `c_<id>` with an id drawn when the server
starts, so two agents on two servers never share variables by accident.
`shared` is the one to name on purpose when they should. Separate namespaces
keep variables apart and nothing else: every one works on the same scene.

What the code leaves in `result` comes back as `result`, turned into JSON the
way every answer from a session is: a node or a parameter as its path, a
vector as its numbers, an array as a list up to the usual caps. What it prints
to either stream comes back as `stdout_tail`. `max_chars` is the text budget
for both together. A result over it comes back as the start of its JSON text,
and whatever does not fit is counted in `elided_chars` and written whole to
the spill folder, with the file named in `spill_path`.

An exception in the code is not a failed call. The result carries `error`
with the type, the message and the last lines of the traceback, places on
disk taken out, and is marked as an error for the client. The namespace is
left as it was at the raise, and so is the scene.

Every call counts as a change, whatever the code does: it runs in one undo
group named `undo_label`, or `hou_python` and the operation id, the caller's
or the one the trace names, and takes a receipt under that id. The same id
sent again after a lost reply gets the first answer back and the code does
not run twice. The same id with other code is `OPERATION_MISMATCH`. A call
that outruns `timeout_s` answers `TIMEOUT` with `still_running`; the code
carries on, and the same operation id fetches its answer once it ends.
`timeout_s` is capped by `python_timeout_cap_s` in the config.

The `mcp` helper in every namespace:

- `mcp.output_path(kind, name=None, ext=None)`: a managed path for this
  session and scene, from the output table: render, flipbook, comp, cache,
  usd, hip, capture or compare. Never a path the code makes up.
- `mcp.progress(done, total=None, message=None)`: a note health shows while
  the call runs.
- `mcp.cancelled()`: whether somebody asked this call to stop, for a long
  loop to look at.

This module never imports `hou`.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from typing import Any

from nscr_houdini_mcp import config as config_module
from nscr_houdini_mcp.bridge.dispatch import DEFAULT_TIMEOUT_S
from nscr_houdini_mcp.results import CallError, Spill, compact
from nscr_houdini_mcp.tools.base import (
    OPERATION_ID,
    SESSION,
    TIMEOUT_S,
    WAIT_S,
    Call,
    ToolSpec,
    inputs,
    outputs,
)

# The text budget for the result and the printed output together.
DEFAULT_MAX_CHARS = 12_000
MAX_MAX_CHARS = 200_000

# The namespace agents name on purpose to share variables.
SHARED = "shared"

# This server's own namespace, drawn once per process. A call that names none
# gets it, so variables are private to the agent behind this server.
SERVER_ID = secrets.token_hex(4)
DEFAULT_NAMESPACE = f"c_{SERVER_ID}"

NAMESPACE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# The run budget, left without its wording here to keep the tool list small.
# It means what it means for every other tool.
RUN_FOR_S = {key: value for key, value in TIMEOUT_S.items() if key != "description"}

LABEL_PREFIX = "hou_python"


def run_python(call: Call) -> dict[str, Any]:
    arguments = call.arguments
    code = arguments["code"]
    namespace = arguments.get("namespace") or DEFAULT_NAMESPACE
    if not NAMESPACE_NAME.match(namespace):
        raise CallError(
            "BAD_ARGUMENTS",
            "namespace must be 1 to 64 letters, digits, dot, dash or underscore",
            details={"argument": "namespace", "default": DEFAULT_NAMESPACE, "shared": SHARED},
        )
    max_chars = arguments.get("max_chars")
    if max_chars is not None and not 1 <= max_chars <= MAX_MAX_CHARS:
        raise CallError(
            "BAD_ARGUMENTS",
            f"max_chars must be from 1 to {MAX_MAX_CHARS}",
            details={"argument": "max_chars", "given": max_chars},
        )
    budget = DEFAULT_MAX_CHARS if max_chars is None else int(max_chars)
    # The session is asked to wait no longer than the config allows. The code
    # is never stopped: past this the call answers TIMEOUT and the code goes on.
    cap = (
        call.config.python_timeout_cap_s
        if call.config
        else config_module.DEFAULT_PYTHON_TIMEOUT_CAP_S
    )
    asked = arguments.get("timeout_s")
    call.arguments["timeout_s"] = min(DEFAULT_TIMEOUT_S if asked is None else float(asked), cap)

    sent: dict[str, Any] = {
        "code": code,
        "namespace": namespace,
        "undo_label": undo_label(arguments, call.operation_id()),
    }
    if arguments.get("reset"):
        sent["reset"] = True
    reply = call.bridge("python.run", sent, mutating=True)
    data = dict(reply.get("data") or {})

    result = data.get("result")
    stdout = str(data.get("stdout") or "")
    dropped = int(data.get("stdout_dropped") or 0)
    error = data.get("error")

    shown, tail, elided = fit(result, stdout, budget)
    elided += dropped
    said: dict[str, Any] = {"result": shown, "stdout_tail": tail, "elided_chars": elided}
    if elided:
        said.update(spill(call, result=result, stdout=stdout, dropped=dropped, error=error))
    if error is not None:
        said["error"] = error
    said["duration_ms"] = data.get("duration_ms")
    undo = reply.get("undo") if isinstance(reply.get("undo"), Mapping) else {}
    said["undo_label"] = undo.get("label") or sent["undo_label"]
    said["namespace"] = data.get("namespace") or namespace
    said["scene_epoch"] = call.trace.get("scene_epoch")
    if reply.get("lossy"):
        said["lossy"] = True
        said["cut"] = reply.get("cut")
    return said


def undo_label(arguments: Mapping[str, Any], operation_id: str) -> str:
    """The caller's name for the undo entry, or one made from the operation id.

    The id is the caller's, or the one this call minted and hands back in its
    trace. Either way a retry under the same id sends the same name, which is
    part of what the receipt compares.
    """
    given = str(arguments.get("undo_label") or "").strip()
    return given or f"{LABEL_PREFIX} {operation_id}"


def fit(result: Any, stdout: str, budget: int) -> tuple[Any, str, int]:
    """The result, the end of the output and how much was left out.

    The result comes first. One whose JSON text is over the budget comes back
    as the start of that text, and no output fits beside it. Otherwise the
    output gets what is left, from its end, because that is where a run says
    how it finished.
    """
    text = "" if result is None else compact(result)
    if len(text) > budget:
        return text[:budget], "", len(text) - budget + len(stdout)
    room = budget - len(text)
    tail = stdout[-room:] if room else ""
    return result, tail, len(stdout) - len(tail)


def spill(call: Call, *, result: Any, stdout: str, dropped: int, error: Any) -> dict[str, Any]:
    """Write the whole of what did not fit, and say where it went."""
    if call.config is None:
        return {}
    whole = {"result": result, "stdout": stdout, "stdout_dropped_chars": dropped, "error": error}
    try:
        written = Spill(call.config.spill_folder, call.config.spill_over_bytes).write(
            compact(whole), tool="hou_python"
        )
    except CallError:
        # The code has run and its receipt is kept, so what did fit still
        # comes back rather than a failure in place of it.
        return {"spill_path": None, "spill_failed": True}
    return {"spill_path": written["path"]}


def failed(data: Mapping[str, Any]) -> bool:
    return data.get("error") is not None


def summary_line(data: Mapping[str, Any]) -> str:
    """What a client that reads only text is shown of a long result."""
    error = data.get("error")
    namespace = data.get("namespace")
    if isinstance(error, Mapping):
        lines = [f"hou_python in {namespace}: {error.get('type')}: {error.get('message')}"]
    else:
        lines = [f"hou_python in {namespace}: ran in {data.get('duration_ms')} ms"]
    if data.get("result") is not None:
        lines.append(f"result: {compact(data['result'])}")
    if data.get("stdout_tail"):
        lines.append(f"stdout_tail:\n{data['stdout_tail']}")
    if isinstance(error, Mapping) and error.get("traceback_tail"):
        lines.append(f"traceback_tail:\n{error['traceback_tail']}")
    if data.get("elided_chars"):
        where = data.get("spill_path") or "nowhere, the spill folder could not be written"
        lines.append(f"{data['elided_chars']} characters left out, all of it in {where}")
    return "\n".join(lines)


HOU_PYTHON = ToolSpec(
    name="hou_python",
    description=(
        "Run Python with the full hou API. Set result to return data. Variables persist "
        "in the echoed namespace; pass it back, or shared. Filter and summarize in "
        "Houdini; return only what you need."
    ),
    input_schema=inputs(
        {
            "code": {"type": "string"},
            "session": SESSION,
            "namespace": {"type": "string"},
            "reset": {"type": "boolean"},
            "operation_id": OPERATION_ID,
            "scene_epoch": {"type": "integer"},
            "undo_label": {"type": "string"},
            "timeout_s": RUN_FOR_S,
            "max_chars": {"type": "integer"},
            "wait_s": WAIT_S,
        },
        required=("code",),
    ),
    output_schema=outputs({}),
    handler=run_python,
    open_world=False,
    summary=summary_line,
    failed=failed,
)
