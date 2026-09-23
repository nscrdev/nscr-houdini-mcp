"""`hou_python`: run Python inside a session, with the whole `hou` API.

The code runs in a namespace the session keeps between calls, so a variable
set in one call is there in the next. Each namespace is one dict, seeded with
`hou` and the `mcp` helper and nothing else, and kept until a call passes
`reset`, the session ends, or nobody has used it for a while: an hour for a
caller's default, a day for one named on purpose. A session keeps at most 32,
the least recently used going first. A call that names no namespace gets this
server's own, `c_<id>` with an id drawn when the server starts, so two agents
on two servers never share variables by accident. `shared` is the one to name
on purpose when they should. Separate namespaces keep variables apart and
nothing else: every one works on the same scene.

What the code leaves in `result` comes back as `result`, turned into JSON on
the session's own thread the way every answer from a session is: a node or a
parameter as its path, a vector as its numbers, an array as a list up to the
usual caps, a lone surrogate as its escape. What it prints to either stream
comes back as `stdout_tail`. `max_chars` is the text budget for both together.
A result over it comes back as the start of its JSON text, and whatever does
not fit is counted in `elided_chars` and written to the spill folder, with the
file named in `spill_path`. The spill holds what the session kept: the last
512,000 characters of output and the result as it was encoded, which is
bounded too. `lossy` and `cut` say where anything was changed or cut.

An exception in the code is not a failed call. The result carries `error`
with the type, the message and the last lines of the traceback, places on
disk taken out, and is marked as an error for the client. Code that will not
compile is the same, with a syntax error's line and offset. The namespace is
left as it was at the raise, and so is the scene.

Every call counts as a change, whatever the code does: it runs in one undo
group named `undo_label`, or `hou_python` and the operation id, the caller's
or the one the trace names, and takes a receipt under that id. The same id
sent again after a lost reply gets the first answer back and the code does
not run twice, from this server or from one started since. The receipt is
bound to the arguments as the caller sent them: a call that named no
namespace sends this server's default under a key of its own that the
receipt leaves out, so the answer it replays names the namespace the code
really ran in. The same id with other code is `OPERATION_MISMATCH`. A call
that outruns `timeout_s` answers `TIMEOUT` with `still_running`; the code
carries on, and the same operation id fetches its answer once it ends.
`timeout_s` above `python_timeout_cap_s` in the config is lowered to it.

Every call is a job, followed by `hou_jobs` under the `job_id` every answer
carries; the id comes from the operation id, so it is known even when a reply
is lost. `background` decides how long the call itself waits:

- `auto`, the default: up to `inline_wait_s` from the config (ten seconds
  unless it says otherwise, or `timeout_s` when that is shorter). Code that
  finishes in that time answers as usual, with `state`. Slower code answers
  with the job handle at that moment and carries on.
- `true`: the job handle as soon as the session has taken the call.
- `false`: up to `timeout_s`, then `TIMEOUT` with `still_running` and the
  `job_id`.

The `mcp` helper in every namespace:

- `mcp.output_path(kind, name=None, ext=None)`: a managed path for this
  session and scene, from the output table: render, flipbook, comp, cache,
  usd, hip, capture, compare, reference or check. Never a path the code
  makes up, and never a spill, which is the server's own.
- `mcp.freeze_parm(parm, path)`: puts a path this call was handed on an
  output parameter, a `hou.Parm` or its path, for as long as the call runs.
  When the call ends, however it ends, the parameter gets the path's `$HIP`
  template back, and `restored_parms` in the answer says so. One changed by
  the code in between is left as the code left it.
- `mcp.progress(done, total=None, message=None)`: a note health and
  `hou_ping` show while the call runs. Finite numbers only.
- `mcp.cancelled()`: whether the call should stop, for a long loop to look
  at. It turns true when `hou_jobs` cancels the job or the session is going
  down. Code that stops once it has seen a cancel ends `cancelled`, and code
  that stops for its session going down ends `lost`; code that never looks
  runs to the end and ends `done`.

This module never imports `hou`.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
from collections.abc import Mapping
from typing import Any

from nscr_houdini_mcp import config as config_module
from nscr_houdini_mcp import jobs as job_rules
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge.dispatch import DEFAULT_TIMEOUT_S
from nscr_houdini_mcp.results import CallError, Spill, compact, scrub
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

NAMESPACE_MAX = 64
NAMESPACE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")

# The run budget, left without its wording or its ceiling here: it means what
# it means for every other tool, and anything above the config's cap is
# lowered to the cap rather than refused.
RUN_FOR_S = {"type": TIMEOUT_S["type"], "minimum": TIMEOUT_S["minimum"]}

LABEL_PREFIX = "hou_python"

# How long the call waits: until the session has taken it, a short while, or
# the whole run budget.
BACKGROUND = ("auto", True, False)

# Errors after which the code may still be running, so the job is worth
# following by its id.
FOLLOWABLE = frozenset({"TIMEOUT", "SESSION_UNREACHABLE", "OUTCOME_UNKNOWN"})


def run_python(call: Call) -> dict[str, Any]:
    arguments = call.arguments
    code = arguments["code"]
    named = arguments.get("namespace")
    if named is not None and not NAMESPACE_NAME.fullmatch(named):
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
    background = arguments.get("background", "auto")
    # The session is asked to wait no longer than the config allows. The code
    # is never stopped: past this the call answers TIMEOUT and the code goes on.
    cap = (
        call.config.python_timeout_cap_s
        if call.config
        else config_module.DEFAULT_PYTHON_TIMEOUT_CAP_S
    )
    asked = arguments.get("timeout_s")
    run_for = min(DEFAULT_TIMEOUT_S if asked is None else float(asked), cap)
    if background is True:
        # Answered as soon as the session has picked the call up.
        run_for = 0.0
    elif background == "auto":
        inline = call.config.inline_wait_s if call.config else config_module.DEFAULT_INLINE_WAIT_S
        run_for = min(float(inline), run_for)
    call.arguments["timeout_s"] = run_for

    # The arguments as the caller chose them, which is what the receipt is
    # bound to. The default namespace goes under a key the receipt leaves out.
    sent: dict[str, Any] = {"code": code}
    if named is not None:
        sent["namespace"] = named
    else:
        sent["default_namespace"] = DEFAULT_NAMESPACE
    sent["undo_label"] = undo_label(arguments, call.operation_id())
    if arguments.get("reset"):
        sent["reset"] = True
    job_id = job_rules.job_id_for(call.operation_id())
    try:
        reply = call.bridge("python.run", sent, mutating=True)
    except CallError as error:
        still = error.code == "TIMEOUT" and error.details.get("still_running")
        if still and background is not False:
            # The code goes on; the caller follows it by id from here.
            return handle(call, job_id, named)
        if error.code in FOLLOWABLE:
            error.details["job_id"] = job_id
        raise
    if background is True:
        # It finished before the session answered, which the row says too.
        return handle(call, job_id, named, state="done")
    said = shape(call, reply, budget=budget, named=named, label=sent["undo_label"])
    said["job_id"] = job_id
    said["state"] = job_state(call, job_id, said)
    return said


def shape(
    call: Call,
    reply: Mapping[str, Any],
    *,
    budget: int,
    named: str | None,
    label: str | None = None,
) -> dict[str, Any]:
    """One answer from the session, fitted to the caller's budget.

    The same for an answer that has just arrived and one read back from its
    receipt when a job is looked at later.
    """
    data = scrub(dict(reply.get("data") or {}))

    result = data.get("result")
    stdout = str(data.get("stdout") or "")
    dropped = int(data.get("stdout_dropped") or 0)
    error = data.get("error")
    # A result the session already cut down to its JSON text.
    as_text = data.get("result_text_chars") is not None

    shown, tail, elided = fit(result, stdout, budget, as_text=as_text)
    elided += dropped
    if as_text:
        elided += int(data["result_text_chars"]) - len(str(result))
    said: dict[str, Any] = {"result": shown, "stdout_tail": tail, "elided_chars": elided}
    if elided:
        said.update(spill(call, result=result, stdout=stdout, dropped=dropped, error=error))
    if error is not None:
        said["error"] = error
    if data.get("restored_parms"):
        said["restored_parms"] = data["restored_parms"]
    said["duration_ms"] = data.get("duration_ms")
    undo = reply.get("undo") if isinstance(reply.get("undo"), Mapping) else {}
    said["undo_label"] = undo.get("label") or label
    said["namespace"] = data.get("namespace") or named or DEFAULT_NAMESPACE
    said["scene_epoch"] = reply.get("scene_epoch", call.trace.get("scene_epoch"))
    cut = list(data.get("cut") or []) + list(reply.get("cut") or [])
    if data.get("lossy") or reply.get("lossy"):
        said["lossy"] = True
        said["cut"] = cut
    return said


def read_job(call: Call, job_id: str) -> store_module.JobRecord | None:
    """The job's row, or nothing when the store will not say."""
    try:
        with call.router.store() as store:
            return None if store is None else store.get_job(job_id)
    except (CallError, store_module.StoreError, sqlite3.Error):
        return None


def job_state(call: Call, job_id: str, said: Mapping[str, Any]) -> str:
    """How the job ended, as its row says, or as the answer implies."""
    record = read_job(call, job_id)
    if record is not None and record.state in store_module.JOB_FINAL_STATES:
        return record.state
    return "failed" if said.get("error") is not None else "done"


def handle(call: Call, job_id: str, named: str | None, *, state: str = "running") -> dict[str, Any]:
    """What a call that is still running answers: the job to follow."""
    record = read_job(call, job_id)
    scene = record.scene if record is not None and isinstance(record.scene, dict) else {}
    spec = record.spec if record is not None and isinstance(record.spec, dict) else {}
    return {
        "job_id": job_id,
        "state": record.state if record is not None else state,
        "session": call.trace.get("session_id"),
        "kind": "python",
        "started_at": (record.started_at or record.created_at) if record is not None else None,
        "namespace": spec.get("namespace") or named or DEFAULT_NAMESPACE,
        "operation_id": call.trace.get("operation_id"),
        "scene_epoch": scene.get("scene_epoch", call.trace.get("scene_epoch")),
    }


def undo_label(arguments: Mapping[str, Any], operation_id: str) -> str:
    """The caller's name for the undo entry, or one made from the operation id.

    The id is the caller's, or the one this call minted and hands back in its
    trace. Either way a retry under the same id sends the same name, which is
    part of what the receipt compares.
    """
    given = str(arguments.get("undo_label") or "").strip()
    return given or f"{LABEL_PREFIX} {operation_id}"


def fit(result: Any, stdout: str, budget: int, *, as_text: bool = False) -> tuple[Any, str, int]:
    """The result, the end of the output and how much was left out.

    The result comes first. One whose JSON text is over the budget comes back
    as the start of that text, and no output fits beside it. Otherwise the
    output gets what is left, from its end, because that is where a run says
    how it finished. `as_text` is for a result the session already turned
    into the start of its JSON text.
    """
    text = str(result) if as_text else ("" if result is None else compact(result))
    if as_text or len(text) > budget:
        kept = text[:budget]
        room = budget - len(kept)
        tail = stdout[-room:] if room else ""
        return kept, tail, len(text) - len(kept) + len(stdout) - len(tail)
    room = budget - len(text)
    tail = stdout[-room:] if room else ""
    return result, tail, len(stdout) - len(tail)


def spill(call: Call, *, result: Any, stdout: str, dropped: int, error: Any) -> dict[str, Any]:
    """Write what the session kept of what did not fit, and say where it went."""
    if call.config is None:
        return {}
    kept = {"result": result, "stdout": stdout, "stdout_dropped_chars": dropped, "error": error}
    try:
        written = Spill(call.config.spill_folder, call.config.spill_over_bytes).write(
            compact(kept), tool="hou_python"
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
    if "result" not in data and data.get("job_id"):
        return f"hou_python in {namespace}: {data.get('state')} as job {data['job_id']}"
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
        lines.append(
            f"{data['elided_chars']} characters left out; what the session kept is in {where}"
        )
    return "\n".join(lines)


HOU_PYTHON = ToolSpec(
    name="hou_python",
    description=(
        "Run Python with the full hou API. Set result to return data. Variables persist "
        "in the echoed namespace; pass it back, or shared. Filter and summarize in "
        "Houdini; return only what you need. Slow code returns a job_id for hou_jobs."
    ),
    input_schema=inputs(
        {
            "code": {"type": "string"},
            "session": SESSION,
            "namespace": {"type": "string", "minLength": 1, "maxLength": NAMESPACE_MAX},
            "reset": {"type": "boolean"},
            "operation_id": OPERATION_ID,
            "scene_epoch": {"type": "integer"},
            "undo_label": {"type": "string"},
            "timeout_s": RUN_FOR_S,
            "max_chars": {"type": "integer"},
            "wait_s": WAIT_S,
            "background": {"enum": list(BACKGROUND)},
        },
        required=("code",),
    ),
    output_schema=outputs({}),
    handler=run_python,
    open_world=False,
    summary=summary_line,
    failed=failed,
)
