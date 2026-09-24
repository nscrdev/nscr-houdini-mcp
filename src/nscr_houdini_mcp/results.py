"""What a tool call hands back to the client: the result shapes, errors and spill.

Every result has two parts that say the same thing. `structuredContent` is the
data. The text block is for a client that reads only text: a small result is
the same JSON, a large one is a line saying where the whole of it went. An
error's text carries the code, the message, the hint and the details, so it is
enough to act on without the structured part.

An error never carries a place on disk: the server replaces them with a
marker the same way the bridge does.

Every result, error or not, carries a `trace`: the session that answered, its
alias and its scene epoch, and the operation id when the call changed the
scene. Before a session is chosen these are empty.

Spill. A result bigger than the configured cap is written to a file in the
spill folder, one dated folder a day, and the call returns the path, the size
and a digest, with the first part of the text in the text block. The folder and
the file are made so only their owner can read them, because a result can hold
scene contents. Files older than the configured number of days are removed when
the server starts.

This module never imports `hou`.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp_types import CallToolResult, TextContent

from nscr_houdini_mcp.bridge.encoding import clean_text
from nscr_houdini_mcp.bridge.errors import CODES as BRIDGE_CODES
from nscr_houdini_mcp.bridge.errors import hide_paths, redact
from nscr_houdini_mcp.bridge.security import InsecureLocation, private_dir, write_private

# Codes only the server raises. The bridge's own table is the rest.
SERVER_CODES: dict[str, str] = {
    "SESSION_UNKNOWN": "no session answers to that id or alias",
    "SESSION_AMBIGUOUS": "several sessions are live and the call named none",
    "SESSION_UNRESPONSIVE": "the session is running but its port does not answer",
    "NO_SESSION": "no Houdini session is live",
    "SESSION_UNREACHABLE": "nothing answered on the session's port",
    "REPLY_NOT_AUTHENTIC": "an answer came back that the session did not sign",
    "BAD_REPLY": "the session answered with something that is not a reply",
    "STORE_UNAVAILABLE": "the coordination store could not be read",
    "CONFIG_INVALID": "the config file could not be used",
    "SPILL_FAILED": "the result was too large to return and could not be written out",
    "RESULT_NOT_JSON": "the tool produced a value that cannot be sent as JSON",
    "POOL_FULL": "the worker pool has no room for another worker",
    "HYTHON_NOT_FOUND": "no hython to start a worker with",
    "WORKER_START_FAILED": "hython started but no bridge came up in it",
    "NOT_A_WORKER": "that session is not a worker the pool started",
    "WORKER_BUSY": "that worker is held by a job or running a call",
    "OUTPUT_REFUSED": "the output conventions do not allow a path for this",
    "OUTPUT_BUSY": "every version tried was taken by another writer first",
    "OUTPUT_UNWRITABLE": "the folder for an output could not be made",
    "BAD_CURSOR": "the page token is not one this read handed out",
    "JOB_UNKNOWN": "no job is kept under that id",
    "HELP_UNAVAILABLE": "no help server and no help folder for this Houdini",
    "DOC_NOT_FOUND": "the help has no page at that path",
    "IMAGE_UNREADABLE": "the file is not an image this can read",
    "IMAGE_TOO_LARGE": "the image has more pixels than this reads",
    "REFERENCE_UNKNOWN": "no reference is registered under that name",
    "JOB_RUNNING": "the job has not ended, so its output is not final",
    "NO_OUTPUT": "no image written by that job or node is on disk",
}

CODES: dict[str, str] = {**BRIDGE_CODES, **SERVER_CODES}

# What to do about each code when the error itself says nothing more useful.
HINTS: dict[str, str] = {
    "SESSION_BUSY": "pass wait_s to queue behind the running call, or call again later",
    "UNKNOWN_SESSION": "list the sessions and address the one you mean",
    "TIMEOUT": (
        "the work may still be running; follow job_id with hou_jobs, or send the same"
        " operation_id again to get its result"
    ),
    "TOOL_FAILED": "the session log has the detail; check the arguments and try again",
    "UNKNOWN_TOOL": "call one of the names in the tool list",
    "BAD_ARGUMENTS": "fix the argument named in the details and call again",
    "NODE_NOT_FOUND": "read the scene again and use a path that exists",
    "TYPE_NOT_FOUND": "use a name from did_you_mean, a context from found_in, or search with query",
    "PARM_NOT_FOUND": "use one of the parameter names in the details",
    "PATH_NOT_A_NODE": "read it with mode parms, or ask for the node named in the details",
    "PATH_NOT_A_PARM": "ask for one of the node's parameters, or read the node itself",
    "BAD_CURSOR": "send next_page back only with the mode and session that gave it",
    "SCENE_REPLACED": "read the new scene, then call again with its scene_epoch",
    "SESSION_DEAD": "address the live session named in the details, or start a new one",
    "OPERATION_MISMATCH": "use a new operation_id for different arguments",
    "PARM_FROZEN": "wait for the run named in the details to end, or use another parameter",
    "OUTCOME_UNKNOWN": "read the scene to see whether the change is there before redoing it",
    "BODY_REFUSED": "send a smaller or less deeply nested request",
    "CAPTURE_EMPTY": "check the camera and the node shown, then capture again",
    "UI_UNAVAILABLE": "use a source this session can show, or a session with a user interface",
    "CAPTURE_FAILED": "check the camera and the node shown, and the error in the details",
    "CLEANUP_FAILED": "look at the scene and the view for the steps named in the details",
    "BAD_ENVELOPE": "the server sent a request the bridge could not read; report it",
    "UNAUTHORIZED": "the session's token changed; call again so the session is read afresh",
    "FLOOD_GUARD": "wait the retry_after_s in the details, then call again",
    "FORBIDDEN": "the request did not come from this machine's loopback; report it",
    "METHOD_REFUSED": "the server used the wrong method; report it",
    "NOT_FOUND": "the bridge is a different version; restart the session",
    "SERVER_BUSY": "too many connections are open to the session; call again shortly",
    "SESSION_UNKNOWN": "pass one of the live sessions in the details",
    "SESSION_AMBIGUOUS": "pass session as one of the candidates, or set default_session in config",
    "SESSION_UNRESPONSIVE": "wait for its next heartbeat, or use another live session",
    "NO_SESSION": "open Houdini with the bridge, or start a worker: bridge worker start",
    "SESSION_UNREACHABLE": "ping the session; if it stays silent, use another one",
    "REPLY_NOT_AUTHENTIC": "the port may belong to another program now; list the sessions again",
    "BAD_REPLY": "ping the session; restart it if this repeats",
    "STORE_UNAVAILABLE": "check that the state folder is on a local disk and readable",
    "CONFIG_INVALID": "fix the key named in the details, then call again",
    "SPILL_FAILED": "free space in the spill folder, or narrow the request",
    "RESULT_NOT_JSON": "the tool is at fault; report it with the tool name",
    "POOL_FULL": "use a worker that is running, stop one you are done with, or raise pool_cap",
    "HYTHON_NOT_FOUND": "name hython or houdini_build in config, then start again",
    "WORKER_START_FAILED": "read the worker log named in the details, then start again",
    "NOT_A_WORKER": "close a Houdini with a user interface yourself; stop only workers here",
    "WORKER_BUSY": "wait for the job or call to end, or pass force true to stop it anyway",
    "OUTPUT_REFUSED": "fix the conventions file or the folder the message names, then call again",
    "OUTPUT_BUSY": "call again; another writer is taking versions in the same folder",
    "OUTPUT_UNWRITABLE": "check that the scene folder is there and writable, then call again",
    "FILE_NOT_FOUND": "check the path and use one that is there",
    "FILE_EXISTS": "ask for the next version rather than writing over this one",
    "UNSAVED_CHANGES": "save the scene first, or pass discard_unsaved true to drop the changes",
    "SCENE_UNTITLED": "use save_increment, which picks a versioned file for the scene",
    "JOB_UNKNOWN": "list the jobs to see the ids that are kept; a job is kept for 7 days",
    "JOB_ID_TAKEN": "use a new operation_id; the job under this one is kept for 7 days",
    "HELP_UNAVAILABLE": (
        "start a session or a worker, or set houdini_build or hython in config to an install"
        " that has its houdini/help folder"
    ),
    "DOC_NOT_FOUND": "use one of the paths in did_you_mean, or find the page with mode search",
    "IMAGE_UNREADABLE": "export the image as PNG, JPEG, TIFF or EXR and pass that file",
    "IMAGE_TOO_LARGE": "pass a smaller copy of the image, or crop it to the part that matters",
    "REFERENCE_UNKNOWN": "pass a registered name from list_references, or a file path",
    "JOB_RUNNING": "wait for the job with hou_jobs and wait_s, then call again",
    "NO_OUTPUT": "make the image first, or pass it as a file",
}

# NO_SESSION says what to do next by what the package files show: a machine
# where the bridge was never installed needs another step than one where
# Houdini is only closed. A state not named here keeps the usual hint.
NO_SESSION_HINTS = {
    "missing": "run: nscr-houdini-mcp bridge install, then open Houdini",
    "stale": "run: nscr-houdini-mcp bridge install again, then restart Houdini",
}
NO_SESSION_NO_AUTOSTART_HINT = (
    "in an open Houdini, run the Python from: nscr-houdini-mcp bridge snippet;"
    " or start a worker: bridge worker start"
)


def no_session_hint(install: Mapping[str, Any]) -> str:
    """The hint for NO_SESSION, given what `install.install_state` found."""
    state = install.get("state")
    if state in NO_SESSION_HINTS:
        return NO_SESSION_HINTS[state]
    if state == "ready":
        ours = [
            entry
            for entry in install.get("checked") or []
            if isinstance(entry, Mapping) and entry.get("found") == "ours"
        ]
        if ours and not any(entry.get("autostart") for entry in ours):
            return NO_SESSION_NO_AUTOSTART_HINT
    return HINTS["NO_SESSION"]


# The largest result whose text block repeats the whole JSON. Larger ones get a
# summary line, so a client that shows both does not pay for the result twice.
MIRROR_CHARS = 2000

# How much of an error's details the text block carries.
DETAILS_CHARS = 1500

# How much of a spilled result comes back as a preview.
PREVIEW_CHARS = 2000

# The most one reply may add up to: the structured result, the text block and
# any other blocks, such as a picture. A block that would go over it is left
# out, and the text says so. A tool that sends a picture sizes it to fit first.
REPLY_BUDGET_BYTES = 1024 * 1024

# More than the text block of a result can be: the mirrored result, or the
# first part of a spilled one, with the line around it.
TEXT_BLOCK_CAP = 8192


class CallError(Exception):
    """A call that did not produce a result, with a code a caller can act on."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
        trace: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint or HINTS.get(code)
        self.details: dict[str, Any] = dict(details or {})
        # Who answered and on which scene, when a session said so.
        self.trace: dict[str, Any] = dict(trace or {})

    def as_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            error["hint"] = self.hint
        if self.details:
            error["details"] = self.details
        return error

    @classmethod
    def from_reply(cls, payload: Mapping[str, Any]) -> CallError:
        """A bridge reply that says the call failed, as a server side error.

        The trace beside the error comes along, and so does the summary of the
        scene a `SCENE_REPLACED` reply carries, because the caller needs it to
        go on.
        """
        error = payload.get("error")
        error = error if isinstance(error, Mapping) else {}
        code = str(error.get("code") or "BAD_REPLY")
        details = error.get("details")
        details = dict(details) if isinstance(details, Mapping) else {}
        if payload.get("scene") is not None:
            details[SCENE_KEY] = payload["scene"]
        return cls(
            code,
            str(error.get("message") or CODES.get(code, "the call failed")),
            hint=error.get("hint") or None,
            details=details,
            trace={key: payload[key] for key in TRACE_KEYS if payload.get(key) is not None},
        )


# Where the scene summary beside a refusal is carried in the details.
SCENE_KEY = "scene"

# What a reply says about who answered it, copied into every result.
TRACE_KEYS = ("session_id", "alias", "scene_epoch", "operation_id", "warnings")


def empty_trace() -> dict[str, Any]:
    return {"session_id": None, "alias": None, "scene_epoch": None}


def compact(value: Any) -> str:
    """JSON with no spaces. Bytes stand as their size, anything else as text."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=_loosely)


def _loosely(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    return str(value)


def scrub(value: Any) -> Any:
    """The same value with every string made safe to write as UTF-8.

    A lone surrogate, which a file name or a stray escape can carry, cannot be
    encoded, and a result that holds one would fail after the work was done.
    It is written as its escape instead. Keys are cleaned the same way.
    """
    if isinstance(value, str):
        return clean_text(value)[0]
    if isinstance(value, Mapping):
        return {
            scrub(key) if isinstance(key, str) else key: scrub(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [scrub(item) for item in value]
    return value


def _strictly(value: Any) -> str:
    """Paths and the like read fine as text. Bytes do not: they are refused."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{type(value).__name__} is not JSON")
    return str(value)


def error_result(error: CallError, trace: Mapping[str, Any] | None = None) -> CallToolResult:
    """An error the client reads as one, with a text block that stands alone.

    The rule the bridge keeps holds here too: no place on disk leaves in an
    error. Exception text the server passes on can hold a config file's or a
    program's full path, so every one is replaced with a marker, and the
    details name files by a name relative to where they belong.
    """
    error = redacted(error)
    body = {"error": error.as_dict(), "trace": dict(trace or empty_trace())}
    return CallToolResult(
        content=[TextContent(type="text", text=error_text(error))],
        structured_content=body,
        is_error=True,
    )


def redacted(error: CallError) -> CallError:
    """The same error with every place on disk in it replaced with a marker.

    The scene summary a session hands back beside a refusal is not part of
    the error. It is what the caller needs to go on after `SCENE_REPLACED`,
    its scene file included, so it is carried as the session gave it.
    """
    details = {key: value for key, value in error.details.items() if key != SCENE_KEY}
    details = redact(details)
    if SCENE_KEY in error.details:
        details[SCENE_KEY] = error.details[SCENE_KEY]
    return CallError(
        error.code,
        hide_paths(error.message),
        hint=hide_paths(error.hint) if error.hint else None,
        details=details,
        trace=error.trace,
    )


def error_text(error: CallError) -> str:
    """One block: code, message, hint and as much of the details as fits."""
    text = f"{error.code}: {error.message}"
    if error.hint:
        text += f"\nhint: {error.hint}"
    if error.details:
        details = compact(error.details)
        if len(details) > DETAILS_CHARS:
            left = len(details) - DETAILS_CHARS
            details = f"{details[:DETAILS_CHARS]} ({left} more characters in structuredContent)"
        text += f"\ndetails: {details}"
    return text


def ok_result(
    data: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    spill: Spill | None = None,
    tool: str = "result",
    summary: str | None = None,
    is_error: bool = False,
    extra: Sequence[Any] = (),
) -> CallToolResult:
    """A result, or the path to it when it is too large to return.

    `is_error` marks a result that is whole and still reports a failure, such
    as code that ran and raised: the client reads it as an error and gets
    everything the call has to say about it. `extra` is content that goes
    after the text block, such as a picture, and goes whether or not the
    result itself was spilled, as long as the text and the blocks together
    stay within `REPLY_BUDGET_BYTES`.
    """
    blocks = list(extra)
    flag = bool(is_error)
    body = {**data, "trace": dict(trace)}
    try:
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False, default=_strictly)
    except (TypeError, ValueError) as error:
        raise CallError(
            "RESULT_NOT_JSON",
            f"{tool} produced a value that cannot be sent as JSON",
            details={"tool": tool, "reason": str(error)},
        ) from None
    if clean_text(text)[1]:
        # Text UTF-8 cannot carry would fail on the way out, after the work
        # was done. It goes as its escapes instead.
        body = scrub(body)
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False, default=_strictly)
    if spill is not None and len(text.encode("utf-8")) > spill.over_bytes:
        spilled = spill.write(text, tool=tool)
        body = {"spilled": spilled, "trace": dict(trace)}
        line = (
            f"{tool}: the result is {spilled['bytes']} bytes, over the cap of"
            f" {spill.over_bytes}, so it was written to {spilled['path']}."
            f" Read that file for all of it. First part:\n{text[:PREVIEW_CHARS]}"
        )
        line, kept = within_budget(line, blocks, structured=body)
        return CallToolResult(
            content=[TextContent(type="text", text=line), *kept],
            structured_content=body,
            is_error=flag,
        )
    if len(text) > MIRROR_CHARS:
        line = summary or f"{tool}: {len(text)} characters, keys {', '.join(sorted(data))}"
        text = f"{line}\ntrace: {compact(dict(trace))}\nThe full result is in structuredContent."
    text, kept = within_budget(text, blocks, structured=body)
    return CallToolResult(
        content=[TextContent(type="text", text=text), *kept],
        structured_content=body,
        is_error=flag,
    )


def block_size(block: Any) -> int:
    """The bytes a content block adds to a reply: its text or its encoded data."""
    for field in ("data", "text"):
        value = getattr(block, field, None)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
    return 0


def structured_size(body: Mapping[str, Any]) -> int:
    """The bytes a structured result adds to a reply."""
    text = json.dumps(body, separators=(",", ":"), ensure_ascii=False, default=str)
    return len(text.encode("utf-8"))


def within_budget(
    text: str, blocks: Sequence[Any], *, structured: Mapping[str, Any] | None = None
) -> tuple[str, list[Any]]:
    """The blocks that fit beside the text and the structured result in one reply.

    The text says what did not fit.
    """
    used = len(text.encode("utf-8"))
    if structured is not None:
        used += structured_size(structured)
    kept: list[Any] = []
    dropped = 0
    for block in blocks:
        size = block_size(block)
        if used + size > REPLY_BUDGET_BYTES:
            dropped += size
            continue
        used += size
        kept.append(block)
    if dropped:
        text += (
            f"\nA {dropped} byte block was left out to keep the reply within"
            f" {REPLY_BUDGET_BYTES} bytes."
        )
    return text, kept


class Spill:
    """Writes results that are too large to return into dated files."""

    def __init__(self, folder: Path, over_bytes: int, *, clock: Any = None) -> None:
        self.folder = Path(folder)
        self.over_bytes = int(over_bytes)
        self._clock = clock or datetime.now

    def write(self, text: str, *, tool: str) -> dict[str, Any]:
        """Write one result and say where it went.

        Raises `CallError` with `SPILL_FAILED` when it cannot be written, so
        the caller hears that rather than getting a truncated answer.
        """
        moment = self._clock()
        day = self.folder / moment.strftime("%Y-%m-%d")
        name = f"{moment.strftime('%H%M%S')}-{_safe_name(tool)}-{secrets.token_hex(3)}.json"
        path = day / name
        data = text.encode("utf-8")
        try:
            private_dir(self.folder)
            write_private(path, text)
        except (OSError, InsecureLocation) as error:
            raise CallError(
                "SPILL_FAILED",
                f"a {len(data)} byte result could not be written to the spill folder",
                details={"exception": type(error).__name__, "reason": hide_paths(str(error))},
            ) from None
        return {
            "path": str(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }


def reap_spill(folder: Path, keep_days: int, *, now: float | None = None) -> int:
    """Remove spilled results older than `keep_days`, and the day folders left
    empty. Returns how many files went. Anything it cannot remove stays."""
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - keep_days * 86400.0
    removed = 0
    for day in sorted(folder.iterdir()):
        if not day.is_dir() or day.is_symlink():
            continue
        for item in day.glob("*.json"):
            try:
                if item.is_file() and item.stat().st_mtime < cutoff:
                    item.unlink()
                    removed += 1
            except OSError:
                continue
        try:
            day.rmdir()
        except OSError:
            pass
    return removed


def _safe_name(tool: str) -> str:
    kept = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in tool)
    return kept.strip("-") or "result"
