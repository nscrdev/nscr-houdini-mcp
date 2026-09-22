"""The error codes a call can come back with, and the exception tools raise.

One table, in one place, so a caller can be written against it and a tool can
never invent a code. Every code here is stable: the text of a message may be
improved, the code may not change meaning.

Two rules hold for everything that leaves this module:

- No raw exception text. What Houdini raises can hold scene contents and the
  name of whoever is logged in, so the reply carries the exception type and a
  message the bridge wrote. The detail goes to the session log.
- No file paths. A node path such as `/obj/geo1` is what the caller asked
  about and stays; a path on disk is replaced with a marker.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping
from typing import Any

# Section: the code table

# What each code means. Reserved codes are listed so the table is the whole
# agreement between the bridge and its callers from the start, even where the
# work that raises them is not written yet.
CODES: dict[str, str] = {
    "SESSION_BUSY": "the session is running another call",
    "UNKNOWN_SESSION": "this bridge is a different session",
    "TIMEOUT": "the call gave up waiting, and says whether the work goes on",
    "TOOL_FAILED": "the tool ran and raised",
    "UNKNOWN_TOOL": "no tool is registered under that name",
    "BAD_ARGUMENTS": "an argument is missing, unknown or of the wrong shape",
    "NODE_NOT_FOUND": "no node at that path",
    "PARM_NOT_FOUND": "the node has no parameter of that name",
    "SCENE_REPLACED": "the scene changed under the call",
    "SESSION_DEAD": "that session is not there any more",
    "OPERATION_MISMATCH": "the same operation id arrived with different arguments",
    "OUTCOME_UNKNOWN": "the work may have happened, and the bridge cannot say",
    "BODY_REFUSED": "the request body was refused before it was read",
    "CAPTURE_EMPTY": "the capture wrote no usable image",
    # The transport refuses these before a tool is ever chosen.
    "BAD_ENVELOPE": "the request envelope could not be read",
    "UNAUTHORIZED": "the request was not signed for this bridge",
    "FORBIDDEN": "the request came from somewhere this bridge does not answer",
    "METHOD_REFUSED": "that endpoint takes POST",
    "NOT_FOUND": "this bridge has nothing on that path",
    "SERVER_BUSY": "too many connections are open to answer another",
}

# Codes the bridge does not raise yet. They are in the table so the meaning is
# fixed now and a caller can handle them before they arrive.
RESERVED = frozenset(
    {
        "SCENE_REPLACED",
        "SESSION_DEAD",
        "OPERATION_MISMATCH",
        "OUTCOME_UNKNOWN",
        "CAPTURE_EMPTY",
    }
)


class BridgeError(Exception):
    """What a tool raises when it wants a code in the reply.

    Anything else a tool raises becomes `TOOL_FAILED`, or the mapped code for
    the Houdini exceptions below.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        *,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, Any] = dict(details or {})
        self.hint = hint

    def safe(self) -> BridgeError:
        """The same error with anything path shaped taken out."""
        return BridgeError(
            self.code,
            hide_paths(self.message),
            {key: _safe_value(value) for key, value in self.details.items()},
            hint=hide_paths(self.hint) if self.hint else None,
        )


# Section: paths

# Where a file path starts on each system. A node path begins with one of
# Houdini's own roots and is never one of these, so it is left alone.
_DISK_ROOTS = (
    "Users",
    "home",
    "root",
    "Applications",
    "Library",
    "Volumes",
    "private",
    "tmp",
    "var",
    "opt",
    "mnt",
    "srv",
    "etc",
    "usr",
)

# A file path starts a word and is followed by a separator or nothing. That
# keeps `/obj/tmp/thing`, which is a node path and the caller's own subject,
# out of it: the `/tmp` in it starts no word.
_POSIX_PATH = re.compile(
    r"(?:\A|(?<=[\s\"'(\[]))"
    r"/(?:" + "|".join(_DISK_ROOTS) + r")"
    r"(?=/|\Z|[\s\"')\],;:])"
    r"(?:/[^\s\"'<>|]*)*"
)
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|]*")
_UNC_PATH = re.compile(r"\\\\[^\s\"'<>|]+")

PATH_MARKER = "<path>"


def hide_paths(text: str) -> str:
    """Replace anything that names a place on disk with a marker."""
    for pattern in (_WINDOWS_PATH, _UNC_PATH, _POSIX_PATH):
        text = pattern.sub(PATH_MARKER, text)
    return text


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return hide_paths(value)
    if isinstance(value, Mapping):
        return {key: _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    return value


# Section: did you mean

MAX_HINTS = 3


def did_you_mean(wanted: str, known: Iterable[str], *, limit: int = MAX_HINTS) -> list[str]:
    """The closest names to one that was not found, nearest first.

    Close spellings come first, then names that start with what was asked for,
    so a caller that typed half a name gets it back.
    """
    names = [str(name) for name in known]
    close = difflib.get_close_matches(wanted, names, n=limit, cutoff=0.5)
    lowered = wanted.lower()
    for name in names:
        if len(close) >= limit:
            break
        if name not in close and lowered and name.lower().startswith(lowered):
            close.append(name)
    return close[:limit]


# Section: what Houdini raises

# Houdini exception name to code. Read by name because the bridge is imported
# where `hou` does not exist, and because a build may not have all of them.
HOU_EXCEPTIONS: dict[str, str] = {
    "ObjectWasDeleted": "NODE_NOT_FOUND",
    "InvalidInput": "BAD_ARGUMENTS",
    "OperationFailed": "TOOL_FAILED",
    "PermissionError": "TOOL_FAILED",
}

# The message for each mapped code, written here so no exception text is
# copied into a reply.
_MAPPED_MESSAGE = {
    "NODE_NOT_FOUND": "the node was deleted while the call was running",
    "BAD_ARGUMENTS": "Houdini refused the values this call passed",
    "TOOL_FAILED": "Houdini refused the operation",
}

_MAPPED_HINT = {
    "NODE_NOT_FOUND": "read the scene again and call with a path that exists",
    "BAD_ARGUMENTS": "check the argument values against what the node accepts",
    "TOOL_FAILED": "the bridge log for this session has the detail",
}


def map_exception(error: BaseException, *, tool: str | None = None) -> BridgeError:
    """Turn whatever a tool raised into one coded error.

    A `BridgeError` is kept as it is. A Houdini exception becomes its mapped
    code. A `TypeError` from calling the tool is an argument mistake. Anything
    else is `TOOL_FAILED`, with the type named and nothing quoted.
    """
    if isinstance(error, BridgeError):
        return error.safe()
    name = type(error).__name__
    module = type(error).__module__
    code = HOU_EXCEPTIONS.get(name) if module.split(".")[0] == "hou" else None
    if code is None and isinstance(error, TypeError):
        code = "BAD_ARGUMENTS"
    if code is None:
        code = "TOOL_FAILED"
    details: dict[str, Any] = {"exception": name}
    if tool:
        details["tool"] = tool
    message = _MAPPED_MESSAGE.get(code, "the tool raised")
    if code == "TOOL_FAILED":
        message = f"the tool raised {name}"
    return BridgeError(code, message, details, hint=_MAPPED_HINT.get(code))
