"""What every tool is made of, and what a tool body is handed when it runs.

A tool is a `ToolSpec`: a name, a description, an input schema written as a
plain dict, an output schema where it has one, and a handler. The handler gets
a `Call`, which is the only way a tool reaches Houdini. The `Call` settles the
session once, carries the trace every result echoes, mints the operation id a
change needs, and passes the caller's `scene_epoch`, `wait_s` and `timeout_s`
through to the bridge unchanged.

Arguments are checked against the schema before the handler runs, so a
misspelled name comes back as `BAD_ARGUMENTS` with the nearest real one and
the handler never sees it.

This module never imports `hou`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from mcp_types import Tool as MCPTool
from mcp_types import ToolAnnotations

from nscr_houdini_mcp.bridge import client
from nscr_houdini_mcp.bridge.errors import did_you_mean
from nscr_houdini_mcp.results import CallError, empty_trace
from nscr_houdini_mcp.router import Router, Target

# Section: schema pieces every tool shares

SESSION = {
    "type": "string",
    "description": "Session id or alias. May be left out when one session is live.",
    "x-mcp-header": "Session",
}

WAIT_S = {
    "type": "number",
    "minimum": 0,
    "maximum": 50,
    "description": "Seconds to queue for a busy session before SESSION_BUSY. Default 1.",
}

TIMEOUT_S = {
    "type": "number",
    "minimum": 0,
    "maximum": 3600,
    "description": "Seconds to wait for running work before TIMEOUT.",
}

SCENE_EPOCH = {
    "type": "integer",
    "minimum": 0,
    "description": "The scene_epoch from an earlier trace. The call is refused if the scene "
    "was replaced since.",
}

OPERATION_ID = {
    "type": "string",
    "maxLength": 128,
    "description": "Send the same id again after a lost reply to get the outcome, not a repeat.",
}

# Described once in the server instructions rather than in every tool.
TRACE = {"type": "object"}
SPILLED = {"type": "object"}


def inputs(properties: Mapping[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    """An input schema that refuses names it does not list."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def outputs(properties: Mapping[str, Any]) -> dict[str, Any]:
    """An output schema with the trace and the spill marker every result may carry.

    No tool property is required: a spilled result carries only the marker and
    the trace, and must still match.
    """
    return {
        "type": "object",
        "properties": {**properties, "trace": TRACE, "spilled": SPILLED},
        "required": ["trace"],
    }


# Section: the tool


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the client sees it, and the function that runs it."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Callable[[Call], Mapping[str, Any]]
    output_schema: Mapping[str, Any] | None = None
    # Only for a tool that never changes anything, in any mode.
    read_only: bool = False
    title: str | None = None
    # One line for a result too long to repeat in the text block.
    summary: Callable[[Mapping[str, Any]], str] | None = None
    _validator: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        Draft202012Validator.check_schema(dict(self.input_schema))
        object.__setattr__(self, "_validator", Draft202012Validator(dict(self.input_schema)))

    def as_tool(self) -> MCPTool:
        return MCPTool(
            name=self.name,
            title=self.title,
            description=self.description,
            input_schema=dict(self.input_schema),
            output_schema=dict(self.output_schema) if self.output_schema else None,
            annotations=ToolAnnotations(read_only_hint=True) if self.read_only else None,
        )

    def check(self, arguments: Mapping[str, Any]) -> CallError | None:
        """`BAD_ARGUMENTS` for arguments the schema refuses, or nothing."""
        known = sorted(self.input_schema.get("properties", {}))
        unknown = [name for name in arguments if name not in known]
        if unknown and self.input_schema.get("additionalProperties") is False:
            name = unknown[0]
            return CallError(
                "BAD_ARGUMENTS",
                f"{self.name} takes no argument named {name}",
                details={
                    "argument": name,
                    "did_you_mean": did_you_mean(name, known),
                    "arguments": known,
                },
            )
        error = best_match(self._validator.iter_errors(dict(arguments)))
        if error is None:
            return None
        where = ".".join(str(part) for part in error.absolute_path) or "arguments"
        return CallError(
            "BAD_ARGUMENTS",
            f"{where}: {error.message[:300]}",
            details={"argument": where, "rule": error.validator, "arguments": known},
        )


# Section: one call


class Call:
    """One tool call on its way to a session."""

    def __init__(
        self,
        spec: ToolSpec,
        arguments: Mapping[str, Any],
        router: Router,
        *,
        transport: str = "stdio",
    ) -> None:
        self.spec = spec
        self.arguments = dict(arguments)
        self.router = router
        self.transport = transport
        self.trace: dict[str, Any] = empty_trace()
        self._target: Target | None = None
        self._epoch: int | None = self.arguments.get("scene_epoch")
        self._operation_id: str | None = self.arguments.get("operation_id")
        self._changes = 0

    def target(self) -> Target:
        """The session this call goes to, settled once."""
        if self._target is None:
            self._target = self.router.resolve(self.arguments.get("session"))
            self.trace = self._target.trace()
        return self._target

    def health(self) -> dict[str, Any]:
        """What the session says about itself. Answers while it is busy."""
        data = self.router.health(self.target())
        self._note(data)
        return data

    def bridge(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        mutating: bool = False,
    ) -> dict[str, Any]:
        """Send one bridge call and hand back the whole reply.

        A change carries an operation id: the caller's, or one minted here. A
        tool that makes more than one change derives the later ids from the
        first, so sending the same id again replays every step from its
        receipt rather than doing any of them twice.
        """
        target = self.target()
        operation_id = self._next_operation_id() if mutating else None
        try:
            reply = self.router.call(
                target,
                tool,
                arguments,
                operation_id=operation_id,
                scene_epoch=self._epoch,
                wait_s=self.arguments.get("wait_s"),
                timeout_s=self.arguments.get("timeout_s"),
            )
        except CallError as error:
            self._note(error.trace)
            if operation_id:
                self.trace["operation_id"] = operation_id
            raise
        self._note(reply)
        if operation_id:
            self.trace["operation_id"] = reply.get("operation_id") or operation_id
        return reply

    def _next_operation_id(self) -> str:
        if self._operation_id is None:
            self._operation_id = client.new_operation_id()
        self._changes += 1
        if self._changes == 1:
            return self._operation_id
        return f"{self._operation_id}.{self._changes}"

    def _note(self, said: Mapping[str, Any]) -> None:
        """Take what a reply says about who answered and which scene it was."""
        for key in ("session_id", "alias", "scene_epoch"):
            if said.get(key) is not None:
                self.trace[key] = said[key]
        warnings = said.get("warnings")
        if warnings:
            self.trace["warnings"] = warnings
        # A caller that guards on the epoch is guarding on the scene, so a
        # later step of the same call follows a scene this call replaced.
        if self._epoch is not None and isinstance(said.get("scene_epoch"), int):
            self._epoch = said["scene_epoch"]
