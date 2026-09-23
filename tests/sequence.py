"""One scripted pass over every tool, driven by a protocol level client.

The sequence is what a person or an agent would do first with this server, in
order: see the tools, find a session, start a worker, read its scene, build a
few nodes with a reply lost on the way and sent again, read them back, look a
node type and its help page up, ask for a managed output path, save the next
version, run slow code that becomes a job and follow it to the end, compare two
images, and stop the worker.

It takes any connected client that has `list_tools()` and
`call_tool(name, arguments)`, which the SDK's classes all do, so the same pass
runs under each protocol revision and each client class. Every step checks the
shape of its result and records, per step, the tool, how long it took, and
whether the reply carried structured content, a text block and an image, and
in what form. `shape` reduces a result to what must match between two runs:
the keys and the kinds of block, never the values, which name paths, ids and
times that differ every run.

Nothing here imports pytest, so a script can drive the pass too. Every file the
pass writes goes under the worker's `$HOUDINI_TEMP_DIR` or the state folder,
which the caller chooses; the caller also stops the worker should the pass
fail before its last step.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REFERENCES = Path(__file__).resolve().parent / "fixtures" / "brief" / "references"
REFERENCE_IMAGE = REFERENCES / "front_shaded.png"
CANDIDATE_IMAGE = REFERENCES / "three_quarter_shaded.png"

# Code the lost reply step runs. It counts its own runs in the namespace, so
# the pass can tell it ran once. The sleep keeps it running after the first
# send is given up on, so the second send meets work that is still going.
BUILD_CODE = (
    "import time\n"
    "try:\n"
    "    seq_runs += 1\n"
    "except NameError:\n"
    "    seq_runs = 1\n"
    "time.sleep(3)\n"
    "made = [hou.node('/obj').createNode('geo', f'seq_{i}') for i in range(3)]\n"
    "result = {'paths': [node.path() for node in made], 'runs': seq_runs}\n"
)
NODE_NAMES = ("seq_0", "seq_1", "seq_2")
NAMESPACE = "seq"

# What a full compare result counts, whatever the images.
METRIC_KEYS = ("mae", "rmse", "psnr_db", "diff_area_pct")

# Longer than `inline_wait_s` in the pass's config, so `auto` hands back a job.
SLOW_CODE = "import time\ntime.sleep(3)\nresult = {'slept': 3}\n"

# How long the first send of the lost reply step is waited for.
GIVE_UP_S = 0.5

# How many held statuses a job is followed for before the pass gives up.
FOLLOW_ROUNDS = 8


class SequenceFailed(AssertionError):
    """A step's result did not have the shape the pass expects."""


def check(condition: Any, message: str) -> None:
    if not condition:
        raise SequenceFailed(message)


@dataclass
class Step:
    """One call the pass made, and what came back."""

    label: str
    tool: str
    arguments: dict[str, Any]
    elapsed_s: float
    result: Any
    is_error: bool
    structured: bool
    text: str
    image: bool
    blocks: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        """What the pass records for this step."""
        return {
            "step": self.label,
            "tool": self.tool,
            "elapsed_s": round(self.elapsed_s, 3),
            "is_error": self.is_error,
            "structured": self.structured,
            "text": self.text,
            "image": self.image,
        }

    @property
    def body(self) -> dict[str, Any]:
        return dict(self.result.structured_content or {})


@dataclass
class Record:
    """Every step of one pass, in order."""

    steps: list[Step] = field(default_factory=list)
    protocol: str | None = None
    tools: list[str] = field(default_factory=list)
    list_elapsed_s: float = 0.0

    def rows(self) -> list[dict[str, Any]]:
        listing = {
            "step": "tools list",
            "tool": "tools/list",
            "elapsed_s": round(self.list_elapsed_s, 3),
            "is_error": False,
            "structured": False,
            "text": "none",
            "image": False,
        }
        return [listing, *(step.row() for step in self.steps)]

    def shapes(self) -> list[dict[str, Any]]:
        """One shape per step. A job followed over several held statuses is one
        step, its last, since how many holds it took depends on timing."""
        kept = [
            step
            for index, step in enumerate(self.steps)
            if index + 1 == len(self.steps) or self.steps[index + 1].label != step.label
        ]
        return [{"step": step.label, **shape(step)} for step in kept]


def text_form(result: Any) -> str:
    """Whether the text block repeats the structured result, sums it up, or is not there."""
    texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    if not texts:
        return "none"
    if result.structured_content is None:
        return "text only"
    if result.is_error:
        code = (result.structured_content.get("error") or {}).get("code")
        return "error" if code and texts[0].startswith(f"{code}: ") else "summary"
    try:
        mirrored = json.loads(texts[0])
    except ValueError:
        return "summary"
    return "mirror" if mirrored == result.structured_content else "summary"


def key_shape(value: Any, depth: int = 2) -> Any:
    """The keys of a mapping, a few levels down, with none of the values."""
    if isinstance(value, Mapping) and depth > 0:
        return {key: key_shape(item, depth - 1) for key, item in sorted(value.items())}
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


# Where a result carries what the protocol revision itself adds, such as the
# server's own stamp. Kept apart, since that is the revision's to change.
META_KEY = "_meta"


def shape(step: Step) -> dict[str, Any]:
    """What two runs of the pass must agree on for this step, and, under
    `meta`, the protocol's own additions, which may differ by revision."""
    dumped = step.result.model_dump(by_alias=True, exclude_none=True, mode="json")
    meta = dumped.get(META_KEY)
    return {
        "tool": step.tool,
        "is_error": step.is_error,
        "result_keys": sorted(key for key in dumped if key != META_KEY),
        "meta": sorted(meta) if isinstance(meta, Mapping) else [],
        "blocks": step.blocks,
        "text": step.text,
        "structured": key_shape(step.result.structured_content or {}, depth=1),
    }


class Pass:
    """Runs the steps over one connected client and keeps the record."""

    def __init__(self, connected: Any, *, record: Record | None = None) -> None:
        self.connected = connected
        self.record = record or Record()

    async def call(self, label: str, tool: str, arguments: dict[str, Any] | None = None) -> Step:
        arguments = dict(arguments or {})
        started = time.monotonic()
        result = await self.connected.call_tool(tool, arguments)
        elapsed = time.monotonic() - started
        blocks = [str(getattr(block, "type", type(block).__name__)) for block in result.content]
        step = Step(
            label=label,
            tool=tool,
            arguments=arguments,
            elapsed_s=elapsed,
            result=result,
            is_error=bool(result.is_error),
            structured=result.structured_content is not None,
            text=text_form(result),
            image="image" in blocks,
            blocks=blocks,
        )
        check(step.structured, f"{label}: no structured content")
        check(step.text != "none", f"{label}: no text block")
        check("trace" in step.body, f"{label}: no trace")
        self.record.steps.append(step)
        return step

    async def ok(self, label: str, tool: str, arguments: dict[str, Any] | None = None) -> Step:
        step = await self.call(label, tool, arguments)
        first = step.result.content[0].text if step.result.content else ""
        check(not step.is_error, f"{label}: {first[:500]}")
        return step

    async def refused(
        self, label: str, tool: str, arguments: dict[str, Any] | None, code: str
    ) -> Step:
        step = await self.call(label, tool, arguments)
        check(step.is_error, f"{label}: expected {code}, the call succeeded")
        error = step.body.get("error") or {}
        check(error.get("code") == code, f"{label}: expected {code}, got {error.get('code')}")
        check(error.get("message"), f"{label}: the error has no message")
        text = step.result.content[0].text
        check(text.startswith(f"{code}: "), f"{label}: the text does not lead with the code")
        return step

    async def follow(self, label: str, job_id: str) -> Step:
        """Held statuses on a job until it ends."""
        step = await self.ok(label, "hou_jobs", {"job_id": job_id, "wait_s": 10})
        for _ in range(FOLLOW_ROUNDS):
            if step.body.get("state") not in ("queued", "running"):
                break
            step = await self.ok(label, "hou_jobs", {"job_id": job_id, "wait_s": 10})
        check(step.body.get("state") == "done", f"{label}: the job ended {step.body.get('state')}")
        return step


def new_operation_id() -> str:
    return f"op-seq-{uuid.uuid4().hex}"


async def run_sequence(connected: Any, *, record: Record | None = None) -> Record:
    """Run the whole pass over a connected client. Raises `SequenceFailed`."""
    run = Pass(connected, record=record)
    record = run.record

    # The tool list.
    started = time.monotonic()
    listed = await connected.list_tools()
    names = [tool.name for tool in listed.tools]
    check(names and all(name.startswith("hou_") for name in names), f"tool names: {names}")
    for tool in listed.tools:
        check(tool.input_schema.get("type") == "object", f"{tool.name}: no input schema")
        check(tool.output_schema is not None, f"{tool.name}: no output schema")
        check(tool.description, f"{tool.name}: no description")
    record.tools = names
    record.list_elapsed_s = time.monotonic() - started

    # Nothing is live before the worker starts.
    await run.refused("ping, nothing live", "hou_ping", {}, "NO_SESSION")

    listing = await run.ok("sessions list, empty", "hou_sessions", {"action": "list"})
    check(listing.body.get("sessions") == [], f"sessions list: {listing.body.get('sessions')}")

    started_step = await run.ok("sessions start", "hou_sessions", {"action": "start"})
    worker = started_step.body.get("session") or {}
    check(worker.get("kind") == "hython", f"sessions start: kind {worker.get('kind')}")
    check(worker.get("state") == "live", f"sessions start: state {worker.get('state')}")
    alias = worker.get("alias")
    check(alias, "sessions start: no alias")
    on = {"session": alias}

    listing = await run.ok("sessions list", "hou_sessions", {"action": "list"})
    rows = {row.get("session_id"): row for row in listing.body.get("sessions") or []}
    row = rows.get(worker.get("session_id")) or {}
    check(row.get("alias") == alias, f"sessions list: no row for the worker in {list(rows)}")
    check(row.get("kind") == "hython", f"sessions list: kind {row.get('kind')}")
    check(row.get("state") == "live", f"sessions list: state {row.get('state')}")

    ping = await run.ok("ping", "hou_ping", on)
    check(ping.body.get("session_id") == worker.get("session_id"), "ping: another session")
    check(ping.body["call"].get("ok") is True, "ping: the call did not answer")

    info = await run.ok("scene info", "hou_scene", {"action": "info", **on})
    check(isinstance(info.body.get("untitled"), bool), "scene info: no untitled flag")
    check(isinstance(info.body["trace"].get("scene_epoch"), int), "scene info: no epoch")

    # Three nodes, with the first reply lost and the call sent again under
    # the same operation id: one set of nodes, not two.
    operation_id = new_operation_id()
    build = {"code": BUILD_CODE, "operation_id": operation_id, "namespace": NAMESPACE, **on}
    try:
        await asyncio.wait_for(connected.call_tool("hou_python", build), GIVE_UP_S)
        check(False, "python build: the first send answered before it was given up on")
    except TimeoutError:
        pass
    sent_again_at = time.time()
    retried = await run.ok("python build, sent again", "hou_python", {**build, "wait_s": 30})
    check(retried.body["trace"].get("operation_id") == operation_id, "python build: another id")
    # The second send met the first one's work still running: it is handed the
    # job, not a result, and that job started before the second send went.
    check("result" not in retried.body, "python build: the second send ran the code itself")
    check(retried.body.get("state") in ("queued", "running"), "python build: not still running")
    check(retried.body.get("job_id") == f"job-{operation_id}", "python build: another job")
    began = retried.body.get("started_at")
    check(isinstance(began, (int, float)) and began < sent_again_at, "python build: started late")
    await run.follow("python build, followed", retried.body["job_id"])
    retried = await run.ok("python build, answered", "hou_python", build)
    made = retried.body.get("result") or {}
    paths = [f"/obj/{name}" for name in NODE_NAMES]
    check(made.get("paths") == paths, f"python build: made {made}")
    check(made.get("runs") == 1, f"python build: the code ran {made.get('runs')} times")
    counted = await run.ok(
        "python build, runs counted",
        "hou_python",
        {"code": "result = seq_runs", "namespace": NAMESPACE, **on},
    )
    check(counted.body.get("result") == 1, f"python build: ran {counted.body.get('result')} times")

    tree = await run.ok("inspect tree", "hou_inspect", {"mode": "tree", "path": "/obj", **on})
    paths = [row.get("path") for row in tree.body.get("rows") or tree.body.get("nodes") or []]
    ours = sorted(path for path in paths if str(path).startswith("/obj/seq_"))
    check(ours == [f"/obj/{name}" for name in NODE_NAMES], f"inspect tree: {paths}")

    parms = await run.ok(
        "inspect parms",
        "hou_inspect",
        {"mode": "parms", "path": "/obj/seq_0", "parm_filter": "all", **on},
    )
    entries = parms.body.get("nodes") or [{}]
    read = [row.get("n") for row in entries[0].get("parms") or []]
    check("t" in read, f"inspect parms: read {read[:10]}")

    node_type = await run.ok(
        "node type",
        "hou_node_type",
        {"type": "attribwrangle", "context": "sop", "detail": "full", **on},
    )
    parm_names = [row.get("name") for row in node_type.body.get("parms") or []]
    check(parm_names, "node type: no parameters")
    check("snippet" in parm_names, f"node type: no snippet in {parm_names[:10]}")

    page = await run.ok(
        "docs page", "hou_docs", {"mode": "page", "path": "nodes/sop/attribwrangle"}
    )
    title = str(page.body.get("title") or "")
    text = str(page.body.get("text") or "")
    check("wrangle" in title.lower(), f"docs page: title {title!r}")
    check(len(text) > 200 and "VEX" in text, "docs page: no text read")

    resolved = await run.ok(
        "outputs resolve",
        "hou_outputs",
        {"action": "resolve", "kind": "cache", "name": "seq", **on},
    )
    check(resolved.body.get("parm_string"), "outputs resolve: no parm string")
    check(Path(resolved.body.get("expanded_path") or "").is_absolute(), "outputs resolve: path")

    saved = await run.ok("scene save_increment", "hou_scene", {"action": "save_increment", **on})
    check(Path(saved.body.get("hip_path") or "").is_file(), "save_increment: no file")
    check(saved.body.get("version") == 1, f"save_increment: version {saved.body.get('version')}")

    slow = await run.ok(
        "python promoted to a job", "hou_python", {"code": SLOW_CODE, "background": "auto", **on}
    )
    check(slow.body.get("state") in ("queued", "running"), f"python job: {slow.body.get('state')}")
    check("result" not in slow.body, "python job: answered inline, not promoted")
    ended = await run.follow("jobs status", slow.body["job_id"])
    check(ended.body.get("outputs", {}).get("result") == {"slept": 3}, "jobs status: outputs")

    compared = await run.ok(
        "compare",
        "hou_compare",
        {
            "reference": str(REFERENCE_IMAGE),
            "candidate": {"source": "file", "path": str(CANDIDATE_IMAGE)},
            **on,
        },
    )
    check(compared.image, "compare: no image block")
    metrics = compared.body.get("metrics") or {}
    missing = [key for key in METRIC_KEYS if key not in metrics]
    check(not missing, f"compare: metrics lack {missing}")
    check(0.0 < float(metrics["mae"]["overall"]) < 1.0, f"compare: mae {metrics['mae']}")

    stopped = await run.ok("sessions stop", "hou_sessions", {"action": "stop", **on})
    check((stopped.body.get("stopped") or {}).get("ended") is True, "sessions stop: not ended")
    return record
