"""Jobs: work that is followed by an id rather than waited on.

The rules both ends keep the same way. A job is the operation that runs it:
a session writes the row when it accepts the work, keeps it up to date while
the work runs and writes how it ended, and a server reads the row to answer
for it, from any process and after any restart, because the row lives in the
coordination store.

- A job's id comes from the operation id it runs under, so whoever holds the
  one holds the other, and a reply that never arrived can still be followed.
- States: `queued` once the session has taken the work and before it runs,
  `running`, and one of `done`, `failed`, `cancelled` or `lost` at the end.
  `lost` means the session running it is gone, or has said nothing for
  longer than `SILENCE_S`; the progress and outputs it wrote are kept.
- A finished job leaves a readable copy of its row beside the scene, under
  the `job` kind of the output table: `$HIP/.agent/jobs/<job_id>.json`, or the
  scratch folder for a scene that was never saved.
- Rows are kept for `KEEP_S`, a week, then pruned.

This module never imports `hou`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import outputs
from nscr_houdini_mcp import store as store_module

# How long a job row is kept after it last changed.
KEEP_S = 7 * 24 * 60 * 60.0

# How long a job may go without a word from its session before it counts as
# lost. A session writes one every few seconds while the work runs; this is
# long enough that a cook holding the interpreter for minutes is not taken
# for a dead one.
SILENCE_S = 15 * 60.0

ID_PREFIX = "job-"

# The output kind a finished job's readable copy is written under.
EXPORT_KIND = "job"


def job_id_for(operation_id: str) -> str:
    """The id of the job an operation runs as."""
    return f"{ID_PREFIX}{operation_id}"


def scratch_root(home: Path | str | None) -> Path | None:
    """The scratch folder a scene with no file writes to, as every output picks it."""
    if os.environ.get("HOUDINI_TEMP_DIR") or home is None:
        return None
    return Path(home) / "temp"


def hip_of(record: store_module.JobRecord) -> str | None:
    """The scene file a job ran against, when it had one."""
    scene = record.scene if isinstance(record.scene, dict) else {}
    hip = scene.get("hip_path")
    return str(hip) if hip and not scene.get("untitled") else None


def export_plan(record: store_module.JobRecord, *, home: Path | str | None) -> outputs.OutputPlan:
    """Where a job's readable copy goes, with its folder made."""
    hip = hip_of(record)
    return outputs.record_path(
        EXPORT_KIND,
        record.job_id,
        hip_path=hip,
        session_id=record.session_id,
        conventions=outputs.load_conventions(home=home, hip_path=hip),
        scratch_root=scratch_root(home),
    )


def export(store: store_module.Store, job_id: str, *, home: Path | str | None) -> str | None:
    """Write a job's readable copy beside its scene. Returns the path.

    Nothing when there is no such job. A failure to work out the place or to
    write raises, and the caller decides whether that matters.
    """
    record = store.get_job(job_id)
    if record is None:
        return None
    plan = export_plan(record, home=home)
    store_module.write_export(store.job_export(job_id), plan.path)
    return plan.path


def export_path(record: store_module.JobRecord, *, home: Path | str | None) -> str | None:
    """Where a job's readable copy is, when it has been written."""
    hip = hip_of(record)
    try:
        plan = outputs.plan_path(
            EXPORT_KIND,
            run_id=record.job_id,
            name=record.job_id,
            hip_path=hip,
            session_id=record.session_id,
            conventions=outputs.load_conventions(home=home, hip_path=hip),
            scratch_root=scratch_root(home),
        )
    except (outputs.OutputError, OSError):
        return None
    return plan.path if Path(plan.path).is_file() else None


def progress_of(note: Any) -> dict[str, Any] | None:
    """A progress note as a job row keeps it: done, total and message."""
    if not isinstance(note, dict):
        return None
    return {key: note.get(key) for key in ("done", "total", "message")}


def ending(
    kind: str,
    operation_id: str | None,
    payload: Mapping[str, Any],
    *,
    cancelled: bool = False,
    session_ended: bool = False,
) -> tuple[str, dict[str, Any], Any]:
    """How a job ended, what it made and what went wrong, from its call's answer.

    Work that saw its cancel and stopped is `cancelled`. Work that stopped
    because its session was going down, with no cancel, is `lost` with
    `SESSION_ENDED`, as it would be had the session gone first. Otherwise a call
    that failed, or Python code that raised, is `failed`, and the rest is
    `done`. The outputs hold the answer the call gave, so a job can be read
    back whole after the receipt that also holds it has gone.
    """
    outputs: dict[str, Any] = {"operation_id": operation_id}
    if not payload.get("ok"):
        if cancelled:
            return "cancelled", outputs, payload.get("error")
        if session_ended:
            return "lost", outputs, store_module.SESSION_ENDED_ERROR
        return "failed", outputs, payload.get("error")
    data = payload.get("data")
    data = dict(data) if isinstance(data, Mapping) else {}
    outputs["answer"] = data
    for key in ("undo", "scene_epoch", "cut", "lossy"):
        if payload.get(key) is not None:
            outputs[key] = payload[key]
    error = data.get("error") if kind == "python" else None
    if cancelled:
        return "cancelled", outputs, error
    if session_ended:
        return "lost", outputs, store_module.SESSION_ENDED_ERROR
    return ("failed" if error is not None else "done"), outputs, error
