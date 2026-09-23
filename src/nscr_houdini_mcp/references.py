"""Reference images a scene is compared against, and the series each one starts.

A reference is registered once and never changed. Registering copies the image
into the `reference` output folder beside the scene (`$HIP/.agent/reference/`
by default) and writes one record next to it, `<ref_id>.json`, created
exclusively so nothing writes over it. The record holds the content hash, the
colour profile the file carries or the assumption made about it, and what a
comparison needs to be repeated the same way: the camera it was framed for,
named detail regions and a mask. Registering the same name again writes a new
record under a new id; the name then points at the newest, and every
comparison made against the new one starts a new series.

A series is the run of comparisons that can be read against each other: the
same reference id and hash, camera, crop, mask, colour records and settings.
Each series keeps an append only log, one line a run, in `series/` under the
reference folder, so the trend survives the server and is shared by every
process working on the scene.

This module never imports `hou`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import outputs as outputs_module

KIND = "reference"
ID_PREFIX = "ref-"
SERIES_DIR = "series"
SERIES_PREFIX = "ser-"

# How many earlier runs a trend carries.
TREND_RUNS = 10

HASH_CHUNK = 1 << 20


def file_hash(path: str | Path) -> str:
    """The SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def folder(
    *,
    home: str | Path,
    hip_path: str | Path | None,
    session_id: str | None,
    scratch_root: str | Path | None = None,
) -> Path:
    """Where this scene's references live. Nothing is made on disk."""
    conventions = outputs_module.load_conventions(home=home, hip_path=hip_path)
    plan = outputs_module.plan_path(
        KIND,
        run_id="probe",
        name="probe",
        hip_path=hip_path,
        session_id=session_id,
        conventions=conventions,
        scratch_root=scratch_root,
    )
    return Path(plan.directory)


def records(place: Path) -> list[dict[str, Any]]:
    """Every record in a reference folder, oldest first. Unreadable ones are skipped."""
    found: list[dict[str, Any]] = []
    try:
        entries = sorted(place.glob(f"{ID_PREFIX}*.json"))
    except OSError:
        return found
    for entry in entries:
        try:
            record = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("ref_id") == entry.stem:
            found.append(record)
    found.sort(key=lambda record: (float(record.get("created", 0.0)), str(record["ref_id"])))
    return found


def current(place: Path) -> dict[str, dict[str, Any]]:
    """The newest record under each name."""
    latest: dict[str, dict[str, Any]] = {}
    for record in records(place):
        latest[str(record.get("name"))] = record
    return latest


def find(place: Path, handle: str) -> dict[str, Any] | None:
    """A record by its id, or the newest one under a name."""
    everything = records(place)
    for record in everything:
        if record.get("ref_id") == handle:
            return record
    named = [record for record in everything if record.get("name") == handle]
    return named[-1] if named else None


def names(place: Path) -> list[str]:
    return sorted(current(place))


def listing(place: Path) -> list[dict[str, Any]]:
    """One row a name: the id in force, and how many were registered before it."""
    everything = records(place)
    rows = []
    for name, record in sorted(current(place).items()):
        earlier = [item["ref_id"] for item in everything if item.get("name") == name]
        rows.append(
            {
                "name": name,
                "ref_id": record["ref_id"],
                "image": record.get("image"),
                "width": record.get("width"),
                "height": record.get("height"),
                "camera": record.get("camera"),
                "regions": sorted(record.get("regions") or {}),
                "mask": bool(record.get("mask")),
                "created": record.get("created_iso"),
                "replaced": earlier[:-1],
            }
        )
    return rows


def register(
    store: Any,
    *,
    home: str | Path,
    hip_path: str | Path | None,
    session_id: str | None,
    name: str,
    source: str | Path,
    colour: Mapping[str, Any],
    size: tuple[int, int] | None,
    camera: str | None = None,
    regions: Mapping[str, list[float]] | None = None,
    mask: str | Path | None = None,
    scratch_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy the image in, write its record, and hand the record back.

    The copy goes through the output allocation, so it gets a run id, a run
    record and a readable sidecar like any other output. The record is made
    with an exclusive create, so an id is written once.
    """
    source_path = Path(source)
    extension = source_path.suffix.lstrip(".").lower() or "png"
    conventions = outputs_module.load_conventions(home=home, hip_path=hip_path)
    plan = outputs_module.allocate(
        store,
        KIND,
        name=name,
        hip_path=hip_path,
        session_id=session_id,
        ext=extension,
        conventions=conventions,
        scratch_root=scratch_root,
    )
    place = Path(plan.directory)
    before = find(place, plan.name)
    try:
        shutil.copyfile(source_path, plan.path)
        mask_record = None
        if mask is not None:
            mask_path = Path(mask)
            mask_suffix = mask_path.suffix.lstrip(".").lower() or "png"
            copied = place / f"{Path(plan.path).stem}_mask.{mask_suffix}"
            shutil.copyfile(mask_path, copied)
            mask_record = {
                "path": str(copied),
                "source": str(mask_path),
                "sha256": file_hash(copied),
            }
        now = time.time()
        ref_id = ID_PREFIX + plan.run_id.removeprefix("run-")
        record = {
            "ref_id": ref_id,
            "name": plan.name,
            "image": str(Path(plan.path)),
            "template": plan.template,
            "source": str(source_path),
            "sha256": file_hash(plan.path),
            "bytes": Path(plan.path).stat().st_size,
            "width": size[0] if size else None,
            "height": size[1] if size else None,
            "colour": dict(colour),
            "camera": camera,
            "regions": {key: list(value) for key, value in (regions or {}).items()},
            "mask": mask_record,
            "hip_family": plan.hip_family,
            "hip_path": None if hip_path is None else str(hip_path),
            "run_id": plan.run_id,
            "replaces": before["ref_id"] if before else None,
            "created": now,
            "created_iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        }
        write_once(place / f"{ref_id}.json", record)
    except BaseException:
        outputs_module.release(store, plan)
        Path(plan.path).unlink(missing_ok=True)
        raise
    return record


def write_once(path: Path, record: Mapping[str, Any]) -> None:
    """Write a record that must never be written again."""
    text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(text)


# Section: series


def series_id(key: Mapping[str, Any]) -> str:
    text = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
    return SERIES_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def series_path(place: Path, identity: str) -> Path:
    return place / SERIES_DIR / f"{identity}.jsonl"


def series_runs(place: Path, identity: str) -> list[dict[str, Any]]:
    """Every run logged under a series, oldest first."""
    try:
        text = series_path(place, identity).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    runs = []
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            runs.append(entry)
    return runs


def last_series_for(place: Path, name: str, *, besides: str) -> dict[str, Any] | None:
    """The newest run logged under the same compare name in another series."""
    newest: dict[str, Any] | None = None
    try:
        logs = list((place / SERIES_DIR).glob(f"{SERIES_PREFIX}*.jsonl"))
    except OSError:
        return None
    for log in logs:
        if log.stem == besides:
            continue
        for entry in series_runs(place, log.stem):
            if entry.get("name") != name:
                continue
            if newest is None or float(entry.get("when", 0)) > float(newest.get("when", 0)):
                newest = dict(entry, series_id=log.stem)
    return newest


def log_run(place: Path, identity: str, entry: Mapping[str, Any]) -> None:
    """Add one run to a series log, in one write."""
    path = series_path(place, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(line)


def changed_fields(before: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """Which parts of a series key differ, by name."""
    fields = sorted(set(before) | set(now))
    return [name for name in fields if before.get(name) != now.get(name)]


def trend(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The earlier runs of a series, newest last, and the change since the last one."""
    earlier = [run for run in runs if isinstance(run.get("metrics"), Mapping)]
    if not earlier:
        return None
    kept = earlier[-TREND_RUNS:]
    points = [
        {
            "run_id": run.get("run_id"),
            "when": run.get("when_iso"),
            "mae": _overall(run["metrics"], "mae"),
            "rmse": _overall(run["metrics"], "rmse"),
            "psnr_db": run["metrics"].get("psnr_db"),
            "diff_area_pct": run["metrics"].get("diff_area_pct"),
        }
        for run in kept
    ]
    return {"runs": len(earlier), "earlier": points}


def change_since(last: Mapping[str, Any] | None, now: Mapping[str, Any] | None) -> dict | None:
    """How the headline numbers moved since the run before. Negative is closer."""
    if not last or not now:
        return None
    moved: dict[str, float] = {}
    for key in ("mae", "rmse", "diff_area_pct"):
        before, after = (
            last.get(key),
            (_overall(now, key) if key != "diff_area_pct" else now.get(key)),
        )
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            moved[key] = round(float(after) - float(before), 6)
    return moved or None


def _overall(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, Mapping):
        overall = value.get("overall")
        return float(overall) if isinstance(overall, (int, float)) else None
    return None
