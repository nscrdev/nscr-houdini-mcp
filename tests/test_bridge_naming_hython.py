"""A session's name against a real Houdini's scene events.

A Houdini started with a file on its command line runs its startup scripts,
and so starts the bridge, before that file loads. The bridge then finds the
untitled scene. This plays that order in a hython of its own: a bridge that
takes the part of one with a user interface starts on the untitled scene, and
the file loads after it. What only a real Houdini can say is whether it calls
the fresh scene new, and which file it reports once the load is over.

Skipped, not failed, when there is no Houdini on this machine. One hython, run
to the end and gone before the test returns, with a state folder of its own.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import support
from nscr_houdini_mcp.bridge.launcher import find_hython, hython_available

pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
]

RUN_TIMEOUT_S = 300.0

SCRIPT = """
import json
import sys
from pathlib import Path

import hou

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import registry
from nscr_houdini_mcp.bridge.app import Bridge, BridgeConfig

home, hip, first, last = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

hou.node("/obj").createNode("null", "kept")
hou.hipFile.save(hip)
hou.hipFile.clear(suppress_save_prompt=True)

found = {"new_at_start": hou.hipFile.isNewFile()}
bridge = Bridge(
    BridgeConfig(home=home, kind="gui", port_range=(first, last), heartbeat_s=3600.0)
)
found["started_as"] = bridge.start().alias
try:
    hou.hipFile.load(hip, suppress_save_prompt=True)
    found["alias"] = bridge.alias
    found["drift"] = bridge.identity.drift()
    found["epoch"] = bridge.scene_epoch
    found["problems"] = list(bridge.problems)
    with store_module.Store(bridge.store_path) as store:
        row = store.get_session(bridge.session_id)
        found["stored"] = [row.alias, row.previous_alias, row.hip_path]
        second = store.register_session(
            "second", kind="gui", pid=1, alias_template="untitled-{n}"
        )
        found["second"] = second.alias
        found["old_name_reaches"] = store.resolve_session("untitled-1").session_id
        found["session_id"] = bridge.session_id
        store.end_session("second")
    entry = registry.read_entry(registry.entry_path(home, bridge.session_id))
    found["file"] = [entry["alias"], entry["hip_path"]]
finally:
    bridge.stop()
print("RESULT " + json.dumps(found))
"""


def test_a_session_started_before_its_file_loads_is_named_after_the_file(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    scratch = tmp_path / "houdini-temp"
    scratch.mkdir()
    script = tmp_path / "naming.py"
    script.write_text(SCRIPT, encoding="utf-8")
    hip = tmp_path / "gui_v001.hip"
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    first, last = support.BRIDGE_PORTS
    done = subprocess.run(
        [str(find_hython()), str(script), str(home), str(hip), str(first), str(last)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": source_root,
            "HOUDINI_TEMP_DIR": str(scratch),
            "NSCR_MCP_HOME": str(home),
        },
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, done.stdout[-2000:]
    found = json.loads(lines[-1][len("RESULT ") :])

    assert found["new_at_start"] is True
    assert found["started_as"] == "untitled-1"
    assert found["alias"] == "gui_v001-1"
    assert found["stored"] == ["gui_v001-1", "untitled-1", hip.as_posix()]
    assert found["file"] == ["gui_v001-1", hip.as_posix()]
    # A second empty Houdini is not handed the name a caller may still hold.
    assert found["second"] == "untitled-2"
    assert found["old_name_reaches"] == found["session_id"]
    assert found["drift"] is None
    assert found["epoch"] == 1
    assert [line for line in found["problems"] if "scene" in line] == []
