"""Bake ONE material to EXR maps in its own hython process, then exit.

usage: hython bake.py <recipes.py> <index or function name> <resolution> <outdir> [b16|b32]

<recipes.py> exposes ALL, a list of functions f(parent, resolution) that build one Copernicus
network each and return the builder (cops_lib.B, usually via Material.done()).

Why it is shaped this way:
  * one process per material, because Copernicus keeps every cooked layer and a long-lived
    session grows until the machine runs out of memory
  * a streamed cook: nodes cook in dependency order and each one's layers are freed as soon
    as nothing downstream still needs them, so the live set is the frontier, not the network
  * each map is written the moment its output cooks, straight from the layer. An Image ROP
    re-cooks the whole network when it runs, which at 8K is what exhausts memory
  * b16 (half float layers) roughly halves the footprint and is indistinguishable in the maps
"""

# Node code and VEX snippets are embedded as strings, so long lines and percent
# formatting are left as they are.
# ruff: noqa: E501, UP031

import importlib.util
import json
import os
import subprocess
import sys
import time

import hou
import numpy as np
import OpenImageIO as oiio

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import cops_lib as L  # noqa: E402

rec, which, res, out = (
    os.path.abspath(sys.argv[1]),
    sys.argv[2],
    int(sys.argv[3]),
    os.path.abspath(sys.argv[4]),
)
precision = sys.argv[5] if len(sys.argv) > 5 else "b16"
sys.path.insert(0, os.path.dirname(rec))
spec = importlib.util.spec_from_file_location("recipes", rec)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
fn = M.ALL[int(which) - 1] if which.isdigit() else [f for f in M.ALL if f.__name__ == which][0]
t0 = time.time()


def free():
    """system-wide free memory, as text, for the log"""
    try:
        o = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
        return o.strip().splitlines()[-1].split(":")[-1].strip()
    except Exception:
        return "?"


def write_layer(node, path, key):
    lay = node.layer()
    w, h = lay.bufferResolution()
    c = lay.channelCount()
    dt = np.float16 if "loat16" in str(lay.storageType()) else np.float32
    a = np.frombuffer(lay.allBufferElements(), dtype=dt).reshape(h, w, c)[::-1]
    if c > 3:
        a = a[:, :, :3]
        c = 3
    sp = oiio.ImageSpec(w, h, c, oiio.FLOAT if key == "height" else oiio.HALF)
    sp.channelnames = ("R", "G", "B") if c == 3 else ("Y",)  # single channel wherever the data is
    sp.attribute("compression", "dwaa" if key == "basecolor" else "zip")
    if key == "basecolor":
        sp.attribute("dwaCompressionLevel", 45.0)
    sp.attribute("oiio:ColorSpace", "lin_rec709" if key == "basecolor" else "Raw")
    o = oiio.ImageOutput.create(path)
    assert o and o.open(path, sp), oiio.geterror()
    ok = o.write_image(
        np.ascontiguousarray(a.astype(np.float32 if key == "height" else np.float16, copy=False))
    )
    o.close()
    assert ok, path


parent = hou.node("/obj").createNode("subnet", "IMAGE_TO_COPS")
b = fn(parent, 1024)
net = b.net
net.parm("setprecision").set(1)
net.parm("precision").set(precision)
net.parmTuple("res").set((res, res))
d = os.path.join(out, b.name)
os.makedirs(d, exist_ok=True)

targets = [net.node("OUT_" + k) for k in L.MAPS]
key_of = dict((t.path(), k) for t, k in zip(targets, L.MAPS, strict=True))
order, seen = [], set()


def visit(n):
    if n.path() in seen:
        return
    seen.add(n.path())
    for x in n.inputs():
        if x:
            visit(x)
    order.append(n)


for t in targets:
    visit(t)
pending = {n.path(): len(set(o.path() for o in n.outputs() if o.path() in seen)) for n in order}
freed = 0
for n in order:
    n.cook(force=False)
    if n == b.Hmm:
        b.set_height_range(force=False)  # displacement range measured at the bake resolution
    if n.path() in key_of:
        k = key_of[n.path()]
        write_layer(n, os.path.join(d, f"{b.name}_{L.MAPS[k]}.exr"), k)
        print(f"  wrote {k}, free {free()}", flush=True)
    for x in set(y for y in n.inputs() if y):
        pending[x.path()] -= 1
        if pending[x.path()] <= 0:
            r = hou.hscript("opunload -R " + x.path())
            assert not r[1].strip(), r[1]
            freed += 1
assert len([f for f in os.listdir(d) if f.endswith(".exr")]) == len(L.MAPS)
json.dump(
    {
        "name": b.name,
        "tile": b.tile,
        "disp_mm": b.ctlnode.evalParm("disp_range_mm"),
        "res": res,
        "controls": dict((k, b.ctlnode.evalParm(k)) for k in b.ctls),
    },
    open(os.path.join(d, "meta.json"), "w"),
    indent=1,
)
print(
    "BAKED %s %d in %.0fs (%d nodes, %d freed)" % (b.name, res, time.time() - t0, len(order), freed)
)
