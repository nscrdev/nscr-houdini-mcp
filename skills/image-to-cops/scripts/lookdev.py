"""image-to-cops: Karma XPU look-dev renders of ONE baked material, in its own hython process.

usage: hython lookdev.py <texroot> <material name> <outdir> [ball_rot] [close_rot] [exposure] [samples] [views] [ball_tiles]
  views       comma list of: ball, front, close, cam   (default ball,front,close)
  ball_rot    HDRI rotation for the shader ball; pick one that puts a highlight on it (default 345)
  close_rot   HDRI rotation for the wall views; pick one that rakes across the surface (default 110)


Two HDRI-lit set-ups, both with true displacement and a MaterialX standard surface:
  build_ball  Houdini's shipped material look-dev scene (standard shader ball + ground), tiles repeated over the ball's UVs
  build       a wall and floor: 'front' is an orthographic 3x3 tiling view, 'close' a raking close-up, 'cam' a wide

Colour: base colour EXRs are linear Rec.709 and are tagged so Karma converts them to the working space
(the OCIO scene_linear role). Renders are written in the working space and converted for display
with the config's default view, so what is judged is what a viewer would show.
"""

# Node code and VEX snippets are embedded as strings, so long lines and percent
# formatting are left as they are.
# ruff: noqa: E501, UP031

import json
import os
import subprocess
import sys
import time

import hou

HDRI = "$HFS/houdini/pic/hdri/HDRIHaven_kiara_5_noon_2k.rat"
BALL_SCENE = "$HFS/houdini/usd/scenes/material_lookdev/standard_shader_ball.usd"
BALL_TILES = 2

MAT = r"""
import hou
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf
node = hou.pwd()
stage = node.editableStage()
tex = node.evalParm('texdir'); name = node.evalParm('matname')
tile = node.evalParm('tile'); rng = node.evalParm('disp_mm') * 0.001

def make_mat(path, disp_scale):
    mat = UsdShade.Material.Define(stage, path)
    def sh(n, idn):
        s = UsdShade.Shader.Define(stage, path + '/' + n); s.CreateIdAttr(idn); return s
    def img(n, idn, suffix, typ, cs):
        s = sh(n, idn)
        a = s.CreateInput('file', Sdf.ValueTypeNames.Asset)
        a.Set(Sdf.AssetPath('%s/%s/%s_%s.exr' % (tex, name, name, suffix)))
        a.GetAttr().SetColorSpace(cs)
        s.CreateOutput('out', typ)
        return s
    bc = img('basecolor', 'ND_image_color3', 'BaseColor_lin_srgb', Sdf.ValueTypeNames.Color3f, 'lin_rec709')
    rg = img('roughness', 'ND_image_float', 'Roughness_raw', Sdf.ValueTypeNames.Float, 'Raw')
    sw = img('specular', 'ND_image_float', 'Specular_raw', Sdf.ValueTypeNames.Float, 'Raw')
    ht = img('height', 'ND_image_float', 'Height_raw', Sdf.ValueTypeNames.Float, 'Raw')
    ss = sh('surface', 'ND_standard_surface_surfaceshader')
    ss.CreateInput('base', Sdf.ValueTypeNames.Float).Set(1.0)
    ss.CreateInput('base_color', Sdf.ValueTypeNames.Color3f).ConnectToSource(bc.ConnectableAPI(), 'out')
    ss.CreateInput('specular', Sdf.ValueTypeNames.Float).ConnectToSource(sw.ConnectableAPI(), 'out')
    ss.CreateInput('specular_roughness', Sdf.ValueTypeNames.Float).ConnectToSource(rg.ConnectableAPI(), 'out')
    ss.CreateOutput('out', Sdf.ValueTypeNames.Token)
    sub = sh('centre', 'ND_subtract_float')
    sub.CreateInput('in1', Sdf.ValueTypeNames.Float).ConnectToSource(ht.ConnectableAPI(), 'out')
    sub.CreateInput('in2', Sdf.ValueTypeNames.Float).Set(0.5)
    sub.CreateOutput('out', Sdf.ValueTypeNames.Float)
    dp = sh('displacement', 'ND_displacement_float')
    dp.CreateInput('displacement', Sdf.ValueTypeNames.Float).ConnectToSource(sub.ConnectableAPI(), 'out')
    dp.CreateInput('scale', Sdf.ValueTypeNames.Float).Set(disp_scale)
    dp.CreateOutput('out', Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput('mtlx').ConnectToSource(ss.ConnectableAPI(), 'out')
    mat.CreateDisplacementOutput('mtlx').ConnectToSource(dp.ConnectableAPI(), 'out')
    return mat
"""

SCENE = (
    MAT
    + r"""
def quad(path, pts, st):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr([4]); m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateSubdivisionSchemeAttr('none')
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set([Gf.Vec2f(*s) for s in st])
    m.GetPrim().CreateAttribute('primvars:karma:object:dicingquality', Sdf.ValueTypeNames.Float).Set(1.5)
    return m

UsdGeom.Xform.Define(stage, '/scene')
W, H, D = 18.0, 13.0, 10.0
wall = quad('/scene/wall', [(-W/2, 0, 0), (W/2, 0, 0), (W/2, H, 0), (-W/2, H, 0)],
            [(-W/2/tile, 0), (W/2/tile, 0), (W/2/tile, H/tile), (-W/2/tile, H/tile)])
floor = quad('/scene/floor', [(-W/2, 0, D), (W/2, 0, D), (W/2, 0, 0), (-W/2, 0, 0)],
             [(-W/2/tile, -D/tile), (W/2/tile, -D/tile), (W/2/tile, 0), (-W/2/tile, 0)])
mat = make_mat('/mat/concrete', rng)
for g in (wall, floor):
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)

def camera(path, eye, tgt, focal, ortho=0.0):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200))
    if ortho:
        cam.CreateProjectionAttr('orthographic')
        cam.CreateHorizontalApertureAttr(ortho * 10.0); cam.CreateVerticalApertureAttr(ortho * 10.0)
    else:
        cam.CreateFocalLengthAttr(focal); cam.CreateHorizontalApertureAttr(36.0); cam.CreateVerticalApertureAttr(24.0)
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 1, 0)).GetInverse())
camera('/cameras/cam', (-2.4, 1.2, 6.4), (0.4, 1.9, 0.0), 30.0)        # wide
camera('/cameras/close', (0.75, 1.45, 1.0), (-0.25, 1.5, 0.0), 40.0)    # raking close-up for relief
camera('/cameras/front', (0.0, 1.5 * tile, 9.0), (0.0, 1.5 * tile, 0.0), 0.0, ortho=3.0 * tile)   # 3x3 tiles, front on
"""
)

BALL = (
    MAT
    + r"""
import numpy as np
for p in ('/World/box', '/World/lights'):          # the scene's own light box goes; the HDRI lights it instead
    prim = stage.GetPrimAtPath(p)
    if prim: prim.SetActive(False)
ball = stage.GetPrimAtPath('/World/shader_ball/material_surface')
mesh = UsdGeom.Mesh(ball)
# how much surface one UV unit covers, so the relief keeps its proportion to the tile
pts = np.array(mesh.GetPointsAttr().Get()) * 0.01
idx = np.array(mesh.GetFaceVertexIndicesAttr().Get()); cnt = np.array(mesh.GetFaceVertexCountsAttr().Get())
st = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').ComputeFlattened())
start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
d3 = np.linalg.norm(pts[idx[start]] - pts[idx[start + 1]], axis=1)
duv = np.linalg.norm(st[start] - st[start + 1], axis=1)
ok = duv > 1e-6
per_uv = float(np.median(d3[ok] / duv[ok]))
rep_n = max(node.evalParm('ball_tiles'), 1)        # tiles across the ball's UV square
pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
pv.Set([Gf.Vec2f(float(u) * rep_n, float(v) * rep_n) for u, v in st]); pv.BlockIndices()
mat = make_mat('/World/materials/image_to_cops', rng * per_uv / (tile * rep_n))
api = UsdShade.MaterialBindingAPI.Apply(ball)
api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)
api.Bind(mat, UsdShade.Tokens.strongerThanDescendants, UsdShade.Tokens.full)
ball.CreateAttribute('primvars:karma:object:dicingquality', Sdf.ValueTypeNames.Float).Set(2.0)
node.setUserData('per_uv', '%g' % per_uv)
"""
)


def _parms(py, texdir):
    ptg = py.parmTemplateGroup()
    ptg.append(hou.StringParmTemplate("texdir", "Texture Dir", 1, default_value=(texdir,)))
    ptg.append(hou.StringParmTemplate("matname", "Material", 1))
    ptg.append(hou.FloatParmTemplate("tile", "Tile Size", 1, default_value=(3.0,)))
    ptg.append(hou.FloatParmTemplate("disp_mm", "Displacement Range mm", 1, default_value=(10.0,)))
    ptg.append(
        hou.IntParmTemplate(
            "ball_tiles", "Tiles Across Shader Ball", 1, default_value=(BALL_TILES,)
        )
    )
    py.setParmTemplateGroup(ptg)


def _tail(stage, src, camera, hdri, rot, exposure, resx):
    sky = stage.createNode("domelight", "hdri")  # image based lighting only
    sky.setInput(0, src)
    sky.parm("primpath").set("/lights/hdri")
    sky.parm("xn__inputstexturefile_r3ah").set(hdri)
    sky.parm("xn__inputsintensity_i0a").set(1.0)
    sky.parm("xn__inputsexposure_vya").set(exposure)
    sky.parm("ry").set(rot)
    krs = stage.createNode("karmarendersettings", "krs")
    krs.setInput(0, sky)
    krs.parm("camera").set(camera)
    krs.parm("resolutionx").set(resx)
    krs.parm("engine").set("xpu")
    krs.parm("pathtracedsamples").set(64)
    krs.parm("diffuselimit").set(
        4
    )  # Karma defaults to 1 diffuse bounce, which leaves shadows too dark
    rop = stage.createNode("usdrender_rop", "render")
    rop.setInput(0, krs)
    stage.layoutChildren()
    return sky, krs, rop


def build(texdir, hdri=HDRI, rot=0.0, exposure=0.0):
    stage = hou.node("/stage")
    for n in stage.children():
        n.destroy()
    py = stage.createNode("pythonscript", "lookdev_scene")
    _parms(py, texdir)
    py.parm("python").set(SCENE)
    sky, krs, rop = _tail(stage, py, "/cameras/cam", hdri, rot, exposure, 1600)
    return py, None, sky, krs, rop


def build_ball(texdir, hdri=HDRI, rot=0.0, exposure=0.0):
    stage = hou.node("/stage")
    for n in stage.children():
        n.destroy()
    sub = stage.createNode("sublayer", "material_lookdev_scene")
    sub.parm("filepath1").set(BALL_SCENE)
    py = stage.createNode("pythonscript", "lookdev_scene")
    py.setInput(0, sub)
    _parms(py, texdir)
    py.parm("python").set(BALL)
    sky, krs, rop = _tail(stage, py, "/World/Camera", hdri, rot, exposure, 1400)
    return py, None, sky, krs, rop


# ---------------------------------------------------------------- command line
if __name__ == "__main__":
    import PyOpenColorIO as OCIO

    arg = sys.argv[1:]
    tex, name, out = os.path.abspath(arg[0]), arg[1], os.path.abspath(arg[2])
    ball_rot = float(arg[3]) if len(arg) > 3 else 345.0
    close_rot = float(arg[4]) if len(arg) > 4 else 110.0
    exposure = float(arg[5]) if len(arg) > 5 else 0.4
    samples = int(arg[6]) if len(arg) > 6 else 64
    views = arg[7].split(",") if len(arg) > 7 else ["ball", "front", "close"]
    BALL_TILES = int(arg[8]) if len(arg) > 8 else 2
    HDRI = os.environ.get("IMAGE_TO_COPS_HDRI", HDRI)
    cfg = OCIO.GetCurrentConfig()
    work = cfg.getCanonicalName("scene_linear")
    display = os.environ.get("IMAGE_TO_COPS_DISPLAY") or cfg.getDefaultDisplay()
    view = os.environ.get("IMAGE_TO_COPS_VIEW") or cfg.getDefaultView(display)
    print(f"working space {work} | display {display} | view {view}")
    meta = json.load(open(os.path.join(tex, name, "meta.json")))
    os.makedirs(out, exist_ok=True)
    oiiotool = hou.text.expandString("$HB/hoiiotool")
    t0 = time.time()

    def shoot(py, krs, rop, label, camera):
        krs.parm("pathtracedsamples").set(samples)
        py.parm("matname").set(name)
        py.parm("tile").set(meta["tile"])
        py.parm("disp_mm").set(meta["disp_mm"])
        krs.parm("camera").set(camera)
        pic = os.path.join(out, f"{name}_{label}.exr")
        krs.parm("picture").set(pic)
        rop.render()
        r = subprocess.run(
            [
                oiiotool,
                pic,
                "--ch",
                "R,G,B",
                "--ociodisplay:from=" + work,
                display,
                view,
                "-o",
                pic.replace(".exr", ".jpg"),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0 and not r.stderr.strip(), r.stderr[-400:]
        os.remove(pic)

    if "ball" in views:
        # the shipped scene's pale ground needs about a stop less light than the wall
        py, _, sky, krs, rop = build_ball(tex, hdri=HDRI, rot=ball_rot, exposure=exposure - 1.0)
        shoot(py, krs, rop, "ball", "/World/Camera")
    wall = [v for v in views if v != "ball"]
    if wall:
        py, _, sky, krs, rop = build(tex, hdri=HDRI, rot=close_rot, exposure=exposure)
        for v in wall:
            sky.parm("ry").set(ball_rot if v == "front" else close_rot)
            krs.parm("resolutionx").set(1400)
            shoot(py, krs, rop, "wide" if v == "cam" else v, "/cameras/" + v)
    print(f"SHOT {name} in {time.time() - t0:.0f}s")
