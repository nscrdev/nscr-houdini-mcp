"""image-to-cops: a thin builder over Copernicus so a material reads as a recipe.

A thin layer over Copernicus nodes so each material reads as a recipe:
native noises / grunge / blur nodes for the signal, small OpenCL COPs for
math, plus a few periodic layout kernels (scattered pits, panel grids, fixings, boards).

Conventions
  * image space is -1..1, every material tiles over that square
  * heights are carried in millimetres until the finishing stage
  * colours are linear Rec.709 (linear sRGB)
"""

# Node code and VEX snippets are embedded as strings, so long lines and percent
# formatting are left as they are.
# ruff: noqa: E501, UP031

import os
import re

import hou

HB = hou.text.expandString("$HB")
OIIO = os.path.join(HB, "hoiiotool")

PRE = r"""
static float sat(float x){return clamp(x,0.0f,1.0f);}
static int wrapi(int a,int n){int r=a%n; return r<0?r+n:r;}
static float h1(int x,int y,int s){uint n=(uint)x*374761393u+(uint)y*668265263u+(uint)s*1442695041u; n=(n^(n>>13))*1274126177u; n^=n>>16; return (float)(n&0xffffffu)/16777215.0f;}

// scattered irregular pits. returns a 0..1 cone (1 = centre)
static float pits(float2 p,int cells,float dens,float rmin,float rmax,float rpow,float irr,float asp,int seed)
{
    float2 u=(p*0.5f+0.5f)*(float)cells; int ix=(int)floor(u.x), iy=(int)floor(u.y); float best=0.0f;
    for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
        int cx=ix+dx, cy=iy+dy; int mx=wrapi(cx,cells), my=wrapi(cy,cells);
        if(h1(mx,my,seed)>dens) continue;
        float2 c=(float2)((float)cx+h1(mx,my,seed+1),(float)cy+h1(mx,my,seed+2));
        float r=mix(rmin,rmax,pow(h1(mx,my,seed+3),rpow));
        float2 d=u-c; d.x*=asp;
        float ang=atan2(d.y,d.x);
        float rr=r*(1.0f+irr*(0.5f*sin(ang*3.0f+h1(mx,my,seed+4)*6.2832f)+0.3f*sin(ang*5.0f+h1(mx,my,seed+5)*6.2832f)));
        best=max(best,1.0f-length(d)/max(rr,1e-5f));
    }
    return sat(best);
}

// formwork panel grid. mode 0 seam distance (image units), 1 panel random,
// 2 vertical seam distance, 3 horizontal seam distance
static float panels(float2 p,int nx,int ny,float stag,int mode)
{
    float2 u=(p*0.5f+0.5f)*(float2)((float)nx,(float)ny); float row=floor(u.y);
    u.x+=stag*fmod(row,2.0f);
    float2 f=u-floor(u);
    float dx=min(f.x,1.0f-f.x)/(float)nx*2.0f, dy=min(f.y,1.0f-f.y)/(float)ny*2.0f;
    if(mode==0) return min(dx,dy);
    if(mode==1) return h1(wrapi((int)floor(u.x),nx),wrapi((int)row,ny),91);
    if(mode==2) return dx;
    return dy;
}

// tie holes laid out cols x rows inside every panel. returns 0..1 cone
static float ties(float2 p,int nx,int ny,float stag,int cols,int rows,float mx,float my,float r)
{
    float2 n=(float2)((float)nx,(float)ny);
    float2 u=(p*0.5f+0.5f)*n; float row=floor(u.y); u.x+=stag*fmod(row,2.0f);
    float2 f=u-floor(u); float best=0.0f;
    for(int j=0;j<rows;j++) for(int i=0;i<cols;i++){
        float hx=cols>1?mx+(1.0f-2.0f*mx)*(float)i/(float)(cols-1):0.5f;
        float hy=rows>1?my+(1.0f-2.0f*my)*(float)j/(float)(rows-1):0.5f;
        float2 d=(f-(float2)(hx,hy))/n*2.0f;
        best=max(best,1.0f-length(d)/r);
    }
    return sat(best);
}

// variable width form boards. axis 0 = boards run vertically.
// mode 0 edge distance, 1 board random, 2 coord across board, 3 end joint distance, 4 segment random
static float bedge(int i,int n,float jit,int seed){return (float)i+(h1(wrapi(i,n),17,seed)-0.5f)*jit;}
static float boards(float2 p,int n,float jit,int seed,int axis,int segs,int mode)
{
    float2 uv=p*0.5f+0.5f; float t=(axis==0?uv.x:uv.y)*(float)n; float a=(axis==0?uv.y:uv.x);
    int i=(int)floor(t); int b=i;
    if(t<bedge(i,n,jit,seed)) b=i-1; else if(t>=bedge(i+1,n,jit,seed)) b=i+1;
    float lo=bedge(b,n,jit,seed), hi=bedge(b+1,n,jit,seed); int bw=wrapi(b,n);
    float r=h1(bw,3,seed);
    float a2=a*(float)segs+r*7.13f; float fa=a2-floor(a2);
    if(mode==0) return min(t-lo,hi-t)/(float)n*2.0f;
    if(mode==1) return r;
    if(mode==2) return (t-lo)/(hi-lo);
    if(mode==3) return min(fa,1.0f-fa)/(float)segs*2.0f;
    return h1(bw,wrapi((int)floor(a2),segs),seed+5);
}
"""

_FLT = re.compile(r"(?<![\w.])(\d+\.\d*|\.\d+)(?![\w.])")


def _f(code):
    return _FLT.sub(lambda m: m.group(1) + "f", code)


def col(c):
    return "(float3)({:.5f}f,{:.5f}f,{:.5f}f)".format(*tuple(c))


def srgb(r, g, b):
    """8 bit sRGB triple -> linear"""

    def lin(v):
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    return (lin(r), lin(g), lin(b))


class B:
    def __init__(s, parent, name, tile_m, res=1024, title=""):
        old = parent.node(name)
        if old:
            old.destroy()
        s.net = parent.createNode("copnet", name)
        s.name = name
        s.tile = tile_m
        s.net.parm("setres").set(1)
        s.net.parmTuple("res").set((res, res))
        s.net.parm("setborder").set(1) if s.net.parm("setborder") else None
        try:
            s.net.parm("border").set("wrap")
        except Exception:
            pass
        s.net.setComment(title)
        s.net.setGenericFlag(hou.nodeFlag.DisplayComment, True)
        s.ctlnode = s.net.createNode("null", "CONTROLS")
        s.ctlnode.setColor(hou.Color(1, 0.85, 0.2))
        ptg = s.ctlnode.parmTemplateGroup()
        ptg.append(
            hou.FloatParmTemplate("tile_size_m", "Tile Size (m, info)", 1, default_value=(tile_m,))
        )
        s.ctlnode.setParmTemplateGroup(ptg)
        s.ctls = {}
        s.seed = 0
        s.ref = s.net.createNode("constant", "size_ref")
        s.ref.parm("signature").set("f1")

    # ---------------------------------------------------------------- core
    def ctl(s, name, default, lo=0.0, hi=1.0, label=None):
        if name not in s.ctls:
            ptg = s.ctlnode.parmTemplateGroup()
            ptg.append(
                hou.FloatParmTemplate(
                    name,
                    label or name.replace("_", " ").title(),
                    1,
                    default_value=(default,),
                    min=lo,
                    max=hi,
                )
            )
            s.ctlnode.setParmTemplateGroup(ptg)
            s.ctls[name] = default
        return "$" + name

    def n(s, typ, name=None, ins=(), **p):
        nd = s.net.createNode(typ, name) if name else s.net.createNode(typ)
        for i, src in enumerate(ins):
            if src is None:
                continue
            if isinstance(src, tuple):
                nd.setInput(i, src[0], src[1])
            else:
                nd.setInput(i, src)
        for k, v in p.items():
            if isinstance(v, str) and v.startswith("$"):
                nd.parm(k).setExpression(f'ch("../CONTROLS/{v[1:]}")')
            elif isinstance(v, (tuple, list)):
                nd.parmTuple(k).set(v)
            else:
                nd.parm(k).set(v)
        return nd

    def cl(s, body, out="float", name=None, **ins):
        """OpenCL COP. body = statements ending in @dst.set(...).
        float inputs by name; names starting c_ are float3; '$ctl' / numbers become bound parms."""
        nd = s.net.createNode("opencl", name) if name else s.net.createNode("opencl")
        layers = [(k, v) for k, v in ins.items() if isinstance(v, (hou.Node, tuple))]
        consts = [(k, v) for k, v in ins.items() if not isinstance(v, (hou.Node, tuple))]
        if not layers:
            layers = [("ref", s.ref)]
        binds = []
        nd.parm("inputs").set(len(layers))
        for i, (k, v) in enumerate(layers):
            t = "float3" if k.startswith("c_") else "float"
            nd.parm("input%d_name" % (i + 1)).set(k)
            nd.parm("input%d_type" % (i + 1)).set(t)
            binds.append(f"#bind layer {k}? {t} val=0")
            if isinstance(v, tuple):
                nd.setInput(i, v[0], v[1])
            else:
                nd.setInput(i, v)
        nd.parm("outputs").set(1)
        nd.parm("output1_name").set("dst")
        nd.parm("output1_type").set(out)
        binds.append(f"#bind layer !&dst {out}")
        nd.parm("bindings").set(len(consts))
        for i, (k, v) in enumerate(consts):
            j = i + 1
            nd.parm("bindings%d_name" % j).set(k)
            nd.parm("bindings%d_type" % j).set("float")
            if isinstance(v, str):
                nd.parm("bindings%d_fval" % j).setExpression(f'ch("../CONTROLS/{v[1:]}")')
            else:
                nd.parm("bindings%d_fval" % j).set(float(v))
            binds.append(f"#bind parm {k} float val=0")
        code = "\n".join(binds) + "\n" + PRE + "\n@KERNEL\n{\n" + _f(body) + "\n}\n"
        nd.parm("kernelcode").set(code)
        return nd

    def x(s, expr, name=None, **ins):
        """float expression over float inputs / controls."""
        decl = ""
        for k in ins:
            decl += (
                f"    float {k}=@{k};\n" if not k.startswith("c_") else f"    float3 {k}=@{k};\n"
            )
        return s.cl(decl + f"    @dst.set({expr});", name=name, **ins)

    def rgb(s, expr, name=None, **ins):
        decl = ""
        for k in ins:
            decl += (
                f"    float {k}=@{k};\n" if not k.startswith("c_") else f"    float3 {k}=@{k};\n"
            )
        return s.cl(decl + f"    @dst.set({expr});", out="float3", name=name, **ins)

    # ------------------------------------------------------------ signals
    def _sd(s, seed):
        if seed is None:
            s.seed += 1
            seed = s.seed
        return (seed * 1.371 + 0.13, seed * 2.113 + 0.57)

    def fn(s, size, oct=6, rough=0.5, typ="perlin", sx=1.0, sy=1.0, seed=None, frac=None, **kw):
        p = dict(elementsize=size, oct=oct, rough=rough, noisetype=typ, off=s._sd(seed))
        if sx != 1.0 or sy != 1.0:
            p["elementsizetype"] = 1
            p["elementscale"] = (sx, sy)
        if frac is not None:
            p["fractaltype"] = frac
        p.update(kw)
        return s.n("fractalnoise", **p)

    def gr(s, kind, size, seed=None, **kw):
        if seed is None:
            s.seed += 1
            seed = s.seed
        if kind == "layerednoise":
            # the stock base octaves are tile sized; keep them small so the tile does not read when repeated
            kw.setdefault("basenoise_elementsize", 0.3)
            kw.setdefault("basecells_elementsize", 0.2)
        nd = s.n("grunge_" + kind, elementsize=size, **kw)
        if nd.parm("seed"):
            nd.parm("seed").set(seed)
        else:
            nd.parmTuple("off").set(s._sd(seed))
        return nd

    def wor(s, size, jit=1.0, out=0, seed=None, **kw):
        nd = s.n("worleynoise", elementsize=size, jitter=jit, off=s._sd(seed), **kw)
        return (nd, out)

    def mm(s, v):
        """millimetres -> image units"""
        return v * 0.002 / s.tile

    def lv(s, a, lo, hi, g=1.0):
        return s.x("pow(sat((a-(LO))/((HI)-(LO))),G)", a=a, LO=lo, HI=hi, G=g)

    def blur(s, a, size):
        return s.n("blur", ins=[a], size=size)

    def sblur(s, a, angle, length):
        return s.n("streakblur", ins=[a], angle=angle, length=length)

    def warp(s, a, amt, size=0.3, seed=None, oct=4):
        nx = s.fn(size, oct=oct, seed=seed)
        ny = s.fn(size, oct=oct, seed=None)
        return s.cl(
            "    float2 o=(float2)(@nx-0.5,@ny-0.5)*@AMT;\n    @dst.set(@a.imageSample(@P+o));",
            a=a,
            nx=nx,
            ny=ny,
            AMT=amt,
        )

    def pits(s, cells, dens=0.5, rmin=0.1, rmax=0.5, rpow=2.0, irr=0.18, asp=1.0, seed=None):
        if seed is None:
            s.seed += 1
            seed = s.seed
        return s.cl(
            "    @dst.set(pits(@P,%d,@D,%f,%f,%f,%f,%f,%d));"
            % (cells, rmin, rmax, rpow, irr, asp, seed * 13),
            D=dens,
        )

    def panels(s, nx, ny, stag=0.0, mode=0):
        return s.cl("    @dst.set(panels(@P,%d,%d,%f,%d));" % (nx, ny, stag, mode))

    def ties(s, nx, ny, cols, rows, mx, my, r, stag=0.0):
        return s.cl(
            "    @dst.set(ties(@P,%d,%d,%f,%d,%d,%f,%f,%f));"
            % (nx, ny, stag, cols, rows, mx, my, r)
        )

    def boards(s, n, jit=0.5, seed=1, axis=0, segs=1, mode=0):
        return s.cl(
            "    @dst.set(boards(@P,%d,%f,%d,%d,%d,%d));" % (n, jit, seed, axis, segs, mode)
        )

    def shift_by(s, a, idn, ax, ay):
        """sample a shifted by a per-id random offset (breaks continuity across boards/panels)"""
        return s.cl(
            f"    float r=@id;\n    @dst.set(@a.imageSample(@P+(float2)({ax:f}*r,{ay:f}*fmod(r*7.31,1.0))));",
            a=a,
            id=idn,
        )

    def cracks(s, size, width, seed=None, cover=0.5, wamt=0.06, msize=0.5):
        b = s.wor(size, jit=1.0, out=2, seed=seed)
        w = s.warp(b, wamt, size=size * 0.6)
        w2 = s.warp(w, wamt * 0.15, size=size * 0.05)
        m = s.fn(msize, oct=3)
        return s.x(
            "(1.0-smoothstep(0.0,W,a))*smoothstep(C-0.08,C+0.08,m)",
            a=w2,
            m=m,
            W=width,
            C=1.0 - cover,
        )

    def colmix(s, base, layers, name=None):
        """base colour, then [(mask, colour, amount)] mixed in order."""
        ins = {}
        body = f"    float3 c={col(base)};\n"
        for i, (m, c, a) in enumerate(layers):
            ins["m%d" % i] = m
            ins["A%d" % i] = a
            body += "    c=mix(c,%s,sat(@m%d*@A%d));\n" % (col(c), i, i)
        body += "    @dst.set(c);"
        return s.cl(body, out="float3", name=name, **ins)

    # ----------------------------------------------------------- finishing
    def finish(s, h_mm, albedo, rough, ao_radius=0.015, spec=None):
        t = s.tile
        s.ctl("height_scale", 1.0, 0, 3)
        s.ctl("albedo_gain", 1.0, 0, 2)
        s.ctl("albedo_saturation", 1.0, 0, 2)
        s.ctl("roughness_gain", 1.0, 0, 2)
        s.ctl("roughness_offset", 0.0, -0.5, 0.5)
        H = s.x("a*K", name="height_mm", a=h_mm, K="$height_scale")
        H.setComment("height in millimetres")
        himg = s.x("a*%g" % (0.002 / t), a=H)
        nrm = s.n("heighttonormal", ins=[himg], normaltype="offset")
        ao0 = s.n("heighttoambientocclusion", ins=[himg], viewradius=ao_radius)
        ao = s.x("sat(a)", a=ao0)
        # normalise height around its own range (set by set_height_range)
        s.rng = s.ctl("disp_range_mm", 10.0, 0.1, 100, "Displacement Range (mm)")
        hn = s.x("sat(a/R+0.5)", a=H, R=s.rng)
        alb = s.cl(
            "    float3 c=@c_a*@G; float l=dot(c,(float3)(0.2126,0.7152,0.0722));\n"
            "    c=mix((float3)(l),c,@S); @dst.set(clamp(c,0.0,1.0));",
            out="float3",
            c_a=albedo,
            G="$albedo_gain",
            S="$albedo_saturation",
        )
        rg = s.x("clamp(a*G+O,0.03,1.0)", a=rough, G="$roughness_gain", O="$roughness_offset")
        s.ctl("specular_gain", 1.0, 0, 2)
        sp = s.x(
            "clamp(a*G,0.0,1.0)", a=spec if spec is not None else s.x("1.0"), G="$specular_gain"
        )
        s.out = {}
        for k, nd in (
            ("basecolor", alb),
            ("roughness", rg),
            ("specular", sp),
            ("height", hn),
            ("normal", nrm),
            ("ao", ao),
        ):
            o = s.n("null", "OUT_" + k, ins=[nd])
            o.setColor(hou.Color(0.3, 0.8, 0.3))
            s.out[k] = o
        s.Hmm = H
        # cheap lit preview (raking light), display referred
        pv = s.cl(
            """    float3 n=normalize(@c_n*2.0-1.0);
    float3 L=normalize((float3)(-0.55,0.62,0.5));
    float3 V=(float3)(0.0,0.0,1.0); float3 Hh=normalize(L+V);
    float r=max(@r,0.05); float a2=r*r*r*r;
    float nh=max(dot(n,Hh),0.0); float d=nh*nh*(a2-1.0)+1.0;
    float spec=0.04*@sw*a2/(3.14159*d*d);
    float nl=max(dot(n,L),0.0);
    float amb=0.22*@ao;
    float3 c=@c_a*(nl*1.5+amb)+(float3)(spec*nl*1.5);
    @dst.set(pow(clamp(c,0.0,1.0),(float3)(1.0/2.2)));""",
            out="float3",
            c_n=nrm,
            c_a=alb,
            r=rg,
            ao=ao,
            sw=sp,
        )
        s.out["preview"] = s.n("null", "OUT_preview", ins=[pv])
        s.net.layoutChildren()
        return s

    def set_height_range(s, force=True):
        s.Hmm.cook(force=force)
        L = s.Hmm.layer()
        m = max(abs(L.computeMin()), abs(L.computeMax()))
        r = max(0.5, round(m * 2.0 * 1.05 + 0.05, 1))
        s.ctlnode.parm("disp_range_mm").set(r)
        ptg = s.ctlnode.parmTemplateGroup()
        pt = ptg.find("disp_range_mm")
        pt.setDefaultValue((r,))
        ptg.replace("disp_range_mm", pt)
        s.ctlnode.setParmTemplateGroup(ptg)
        return r


class Material:
    """material state: height (mm), colour layers, tone multipliers, roughness deltas"""

    def __init__(s, b, base_col, base_rough, base_spec=0.8):
        s.b = b
        s.h = None
        s.base_col = base_col
        s.base_rough = base_rough
        s.col = []  # (mask, colour, amount)
        s.tone = []  # (signed mask, amount)  multiplies albedo
        s.r = []  # (mask, delta)
        s.base_spec = base_spec
        s.sp = []  # (mask, delta)  specular weight

    def addh(s, mask, k):
        s.h = s.b.x("h+m*K", h=s.h, m=mask, K=k)

    def done(s, ao_radius=0.015):
        b = s.b
        alb = b.colmix(s.base_col, s.col)
        if s.tone:
            ins = {"c_a": alb}
            body = "    float3 c=@c_a; float t=1.0;\n"
            for i, (m, a) in enumerate(s.tone):
                ins["t%d" % i] = m
                ins["T%d" % i] = a
                body += "    t*=1.0+@t%d*@T%d;\n" % (i, i)
            body += "    @dst.set(c*max(t,0.05));"
            alb = b.cl(body, out="float3", **ins)
        ins = {}
        body = f"    float r={s.base_rough:f};\n"
        for i, (m, d) in enumerate(s.r):
            ins["m%d" % i] = m
            ins["D%d" % i] = d
            body += "    r+=@m%d*@D%d;\n" % (i, i)
        body += "    @dst.set(r);"
        rough = b.cl(body, **ins)
        ins = {}
        body = f"    float w={s.base_spec:f};\n"
        for i, (m, d) in enumerate(s.sp):
            ins["m%d" % i] = m
            ins["D%d" % i] = d
            body += "    w+=@m%d*@D%d;\n" % (i, i)
        body += "    @dst.set(w);"
        spec = b.cl(body, **ins)
        b.finish(s.h, alb, rough, ao_radius=ao_radius, spec=spec)
        return b


# map key -> file suffix. base colour is linear Rec.709, everything else is data
MAPS = {
    "basecolor": "BaseColor_lin_srgb",
    "normal": "Normal_raw",
    "roughness": "Roughness_raw",
    "specular": "Specular_raw",
    "ao": "AO_raw",
    "height": "Height_raw",
}
