"""Worked example: nineteen concretes, each matched to a film still.

Read it for the pattern, not the numbers: one layer function per physical cause,
and a recipe per material that reads as a list of what happened to the surface.
Run one with:  hython scripts/bake.py examples/concrete_recipes.py 5 2048 <outdir>
"""

# Node code and VEX snippets are embedded as strings, so long lines and percent
# formatting are left as they are.
# ruff: noqa: E501, UP031

import cops_lib as L
from cops_lib import Material as S
from cops_lib import srgb


# ------------------------------------------------------------------ layers
def base(
    s,
    macro=0.6,
    meso=0.18,
    fine=0.06,
    sand=0.035,
    speck=0.06,
    micro=0.04,
    pores=0.3,
    pore_depth=0.35,
    rvar=0.12,
    boost=1.4,
):
    """cast cement paste: slab undulation, trowel-scale relief, sand grain, paste micro-texture and pores,
    plus a roughness structure that does not follow the colour"""
    b = s.b
    meso, fine, sand, micro, pore_depth = (
        meso * boost,
        fine * boost,
        sand * boost,
        micro * boost,
        pore_depth * boost,
    )
    s.macro = b.fn(0.9, oct=4)
    s.meso = b.fn(0.14, oct=6, rough=0.55)
    s.fine = b.fn(0.022, oct=5, rough=0.6)
    s.sand = b.fn(0.0032, oct=3, rough=0.6, typ="sparse")
    s.micro = b.fn(0.0011, oct=3, rough=0.65)
    s.h = b.x(
        "(a-0.5)*MA+(b-0.5)*ME+(c-0.5)*FI+(d-0.5)*SA+(e-0.5)*MI",
        a=s.macro,
        b=s.meso,
        c=s.fine,
        d=s.sand,
        e=s.micro,
        MA=macro * 2,
        ME=meso * 2,
        FI=fine * 2,
        SA=b.ctl("sand_grain", sand * 2, 0, 0.5),
        MI=b.ctl("micro_grain", micro * 2, 0, 0.3),
    )
    # tonal grain of the paste + sparse aggregate specks
    s.tone.append((b.x("(a-0.5)*2.0", a=s.sand), 0.05))
    s.tone.append((b.x("(a-0.5)*2.0", a=s.fine), 0.05))
    s.tone.append((b.x("(a-0.5)*2.0", a=s.micro), 0.035))
    sp = b.pits(520, 0.22, 0.12, 0.45, rpow=2.0)
    sp2 = b.pits(380, 0.12, 0.12, 0.5, rpow=2.0)
    s.tone.append((b.x("smoothstep(0.1,0.6,a)-smoothstep(0.1,0.6,b)", a=sp, b=sp2), speck))
    # paste pores: pin pricks of a millimetre or so, everywhere but denser in drifts
    pm = b.fn(0.07, oct=4)
    po = b.pits(760, 0.55, 0.08, 0.36, rpow=2.5, irr=0.3)
    pore = b.x(
        "smoothstep(0.0,0.5,p)*smoothstep(T-0.12,T+0.12,m)",
        p=po,
        m=pm,
        T=b.x("0.78-D*0.6", D=b.ctl("pore_amount", pores, 0, 1)),
    )
    s.addh(pore, -pore_depth)
    s.col.append((pore, tuple(c * 0.5 for c in s.base_col), 0.45))
    s.pore = pore
    # roughness: broad sheen drift, cloudy mid scale, then grain
    g1 = b.fn(0.24, oct=5, rough=0.6)
    g2 = b.gr("moisture", 0.035)
    g3 = b.fn(0.008, oct=4, rough=0.6)
    s.r.append(
        (
            b.x("(a-0.5)*2.0+(b-0.5)*1.3+(c-0.5)*0.8", a=g1, b=g2, c=g3),
            b.ctl("roughness_variation", rvar, 0, 0.5),
        )
    )
    s.r.append((b.x("(a-0.5)*2.0", a=s.fine), 0.05))
    s.r.append((b.x("(a-0.5)*2.0", a=s.sand), 0.04))
    s.r.append((b.x("(a-0.5)*2.0", a=s.micro), 0.03))
    s.r.append((pore, 0.12))
    s.sp.append((pore, -0.4))
    s.sp.append(
        (b.x("(a-0.5)*2.0+(b-0.5)*1.3", a=g1, b=g2), b.ctl("specular_variation", 0.12, 0, 0.5))
    )
    s.sp.append((b.x("(a-0.5)*2.0", a=s.sand), -0.06))


def mottle(s, light, dark, amt=0.4, size=0.07, big=0.2, fine=1.0):
    """cloudy laitance / moisture mottling. weighted towards small scales so the tile does not read"""
    b = s.b
    A = b.ctl("mottle_amount", amt, 0, 1)
    big = min(big, 0.2)
    m1 = b.gr("moisture", size)
    m2 = b.fn(big, oct=6, rough=0.6)
    m3 = b.gr("layerednoise", big * 0.6)
    m4 = b.gr("moisture", size * 0.28)
    m5 = b.fn(size * 0.12, oct=4, rough=0.6)
    s.cloud = b.x(
        "sat((a-0.5)*1.0+(b-0.5)*0.4+(c-0.3)*0.25+((d-0.5)*0.8+(e-0.5)*0.5)*F+0.5)",
        a=m1,
        b=m2,
        c=m3,
        d=m4,
        e=m5,
        F=fine,
    )
    s.col.append((b.x("smoothstep(0.5,0.95,a)", a=s.cloud), light, A))
    s.col.append((b.x("smoothstep(0.5,0.05,a)", a=s.cloud), dark, A))
    s.r.append((b.x("a-0.5", a=s.cloud), -0.08))


def pinholes(s, dens=0.25, cells=440, depth=0.9, rmax=0.3, cluster=0.3, dirt=0.5, name="pinhole"):
    """fine entrapped air: 1-4 mm pin holes in loose drifts, a few slightly larger"""
    b = s.b
    D = b.ctl(name + "_amount", dens, 0, 1)
    m = b.fn(cluster, oct=4)
    m2 = b.fn(0.04, oct=3)
    p1 = b.pits(cells, 0.6, 0.06, rmax, rpow=3.0, irr=0.25)
    p2 = b.pits(int(cells * 0.42), 0.22, 0.05, rmax * 0.85, rpow=4.0, irr=0.3)
    hole = b.x(
        "max(smoothstep(0.0,0.5,p1),smoothstep(0.0,0.45,p2))*smoothstep(T-0.1,T+0.1,m*0.65+m2*0.35)",
        p1=p1,
        p2=p2,
        m=m,
        m2=m2,
        T=b.x("0.74-D*0.55", D=D),
    )
    s.hole = hole
    s.addh(hole, -depth * 1.3)
    s.col.append((hole, tuple(c * 0.35 for c in s.base_col), dirt))
    s.r.append((hole, 0.12))
    s.sp.append((hole, -0.4))
    return hole


def joint_wear(
    s,
    dist,
    width_mm=25.0,
    void=0.5,
    chip=0.4,
    depth=1.5,
    gain=1.3,
    stain=None,
    stain_amt=0.5,
    name="joint",
):
    """damage that lives at formwork joints: grout-loss voids, ragged chipped arrises, staining"""
    b = s.b
    A = b.ctl(name + "_wear", 1.0, 0, 2)
    band = b.x("exp(-a/W)", a=dist, W=b.mm(width_mm))
    n = b.fn(0.05, oct=5, rough=0.6)
    n2 = b.fn(0.012, oct=3, typ="alligator")
    p = b.pits(540, 0.75, 0.06, 0.38, rpow=2.5, irr=0.35)
    voids = b.x(
        "smoothstep(0.0,0.5,p)*sat(band*1.7)*smoothstep(T-0.1,T+0.1,n)*A",
        p=p,
        band=band,
        n=n,
        T=0.72 - void * 0.4,
        A=A,
    )
    chips = b.x(
        "smoothstep(0.42,0.6,band*(0.35+1.3*n))*smoothstep(0.48,0.6,n)*(0.4+0.6*n2)*A*C",
        band=band,
        n=n,
        n2=n2,
        A=A,
        C=chip,
    )
    s.addh(voids, -depth * gain)
    s.addh(chips, -depth * 0.6 * gain)
    s.col.append((voids, tuple(c * 0.35 for c in s.base_col), 0.6))
    s.col.append((chips, tuple(c * 0.72 for c in s.base_col), 0.5))
    s.r.append((b.x("max(a,b)", a=voids, b=chips), 0.12))
    if stain:
        s.col.append(
            (
                b.x("band*band*(0.25+0.75*n)", band=band, n=n),
                stain,
                b.ctl(name + "_stain", stain_amt, 0, 1),
            )
        )
    s.jband = band
    return voids


def scaling(s, cover=0.3, depth=0.6, size=0.16, colour=None, name="scaling", rough=0.14):
    """shallow delamination: the thin top skin lost in crisp-edged patches, sandier underneath"""
    b = s.b
    K = b.ctl(name + "_depth", depth, 0, 4)
    n = b.warp(b.fn(size, oct=6, rough=0.62), size * 0.6, size * 0.3)
    f = b.fn(size * 0.06, oct=4)
    T = 0.5 + (0.5 - cover) * 0.45
    m = b.x("smoothstep(T,T+0.012,n+(f-0.5)*0.05)", n=n, f=f, T=T)
    agg = b.fn(0.006, oct=3, typ="alligator")
    s.h = b.x("h-m*(0.8+(a-0.5)*0.9)*K", h=s.h, m=m, a=agg, K=K)
    s.col.append((m, colour or tuple(c * 0.8 for c in s.base_col), 0.55))
    s.r.append((m, rough))
    s.sp.append((m, -0.25))
    s.scale = m
    return m


def crazing(s, size=0.08, width_mm=0.3, cover=0.5, dark=0.35, depth=0.15, name="crazing"):
    """map cracking: a fine net of hairlines in the surface skin"""
    b = s.b
    A = b.ctl(name + "_amount", 1.0, 0, 1)
    m = b.x("a*A", a=b.cracks(size, b.mm(width_mm), cover=cover, wamt=0.02, msize=0.25), A=A)
    s.addh(m, -depth)
    s.col.append((m, tuple(c * 0.4 for c in s.base_col), dark))
    return m


def polish(s, amt=0.25, size=0.3, tint=0.0, name="polish"):
    """burnished / trafficked lanes: lower roughness in soft lanes, broken up at two finer scales"""
    b = s.b
    a = b.warp(b.fn(size, oct=5, rough=0.55), 0.1, 0.3)
    q = b.gr("moisture", 0.04)
    c = b.fn(0.008, oct=4)
    m = b.x("smoothstep(0.4,0.62,a)*(0.4+0.6*q)*(0.7+0.6*(c-0.5))", a=a, q=q, c=c)
    s.r.append((b.x("0.0-a", a=m), b.ctl(name + "_amount", amt, 0, 0.6)))
    s.sp.append((m, 0.3))
    if tint:
        s.tone.append((b.x("0.0-a", a=m), tint))
    s.polish = m
    return m


def dust(s, colour, amt=0.35, rough=0.12, size=0.05, name="dust"):
    """settled dust: pale, matte, gathers in fine cloudy drifts and in the pores"""
    b = s.b
    m = b.x(
        "sat(smoothstep(0.45,0.85,a)*(0.5+0.5*c)+p*0.5)",
        a=b.gr("moisture", size),
        c=b.fn(0.2, oct=4),
        p=s.pore,
    )
    s.col.append((m, colour, b.ctl(name + "_amount", amt, 0, 1)))
    s.r.append((m, rough))
    s.sp.append((m, -0.4))
    return m


def scuffs(s, colour, amt=0.3, size=0.16, rough=0.08, name="scuff"):
    """thin drag marks and scratches"""
    b = s.b
    t = b.gr("wipetrails", size, density=0.7, wipewidth=0.2)
    f = b.fn(0.003, oct=3)
    m = b.x("smoothstep(0.3,0.9,a)*(0.4+1.2*(f-0.3))", a=t, f=f)
    s.col.append((m, colour, b.ctl(name + "_amount", amt, 0, 1)))
    s.r.append((m, rough))
    s.addh(m, -0.04)
    return m


def spots(s, colour, amt=0.6, cells=14, dens=0.35, rmax=0.3, rough=-0.15, name="spot"):
    """soft-edged drips and spills"""
    b = s.b
    p = b.warp(b.pits(cells, dens, 0.04, rmax, rpow=2.5, irr=0.5), 0.03, 0.06)
    m = b.x("smoothstep(0.0,0.5,a)*(0.55+0.45*n)", a=p, n=b.gr("moisture", 0.03))
    s.col.append((m, colour, b.ctl(name + "_amount", amt, 0, 1)))
    s.r.append((m, rough))
    return m


def slats(s, n=24, tone=0.035, step=0.12, axis=0, seed=5, line=0.15):
    """faint board-width banding telegraphed through sheet formwork"""
    b = s.b
    edge = b.boards(n, 0.35, seed, axis, 1, 0)
    bid = b.boards(n, 0.35, seed, axis, 1, 1)
    st = (1.0, 40.0) if axis == 0 else (40.0, 1.0)
    f = b.fn(0.008, oct=4, rough=0.6, sx=st[0], sy=st[1])
    K = b.ctl("slat_depth", step, 0, 2)
    s.h = b.x("h+((a-0.5)+(f-0.5)*0.6)*K", h=s.h, a=bid, f=f, K=K)
    s.tone.append((b.x("(a-0.5)*2.0", a=bid), b.ctl("slat_tone", tone, 0, 0.3)))
    s.tone.append((b.x("(a-0.5)*2.0", a=f), tone * 0.9))
    s.col.append(
        (
            b.x("1.0-smoothstep(0.0,W,a)", a=edge, W=b.mm(1.5)),
            tuple(c * 0.55 for c in s.base_col),
            line,
        )
    )
    s.r.append((b.x("(a-0.5)*2.0", a=bid), 0.05))
    return bid


def striations(s, axis=1, depth=0.35, tone=0.1, dark=None, dark_amt=0.5):
    """fine linear drag texture left by the form face, with dirt caught along the lines"""
    b = s.b
    st = (1.0, 45.0) if axis == 0 else (45.0, 1.0)
    a = b.fn(0.0035, oct=4, rough=0.65, sx=st[0], sy=st[1])
    c = b.fn(0.011, oct=4, rough=0.6, sx=st[0], sy=st[1] * 0.6)
    d = b.fn(0.03, oct=4, rough=0.55, sx=st[0], sy=st[1] * 0.35)
    K = b.ctl("striation_depth", depth, 0, 2)
    s.h = b.x("h+((a-0.5)*0.9+(c-0.5)*0.7+(d-0.5)*0.4)*K", h=s.h, a=a, c=c, d=d, K=K)
    s.tone.append((b.x("(a-0.5)*1.2+(c-0.5)*1.2", a=a, c=c), b.ctl("striation_tone", tone, 0, 0.4)))
    s.r.append((b.x("(a-0.5)+(c-0.5)", a=a, c=c), 0.08))
    if dark:
        g = b.fn(0.18, oct=4)
        m = b.x("smoothstep(0.55,0.8,c*0.45+d*0.55)*smoothstep(0.3,0.6,g)", c=c, d=d, g=g)
        s.col.append((m, dark, b.ctl("striation_dirt", dark_amt, 0, 1)))
    return a


def bugholes(s, cells=90, dens=0.3, depth=2.5, rmax=0.36, dirt=0.6, mask_size=0.4, name="bughole"):
    """entrapped air voids (blowholes)"""
    b = s.b
    D = b.ctl(name + "_amount", dens, 0, 1)
    m = b.fn(mask_size, oct=3)
    p1 = b.pits(cells, D, 0.05, rmax, rpow=3.5, irr=0.2)
    p2 = b.pits(cells * 3, D, 0.06, 0.3, rpow=3.0, irr=0.15)
    # holes cluster where the low frequency mask is high
    hole = b.x(
        "max(smoothstep(0.0,0.45,p1),0.35*smoothstep(0.0,0.5,p2))*smoothstep(0.3,0.45,m+(D-0.3))",
        p1=p1,
        p2=p2,
        m=m,
        D=D,
    )
    s.hole = hole
    s.addh(hole, -depth)
    s.col.append((hole, tuple(c * 0.35 for c in s.base_col), dirt))
    s.r.append((hole, 0.12))


def panels(
    s,
    nx,
    ny,
    stag=0.0,
    seam_mm=2.5,
    seam_d=-0.7,
    step=0.35,
    tone=0.06,
    ties=None,
    tie_depth=5.0,
    seam_dirt=0.5,
):
    """sheet formwork: seams, per-panel offset/tone and tie holes (cols, rows, mx, my, radius_mm)"""
    b = s.b
    d = b.warp(b.panels(nx, ny, stag, 0), b.mm(1.2), size=0.015)
    s.pid = b.panels(nx, ny, stag, 1)
    line = b.x("1.0-smoothstep(0.0,W,a)", a=d, W=b.mm(seam_mm))
    K = b.ctl("seam_depth", 1.0, 0, 3)
    trough = b.x("1.0-smoothstep(0.0,W,a)", a=d, W=b.mm(seam_mm * 4.0))
    s.h = b.x(
        "h+(line*D+tr*tr*D*0.45+(pid-0.5)*ST)*K",
        h=s.h,
        line=line,
        tr=trough,
        pid=s.pid,
        D=seam_d,
        ST=step,
        K=K,
    )
    s.tone.append((b.x("(a-0.5)*2.0", a=s.pid), tone))
    s.seam = b.x("1.0-smoothstep(0.0,W,a)", a=d, W=b.mm(seam_mm * 5))
    s.col.append((b.x("a*a", a=s.seam), tuple(c * 0.4 for c in s.base_col), seam_dirt))
    s.seamdist = d
    if ties:
        cols, rows, mx, my, r = ties
        cone = b.ties(nx, ny, cols, rows, mx, my, b.mm(r), stag)
        hole = b.x("smoothstep(0.0,0.35,a)-0.15*smoothstep(0.75,1.0,a)", a=cone)
        s.h = b.x("h-m*K", h=s.h, m=hole, K=b.ctl("tie_hole_depth", tie_depth, 0, 20))
        halo = b.ties(nx, ny, cols, rows, mx, my, b.mm(r * 2.6), stag)
        # grout squeezed out round the cone leaves a slightly proud, rough collar
        lip = b.x(
            "smoothstep(0.0,0.3,h)*(1.0-smoothstep(0.0,0.08,c))*(0.6+0.8*n)",
            h=halo,
            c=cone,
            n=b.fn(0.01, oct=3),
        )
        s.addh(lip, b.ctl("tie_collar", tie_depth * 0.08, 0, 2))
        s.col.append(
            (
                b.x("max(smoothstep(0.0,0.35,c),0.5*h*h)", c=cone, h=halo),
                tuple(c * 0.45 for c in s.base_col),
                0.6,
            )
        )
        s.tie = hole


def boards(
    s,
    n,
    axis=0,
    jit=0.5,
    segs=2,
    seed=1,
    step=1.2,
    grain=0.35,
    fibre=0.12,
    fin=0.8,
    gap_mm=2.0,
    rings=4.0,
    tone=0.08,
):
    """sawn timber board formwork imprint"""
    b = s.b
    edge = b.boards(n, jit, seed, axis, segs, 0)
    bid = b.boards(n, jit, seed, axis, segs, 1)
    acr = b.boards(n, jit, seed, axis, segs, 2)
    endj = b.boards(n, jit, seed, axis, segs, 3)
    sid = b.boards(n, jit, seed, axis, segs, 4)
    st = (1.0, 14.0) if axis == 0 else (14.0, 1.0)
    g = b.fn(0.05, oct=4, sx=st[0], sy=st[1])
    fib = b.fn(0.0035, oct=4, rough=0.65, sx=st[0], sy=st[1] * 3)
    fib2 = b.fn(0.012, oct=4, rough=0.6, sx=st[0], sy=st[1] * 2)
    sh = (0.9, 1.7) if axis == 0 else (1.7, 0.9)
    ring = b.cl(
        f"""    float id=@sid; float2 o=(float2)({sh[0]:f}*id,{sh[1]:f}*fmod(id*7.31,1.0));
    float gg=@g.imageSample(@P+o); float f1=@fib.imageSample(@P+o); float f2=@fib2.imageSample(@P+o);
    float r=sin((@ac*({rings:f}+3.0*id)+gg*9.0)*6.2832);
    r=pow(0.5+0.5*r,0.7);
    @dst.set((r-0.5)*@G+(f1-0.5)*@F+(f2-0.5)*@F*1.5);""",
        g=g,
        fib=fib,
        fib2=fib2,
        ac=acr,
        sid=sid,
        G=b.ctl("wood_grain", grain, 0, 2),
        F=b.ctl("wood_fibre", fibre, 0, 1),
    )
    s.grain = ring
    gapw = b.mm(gap_mm)
    joint = b.x("max(1.0-smoothstep(0.0,W,e),1.0-smoothstep(0.0,W,j))", e=edge, j=endj, W=gapw)
    finn = b.fn(0.02, oct=3, sx=st[0] * 0.3, sy=st[1] * 0.3)
    K = b.ctl("board_depth", 1.0, 0, 3)
    s.h = b.x(
        "h+ring+((bid-0.5)*ST+(sid-0.5)*ST*0.5+(ac-0.5)*(ac-0.5)*1.2+joint*FIN*smoothstep(0.25,0.6,fn))*K",
        h=s.h,
        ring=ring,
        bid=bid,
        sid=sid,
        ac=acr,
        joint=joint,
        fn=finn,
        ST=step,
        FIN=fin,
        K=K,
    )
    s.tone.append((b.x("(a-0.5)*2.0", a=sid), tone))
    s.tone.append((b.x("clamp(a*3.0,-1.0,1.0)", a=ring), 0.07))
    s.joint = b.x(
        "max(1.0-smoothstep(0.0,W,e),1.0-smoothstep(0.0,W,j))", e=edge, j=endj, W=gapw * 3
    )
    s.col.append(
        (s.joint, tuple(c * 0.38 for c in s.base_col), b.ctl("board_joint_dirt", 0.7, 0, 1))
    )
    s.bid = sid
    s.bdist = b.x("min(e,j)", e=edge, j=endj)
    s.r.append((b.x("clamp(a*2.0,-1.0,1.0)", a=ring), 0.06))


def streaks(s, colour, amt=0.5, kind="fine", size=0.3, name="streak", rough=0.0, gamma=1.0):
    """vertical run-off: 'fine' rain streaks, 'leak' wide leaks, 'drips' short drips"""
    b = s.b
    A = b.ctl(name + "_amount", amt, 0, 1)
    if kind == "fine":
        a = b.fn(size * 0.06, oct=5, rough=0.6, sx=1.0, sy=40.0)
        c = b.fn(0.22, oct=4)
        m = b.x("smoothstep(0.5,0.85,a)*smoothstep(0.3,0.7,c)", a=a, c=c)
    elif kind == "leak":
        # leak shapes softened down the wall and broken up by fine run-off lines
        g = b.sblur(b.gr("leakingstreaks", size), 90.0, 0.05)
        f = b.fn(0.012, oct=5, rough=0.6, sx=1.0, sy=50.0)
        m = b.x("pow(sat((g-0.06)/0.6),GA)*(0.55+0.9*(f-0.25))", g=g, f=f, GA=gamma)
    elif kind == "broad":
        # wide soft mineral runs: tall warped islands with feathered lower ends
        n = b.warp(b.fn(size * 0.35, oct=5, rough=0.55, sx=1.0, sy=10.0), 0.03, 0.2)
        f = b.fn(0.01, oct=5, rough=0.6, sx=1.0, sy=60.0)
        d = b.gr("moisture", 0.05)
        m = b.x("pow(smoothstep(0.52,0.78,n+(f-0.5)*0.12),GA)*(0.5+0.5*d)", n=n, f=f, d=d, GA=gamma)
    else:
        g = b.gr("drips", size)
        m = b.lv(b.sblur(g, 90.0, 0.01), 0.02, 0.6, gamma)
    s.col.append((m, colour, A))
    if rough:
        s.r.append((m, rough))
    return m


def stains(
    s,
    colour,
    amt=0.5,
    kind="moisture",
    size=0.1,
    lo=0.45,
    hi=0.8,
    name="stain",
    rough=0.0,
    rim=0.22,
    **kw,
):
    b = s.b
    A = b.ctl(name + "_amount", amt, 0, 1)
    if kind == "blotch":
        size = min(size * 0.5, 0.3)
        # built-up stain: warped fractal island, moisture breakup inside, tide mark at the rim. lo = coverage
        n = b.warp(b.fn(size, oct=6, rough=0.62), size * 0.9, size * 0.35)
        d = b.gr("moisture", size * 0.2)
        f = b.fn(size * 0.05, oct=4)
        T = 0.5 + (0.5 - lo) * 0.45
        m = b.x(
            "sat(smoothstep(T,T+0.14,n+(f-0.5)*0.06)*(0.35+0.65*d)+RIM*d*exp(-pow((n+(f-0.5)*0.06-T)/0.015,2.0)))",
            n=n,
            d=d,
            f=f,
            T=T,
            RIM=rim,
        )
    else:
        g = b.gr(kind, size, **kw)
        m = b.lv(g, lo, hi)
    s.col.append((m, colour, A))
    if rough:
        s.r.append((m, rough))
    return m


def cracks(s, size=0.7, width_mm=0.8, depth=1.2, cover=0.45, dark=0.7, name="crack", shoulder=0.25):
    b = s.b
    A = b.ctl(name + "_amount", 1.0, 0, 1)
    m = b.cracks(size, b.mm(width_mm), cover=cover)
    m = b.x("a*A", a=m, A=A)
    s.addh(m, -depth)
    if shoulder:
        # the arris either side of a crack crumbles: a shallow ragged trough around the sharp line
        sh = b.x(
            "sat(a*6.0)*(0.5+n)",
            a=b.blur(m, b.mm(width_mm * 5.0)),
            n=b.fn(0.01, oct=3, typ="alligator"),
        )
        s.addh(sh, -depth * shoulder)
        s.r.append((sh, 0.1))
        s.sp.append((sh, -0.2))
    s.col.append((b.x("sat(a*1.5)", a=m), tuple(c * 0.25 for c in s.base_col), dark))
    s.crack = m
    return m


# ------------------------------------------------------------------ recipes
def m01(par, res):
    """499 - weathered cast in place flyover"""
    b = L.B(par, "Concrete_01_Overpass_Weathered", 4.0, res, m01.__doc__)
    s = S(b, srgb(150, 148, 143), 0.72, 0.55)
    base(
        s,
        macro=0.8,
        meso=0.25,
        fine=0.09,
        sand=0.05,
        micro=0.035,
        pores=0.4,
        pore_depth=0.45,
        rvar=0.16,
    )
    mottle(s, srgb(174, 172, 166), srgb(120, 118, 114), 0.42)
    panels(s, 3, 2, seam_mm=3, seam_d=0.6, step=0.6, tone=0.05)
    # horizontal pour lines between lifts
    pl = b.warp(b.panels(1, 5, 0.0, 3), 0.01, 0.3)
    pline = b.x(
        "(1.0-smoothstep(0.0,W,a))*smoothstep(0.3,0.6,n)",
        a=pl,
        n=b.fn(0.3, oct=4, sx=4.0, sy=1.0),
        W=b.mm(5.0),
    )
    s.addh(pline, -0.8)
    s.col.append((pline, srgb(82, 80, 77), b.ctl("pour_line", 0.7, 0, 1)))
    s.col.append((b.x("exp(-a/W)", a=pl, W=b.mm(60.0)), srgb(120, 118, 113), 0.3))
    joint_wear(s, pl, 22.0, void=0.55, chip=0.45, depth=1.6, name="pour_joint")
    pinholes(s, 0.22, 420, 1.0)
    streaks(s, srgb(84, 83, 80), 0.6, "fine", 0.3, "rain_streak", 0.05)
    streaks(s, srgb(188, 186, 180), 0.35, "leak", 0.35, "lime_streak", -0.03)
    stains(s, srgb(92, 90, 86), 0.35, "blotch", 0.22, 0.3, 0.0, "soot", rim=0.1)
    crazing(s, 0.1, 0.3, 0.4, 0.3)
    cracks(s, 0.9, 0.6, 0.6, 0.3)
    return s.done()


def m02(par, res):
    """Blade Runner 2049 - pale burnished floor"""
    b = L.B(par, "Concrete_02_Burnished_Pale", 3.0, res, m02.__doc__)
    s = S(b, srgb(178, 182, 174), 0.4, 0.95)
    base(
        s,
        macro=0.25,
        meso=0.05,
        fine=0.02,
        sand=0.012,
        speck=0.09,
        micro=0.012,
        pores=0.25,
        pore_depth=0.18,
        rvar=0.2,
    )
    mottle(s, srgb(194, 197, 190), srgb(160, 165, 158), 0.4, 0.09, 0.2)
    tr = b.blur(b.gr("wipetrails", 0.35, density=1.0, wipewidth=0.8), 0.003)
    tr2 = b.gr("wipetrails", 0.12, density=1.0, wipewidth=0.5)
    s.tone.append((b.x("a-0.15", a=tr), b.ctl("trowel_burnish", 0.07, 0, 0.5)))
    s.r.append((b.x("a+b*0.6", a=tr, b=tr2), -0.14))
    s.addh(b.x("a+b*0.5", a=tr, b=tr2), b.ctl("trowel_relief", 0.4, 0, 1))
    chatter = b.fn(0.004, oct=3, sx=30.0, sy=1.0)
    s.addh(b.x("(a-0.5)*(0.3+t)", a=chatter, t=tr), 0.3)
    polish(s, 0.18, 0.3)
    scuffs(s, srgb(206, 208, 202), 0.25, 0.14, 0.1)
    pinholes(s, 0.12, 620, 0.35, rmax=0.26, dirt=0.4)
    crazing(s, 0.07, 0.2, 0.45, 0.25, 0.05)
    cracks(s, 1.0, 0.25, 0.15, 0.25, 0.35, "hairline")
    return s.done(0.008)


def m03(par, res):
    """Eddy De Pretto - shaft panels: rusting seams, wet and dry faces, efflorescence runs"""
    b = L.B(par, "Concrete_03_Shaft_Efflorescence", 4.0, res, m03.__doc__)
    s = S(b, srgb(122, 128, 126), 0.62, 0.75)
    base(s, macro=0.5, meso=0.14, fine=0.06, micro=0.03, pores=0.3, rvar=0.16)
    mottle(s, srgb(140, 146, 144), srgb(100, 106, 106), 0.38)
    panels(
        s,
        2,
        2,
        seam_mm=3,
        seam_d=-0.9,
        step=0.5,
        tone=0.05,
        seam_dirt=0.6,
        ties=(2, 2, 0.12, 0.12, 9.0),
        tie_depth=3.0,
    )
    # each panel took the form oil differently: some faces are dense and glossy, others chalky
    s.r.append((b.x("(a-0.5)*2.0", a=s.pid), b.ctl("panel_gloss_variation", 0.16, 0, 0.4)))
    wet = b.warp(b.fn(0.2, oct=5, rough=0.55, sx=1.0, sy=5.0), 0.04, 0.2)
    wetm = b.x("smoothstep(0.48,0.66,a)*(0.5+0.5*d)", a=wet, d=b.gr("moisture", 0.04))
    s.r.append((b.x("0.0-a", a=wetm), b.ctl("wet_sheen", 0.3, 0, 0.6)))
    s.sp.append((wetm, 0.3))
    s.col.append((wetm, srgb(92, 98, 98), 0.3))
    # rust bleeding out of the seams and down from the horizontal ones
    joint_wear(
        s,
        s.seamdist,
        75.0,
        void=0.3,
        chip=0.3,
        depth=1.0,
        stain=srgb(104, 80, 60),
        stain_amt=0.42,
        name="seam",
    )
    run = b.sblur(
        b.x("a*a*smoothstep(0.4,0.7,n)", a=s.jband, n=b.fn(0.03, oct=4, sx=1.0, sy=6.0)), 90.0, 0.07
    )
    s.col.append((b.lv(run, 0.02, 0.3), srgb(108, 86, 66), b.ctl("rust_run", 0.4, 0, 1)))
    pinholes(s, 0.15, 460, 0.8)
    streaks(s, srgb(206, 209, 206), 0.6, "broad", 0.7, "efflorescence", 0.1, 1.5)
    streaks(s, srgb(186, 190, 188), 0.3, "leak", 0.5, "lime_run", 0.06)
    streaks(s, srgb(70, 76, 77), 0.5, "fine", 0.5, "wet_streak", -0.18)
    return s.done()


def m04(par, res):
    """Gattaca - warm smooth architectural concrete with tie holes"""
    b = L.B(par, "Concrete_04_Architectural_Warm", 3.6, res, m04.__doc__)
    s = S(b, srgb(170, 158, 140), 0.78, 0.45)  # chalky: rough, weak specular
    base(
        s,
        macro=0.4,
        meso=0.18,
        fine=0.1,
        sand=0.07,
        micro=0.065,
        pores=0.45,
        pore_depth=0.45,
        rvar=0.08,
        boost=1.0,
    )
    mottle(s, srgb(184, 173, 156), srgb(150, 138, 122), 0.34, 0.07, 0.2)
    panels(
        s,
        3,
        2,
        seam_mm=2.5,
        seam_d=-1.6,
        step=0.6,
        tone=0.04,
        seam_dirt=0.4,
        ties=(2, 3, 0.16, 0.14, 11.0),
        tie_depth=6.5,
    )
    joint_wear(s, s.seamdist, 14.0, void=0.35, chip=0.4, depth=1.3, name="seam")
    pinholes(s, 0.22, 500, 1.1, rmax=0.28)
    crazing(s, 0.09, 0.25, 0.35, 0.2, 0.05)
    streaks(s, srgb(140, 128, 112), 0.28, "fine", 0.4, "rain_streak", 0.04)
    return s.done(0.012)


def m05(par, res):
    """Inception - worn factory floor: polished traffic lanes, dust, shallow scaling and cracks"""
    b = L.B(par, "Concrete_05_Factory_Floor_Worn", 4.0, res, m05.__doc__)
    s = S(b, srgb(136, 135, 130), 0.62, 0.8)  # trafficked: glossy lanes against dusty matte
    base(
        s,
        macro=1.0,
        meso=0.36,
        fine=0.15,
        sand=0.085,
        speck=0.1,
        micro=0.065,
        pores=0.45,
        pore_depth=0.55,
        rvar=0.12,
        boost=1.0,
    )
    mottle(s, srgb(156, 155, 150), srgb(108, 107, 103), 0.3, 0.05, 0.12)
    polish(s, 0.46, 0.28, tint=0.1)
    polish(s, 0.18, 0.07, name="fine_polish")
    polish(s, 0.1, 0.02, name="micro_polish")
    dust(s, srgb(176, 174, 166), 0.4, 0.2, 0.05)
    scaling(s, 0.2, 1.3, 0.14, srgb(118, 114, 106))
    pid = b.panels(3, 4, 0.5, 1)
    pd = b.warp(b.panels(3, 4, 0.5, 0), 0.01, 0.2)
    patch = b.x("step(0.86,p)*smoothstep(0.0,0.004,d)", p=pid, d=pd)
    s.col.append((patch, srgb(162, 160, 152), b.ctl("patch_amount", 0.3, 0, 1)))
    s.addh(patch, 0.5)
    stains(s, srgb(78, 75, 70), 0.3, "blotch", 0.36, 0.3, 0.0, "oil_stain", -0.2, rim=0.05)
    spots(s, srgb(60, 58, 55), 0.45, 18, 0.25, 0.25, -0.18, "oil_spot")
    scuffs(s, srgb(190, 189, 184), 0.3, 0.2, 0.1)
    scuffs(s, srgb(70, 69, 66), 0.25, 0.3, -0.05, "drag_mark")
    pinholes(s, 0.3, 380, 1.5, name="pit")
    cracks(s, 0.8, 1.8, 3.5, 0.5, shoulder=0.3)
    cracks(s, 0.3, 0.7, 1.4, 0.35, 0.6, "fine_crack", shoulder=0.25)
    return s.done()


def m06(par, res):
    """Inception - light grey dusty concrete, old limewash worn to a film"""
    b = L.B(par, "Concrete_06_Dusty_Light_Grey", 3.0, res, m06.__doc__)
    s = S(b, srgb(158, 156, 148), 0.82, 0.4)
    base(
        s,
        macro=0.9,
        meso=0.45,
        fine=0.18,
        sand=0.1,
        micro=0.045,
        pores=0.5,
        pore_depth=0.5,
        rvar=0.1,
    )
    mottle(s, srgb(178, 176, 168), srgb(130, 128, 121), 0.34, 0.06, 0.15)
    # limewash survives as a thin film on the high spots
    f1 = b.gr("rustysurface", 0.12)
    f2 = b.warp(b.fn(0.07, oct=6, rough=0.6), 0.04, 0.1)
    brush = b.fn(0.006, oct=4, sx=1.0, sy=22.0)
    wash = b.x(
        "smoothstep(0.5,0.62,a*0.55+b*0.6+(c-0.5)*0.3)*(1.0-p)", a=f1, b=f2, c=s.fine, p=s.pore
    )
    s.col.append((wash, srgb(198, 196, 188), b.ctl("limewash_amount", 0.5, 0, 1)))
    s.h = b.x("h+w*(0.12+(br-0.5)*0.1)", h=s.h, w=wash, br=brush)
    s.r.append((wash, -0.1))
    dust(s, srgb(190, 187, 176), 0.3, 0.08, 0.04)
    scaling(s, 0.2, 0.5, 0.1, srgb(134, 131, 122))
    pinholes(s, 0.32, 400, 1.1)
    crazing(s, 0.06, 0.3, 0.5, 0.3)
    streaks(s, srgb(112, 104, 90), 0.25, "drips", 0.06, "grime_drip", 0.05)
    streaks(s, srgb(128, 122, 110), 0.25, "fine", 0.4, "dirt_streak")
    cracks(s, 0.5, 0.5, 0.8, 0.35)
    return s.done()


def m07(par, res):
    """JD.com - hard trowelled skate bowl"""
    b = L.B(par, "Concrete_07_Skate_Bowl_Trowelled", 4.0, res, m07.__doc__)
    s = S(b, srgb(142, 148, 144), 0.5, 0.9)
    base(
        s,
        macro=1.2,
        meso=0.2,
        fine=0.04,
        sand=0.022,
        speck=0.07,
        micro=0.02,
        pores=0.3,
        pore_depth=0.25,
        rvar=0.18,
    )
    mottle(s, srgb(162, 167, 162), srgb(114, 120, 118), 0.36, 0.07, 0.2)
    tr = b.blur(
        b.gr("wipetrails", 0.5, density=1.0, wipewidth=1.0, shapedetail=0.4, mixing=0.6), 0.002
    )
    tr2 = b.gr("wipetrails", 0.25)
    tr3 = b.gr("wipetrails", 0.09, density=1.0, wipewidth=0.4)
    s.addh(b.x("a+b*0.4+c*0.25", a=tr, b=tr2, c=tr3), b.ctl("trowel_marks", 1.1, 0, 2))
    s.tone.append((b.x("a-0.25", a=tr), 0.26))
    s.r.append((b.x("a+c*0.5", a=tr, c=tr3), -0.16))
    polish(s, 0.2, 0.25)
    scuffs(s, srgb(192, 195, 190), 0.4, 0.18, 0.1)
    scuffs(s, srgb(60, 64, 64), 0.3, 0.35, -0.04, "wheel_mark")
    stains(s, srgb(84, 90, 88), 0.28, "blotch", 0.4, 0.4, 0.0, "grime", 0.06, rim=0.08)
    pinholes(s, 0.12, 520, 0.6, rmax=0.26, name="pit")
    crazing(s, 0.08, 0.25, 0.5, 0.3, 0.06)
    cracks(s, 0.45, 0.45, 0.6, 0.6, 0.75)
    return s.done(0.01)


def m08(par, res):
    """K.ZIA Sanctuary - narrow board-formed wall: tie cones, voids strung along the board joints"""
    b = L.B(par, "Concrete_08_BoardFormed_Narrow", 3.0, res, m08.__doc__)
    s = S(b, srgb(160, 153, 143), 0.72, 0.55)
    base(s, macro=0.5, meso=0.15, fine=0.06, micro=0.03, pores=0.35, rvar=0.12)
    mottle(s, srgb(176, 168, 157), srgb(134, 128, 120), 0.3)
    boards(s, 26, 0, 0.55, 2, 3, step=1.5, grain=0.2, fibre=0.12, fin=0.9, gap_mm=2.0, tone=0.07)
    joint_wear(s, s.bdist, 9.0, void=0.75, chip=0.45, depth=1.8, name="board_joint")
    # form tie cones on a regular grid, each with a set-back plug
    cone = b.ties(5, 4, 1, 1, 0.5, 0.5, b.mm(15.0), 0.5)
    dish = b.x("smoothstep(0.0,0.12,a)*(0.75+0.25*smoothstep(0.55,0.7,a))", a=cone)
    s.addh(dish, b.x("0.0-K", K=b.ctl("tie_cone_depth", 5.0, 0, 20)))
    s.col.append((b.x("smoothstep(0.0,0.2,a)", a=cone), tuple(c * 0.62 for c in s.base_col), 0.5))
    # day joint: a paler laitance band across the boards
    lj = b.warp(b.panels(1, 2, 0.0, 3), 0.006, 0.3)
    s.col.append(
        (b.x("exp(-a/W)", a=lj, W=b.mm(26.0)), srgb(192, 186, 176), b.ctl("lift_band", 0.4, 0, 1))
    )
    s.addh(b.x("exp(-a/W)", a=lj, W=b.mm(12.0)), 0.6)
    pinholes(s, 0.14, 460, 0.9)
    stains(s, srgb(186, 172, 148), 0.3, "moisture", 0.08, 0.5, 0.85, "warm_stain")
    streaks(s, srgb(112, 106, 98), 0.35, "fine", 0.35, "rain_streak")
    return s.done()


def m09(par, res):
    """K.ZIA Sanctuary - rough dark board-formed wall, honeycombed at the joints"""
    b = L.B(par, "Concrete_09_BoardFormed_Rough", 3.0, res, m09.__doc__)
    s = S(b, srgb(108, 105, 100), 0.72, 0.5)
    base(
        s,
        macro=0.8,
        meso=0.4,
        fine=0.16,
        sand=0.08,
        micro=0.04,
        pores=0.5,
        pore_depth=0.5,
        rvar=0.16,
    )
    mottle(s, srgb(130, 127, 121), srgb(80, 78, 75), 0.5)
    boards(s, 20, 0, 0.6, 2, 7, step=2.2, grain=0.5, fibre=0.22, fin=1.6, gap_mm=2.4, tone=0.1)
    joint_wear(s, s.bdist, 14.0, void=0.8, chip=0.7, depth=2.6, name="board_joint")
    pinholes(s, 0.42, 300, 1.7, rmax=0.34, dirt=0.7)
    hc = b.lv(b.gr("rustysurface", 0.3), 0.55, 0.85)
    agg = b.fn(0.012, oct=3, typ="alligator")
    s.addh(b.x("m*(0.3+a)", m=hc, a=agg), b.x("0.0-K", K=b.ctl("honeycomb_depth", 2.4, 0, 10)))
    s.col.append((hc, srgb(70, 67, 62), 0.55))
    streaks(s, srgb(52, 51, 49), 0.55, "fine", 0.3, "dirt_streak", 0.04)
    streaks(s, srgb(150, 148, 142), 0.3, "drips", 0.04, "lime_drip")
    return s.done(0.02)


def m10(par, res):
    """Kinetta - sun-warmed retaining wall with pour lifts"""
    b = L.B(par, "Concrete_10_Pour_Lift_Warm", 4.0, res, m10.__doc__)
    s = S(b, srgb(170, 158, 134), 0.7, 0.5)
    base(s, macro=0.7, meso=0.22, fine=0.09, sand=0.05, micro=0.035, pores=0.4, rvar=0.14)
    mottle(s, srgb(184, 174, 150), srgb(146, 136, 116), 0.36)
    big = b.fn(0.2, oct=6, rough=0.6)
    s.col.append((b.lv(big, 0.5, 0.75), srgb(152, 128, 98), b.ctl("ochre_patch", 0.45, 0, 1)))
    s.col.append((b.lv(big, 0.5, 0.25), srgb(142, 144, 130), b.ctl("grey_patch", 0.45, 0, 1)))
    panels(
        s,
        2,
        2,
        seam_mm=3,
        seam_d=-0.6,
        step=0.8,
        tone=0.05,
        ties=(2, 1, 0.3, 0.5, 13.0),
        tie_depth=4.0,
    )
    # horizontal lift joints: grout loss and exposed aggregate along the cold joint
    dy = b.warp(b.panels(2, 2, 0.0, 3), 0.012, 0.25)
    band = b.x("exp(-a/W)", a=dy, W=b.mm(35.0))
    hc = b.x("band*smoothstep(0.35,0.7,n)", band=band, n=b.gr("rustysurface", 0.25))
    agg = b.fn(0.01, oct=3, typ="alligator")
    s.addh(b.x("m*(0.25+a)", m=hc, a=agg), b.x("0.0-K", K=b.ctl("lift_joint_depth", 4.0, 0, 12)))
    s.col.append((hc, srgb(112, 104, 90), 0.7))
    s.col.append((b.x("exp(-a/W)", a=dy, W=b.mm(90.0)), srgb(190, 182, 164), 0.3))
    joint_wear(s, dy, 45.0, void=0.6, chip=0.5, depth=1.8, name="lift_joint")
    joint_wear(s, b.panels(2, 2, 0.0, 2), 12.0, void=0.3, chip=0.3, depth=1.0, name="seam")
    pinholes(s, 0.2, 440, 0.9)
    stains(s, srgb(124, 112, 92), 0.32, "blotch", 0.3, 0.3, 0.0, "damp_stain", rim=0.1)
    streaks(s, srgb(124, 112, 94), 0.4, "fine", 0.4, "rain_streak")
    crazing(s, 0.1, 0.3, 0.4, 0.25)
    cracks(s, 1.0, 0.6, 0.8, 0.3)
    return s.done()


def m11(par, res):
    """Terminal 27 - sun-bleached skim coat, trowel dragged, flaking in small patches"""
    b = L.B(par, "Concrete_11_Sunbleached_Skim", 3.0, res, m11.__doc__)
    s = S(b, srgb(196, 174, 142), 0.84, 0.42)
    base(
        s,
        macro=0.7,
        meso=0.35,
        fine=0.18,
        sand=0.11,
        speck=0.1,
        micro=0.05,
        pores=0.45,
        pore_depth=0.45,
        rvar=0.12,
    )
    mottle(s, srgb(206, 186, 156), srgb(174, 152, 122), 0.4, 0.05, 0.14)
    f1 = b.gr("layerednoise", 0.09, basenoise_elementsize=0.12, basecells_elementsize=0.08)
    f2 = b.gr("rustysurface", 0.09)
    f3 = b.sblur(b.fn(0.02, oct=5), 0.0, 0.06)
    f4 = b.fn(0.012, oct=4, sx=14.0, sy=1.0)
    CV = b.ctl("skim_coverage", 0.7, 0, 1)
    skim = b.x(
        "smoothstep(-0.03,0.03,CV-sat(a*0.5+(1.0-b)*0.45+(c-0.5)*0.4+(d-0.5)*0.35))",
        a=f1,
        b=f2,
        c=f3,
        d=f4,
        CV=CV,
    )
    smear = b.fn(0.008, oct=4, sx=34.0, sy=1.0)
    smear2 = b.fn(0.03, oct=4, sx=20.0, sy=1.0)
    s.h = b.x("mix(h,h*0.45+0.9+(sm-0.5)*0.7+(s2-0.5)*0.9,p)", h=s.h, p=skim, sm=smear, s2=smear2)
    s.col.append((skim, srgb(220, 205, 180), 1.0))
    s.col.append(
        (b.x("p*smoothstep(0.4,0.8,g)", p=skim, g=b.gr("moisture", 0.05)), srgb(214, 196, 166), 0.6)
    )
    s.tone.append((b.x("p*((sm-0.5)*1.4+(s2-0.5)*1.2)", p=skim, sm=smear, s2=smear2), 0.12))
    s.r.append((skim, -0.14))
    s.r.append((b.x("p*(s2-0.5)*2.0", p=skim, s2=smear2), 0.1))
    gouge = b.lv(b.fn(0.02, oct=3, sx=40.0, sy=1.0, typ="sparse"), 0.72, 0.9)
    s.addh(gouge, -0.5)
    s.col.append((gouge, srgb(150, 128, 100), 0.5))
    pinholes(s, 0.25, 420, 0.8, dirt=0.35)
    crazing(s, 0.07, 0.3, 0.55, 0.35, 0.1)
    stains(s, srgb(176, 152, 120), 0.2, "blotch", 0.2, 0.3, 0.0, "earth_stain", rim=0.1)
    cracks(s, 0.8, 0.9, 1.4, 0.45, 0.6)
    return s.done()


def m12(par, res):
    """The Brutalist - cool fair-faced panels, board widths ghosting through, single ties"""
    b = L.B(par, "Concrete_12_Fairfaced_Cool", 3.6, res, m12.__doc__)
    s = S(b, srgb(122, 127, 130), 0.6, 0.65)
    base(
        s,
        macro=0.3,
        meso=0.08,
        fine=0.04,
        sand=0.028,
        micro=0.028,
        pores=0.35,
        pore_depth=0.3,
        rvar=0.14,
    )
    mottle(s, srgb(138, 143, 145), srgb(104, 109, 112), 0.3, 0.045, 0.12)
    slats(s, 30, 0.06, 0.2, 0, 5, line=0.16)
    panels(
        s,
        3,
        2,
        stag=0.5,
        seam_mm=1.8,
        seam_d=-1.0,
        step=0.5,
        tone=0.1,
        seam_dirt=0.5,
        ties=(1, 2, 0.5, 0.22, 8.0),
        tie_depth=5.0,
    )
    joint_wear(s, s.seamdist, 8.0, void=0.25, chip=0.25, depth=0.7, name="seam")
    pinholes(s, 0.14, 500, 0.7, rmax=0.27)
    streaks(s, srgb(118, 122, 124), 0.25, "fine", 0.3, "rain_streak", 0.04)
    stains(s, srgb(126, 130, 132), 0.16, "moisture", 0.05, 0.55, 0.9, "damp")
    return s.done(0.012)


def m13(par, res):
    """The Brutalist - pale cast wall: smooth, fine joints, the odd tie hole"""
    b = L.B(par, "Concrete_13_Cast_Wall_Pale", 4.0, res, m13.__doc__)
    s = S(b, srgb(150, 156, 150), 0.56, 0.7)
    base(
        s,
        macro=0.3,
        meso=0.08,
        fine=0.035,
        sand=0.025,
        micro=0.025,
        pores=0.3,
        pore_depth=0.28,
        rvar=0.14,
    )
    mottle(s, srgb(162, 168, 162), srgb(132, 138, 134), 0.26, 0.045, 0.12)
    slats(s, 22, 0.022, 0.08, 0, 9, line=0.06)
    panels(
        s,
        2,
        3,
        stag=0.5,
        seam_mm=1.4,
        seam_d=-0.6,
        step=0.25,
        tone=0.035,
        seam_dirt=0.6,
        ties=(2, 1, 0.25, 0.5, 9.0),
        tie_depth=5.0,
    )
    joint_wear(
        s,
        s.seamdist,
        9.0,
        void=0.2,
        chip=0.2,
        depth=0.6,
        stain=srgb(100, 106, 104),
        stain_amt=0.3,
        name="seam",
    )
    pinholes(s, 0.1, 520, 0.6, rmax=0.26)
    streaks(s, srgb(120, 126, 124), 0.22, "fine", 0.35, "rain_streak", 0.04)
    streaks(s, srgb(176, 181, 176), 0.14, "broad", 0.5, "lime_bloom", 0.06, 1.6)
    crazing(s, 0.09, 0.22, 0.3, 0.18, 0.04)
    return s.done(0.012)


def m14(par, res):
    """The Brutalist - raw hand-floated wall, blue grey"""
    b = L.B(par, "Concrete_14_Raw_Floated_Blue", 3.0, res, m14.__doc__)
    s = S(b, srgb(94, 102, 110), 0.8, 0.55)
    base(
        s,
        macro=1.6,
        meso=0.7,
        fine=0.22,
        sand=0.12,
        speck=0.08,
        micro=0.05,
        pores=0.45,
        pore_depth=0.5,
        rvar=0.14,
    )
    mottle(s, srgb(108, 116, 123), srgb(78, 86, 93), 0.3, 0.05, 0.14)
    t1 = b.gr("wipetrails", 0.3, density=1.0, wipewidth=1.0, shapedetail=0.3, mixing=0.6)
    t2 = b.gr("wipetrails", 0.12, density=1.0, wipewidth=0.5)
    sm = b.warp(b.fn(0.012, oct=4, sx=1.0, sy=18.0), 0.25, 0.6)
    K = b.ctl("float_marks", 3.0, 0, 6)
    s.h = b.x("h+((a-0.3)*1.6+(b-0.3)*0.7+(c-0.5)*0.6)*K", h=s.h, a=t1, b=t2, c=sm, K=K)
    s.tone.append((b.x("(a-0.3)+(b-0.3)*0.5", a=t1, b=t2), 0.15))
    s.r.append((t1, -0.12))
    drag = b.lv(b.gr("rustysurface", 0.2), 0.55, 0.82)
    s.addh(b.x("m*(0.3+a)", m=drag, a=b.fn(0.008, oct=3, typ="alligator")), -1.6)
    s.col.append((drag, srgb(70, 77, 84), 0.4))
    pinholes(s, 0.28, 360, 1.5, rmax=0.33)
    stains(s, srgb(128, 135, 140), 0.22, "moisture", 0.04, 0.6, 0.9, "dust")
    return s.done(0.02)


def m15(par, res):
    """The Brutalist - dark green cast wall peppered with blowholes"""
    b = L.B(par, "Concrete_15_Blowholed_Dark", 3.0, res, m15.__doc__)
    s = S(b, srgb(80, 90, 86), 0.64, 0.7)
    base(s, macro=0.4, meso=0.12, fine=0.06, micro=0.03, pores=0.35, rvar=0.16)
    mottle(s, srgb(94, 104, 99), srgb(64, 73, 70), 0.36, 0.05, 0.14)
    panels(s, 2, 1, seam_mm=2.5, seam_d=0.5, step=0.4, tone=0.04)
    bugholes(s, 45, 0.3, 5.0, rmax=0.3, dirt=0.85, name="blowhole")
    bugholes(s, 100, 0.45, 2.4, rmax=0.36, dirt=0.8)
    bugholes(s, 220, 0.4, 1.0, rmax=0.36, dirt=0.7, name="pinhole")
    stains(s, srgb(112, 121, 115), 0.26, "moisture", 0.05, 0.55, 0.9, "bloom")
    streaks(s, srgb(52, 58, 57), 0.35, "fine", 0.4, "wet_streak", -0.1)
    return s.done()


def m16(par, res):
    """The Brutalist - blue grey slab face with drip lines and repairs"""
    b = L.B(par, "Concrete_16_Drip_Stained_Slab", 4.0, res, m16.__doc__)
    s = S(b, srgb(114, 126, 130), 0.68, 0.65)
    base(
        s,
        macro=0.5,
        meso=0.18,
        fine=0.09,
        sand=0.06,
        micro=0.04,
        pores=0.4,
        pore_depth=0.4,
        rvar=0.15,
    )
    mottle(s, srgb(130, 142, 145), srgb(94, 105, 110), 0.36)
    # sawn-board drag lines down the face
    vg = b.fn(0.005, oct=4, rough=0.65, sx=1.0, sy=60.0)
    vg2 = b.fn(0.02, oct=4, sx=1.0, sy=30.0)
    s.addh(b.x("(a-0.5)+(b-0.5)*0.8", a=vg, b=vg2), b.ctl("drag_lines", 0.55, 0, 2))
    s.tone.append((b.x("(a-0.5)+(b-0.5)", a=vg, b=vg2), 0.07))
    pid = b.panels(4, 3, 0.5, 1)
    pd = b.panels(4, 3, 0.5, 0)
    patch = b.x("step(0.84,p)*smoothstep(0.0,0.003,d)", p=pid, d=pd)
    edge = b.x("step(0.84,p)*(1.0-smoothstep(0.0,0.006,d))", p=pid, d=pd)
    s.col.append((patch, srgb(134, 146, 149), b.ctl("repair_patch", 0.14, 0, 1)))
    s.addh(patch, 0.5)
    s.addh(edge, -0.5)
    s.col.append((edge, srgb(60, 68, 72), 0.5))
    lj = b.warp(b.panels(1, 2, 0.0, 3), 0.004, 0.3)
    joint_wear(
        s,
        lj,
        14.0,
        void=0.5,
        chip=0.5,
        depth=1.4,
        stain=srgb(70, 80, 84),
        stain_amt=0.4,
        name="lift_joint",
    )
    dr = b.fn(0.006, oct=4, rough=0.7, sx=1.0, sy=120.0)
    gate = b.fn(0.12, oct=4, sx=1.0, sy=6.0)
    drip = b.x("smoothstep(0.56,0.7,a)*smoothstep(0.3,0.55,g)", a=dr, g=gate)
    s.col.append((drip, srgb(48, 55, 59), b.ctl("drip_line_amount", 0.8, 0, 1)))
    s.r.append((drip, -0.1))
    streaks(s, srgb(70, 80, 84), 0.45, "fine", 0.5, "rain_streak")
    streaks(s, srgb(160, 170, 172), 0.3, "leak", 0.3, "lime_streak")
    spots(s, srgb(22, 24, 26), 0.85, 26, 0.2, 0.16, -0.1, "black_spot")
    pinholes(s, 0.18, 440, 0.9)
    cracks(s, 0.6, 0.5, 0.7, 0.5, 0.8)
    return s.done()


def m17(par, res):
    """The Dark Knight Rises - bunker floor slab: saw cut bays, oil spots, traffic sheen"""
    b = L.B(par, "Concrete_17_Bunker_Floor_Slab", 4.0, res, m17.__doc__)
    s = S(b, srgb(124, 128, 122), 0.62, 0.8)
    base(
        s,
        macro=0.8,
        meso=0.22,
        fine=0.09,
        sand=0.055,
        speck=0.09,
        micro=0.035,
        pores=0.4,
        pore_depth=0.4,
        rvar=0.14,
    )
    mottle(s, srgb(142, 146, 140), srgb(100, 104, 100), 0.32, 0.055, 0.15)
    panels(s, 2, 2, seam_mm=3.0, seam_d=-3.5, step=0.3, tone=0.045, seam_dirt=0.75)
    joint_wear(
        s,
        s.seamdist,
        9.0,
        void=0.35,
        chip=0.7,
        depth=1.6,
        stain=srgb(74, 77, 74),
        stain_amt=0.4,
        name="saw_cut",
    )
    polish(s, 0.24, 0.3, tint=0.06)
    polish(s, 0.1, 0.06, name="fine_polish")
    dust(s, srgb(160, 162, 154), 0.3, 0.12, 0.05)
    spots(s, srgb(54, 56, 54), 0.6, 16, 0.3, 0.3, -0.16, "oil_spot")
    spots(s, srgb(70, 72, 70), 0.4, 44, 0.2, 0.3, -0.1, "drip_spot")
    stains(s, srgb(92, 96, 92), 0.3, "blotch", 0.3, 0.35, 0.0, "damp_patch", -0.1, rim=0.12)
    scuffs(s, srgb(64, 66, 64), 0.3, 0.4, -0.05, "tyre_mark")
    scuffs(s, srgb(168, 170, 164), 0.22, 0.16, 0.1)
    pinholes(s, 0.24, 400, 1.0, name="pit")
    crazing(s, 0.09, 0.3, 0.45, 0.3)
    cracks(s, 1.0, 0.7, 1.2, 0.3)
    return s.done()


def m18(par, res):
    """The Dark Knight - brown oxidised bunker wall: damp glossy runs, flaking skin"""
    b = L.B(par, "Concrete_18_Oxidised_Brown", 4.0, res, m18.__doc__)
    s = S(b, srgb(80, 76, 66), 0.66, 0.7)
    base(
        s,
        macro=0.7,
        meso=0.26,
        fine=0.11,
        sand=0.06,
        micro=0.04,
        pores=0.4,
        pore_depth=0.45,
        rvar=0.2,
    )
    mottle(s, srgb(100, 94, 80), srgb(54, 51, 45), 0.42)
    vg = b.fn(0.006, oct=4, rough=0.65, sx=1.0, sy=50.0)
    s.addh(b.x("a-0.5", a=vg), b.ctl("form_drag", 0.7, 0, 2))
    s.tone.append((b.x("(a-0.5)*2.0", a=vg), 0.06))
    panels(s, 4, 1, seam_mm=5, seam_d=-2.5, step=1.2, tone=0.08, seam_dirt=0.8)
    joint_wear(
        s,
        s.seamdist,
        22.0,
        void=0.3,
        chip=0.6,
        depth=1.6,
        stain=srgb(40, 37, 33),
        stain_amt=0.6,
        name="seam",
    )
    # the oxidised skin lifts off in flakes, paler cement underneath
    scaling(s, 0.12, 0.9, 0.07, srgb(104, 98, 84), "flaking", 0.2)
    stains(s, srgb(104, 82, 60), 0.28, "rustysurface", 0.2, 0.45, 0.85, "rust", 0.06)
    stains(s, srgb(66, 72, 58), 0.5, "moisture", 0.08, 0.5, 0.85, "algae_tint")
    stains(s, srgb(124, 106, 82), 0.18, "moisture", 0.05, 0.5, 0.85, "ochre_bloom")
    wet = streaks(s, srgb(34, 32, 29), 0.65, "broad", 0.55, "wet_run")
    s.r.append((b.x("0.0-a", a=wet), b.ctl("wet_gloss", 0.32, 0, 0.6)))
    s.sp.append((wet, 0.35))
    streaks(s, srgb(142, 126, 104), 0.35, "broad", 0.4, "mineral_run", 0.08, 1.4)
    streaks(s, srgb(58, 52, 46), 0.5, "leak", 0.45, "dark_leak", -0.12)
    streaks(s, srgb(64, 56, 48), 0.45, "fine", 0.35, "dirt_streak")
    pinholes(s, 0.14, 440, 0.9)
    crazing(s, 0.08, 0.3, 0.45, 0.35, 0.1)
    return s.done()


def m19(par, res):
    """V-Zug - brown cast concrete with a fine linear form texture and dirt along the lines"""
    b = L.B(par, "Concrete_19_Striated_Brown", 3.0, res, m19.__doc__)
    s = S(b, srgb(112, 101, 94), 0.7, 0.6)
    base(
        s,
        macro=0.3,
        meso=0.1,
        fine=0.045,
        sand=0.03,
        micro=0.03,
        pores=0.3,
        pore_depth=0.3,
        rvar=0.14,
    )
    mottle(s, srgb(128, 116, 108), srgb(90, 80, 75), 0.34, 0.06, 0.16)
    striations(s, 1, 0.55, 0.15, srgb(62, 55, 52), 0.7)
    panels(s, 2, 4, stag=0.5, seam_mm=1.6, seam_d=-0.7, step=0.3, tone=0.05, seam_dirt=0.55)
    joint_wear(
        s,
        s.seamdist,
        10.0,
        void=0.3,
        chip=0.3,
        depth=0.8,
        stain=srgb(72, 62, 58),
        stain_amt=0.35,
        name="seam",
    )
    band = b.fn(0.12, oct=4, sx=8.0, sy=1.0)
    s.col.append((b.lv(band, 0.55, 0.8), srgb(92, 80, 76), b.ctl("dark_band", 0.35, 0, 1)))
    pinholes(s, 0.12, 480, 0.7, rmax=0.27)
    streaks(s, srgb(96, 86, 82), 0.25, "fine", 0.4, "rain_streak")
    return s.done(0.01)


ALL = [
    m01,
    m02,
    m03,
    m04,
    m05,
    m06,
    m07,
    m08,
    m09,
    m10,
    m11,
    m12,
    m13,
    m14,
    m15,
    m16,
    m17,
    m18,
    m19,
]
