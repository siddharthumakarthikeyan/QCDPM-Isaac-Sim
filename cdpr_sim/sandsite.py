"""Sand-lot delivery environment: a flatbed truck has unloaded the blocks next to a levelled footing and the robots
build beside it.

Everything is procedural except the sand / concrete textures and a few small Poly Haven props (all local files, no
asset server). Background objects have no collision shapes; physics still sees one flat ground plane, i.e. the work
area is modelled as levelled, compacted sand (wheel sinkage is not simulated).

Layout (world frame, metres):
    footing slab     at the build centre, flush with the ground
    block stock      on boards at blocks.staging.center
    truck            parked east of the stock, parallel to y, near dropside lowered; never inside the robots' reach
                     (UGVs reach x <= staging_x + ugv_radius)
    loose sand       flat inside the work zone, rolling beyond ~10 m, dunes on the horizon
"""

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

from .scene import _bind, _bind_phys, _cyl, _material, _uv_box, _xform
from .site import _ph, _place, _ref, _sky, flat_material, pbr_material

E = "/World/Environment"


# ------------------------------------------------------------------------------------------------ geometry helpers

def _b(stage, path, size, pos, mat, tile=1.0):
    return _uv_box(stage, path, size, pos, mat=mat, tile=tile)


def _wheel(stage, path, pos, r, w, m):
    _cyl(stage, path + "_tire", r, w, "Y", pos, m["rubber"])
    _cyl(stage, path + "_rim", r * 0.58, w + 0.012, "Y", pos, m["rim"])
    _cyl(stage, path + "_hub", r * 0.20, w + 0.05, "Y", pos, m["dark"])


def _prism(stage, path, profile, y0, y1, mat):
    """Extrude a convex (x, z) outline (counter-clockwise seen from -y) between y0 and y1, flat shaded."""
    pts, cnt, nrm = [], [], []
    n = len(profile)

    def face(vs):
        v = np.array(vs, float)                           # Newell normal: robust when the first corners are collinear
        nn = np.cross(v, np.roll(v, -1, axis=0)).sum(0)
        nn /= np.linalg.norm(nn)
        pts.extend(vs)
        cnt.append(len(vs))
        nrm.extend([nn] * len(vs))

    face([(x, y0, z) for x, z in profile])
    face([(x, y1, z) for x, z in reversed(profile)])
    for i in range(n):
        (xa, za), (xb, zb) = profile[i], profile[(i + 1) % n]
        face([(xa, y0, za), (xa, y1, za), (xb, y1, zb), (xb, y0, zb)])
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    m.CreateFaceVertexCountsAttr(cnt)
    m.CreateFaceVertexIndicesAttr(list(range(len(pts))))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in nrm]))
    m.SetNormalsInterpolation("faceVarying")
    m.CreateSubdivisionSchemeAttr("none")
    _bind(m.GetPrim(), mat)
    return m


def _pallet(stage, path, pos, yaw_deg, m):
    """EUR pallet 1.2 x 0.8 x 0.144 m; origin on the ground at its centre."""
    _place(stage.DefinePrim(path, "Xform"), pos, yaw_deg)
    for i, y in enumerate((-0.35, -0.175, 0.0, 0.175, 0.35)):
        _b(stage, f"{path}/top_{i}", (1.2, 0.1 if i % 2 else 0.145, 0.022), (0, y * 0.93, 0.133), m["wood"], 0.6)
    for i, x in enumerate((-0.5275, 0.0, 0.5275)):
        _b(stage, f"{path}/cross_{i}", (0.145, 0.8, 0.022), (x, 0, 0.111), m["wood"], 0.6)
        for j, y in enumerate((-0.35, 0.0, 0.35)):
            _b(stage, f"{path}/blk_{i}_{j}", (0.145, 0.1, 0.078), (x, y, 0.061), m["wood"], 0.6)
    for j, y in enumerate((-0.35, 0.0, 0.35)):
        _b(stage, f"{path}/skid_{j}", (1.2, 0.1, 0.022), (0, y, 0.011), m["wood"], 0.6)


def _block_stack(stage, path, cfg, pos, yaw_deg, layers, m, strap=True):
    """Pallet with a 3 x 6 x layers stack of the same hollow blocks (drawn as solid boxes), optionally strapped."""
    L, W, H = cfg["blocks"]["size"]
    _pallet(stage, path + "/pallet", pos, yaw_deg, m)
    _place(stage.DefinePrim(path + "/stack", "Xform"), (pos[0], pos[1], pos[2] + 0.144), yaw_deg)
    for c in range(layers):
        for r in range(6):
            for n in range(3):
                x, y = (n - 1) * (L + 0.012), (r - 2.5) * (W + 0.012)
                p = (y, x, H / 2 + c * H) if c % 2 else (x, y, H / 2 + c * H)       # alternate courses turned 90 deg
                s = (W - 0.002, L, H - 0.002) if c % 2 else (L, W - 0.002, H - 0.002)
                if c % 2 and abs(y) > 0.42:                                           # turned course is shorter
                    continue
                _b(stage, f"{path}/stack/b_{c}_{r}_{n}", (s[0] - 0.006, s[1] - 0.006, s[2] - 0.004), p, m["block"], 0.45)
    if strap and layers:
        for i, x in enumerate((-0.2, 0.2)):
            _b(stage, f"{path}/stack/strap_{i}", (0.016, 0.76, layers * H + 0.004), (x, 0, layers * H / 2), m["strap"])


def _cone(stage, path, pos, m):
    _place(stage.DefinePrim(path, "Xform"), pos)
    _b(stage, path + "/base", (0.36, 0.36, 0.03), (0, 0, 0.015), m["dark"])
    for i, (z0, z1, mat) in enumerate(((0.03, 0.30, "cone"), (0.30, 0.46, "white"), (0.46, 0.70, "cone"))):
        c = UsdGeom.Cone.Define(stage, f"{path}/body_{i}")       # stacked frusta drawn as nested cones clipped by order
        r0 = 0.14 * (0.72 - z0) / 0.69
        c.CreateRadiusAttr(r0)
        c.CreateHeightAttr(0.72 - z0)
        c.CreateAxisAttr("Z")
        c.ClearXformOpOrder()
        c.AddTranslateOp().Set(Gf.Vec3d(0, 0, z0 + (0.72 - z0) / 2 + 0.0005 * i))
        c.AddScaleOp().Set(Gf.Vec3f(1 + 0.004 * i, 1 + 0.004 * i, 1))
        _bind(c.GetPrim(), m[mat])


# ------------------------------------------------------------------------------------------------ terrain

def _height(x, y, c, half):
    """Sand surface: exactly 0 inside the work zone (rectangle c +- half), rolling outside, dunes far away."""
    d = np.hypot(np.maximum(np.abs(x - c[0]) - half[0], 0), np.maximum(np.abs(y - c[1]) - half[1], 0))
    w = np.clip(d / 14.0, 0, 1)
    w = w * w * (3 - 2 * w)
    rng = np.random.default_rng(11)
    ripple = np.zeros_like(x)
    for _ in range(6):                                  # low mounds and hollows
        k, ph = rng.uniform(0.12, 0.5, 2) * rng.choice([-1, 1], 2), rng.uniform(0, 2 * np.pi)
        ripple += 0.09 * np.sin(k[0] * x + k[1] * y + ph)
    far = np.clip((d - 18.0) / 70.0, 0, 1)
    wx = x + 9 * np.sin(0.021 * y + 1.3)                # warped ridges -> dune crests
    dune = (0.5 + 0.5 * np.sin(0.045 * wx + 0.6)) ** 1.6 * (0.6 + 0.4 * np.sin(0.017 * y + 0.031 * x))
    dune += 0.5 * (0.5 + 0.5 * np.sin(0.083 * y - 0.02 * x + 2.0)) ** 2
    return w * (0.12 + ripple) + far ** 1.3 * 9.0 * dune


def _terrain(stage, cfg, m, pm_ground, c, half):
    n, ext = 280, 220.0
    s = np.linspace(-1, 1, n)
    g = np.sign(s) * np.abs(s) ** 1.9 * ext              # fine cells near the site, coarse towards the horizon
    X, Y = np.meshgrid(g + c[0], g + c[1], indexing="ij")
    Z = _height(X, Y, c, half)
    P = np.stack([X, Y, Z], -1)
    N = np.zeros_like(P)
    N[..., 0] = -np.gradient(Z, axis=0) / np.gradient(X, axis=0)
    N[..., 1] = -np.gradient(Z, axis=1) / np.gradient(Y, axis=1)
    N[..., 2] = 1.0
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    i = np.arange(n * n).reshape(n, n)
    quads = np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], -1).reshape(-1)
    mesh = UsdGeom.Mesh.Define(stage, E + "/Sand")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.reshape(-1, 3).astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full((n - 1) ** 2, 4, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.reshape(-1, 3).astype(np.float32)))
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateSubdivisionSchemeAttr("none")
    uv = (P[..., :2] / 2.2).reshape(-1, 2).astype(np.float32)          # one texture tile = 2.2 m
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(uv))
    _bind(mesh.GetPrim(), m["sand"])
    plane = UsdGeom.Plane.Define(stage, E + "/GroundCollider")
    plane.CreateAxisAttr("Z")
    plane.CreateWidthAttr(300.0)
    plane.CreateLengthAttr(300.0)
    UsdGeom.Imageable(plane).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    _bind_phys(plane.GetPrim(), pm_ground)


# ------------------------------------------------------------------------------------------------ truck

def _truck(stage, cfg, T, pos, yaw_deg, m, drop_left=True, body="flatbed"):
    """Two-axle truck, 6.9 x 2.3 x 2.55 m: a flatbed with part of its load, or (body="tank") a tanker with a pump.
    Local frame: x forward, y left, origin on the ground below the middle of the bed."""
    _place(stage.DefinePrim(T, "Xform"), pos, yaw_deg)
    wr, hw = 0.47, 1.15                                   # wheel radius, half width
    xf, xr = 3.55, -0.75                                  # axle positions
    bed0, bed1, zb = -2.35, 2.35, 1.08                    # bed extent and floor height
    # chassis
    for i, y in enumerate((-0.42, 0.42)):
        _b(stage, f"{T}/rail_{i}", (6.6, 0.09, 0.24), (1.05, y, 0.78), m["frame"])
    for i, x in enumerate(np.linspace(bed0 + 0.2, bed1 - 0.2, 7)):
        _b(stage, f"{T}/xmember_{i}", (0.08, 2.2, 0.12), (x, 0, 0.96), m["frame"])
    _cyl(stage, T + "/axle_f", 0.06, 2.0, "Y", (xf, 0, wr), m["frame"])
    _cyl(stage, T + "/axle_r", 0.08, 2.0, "Y", (xr, 0, wr), m["frame"])
    _cyl(stage, T + "/tank", 0.27, 0.9, "X", (1.6, -0.82, 0.62), m["alu"])
    _b(stage, T + "/battery_box", (0.7, 0.45, 0.42), (1.6, 0.86, 0.62), m["frame"])
    for sy in (-1, 1):
        _wheel(stage, f"{T}/wf_{'lr'[sy < 0]}", (xf, sy * (hw - 0.16), wr), wr, 0.28, m)
        _wheel(stage, f"{T}/wr_o_{'lr'[sy < 0]}", (xr, sy * (hw - 0.15), wr), wr, 0.26, m)
        _wheel(stage, f"{T}/wr_i_{'lr'[sy < 0]}", (xr, sy * (hw - 0.45), wr), wr, 0.26, m)
        _b(stage, f"{T}/mudguard_{'lr'[sy < 0]}", (1.25, 0.62, 0.035), (xr, sy * (hw - 0.31), 1.0), m["dark"])
        _b(stage, f"{T}/flap_{'lr'[sy < 0]}", (0.02, 0.56, 0.42), (xr - 0.64, sy * (hw - 0.31), 0.78), m["rubber"])
        _b(stage, f"{T}/underrun_{'lr'[sy < 0]}", (1.5, 0.03, 0.1), (1.1, sy * (hw - 0.02), 0.6), m["alu"])
    if body == "tank":
        _tank(stage, T, m, zb)
    else:
        # bed
        _b(stage, T + "/bed_floor", (bed1 - bed0, 2 * hw, 0.06), (0, 0, zb - 0.03), m["wood"], 0.8)
        _b(stage, T + "/bed_frame", (bed1 - bed0 + 0.02, 2 * hw + 0.02, 0.1), (0, 0, zb - 0.11), m["frame"])
        _b(stage, T + "/headboard", (0.06, 2 * hw, 1.25), (bed1 - 0.03, 0, zb + 0.625), m["alu"])
        for i, y in enumerate((-0.8, 0.0, 0.8)):
            _b(stage, f"{T}/headboard_rib_{i}", (0.03, 0.06, 1.25), (bed1 - 0.075, y, zb + 0.625), m["frame"])
        side_h = 0.5
        for sy in (-1, 1):
            dropped = (sy > 0) == drop_left
            for k in range(2):                                # two dropside panels per side with a centre post
                xc = bed0 + (bed1 - bed0) * (0.25 + 0.5 * k)
                z = zb - side_h / 2 - 0.02 if dropped else zb + side_h / 2
                y = sy * (hw + 0.035) if dropped else sy * (hw - 0.02)
                _b(stage, f"{T}/side_{'lr'[sy < 0]}_{k}", ((bed1 - bed0) / 2 - 0.06, 0.04, side_h), (xc, y, z), m["alu"])
                for r in range(3):                            # panel ribs
                    _b(stage, f"{T}/side_rib_{'lr'[sy < 0]}_{k}_{r}", ((bed1 - bed0) / 2 - 0.06, 0.012, 0.03),
                       (xc, y + sy * 0.024, z - side_h / 2 + 0.08 + r * 0.17), m["frame"])
            for k, x in enumerate((bed0 + 0.03, 0.0, bed1 - 0.1)):
                _b(stage, f"{T}/post_{'lr'[sy < 0]}_{k}", (0.06, 0.06, side_h), (x, sy * (hw - 0.03), zb + side_h / 2), m["frame"])
        _b(stage, T + "/tailgate", (0.04, 2 * hw - 0.14, side_h), (bed0 + 0.02, 0, zb + side_h / 2), m["alu"])
        _b(stage, T + "/rear_bar", (0.1, 2.2, 0.12), (bed0 + 0.1, 0, 0.62), m["frame"])
        for sy in (-1, 1):
            _b(stage, f"{T}/tail_light_{'lr'[sy < 0]}", (0.03, 0.3, 0.1), (bed0 + 0.04, sy * 0.85, 0.82), m["red"])
            for k, z in enumerate((0.62,)):
                _b(stage, f"{T}/chevron_{'lr'[sy < 0]}_{k}", (0.012, 0.5, 0.1), (bed0 + 0.045, sy * 0.55, z), m["cone" if sy > 0 else "white"])
    # cab: extruded side profile with a raked windscreen
    x0, x1, z0, z1 = 2.6, 4.45, 0.78, 2.55
    prof = [(x0, z0), (x1 - 0.06, z0), (x1, z0 + 0.25), (x1, 1.62), (x1 - 0.33, z1 - 0.07), (x1 - 0.5, z1), (x0, z1)]
    _prism(stage, T + "/cab", prof, -hw + 0.02, hw - 0.02, m["cab"])
    wa, wb = np.array([x1 + 0.004, 1.66]), np.array([x1 - 0.322, z1 - 0.13])       # windscreen on the raked face
    wc, wl = (wa + wb) / 2, np.linalg.norm(wb - wa)
    ang = np.arctan2(wb[1] - wa[1], wb[0] - wa[0])
    q = (np.cos(-ang / 2 + np.pi / 4), 0, np.sin(-ang / 2 + np.pi / 4), 0)             # box z-axis along the glass
    _uv_box(stage, T + "/windscreen", (0.015, 2 * hw - 0.3, wl), (wc[0], 0, wc[1]), quat=q, mat=m["glass"])
    for sy in (-1, 1):
        _b(stage, f"{T}/side_glass_{'lr'[sy < 0]}", (0.95, 0.012, 0.6), (3.72, sy * (hw - 0.016), 1.98), m["glass"])
        _b(stage, f"{T}/door_gap_{'lr'[sy < 0]}", (0.012, 0.01, 1.55), (3.12, sy * (hw - 0.017), 1.6), m["dark"])
        _b(stage, f"{T}/door_handle_{'lr'[sy < 0]}", (0.14, 0.02, 0.035), (3.3, sy * (hw - 0.012), 1.55), m["dark"])
        _b(stage, f"{T}/step_{'lr'[sy < 0]}", (0.5, 0.2, 0.04), (3.0, sy * (hw - 0.1), 0.55), m["alu"])
        _b(stage, f"{T}/arch_{'lr'[sy < 0]}", (1.2, 0.34, 0.05), (xf, sy * (hw - 0.17), 1.03), m["dark"])
        _b(stage, f"{T}/mirror_arm_{'lr'[sy < 0]}", (0.03, 0.26, 0.03), (4.2, sy * (hw + 0.1), 2.2), m["dark"])
        _b(stage, f"{T}/mirror_{'lr'[sy < 0]}", (0.05, 0.14, 0.36), (4.2, sy * (hw + 0.24), 2.05), m["dark"])
        _b(stage, f"{T}/headlight_{'lr'[sy < 0]}", (0.03, 0.34, 0.14), (x1 - 0.02, sy * 0.8, 0.95), m["lens"])
        _b(stage, f"{T}/indicator_{'lr'[sy < 0]}", (0.03, 0.1, 0.14), (x1 - 0.02, sy * 1.04, 0.95), m["cone"])
    _b(stage, T + "/grille", (0.03, 1.5, 0.42), (x1 + 0.005, 0, 1.32), m["dark"])
    for i in range(4):
        _b(stage, f"{T}/grille_bar_{i}", (0.02, 1.46, 0.025), (x1 + 0.022, 0, 1.17 + i * 0.1), m["alu"])
    _b(stage, T + "/bumper", (0.22, 2 * hw, 0.3), (x1 - 0.02, 0, 0.66), m["frame"])
    _b(stage, T + "/plate", (0.012, 0.5, 0.12), (x1 + 0.095, 0, 0.66), m["white"])
    _b(stage, T + "/stripe", (0.012, 2 * hw - 0.2, 0.09), (x1 + 0.004, 0, 1.56), m["cone"])
    _b(stage, T + "/sunvisor", (0.25, 2 * hw - 0.2, 0.03), (x1 - 0.4, 0, z1 + 0.0), m["dark"])
    for i, y in enumerate((-0.5, 0.5)):
        _cyl(stage, f"{T}/beacon_{i}", 0.06, 0.1, "Z", (3.3, y, z1 + 0.05), m["cone"])
    _b(stage, T + "/cab_back_guard", (0.05, 2 * hw - 0.2, 1.3), (x0 - 0.12, 0, zb + 0.7), m["frame"])
    if body == "flatbed":
        # the part of the load that is still on the bed
        _block_stack(stage, T + "/load_0", cfg, (1.55, -0.55, zb), 90, 4, m)
        _block_stack(stage, T + "/load_1", cfg, (1.55, 0.55, zb), 90, 4, m)
        _block_stack(stage, T + "/load_2", cfg, (0.6, -0.55, zb), 90, 3, m)
        _pallet(stage, T + "/empty_0", (-1.5, 0.3, zb), 4, m)
        _pallet(stage, T + "/empty_1", (-1.5, 0.3, zb + 0.144), -3, m)


TANK_OUTLET = (-1.95, 1.22, 0.86)          # pump outlet in the truck frame (left side, towards the rear)


def _tank(stage, T, m, zb):
    """Tank body on saddles with a pump unit at the rear left; the hose leaves from TANK_OUTLET."""
    from .dressup import lathe

    r, zc = 0.92, zb + 0.98
    xs = np.linspace(-2.25, 2.2, 40)
    prof = [(float(r * np.sqrt(max(0.0, 1 - max(0.0, abs(x + 0.025) - 1.9) ** 2 / 0.33 ** 2))), float(x)) for x in xs]
    lathe(stage, T + "/tank", [(0.0, -2.25)] + prof[1:-1] + [(0.0, 2.2)], m["tank"], "X", (0, 0, zc), smooth=True)
    for i, x in enumerate((-1.5, 0.0, 1.5)):
        _b(stage, f"{T}/saddle_{i}", (0.16, 1.5, 0.34), (x, 0, zb + 0.1), m["frame"])
        lathe(stage, f"{T}/band_{i}", [(r + 0.006, -0.05), (r + 0.016, -0.05), (r + 0.016, 0.05), (r + 0.006, 0.05)], m["alu"], "X", (x, 0, zc))
    _b(stage, T + "/deck", (4.5, 2.2, 0.08), (0, 0, zb - 0.04), m["frame"])
    _cyl(stage, T + "/hatch", 0.28, 0.14, "Z", (0.6, 0, zc + r + 0.02), m["alu"])
    _b(stage, T + "/walkway", (3.2, 0.4, 0.03), (0, 0, zc + r + 0.0), m["alu"])
    for k in range(6):                                    # rear ladder
        _b(stage, f"{T}/rung_{k}", (0.03, 0.42, 0.03), (-2.32, 0, zb + 0.25 + 0.3 * k), m["alu"])
    for i, y in enumerate((-0.21, 0.21)):
        _b(stage, f"{T}/ladder_rail_{i}", (0.03, 0.03, 1.9), (-2.32, y, zb + 0.95), m["alu"])
    ox, oy, oz = TANK_OUTLET
    _b(stage, T + "/pump", (0.8, 0.55, 0.6), (ox + 0.05, oy - 0.36, oz - 0.02), m["cone"])
    _b(stage, T + "/pump_panel", (0.5, 0.02, 0.3), (ox + 0.05, oy - 0.08, oz + 0.08), m["dark"])
    _cyl(stage, T + "/outlet", 0.06, 0.16, "Y", (ox, oy, oz), m["alu"])
    _cyl(stage, T + "/suction", 0.05, 0.9, "Z", (ox + 0.3, oy - 0.5, oz + 0.6), m["alu"])


def build_tanker(stage, cfg, path, pos, yaw_deg=90.0):
    """A tanker truck on its own (any environment). Returns the world position of the pump outlet."""
    E_ = path + "_looks"
    col = {"block": ((0.6, 0.6, 0.6), 0.8), "strap": ((0.05, 0.25, 0.6), 0.5), "wood": ((0.60, 0.46, 0.29), 0.82), "cab": ((0.78, 0.10, 0.06), 0.28, 0.35),
           "tank": ((0.86, 0.87, 0.88), 0.22, 0.75), "frame": ((0.06, 0.06, 0.065), 0.55, 0.4), "alu": ((0.72, 0.73, 0.74), 0.38, 0.85),
           "rim": ((0.80, 0.80, 0.80), 0.35, 0.9), "rubber": ((0.02, 0.02, 0.02), 0.92), "dark": ((0.03, 0.03, 0.035), 0.5),
           "glass": ((0.02, 0.03, 0.04), 0.06, 0.6), "lens": ((0.9, 0.9, 0.85), 0.1, 0.5), "red": ((0.6, 0.02, 0.02), 0.3),
           "cone": ((1.0, 0.33, 0.03), 0.5), "white": ((0.9, 0.9, 0.9), 0.5)}
    m = {k: flat_material(stage, f"{E_}/{k}", *v) for k, v in col.items()}
    _truck(stage, cfg, path, pos, yaw_deg, m, body="tank")
    a = np.deg2rad(yaw_deg)
    ox, oy, oz = TANK_OUTLET
    return np.array([pos[0] + np.cos(a) * ox - np.sin(a) * (oy + 0.08), pos[1] + np.sin(a) * ox + np.cos(a) * (oy + 0.08), oz])


# ------------------------------------------------------------------------------------------------ site

def build_sand(stage, cfg, pm_ground, tex):
    env = dict(cfg["environment"])
    env.update(env.get("sand") or {})
    _xform(stage, E)
    m = {
        "sand": pbr_material(stage, E + "/Looks/sand", "dense_sand", tint=tuple(env.get("sand_tint", (0.86, 0.70, 0.50)))),
        "track": pbr_material(stage, E + "/Looks/sand_track", "dense_sand", tint=(0.60, 0.48, 0.34)),
        "pad": pbr_material(stage, E + "/Looks/footing", "concrete_floor_02", tint=(0.62, 0.61, 0.59)),
        "block": pbr_material(stage, E + "/Looks/block", "rough_concrete", res="1k", tint=(0.90, 0.89, 0.87)),
        "wood": flat_material(stage, E + "/Looks/wood", (0.60, 0.46, 0.29), 0.82),
        "cab": flat_material(stage, E + "/Looks/cab", (0.05, 0.16, 0.42), 0.28, 0.35),
        "frame": flat_material(stage, E + "/Looks/frame", (0.06, 0.06, 0.065), 0.55, 0.4),
        "alu": flat_material(stage, E + "/Looks/alu", (0.72, 0.73, 0.74), 0.38, 0.85),
        "rim": flat_material(stage, E + "/Looks/rim", (0.80, 0.80, 0.80), 0.35, 0.9),
        "rubber": flat_material(stage, E + "/Looks/rubber", (0.02, 0.02, 0.02), 0.92),
        "dark": flat_material(stage, E + "/Looks/dark", (0.03, 0.03, 0.035), 0.5),
        "glass": flat_material(stage, E + "/Looks/glass", (0.02, 0.03, 0.04), 0.06, 0.6),
        "lens": flat_material(stage, E + "/Looks/lens", (0.9, 0.9, 0.85), 0.1, 0.5),
        "red": flat_material(stage, E + "/Looks/red", (0.6, 0.02, 0.02), 0.3),
        "cone": flat_material(stage, E + "/Looks/cone", (1.0, 0.33, 0.03), 0.5),
        "white": flat_material(stage, E + "/Looks/white", (0.9, 0.9, 0.9), 0.5),
        "strap": flat_material(stage, E + "/Looks/strap", (0.05, 0.25, 0.6), 0.5),
    }
    bx, by = cfg["task"]["build_center"]
    sx, sy = cfg["blocks"]["staging"]["center"]
    reach = sx + cfg["layout"]["ugv_radius"] * np.cos(np.pi / 4) + 0.45      # east edge of the robots' footprint
    centre, half = ((bx + sx) / 2 + 1.5, (by + sy) / 2), (abs(sx - bx) / 2 + 7.5, 7.0)
    _terrain(stage, cfg, m, pm_ground, centre, half)
    _sky(stage, E, {"environment": env})

    # levelled footing for the structure and boards under the unloaded stock (2 mm proud: visual only)
    _b(stage, E + "/Footing", (2.6, 2.6, 0.08), (bx, by, -0.038), m["pad"], 1.6)
    st = cfg["blocks"]["staging"]
    L, W, _ = cfg["blocks"]["size"]
    bw, bl = st["cols"] * (L + st["gap"][0]) + 0.25, st["rows"] * (W + st["gap"][1]) + 0.25
    if env.get("stock", True):
        _b(stage, E + "/StockBoards", (bw, bl, 0.03), (sx, sy, -0.013), m["wood"], 0.7)

    # truck broadside to the stock, bed centred on it, dropside towards the robots lowered
    tx = max(reach + 0.75, sx + 3.0) + 1.15
    _truck(stage, cfg, E + "/Truck", (tx, sy, 0.0), 90.0, m, drop_left=True)
    for i, x in enumerate((tx - 0.86, tx + 0.86)):          # tyre tracks where it drove in
        _b(stage, f"{E}/Track_{i}", (0.34, 70.0, 0.004), (x, sy - 38.0, 0.001), m["track"], 2.2)

    # a little site clutter behind the tail of the truck and cones around the work zone
    _block_stack(stage, E + "/Yard/stack_0", cfg, (tx + 0.3, sy - 5.2, 0), 8, 4, m)
    _block_stack(stage, E + "/Yard/stack_1", cfg, (tx - 1.3, sy - 5.6, 0), -6, 2, m, strap=False)
    _pallet(stage, E + "/Yard/pallet_0", (tx + 1.9, sy - 5.0, 0), 80, m)
    _pallet(stage, E + "/Yard/pallet_1", (tx + 1.9, sy - 5.0, 0.144), 86, m)
    for i in range(5):
        _ref(stage, f"{E}/Yard/bag_{i}", _ph("cement_bag"), (tx + 1.75 + (i % 2) * 0.1, sy - 5.25 + (i % 3) * 0.26, 0.288 + (i // 3) * 0.17), 80)
    _ref(stage, E + "/Yard/generator", _ph("portable_generator"), (tx - 2.6, sy - 5.4, 0), 200)
    _ref(stage, E + "/Yard/toolbox", _ph("metal_toolbox"), (tx - 2.0, sy - 4.6, 0), 35)
    _ref(stage, E + "/Yard/jerrycan", _ph("metal_jerrycan"), (tx - 3.0, sy - 4.9, 0), 110)
    x_w = bx - cfg["layout"]["ugv_radius"] - 1.2
    cones = ((x_w, -4.2), (x_w, 4.2), (reach + 0.3, -4.2), (reach + 0.3, 4.2), (x_w, 0.0), ((x_w + reach) / 2, -4.2), ((x_w + reach) / 2, 4.2))
    for i, (x, y) in enumerate(cones if env.get("cones", True) else ()):
        _cone(stage, f"{E}/Cones/cone_{i}", (x, y, 0), m)
