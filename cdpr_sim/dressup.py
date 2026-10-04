"""Render-only detail for the robots (used by the video tools, never by the physics runs).

The simulation models are plain boxes and cylinders, which is all the physics needs. For film close-ups this hides
those visuals and hangs detailed ones under the same rigid bodies, so they follow the recorded poses. Dimensions
come from the same config (arm length, prop radius, chassis size, wheel radius, ...): nothing changes size.
No collision shapes or mass properties are touched.
"""

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from .scene import _bind
from .site import _tube


# ------------------------------------------------------------------------------------------------ materials

def _mat(stage, name, color, rough=0.5, metal=0.0, opacity=1.0, emissive=None, coat=0.0, ior=None):
    path = "/World/Looks/dress_" + name
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if coat:
        sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(coat)
        sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.08)
    if opacity < 1.0:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if ior is not None:                                   # ior 1: no refraction or mirror reflection (propeller blur)
        sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
        sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    if emissive:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def materials(stage):
    return {
        "shell": _mat(stage, "shell", (0.82, 0.83, 0.84), 0.28, 0.0, coat=0.6),
        "graphite": _mat(stage, "graphite", (0.055, 0.058, 0.065), 0.38, 0.3, coat=0.3),
        "carbon": _mat(stage, "carbon", (0.02, 0.02, 0.022), 0.3, 0.1, coat=0.5),
        "alu": _mat(stage, "alu", (0.78, 0.79, 0.80), 0.3, 1.0),
        "steel": _mat(stage, "steel", (0.35, 0.36, 0.38), 0.35, 1.0),
        "accent": _mat(stage, "accent", (1.0, 0.33, 0.04), 0.35, 0.0, coat=0.5),
        "rubber": _mat(stage, "rubber", (0.018, 0.018, 0.018), 0.9),
        "rope": _mat(stage, "rope", (0.95, 0.76, 0.10), 0.7),
        "glass": _mat(stage, "glass", (0.01, 0.012, 0.016), 0.05, 0.4, coat=1.0),
        "blur": _mat(stage, "prop_blur", (0.03, 0.03, 0.035), 1.0, opacity=0.22, ior=1.0),
        "led_w": _mat(stage, "led_white", (1, 1, 1), 0.3, emissive=(6.0, 6.0, 5.4)),
        "led_r": _mat(stage, "led_red", (1, 0.05, 0.02), 0.3, emissive=(6.0, 0.25, 0.1)),
        "led_g": _mat(stage, "led_green", (0.05, 1, 0.2), 0.3, emissive=(0.3, 6.0, 1.0)),
        "red": _mat(stage, "red", (0.75, 0.03, 0.03), 0.35, coat=0.5),
        "yellow": _mat(stage, "yellow", (0.95, 0.75, 0.05), 0.45),
    }


# ------------------------------------------------------------------------------------------------ meshes

def _mesh(stage, path, P, N, mat, pos=(0, 0, 0), quat=(1, 0, 0, 0), scale=None, st=None):
    """Quad grid mesh from point / normal arrays of shape (a, b, 3)."""
    a, b = P.shape[:2]
    i = np.arange(a * b).reshape(a, b)
    q = np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], -1).reshape(-1)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.reshape(-1, 3).astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full((a - 1) * (b - 1), 4, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(q.astype(np.int32)))
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.reshape(-1, 3).astype(np.float32)))
    m.SetNormalsInterpolation("vertex")
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    if st is not None:
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(st.reshape(-1, 2).astype(np.float32)))
    m.ClearXformOpOrder()
    m.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    m.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
    if scale is not None:
        m.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    _bind(m.GetPrim(), mat)
    return m


def lathe(stage, path, profile, mat, axis="Z", pos=(0, 0, 0), quat=(1, 0, 0, 0), scale=None, n=56, smooth=False,
          arc=(0.0, 360.0)):
    """Surface of revolution. profile: (radius, height) points along the axis; sharp profile corners unless smooth.
    arc: swept angle range [deg] (for axis Y the angle is measured from +z towards +x)."""
    pr = np.asarray(profile, float)
    if not smooth:                                        # duplicate inner points so each segment is flat-shaded
        pr = np.concatenate([pr[:1]] + [np.repeat(pr[k:k + 1], 2, 0) for k in range(1, len(pr) - 1)] + [pr[-1:]])
        seg = pr[1::2] - pr[0::2]
        nr = np.repeat(np.stack([seg[:, 1], -seg[:, 0]], -1), 2, 0)
    else:
        d = np.gradient(pr, axis=0)
        nr = np.stack([d[:, 1], -d[:, 0]], -1)
    nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-9)
    a = np.deg2rad(np.linspace(arc[0], arc[1], n + 1))
    c, s = np.cos(a)[None, :], np.sin(a)[None, :]
    i, j, k = {"X": (1, 2, 0), "Y": (2, 0, 1), "Z": (0, 1, 2)}[axis]
    P = np.zeros((len(pr), n + 1, 3))
    N = np.zeros_like(P)
    P[..., i], P[..., j], P[..., k] = pr[:, :1] * c, pr[:, :1] * s, pr[:, 1:2]
    N[..., i], N[..., j], N[..., k] = nr[:, :1] * c, nr[:, :1] * s, nr[:, 1:2]
    return _mesh(stage, path, P, N, mat, pos, quat, scale)


def rbox(stage, path, size, radius, mat, pos=(0, 0, 0), quat=(1, 0, 0, 0), n=6, tile=None, uv0=(0.0, 0.0)):
    """Box with rounded edges and corners (one mesh per face). tile [m]: texture repeat for planar UVs."""
    h = np.asarray(size, float) / 2
    r = min(radius, h.min() - 1e-4)
    x = UsdGeom.Xform.Define(stage, path)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
    ang = np.linspace(0, np.pi / 2, n)

    def coords(hh):                                       # fine samples in the rounded zones
        e = hh - r + r * np.sin(ang)
        return np.concatenate([-e[::-1], e])

    for f, (ax, sg) in enumerate(((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))):
        u, v = [(1, 2), (2, 0), (0, 1)][ax]
        U, V = np.meshgrid(coords(h[u]), coords(h[v]), indexing="ij")
        P = np.zeros(U.shape + (3,))
        P[..., u], P[..., v], P[..., ax] = U, V, sg * h[ax]
        inner = np.clip(P, -(h - r), h - r)
        d = P - inner
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        st = None if tile is None else np.stack([U, V], -1) / tile + np.asarray(uv0) + 0.37 * f
        _mesh(stage, f"{path}/f{f}", inner + r * d, d, mat, st=st)
    return x


def box(stage, path, size, pos, mat, quat=(1, 0, 0, 0)):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.ClearXformOpOrder()
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    c.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    _bind(c.GetPrim(), mat)
    return c


def _qz(deg):
    a = np.deg2rad(deg) / 2
    return (np.cos(a), 0, 0, np.sin(a))


def _qy(deg):
    a = np.deg2rad(deg) / 2
    return (np.cos(a), 0, np.sin(a), 0)


def _hide(stage, root, keep=()):
    for c in stage.GetPrimAtPath(root).GetChildren():
        if c.IsA(UsdGeom.Gprim) and c.GetName() not in keep:
            UsdGeom.Imageable(c).MakeInvisible()


def _winch(stage, path, pos, r, w, m):
    """Cable drum on a bracket, axis along y."""
    fl = r * 1.3
    prof = [(0, -w / 2 - 0.004), (fl, -w / 2 - 0.004), (fl, -w / 2), (r * 0.75, -w / 2), (r * 0.75, w / 2), (fl, w / 2),
            (fl, w / 2 + 0.004), (0, w / 2 + 0.004)]
    lathe(stage, path + "_drum", prof, m["accent"], "Y", pos)
    lathe(stage, path + "_rope", [(r, -w / 2 + 0.002), (r, w / 2 - 0.002)], m["rope"], "Y", pos)
    for i, sy in enumerate((-1, 1)):
        box(stage, f"{path}_cheek_{i}", (r * 1.1, 0.006, fl * 1.9), (pos[0], pos[1] + sy * (w / 2 + 0.008), pos[2] + fl * 0.25), m["graphite"])
    lathe(stage, path + "_motor", [(0, 0), (r * 0.6, 0), (r * 0.6, w * 0.7), (0, w * 0.7)], m["steel"], "Y",
          (pos[0], pos[1] + w / 2 + 0.011, pos[2]))


# ------------------------------------------------------------------------------------------------ robots

def dress_drone(stage, cfg, path, m):
    d = cfg["drone"]
    a, pr = d["arm_length"], 0.127
    _hide(stage, path, keep=("fairlead",))
    V = path + "/vis"
    UsdGeom.Xform.Define(stage, V)
    rbox(stage, V + "/hull", (0.21, 0.115, 0.056), 0.024, m["shell"])
    lathe(stage, V + "/canopy", [(np.cos(t), np.sin(t)) for t in np.linspace(0, np.pi / 2, 10)], m["graphite"],
          pos=(0.012, 0, 0.022), scale=(0.078, 0.046, 0.034), smooth=True)
    box(stage, V + "/strap", (0.026, 0.1165, 0.0575), (-0.045, 0, 0), m["accent"])
    box(stage, V + "/battery", (0.12, 0.07, 0.03), (-0.005, 0, -0.04), m["graphite"])
    lathe(stage, V + "/gimbal", [(np.sin(t), -np.cos(t)) for t in np.linspace(0, np.pi, 14)], m["glass"],
          pos=(0.098, 0, -0.03), scale=(0.024, 0.024, 0.024), smooth=True)
    box(stage, V + "/gimbal_yoke", (0.02, 0.058, 0.012), (0.092, 0, -0.012), m["graphite"])
    for i in range(4):
        ang = np.deg2rad(45 + 90 * i)
        c, s = np.cos(ang), np.sin(ang)
        tip = np.array([c * a, s * a, 0.0])
        _tube(stage, f"{V}/arm_{i}", (c * 0.06, s * 0.06, 0.0), tip, 0.0105, m["carbon"])
        lathe(stage, f"{V}/mount_{i}", [(0, -0.016), (0.02, -0.016), (0.024, -0.008), (0.024, 0.004), (0, 0.004)], m["graphite"], pos=tip)
        lathe(stage, f"{V}/motor_{i}", [(0, 0.004), (0.021, 0.004), (0.021, 0.026), (0.017, 0.032), (0.005, 0.032), (0.005, 0.04),
                                         (0, 0.04)], m["alu"], pos=tip)
        lathe(stage, f"{V}/spinner_{i}", [(0, 0.036), (0.011, 0.036), (0.009, 0.046), (0.004, 0.054), (0, 0.056)], m["accent"],
              pos=tip, smooth=True)
        lathe(stage, f"{V}/prop_{i}", [(0.008, 0.0405), (pr, 0.0405), (pr, 0.0425), (0.008, 0.0425)], m["blur"], pos=tip)
        lathe(stage, f"{V}/proptip_{i}", [(pr - 0.004, 0.040), (pr, 0.040), (pr, 0.043), (pr - 0.004, 0.043), (pr - 0.004, 0.040)],
              m["blur"], pos=tip)
        box(stage, f"{V}/led_{i}", (0.012, 0.012, 0.005), tip + [0, 0, -0.0185], m["led_g" if c > 0 else "led_r"], _qz(45 + 90 * i))
    for i, sy in enumerate((-1, 1)):                      # landing gear
        y = sy * 0.095
        _tube(stage, f"{V}/skid_{i}", (-0.11, y, -0.135), (0.11, y, -0.135), 0.006, m["carbon"])
        for k, x in enumerate((-0.06, 0.06)):
            _tube(stage, f"{V}/strut_{i}_{k}", (x * 0.75, sy * 0.05, -0.025), (x, y, -0.135), 0.005, m["carbon"])
            lathe(stage, f"{V}/foot_{i}_{k}", [(0, -0.008), (0.009, -0.008), (0.009, 0.008), (0, 0.008)], m["rubber"], "X",
                  (np.sign(x) * 0.108, y, -0.135))
    _winch(stage, V + "/winch", (0.0, 0.0, -0.055), 0.022, 0.05, m)
    _tube(stage, V + "/gps_mast", (-0.075, 0, 0.025), (-0.075, 0, 0.085), 0.0035, m["carbon"])
    lathe(stage, V + "/gps", [(0, 0.083), (0.022, 0.083), (0.022, 0.092), (0.016, 0.097), (0, 0.097)], m["shell"], pos=(-0.075, 0, 0))


def dress_ugv(stage, cfg, path, m):
    u = cfg["ugv"]
    cs, r, ww = u["chassis_size"], u["wheel_radius"], u["wheel_width"]
    cz = u["ground_clearance"] + cs[2] / 2
    ch = path + "/chassis"
    _hide(stage, ch, keep=("fairlead", "caster_0", "caster_1"))
    V = ch + "/vis"
    UsdGeom.Xform.Define(stage, V)
    top = cs[2] / 2
    rbox(stage, V + "/tub", (cs[0], cs[1] - 0.05, cs[2] * 0.62), 0.022, m["graphite"], (0, 0, -cs[2] * 0.19))
    rbox(stage, V + "/cover", (cs[0] - 0.05, cs[1] - 0.07, cs[2] * 0.42), 0.03, m["shell"], (0, 0, top - cs[2] * 0.21))
    box(stage, V + "/band", (cs[0] - 0.02, cs[1] - 0.046, 0.016), (0, 0, cs[2] * 0.115), m["accent"])
    box(stage, V + "/deck", (cs[0] - 0.16, cs[1] - 0.16, 0.004), (-0.01, 0, top + 0.002), m["graphite"])
    for i, sy in enumerate((-1, 1)):                      # fenders over the drive wheels
        lathe(stage, f"{V}/fender_{i}", [(r + 0.022, -0.03), (r + 0.028, -0.03), (r + 0.028, 0.03), (r + 0.022, 0.03), (r + 0.022, -0.03)],
              m["graphite"], "Y", (0, sy * u["track_width"] / 2, r - cz), arc=(-80, 80))
        box(stage, f"{V}/headlight_{i}", (0.008, 0.07, 0.022), (cs[0] / 2 - 0.002, sy * 0.105, -0.01), m["led_w"])
        box(stage, f"{V}/taillight_{i}", (0.008, 0.05, 0.016), (-cs[0] / 2 + 0.002, sy * 0.115, -0.01), m["led_r"])
        _tube(stage, f"{V}/bumper_arm_{i}", (cs[0] / 2 - 0.01, sy * 0.13, -0.065), (cs[0] / 2 + 0.03, sy * 0.13, -0.065), 0.008, m["steel"])
    _tube(stage, V + "/bumper", (cs[0] / 2 + 0.03, -0.17, -0.065), (cs[0] / 2 + 0.03, 0.17, -0.065), 0.011, m["rubber"])
    lathe(stage, V + "/lidar_base", [(0, 0), (0.04, 0), (0.04, 0.022), (0, 0.022)], m["graphite"], pos=(0.15, 0, top))
    lathe(stage, V + "/lidar", [(0.036, 0.022), (0.036, 0.05), (0.03, 0.058), (0, 0.058)], m["glass"], pos=(0.15, 0, top))
    mast_h = u["attach_height"] - cz
    lathe(stage, V + "/mast_foot", [(0, 0), (0.038, 0), (0.038, 0.008), (0.026, 0.02), (0, 0.02)], m["steel"], pos=(0, 0, top))
    _tube(stage, V + "/mast", (0, 0, top), (0, 0, mast_h - 0.012), 0.014, m["alu"])
    lathe(stage, V + "/pulley", [(0, -0.008), (0.024, -0.008), (0.024, -0.005), (0.016, -0.004), (0.016, 0.004), (0.024, 0.005), (0.024, 0.008),
                                 (0, 0.008)], m["steel"], "Y", (0, 0, mast_h - 0.004))
    _winch(stage, V + "/winch", (-0.13, 0.0, top + 0.05), 0.034, 0.075, m)
    _tube(stage, V + "/antenna", (-0.2, 0.12, top), (-0.2, 0.12, top + 0.26), 0.003, m["carbon"])
    lathe(stage, V + "/antenna_base", [(0, 0), (0.012, 0), (0.008, 0.02), (0, 0.02)], m["graphite"], pos=(-0.2, 0.12, top))
    lathe(stage, V + "/estop_base", [(0, 0), (0.026, 0), (0.026, 0.012), (0, 0.012)], m["yellow"], pos=(-0.2, -0.12, top))
    lathe(stage, V + "/estop", [(0, 0.012), (0.018, 0.012), (0.018, 0.026), (0.012, 0.03), (0, 0.03)], m["red"], pos=(-0.2, -0.12, top))
    for name in ("left", "right"):
        wp = f"{path}/wheel_{name}"
        _hide(stage, wp)
        W = wp + "/vis"
        UsdGeom.Xform.Define(stage, W)
        h = ww / 2
        lathe(stage, W + "/tire", [(r * 0.62, -h), (r * 0.88, -h), (r * 0.965, -h + 0.006), (r, -h + 0.014), (r, h - 0.014),
                                   (r * 0.965, h - 0.006), (r * 0.88, h), (r * 0.62, h)], m["rubber"], "Y")
        lathe(stage, W + "/rim", [(0, -h * 0.7), (r * 0.2, -h * 0.7), (r * 0.26, -h * 0.9), (r * 0.56, -h * 0.9), (r * 0.64, -h * 0.75),
                                  (r * 0.64, h * 0.75), (r * 0.56, h * 0.9), (r * 0.26, h * 0.9), (r * 0.2, h * 0.7), (0, h * 0.7)], m["alu"], "Y")
        lathe(stage, W + "/cap", [(0, -h - 0.004), (r * 0.16, -h - 0.004), (r * 0.16, h + 0.004), (0, h + 0.004)], m["accent"], "Y")
        for k in range(20):                               # tread lugs and wheel bolts: make the rotation visible
            ang = 360.0 * k / 20
            c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            box(stage, f"{W}/lug_{k}", (0.012, ww - 0.022, 0.007), (s * r, 0, c * r), m["rubber"], _qy(ang))
        for k in range(5):
            ang = np.deg2rad(72 * k)
            lathe(stage, f"{W}/bolt_{k}", [(0, -h * 0.95), (0.006, -h * 0.95), (0.006, h * 0.95), (0, h * 0.95)], m["steel"], "Y",
                  (np.sin(ang) * r * 0.41, 0, np.cos(ang) * r * 0.41), n=12)


def dress_platform(stage, cfg, m):
    s = cfg["platform"]["side"]
    base = "/World/Platform/base"
    V = base + "/vis"
    UsdGeom.Xform.Define(stage, V)
    h = s / 2
    k = 0
    for ax in range(3):                                   # aluminium edge frame
        u, v = [(1, 2), (2, 0), (0, 1)][ax]
        for su in (-1, 1):
            for sv in (-1, 1):
                p0, p1 = np.zeros(3), np.zeros(3)
                p0[u] = p1[u] = su * h
                p0[v] = p1[v] = sv * h
                p0[ax], p1[ax] = -h, h
                _tube(stage, f"{V}/edge_{k}", p0, p1, 0.009, m["alu"])
                k += 1
    for i, sx in enumerate((-1, 1)):                      # side panels with an accent line
        box(stage, f"{V}/line_x{i}", (0.002, s * 0.8, 0.012), (sx * (h + 0.0005), 0, -h * 0.55), m["accent"])
        box(stage, f"{V}/line_y{i}", (s * 0.8, 0.002, 0.012), (0, sx * (h + 0.0005), -h * 0.55), m["accent"])


def dress_gripper(stage, cfg, m):
    g = cfg["gripper"]
    w, t, f = g["wrist"], g["tool"], g["finger"]
    R = "/World/Platform"
    # wrist actuator
    _hide(stage, R + "/wrist", keep=("index_mark",))
    V = R + "/wrist/vis"
    UsdGeom.Xform.Define(stage, V)
    hw, rw = w["height"] / 2, w["radius"]
    lathe(stage, V + "/housing", [(0, hw), (rw * 1.08, hw), (rw * 1.08, hw - 0.008), (rw * 0.96, hw - 0.008), (rw * 0.96, -hw + 0.012),
                                  (rw * 1.04, -hw + 0.012), (rw * 1.04, -hw), (0, -hw)], m["graphite"])
    lathe(stage, V + "/ring", [(rw * 0.965, 0.006), (rw * 0.985, 0.006), (rw * 0.985, 0.018), (rw * 0.965, 0.018)], m["accent"])
    for k in range(8):
        a = np.deg2rad(45 * k + 22.5)
        lathe(stage, f"{V}/bolt_{k}", [(0, -hw - 0.003), (0.004, -hw - 0.003), (0.004, -hw + 0.001), (0, -hw + 0.001)], m["steel"],
              pos=(np.cos(a) * rw * 0.82, np.sin(a) * rw * 0.82, 0), n=10)
    box(stage, V + "/connector", (0.022, 0.03, 0.02), (-rw * 0.98, 0, 0.005), m["steel"])
    # load cell, tube, palm
    _hide(stage, R + "/tool")
    V = R + "/tool/vis"
    UsdGeom.Xform.Define(stage, V)
    L, rt = t["tube_length"], t["tube_radius"]
    z1 = L / 2
    lathe(stage, V + "/ft", [(0, z1), (0.04, z1), (0.04, z1 - 0.007), (0.037, z1 - 0.009), (0.037, z1 - 0.021), (0.04, z1 - 0.023),
                             (0.04, z1 - 0.03), (0, z1 - 0.03)], m["alu"])
    lathe(stage, V + "/ft_band", [(0.0375, z1 - 0.011), (0.0385, z1 - 0.011), (0.0385, z1 - 0.019), (0.0375, z1 - 0.019)], m["accent"])
    box(stage, V + "/ft_plug", (0.018, 0.02, 0.014), (-0.044, 0, z1 - 0.015), m["graphite"])
    palm_z = -(L / 2 + t["palm"][2] / 2)
    lathe(stage, V + "/tube", [(rt, z1 - 0.03), (rt, palm_z + t["palm"][2] / 2)], m["carbon"])
    for i, z in enumerate((z1 - 0.036, palm_z + t["palm"][2] / 2 + 0.012)):
        lathe(stage, f"{V}/collar_{i}", [(rt, z - 0.01), (rt + 0.006, z - 0.01), (rt + 0.006, z + 0.01), (rt, z + 0.01)], m["alu"])
    px, py, pz = t["palm"]
    rbox(stage, V + "/palm", (px, py, pz), 0.006, m["graphite"], (0, 0, palm_z))
    for i, sx in enumerate((-1, 1)):                      # guide rails the jaws run on
        box(stage, f"{V}/rail_{i}", (0.005, py * 0.96, 0.012), (sx * (px / 2 + 0.0015), 0, palm_z - 0.006), m["alu"])
    box(stage, V + "/drive", (0.034, 0.09, 0.034), (-px / 2 - 0.017, 0, palm_z + 0.004), m["graphite"])
    box(stage, V + "/drive_label", (0.001, 0.06, 0.012), (-px / 2 - 0.0345, 0, palm_z + 0.006), m["accent"])
    _tube(stage, V + "/cable_loom", (-0.03, 0.0, z1 - 0.02), (-px / 2 - 0.017, 0.0, palm_z + 0.021), 0.004, m["rubber"])
    # jaws
    sx, sy, sz = f["size"]
    for side, name in ((1, "l"), (-1, "r")):
        F = f"{R}/finger_{name}"
        UsdGeom.Imageable(stage.GetPrimAtPath(F + "/body")).MakeInvisible()
        UsdGeom.Imageable(stage.GetPrimAtPath(F + "/pad")).MakeInvisible()
        V = F + "/vis"
        UsdGeom.Xform.Define(stage, V)
        rbox(stage, V + "/jaw", (sx, sy, sz), 0.004, m["alu"])
        box(stage, V + "/carriage", (sx + 0.016, sy + 0.008, 0.02), (0, 0, sz / 2 - 0.01), m["graphite"])
        box(stage, V + "/pad", (sx * 0.9, 0.004, sz * 0.7), (0, -side * (sy / 2 + 0.002), -sz * 0.1), m["rubber"])
        for k in range(7):                                # grip ribs on the pad
            box(stage, f"{V}/rib_{k}", (sx * 0.9, 0.0016, 0.004), (0, -side * (sy / 2 + 0.0045), -sz * 0.1 + (k - 3) * 0.011), m["rubber"])
        for k, z in enumerate((-0.02, 0.012)):            # lightening slots and screws on the outer face
            box(stage, f"{V}/slot_{k}", (sx * 0.5, 0.001, 0.016), (0, side * (sy / 2 + 0.0003), z), m["graphite"])
        for k, x in enumerate((-0.018, 0.018)):
            lathe(stage, f"{V}/screw_{k}", [(0, 0), (0.004, 0), (0.004, 0.002), (0, 0.002)], m["steel"], "Y",
                  (x, side * (sy / 2) - (0.002 if side < 0 else 0.0), sz / 2 - 0.022), n=10)


def dress_blocks(stage, cfg):
    """Masonry look for every hollow block: the five collision boxes are redrawn with rounded arrises and a
    concrete texture, and each block gets one of a few slightly different shades so courses read as separate units."""
    from .blocks import BLOCK_ROOT, block_parts
    from .site import pbr_material

    tints = [(0.93, 0.92, 0.90), (0.86, 0.855, 0.84), (0.98, 0.965, 0.94), (0.89, 0.875, 0.85), (0.81, 0.805, 0.795), (0.95, 0.93, 0.89)]
    mats = [pbr_material(stage, f"/World/Looks/dress_block_{i}", "rough_concrete", res="1k", tint=t) for i, t in enumerate(tints)]
    rng = np.random.default_rng(12)
    parts = block_parts(cfg)
    for prim in stage.GetPrimAtPath(BLOCK_ROOT).GetChildren():
        path = prim.GetPath().pathString
        _hide(stage, path)
        mat = mats[int(rng.integers(len(mats)))]
        uv0 = rng.uniform(0, 1, 2)
        UsdGeom.Xform.Define(stage, path + "/vis")
        for name, size, c in parts:
            rbox(stage, f"{path}/vis/{name}", size, 0.005, mat, c, n=4, tile=0.45, uv0=uv0)
    return mats


def dress_all(stage, cfg, gripper=True, blocks=True):
    """gripper=False leaves the hanging tool for a process end-effector (cdpr_sim/process_tools.py)."""
    m = materials(stage)
    for k in range(4):
        dress_drone(stage, cfg, f"/World/Drone_{k}", m)
        dress_ugv(stage, cfg, f"/World/UGV_{k}", m)
    dress_platform(stage, cfg, m)
    if gripper:
        dress_gripper(stage, cfg, m)
    if blocks and stage.GetPrimAtPath("/World/Blocks"):
        dress_blocks(stage, cfg)
