"""USD scene construction (pure pxr): physics scene, environment, platform with AprilTags, drones, UGV articulations.

Everything is built procedurally so the sim has no dependency on the (remote) Isaac asset library.
"""

import os

import cv2
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from .cdpr_control import corner_points, formation

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


# ----------------------------------------------------------------------------------------------- textures

def make_textures():
    os.makedirs(ASSET_DIR, exist_ok=True)
    paths = {}
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id in range(5):
        cell = 100
        img = cv2.aruco.generateImageMarker(d, tag_id, 8 * cell, borderBits=1)  # 8x8 cells incl. black border
        img = cv2.copyMakeBorder(img, cell, cell, cell, cell, cv2.BORDER_CONSTANT, value=255)  # 1-cell white quiet zone
        p = os.path.join(ASSET_DIR, f"tag36h11_{tag_id}.png")
        if not os.path.exists(p):
            cv2.imwrite(p, img)
        paths[f"tag{tag_id}"] = p
    # plain white floor tiles (60 cm, light grey grout) - one tile per texture repeat
    n = 512
    tile = np.empty((n, n, 3), np.uint8)
    tile[:] = (108, 110, 114)                                      # BGR: warm mid grey (~0.43 albedo)
    tile += np.random.default_rng(1).integers(0, 5, (n, n, 1)).astype(np.uint8)   # faint surface variation
    g = 5
    tile[:g, :] = tile[-g:, :] = tile[:, :g] = tile[:, -g:] = (70, 72, 75)       # darker grout
    p = os.path.join(ASSET_DIR, "white_tile.png")
    if not os.path.exists(p):
        cv2.imwrite(p, tile)
    paths["white_tile"] = p
    # feature-rich ground / wall textures for visual odometry (multi-scale noise + speckles), deterministic
    rng = np.random.default_rng(7)
    for name, base in (("ground", (120, 118, 112)), ("wall", (150, 152, 158)), ("crate", (60, 100, 140)),
                       ("site", (138, 136, 130)), ("block", (168, 167, 162))):
        n = 2048 if name in ("ground", "wall", "site") else 512
        tex = np.zeros((n, n), np.float32)
        for scale, amp in ((8, 40), (32, 25), (128, 15), (512, 10)):
            tex += cv2.resize(rng.normal(0, amp, (scale, scale)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        img = np.clip(np.stack([tex + c for c in base], -1), 0, 255).astype(np.uint8)
        n_dots = {"ground": 4000, "wall": 4000, "crate": 300, "site": 1500, "block": 900}[name]
        if name in ("site", "block"):  # fine pores / aggregate, low contrast (concrete-like, still trackable)
            tex *= 0.5
        for _ in range(n_dots):
            c = tuple(int(v) for v in rng.integers(20, 235, 3)) if name not in ("site", "block") else \
                tuple(int(v) for v in np.clip(np.array(base) + rng.integers(-45, 30), 0, 255))
            cv2.circle(img, tuple(int(v) for v in rng.integers(0, n, 2)), int(rng.integers(2, 9) * n / 2048 + 1), c, -1)
        p = os.path.join(ASSET_DIR, f"{name}.png")
        if not os.path.exists(p):
            cv2.imwrite(p, img)
        paths[name] = p
    # fired-clay brick for the video renders: terracotta with faint firing variation and fine pores
    rng = np.random.default_rng(21)
    n = 512
    v = np.zeros((n, n), np.float32)
    for scale, amp in ((4, 9), (16, 6), (128, 5), (512, 6)):
        v += cv2.resize(rng.normal(0, amp, (scale, scale)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
    img = np.clip(np.stack([v * 0.6 + 58, v * 0.8 + 84, v + 158], -1), 0, 255).astype(np.uint8)   # BGR
    for _ in range(500):
        c = int(rng.integers(-30, 12))
        cv2.circle(img, tuple(int(q) for q in rng.integers(0, n, 2)), int(rng.integers(1, 3)),
                   (58 + c // 2, 84 + c, 158 + c), -1)
    p = os.path.join(ASSET_DIR, "brick.png")
    if not os.path.exists(p):
        cv2.imwrite(p, img)
    paths["brick"] = p
    return paths


# ----------------------------------------------------------------------------------------------- helpers

def _xform(stage, path, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    x = UsdGeom.Xform.Define(stage, path)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
    return x


def _material(stage, path, color, roughness=0.6, metallic=0.0, texture=None, tex_scale=1.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(texture))
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _phys_material(stage, path, static, dynamic, restitution=0.0):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(restitution)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr("min")
    return mat


def _bind_phys(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), mat=None, collide=False):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.ClearXformOpOrder()
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    c.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if mat:
        _bind(c.GetPrim(), mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def _uv_box(stage, path, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), mat=None, collide=False, tile=1.0):
    """Box mesh with per-face UVs scaled to metres / `tile`, so textures keep a constant physical scale."""
    hx, hy, hz = (v / 2.0 for v in size)
    faces = [  # (normal, u-axis, v-axis, half extents along n,u,v)
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), hx, hy, hz), ((-1, 0, 0), (0, -1, 0), (0, 0, 1), hx, hy, hz),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1), hy, hx, hz), ((0, -1, 0), (1, 0, 0), (0, 0, 1), hy, hx, hz),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0), hz, hx, hy), ((0, 0, -1), (-1, 0, 0), (0, 1, 0), hz, hx, hy),
    ]
    pts, nrm, uv = [], [], []
    for n, a, b, hn, ha, hb in faces:
        n, a, b = np.array(n, float), np.array(a, float), np.array(b, float)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pts.append(n * hn + a * su * ha + b * sv * hb)
            nrm.append(n)
            uv.append(((su + 1) * ha / tile, (sv + 1) * hb / tile))
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr(list(range(24)))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in nrm]))
    m.SetNormalsInterpolation("faceVarying")
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in uv]))
    m.CreateSubdivisionSchemeAttr("none")
    m.ClearXformOpOrder()
    m.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    m.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
    if mat:
        _bind(m.GetPrim(), mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    return m


def _cyl(stage, path, radius, height, axis="Z", pos=(0, 0, 0), mat=None):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    c.ClearXformOpOrder()
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if mat:
        _bind(c.GetPrim(), mat)
    return c


def _sphere(stage, path, radius, pos=(0, 0, 0), mat=None, collide=False, visible=True):
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(radius)
    s.ClearXformOpOrder()
    s.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if mat:
        _bind(s.GetPrim(), mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(s.GetPrim())
    if not visible:
        UsdGeom.Imageable(s).MakeInvisible()
    return s


def _quad(stage, path, size, z_offset, flip, mat, uv_repeat=1.0):
    """Square textured quad in the parent's xy-plane at height z_offset; flip=True faces -z."""
    h = size / 2.0
    m = UsdGeom.Mesh.Define(stage, path)
    pts = [(-h, -h, z_offset), (h, -h, z_offset), (h, h, z_offset), (-h, h, z_offset)]
    uv = [(0, 0), (uv_repeat, 0), (uv_repeat, uv_repeat), (0, uv_repeat)]
    idx = [0, 1, 2, 3]
    if flip:
        # seen from below: mirror x so the tag reads correctly (not mirrored) from underneath
        pts = [(h, -h, z_offset), (-h, -h, z_offset), (-h, h, z_offset), (h, h, z_offset)]
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, -1 if flip else 1)] * 4))
    m.SetNormalsInterpolation("vertex")
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in uv]))
    m.CreateDoubleSidedAttr(False)
    _bind(m.GetPrim(), mat)
    return m


def _rigid(prim, mass, inertia, com=(0, 0, 0)):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    px.CreateSleepThresholdAttr(0.0)  # never sleep: cable forces are external and small
    px.CreateLinearDampingAttr(0.0)
    px.CreateAngularDampingAttr(0.0)
    px.CreateMaxDepenetrationVelocityAttr(1.0)


def _box_inertia(m, s):
    x, y, z = s
    return (m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12)


# ----------------------------------------------------------------------------------------------- builders

def build_scene(stage, cfg):
    tex = make_textures()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    _xform(stage, "/World")

    # physics scene
    sc = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr(cfg["sim"]["gravity"])
    px = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim())
    px.CreateTimeStepsPerSecondAttr(int(round(1.0 / cfg["sim"]["physics_dt"])))
    px.CreateSolverTypeAttr(cfg["sim"]["solver"])
    px.CreateEnableGPUDynamicsAttr(cfg["sim"]["device"] != "cpu")
    px.CreateBroadphaseTypeAttr("MBP" if cfg["sim"]["device"] == "cpu" else "GPU")
    px.CreateEnableStabilizationAttr(False)
    px.CreateEnableCCDAttr(False)

    mats = {
        "ground": _material(stage, "/World/Looks/ground", (0.5, 0.5, 0.5), 0.8, texture=tex["ground"]),
        "wall": _material(stage, "/World/Looks/wall", (0.6, 0.6, 0.6), 0.8, texture=tex["wall"]),
        "tag0": _material(stage, "/World/Looks/tag0", (1, 1, 1), 0.7, texture=tex["tag0"]),
        "site": _material(stage, "/World/Looks/site", (0.55, 0.55, 0.53), 0.9, texture=tex["site"]),
        "tagb0": _material(stage, "/World/Looks/tagb0", (1, 1, 1), 0.7, texture=tex["tag1"]),
        "tagb1": _material(stage, "/World/Looks/tagb1", (1, 1, 1), 0.7, texture=tex["tag2"]),
        "tagb2": _material(stage, "/World/Looks/tagb2", (1, 1, 1), 0.7, texture=tex["tag3"]),
        "tagb3": _material(stage, "/World/Looks/tagb3", (1, 1, 1), 0.7, texture=tex["tag4"]),
        "alu": _material(stage, "/World/Looks/alu", (0.75, 0.76, 0.78), 0.35, 0.8),
        "graphite": _material(stage, "/World/Looks/graphite", (0.13, 0.135, 0.145), 0.42, 0.25),
        "accent": _material(stage, "/World/Looks/accent", (1.0, 0.36, 0.06), 0.45),
        "dark": _material(stage, "/World/Looks/dark", (0.08, 0.08, 0.09), 0.5),
        "rubber": _material(stage, "/World/Looks/rubber", (0.03, 0.03, 0.03), 0.9),
        "orange": _material(stage, "/World/Looks/orange", (0.95, 0.45, 0.05), 0.5),
        "blue": _material(stage, "/World/Looks/blue", (0.10, 0.30, 0.80), 0.5),
        "prop": _material(stage, "/World/Looks/prop", (0.15, 0.15, 0.15), 0.4),
        "cable": _material(stage, "/World/Looks/cable", (0.98, 0.80, 0.12), 0.55),
        "crate_tex": _material(stage, "/World/Looks/crate", (0.55, 0.40, 0.22), 0.8, texture=tex["crate"]),
    }
    pm_ground = _phys_material(stage, "/World/PhysicsMaterials/ground", **cfg["ground"]["friction"])
    pm_tire = _phys_material(stage, "/World/PhysicsMaterials/tire", **cfg["ugv"]["friction"])
    pm_caster = _phys_material(stage, "/World/PhysicsMaterials/caster", **cfg["ugv"]["caster_friction"])

    if cfg["environment"]["type"] == "tiles":
        _build_tiles(stage, cfg, tex, pm_ground)
    elif cfg["environment"]["type"] == "lab":
        _build_environment(stage, cfg, mats, pm_ground)
    elif cfg["environment"]["type"] == "hangar":
        from .hangar import build_hangar

        build_hangar(stage, cfg, pm_ground)
    elif cfg["environment"]["type"] == "sand":
        from .sandsite import build_sand

        build_sand(stage, cfg, pm_ground, tex)
    else:
        from .site import build_site  # local import: site.py uses the helpers defined in this module

        build_site(stage, cfg, pm_ground)
    from .tool import build_platform  # local import: tool.py uses the helpers defined in this module

    info = {"tool": build_platform(stage, cfg, mats), "textures": tex}
    drones, ugvs = formation(cfg, np.array(cfg["platform"]["start_position"][:2]))
    att = np.array(cfg["drone"]["attach_offset"])
    for k in range(4):
        _build_drone(stage, cfg, f"/World/Drone_{k}", drones[k] - att, mats, k)
    for k in range(4):
        p = ugvs[k]
        yaw = np.arctan2(p[1], p[0]) + np.pi / 2  # tangential heading
        _build_ugv(stage, cfg, f"/World/UGV_{k}", p[:2], yaw, mats, pm_tire, pm_caster)
    _build_cables(stage, mats)
    return info


def _build_tiles(stage, cfg, tex, pm_ground):
    """Light default scene: plain white tiled floor, uniform sky light and a sun."""
    size = cfg["ground"]["size"]
    _xform(stage, "/World/Environment")
    mat = _material(stage, "/World/Looks/floor_tile", (0.45, 0.44, 0.42), 0.38, texture=tex["white_tile"])
    _quad(stage, "/World/Environment/Floor", size, 0.0, False, mat, uv_repeat=size / 0.6)
    plane = UsdGeom.Plane.Define(stage, "/World/Environment/GroundCollider")
    plane.CreateAxisAttr("Z")
    plane.CreateWidthAttr(size)
    plane.CreateLengthAttr(size)
    UsdGeom.Imageable(plane).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    _bind_phys(plane.GetPrim(), pm_ground)
    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/Sky")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateTextureFileAttr(_studio_gradient())
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    sun = UsdLux.DistantLight.Define(stage, "/World/Environment/Sun")
    sun.CreateIntensityAttr(2200.0)
    sun.CreateAngleAttr(2.0)                       # slightly soft key light
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
    _xform_rot(sun, (-50.0, 30.0, 0.0))


def _studio_gradient(width=1024):
    """Generated latlong backdrop: slate blue overhead fading to a light blue-grey horizon, grey below."""
    p = os.path.join(ASSET_DIR, "studio_gradient.hdr")
    if not os.path.exists(p):
        h = width // 2
        el = (0.5 - (np.arange(h) + 0.5) / h) * np.pi           # +pi/2 at the top row
        t = np.clip(np.sin(np.maximum(el, 0)), 0, 1) ** 0.6
        zen, hor = np.array([0.20, 0.30, 0.46]), np.array([0.70, 0.75, 0.80])
        rows = hor[None] * (1 - t[:, None]) + zen[None] * t[:, None]
        rows[el < 0] = np.array([0.30, 0.30, 0.30])
        img = np.repeat(rows[:, None, :], width, axis=1).astype(np.float32)
        cv2.imwrite(p, img[..., ::-1])
    return os.path.abspath(p)


def _build_environment(stage, cfg, mats, pm_ground):
    size = cfg["ground"]["size"]
    _xform(stage, "/World/Environment")
    g = _quad(stage, "/World/Environment/GroundVisual", size, 0.0, False, mats["ground"], uv_repeat=size / 8.0)
    plane = UsdGeom.Plane.Define(stage, "/World/Environment/GroundCollider")
    plane.CreateAxisAttr("Z")
    plane.CreateWidthAttr(size)
    plane.CreateLengthAttr(size)
    UsdGeom.Imageable(plane).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    _bind_phys(plane.GetPrim(), pm_ground)
    # 16 m x 16 m lab: textured walls + some crates and pillars (lidar returns, VO features, occluders)
    half, hgt, th = 8.0, 3.0, 0.2
    for i, (pos, sz) in enumerate([((half, 0, hgt / 2), (th, 2 * half, hgt)), ((-half, 0, hgt / 2), (th, 2 * half, hgt)),
                                   ((0, half, hgt / 2), (2 * half, th, hgt)), ((0, -half, hgt / 2), (2 * half, th, hgt))]):
        _uv_box(stage, f"/World/Environment/Wall_{i}", sz, pos, mat=mats["wall"], collide=True, tile=4.0)
    rng = np.random.default_rng(3)
    for i in range(10):
        r = rng.uniform(4.0, 7.0)
        a = rng.uniform(0, 2 * np.pi)
        s = rng.uniform(0.3, 0.8)
        _uv_box(stage, f"/World/Environment/Crate_{i}", (s, s, s), (r * np.cos(a), r * np.sin(a), s / 2),
                quat=(np.cos(a / 2), 0, 0, np.sin(a / 2)), mat=mats["crate_tex"], collide=True, tile=1.0)
    for i, (x, y) in enumerate([(5.5, 5.5), (-5.5, 5.5), (-5.5, -5.5), (5.5, -5.5)]):
        c = _cyl(stage, f"/World/Environment/Pillar_{i}", 0.2, 3.0, pos=(x, y, 1.5), mat=mats["wall"])
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    # lights
    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/Dome")
    dome.CreateIntensityAttr(600.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Environment/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    _xform_rot(sun, (-35.0, 25.0, 0.0))


def _xform_rot(prim, rot_xyz_deg):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz_deg))


def _build_drone(stage, cfg, path, pos, mats, k):
    d = cfg["drone"]
    root = _xform(stage, path, pos)
    _rigid(root.GetPrim(), d["mass"], d["inertia"])
    bs = d["body_size"]
    _box(stage, path + "/body", bs, mat=mats["graphite"], collide=True)
    _box(stage, path + "/battery", (bs[0] * 0.8, bs[1] * 0.5, 0.04), (0, 0, bs[2] / 2 + 0.02), mat=mats["accent"])
    a = d["arm_length"]
    for i in range(4):
        ang = np.deg2rad(45 + 90 * i)
        c, sn = np.cos(ang), np.sin(ang)
        _box(stage, f"{path}/arm_{i}", (a, 0.025, 0.02), (c * a / 2, sn * a / 2, 0),
             quat=(np.cos(ang / 2), 0, 0, np.sin(ang / 2)), mat=mats["dark"])
        _cyl(stage, f"{path}/motor_{i}", 0.018, 0.035, pos=(c * a, sn * a, 0.02), mat=mats["alu"])
        disc = _cyl(stage, f"{path}/prop_{i}", 0.127, 0.004, pos=(c * a, sn * a, 0.04), mat=mats["prop"])
        UsdGeom.Gprim(disc).CreateDisplayOpacityAttr([0.35])
    for i in range(2):  # landing skids
        y = (-1) ** i * 0.08
        _box(stage, f"{path}/skid_{i}", (0.20, 0.012, 0.012), (0, y, -0.12), mat=mats["dark"])
        _box(stage, f"{path}/strut_{i}", (0.012, 0.012, 0.08), (0, y, -0.08), mat=mats["dark"])
    # winch + fairlead under the body
    _cyl(stage, path + "/winch", 0.03, 0.06, axis="Y", pos=(0, 0, -0.05), mat=mats["orange"])
    _sphere(stage, path + "/fairlead", 0.008, d["attach_offset"], mats["alu"])


def _build_ugv(stage, cfg, path, xy, yaw, mats, pm_tire, pm_caster):
    u = cfg["ugv"]
    r, wwid = u["wheel_radius"], u["wheel_width"]
    cs = u["chassis_size"]
    cz = u["ground_clearance"] + cs[2] / 2.0  # chassis centre height
    q = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    root = _xform(stage, path)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    pxa = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    pxa.CreateEnabledSelfCollisionsAttr(False)
    pxa.CreateSolverPositionIterationCountAttr(16)
    pxa.CreateSolverVelocityIterationCountAttr(4)
    pxa.CreateSleepThresholdAttr(0.0)

    ch_path = path + "/chassis"
    ch = _xform(stage, ch_path, (xy[0], xy[1], cz), q)
    _rigid(ch.GetPrim(), u["chassis_mass"], _box_inertia(u["chassis_mass"], cs))
    _box(stage, ch_path + "/hull", cs, mat=mats["graphite"], collide=True)
    for side in (1, -1):   # accent stripes along both flanks
        _box(stage, f"{ch_path}/stripe_{'l' if side > 0 else 'r'}", (cs[0] * 0.92, 0.004, 0.035),
             (0, side * (cs[1] / 2 + 0.002), cs[2] * 0.18), mat=mats["accent"])
    _box(stage, ch_path + "/deck", (cs[0] * 0.9, cs[1] * 0.9, 0.01), (0, 0, cs[2] / 2 + 0.005), mat=mats["dark"])
    mast_h = u["attach_height"] - cz
    _cyl(stage, ch_path + "/mast", 0.02, mast_h - cs[2] / 2, pos=(0, 0, cs[2] / 2 + (mast_h - cs[2] / 2) / 2), mat=mats["alu"])
    _cyl(stage, ch_path + "/winch", 0.04, 0.08, axis="Y", pos=(-0.12, 0, cs[2] / 2 + 0.05), mat=mats["orange"])
    _sphere(stage, ch_path + "/fairlead", 0.012, (0, 0, mast_h), mats["alu"])
    cr = u["caster_radius"]
    for i, sx in enumerate((1, -1)):
        c = _sphere(stage, f"{ch_path}/caster_{i}", cr, (sx * u["caster_offset_x"], 0, cr - cz + u.get("caster_lift", 0.0)), mats["alu"], collide=True)
        _bind_phys(c.GetPrim(), pm_caster)

    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    for side, name in ((1, "left"), (-1, "right")):
        local = np.array([0.0, side * u["track_width"] / 2, r - cz])
        wpos = np.array([xy[0], xy[1], cz]) + R @ local
        wp = f"{path}/wheel_{name}"
        w = _xform(stage, wp, wpos, q)
        mw = u["wheel_mass"]
        _rigid(w.GetPrim(), mw, (mw * (3 * r * r + wwid * wwid) / 12, mw * r * r / 2, mw * (3 * r * r + wwid * wwid) / 12))
        _cyl(stage, wp + "/tire", r, wwid, axis="Y", mat=mats["rubber"])
        _cyl(stage, wp + "/hub", r * 0.5, wwid * 1.05, axis="Y", mat=mats["alu"])
        col = _sphere(stage, wp + "/collider", r, collide=True, visible=False)
        _bind_phys(col.GetPrim(), pm_tire)
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/joint_wheel_{name}")
        j.CreateBody0Rel().SetTargets([ch_path])
        j.CreateBody1Rel().SetTargets([wp])
        j.CreateAxisAttr("Y")
        j.CreateLocalPos0Attr(Gf.Vec3f(*local))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(0.0)
        drv.CreateDampingAttr(float(u["wheel_drive_damping"] * np.pi / 180.0))  # USD angular drive units: per degree
        drv.CreateMaxForceAttr(float(u["wheel_max_torque"]))
        drv.CreateTargetVelocityAttr(0.0)
        PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateMaxJointVelocityAttr(float(np.rad2deg(u["wheel_max_speed"] * 1.5)))


def _build_cables(stage, mats):
    curves = UsdGeom.BasisCurves.Define(stage, "/World/Cables")
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([2] * 8)
    curves.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0)] * 16))
    curves.CreateWidthsAttr(Vt.FloatArray([0.006] * 16))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    _bind(curves.GetPrim(), mats["cable"])
