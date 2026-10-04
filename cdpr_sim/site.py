"""Construction-site environment built from 3D assets (no photographic backdrop).

Assets: Poly Haven CC0 models/textures (assets/polyhaven, fetched by tools/fetch_assets.py), NVIDIA Isaac/ArchVis
assets from the Isaac asset server (forklift, pallets, trees, procedural "Dynamic" sky), and procedural geometry
(building frame under construction, scaffolding, site office container).

Layout (world frame, metres). The robots' working area must stay empty:
    clear zone      x in [-4.5, 8.0], y in [-4.5, 4.5]   (build centre (0,0), block staging (3.2,0), formation r 2.75)
    site fence      x in [-9, 14],   y in [-9, 9], gate on the east side
    neighbouring plot (building under construction) north of the fence, y in [11, 19]
Background objects have no collision shapes: the robots never reach them, and RTX cameras/lidars see render
geometry anyway.
"""

import os

import carb
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .scene import _bind, _bind_phys, _quad, _uv_box, _xform

PH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "polyhaven"))


# ------------------------------------------------------------------------------------------------ materials

def pbr_material(stage, path, name, res="2k", tint=(1.0, 1.0, 1.0), metallic=0.0):
    """UsdPreviewSurface with diffuse + OpenGL normal + roughness maps from assets/polyhaven/<name>/."""
    d = os.path.join(PH, name)
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    st = UsdShade.Shader.Define(stage, path + "/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def tex(kind, cs, out, out_type):
        t = UsdShade.Shader.Define(stage, f"{path}/{kind}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.join(d, f"{name}_{kind}_{res}.jpg"))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
        t.CreateOutput(out, out_type)
        return t

    diff = tex("diff", "sRGB", "rgb", Sdf.ValueTypeNames.Float3)
    diff.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1.0))
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diff.ConnectableAPI(), "rgb")
    nrm = tex("nor_gl", "raw", "rgb", Sdf.ValueTypeNames.Float3)
    nrm.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
    nrm.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
    sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nrm.ConnectableAPI(), "rgb")
    rough = tex("rough", "raw", "r", Sdf.ValueTypeNames.Float)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(rough.ConnectableAPI(), "r")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def flat_material(stage, path, color, roughness=0.5, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


# ------------------------------------------------------------------------------------------------ helpers

def _place(prim, pos, yaw_deg=0.0, scale=1.0):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddRotateZOp().Set(float(yaw_deg))
    if scale != 1.0:
        x.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))


def _ref(stage, path, asset, pos, yaw_deg=0.0, scale=1.0, keep=None):
    """Reference an asset file under `path`; `keep` = names of the default prim's children to keep active."""
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(asset)
    _place(prim, pos, yaw_deg, scale)
    if keep is not None:
        for c in prim.GetChildren():
            if c.GetName() not in keep and not c.GetName().startswith("_"):
                c.SetActive(False)
    return prim


def _ph(name):
    return os.path.join(PH, "models", name, f"{name}_1k.usdc")


REMOTE = {"enabled": False}


def _isaac(rel):
    if not REMOTE["enabled"]:
        return None
    try:
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
    except Exception:
        root = None
    if not root:
        carb.log_warn(f"Isaac asset server unavailable, skipping {rel}")
        return None
    return root + rel


def _box(stage, path, size, pos, mat, yaw_deg=0.0, tile=1.0):
    q = (np.cos(np.deg2rad(yaw_deg) / 2), 0, 0, np.sin(np.deg2rad(yaw_deg) / 2))
    return _uv_box(stage, path, size, pos, q, mat=mat, tile=tile)


def _tube(stage, path, p0, p1, r, mat):
    """Cylinder between two points (scaffold tubes)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(r)
    c.CreateHeightAttr(L)
    c.CreateAxisAttr("Z")
    z = d / L
    a = np.cross([0, 0, 1.0], z)
    s = np.linalg.norm(a)
    ang = np.arctan2(s, z[2])
    q = Gf.Quatf(1, 0, 0, 0) if s < 1e-9 else Gf.Quatf(float(np.cos(ang / 2)), *map(float, a / s * np.sin(ang / 2)))
    c.ClearXformOpOrder()
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, (p0 + p1) / 2)))
    c.AddOrientOp().Set(q)
    _bind(c.GetPrim(), mat)


# ------------------------------------------------------------------------------------------------ builders

def build_site(stage, cfg, pm_ground):
    E = "/World/Environment"
    REMOTE["enabled"] = cfg["environment"].get("remote_assets", False)
    _xform(stage, E)
    m = {
        "ground": pbr_material(stage, E + "/Looks/dry_ground", "dry_ground_01"),
        "pad": pbr_material(stage, E + "/Looks/concrete_pad", "concrete_floor_02"),
        "concrete": pbr_material(stage, E + "/Looks/concrete_frame", "concrete_slab_wall", res="1k"),
        "corrugated": pbr_material(stage, E + "/Looks/corrugated", "corrugated_iron", res="1k", tint=(0.55, 0.75, 0.95)),
        "steel": flat_material(stage, E + "/Looks/galvanised", (0.62, 0.63, 0.64), 0.35, 0.9),
        "plank": flat_material(stage, E + "/Looks/plank", (0.55, 0.42, 0.26), 0.8),
        "dark": flat_material(stage, E + "/Looks/window", (0.05, 0.06, 0.07), 0.15, 0.2),
        "block": flat_material(stage, E + "/Looks/infill_block", (0.66, 0.65, 0.62), 0.9),
        "paint": flat_material(stage, E + "/Looks/line_paint", (0.95, 0.80, 0.10), 0.6),
    }
    _ground(stage, cfg, E, m, pm_ground)
    _sky(stage, E, cfg)
    _fence(stage, E + "/Fence")
    _building(stage, E + "/Building", m)
    _container(stage, E + "/SiteOffice", m)
    _yard(stage, E + "/Yard", m)
    _props(stage, E + "/Props")
    if cfg["environment"].get("trees", False):
        _trees(stage, E + "/Trees")


def _ground(stage, cfg, E, m, pm_ground):
    _quad(stage, E + "/Ground", 300.0, 0.0, False, m["ground"], uv_repeat=300.0 / 6.0)
    pad = _quad(stage, E + "/ConcretePad", 26.0, 0.004, False, m["pad"], uv_repeat=26.0 / 3.0)
    UsdGeom.XformCommonAPI(pad).SetTranslate(Gf.Vec3d(2.5, 0, 0))
    # painted work-area outline around the robots' clear zone (as on real sites)
    for i, (c, s) in enumerate((((1.75, 4.6), (12.7, 0.08)), ((1.75, -4.6), (12.7, 0.08)), ((-4.6, 0), (0.08, 9.2)), ((8.1, 0), (0.08, 9.2)))):
        _box(stage, f"{E}/Marking_{i}", (s[0], s[1], 0.002), (c[0], c[1], 0.006), m["paint"])
    plane = UsdGeom.Plane.Define(stage, E + "/GroundCollider")
    plane.CreateAxisAttr("Z")
    plane.CreateWidthAttr(300.0)
    plane.CreateLengthAttr(300.0)
    UsdGeom.Imageable(plane).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    _bind_phys(plane.GetPrim(), pm_ground)


def preetham_sky(path, sun_elev_deg, sun_az_deg, turbidity=2.5, width=2048):
    """Analytic daylight sky (Preetham, Shirley & Smits 1999) as an equirectangular HDR (latlong, z up).
    Radiance is normalised to 1 at the zenith; below the horizon a dim ground colour."""
    import cv2

    T = turbidity
    ts = np.deg2rad(90.0 - sun_elev_deg)
    h = width // 2
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(h) + 0.5) / h
    az = (u - 0.5) * 2 * np.pi                      # longitude
    el = (0.5 - v) * np.pi                          # latitude, +pi/2 at the top row
    AZ, EL = np.meshgrid(az, el)
    dx, dy, dz = np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)
    sa = np.deg2rad(sun_az_deg)
    s = np.array([np.sin(ts) * np.cos(sa), np.sin(ts) * np.sin(sa), np.cos(ts)])
    theta = np.arccos(np.clip(dz, 0.01, 1.0))
    gamma = np.arccos(np.clip(dx * s[0] + dy * s[1] + dz * s[2], -1, 1))

    def perez(A, B, C, D, E_, th, g):
        return (1 + A * np.exp(B / np.cos(th))) * (1 + C * np.exp(D * g) + E_ * np.cos(g) ** 2)

    coef = {"Y": (0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771, -0.0670 * T + 0.3703),
            "x": (-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989, -0.0033 * T + 0.0452),
            "y": (-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537, -0.0109 * T + 0.0529)}
    chi = (4 / 9 - T / 120) * (np.pi - 2 * ts)
    Yz = (4.0453 * T - 4.9710) * np.tan(chi) - 0.2155 * T + 2.4192
    t3, t2 = ts**3, ts**2
    xz = T * T * (0.00166 * t3 - 0.00375 * t2 + 0.00209 * ts) + T * (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * ts + 0.00394) \
        + (0.11693 * t3 - 0.21196 * t2 + 0.06052 * ts + 0.25886)
    yz = T * T * (0.00275 * t3 - 0.00610 * t2 + 0.00317 * ts) + T * (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * ts + 0.00516) \
        + (0.15346 * t3 - 0.26756 * t2 + 0.06670 * ts + 0.26688)
    val = {}
    for k, z in (("Y", Yz), ("x", xz), ("y", yz)):
        val[k] = z * perez(*coef[k], theta, gamma) / perez(*coef[k], 0.0, ts)
    Y = val["Y"] / Yz
    X = val["x"] / val["y"] * Y
    Z = (1 - val["x"] - val["y"]) / val["y"] * Y
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(np.einsum("ij,jhw->hwi", M, np.stack([X, Y, Z])), 0, None)
    ground = np.array([0.32, 0.29, 0.25]) * 0.35 * rgb[np.argmin(np.abs(el)), :, :].mean(0)
    rgb[dz < 0] = ground
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, rgb[..., ::-1].astype(np.float32))
    return s


def _sky(stage, E, cfg):
    """Simulated daylight: analytic Preetham sky on the dome + a matching sun (no photographs)."""
    env = cfg["environment"]
    el, az = env["sun_elevation_deg"], env["sun_azimuth_deg"]
    hdr = os.path.join(PH, "..", "generated", f"sky_preetham_e{el:g}_a{az:g}_t{env['turbidity']:g}.hdr")
    s = preetham_sky(hdr, el, az, env["turbidity"]) if not os.path.exists(hdr) else _sun_dir(el, az)
    dome = UsdLux.DomeLight.Define(stage, E + "/Sky")
    dome.CreateIntensityAttr(float(env["sky_intensity"]))
    dome.CreateTextureFileAttr(os.path.abspath(hdr))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    sun = UsdLux.DistantLight.Define(stage, E + "/Sun")
    sun.CreateIntensityAttr(float(env["sun_intensity"]))
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
    from .mathutil import look_at_quat

    q = look_at_quat(np.zeros(3), -s)   # DistantLight shines along its local -Z, like a camera looks
    x = UsdGeom.Xformable(sun)
    x.ClearXformOpOrder()
    x.AddOrientOp().Set(Gf.Quatf(*map(float, q)))


def _sun_dir(el, az):
    e, a = np.deg2rad(el), np.deg2rad(az)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _fence(stage, F):
    """Chain-link panels (1.91 m) with posts around x [-9, 14], y [-9, 9]; 4 m gate gap on the east side."""
    kit = _ph("modular_chainlink_fence")
    panel_c = np.array([-0.995, 0.0])      # centre of the 'double' panel in the kit frame
    post_c = np.array([-0.995, 1.005])     # centre of a post in the kit frame
    x0, x1, y0, y1 = -9.0, 14.0, -9.0, 9.0
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    k = 0
    for (ax, ay), (bx, by) in edges:
        a, b = np.array([ax, ay]), np.array([bx, by])
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        yaw = np.rad2deg(np.arctan2(u[1], u[0]))
        R = np.array([[np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw))], [np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))]])
        n = int(np.floor(L / 1.95))
        for i in range(n + 1):
            p = a + u * min(i * 1.95, L)
            _ref(stage, f"{F}/post_{k}", kit, (*(p - R @ post_c), 0), yaw, keep={"modular_chainlink_fence_post"})
            mid = a + u * (i * 1.95 + 0.975)
            gate = abs(mid[0] - x1) < 0.1 and abs(mid[1]) < 2.0
            if i < n and not gate:
                _ref(stage, f"{F}/panel_{k}", kit, (*(mid - R @ panel_c), 0), yaw, keep={"modular_chainlink_fence_double"})
            k += 1


def _building(stage, B, m):
    """3-storey reinforced-concrete frame under construction on the neighbouring plot (north of the fence)."""
    xs, ys = np.arange(-8.0, 8.1, 4.0), np.array([11.0, 15.0, 19.0])
    h, slab, col = 3.2, 0.22, 0.35
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            top = 3 * h if (i + j) % 2 == 0 or i < 3 else 2 * h + slab   # some top-floor columns not cast yet
            _box(stage, f"{B}/col_{i}_{j}", (col, col, top), (x, y, top / 2), m["concrete"], tile=1.0)
            if top > 2 * h + slab:   # starter rebars on the top
                for r, (dx, dy) in enumerate(((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1))):
                    _tube(stage, f"{B}/rebar_{i}_{j}_{r}", (x + dx, y + dy, top), (x + dx, y + dy, top + 0.6), 0.008, m["steel"])
    for f in (1, 2):
        _box(stage, f"{B}/slab_{f}", (16.6, 8.6, slab), (0, 15.0, f * h + slab / 2), m["concrete"], tile=2.0)
        for i, x in enumerate(xs):   # edge beams
            _box(stage, f"{B}/beam_{f}_{i}", (0.3, 8.35, 0.45), (x, 15.0, f * h - 0.22), m["concrete"], tile=1.0)
    # ground-floor hollow-block infill walls, partly built (same blocks as the robots lay)
    for b in range(4):
        x_a, courses = xs[b] + 0.2, (9, 6, 12, 3)[b]
        for c in range(courses):
            for n in range(12):
                x = x_a + 0.15 + n * 0.305 + (0.15 if c % 2 else 0.0)
                if x > xs[b + 1] - 0.3:
                    break
                _box(stage, f"{B}/infill_{b}_{c}_{n}", (0.30, 0.15, 0.148), (x, 11.0, 0.075 + c * 0.15), m["block"])
    # scaffolding along the south face: standards every 2 m, ledgers every 2 m height, planks per lift
    y_out, y_in = 10.0, 10.6
    for i, x in enumerate(np.arange(-8.0, 8.01, 2.0)):
        for y in (y_out, y_in):
            _tube(stage, f"{B}/std_{i}_{int(y * 10)}", (x, y, 0), (x, y, 8.0), 0.024, m["steel"])
    for lvl, z in enumerate(np.arange(2.0, 8.01, 2.0)):
        for y in (y_out, y_in):
            _tube(stage, f"{B}/ledger_{lvl}_{int(y * 10)}", (-8.0, y, z), (8.0, y, z), 0.024, m["steel"])
        _box(stage, f"{B}/deck_{lvl}", (16.0, 0.55, 0.04), (0, (y_out + y_in) / 2, z + 0.04), m["plank"], tile=1.0)
        _tube(stage, f"{B}/guard_{lvl}", (-8.0, y_out, z + 1.0), (8.0, y_out, z + 1.0), 0.024, m["steel"])
    for i, x in enumerate(np.arange(-8.0, 6.01, 4.0)):   # facade bracing
        _tube(stage, f"{B}/brace_{i}", (x, y_out, 0.1), (x + 4.0, y_out, 7.9), 0.024, m["steel"])


def _container(stage, C, m):
    """20 ft site office container: corrugated walls, door, windows."""
    L, W, H = 6.06, 2.44, 2.59
    cx, cy = 11.0, -6.6
    _box(stage, C + "/body", (L, W, H), (cx, cy, H / 2), m["corrugated"], tile=1.2)
    _box(stage, C + "/door", (0.9, 0.02, 2.0), (cx - 1.8, cy + W / 2 + 0.01, 1.0), m["dark"])
    for i, x in enumerate((cx + 0.2, cx + 1.9)):
        _box(stage, f"{C}/window_{i}", (1.1, 0.02, 0.9), (x, cy + W / 2 + 0.01, 1.5), m["dark"])
    _box(stage, C + "/step", (1.0, 0.5, 0.2), (cx - 1.8, cy + W / 2 + 0.25, 0.1), m["concrete"])


def _yard(stage, Y, m):
    """Material yard: pallets of hollow blocks, cement bags, crates (north-east, outside the clear zone)."""
    pallet = _isaac("/NVIDIA/Assets/ArchVis/Industrial/Pallets/Pallet_A1.usd")
    for p, (x, y, rot) in enumerate(((10.0, 4.8, 0), (11.5, 4.8, 0), (13.0, 4.8, 0), (10.0, 6.8, 90))):
        if pallet:
            _ref(stage, f"{Y}/pallet_{p}", pallet, (x, y, 0), rot)
        if p < 3:   # stacked blocks on the pallets (visual)
            for c in range(4):
                for r in range(3):
                    for n in range(3):
                        _box(stage, f"{Y}/stack_{p}_{c}_{r}_{n}", (0.30, 0.11, 0.148),
                             (x - 0.33 + n * 0.33, y - 0.25 + r * 0.25, 0.16 + 0.075 + c * 0.15), m["block"])
    bag = _ph("cement_bag")
    for i in range(6):
        _ref(stage, f"{Y}/bag_{i}", bag, (10.0 + (i % 3) * 0.47 - 0.47, 6.8 + (i // 3) * 0.02, 0.16 + (i // 3) * 0.18), 90)
    for i, (x, y) in enumerate(((12.6, 6.6), (12.95, 6.6), (12.6, 7.1))):
        _ref(stage, f"{Y}/crate_{i}", _ph("plastic_crate_01"), (x, y, 0), 0)
    _ref(stage, Y + "/wooden_crate", _ph("wooden_crate_01"), (13.3, 3.4, 0), 20)


def _props(stage, P):
    fork = _isaac("/Isaac/Props/Forklift/forklift.usd")
    if fork:
        _ref(stage, P + "/forklift", fork, (11.6, 1.4, 0), 200)
    for i, x in enumerate((9.5, 11.1, 12.7)):
        _ref(stage, f"{P}/barrier_{i}", _ph("concrete_road_barrier"), (x, -8.3, 0), 0)
    for i, y in enumerate((-2.5, -0.9)):
        _ref(stage, f"{P}/barrier2_{i}", _ph("concrete_road_barrier_02"), (-7.8, y, 0), 90)
    _ref(stage, P + "/generator", _ph("portable_generator"), (8.9, -6.2, 0), 75)
    _ref(stage, P + "/toolbox", _ph("metal_toolbox"), (9.0, -5.4, 0), -20)
    for i, (x, y) in enumerate(((13.4, -4.9), (13.4, -4.3), (12.9, -4.6))):
        _ref(stage, f"{P}/barrel_{i}", _ph("Barrel_01"), (x, y, 0), 30 * i)
    for i, (x, y) in enumerate(((9.4, -6.9), (9.75, -6.9))):
        _ref(stage, f"{P}/jerrycan_{i}", _ph("metal_jerrycan"), (x, y, 0), 90)
    _ref(stage, P + "/ladder", _ph("ladder_sectioned_01"), (7.9, -7.9, 0), 0)
    for i, (x, y, yaw) in enumerate(((-8.6, -8.6, 45), (13.6, 8.6, 225), (13.6, -8.6, 135), (-8.6, 8.6, -45))):
        _ref(stage, f"{P}/lamp_{i}", _ph("street_lamp_01"), (x, y, 0), yaw)


def _trees(stage, T):
    rng = np.random.default_rng(5)
    species = ["Black_Oak", "American_Beech", "Colorado_Spruce", "Gray_Birch", "Honey_Locust", "Douglas_Fir"]
    spots = [(-16, -12), (-19, -2), (-17, 7), (-13, 24), (-2, 26), (12, 25), (22, 16), (24, 3), (23, -10),
             (12, -17), (0, -18), (-11, -18)]
    for i, (x, y) in enumerate(spots):
        a = _isaac(f"/NVIDIA/Assets/Vegetation/Trees/{species[i % len(species)]}.usd")
        if a:
            _ref(stage, f"{T}/tree_{i}", a, (x + rng.uniform(-1, 1), y + rng.uniform(-1, 1), 0), rng.uniform(0, 360))
