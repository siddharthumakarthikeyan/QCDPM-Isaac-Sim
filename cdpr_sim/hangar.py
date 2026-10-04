"""Aircraft hangar interior for the painting application: sealed concrete floor with markings, concrete dado and
corrugated cladding, steel portal frames, a pitched roof with high-bay lights, and the main door open to daylight.
Procedural geometry and local textures only. Physics sees one flat floor.
"""

import numpy as np
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

from .scene import _bind_phys, _quad, _uv_box, _xform
from .site import _ph, _ref, _sky, flat_material, pbr_material

E = "/World/Environment"
X0, X1, Y0, Y1 = -12.0, 10.0, -16.0, 9.0          # inside faces of the walls; the door is the whole +x end
H_EAVE, RISE = 7.2, 2.2


def _rx(deg):
    a = np.deg2rad(deg) / 2
    return (np.cos(a), np.sin(a), 0, 0)


def build_hangar(stage, cfg, pm_ground):
    env = dict(cfg["environment"])
    env.update(env.get("hangar") or {})
    _xform(stage, E)
    m = {
        "floor": pbr_material(stage, E + "/Looks/floor", "concrete_floor_02", tint=(1.25, 1.25, 1.22)),
        "apron": pbr_material(stage, E + "/Looks/apron", "concrete_floor_02", tint=(0.9, 0.88, 0.84)),
        "dado": pbr_material(stage, E + "/Looks/dado", "concrete_slab_wall", res="1k", tint=(0.85, 0.85, 0.85)),
        "clad": pbr_material(stage, E + "/Looks/cladding", "corrugated_iron", res="1k", tint=(0.86, 0.88, 0.9)),
        "roof": pbr_material(stage, E + "/Looks/roof", "corrugated_iron", res="1k", tint=(0.62, 0.64, 0.66)),
        "steel": flat_material(stage, E + "/Looks/steel", (0.16, 0.24, 0.36), 0.45, 0.6),
        "yellow": flat_material(stage, E + "/Looks/yellow", (0.95, 0.74, 0.08), 0.5),
        "white": flat_material(stage, E + "/Looks/white", (0.92, 0.92, 0.9), 0.5),
        "red": flat_material(stage, E + "/Looks/red", (0.75, 0.06, 0.05), 0.5),
        "dark": flat_material(stage, E + "/Looks/dark", (0.04, 0.045, 0.05), 0.5),
        "cone": flat_material(stage, E + "/Looks/cone", (1.0, 0.33, 0.03), 0.5),
        "rubber": flat_material(stage, E + "/Looks/rubber", (0.02, 0.02, 0.02), 0.9),
    }
    xm, ym, L, Wd = (X0 + X1) / 2, (Y0 + Y1) / 2, X1 - X0, Y1 - Y0
    # floor inside, apron outside the door
    f = _quad(stage, E + "/Floor", 1.0, 0.0, False, m["floor"], uv_repeat=1.0)
    f.GetPointsAttr().Set([Gf.Vec3f(X0, Y0, 0), Gf.Vec3f(X1, Y0, 0), Gf.Vec3f(X1, Y1, 0), Gf.Vec3f(X0, Y1, 0)])
    UsdGeom.PrimvarsAPI(f).GetPrimvar("st").Set([Gf.Vec2f(0, 0), Gf.Vec2f(L / 3, 0), Gf.Vec2f(L / 3, Wd / 3), Gf.Vec2f(0, Wd / 3)])
    a = _quad(stage, E + "/Apron", 160.0, -0.004, False, m["apron"], uv_repeat=160.0 / 4)
    plane = UsdGeom.Plane.Define(stage, E + "/GroundCollider")
    plane.CreateAxisAttr("Z")
    plane.CreateWidthAttr(300.0)
    plane.CreateLengthAttr(300.0)
    UsdGeom.Imageable(plane).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    _bind_phys(plane.GetPrim(), pm_ground)
    # walls: concrete dado, corrugated cladding above (back wall and both sides)
    t = 0.2
    for name, size, pos in (("back", (t, Wd + 2 * t, H_EAVE), (X0 - t / 2, ym, 0)), ("left", (L, t, H_EAVE), (xm, Y1 + t / 2, 0)),
                            ("right", (L, t, H_EAVE), (xm, Y0 - t / 2, 0))):
        _uv_box(stage, f"{E}/wall_{name}_dado", (size[0], size[1], 1.4), (pos[0], pos[1], 0.7), mat=m["dado"], tile=2.0)
        _uv_box(stage, f"{E}/wall_{name}", (size[0], size[1], H_EAVE - 1.4), (pos[0], pos[1], 1.4 + (H_EAVE - 1.4) / 2), mat=m["clad"], tile=1.4)
        _uv_box(stage, f"{E}/wall_{name}_band", (size[0] + 0.01, size[1] + 0.01, 0.12), (pos[0], pos[1], 1.46), mat=m["steel"])
    # back-wall gable and the door header at the open end
    for i, x in enumerate((X0 - t / 2, X1 + t / 2)):
        for s in (-1, 1):                                  # two triangles approximated by stepped panels under the roof
            for k in range(6):
                y0 = ym + s * Wd / 2 * k / 6
                h = RISE * (1 - (k + 0.5) / 6)
                _uv_box(stage, f"{E}/gable_{i}_{s > 0}_{k}", (t, Wd / 12, h), (x, y0 + s * Wd / 24, H_EAVE + h / 2), mat=m["clad"], tile=1.4)
    _uv_box(stage, E + "/door_header", (0.5, Wd + 2 * t, 0.9), (X1 + 0.05, ym, H_EAVE - 0.45), mat=m["steel"])
    for i, y in enumerate((Y0 + 2.2, Y1 - 2.2)):           # sliding door leaves parked at the sides
        _uv_box(stage, f"{E}/door_leaf_{i}", (0.12, 4.4, H_EAVE - 0.9), (X1 + 0.3, y, (H_EAVE - 0.9) / 2), mat=m["clad"], tile=1.4)
        _uv_box(stage, f"{E}/door_rail_{i}", (0.2, 4.6, 0.14), (X1 + 0.3, y, 0.07), mat=m["steel"])
    # portal frames: columns on both side walls and rafters to the ridge, purlins along the roof
    slope = np.rad2deg(np.arctan2(RISE, Wd / 2))
    rl = np.hypot(RISE, Wd / 2)
    frames = np.linspace(X0 + 0.4, X1 - 0.4, 5)
    for i, x in enumerate(frames):
        for s in (-1, 1):
            y = ym + s * (Wd / 2 - 0.2)
            _uv_box(stage, f"{E}/column_{i}_{s > 0}", (0.3, 0.36, H_EAVE), (x, y, H_EAVE / 2), mat=m["steel"])
            _uv_box(stage, f"{E}/rafter_{i}_{s > 0}", (0.28, rl, 0.42), (x, ym + s * Wd / 4, H_EAVE + RISE / 2 - 0.25), quat=_rx(-s * slope), mat=m["steel"])
            _uv_box(stage, f"{E}/haunch_{i}_{s > 0}", (0.26, 1.6, 0.5), (x, y - s * 0.7, H_EAVE - 0.45), quat=_rx(-s * 28), mat=m["steel"])
    for s in (-1, 1):
        _uv_box(stage, f"{E}/roof_{s > 0}", (L + 0.6, rl + 0.5, 0.06), (xm, ym + s * Wd / 4, H_EAVE + RISE / 2 + 0.05), quat=_rx(-s * slope), mat=m["roof"], tile=1.4)
        for k in range(1, 5):
            u = k / 5
            _uv_box(stage, f"{E}/purlin_{s > 0}_{k}", (L, 0.1, 0.16), (xm, ym + s * Wd / 2 * (1 - u), H_EAVE + RISE * u - 0.12), mat=m["steel"])
    # lights: daylight through the door plus rows of high-bay fittings
    env.setdefault("sun_elevation_deg", 38.0)
    env.setdefault("sun_azimuth_deg", -20.0)                # low sun coming in through the door
    env["sky_intensity"] = env.get("hangar_sky_intensity", 2600.0)
    _sky(stage, E, {"environment": env})
    k = 0
    for x in np.linspace(X0 + 3.0, X1 - 3.0, 4):
        for y in np.linspace(Y0 + 3.5, Y1 - 3.5, 4):
            z = H_EAVE - 0.5
            _uv_box(stage, f"{E}/Lights/fitting_{k}", (1.5, 0.42, 0.1), (x, y, z + 0.08), mat=m["dark"])
            _uv_box(stage, f"{E}/Lights/drop_{k}", (0.03, 0.03, 0.9), (x, y, z + 0.55), mat=m["dark"])
            lt = UsdLux.RectLight.Define(stage, f"{E}/Lights/light_{k}")
            lt.CreateWidthAttr(1.4)
            lt.CreateHeightAttr(0.36)
            lt.CreateIntensityAttr(float(env.get("bay_light_intensity", 45000.0)))
            lt.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
            lt.ClearXformOpOrder()
            lt.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), z))      # RectLight emits along its local -z: straight down
            k += 1
    # floor markings: a taxi centre line through the door and a hatched work zone around the wing
    _uv_box(stage, E + "/Marks/centre_line", (L + 30, 0.15, 0.002), (xm + 15, -3.6, 0.002), mat=m["yellow"])
    for i, (c, sz) in enumerate((((0.0, 5.2), (7.0, 0.1)), ((0.0, -3.0 + 0.0), (0.0, 0.0)), ((-3.5, 1.0), (0.1, 8.4)), ((3.5, 1.0), (0.1, 8.4)))):
        if sz[0] > 0:
            _uv_box(stage, f"{E}/Marks/zone_{i}", (sz[0], sz[1], 0.002), (c[0], c[1], 0.002), mat=m["red"])
    # hangar clutter along the walls (visual only), well away from the team
    for i, (x, y) in enumerate(((-10.6, 6.8), (-10.6, 6.1), (-9.9, 6.6), (-10.5, 5.3))):
        _ref(stage, f"{E}/Props/drum_{i}", _ph("Barrel_01"), (x, y, 0), 40 * i)
    _ref(stage, E + "/Props/toolbox", _ph("metal_toolbox"), (-9.0, 7.6, 0), 15)
    _ref(stage, E + "/Props/ladder", _ph("ladder_sectioned_01"), (-6.5, 8.3, 0), 0)
    _ref(stage, E + "/Props/generator", _ph("portable_generator"), (-11.0, -1.0, 0), 95)
    _ref(stage, E + "/Props/crate", _ph("wooden_crate_01"), (-10.8, -13.5, 0), 20)
    _ref(stage, E + "/Props/jerrycan", _ph("metal_jerrycan"), (-9.6, 5.6, 0), 60)
    for i, (x, y) in enumerate(((4.2, 5.6), (-4.2, 5.6), (4.2, -1.6), (-4.2, 3.0))):
        from .sandsite import _cone

        _cone(stage, f"{E}/Props/cone_{i}", (x, y, 0), m)
