"""Light aircraft for the painting application. The team paints the upper surface of the right wing: the wing
passes under the platform, between the two pairs of ground robots, and the lower cables have to clear its leading
and trailing edges.

World frame: the aircraft points along +x, its centre line is at y = Y_CL, and the right wing runs towards +y
through the origin. The physics run only needs the wing (a static collision slab); the film gets the whole
aircraft (visual only).
"""

import numpy as np

ROOT = "/World/Aircraft"
Y_CL = -3.6                 # fuselage centre line
Y_ROOT, Y_TIP = -3.0, 2.4   # right wing, root to tip
CHORD = 1.3                 # outer-wing chord; leading edge at x = +CHORD/2
Z_LOW, Z_TOP = 0.70, 0.90   # wing lower surface and highest point of the upper surface (low-wing aircraft)
WING_BOX = ((0.0, (Y_ROOT + Y_TIP) / 2, (Z_LOW + Z_TOP) / 2), (CHORD, Y_TIP - Y_ROOT, Z_TOP - Z_LOW))


def wing_top(x):
    """Height of the upper surface at chordwise position x (simple cambered section, thickest at 30% chord)."""
    s = np.clip((CHORD / 2 - np.asarray(x, float)) / CHORD, 0, 1)        # 0 at the leading edge, 1 at the trailing edge
    t = 0.2969 * np.sqrt(s) - 0.126 * s - 0.3516 * s**2 + 0.2843 * s**3 - 0.1015 * s**4      # NACA 4-digit thickness shape
    return Z_LOW + 0.02 + (Z_TOP - Z_LOW - 0.02) * t / 0.100


def wing_boxes(n=8):
    """The wing section as n chordwise slabs (centre, size), each as high as the skin at its highest point. Used both
    as the collision shape in the physics run and as the obstacle in the team clearance check."""
    e = np.linspace(-CHORD / 2, CHORD / 2, n + 1)
    out = []
    for a, b in zip(e[:-1], e[1:]):
        top = float(wing_top(np.linspace(a, b, 9)).max())
        out.append((((a + b) / 2, (Y_ROOT + Y_TIP) / 2, (Z_LOW + top) / 2), (b - a, Y_TIP - Y_ROOT, top - Z_LOW)))
    return out


def build_collider(stage):
    """Physics run: the wing as a static slab (anything touching it is stopped by PhysX)."""
    from .scene import _material, _uv_box, _xform

    _xform(stage, ROOT)
    mat = _material(stage, "/World/Looks/wing", (0.8, 0.8, 0.82), 0.4)
    from pxr import UsdGeom

    for i, (c, sz) in enumerate(wing_boxes()):
        b = _uv_box(stage, f"{ROOT}/wing_collider_{i}", sz, c, mat=mat, collide=True)
        UsdGeom.Imageable(b).MakeInvisible()              # the drawn wing is build_film's; these only collide


def build_film(stage):
    """The whole aircraft as drawn (no collision): fuselage, both wings, tail, propeller and gear. The right wing is
    in primer, ready for paint."""
    from .dressup import _mat, lathe
    from .scene import _uv_box, _xform
    from .sandsite import _prism
    from .site import _tube

    _xform(stage, ROOT)
    primer = _mat(stage, "aircraft_primer", (0.60, 0.61, 0.62), 0.55)
    white = _mat(stage, "aircraft_white", (0.90, 0.90, 0.88), 0.25, coat=0.6)
    dark = _mat(stage, "aircraft_dark", (0.03, 0.03, 0.035), 0.4, 0.3)
    glass = _mat(stage, "aircraft_glass", (0.02, 0.03, 0.04), 0.05, 0.5, coat=1.0)
    alu = _mat(stage, "aircraft_alu", (0.78, 0.79, 0.80), 0.3, 1.0)
    rubber = _mat(stage, "aircraft_rubber", (0.02, 0.02, 0.02), 0.9)
    zc = 1.55                                                            # fuselage axis height
    # fuselage: smooth body of revolution, nose at x = +3.4, tail cone to x = -4.6
    st = np.array([(-4.6, 0.05), (-4.2, 0.12), (-3.2, 0.26), (-2.0, 0.44), (-0.9, 0.58), (0.2, 0.64), (1.2, 0.62), (2.1, 0.52), (2.8, 0.40),
                   (3.25, 0.30), (3.4, 0.24)])
    xs = np.linspace(-4.6, 3.4, 60)
    lathe(stage, ROOT + "/fuselage", [(float(np.interp(x, st[:, 0], st[:, 1])), float(x)) for x in xs], white, "X", (0, Y_CL, zc), smooth=True)
    lathe(stage, ROOT + "/cowl_ring", [(0.25, 3.38), (0.25, 3.44), (0.12, 3.44)], dark, "X", (0, Y_CL, zc))
    lathe(stage, ROOT + "/spinner", [(0.12, 3.44), (0.11, 3.52), (0.07, 3.64), (0.0, 3.72)], alu, "X", (0, Y_CL, zc), smooth=True)
    for i in range(3):                                                   # three-blade propeller
        a = np.deg2rad(20 + 120 * i)
        _tube(stage, f"{ROOT}/blade_{i}", (3.5, Y_CL, zc), (3.5, Y_CL + 0.95 * np.cos(a), zc + 0.95 * np.sin(a)), 0.035, dark)
    lathe(stage, ROOT + "/canopy", [(np.cos(t), np.sin(t)) for t in np.linspace(0, np.pi / 2, 12)], glass, "Z", (0.55, Y_CL, zc + 0.36),
          scale=(1.35, 0.52, 0.42), smooth=True)
    # wings: extruded section. The right wing (the one being painted) is in primer, the rest is finished white
    xs = np.linspace(CHORD / 2, -CHORD / 2, 26)
    # _prism wants a convex outline, counter-clockwise seen from -y: lower surface front to back, then upper back to front
    prof = [(CHORD / 2, Z_LOW + 0.03)] + [(float(x), float(wing_top(x))) for x in xs[1:-1]] + [(-CHORD / 2, Z_LOW + 0.012)] + \
           [(float(x), Z_LOW) for x in xs[::-1][2:-2]]
    _prism(stage, ROOT + "/wing_right", prof, Y_ROOT, Y_TIP, primer)
    _prism(stage, ROOT + "/wing_left", prof, 2 * Y_CL - Y_TIP, 2 * Y_CL - Y_ROOT, white)
    for i, y in enumerate((Y_TIP, 2 * Y_CL - Y_TIP - 0.05)):             # wing-tip caps
        _uv_box(stage, f"{ROOT}/tip_{i}", (CHORD * 0.96, 0.05, 0.1), (0, y + 0.025, Z_LOW + 0.08), mat=white)
    # tail: horizontal stabiliser and fin
    hp = [(-3.55, zc + 0.12), (-3.9, zc + 0.17), (-4.45, zc + 0.13), (-3.9, zc + 0.09)]
    _prism(stage, ROOT + "/stabiliser", hp, Y_CL - 1.6, Y_CL + 1.6, white)
    fin = [(-3.3, zc + 0.2), (-4.5, zc + 0.2), (-4.75, zc + 1.45), (-4.25, zc + 1.45)]
    _prism(stage, ROOT + "/fin", fin, Y_CL - 0.045, Y_CL + 0.045, white)
    _uv_box(stage, ROOT + "/fin_stripe", (0.95, 0.095, 0.16), (-4.38, Y_CL, zc + 1.2), mat=_mat(stage, "aircraft_stripe", (0.05, 0.22, 0.55), 0.3, coat=0.6))
    # landing gear: nose leg and two mains under the wings
    for name, (x, y) in (("nose", (2.5, Y_CL)), ("main_r", (-0.25, Y_CL + 1.25)), ("main_l", (-0.25, Y_CL - 1.25))):
        top = zc - 0.45 if name == "nose" else Z_LOW
        _tube(stage, f"{ROOT}/leg_{name}", (x, y, top), (x, y, 0.2), 0.03, alu)
        lathe(stage, f"{ROOT}/wheel_{name}", [(0.07, -0.06), (0.17, -0.06), (0.2, -0.03), (0.2, 0.03), (0.17, 0.06), (0.07, 0.06)], rubber, "Y", (x, y, 0.2))
        lathe(stage, f"{ROOT}/hub_{name}", [(0.0, -0.065), (0.08, -0.065), (0.08, 0.065), (0.0, 0.065)], alu, "Y", (x, y, 0.2))
