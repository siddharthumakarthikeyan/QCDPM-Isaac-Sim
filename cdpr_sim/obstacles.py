"""Site obstacles for the planning demo: plain collision boxes in the physics run, dressed versions in the film.

Each obstacle is (kind, centre, size): "cabin" (site cabin the team has to go around) or "pallet" (a low pallet of
blocks the ground robots straddle while the payload is lifted over it).
"""

import numpy as np

from .scene import _material, _uv_box, _xform

ROOT = "/World/Obstacles"


def build_colliders(stage, obstacles):
    """Physics run: static boxes with collision (anything that touches them is stopped by PhysX)."""
    _xform(stage, ROOT)
    mat = _material(stage, "/World/Looks/obstacle", (0.75, 0.35, 0.12), 0.7)
    for i, (kind, c, s) in enumerate(obstacles):
        _uv_box(stage, f"{ROOT}/{kind}_{i}", s, c, mat=mat, collide=True)


def build_film(stage, cfg, obstacles):
    """Film: the same volumes as a corrugated site cabin and a strapped pallet of blocks (visual only)."""
    from .sandsite import _b, _block_stack
    from .site import flat_material, pbr_material

    _xform(stage, ROOT)
    E = "/World/Environment"
    m = {
        "corr": pbr_material(stage, ROOT + "/Looks/corrugated", "corrugated_iron", res="1k", tint=(0.92, 0.80, 0.42)),
        "dark": flat_material(stage, ROOT + "/Looks/dark", (0.04, 0.045, 0.05), 0.2, 0.3),
        "frame": flat_material(stage, ROOT + "/Looks/frame", (0.08, 0.08, 0.09), 0.5, 0.4),
        "white": flat_material(stage, ROOT + "/Looks/white", (0.9, 0.9, 0.88), 0.5),
    }
    sand = {k: stage.GetPrimAtPath(f"{E}/Looks/{k}") for k in ("wood", "block", "strap")}
    from pxr import UsdShade

    sm = {k: UsdShade.Material(v) for k, v in sand.items()}
    for i, (kind, c, s) in enumerate(obstacles):
        c, s = np.asarray(c, float), np.asarray(s, float)
        p = f"{ROOT}/{kind}_{i}"
        if kind == "cabin":
            _xform(stage, p, (c[0], c[1], 0.0))
            h = s[2]
            _b(stage, p + "/body", (s[0] - 0.06, s[1] - 0.06, h - 0.12), (0, 0, h / 2 + 0.03), m["corr"], 0.9)
            _b(stage, p + "/roof", (s[0], s[1], 0.06), (0, 0, h - 0.03), m["frame"])
            _b(stage, p + "/skid", (s[0], s[1], 0.09), (0, 0, 0.045), m["frame"])
            for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
                _b(stage, f"{p}/post_{k}", (0.07, 0.07, h), (sx * (s[0] / 2 - 0.035), sy * (s[1] / 2 - 0.035), h / 2), m["frame"])
            _b(stage, p + "/door", (0.02, 0.8, 1.85), (-s[0] / 2 + 0.02, -0.1, 1.02), m["white"])
            _b(stage, p + "/door_handle", (0.03, 0.04, 0.14), (-s[0] / 2 - 0.0, 0.2, 1.0), m["frame"])
            _b(stage, p + "/window", (0.9, 0.02, 0.6), (0.0, -s[1] / 2 + 0.02, 1.45), m["dark"])
            _b(stage, p + "/window_frame", (1.0, 0.015, 0.7), (0.0, -s[1] / 2 + 0.028, 1.45), m["white"])
        else:
            layers = int(round((s[2] - 0.144) / cfg["blocks"]["size"][2]))
            _block_stack(stage, p, cfg, (c[0], c[1], 0.0), 0.0, layers, sm)
