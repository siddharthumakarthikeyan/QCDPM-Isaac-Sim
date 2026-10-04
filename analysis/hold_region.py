"""How far can the platform move with the robots standing still?

For a fixed formation (crossed layout) this finds, at several heights, the largest horizontal offset of the platform
from the formation centre for which the controller's tension distribution is feasible: every cable between the
winch limits, and every drone cable below what the drone can hold at its tilt limit (with margin) and 85% thrust.
The smallest value over 16 directions is the radius of the region the winches alone can serve; the runtime keeps
the robots still inside a fraction of it (layout.hold_radius).

    python3 analysis/hold_region.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory as th  # noqa: E402

g = th.cfg["gripper"]
m_tool = th.cfg["platform"]["mass"] + g["wrist"]["mass"] + g["tool"]["mass"] + 2 * g["finger"]["mass"]
A = th.anchors(*th.CROSSED)
tilt = th.cfg["drone"]["control"]["max_tilt_deg"] - th.cfg["drone"]["control"]["tension_caps"]["tilt_margin_deg"]


def ok(p, m):
    W, u, _ = th.W_at(A, p)
    w = np.array([0, 0, m * th.G, 0, 0, 0])
    tb = th.drone_tension_bounds(u, tilt, 0.85 * th.F_MAX)
    t, feas = th.tension_distribution(W, w, th.T_MIN, tb, th.T_REF)
    return feas, t, u


print(f"tool {m_tool:.2f} kg; drone tilt cap {tilt:.0f} deg, thrust cap 85%")
for m, name in ((m_tool, "empty tool"), (m_tool + th.cfg["blocks"]["mass"], "tool + 2.2 kg block")):
    for z in (0.7, 1.0, 1.3, 1.6, 2.0):
        radii = []
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            r = 0.0
            while r < 2.5 and ok(np.array([(r + 0.02) * np.cos(a), (r + 0.02) * np.sin(a), z]), m)[0]:
                r += 0.02
            radii.append(r)
        R = min(radii)
        _, t, u = ok(np.array([0.8 * R, 0, z]), m)
        print(f"{name:20s} z {z:.1f} m: feasible radius {R:.2f} m (max over directions {max(radii):.2f}); at 0.8 R: "
              f"tensions {t.min():.0f}-{t.max():.0f} N, drone tilt {th.drone_tilt(t, u).max():.0f} deg")
