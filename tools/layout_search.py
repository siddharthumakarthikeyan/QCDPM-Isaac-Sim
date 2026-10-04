"""Grid search over anchor layouts: minimise worst-case drone thrust over a platform pose set.

For every candidate layout and pose, a linear program finds the tension vector (t >= t_min) that minimises the
largest tension while balancing the platform weight; the drone requirement is |m_d g e3 + t_k u_k|.
Run: python3 tools/layout_search.py
"""

import itertools
import os
import sys

import numpy as np
import yaml
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cdpr_sim.cdpr_control import corner_points, structure_matrix  # noqa: E402
from cdpr_sim.mathutil import rpy_to_rot  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "cdpr.yaml")))
g = cfg["sim"]["gravity"]
b = corner_points(cfg["platform"]["side"])
w_g = np.array([0, 0, cfg["platform"]["mass"] * g, 0, 0, 0])
z0 = cfg["platform"]["start_position"][2]
t_min = cfg["winch"]["tension_min"]
m_d = cfg["drone"]["mass"]
z_g = cfg["ugv"]["attach_height"]
POSES = list(itertools.product((-0.3, 0, 0.3), (-10, 0, 10), (-10, 0, 10), (-15, 0, 15)))


def evaluate(Rd, zd, Rg, cross_top, cross_bot):
    az = np.deg2rad(45 + 90 * np.arange(4))
    ct, cb = np.deg2rad(cross_top), np.deg2rad(cross_bot)
    A = np.vstack([np.c_[Rd * np.cos(az + ct), Rd * np.sin(az + ct), np.full(4, zd)],
                   np.c_[Rg * np.cos(az + cb), Rg * np.sin(az + cb), np.full(4, z_g)]])
    worst_t = worst_thrust = 0.0
    for dz, r, p, y in POSES:
        W, u, _ = structure_matrix(np.array([0, 0, z0 + dz]), rpy_to_rot(np.deg2rad([r, p, y])), b, A)
        res = linprog(np.r_[np.zeros(8), 1], A_ub=np.c_[np.eye(8), -np.ones(8)], b_ub=np.zeros(8),
                      A_eq=np.c_[W, np.zeros(6)], b_eq=w_g, bounds=[(t_min, None)] * 8 + [(0, None)])
        if res.status != 0:
            return np.inf, np.inf
        t = res.x[:8]
        worst_t = max(worst_t, t.max())
        worst_thrust = max(worst_thrust, np.linalg.norm(m_d * g * np.array([0, 0, 1]) + t[:4, None] * u[:4], axis=1).max())
    return worst_thrust, worst_t


if __name__ == "__main__":
    rows = []
    for (ct, cb), Rd, zd, Rg in itertools.product([(90, -90), (75, -75), (105, -105), (60, -60)],
                                                  (1.6, 1.8, 2.0, 2.2), (3.0, 3.25, 3.5, 3.75), (2.0, 2.5, 2.75, 3.0)):
        rows.append((*evaluate(Rd, zd, Rg, ct, cb), ct, cb, Rd, zd, Rg))
    rows.sort()
    print("worst drone thrust [N] | worst tension [N] | cross top/bot | drone R, z | UGV R")
    for r in rows[:10]:
        print("%6.1f | %6.1f | %4d/%4d | %.2f %.2f | %.2f" % r)
    L = cfg["layout"]
    print("configured:", evaluate(L["drone_radius"], L["drone_altitude"], L["ugv_radius"],
                                  L["drone_cross_angle_deg"], L["ugv_cross_angle_deg"]))
