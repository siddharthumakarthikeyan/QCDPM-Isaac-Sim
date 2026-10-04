"""Sensor-based 3-D team planning with the C++ core, Isaac-free. Run:  python3 tests/core_plan3d_test.py [course]

`course`: the harder site of `run_traj.py --scenario course` (a 4.4 m gate in a wall, a crate, two 3 m pillars), where the
team has to pull the ground robots' ring in to pass; the planner then searches (x, y, z, ring scale).

Same site as `run_traj.py --scenario plan` (a 2.2 m cabin and a 0.6 m pallet), not given to the planner. The map is an
elevation grid built from
  - a downward depth camera on each drone (90 deg, 64 x 64 rays, 20 mm noise), which gives obstacle heights, and
  - the 2-D lidar of each ground robot (270 deg, 811 beams, 10 mm noise), which only says that something reaches the
    scan plane,
taken at the robots' estimated positions (5 mm, 0.3 deg). The planner searches platform positions (x, y, z) with the
formation centred on the platform: the ground robots need flat ground, the eight cables and the tool need vertical
clearance over the map. It is re-run every 0.5 m as the map grows, and the executed motion is checked against the
true obstacles.

Not modelled: the sensors seeing the team itself, sensor latency, the drones' own clearance to tall obstacles.
"""

import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from cdpr_sim import cdpr_core as cc  # noqa: E402
from cdpr_sim.cdpr_control import corner_points, formation  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
rng = np.random.default_rng(5)
COURSE = "course" in sys.argv[1:]
if COURSE:
    obstacles = [((-5.95, -3.0, 1.0), (7.1, 0.3, 2.0)), ((5.75, -3.0, 1.0), (7.5, 0.3, 2.0)), ((-0.2, 2.2, 0.3), (1.0, 1.0, 0.6)),
                 ((1.6, 4.4, 1.5), (0.5, 0.5, 3.0)), ((-2.8, 3.6, 1.5), (0.5, 0.5, 3.0))]
else:
    obstacles = [((-0.5, -1.3, 1.1), (1.6, 1.2, 2.2)), ((-0.6, 2.4, 0.3), (1.2, 0.8, 0.6))]     # centre, size: cabin, pallet
lo = [np.array(c) - np.array(s) / 2 for c, s in obstacles]
hi = [np.array(c) + np.array(s) / 2 for c, s in obstacles]
boxes3 = [cc.Box3(a, b) for a, b in zip(lo, hi)]
start, goal = (np.array([-0.2, -6.5, 1.3]), np.array([-0.2, 7.6, 1.3])) if COURSE else (np.array([-0.5, -4.4, 1.3]), np.array([-0.5, 4.6, 1.3]))

dr, ug = formation(cfg, np.zeros(2))
sh = cc.TeamShape()
sh.ugv_offset, sh.drone_offset = ug[:, :2], dr[:, :2]
b = corner_points(cfg["platform"]["side"])
sh.corner = b[:4, :2]
sh.fairlead_h, sh.drone_alt, sh.half_side = cfg["ugv"]["attach_height"], cfg["layout"]["drone_altitude"], cfg["platform"]["side"] / 2
sh.tool_drop = 0.67
sh.robot_radius, sh.cable_margin, sh.body_radius, sh.clearance = 0.55, 0.20, 0.35, 0.15
sh.z_min, sh.z_max = 0.9, 2.0
sh.step_max = 0.10                        # ground counts as flat below this: the map keeps the highest of noisy returns
if COURSE:
    sh.scales = [0.6, 0.7, 0.8, 0.9, 1.0]
RES = 0.10
grid = cc.ElevationGrid(np.array([-9.0, -9.0]), RES, 180, 180)
LIDAR_H = cfg["ugv"]["ground_clearance"] + cfg["ugv"]["chassis_size"][2] / 2 + cfg["sensors"]["lidar_2d"]["mount_offset"][2]
FOV_L, BEAMS, R_MAX = np.deg2rad(270.0), 811, 25.0


def boxes2_at(height):
    """Footprints of the obstacles that reach a horizontal plane (what a 2-D lidar at that height can hit)."""
    return [cc.Box2(a[:2], b_[:2]) for a, b_ in zip(lo, hi) if a[2] <= height <= b_[2]]


def sense(c, heading):
    for k in range(4):
        # drone depth camera
        p = np.r_[c[:2] + sh.drone_offset[k], sh.drone_alt] + rng.normal(0, 0.005, 3)
        pts = cc.depth_scan(p, heading, boxes3, 64, np.deg2rad(90.0), 12.0)
        grid.insert_points(pts + rng.normal(0, 0.02, pts.shape) * [0.3, 0.3, 1.0])
        # ground-robot lidar
        pose = np.r_[c[:2] + scale * sh.ugv_offset[k], heading]
        ranges = cc.lidar_scan(pose, boxes2_at(LIDAR_H), BEAMS, FOV_L, 0.05, R_MAX)
        ranges = np.where(ranges < R_MAX, ranges + rng.normal(0, 0.01, BEAMS), ranges)
        est = pose + np.r_[rng.normal(0, 0.005, 2), rng.normal(0, np.deg2rad(0.3))]
        grid.insert_lidar(est, ranges, FOV_L, R_MAX, LIDAR_H)
    grid.update(sh.robot_radius, sh.cable_margin, sh.body_radius)


def seg_box_dist(p0, p1, a, b_, n=40):
    s = np.linspace(0, 1, n)[:, None]
    P = p0 + s * (p1 - p0)
    return np.linalg.norm(np.maximum(np.maximum(a - P, P - b_), 0.0), axis=1).min()


c, heading, scale = start.copy(), np.pi / 2, 1.0
executed, plans, t_plan, n_exp = [np.r_[c, scale]], 0, 0.0, 0
for it in range(300):
    sense(c, heading)
    planner = cc.TeamPlanner3(grid, sh)
    t0 = time.perf_counter()
    path, n = planner.plan4(np.r_[c, scale], np.r_[goal, 1.0])
    t_plan += time.perf_counter() - t0
    n_exp += n
    plans += 1
    if not path:
        print(f"  no path from {np.round(c, 2)}")
        break
    left = 0.5
    for q4 in path[1:]:
        q = np.array(q4[:3])
        seg = np.linalg.norm(q - c)
        if np.linalg.norm((q - c)[:2]) > 1e-6:
            heading = np.arctan2(q[1] - c[1], q[0] - c[0])
        if seg <= left:
            left -= seg
            c, scale = q, q4[3]
            executed.append(np.r_[c, scale])
        else:
            c, scale = c + (q - c) / seg * left, scale + (q4[3] - scale) * left / seg
            executed.append(np.r_[c, scale])
            break
    if np.linalg.norm(c - goal) < 1e-6 and abs(scale - 1.0) < 1e-9:
        break

executed = np.array(executed)
scales_run = executed[:, 3]
executed4, executed = executed, executed[:, :3]
length = np.linalg.norm(np.diff(executed, axis=0), axis=1).sum()
d_robot = d_cable = d_tool = 1e9
for p0, p1 in zip(executed4[:-1], executed4[1:]):
    for s in np.linspace(0, 1, max(2, int(max(np.linalg.norm((p1 - p0)[:3]), abs(p1[3] - p0[3]) * 2.75) / 0.03))):
        q, sc_ = (p0 + s * (p1 - p0))[:3], (p0 + s * (p1 - p0))[3]
        for a, b_ in zip(lo, hi):
            tip = q - [0, 0, sh.tool_drop]
            d_tool = min(d_tool, seg_box_dist(q, tip, a, b_, 8))
            for k in range(4):
                g = np.r_[q[:2] + sc_ * sh.ugv_offset[k], 0.0]
                d_robot = min(d_robot, np.linalg.norm(np.maximum(np.maximum(a[:2] - g[:2], g[:2] - b_[:2]), 0.0)))
                d_cable = min(d_cable, seg_box_dist(q + np.r_[sh.corner[k], -sh.half_side], np.r_[g[:2], sh.fairlead_h], a, b_))
                d_cable = min(d_cable, seg_box_dist(q + np.r_[sh.corner[k], sh.half_side],
                                                    np.r_[q[:2] + sh.drone_offset[k], sh.drone_alt], a, b_))
over = executed[(np.abs(executed[:, 0] + 0.2) < 0.5) & (np.abs(executed[:, 1] - 2.2) < 0.5)] if COURSE else \
    executed[(np.abs(executed[:, 0] + 0.6) < 0.6) & (np.abs(executed[:, 1] - 2.4) < 0.4)]
m, seen = grid.map(), grid.seen()
print(f"  reached the goal: {np.linalg.norm(c - goal) < 1e-6}; path {length:.2f} m (straight line {np.linalg.norm(goal - start):.2f} m); "
      f"{plans} plans, {1e3 * t_plan / plans:.1f} ms and {n_exp // plans} expansions each")
print(f"  map: {100 * seen.mean():.0f}% of the grid seen from above; highest cell {m.max():.2f} m (tallest obstacle {max(b_[2] for b_ in hi):.2f} m); "
      f"cells above 0.3 m: {int((m > 0.3).sum())}")
print(f"  platform height along the path: {executed[:, 2].min():.2f} .. {executed[:, 2].max():.2f} m; "
      + (f"passes over the {'crate' if COURSE else 'pallet'} at z = {over[:, 2].min():.2f} m" if len(over) else "goes around the low obstacle"))
if COURSE:
    print(f"  ring of the ground robots: {2.75 * scales_run.min():.2f} m at the gate (2.75 m nominal), back to {2.75 * scales_run[-1]:.2f} m at the goal")
print(f"  true clearances of the executed motion: ground robots {d_robot:.2f} m (required {sh.robot_radius}), cables {d_cable:.2f} m, "
      f"tool {d_tool:.2f} m (vertical clearance required {sh.clearance})")
ok = (np.linalg.norm(c - goal) < 1e-6 and d_robot > sh.robot_radius - 0.03 and d_cable > 0.10 and d_tool > 0.10)
np.savez(os.path.join(ROOT, "results", "core_course.npz" if COURSE else "core_plan3d.npz"), executed=executed, map=m, origin=np.array([-9.0, -9.0]), res=RES)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
