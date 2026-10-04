"""Sensor-based team planning with the C++ core, Isaac-free. Run:  python3 tests/core_plan_test.py

The team crosses the site of `run_traj.py --scenario plan` (a cabin and a pallet) without being given the obstacles.
Each ground robot carries a 2-D lidar (SICK TiM781 class: 270 deg, 811 beams, 25 m, 10 mm range noise) and knows its
pose only to the accuracy of its estimator (5 mm, 0.3 deg). The scans build an occupancy grid; the planner (A* for
the platform with the formation centred on it: four robots, their cables' ground projections and the tool must
clear the map) is re-run every 0.5 m of travel as the map grows. The executed path is then checked against the true
obstacles.

Not modelled here: the lidars seeing the other robots and the platform, and obstacles lower than the lidar plane.
"""

import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from cdpr_sim import cdpr_core as cc  # noqa: E402
from cdpr_sim.cdpr_control import formation  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
rng = np.random.default_rng(3)
obstacles = [((-0.5, -1.3), (1.6, 1.2)), ((-0.6, 2.4), (1.2, 0.8))]          # centre, size: cabin, pallet
boxes = [cc.Box2(np.array(c) - np.array(s) / 2, np.array(c) + np.array(s) / 2) for c, s in obstacles]
start, goal = np.array([-0.5, -4.4]), np.array([-0.5, 4.6])

fp = cc.TeamFootprint()
fp.ugv_offset = formation(cfg, np.zeros(2))[1][:, :2]
fp.robot_radius, fp.cable_margin, fp.body_radius = 0.55, 0.20, 0.35
RES = 0.10
grid = cc.OccupancyGrid(np.array([-9.0, -9.0]), RES, 180, 180)
FOV, BEAMS, R_MAX = np.deg2rad(270.0), 811, 25.0


def true_distance(p):
    d = 1e9
    for b in boxes:
        d = min(d, np.linalg.norm(np.maximum(np.maximum(b.lo - p, p - b.hi), 0.0)))
    return d


def sense(c, heading):
    for k in range(4):
        pose = np.r_[c + fp.ugv_offset[k], heading]
        ranges = cc.lidar_scan(pose, boxes, BEAMS, FOV, 0.05, R_MAX)
        ranges = np.where(ranges < R_MAX, ranges + rng.normal(0, 0.01, BEAMS), ranges)
        est = pose + np.r_[rng.normal(0, 0.005, 2), rng.normal(0, np.deg2rad(0.3))]    # the robot's own pose estimate
        grid.insert_scan(est, ranges, FOV, R_MAX)
    grid.update_distance()


c, heading = start.copy(), np.pi / 2
executed, replans, t_plan, n_exp = [c.copy()], 0, 0.0, 0
path = []
for it in range(200):
    sense(c, heading)
    planner = cc.TeamPlanner(grid, fp)
    t0 = time.perf_counter()
    path, n = planner.plan(c, goal)
    t_plan += time.perf_counter() - t0
    n_exp += n
    replans += 1
    if not path:
        print(f"  no path from {np.round(c, 2)}")
        break
    # advance 0.5 m along the plan
    left = 0.5
    for q in path[1:]:
        seg = np.linalg.norm(q - c)
        if seg <= left:
            left -= seg
            heading = np.arctan2(*(q - c)[::-1]) if seg > 1e-6 else heading
            c = np.array(q)
            executed.append(c.copy())
        else:
            heading = np.arctan2(*(q - c)[::-1])
            c = c + (q - c) / seg * left
            executed.append(c.copy())
            break
    if np.linalg.norm(c - goal) < 1e-6:
        break

executed = np.array(executed)
length = np.linalg.norm(np.diff(executed, axis=0), axis=1).sum()
# clearances of the executed motion against the TRUE obstacles, sampled every 2 cm
d_robot = d_cable = d_body = 1e9
for a, b in zip(executed[:-1], executed[1:]):
    for s in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.02))):
        q = a + s * (b - a)
        d_body = min(d_body, true_distance(q))
        for k in range(4):
            d_robot = min(d_robot, true_distance(q + fp.ugv_offset[k]))
            for u in np.linspace(0, 1, 30):
                d_cable = min(d_cable, true_distance(q + u * fp.ugv_offset[k]))
m = grid.map()
print(f"  reached the goal: {np.linalg.norm(c - goal) < 1e-6}; path {length:.2f} m (straight line {np.linalg.norm(goal - start):.2f} m); "
      f"{replans} plans, {1e3 * t_plan / replans:.2f} ms and {n_exp // replans} expansions each")
print(f"  map: {int((m > 0.7).sum())} occupied cells, {100 * (m != 0).mean():.0f}% of the {m.shape[1] * RES:.0f} x {m.shape[0] * RES:.0f} m grid observed")
print(f"  true clearances of the executed motion: ground robots {d_robot:.2f} m (required {fp.robot_radius}), cables {d_cable:.2f} m "
      f"(required {fp.cable_margin}), tool {d_body:.2f} m (required {fp.body_radius})")
tol = 0.03                                                           # scan and pose noise
ok = (np.linalg.norm(c - goal) < 1e-6 and d_robot > fp.robot_radius - tol and d_cable > fp.cable_margin - tol
      and d_body > fp.body_radius - tol)
np.savez(os.path.join(ROOT, "results", "core_plan.npz"), executed=executed, map=m, origin=np.array([-9.0, -9.0]), res=RES)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
