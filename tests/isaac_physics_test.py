"""Headless physics test in Isaac Sim (no sensors, no ROS).

Holds the platform for 3 s, then commands +0.3 m lift, 10 deg roll, 15 deg yaw; then translates the whole
formation 1 m in x (UGVs drive, drones fly). Checks tracking, tension bounds and real-time factor.
Run:  ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tests/isaac_physics_test.py [--gui] [--core | --core-truth]
  --core        C++ core in the loop: simulated sensors -> estimators -> controllers
  --core-truth  C++ controllers on the true state
"""

import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--gui", action="store_true")
ap.add_argument("--core", action="store_true")
ap.add_argument("--core-truth", action="store_true")
args, _ = ap.parse_known_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from cdpr_sim.mathutil import rot_to_quat  # noqa: E402
from cdpr_sim.runtime import CdprRuntime  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
T_END = float(os.environ.get("T_END", 22.0))
LOG_DT = float(os.environ.get("LOG_DT", 0.5))
if args.core or args.core_truth:
    cfg.setdefault("core", {}).update(enabled=True, use_truth=args.core_truth)
for kv in os.environ.get("OVERRIDES", "").split():       # e.g. OVERRIDES="core.k_adm=0.002 layout.hold_fraction=0.5"
    key, val = kv.split("=")
    d = cfg
    for part in key.split(".")[:-1]:
        d = d.setdefault(part, {})
    d[key.split(".")[-1]] = yaml.safe_load(val)

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
build_scene(stage, cfg)
app.update()
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(cfg["sim"]["physics_dt"])
RenderingManager.set_dt(cfg["sim"]["render_dt"])
app.update()

omni.timeline.get_timeline_interface().play()
app.update()
rt = CdprRuntime(cfg)
rt.initialize()
print("initial tension targets", np.round(rt.ctrl.t_des, 2))
if args.gui:
    from isaacsim.core.rendering_manager import ViewportManager
    from pxr import UsdGeom, Vt

    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[-5.5, -6.5, 3.6], target=[0.0, 0, 1.6])
    cable_prim = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Cables"))


def draw_cables():
    pts = np.empty((16, 3))
    pts[0::2], pts[1::2] = rt.last["pa"], rt.last["pb"]
    cable_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))


log, est_log = [], []
slack_t = []
stat = dict(n=0, slack=0, low=0, e2=0.0, emax=0.0, r_sum=0.0, rmax=0.0, tmin=1e9)


def every_step():
    """Tension and tracking statistics over every physics step after the start-up second."""
    if rt.sim_time < 1.0:
        return
    T, s = rt.last["T"], rt.state
    e = np.linalg.norm(s["pos"][0] - rt.ctrl.p_ref)
    r = np.rad2deg(2 * np.linalg.norm(rot_to_quat(rt.ctrl.R_ref.T @ s["R"][0])[1:]))
    stat["n"] += 1
    stat["slack"] += T.min() <= 0.0
    if T.min() <= 0.0 and (not slack_t or rt.sim_time - slack_t[-1][1] > 0.05):
        slack_t.append([rt.sim_time, rt.sim_time, int(T.argmin())])
    elif T.min() <= 0.0:
        slack_t[-1][1] = rt.sim_time
    stat["low"] += T.min() < 2.0
    stat["tmin"] = min(stat["tmin"], T.min())
    stat["e2"] += e * e
    stat["emax"] = max(stat["emax"], e)
    stat["r_sum"] += r
    stat["rmax"] = max(stat["rmax"], r)


rt.step_hooks.append(every_step)
wall0 = time.time()
next_log = 0.0
phase = 0
while rt.sim_time < T_END:
    app.update()
    if args.gui:
        draw_cables()
    t = rt.sim_time
    if phase == 0 and t > 3.0:
        crossed = abs(cfg["layout"]["drone_cross_angle_deg"]) > 1e-6  # straight cables: level only
        rt.set_platform_target(rt.p0 + [0, 0, 0.3], rpy=np.deg2rad([10, 0, 15] if crossed else [0, 0, 0]))
        phase = 1
    if phase == 1 and t > 11.0:
        rt.set_platform_target(rt.p0 + [1.0, 0, 0.3], rpy=np.deg2rad([0, 0, 0]))
        phase = 2
    if t >= next_log:
        s = rt.state
        e_p = s["pos"][0] - rt.ctrl.p_ref
        Rerr = rt.ctrl.R_ref.T @ s["R"][0]
        e_r = np.rad2deg(2 * np.linalg.norm(rot_to_quat(Rerr)[1:]))
        T = rt.last["T"]
        d_err = np.linalg.norm(s["pos"][1:5] - rt.quads.p_ref, axis=1).max()
        u_err = np.linalg.norm(s["pos"][5:9, :2] - rt.ugv_cmd.goal[:, :2], axis=1).max()
        log.append((t, *e_p, e_r, T.min(), T.max(), d_err, u_err, rt.quads.f_cmd.max(), s["pos"][5:9, 2].min()))
        if rt.core is not None and not args.core_truth:
            est = rt.core.core.est
            dR = est.R.T @ s["R"][0]
            est_log.append((t, np.linalg.norm(est.p - s["pos"][0]) * 1e3, np.rad2deg(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))),
                            max(np.linalg.norm(est.drone[k].p - s["pos"][1 + k]) for k in range(4)) * 1e3,
                            np.linalg.norm(est.anchors[4:] - rt.last["pb"][4:], axis=1).max() * 1e3, est.d_hat[2],
                            rt.core.core.observer.rejected))
        next_log += LOG_DT
wall = time.time() - wall0
log = np.array(log)
print("     t     ex     ey     ez  erot[deg]  Tmin   Tmax  drone_err ugv_err rotor_f  ugv_z_min")
for r in log:
    print("%6.1f %6.3f %6.3f %6.3f %8.2f %6.2f %6.2f %8.3f %7.3f %7.2f %7.3f" % tuple(r))
if est_log:
    print("estimators:  t  platform[mm]  platform[deg]  drone[mm]  ugv[mm]  d_hat_z[N]  rejected")
    for r in est_log[:: 4 if LOG_DT >= 0.5 else 1]:
        print("          %5.1f %10.2f %12.3f %10.2f %8.2f %10.2f %8d" % tuple(r))
    e = np.array(est_log)
    print(f"platform estimate error: mean {e[:, 1].mean():.2f} mm, max {e[:, 1].max():.2f} mm, {e[:, 2].max():.3f} deg max")
n = max(stat["n"], 1)
print(f"every step (t > 1 s): position error rms {np.sqrt(stat['e2'] / n) * 1e3:.2f} mm, max {stat['emax'] * 1e3:.2f} mm; rotation mean "
      f"{stat['r_sum'] / n:.3f} deg, max {stat['rmax']:.3f} deg; a cable slack {100 * stat['slack'] / n:.2f}% of the time, below 2 N "
      f"{100 * stat['low'] / n:.2f}%; lowest tension {stat['tmin']:.2f} N")
print("slack episodes (start, end, cable):", [(round(a, 2), round(b, 2), c) for a, b, c in slack_t][:20])
print(f"real-time factor: {rt.sim_time / wall:.2f} ({rt.step_count} physics steps in {wall:.1f} s)")
hold = log[(log[:, 0] > 9.0) & (log[:, 0] < 11.0)]
final = log[-1]
checks = {
    "finite": np.all(np.isfinite(log)),
    "pose_hold_after_rotation": np.all(np.abs(hold[:, 1:4]) < 0.01) and np.all(hold[:, 4] < 1.0),
    "final_position": np.all(np.abs(final[1:4]) < 0.02) and final[4] < 1.0,
    "cables_taut": log[2:, 5].min() > 1.0,
    "ugv_on_ground": log[:, 10].min() > 0.10,
}
for k, v in checks.items():
    print(f"{k:28s} {'PASS' if v else 'FAIL'}")
if args.gui:                                             # leave the window open
    while app.is_running():
        app.update()
        draw_cables()
omni.timeline.get_timeline_interface().stop()
app.close()
sys.exit(0 if all(checks.values()) else 1)
