"""Headless physics test in Isaac Sim (no sensors, no ROS).

Holds the platform for 3 s, then commands +0.3 m lift, 10 deg roll, 15 deg yaw; then translates the whole
formation 1 m in x (UGVs drive, drones fly). Checks tracking, tension bounds and real-time factor.
Run:  ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tests/isaac_physics_test.py
"""

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

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

log = []
wall0 = time.time()
next_log = 0.0
phase = 0
while rt.sim_time < T_END:
    app.update()
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
        next_log += 0.5
wall = time.time() - wall0
log = np.array(log)
print("     t     ex     ey     ez  erot[deg]  Tmin   Tmax  drone_err ugv_err rotor_f  ugv_z_min")
for r in log:
    print("%6.1f %6.3f %6.3f %6.3f %8.2f %6.2f %6.2f %8.3f %7.3f %7.2f %7.3f" % tuple(r))
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
omni.timeline.get_timeline_interface().stop()
app.close()
sys.exit(0 if all(checks.values()) else 1)
