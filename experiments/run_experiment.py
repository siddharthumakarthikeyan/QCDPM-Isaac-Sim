"""Isaac Sim experiments that test the analytical predictions of analysis/theory.py (headless, no sensors/ROS).

  E1_straight / E1_crossed  hold 3 s; 0.3 N m yaw torque on the platform for 5 s; release; command 10 deg roll.
                            theory: yaw deflection 0.78 deg vs 0.05 deg; straight cannot exceed ~1.9 deg roll.
  E2_reconfig               start straight; try 10 deg roll (fails); level; morph crossing 0 -> +-90 deg live
                            (robots fly / drive around the platform); then 10 deg roll + 15 deg yaw (succeeds).
  E3_tilt35 / E3_tilt45     crossed layout, drones at 3.0 m; command the pose from theory.json (needs ~38 deg drone
                            tilt; infeasible for any tensions at 35 deg) with drone tilt limit 35 vs 45 deg.
Run: ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh experiments/run_experiment.py --scenario E1_straight
Writes results/<scenario>.csv (100 Hz).
"""

import argparse
import copy
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", required=True,
                    choices=["E1_straight", "E1_crossed", "E2_reconfig", "E3_tilt35", "E3_tilt45", "E3"])
parser.add_argument("--tilt", type=float, default=None, help="E3: drone tilt limit [deg]")
parser.add_argument("--drone-aware", action="store_true", help="E3: drone-aware tension caps in the controller")
parser.add_argument("--hybrid", action="store_true", help="drone winches in tension mode, UGV winches in length mode")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from cdpr_sim.runtime import CdprRuntime  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

base = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
cfg = copy.deepcopy(base)
sc = args.scenario
L = cfg["layout"]
if sc in ("E1_crossed", "E3_tilt35", "E3_tilt45", "E3"):
    L["drone_cross_angle_deg"], L["ugv_cross_angle_deg"] = 90.0, -90.0
else:
    L["drone_cross_angle_deg"], L["ugv_cross_angle_deg"] = 0.0, 0.0
if sc.startswith("E3"):
    L["drone_altitude"] = 3.0
    tilt_lim = args.tilt if args.tilt is not None else (35.0 if sc == "E3_tilt35" else 45.0)
    cfg["drone"]["control"]["max_tilt_deg"] = tilt_lim
    cfg["drone"]["control"]["tension_caps"]["enabled"] = args.drone_aware
    if sc == "E3":
        sc = f"E3_tilt{tilt_lim:g}" + ("_aware" if args.drone_aware else "") + ("_hybrid" if args.hybrid else "")
    e3 = json.load(open(os.path.join(ROOT, "results", "theory.json")))["E3_pose"]
if args.hybrid:
    cfg["winch"]["drone_cable_mode"] = "tension"

omni.usd.get_context().new_stage()
app.update()
build_scene(omni.usd.get_context().get_stage(), cfg)
app.update()
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(cfg["sim"]["physics_dt"])
RenderingManager.set_dt(cfg["sim"]["render_dt"])
app.update()
omni.timeline.get_timeline_interface().play()
app.update()
rt = CdprRuntime(cfg)
rt.initialize()


def rpy(R):
    return np.rad2deg([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])


rows = []


def log():
    if rt.step_count % 10:
        return
    s = rt.state
    tilt = np.rad2deg(np.arccos(np.clip(s["R"][1:5, 2, 2], -1, 1)))
    derr = np.linalg.norm(s["pos"][1:5] - rt.quads.p_ref, axis=1)
    f = rt.quads.f_cmd
    sat = ((f >= 0.999 * rt.quads.f_max) | (f <= 1e-6)).sum(1)        # rotors at a thrust limit
    thrust = f.sum(1) / (4 * rt.quads.f_max)                           # commanded thrust fraction
    rows.append(np.r_[rt.sim_time, s["pos"][0], rpy(s["R"][0]), rt.ctrl.p_ref, rpy(rt.ctrl.R_ref), rt.last["T"], rt.ctrl.t_des,
                      tilt, derr, np.rad2deg([rt.cross_top, rt.cross_bot]), rt.ext_wrench[5], float(rt.ctrl.feasible),
                      sat, thrust])


rt.step_hooks.append(log)

if sc.startswith("E1"):
    events = [(3.0, lambda: rt.ext_wrench.__setitem__(5, 0.3)), (8.0, lambda: rt.ext_wrench.__setitem__(5, 0.0)),
              (10.0, lambda: rt.set_platform_target(rt.p0, rpy=np.deg2rad([10, 0, 0])))]
    T_END = 22.0
elif sc == "E2_reconfig":
    def morph():
        rt.cross_top_target, rt.cross_bot_target = np.pi / 2, -np.pi / 2
    events = [(3.0, lambda: rt.set_platform_target(rt.p0, rpy=np.deg2rad([10, 0, 0]))),
              (9.0, lambda: rt.set_platform_target(rt.p0, rpy=[0, 0, 0])),
              (12.0, morph),
              (35.0, lambda: rt.set_platform_target(rt.p0, rpy=np.deg2rad([10, 0, 15])))]
    T_END = 46.0
else:
    events = [(3.0, lambda: rt.set_platform_target(rt.p0 + [0, 0, e3["dz"]], rpy=np.deg2rad(e3["rpy"])))]
    T_END = 22.0

k = 0
while rt.sim_time < T_END:
    while k < len(events) and rt.sim_time >= events[k][0]:
        events[k][1]()
        k += 1
    app.update()

cols = (["t", "x", "y", "z", "roll", "pitch", "yaw", "xr", "yr", "zr", "rollr", "pitchr", "yawr"]
        + [f"T{i}" for i in range(8)] + [f"Td{i}" for i in range(8)] + [f"tilt{i}" for i in range(4)]
        + [f"derr{i}" for i in range(4)] + ["cross_top", "cross_bot", "tau_z", "feasible"]
        + [f"sat{i}" for i in range(4)] + [f"thrust{i}" for i in range(4)])
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
np.savetxt(os.path.join(ROOT, "results", f"{sc}.csv"), np.array(rows), delimiter=",", header=",".join(cols), comments="")
print(f"[exp] {sc} done: {len(rows)} rows")
omni.timeline.get_timeline_interface().stop()
app.close()
