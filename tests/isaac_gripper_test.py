"""Headless end-effector test: F/T at rest, then pick the top staging block and lift it.

Checks: wrist F/T at rest ~ tool weight; block lifts with the gripper; F/T and the tension feed-forward pick up
the block weight (2.4 kg -> ~23.5 N); platform stays on its reference.
Run: ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tests/isaac_gripper_test.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from isaacsim import SimulationApp  # noqa: E402

GUI = "--headless" not in sys.argv
app = SimulationApp({"headless": not GUI, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from cdpr_sim.blocks import build_blocks, pattern  # noqa: E402
from cdpr_sim.runtime import CdprRuntime  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402
from cdpr_sim.tool import grasp_height  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
cfg["task"]["pattern"] = "column"
for kv in [a for a in sys.argv[1:] if "=" in a]:   # overrides, e.g. winch.hybrid_pd.wn=6
    k, v = kv.split("=")
    d = cfg
    for kk in k.split(".")[:-1]:
        d = d[kk]
    d[k.split(".")[-1]] = float(v)
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
info = build_scene(stage, cfg)
paths, stock = build_blocks(stage, cfg, len(pattern(cfg)), info["textures"]["block"])
app.update()
if GUI:
    from isaacsim.core.rendering_manager import ViewportManager

    x0, y0 = cfg["blocks"]["staging"]["center"]
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[x0 - 3.5, y0 - 4.5, 2.6], target=[x0 - 1.2, y0, 0.6])
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(cfg["sim"]["physics_dt"])
RenderingManager.set_dt(cfg["sim"]["render_dt"])
app.update()
omni.timeline.get_timeline_interface().play()
app.update()
rt = CdprRuntime(cfg)
rt.initialize()
blk = rt.sim_view.create_rigid_body_view([paths[0]])
geo = info["tool"]
g = 9.81


from pxr import UsdGeom, Vt  # noqa: E402

cables = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Cables"))


DLOG = []


def _drone_log():
    if rt.step_count % 10:
        return
    st = rt.state
    tilt = np.rad2deg(np.arccos(np.clip(st["R"][1:5, 2, 2], -1, 1)))
    derr = np.linalg.norm(st["pos"][1:5] - rt.quads.p_ref, axis=1)
    f = rt.quads.f_cmd
    sat = ((f >= 0.999 * rt.quads.f_max) | (f <= 1e-6)).sum(1)
    DLOG.append(np.r_[rt.sim_time, tilt, derr, sat, f.sum(1) / (4 * rt.quads.f_max), rt.last["T"], rt.ctrl.t_des,
                      rt.tool.ft_raw, rt.tool.w_on_base[2], st["pos"][0] - rt.ctrl.p_ref, float(rt.ctrl.feasible)])


rt.step_hooks.append(_drone_log)


def run_until(t):
    while rt.sim_time < t:
        app.update()
        pts = np.empty((16, 3))
        pts[0::2], pts[1::2] = rt.last["pa"], rt.last["pb"]
        cables.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))


def report(tag):  # noqa: E302
    s = rt.state
    bz = blk.get_transforms().reshape(7)[2]
    e = np.linalg.norm(s["pos"][0] - rt.ctrl.p_ref) * 1000
    print(f"[{tag:12s}] t={rt.sim_time:5.1f}  platform err {e:5.1f} mm  F/T z {rt.tool.ft_z:7.2f} N  "
          f"w_ext z {rt.tool.w_on_base[2]:7.2f} N  width {rt.tool.width * 1000:5.1f} mm  block z {bz:.3f}  "
          f"T [{rt.last['T'].min():.1f}, {rt.last['T'].max():.1f}] N", flush=True)


run_until(3.0)
report("rest")
ft_rest, w_rest = rt.tool.ft_z, rt.tool.w_on_base[2]
x, y, zc, yaw = stock[0]
rt.max_lin_speed = 0.3
z_grasp = grasp_height(cfg, zc)
rt.set_platform_target([x, y, z_grasp + 0.30])
rt.tool.set_yaw(yaw)
run_until(18.0)
report("above block")
# descend only once the platform has settled over the block (xy error < 3 mm)
while np.linalg.norm(rt.state["pos"][0, :2] - np.array([x, y])) > 0.003 or np.linalg.norm(rt.state["v"][0]) > 0.01:
    run_until(rt.sim_time + 0.1)
report("settled")
rt.max_lin_speed = 0.06
rt.set_platform_target([x, y, z_grasp])
# guarded descent: stop if the load cell sees >8 N of unexpected load change (contact)
ft0 = rt.tool.ft_z
t_end = rt.sim_time + 8.0
while rt.sim_time < t_end and abs(rt.tool.ft_z - ft0) < 8.0:
    run_until(rt.sim_time + 0.02)
if abs(rt.tool.ft_z - ft0) >= 8.0:
    rt.set_platform_target(rt.state["pos"][0] + [0, 0, 0.01])
    print(f"[guard       ] contact detected at t={rt.sim_time:.2f}, load change {rt.tool.ft_z - ft0:.1f} N", flush=True)
run_until(rt.sim_time + 1.0)
report("at grasp")
rt.tool.close()
run_until(rt.sim_time + 2.0)
report("closed")
rt.set_platform_target([x, y, z_grasp + 0.35])
run_until(rt.sim_time + 8.0)
report("lifted")
run_until(rt.sim_time + 4.0)
report("held")
print(f"  load cell {rt.tool.ft_z:.2f} N  (tool + fingers + block = "
      f"{(cfg['gripper']['tool']['mass'] + 2 * cfg['gripper']['finger']['mass'] + cfg['blocks']['mass']) * g:.2f} N)")
dz = blk.get_transforms().reshape(7)[2] - zc
d_ft = rt.tool.ft_z - ft_rest
d_w = rt.tool.w_on_base[2] - w_rest
print(f"block raised {dz * 100:.1f} cm; F/T change {d_ft:.2f} N; feed-forward change {d_w:.2f} N; "
      f"expected {cfg['blocks']['mass'] * g:.2f} N")
ok = dz > 0.25 and abs(abs(d_ft) - cfg["blocks"]["mass"] * g) < 2.0
print("PASS" if ok else "FAIL", flush=True)
np.savetxt(os.path.join(ROOT, "results", "gripper_test_log.csv"), np.array(DLOG), delimiter=",",
           header="t,tilt0,tilt1,tilt2,tilt3,derr0,derr1,derr2,derr3,sat0,sat1,sat2,sat3,thr0,thr1,thr2,thr3,"
                  + ",".join(f"T{i}" for i in range(8)) + "," + ",".join(f"Td{i}" for i in range(8))
                  + ",ft_raw,wext_z,ex,ey,ez,feasible", comments="")
if GUI:  # keep the scene live (holding the block) until the window is closed
    while app.is_running():
        app.update()
omni.timeline.get_timeline_interface().stop()
app.close()
