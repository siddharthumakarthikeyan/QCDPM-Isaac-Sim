"""Run one block-assembly task and record it for offline rendering.

    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh run_task.py --pattern dome [--gui] [--build-center X Y]
Writes recordings/<name>.npz: 30 fps poses of every moving prim, cable end points, rotor speeds and telemetry.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--pattern", required=True, choices=["dome", "compound", "tower", "pyramid", "column"])
ap.add_argument("--name", default=None)
ap.add_argument("--gui", action="store_true")
ap.add_argument("--build-center", type=float, nargs=2, default=None)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--max-blocks", type=int, default=None)
ap.add_argument("--hold-fraction", type=float, default=0.0, help="hold-region rule for the robots (0 = formation centred on the platform)")
ap.add_argument("--core", action="store_true", help="C++ core in the loop: simulated sensors -> estimators -> controllers")
ap.add_argument("--core-truth", action="store_true", help="C++ controllers on the true state")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import UsdGeom, Vt  # noqa: E402

from cdpr_sim.assembly import AssemblyTask  # noqa: E402
from cdpr_sim.blocks import build_blocks, pattern  # noqa: E402
from cdpr_sim.runtime import CdprRuntime  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
cfg["task"]["pattern"] = args.pattern
if args.core or args.core_truth:
    cfg.setdefault("core", {}).update(enabled=True, use_truth=args.core_truth)
# Block handling keeps the formation centred on the platform unless --hold-fraction is given (docs/RESEARCH_NOTES.md
# sec. 10-11: the hold rule needs the acceleration-limited formation centre and lifted casters to work with a block).
cfg["layout"]["hold_fraction"] = args.hold_fraction
if args.build_center:
    cfg["task"]["build_center"] = list(args.build_center)
name = args.name or args.pattern
targets = pattern(cfg)[: args.max_blocks]

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
info = build_scene(stage, cfg)
paths, stock = build_blocks(stage, cfg, len(targets), info["textures"]["block"])
app.update()
if args.gui:
    from isaacsim.core.rendering_manager import ViewportManager

    bx, by = cfg["task"]["build_center"]
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[bx - 5.5, by - 6.5, 3.6], target=[bx + 1.2, by, 0.8])
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(cfg["sim"]["physics_dt"])
RenderingManager.set_dt(cfg["sim"]["render_dt"])
app.update()
omni.timeline.get_timeline_interface().play()
app.update()
rt = CdprRuntime(cfg)
rt.initialize()
task = AssemblyTask(rt, cfg, stock, targets, paths)

# ---------------------------------------------------------------- recorder
all_blocks = sorted(p.GetPath().pathString for p in stage.GetPrimAtPath("/World/Blocks").GetChildren())
blk_view = rt.sim_view.create_rigid_body_view(all_blocks)
ugv_links = [f"/World/UGV_{k}/{n}" for k in range(4) for n in rt.art.shared_metatype.link_names]
tool_links = [f"/World/Platform/{n}" for n in rt.tool.view.shared_metatype.link_names]
drones = [f"/World/Drone_{k}" for k in range(4)]
prim_paths = tool_links + drones + ugv_links + all_blocks
frames, cables, rotors, tele, tele_t = [], [], [], [], []
next_rec = [0.0]


def record():
    if rt.sim_time + 1e-9 < next_rec[0]:
        return
    next_rec[0] += 1.0 / args.fps
    tl = rt.tool.view.get_link_transforms().reshape(-1, 7)
    dr = rt.view.get_transforms().reshape(9, 7)[1:5]
    ug = rt.art.get_link_transforms().reshape(-1, 7)
    bl = blk_view.get_transforms().reshape(-1, 7)
    frames.append(np.vstack([tl, dr, ug, bl]).astype(np.float32))          # xyzw quaternions (PhysX order)
    cables.append(np.r_[rt.last["pa"].ravel(), rt.last["pb"].ravel()].astype(np.float32))
    rotors.append(rt.quads.w.ravel().astype(np.float32))
    s = rt.state
    tilt = np.rad2deg(np.arccos(np.clip(s["R"][1:5, 2, 2], -1, 1)))
    tele.append(np.r_[rt.last["T"], rt.tool.ft_z, tilt, rt.quads.f_cmd.sum(1) / (4 * rt.quads.f_max),
                      np.linalg.norm(s["pos"][0] - rt.ctrl.p_ref), len(task.placed), task.i].astype(np.float32))
    tele_t.append(rt.sim_time)


rt.step_hooks.append(record)
cable_prim = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Cables"))
wall0 = time.time()
last_print = 0.0
while not task.done and (args.gui is False or app.is_running()):
    task.tick()
    app.update()
    if args.gui:
        pts = np.empty((16, 3))
        pts[0::2], pts[1::2] = rt.last["pa"], rt.last["pb"]
        cable_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    if rt.sim_time - last_print > 30:
        last_print = rt.sim_time
        print(f"[run] t={rt.sim_time:6.1f}s  block {task.i}/{len(targets)}  state {task.state}  RTF {rt.sim_time / (time.time() - wall0):.2f}", flush=True)
# a few seconds of the finished structure for the edit
t_end = rt.sim_time + 4.0
while rt.sim_time < t_end:
    app.update()

os.makedirs(os.path.join(ROOT, "recordings"), exist_ok=True)
out = os.path.join(ROOT, "recordings", f"{name}.npz")
np.savez_compressed(out, frames=np.array(frames), cables=np.array(cables), rotors=np.array(rotors), tele=np.array(tele),
                    t=np.array(tele_t), prim_paths=np.array(prim_paths), targets=np.array(targets),
                    events=json.dumps(task.events), meta=json.dumps(dict(pattern=args.pattern, fps=args.fps,
                                                                         build_center=cfg["task"]["build_center"],
                                                                         placed=len(task.placed), failed=task.failed,
                                                                         staging=cfg["blocks"]["staging"]["center"])))
print(f"[run] saved {out}: {len(frames)} frames, {len(task.placed)}/{len(targets)} placed, failed {task.failed}", flush=True)
if args.gui:
    while app.is_running():
        app.update()
omni.timeline.get_timeline_interface().stop()
app.close()
