"""Do the ground robots stay put when the platform works off-centre with the formation held?

The platform carries a block-equivalent load (applied as an external wrench) at a given height and is moved out to a
fraction of the hold radius in several directions while the robots hold position. Logged per ground robot: how far it
is from its goal (along / across its wheels), its cable's horizontal and vertical pull, and the platform error.

    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh experiments/hold_drag.py [--z 1.7] [--frac 0.95] [--gui]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--z", type=float, default=1.7)
ap.add_argument("--frac", type=float, default=0.95, help="how far out, as a fraction of the hold radius at --z")
ap.add_argument("--hold-fraction", type=float, default=None, help="override layout.hold_fraction")
ap.add_argument("--load", type=float, default=None, help="payload [kg] (default: one block)")
ap.add_argument("--dirs", type=float, nargs="+", default=[0.0, 45.0, 135.0], help="directions [deg]")
ap.add_argument("--dwell", type=float, default=8.0)
ap.add_argument("--transit", type=float, default=0.0, help="then travel this far [m] from the start, out of the hold region")
ap.add_argument("--speed", type=float, default=0.45, help="platform speed for --transit [m/s]")
ap.add_argument("--formation-acc", type=float, default=None, help="override layout.formation_acc")
ap.add_argument("--caster-lift", type=float, default=None, help="override ugv.caster_lift [m]")
ap.add_argument("--gui", action="store_true")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from cdpr_sim.mathutil import yaw_of  # noqa: E402
from cdpr_sim.runtime import CdprRuntime  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
cfg["task"]["pattern"] = "column"
if args.hold_fraction is not None:
    cfg["layout"]["hold_fraction"] = args.hold_fraction
if args.formation_acc is not None:
    cfg["layout"]["formation_acc"] = args.formation_acc
if args.caster_lift is not None:
    cfg["ugv"]["caster_lift"] = args.caster_lift
m_load = cfg["blocks"]["mass"] if args.load is None else args.load

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
build_scene(stage, cfg)
app.update()
if args.gui:
    from isaacsim.core.rendering_manager import ViewportManager

    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[-5.5, -6.5, 3.6], target=[0.6, 0, 0.9])
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(cfg["sim"]["physics_dt"])
RenderingManager.set_dt(cfg["sim"]["render_dt"])
app.update()
omni.timeline.get_timeline_interface().play()
app.update()
rt = CdprRuntime(cfg)
rt.initialize()
r_hold = float(np.interp(args.z, *rt.hold))
print(f"[drag] hold radius at z {args.z:.2f} m: {r_hold:.3f} m (hold_fraction {cfg['layout']['hold_fraction']}), "
      f"going to {args.frac:.2f} of it, load {m_load:.1f} kg", flush=True)

_rw = rt.ctrl.required_wrench
rt.ctrl.required_wrench = lambda *a, **k: _rw(*a, **k) - rt.ext_wrench   # the load is known to the controller, as when
                                                                         # the wrist load cell measures a gripped block
log = []


def sample():
    s = rt.state
    yaw = yaw_of(s["R"][5:9])
    e = rt.ugv_cmd.goal[:, :2] - s["pos"][5:9, :2]
    fwd = np.column_stack([np.cos(yaw), np.sin(yaw)])
    lat = np.column_stack([-np.sin(yaw), np.cos(yaw)])
    F = -rt.last["T"][4:, None] * rt.last["u"][4:]                 # cable force on each ground robot
    log.append(np.r_[rt.sim_time, np.linalg.norm(e, axis=1), (e * fwd).sum(1), (e * lat).sum(1), rt.last["T"][4:],
                     np.linalg.norm(F[:, :2], axis=1), F[:, 2], (F[:, :2] * fwd).sum(1), (F[:, :2] * lat).sum(1),
                     np.linalg.norm(s["pos"][0] - rt.ctrl.p_ref), float(rt.ctrl.feasible), rt.last["T"][:4],
                     yaw, rt.ugv_cmd.goal[:, 2], rt.form_center, rt.ctrl.p_ref[:2], rt.ugv_cmd.v, rt.ugv_cmd.om])


def run(seconds, tag):
    k0, t_end = len(log), rt.sim_time + seconds
    while rt.sim_time < t_end and (not args.gui or app.is_running()):
        app.update()
        sample()
    a = np.array(log[k0:])
    last = a[a[:, 0] > a[-1, 0] - 1.0]                             # last second of the phase
    c = lambda i: slice(1 + 4 * i, 5 + 4 * i)                      # noqa: E731  (columns of the i-th per-robot block)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(f"[drag] {tag}: t {rt.sim_time:5.1f}  platform err {1e3 * last[:, 33].mean():5.1f} mm (max in phase "
          f"{1e3 * a[:, 33].max():5.1f})  feasible {a[:, 34].mean():.2f}\n"
          f"       robot off goal [m] {last[:, c(0)].mean(0)}  max in phase {a[:, c(0)].max(0)}\n"
          f"       along wheels {last[:, c(1)].mean(0)}  across {last[:, c(2)].mean(0)}\n"
          f"       T lower [N] {last[:, c(3)].mean(0)}  max {a[:, c(3)].max(0)}   T upper max {a[:, 35:39].max(0)}\n"
          f"       pull horiz [N] {last[:, c(4)].mean(0)}  up {last[:, c(5)].mean(0)}", flush=True)


p0 = rt.p0.copy()
run(3.0, "settle")
rt.set_platform_target([p0[0], p0[1], args.z])
rt.ext_wrench[2] = -m_load * rt.g
run(abs(args.z - p0[2]) / rt.max_lin_speed + 4.0, "raise + load")
for deg in args.dirs:
    a = np.deg2rad(deg)
    r = args.frac * r_hold
    rt.set_platform_target([p0[0] + r * np.cos(a), p0[1] + r * np.sin(a), args.z])
    run(r / rt.max_lin_speed + args.dwell, f"out {deg:5.1f} deg, r {r:.2f} m")
    rt.set_platform_target([p0[0], p0[1], args.z])
    run(r / rt.max_lin_speed + 3.0, "back to centre")
if args.transit > 0:                                              # leave the hold region at carrying speed
    rt.max_lin_speed = args.speed
    for deg in args.dirs:
        a = np.deg2rad(deg)
        rt.set_platform_target([p0[0] + args.transit * np.cos(a), p0[1] + args.transit * np.sin(a), args.z])
        run(args.transit / args.speed + 4.0, f"transit {deg:5.1f} deg, {args.transit:.1f} m at {args.speed} m/s")
        rt.set_platform_target([p0[0], p0[1], args.z])
        run(args.transit / args.speed + 4.0, "transit back")
print(f"[drag] formation centre moved {np.linalg.norm(rt.form_center - p0[:2]):.3f} m", flush=True)
np.save(os.path.join(ROOT, "results", "hold_drag.npy"), np.array(log))
print("[drag] done", flush=True)
if args.gui:
    while app.is_running():
        app.update()
omni.timeline.get_timeline_interface().stop()
app.close()
