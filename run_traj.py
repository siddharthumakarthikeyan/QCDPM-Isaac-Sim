"""Fly the end-effector along a defined trajectory and record it for offline rendering (same recording format as
run_task.py, plus the reference path).

    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh run_traj.py --scenario track [--gui]
  track   3D Lissajous figure-eight (1.8 x 1.2 x 0.5 m, 24 s per lap, 2 laps): controller tracking test
  plan    RRT* -> prune -> Bezier path for the whole team (platform, ground robots, cables) past a site cabin and
          a pallet (cdpr_sim/planning.py); the obstacles are real colliders in the run
  paint   raster over the upper surface of an aircraft wing (cdpr_sim/aircraft.py), spray lance 0.15 m above the skin;
          the wing is a real collider and every sample is checked for the platform, ground robots and cables
  print   one continuous spiral for 3D printing a twisted, four-lobed wall 0.42 m high in 30 mm layers; the nozzle
          path is what is recorded (the deposited material itself is drawn in the film, not simulated)
Writes recordings/<scenario>.npz. `ref` is the commanded platform position at every recorded frame.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True, choices=["track", "plan", "paint", "print"])
ap.add_argument("--gui", action="store_true")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--seed", type=int, default=4)
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

from cdpr_sim import planning  # noqa: E402
from cdpr_sim.blocks import build_blocks  # noqa: E402
from cdpr_sim.obstacles import build_colliders  # noqa: E402
from cdpr_sim.runtime import CdprRuntime  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
cfg["task"]["pattern"] = "column"
dt_ref = cfg["sim"]["physics_dt"]
if args.scenario == "paint":
    cfg["environment"]["type"] = "hangar"
elif args.scenario == "print":
    cfg["environment"]["type"] = "site"                  # fenced construction site (cdpr_sim/site.py), local assets only

# ---------------------------------------------------------------- reference trajectory (sampled at the physics rate)
extra = {}
if args.scenario == "track":
    c, T, laps = np.array([0.0, 0.0, 1.25]), 24.0, 2
    tt = np.arange(0, laps * T + dt_ref, dt_ref)
    ramp = np.clip(tt / 4.0, 0, 1) ** 2 * (3 - 2 * np.clip(tt / 4.0, 0, 1)) * np.clip((laps * T - tt) / 4.0, 0, 1)
    w = 2 * np.pi / T
    traj = c + np.stack([0.9 * np.sin(w * tt), 0.6 * np.sin(2 * w * tt), 0.25 * np.sin(w * tt + np.pi / 2) - 0.25], 1) * np.r_[1, 1, 1]
    traj = c + (traj - c) * ramp[:, None] + np.array([0, 0, 0.0])
elif args.scenario == "paint":
    from cdpr_sim import aircraft

    obstacles = []
    z_p = aircraft.Z_TOP + 1.10                           # platform height; a 0.3 m spray lance reaches to 0.15 m above the skin
    xs_pass = np.arange(0.4, -0.201, -0.2)                # four lanes 0.2 m apart, leading edge back to the flap line: further
                                                          # aft a lower cable would cross the trailing edge (team check below)
    y0, y1 = -0.5, 2.15
    wp = []
    for i, x in enumerate(xs_pass):
        ya, yb = (y0, y1) if i % 2 == 0 else (y1, y0)
        wp += [(x, ya, z_p), (x, yb, z_p)]
    wp = np.array(wp)
    cfg["platform"]["start_position"] = [float(v) for v in wp[0]]
    team = planning.Team(cfg, aircraft.wing_boxes(), robot_radius=0.55, cable_margin=0.10)
    team.body = planning.inflate([aircraft.WING_BOX], xy=0.0, below=0.0, above=0.74)     # the tool works just above the skin
    smooth = planning.bezier_smooth(wp, team, d=0.09)
    traj = planning.resample(smooth, 0.30, dt_ref)
    centres = planning.lazy_centers(traj[::20], cfg)     # the robots hold position inside the hold region
    bad = [i for i, (q, c) in enumerate(zip(traj[::20], centres)) if not team(q, c)]
    if bad:
        raise RuntimeError(f"paint raster violates the team clearance at {len(bad)} samples, first {traj[bad[0]]}")
    extra = dict(smooth=smooth, lanes=xs_pass, lane_y=np.array([y0, y1]))
    print(f"[paint] raster {len(xs_pass)} lanes, {float(np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum()):.1f} m, platform z {z_p:.2f} m", flush=True)
elif args.scenario == "print":
    layer, n_layers, tip = 0.03, 14, 0.67                 # bead height, layers, nozzle tip below the platform centre
    th = np.linspace(0, 2 * np.pi * n_layers, 220 * n_layers)
    k = th / (2 * np.pi)                                  # layer number, continuous ("vase mode": one unbroken bead)
    r = 0.55 + 0.07 * np.cos(4 * (th - np.deg2rad(7.0) * k))          # four lobes, twisted 7 deg per layer
    path = np.stack([r * np.cos(th), r * np.sin(th), layer * (1 + np.clip(k - 1, 0, None)) + 0.01 + tip], 1)
    path[:, 2] = layer * np.maximum(k, 1.0) + 0.01 + tip  # first lap flat on the footing, then a steady climb
    cfg["platform"]["start_position"] = [float(v) for v in path[0]]
    traj = planning.resample(path, 0.30, dt_ref)
    centres = planning.lazy_centers(traj[::40], cfg)     # where the formation will be: the robots hold position if they can
    for j in range(n_layers):                             # the wall printed so far is an obstacle for the cables
        h = layer * max(j, 1)
        team = planning.Team(cfg, [((0, 0, h / 2), (1.3, 1.3, h))], robot_radius=0.55, cable_margin=0.08)
        team.body = []                                    # the nozzle works on top of the wall by design
        m = (traj[:, 2] - 0.01 - tip >= layer * max(j, 1) - 1e-9) & (traj[:, 2] - 0.01 - tip < layer * (j + 1) + 1e-9)
        bad = [i for i in np.where(m)[0][::40] if not team(traj[i], centres[i // 40])]
        if bad:
            raise RuntimeError(f"print path: a cable or ground robot would touch the wall at layer {j} ({len(bad)} samples)")
    extra = dict(smooth=path, layer=layer, n_layers=n_layers)
    print(f"[print] spiral {n_layers} layers, {float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()):.1f} m, "
          f"platform z {path[0, 2]:.2f} -> {path[-1, 2]:.2f} m", flush=True)
else:
    # travel 9 m along +y: a site cabin that the whole team has to go around, then a low pallet of blocks that the
    # ground robots straddle while the payload is lifted over it
    obstacles = [("cabin", (-0.5, -1.3, 1.1), (1.6, 1.2, 2.2)), ("pallet", (-0.6, 2.4, 0.3), (1.2, 0.8, 0.6))]
    start, goal = np.array([-0.5, -4.4, 1.3]), np.array([-0.5, 4.6, 1.3])
    cfg["platform"]["start_position"] = [float(v) for v in start]    # the team spawns at the planned start: no unplanned leg
    team = planning.Team(cfg, [(c, s) for _, c, s in obstacles], robot_radius=0.55, cable_margin=0.2)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    nodes, parent, raw = planning.rrt_star(start, goal, team, np.array([-4.2, -4.6, 1.0]), np.array([1.2, 4.8, 1.7]), rng,
                                           n_iter=3500, step=0.35, radius=0.9)
    pruned = planning.prune(raw, team)
    smooth = planning.bezier_smooth(pruned, team, d=0.9)
    traj = planning.resample(smooth, 0.35, dt_ref)
    bad = [i for i in range(0, len(traj), 20) if not team(traj[i])]              # every 7 mm of the commanded motion
    if bad:
        raise RuntimeError(f"planned trajectory violates the team clearance at {len(bad)} samples, first {traj[bad[0]]}")
    plen = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())   # noqa: E731
    extra = dict(nodes=nodes, parent=parent, raw=raw, pruned=pruned, smooth=smooth,
                 obstacles=json.dumps([[k, list(c), list(s)] for k, c, s in obstacles]))
    print(f"[plan] RRT* {len(nodes)} nodes in {time.time() - t0:.1f} s; length raw {plen(raw):.2f} -> pruned {plen(pruned):.2f} "
          f"-> smooth {plen(smooth):.2f} m; {len(raw)} -> {len(pruned)} waypoints", flush=True)

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
info = build_scene(stage, cfg)
trail, spray, tip_drop, hose = None, [], 0.0, None
if args.scenario in ("paint", "print"):                  # process tools: the scene is drawn as it is in the film
    from cdpr_sim.dressup import _mat, dress_all
    from cdpr_sim.process_tools import PAINT, Trail, build_print_head, build_spray_tool, tool_tip

    dress_all(stage, cfg, gripper=False, blocks=False)
    if args.scenario == "paint":
        aircraft.build_collider(stage)
        aircraft.build_film(stage)
        tip_drop, spray = build_spray_tool(stage, cfg)
        trail = Trail(stage, "/World/Paint", "film", 0.21, _mat(stage, "paint_film", PAINT, 0.18, coat=1.0), zfun=aircraft.wing_top,
                      clip=(-aircraft.CHORD / 2 + 0.02, aircraft.CHORD / 2 - 0.02, aircraft.Y_ROOT, aircraft.Y_TIP))
    else:
        from cdpr_sim.site import pbr_material

        from cdpr_sim.process_tools import HOSE_TOUCHDOWN, TANKER_POSE, Hose, feed_point
        from cdpr_sim.sandsite import build_tanker

        outlet = build_tanker(stage, cfg, "/World/Environment/Tanker", *TANKER_POSE)
        hose = Hose(stage, "/World/Hose", outlet, HOSE_TOUCHDOWN, _mat(stage, "hose", (0.03, 0.03, 0.035), 0.6))
        tip_drop = build_print_head(stage, cfg)
        trail = Trail(stage, "/World/Print", "bead", 0.05, pbr_material(stage, "/World/Looks/print_bead", "rough_concrete", res="1k",
                                                                         tint=(0.80, 0.79, 0.77)), height=0.031, spacing=0.02)
else:
    build_blocks(stage, cfg, 0, info["textures"]["block"])
if args.scenario == "plan":
    build_colliders(stage, obstacles)
app.update()
if args.gui:
    import carb
    from isaacsim.core.rendering_manager import ViewportManager

    carb.settings.get_settings().set("/rtx/raytracing/fractionalCutoutOpacity", True)     # translucent spray and propellers

    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[-11.0, -7.5, 6.5] if args.scenario == "plan" else ([6.0, 6.5, 3.4] if args.scenario == "paint" else ([1.5, -7.0, 3.0] if args.scenario == "print" else [-5.5, -6.5, 3.6])),
                                    target=[-0.5, 0.6, 1.3] if args.scenario == "paint" else ([1.5, 0, 0.8] if args.scenario == "print" else [-1.0, 0, 1.0]))
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(cfg["sim"]["physics_dt"])
RenderingManager.set_dt(cfg["sim"]["render_dt"])
app.update()
omni.timeline.get_timeline_interface().play()
app.update()
rt = CdprRuntime(cfg)
rt.initialize()

blocks_prim = stage.GetPrimAtPath("/World/Blocks")
all_blocks = sorted(p.GetPath().pathString for p in blocks_prim.GetChildren()) if blocks_prim else []
blk_view = rt.sim_view.create_rigid_body_view(all_blocks) if all_blocks else None
ugv_links = [f"/World/UGV_{k}/{n}" for k in range(4) for n in rt.art.shared_metatype.link_names]
tool_links = [f"/World/Platform/{n}" for n in rt.tool.view.shared_metatype.link_names]
prim_paths = tool_links + [f"/World/Drone_{k}" for k in range(4)] + ugv_links + all_blocks
frames, cables, rotors, tele, tele_t, ref = [], [], [], [], [], []
next_rec = [0.0]
phase = {"name": "HOLD", "i": 0, "t0": None}
events = []
T_HOLD = 3.0


def step():
    """Reference generator (runs every physics step) and 30 fps recorder."""
    if phase["name"] == "HOLD" and rt.sim_time > T_HOLD:
        phase["name"] = "APPROACH"
        rt.max_lin_speed = 0.45
        rt.set_platform_target(traj[0])
        events.append((rt.sim_time, 0, "APPROACH"))
    elif phase["name"] == "APPROACH" and np.linalg.norm(rt.ctrl.p_ref - traj[0]) < 1e-6 and np.linalg.norm(rt.state["v"][0]) < 0.02:
        phase["name"] = "TRACK"
        rt.max_lin_speed = 5.0                           # the trajectory itself is the reference now
        events.append((rt.sim_time, 0, "TRACK"))
    elif phase["name"] == "TRACK":
        i = phase["i"]
        if i < len(traj):
            rt.set_platform_target(traj[i])
            phase["i"] += 1
        else:
            phase["name"], phase["t0"] = "DONE", rt.sim_time
            events.append((rt.sim_time, 0, "DONE"))
    if rt.sim_time + 1e-9 < next_rec[0]:
        return
    next_rec[0] += 1.0 / args.fps
    tl = rt.tool.view.get_link_transforms().reshape(-1, 7)
    dr = rt.view.get_transforms().reshape(9, 7)[1:5]
    ug = rt.art.get_link_transforms().reshape(-1, 7)
    bl = blk_view.get_transforms().reshape(-1, 7) if blk_view is not None else np.zeros((0, 7))
    frames.append(np.vstack([tl, dr, ug, bl]).astype(np.float32))
    cables.append(np.r_[rt.last["pa"].ravel(), rt.last["pb"].ravel()].astype(np.float32))
    rotors.append(rt.quads.w.ravel().astype(np.float32))
    s = rt.state
    tilt = np.rad2deg(np.arccos(np.clip(s["R"][1:5, 2, 2], -1, 1)))
    tele.append(np.r_[rt.last["T"], rt.tool.ft_z, tilt, rt.quads.f_cmd.sum(1) / (4 * rt.quads.f_max),
                      np.linalg.norm(s["pos"][0] - rt.ctrl.p_ref), 0, 0].astype(np.float32))
    tele_t.append(rt.sim_time)
    ref.append(rt.ctrl.p_ref.astype(np.float32).copy())


i_tool = tool_links.index("/World/Platform/tool")


def deposit():
    """Lay material where the tool tip is (drawn, not simulated) and switch the spray on while tracking."""
    if trail is None:
        return
    if hose is not None:
        hose.update(feed_point(rt.tool.view.get_link_transforms().reshape(-1, 7)[i_tool], cfg))
    on = phase["name"] == "TRACK"
    for sp in spray:
        UsdGeom.Imageable(stage.GetPrimAtPath(sp)).MakeVisible() if on else UsdGeom.Imageable(stage.GetPrimAtPath(sp)).MakeInvisible()
    if on:
        trail.add(tool_tip(rt.tool.view.get_link_transforms().reshape(-1, 7)[i_tool], tip_drop))
        trail.show()


rt.step_hooks.append(step)
cable_prim = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Cables"))
wall0, last_print = time.time(), 0.0
while (args.gui is False or app.is_running()) and not (phase["name"] == "DONE" and rt.sim_time > phase["t0"] + 3.0):
    app.update()
    deposit()
    if args.gui:
        pts = np.empty((16, 3))
        pts[0::2], pts[1::2] = rt.last["pa"], rt.last["pb"]
        cable_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    if rt.sim_time - last_print > 10:
        last_print = rt.sim_time
        print(f"[traj] t={rt.sim_time:6.1f}s  {phase['name']} {phase['i']}/{len(traj)}  RTF {rt.sim_time / (time.time() - wall0):.2f}", flush=True)

tele_a, t_a = np.array(tele), np.array(tele_t)
t_tr = next(t for t, _, s in events if s == "TRACK")
t_dn = next((t for t, _, s in events if s == "DONE"), t_a[-1])
m = (t_a >= t_tr) & (t_a <= t_dn)
err = tele_a[m, 17] * 1000
print(f"[traj] tracking error: mean {err.mean():.2f} mm, rms {np.sqrt((err**2).mean()):.2f} mm, max {err.max():.2f} mm "
      f"over {t_dn - t_tr:.1f} s", flush=True)
if args.scenario == "paint":
    obstacles = [("wing", *b) for b in aircraft.wing_boxes()]
if args.scenario in ("plan", "paint"):                  # what actually happened: clearances from the recorded motion
    F, C = np.array(frames), np.array(cables).reshape(len(frames), 2, 8, 3)
    iu = [prim_paths.index(f"/World/UGV_{k}/chassis") for k in range(4)]
    ip = prim_paths.index("/World/Platform/base")
    d_robot = d_cable = d_plat = 1e9
    for _, c, sz in obstacles:
        lo, hi = np.array(c) - np.array(sz) / 2, np.array(c) + np.array(sz) / 2

        def dist(P):                                     # distance of points to the box (0 inside)
            return np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=-1)

        d_robot = min(d_robot, dist(F[:, iu, :3] * [1, 1, 0] + [0, 0, lo[2] + 0.01]).min())
        d_plat = min(d_plat, dist(F[:, ip, :3] - [0, 0, 0.67 if args.scenario == "paint" else 0.45]).min())
        for sfrac in np.linspace(0, 1, 60):
            d_cable = min(d_cable, dist(C[:, 0] + sfrac * (C[:, 1] - C[:, 0])).min())
    moved = np.abs(F[-1, iu, 2] - F[0, iu, 2]).max()
    print(f"[plan] recorded clearance to the obstacles: ground-robot centres {d_robot:.2f} m, cables {d_cable:.2f} m, "
          f"tool {'tip' if args.scenario == 'paint' else 'centre'} {d_plat:.2f} m; robot height change {moved * 1000:.1f} mm", flush=True)
    extra["clearance"] = np.array([d_robot, d_cable, d_plat])
F_ = np.array(frames)
k0, k1 = np.searchsorted(t_a, [t_tr, t_dn])
ip_ = prim_paths.index("/World/Platform/base")
travel = lambda i: float(np.linalg.norm(np.diff(F_[k0:k1, i, :2], axis=0), axis=1).sum())   # noqa: E731
tr_plat = travel(ip_)
tr_rob = [travel(prim_paths.index(n)) for n in [f"/World/UGV_{k}/chassis" for k in range(4)] + [f"/World/Drone_{k}" for k in range(4)]]
print(f"[traj] platform travelled {tr_plat:.1f} m; robots travelled {np.mean(tr_rob):.1f} m on average "
      f"({100 * np.mean(tr_rob) / max(tr_plat, 1e-9):.0f}% of the platform path)", flush=True)
os.makedirs(os.path.join(ROOT, "recordings"), exist_ok=True)
out = os.path.join(ROOT, "recordings", f"{args.scenario}.npz")
np.savez_compressed(out, frames=np.array(frames), cables=np.array(cables), rotors=np.array(rotors), tele=tele_a, t=t_a,
                    prim_paths=np.array(prim_paths), targets=np.zeros((0, 4)), ref=np.array(ref), traj=traj[:: max(1, int(round(1 / args.fps / dt_ref)))],
                    events=json.dumps(events), meta=json.dumps(dict(pattern="column", fps=args.fps, build_center=cfg["task"]["build_center"],
                                                                    placed=0, failed=[], staging=cfg["blocks"]["staging"]["center"],
                                                                    scenario=args.scenario, travel_platform=tr_plat, travel_robots=float(np.mean(tr_rob)), blocks=bool(all_blocks), env=cfg["environment"]["type"], tip_drop=float(tip_drop), err_mean_mm=float(err.mean()),
                                                                    err_rms_mm=float(np.sqrt((err**2).mean())), err_max_mm=float(err.max()))), **extra)
print(f"[traj] saved {out}: {len(frames)} frames", flush=True)
if args.gui:                                             # keep the window open; report the state so a late failure is logged
    last_print = rt.sim_time
    while app.is_running():
        app.update()
        deposit()
        pts = np.empty((16, 3))
        pts[0::2], pts[1::2] = rt.last["pa"], rt.last["pb"]
        cable_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        if rt.sim_time - last_print > 5:
            last_print = rt.sim_time
            s = rt.state
            print(f"[hold] t={rt.sim_time:6.1f}s platform {np.round(s['pos'][0], 3)} err {np.linalg.norm(s['pos'][0] - rt.ctrl.p_ref) * 1000:.1f} mm "
                  f"drone z {np.round(s['pos'][1:5, 2], 2)} T {np.round(rt.last['T'], 1)}", flush=True)
omni.timeline.get_timeline_interface().stop()
app.close()
