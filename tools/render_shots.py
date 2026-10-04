"""Render the trailer shots of one recording (tools/trailer_plan.py) with RTX, headless.

    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tools/render_shots.py --rec tower [--stills] [--only id ...]
Writes renders/trailer/<shot>/f_XXXX.png and renders/trailer/<shot>/meta.json (per frame: sim time, recording index,
camera eye / target / focal length, used by the edit to pin labels to scene points). Physics is not run.
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
ap = argparse.ArgumentParser()
ap.add_argument("--rec", required=True)
ap.add_argument("--quality", default="final", choices=["preview", "final"])
ap.add_argument("--stills", action="store_true", help="first / middle / last frame of every shot only (framing check)")
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--out", default="trailer")
ap.add_argument("--res", type=int, nargs=2, default=None, help="override the picture size (e.g. 2560 1440 for a poster still)")
ap.add_argument("--spp", type=int, default=8)          # 16 samples + OptiX denoiser is visually identical to 96 here
ap.add_argument("--subframes", type=int, default=2)
ap.add_argument("--max-frames", type=int, default=None)
ap.add_argument("--fstops", type=float, nargs="*", default=None, help="depth-of-field test: middle frame at each f-stop")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, Vt  # noqa: E402

import trailer_plan as plan  # noqa: E402
from cdpr_sim.blocks import build_blocks  # noqa: E402
from cdpr_sim.dressup import dress_all  # noqa: E402
from cdpr_sim.obstacles import build_film  # noqa: E402
from cdpr_sim.mathutil import look_at_quat  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

rec = np.load(os.path.join(ROOT, "recordings", f"{args.rec}.npz"), allow_pickle=False)
meta = json.loads(str(rec["meta"]))
T = rec["t"]
frames, cables = rec["frames"], rec["cables"]
paths = [str(p) for p in rec["prim_paths"]]

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
cfg["task"]["pattern"] = meta["pattern"]
cfg["task"]["build_center"] = meta["build_center"]
scenario = meta.get("scenario", "task")
cfg["environment"]["type"] = meta.get("env", "sand") if meta.get("env") in ("sand", "hangar", "site") else "sand"
cfg["environment"].setdefault("sand", {}).update(plan.SUN)
obstacles = json.loads(str(rec["obstacles"])) if "obstacles" in rec.files else []
if obstacles:                                         # the planning run drives through the cone line
    cfg["environment"]["sand"]["cones"] = False
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
info = build_scene(stage, cfg)
if meta.get("blocks", True):
    build_blocks(stage, cfg, len(rec["targets"]), info["textures"]["block"])
if obstacles:
    build_film(stage, cfg, obstacles)
dress_all(stage, cfg, gripper=scenario not in ("paint", "print"))      # film detail on the robots (visual only)

# process applications: the tool and the material it lays down, replayed from the recorded tool pose
trail, spray, n_rows, t_on, hose = None, [], None, (0.0, 0.0), None
if scenario in ("paint", "print"):
    from cdpr_sim.dressup import _mat
    from cdpr_sim.process_tools import PAINT, Trail, build_print_head, build_spray_tool, tool_tip
    from cdpr_sim.site import pbr_material

    if scenario == "paint":
        from cdpr_sim import aircraft

        aircraft.build_film(stage)
        tip_drop, spray = build_spray_tool(stage, cfg)
        trail = Trail(stage, "/World/Paint", "film", 0.21, _mat(stage, "paint_film", PAINT, 0.18, coat=1.0), zfun=aircraft.wing_top,
                      clip=(-aircraft.CHORD / 2 + 0.02, aircraft.CHORD / 2 - 0.02, aircraft.Y_ROOT, aircraft.Y_TIP))
    else:
        from cdpr_sim.process_tools import HOSE_TOUCHDOWN, TANKER_POSE, Hose, feed_point
        from cdpr_sim.sandsite import build_tanker

        outlet = build_tanker(stage, cfg, "/World/Environment/Tanker", *TANKER_POSE)
        hose = Hose(stage, "/World/Hose", outlet, HOSE_TOUCHDOWN, _mat(stage, "hose", (0.03, 0.03, 0.035), 0.6))
        tip_drop = build_print_head(stage, cfg)
        trail = Trail(stage, "/World/Print", "bead", 0.05, pbr_material(stage, "/World/Looks/print_bead", "rough_concrete", res="1k",
                                                                         tint=(0.80, 0.79, 0.77)), height=0.031, spacing=0.02)
    ev = json.loads(str(rec["events"]))
    t_on = (next(t for t, _, st in ev if st == "TRACK"), next((t for t, _, st in ev if st == "DONE"), float(T[-1])))
    i_tool = paths.index("/World/Platform/tool")
    n_rows = np.zeros(len(T), int)
    for k in range(len(T)):
        if t_on[0] <= T[k] <= t_on[1]:
            trail.add(tool_tip(frames[k, i_tool], tip_drop))
        n_rows[k] = len(trail.rows)
app.update()


def round_cylinders(n=48):
    """RTX tessellates Cylinder gprims as octagons: turn them into smooth-shaded meshes (render only)."""
    a = np.arange(n) * 2 * np.pi / n
    ring = np.stack([np.cos(a), np.sin(a)], -1)
    for prim in list(stage.Traverse()):
        if prim.GetTypeName() != "Cylinder":
            continue
        c = UsdGeom.Cylinder(prim)
        r, h, ax = c.GetRadiusAttr().Get(), c.GetHeightAttr().Get(), "XYZ".index(c.GetAxisAttr().Get())
        i, j = [(1, 2), (2, 0), (0, 1)][ax]

        def P(xy, z):
            v = np.zeros((len(xy), 3))
            v[:, i], v[:, j], v[:, ax] = xy[:, 0] * r, xy[:, 1] * r, z
            return v

        top, bot = P(ring, h / 2), P(ring, -h / 2)
        nrm_side, e = P(ring, 0) / r, np.eye(3)[ax]
        pts, nrm, cnt = [], [], []
        for k in range(n):
            m = (k + 1) % n
            pts += [bot[k], bot[m], top[m], top[k]]
            nrm += [nrm_side[k], nrm_side[m], nrm_side[m], nrm_side[k]]
            cnt.append(4)
        pts += list(top) + list(bot[::-1])
        nrm += [e] * n + [-e] * n
        cnt += [n, n]
        opacity = UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get()
        prim.SetTypeName("Mesh")
        m = UsdGeom.Mesh(prim)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, np.float32)))
        m.CreateFaceVertexCountsAttr(cnt)
        m.CreateFaceVertexIndicesAttr(list(range(len(pts))))
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.array(nrm, np.float32)))
        m.SetNormalsInterpolation("faceVarying")
        m.CreateSubdivisionSchemeAttr("none")
        ext = np.abs(np.array(pts)).max(0)
        m.CreateExtentAttr([Gf.Vec3f(*map(float, -ext)), Gf.Vec3f(*map(float, ext))])
        if opacity:
            m.GetDisplayOpacityAttr().Set(opacity)


round_cylinders()

ops = []
for p in paths:
    x = UsdGeom.Xformable(stage.GetPrimAtPath(p))
    tr = rt = None
    for op in x.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            tr = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            rt = op
    ops.append((tr, rt))
cable_prim = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Cables"))


def apply_pose(k):
    for (tr, rt), x in zip(ops, frames[k]):
        if tr is None:
            continue
        tr.Set(Gf.Vec3d(float(x[0]), float(x[1]), float(x[2])))
        rt.Set(Gf.Quatf(float(x[6]), float(x[3]), float(x[4]), float(x[5])))   # recorded xyzw -> wxyz
    c = cables[k].reshape(2, 8, 3)
    pts = np.empty((16, 3), np.float32)
    pts[0::2], pts[1::2] = c[0], c[1]
    cable_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
    if hose is not None:
        hose.update(feed_point(frames[k, i_tool], cfg))
    if trail is not None:
        trail.show(int(n_rows[k]))
        on = t_on[0] <= T[k] <= t_on[1]
        for sp in spray:
            UsdGeom.Imageable(stage.GetPrimAtPath(sp)).MakeVisible() if on else UsdGeom.Imageable(stage.GetPrimAtPath(sp)).MakeInvisible()


# ---------------------------------------------------------------- anchors
B = np.array([*meta["build_center"], 0.0])
S = np.array([*meta["staging"], 0.0])
FIXED = {"build": B, "staging": S, "mid": (B + S) / 2}
TRACK = {"platform": "/World/Platform/base", **{f"drone_{k}": f"/World/Drone_{k}" for k in range(4)},
         **{f"ugv_{k}": f"/World/UGV_{k}/chassis" for k in range(4)}}


def anchor(name, k, smooth):
    if not isinstance(name, str):                        # a fixed world point
        return np.asarray(name, float)
    if name in FIXED:
        return FIXED[name]
    off = np.zeros(3)
    if name == "grip":
        name, off = "platform", np.array([0, 0, -0.55])
    i = paths.index(TRACK[name])
    a, b = max(0, k - smooth), min(len(T), k + smooth + 1)
    return frames[a:b, i, :3].astype(np.float64).mean(0) + off


def camera(sh, u, k):
    lerp = lambda v: v[0] + (v[1] - v[0]) * u   # noqa: E731
    az = np.deg2rad(lerp(sh["az"]))
    eye = anchor(sh["anchor"], k, sh["smooth"]) + np.array([lerp(sh["r"]) * np.cos(az), lerp(sh["r"]) * np.sin(az), lerp(sh["z"])])
    tgt = anchor(sh["look_anchor"], k, sh["smooth"]) + np.array(sh["look"])
    return eye, tgt, lerp(sh["focal"])


# ---------------------------------------------------------------- render settings
s = carb.settings.get_settings()
if args.quality == "final":
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", args.spp)
    s.set("/rtx/pathtracing/totalSpp", args.spp * args.subframes)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    subframes = args.subframes
else:
    s.set("/rtx/rendermode", "RaytracedLighting")
    subframes = 2
if args.stills:
    subframes = 30
for key in ("/rtx/raytracing/fractionalCutoutOpacity", "/rtx/pathtracing/fractionalCutoutOpacity"):
    s.set(key, True)                                     # translucent propeller blur discs
s.set("/rtx/post/aa/op", 3)
s.set("/rtx/post/tonemap/op", 6)

W, H = args.res or plan.RES
cam = RtxCamera("/World/Film/camera", translations=np.zeros((1, 3)), orientations=np.array([[1.0, 0, 0, 0]]))
cam_prim = cam.prims[0]
cam_tr = cam_rt = None
for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        cam_tr = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        cam_rt = op
cam_prim.GetAttribute("horizontalAperture").Set(36.0)
cam_prim.GetAttribute("verticalAperture").Set(36.0 * H / W)
UsdGeom.Camera(cam_prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 5000.0))
sensor = CameraSensor(cam, resolution=(H, W), annotators=["rgb"])
s.set("/app/player/playSimulations", False)
omni.timeline.get_timeline_interface().play()


def set_cam(eye, tgt, focal, fstop=0.0):
    c = UsdGeom.Camera(cam_prim)
    c.GetFStopAttr().Set(float(fstop))                   # 0 = pinhole (everything sharp)
    c.GetFocusDistanceAttr().Set(float(np.linalg.norm(np.asarray(tgt) - np.asarray(eye))))
    cam_tr.Set(Gf.Vec3d(*map(float, eye)))
    cam_rt.Set(Gf.Quatd(*map(float, look_at_quat(eye, tgt))))
    cam_prim.GetAttribute("focalLength").Set(float(focal))


import time  # noqa: E402

todo = [sh for sh in plan.shots() if sh["rec"] == args.rec and (not args.only or sh["id"] in args.only)]
warm = True
for sh in todo:
    n = max(2, int(round(sh["dur"] * plan.FPS)))
    idx = [0, n // 2, n - 1] if args.stills else range(n)
    if args.fstops:
        idx = [n // 2] * len(args.fstops)
    if args.max_frames:
        idx = list(idx)[: args.max_frames]
    out_dir = os.path.join(ROOT, "renders", args.out + ("_stills" if args.stills else ""), sh["id"])
    os.makedirs(out_dir, exist_ok=True)
    log, t0 = [], time.time()
    for q, j in enumerate(idx):
        if args.fstops:
            sh = dict(sh, fstop=args.fstops[q])
        u = j / (n - 1)
        t = sh["t"][0] + (sh["t"][1] - sh["t"][0]) * u
        k = int(np.clip(np.searchsorted(T, t), 0, len(T) - 1))
        apply_pose(k)
        eye, tgt, focal = camera(sh, u, k)
        set_cam(eye, tgt, focal, sh.get("fstop", 0.0))
        for _ in range(40 if warm else (12 if j == idx[0] else subframes)):   # settle after a cut
            app.update()
        warm = False
        data, _ = sensor.get_data("rgb")
        if data is None:
            carb.log_warn(f"no image for {sh['id']} frame {j}")
            continue
        Image.fromarray(data.numpy()[..., :3]).save(os.path.join(out_dir, f"f_{j:04d}.png" if not args.fstops else f"f_stop{q}.png"), compress_level=1)
        log.append(dict(f=j, t=float(t), k=k, eye=[float(v) for v in eye], tgt=[float(v) for v in tgt], focal=float(focal)))
    speed = (sh["t"][1] - sh["t"][0]) / sh["dur"]
    # rec_id ties the frames to the exact recording they were rendered from (the editor draws data from it over them)
    json.dump(dict(shot=sh, rec=args.rec, rec_id=[len(T), float(T[-1])], fps=plan.FPS, speed=speed, frames=log),
              open(os.path.join(out_dir, "meta.json"), "w"))
    print(f"[shots] {sh['id']}: {len(log)} frames, {(time.time() - t0) / max(len(log), 1):.2f} s/frame", flush=True)
app.close()
