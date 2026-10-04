"""Cinematic offline render of a recorded task (recordings/<name>.npz from run_task.py).

    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tools/render_video.py --rec recordings/tower.npz \
        [--quality preview|final] [--res 1920 1080]
Writes renders/<name>/frame_XXXXX.png plus renders/<name>/frames.json (sim time, speed, shot, per frame) that
tools/compose_video.py uses for overlays and editing.

Physics is not run: the scene is rebuilt and every recorded prim pose is written each frame. Shots are generated
from the task events, so close-ups always land on a grasp or a placement:
    establish  crane up over the site                          real time
    follow     chase camera on the first block                 real time (x2 during transit)
    closeup    gripper close-up for the first grasp and placement   real time
    orbit      slow orbit around the build                     time-lapse
    top        overhead                                        time-lapse
    hero       low orbit around the finished structure         static
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--rec", required=True)
ap.add_argument("--quality", default="preview", choices=["preview", "final"])
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--timelapse", type=float, default=12.0)
ap.add_argument("--max-frames", type=int, default=None)
ap.add_argument("--env", default=None, choices=["tiles", "sand", "site", "lab"], help="render in this environment instead of the config one")
ap.add_argument("--stills", action="store_true", help="one frame per shot only (look check), to renders/<name>_stills_<env>/")
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

from cdpr_sim.blocks import build_blocks, pattern  # noqa: E402
from cdpr_sim.mathutil import look_at_quat  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

rec = np.load(args.rec, allow_pickle=False)
meta = json.loads(str(rec["meta"]))
events = json.loads(str(rec["events"]))
T = rec["t"]
frames, cables = rec["frames"], rec["cables"]
paths = [str(p) for p in rec["prim_paths"]]
name = os.path.splitext(os.path.basename(args.rec))[0]
out_dir = os.path.join(ROOT, "renders", f"{name}_stills_{args.env or 'cfg'}" if args.stills else name)
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------- scene (same build as the recording)
cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
cfg["task"]["pattern"] = meta["pattern"]
cfg["task"]["build_center"] = meta["build_center"]
if args.env:                                         # physics is not run here, so the backdrop is free to differ
    cfg["environment"]["type"] = args.env
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
info = build_scene(stage, cfg)
build_blocks(stage, cfg, len(rec["targets"]), info["textures"]["block"])
app.update()

def round_cylinders(n=48):
    """RTX tessellates Cylinder gprims as octagons: turn them into smooth-shaded meshes (render only, no physics here)."""
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


round_cylinders()

ops = []
for p in paths:
    prim = stage.GetPrimAtPath(p)
    x = UsdGeom.Xformable(prim)
    tr = rt = None
    for op in x.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            tr = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            rt = op
    if tr is None or rt is None:
        carb.log_warn(f"no translate/orient op on {p}")
    ops.append((tr, rt))
cable_prim = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Cables"))


def apply_pose(k):
    f = frames[k]
    for (tr, rt), x in zip(ops, f):
        if tr is None:
            continue
        tr.Set(Gf.Vec3d(float(x[0]), float(x[1]), float(x[2])))
        rt.Set(Gf.Quatf(float(x[6]), float(x[3]), float(x[4]), float(x[5])))   # recorded xyzw -> wxyz
    c = cables[k].reshape(2, 8, 3)
    pts = np.empty((16, 3), np.float32)
    pts[0::2], pts[1::2] = c[0], c[1]
    cable_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))


# ---------------------------------------------------------------- shot plan
B = np.array([*meta["build_center"], 0.0])
S = np.array([*meta["staging"], 0.0])
i_base = paths.index("/World/Platform/base")
P = frames[:, i_base, :3].astype(np.float64)          # platform trajectory


def ev_time(block, state, default=None):
    for t, b, s in events:
        if b == block and s == state:
            return t
    return default


t_end = float(T[-1])
t_lift0 = ev_time(0, "LIFT", t_end)
t_place0 = ev_time(0, "PLACE", t_end)
t_ret0 = ev_time(0, "RETRACT", t_end)
t_grasp0 = ev_time(0, "GRASP", t_lift0)
t_done = ev_time(len(rec["targets"]), "DONE", t_end) if any(e[2] == "DONE" for e in events) else t_end
mid = (B + S) / 2
segments = [  # (shot, sim_t0, sim_t1, playback speed)
    ("establish", 0.0, 4.0, 1.0),
    ("follow", 4.0, max(4.0, t_grasp0 - 3.0), 2.0),
    ("closeup", max(4.0, t_grasp0 - 3.0), t_lift0 + 4.0, 1.0),
    ("follow", t_lift0 + 4.0, max(t_lift0 + 4.0, t_place0 - 1.0), 2.0),
    ("closeup", max(t_lift0 + 4.0, t_place0 - 1.0), t_ret0 + 2.0, 1.0),
    ("orbit", t_ret0 + 2.0, t_ret0 + 2.0 + 0.6 * (t_done - t_ret0 - 2.0), args.timelapse),
    ("top", t_ret0 + 2.0 + 0.6 * (t_done - t_ret0 - 2.0), t_done, args.timelapse),
    ("hero", t_end, t_end, 0.0),                         # static scene, 10 s of camera motion
]
plan = []
for shot, a, b, sp in segments:
    if shot == "hero":
        n = int(10.0 * args.fps)
        plan += [(shot, t_end, j / n, 0.0) for j in range(n)]
        continue
    if b <= a:
        continue
    n = max(1, int((b - a) / sp * args.fps))
    plan += [(shot, a + (b - a) * j / n, j / n, sp) for j in range(n)]
if args.max_frames:
    plan = plan[: args.max_frames]
if args.stills:                                      # middle frame of every shot
    cuts = [0] + [j for j in range(1, len(plan)) if plan[j][0] != plan[j - 1][0]] + [len(plan)]
    plan = [plan[(a + b) // 2] for a, b in zip(cuts[:-1], cuts[1:])]


def seg_dist(p, a, b):
    """Distance from point p to segments a[i]-b[i]."""
    ab = b - a
    s_ = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    return np.linalg.norm(a + s_[:, None] * ab - p, axis=-1)


CLOSE = np.array([0.75, -0.95, -0.15])                 # default close-up offset from the gripper


def closeup_yaw(a, b):
    """Rotate the close-up camera about the gripper so that no cable runs just in front of the lens and none
    crosses the line of sight near the camera (a cable 5 cm from the lens is a black bar across the frame)."""
    ks = np.unique(np.clip(np.searchsorted(T, np.linspace(a, b, 12)), 0, len(T) - 1))
    best, best_score = 0.0, -1.0
    for deg in (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180):
        c, s_ = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        off = np.array([c * CLOSE[0] - s_ * CLOSE[1], s_ * CLOSE[0] + c * CLOSE[1], CLOSE[2]])
        score = 1e9
        for k in ks:
            grip = smooth(k) - np.array([0, 0, 0.55])
            cab = cables[k].reshape(2, 8, 3).astype(np.float64)
            for f in (0.0, 0.25, 0.5):                # the eye and the first half of the view ray
                score = min(score, seg_dist(grip + off * (1 - f), cab[0], cab[1]).min())
        if score > best_score + 0.03 or (deg == 0 and score > 0.35):     # prefer small rotations
            best, best_score = deg, score
            if deg == 0 and score > 0.35:
                break
    return np.deg2rad(best)


def smooth(i, k=15):
    a, b = max(0, i - k), min(len(P), i + k + 1)
    return P[a:b].mean(0)


close_yaw = [(a, b, closeup_yaw(a, b)) for shot, a, b, sp in segments if shot == "closeup" and b > a]
print("[render] close-up camera yaw [deg]:", [round(float(np.rad2deg(y))) for _, _, y in close_yaw], flush=True)


def camera(shot, t, u):
    k = int(np.clip(np.searchsorted(T, t), 0, len(T) - 1))
    p = smooth(k)
    grip = p - np.array([0, 0, 0.55])
    if shot == "establish":
        e = u * u * (3 - 2 * u)
        eye = mid + np.array([-6.5, -8.0, 1.0 + 4.0 * e])
        return eye, mid + np.array([0, 0, 1.0]), 22.0
    if shot == "follow":
        return p + np.array([-2.6, -3.4, 0.6]), grip + np.array([0, 0, 0.1]), 28.0
    if shot == "closeup":
        yaw = next((y for a, b, y in close_yaw if a - 1e-6 <= t <= b + 1e-6), 0.0)
        c, s_ = np.cos(yaw), np.sin(yaw)
        off = np.array([c * CLOSE[0] - s_ * CLOSE[1], s_ * CLOSE[0] + c * CLOSE[1], CLOSE[2]])
        return grip + off, grip + np.array([0, 0, -0.08]), 32.0
    if shot == "orbit":
        a = np.deg2rad(-110 + 130 * u)
        return B + np.array([6.0 * np.cos(a), 6.0 * np.sin(a), 3.2]), B + np.array([0.8, 0, 0.7]), 22.0
    if shot == "top":
        a = np.deg2rad(20 * u)
        return B + np.array([0.6 * np.cos(a), 0.6 * np.sin(a), 9.5]), B + np.array([0, 0, 0.3]), 20.0
    if shot == "hero":
        a = np.deg2rad(-150 + 100 * u)
        return B + np.array([3.6 * np.cos(a), 3.6 * np.sin(a), 0.55 + 0.6 * u]), B + np.array([0, 0, 0.45]), 20.0
    raise ValueError(shot)


# ---------------------------------------------------------------- render settings
s = carb.settings.get_settings()
if args.quality == "final":
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", 32)
    s.set("/rtx/pathtracing/totalSpp", 128)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    subframes = 4
else:
    s.set("/rtx/rendermode", "RaytracedLighting")
    subframes = 2
if args.stills:
    subframes = 40                               # the camera jumps between stills: let textures and samples settle
s.set("/rtx/post/aa/op", 3)                      # DLAA
s.set("/rtx/post/tonemap/op", 6)                 # ACES-like filmic tonemapping

W, H = args.res
cam = RtxCamera("/World/Film/camera", translations=np.zeros((1, 3)), orientations=np.array([[1.0, 0, 0, 0]]))
cam_prim = cam.prims[0]
x = UsdGeom.Xformable(cam_prim)
cam_tr = cam_rt = None
for op in x.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        cam_tr = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        cam_rt = op
cam_prim.GetAttribute("horizontalAperture").Set(36.0)          # full-frame 36 x 20.25 mm for 16:9
cam_prim.GetAttribute("verticalAperture").Set(36.0 * H / W)
UsdGeom.Camera(cam_prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 5000.0))   # default near plane (1 m) cuts close-ups open
sensor = CameraSensor(cam, resolution=(H, W), annotators=["rgb"])
s.set("/app/player/playSimulations", False)                   # annotators only deliver while the timeline plays;
omni.timeline.get_timeline_interface().play()                 # keep physics off so the recorded poses stand
if plan:                                                      # warm up on the opening view, or frame 0 is blank
    apply_pose(0)
    eye, tgt, focal = camera(*plan[0][:3])
    cam_tr.Set(Gf.Vec3d(*map(float, eye)))
    cam_rt.Set(Gf.Quatd(*map(float, look_at_quat(eye, tgt))))
    cam_prim.GetAttribute("focalLength").Set(float(focal))
for _ in range(30):                                           # warm-up: textures and shaders
    app.update()

log = []
prev_eye = prev_tgt = None
for j, (shot, t, u, sp) in enumerate(plan):
    k = int(np.clip(np.searchsorted(T, t), 0, len(T) - 1))
    apply_pose(k)
    eye, tgt, focal = camera(shot, t, u)
    if prev_eye is not None and plan[j - 1][0] == shot and not args.stills:          # damp camera jitter within a shot
        eye = 0.7 * eye + 0.3 * prev_eye
        tgt = 0.7 * tgt + 0.3 * prev_tgt
    prev_eye, prev_tgt = eye, tgt
    q = look_at_quat(eye, tgt)
    cam_tr.Set(Gf.Vec3d(*map(float, eye)))
    cam_rt.Set(Gf.Quatd(*map(float, q)))           # RtxCamera authors a double-precision orient op
    cam_prim.GetAttribute("focalLength").Set(float(focal))
    for _ in range(subframes):
        app.update()
    data, _ = sensor.get_data("rgb")
    if data is None:
        carb.log_warn(f"no image for frame {j}")
        continue
    Image.fromarray(data.numpy()[..., :3]).save(os.path.join(out_dir, f"frame_{j:05d}.png"))
    log.append(dict(frame=j, t=float(t), k=k, shot=shot, speed=sp))
    if j % 60 == 0:
        print(f"[render] {name}: frame {j}/{len(plan)} shot {shot} t={t:.1f}s", flush=True)
json.dump(dict(fps=args.fps, frames=log, task=meta["pattern"], n_blocks=int(len(rec["targets"]))),
          open(os.path.join(out_dir, "frames.json"), "w"))
print(f"[render] {name}: done, {len(log)} frames -> {out_dir}", flush=True)
app.close()
