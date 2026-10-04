"""Edit the rendered shots (tools/render_shots.py) into the thesis trailer.

    python3 tools/compose_trailer.py                 -> renders/trailer.mp4 (1920x1080, 2.39:1 picture, H.264)
    python3 tools/compose_trailer.py --sample 12     -> renders/trailer_samples/*.jpg, 12 frames spread over the edit
    python3 tools/compose_trailer.py --src trailer_stills --sample 20     (typography check on the framing stills)
The edit, texts and credits are in tools/trailer_plan.py. Everything shown as a number comes from the recording
(cable tensions, wrist load cell, platform error), read at the simulation time of the frame.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trailer_plan as plan  # noqa: E402

ROOT = plan.ROOT
FPS = plan.FPS
W, H = 1920, 1080
PW, PH = plan.RES
Y0 = (H - PH) // 2
LATO = "/usr/share/fonts/truetype/lato/Lato-%s.ttf"
MONO = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf"
ORANGE, BLUE, WHITE, GREY = (255, 122, 26), (110, 178, 240), (246, 246, 244), (176, 180, 186)
_fonts = {}


def font(weight, size):
    key = (weight, int(size))
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(MONO if weight == "Mono" else LATO % weight, int(size))
    return _fonts[key]


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def tracked(d, xy, text, f, fill, tracking=3, anchor="l"):
    w = sum(d.textlength(ch, font=f) + tracking for ch in text) - tracking
    x, y = xy
    if anchor == "m":
        x -= w / 2
    elif anchor == "r":
        x -= w
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + tracking
    return w


def rgba(c, a):
    return (*c[:3], int(255 * np.clip(a, 0, 1)))


# ------------------------------------------------------------------------------------------------ sources

class Shot:
    def __init__(self, src, sid):
        self.dir = os.path.join(ROOT, "renders", src, sid)
        self.meta = json.load(open(os.path.join(self.dir, "meta.json")))
        self.frames = {f["f"]: f for f in self.meta["frames"]}
        self.idx = np.array(sorted(self.frames))
        self.dur = self.meta["shot"]["dur"]
        self.rec = REC(self.meta["rec"])
        rid = [len(self.rec["t"]), float(self.rec["t"][-1])]
        if "rec_id" in self.meta and (self.meta["rec_id"][0] != rid[0] or abs(self.meta["rec_id"][1] - rid[1]) > 1e-6):
            STALE.append(sid)
        elif "rec_id" not in self.meta and len(self.rec["t"]) <= max(f["k"] for f in self.meta["frames"]):
            STALE.append(sid)

    def at(self, t):
        """Frame record nearest to film time t [s] within the shot."""
        j = int(round(t * FPS))
        j = int(self.idx[np.abs(self.idx - j).argmin()])
        return self.frames[j], os.path.join(self.dir, f"f_{j:04d}.png")


_recs = {}
STALE = []          # shots whose frames were rendered from a different recording than the one on disk now


def REC(name):
    if name not in _recs:
        r = np.load(os.path.join(ROOT, "recordings", f"{name}.npz"))
        paths = [str(p) for p in r["prim_paths"]]
        _recs[name] = dict(t=r["t"], tele=r["tele"], cables=r["cables"], frames=r["frames"], paths=paths,
                           n=len(r["targets"]), meta=json.loads(str(r["meta"])), ev=json.loads(str(r["events"])),
                           **{k: r[k] for k in ("ref", "nodes", "parent", "raw", "pruned", "smooth") if k in r.files},
                           obstacles=json.loads(str(r["obstacles"])) if "obstacles" in r.files else [])
    return _recs[name]


def project(fr, p):
    """World point -> picture pixel (x, y) and depth, with the camera of frame record fr."""
    eye, tgt = np.array(fr["eye"]), np.array(fr["tgt"])
    f = tgt - eye
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.999 else np.array([1.0, 0, 0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    d = np.asarray(p, float) - eye
    z = d @ f
    fx = fr["focal"] / 36.0 * PW
    return PW / 2 + fx * (d @ x) / max(z, 1e-6), PH / 2 - fx * (d @ y) / max(z, 1e-6), z


def project_n(fr, P):
    """Vectorised project(): (n, 3) world points -> (n, 3) [x, y on the full 1080p frame, depth]."""
    eye, tgt = np.array(fr["eye"]), np.array(fr["tgt"])
    f = tgt - eye
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.999 else np.array([1.0, 0, 0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    d = np.asarray(P, float) - eye
    z = d @ f
    fx = fr["focal"] / 36.0 * PW
    zz = np.maximum(z, 1e-6)
    return np.stack([PW / 2 + fx * (d @ x) / zz, Y0 + PH / 2 - fx * (d @ y) / zz, z], -1)


def polyline(d, fr, P, fill, width=2, closed=False):
    q = project_n(fr, P)
    if closed:
        q = np.vstack([q, q[:1]])
    run = []
    for x, y, z in q:
        if z > 0.3 and -2000 < x < W + 2000 and -2000 < y < H + 2000:
            run.append((float(x), float(y)))
        else:
            if len(run) > 1:
                d.line(run, fill=fill, width=width, joint="curve")
            run = []
    if len(run) > 1:
        d.line(run, fill=fill, width=width, joint="curve")


def panel(d, x0, y0, x1, y1, a):
    d.rounded_rectangle((x0, y0, x1, y1), 12, fill=(8, 10, 12, int(175 * a)))


_ws = {}


def o_workspace(d, fr, u, a, sh=None):
    """Wrench-feasible positions of the platform (LP), then the Random Forest's prediction of the same set."""
    if not _ws:
        w = np.load(os.path.join(ROOT, "results", "workspace_ml.npz"))
        show = w["feas"] | w["pred"]
        _ws.update(P=w["P"][show], feas=w["feas"][show], pred=w["pred"][show], acc=float(w["acc"]), n_train=len(w["train"]),
                   n_test=len(w["test"]), mass=float(w["mass"]), speedup=float(w["t_lp"] / w["t_pred"]))
    P, feas, pred = _ws["P"], _ws["feas"], _ws["pred"]
    if sh is not None:                                    # the region belongs to the formation: draw it around the robots
        rec = sh.rec
        c = np.mean([rec["frames"][fr["k"], rec["paths"].index(f"/World/UGV_{i}/chassis"), :2] for i in range(4)], axis=0)
        P = P + np.r_[c, 0.0]
    q = project_n(fr, P)
    learned = u > 0.5
    zr = (P[:, 2] - P[:, 2].min()) / np.ptp(P[:, 2])
    grow = ease(u / 0.38) if not learned else 1.0
    vis = (q[:, 2] > 0.5) & (zr <= grow + 1e-6) & ((pred | feas) if learned else feas)
    TEAL = (72, 214, 190)
    for i in np.argsort(-q[:, 2]):
        if not vis[i]:
            continue
        r = float(np.clip(26.0 / q[i, 2], 1.6, 4.2))
        col = TEAL if (not learned or pred[i] == feas[i]) else ORANGE
        d.ellipse((q[i, 0] - r, q[i, 1] - r, q[i, 0] + r, q[i, 1] + r), fill=rgba(col, (0.55 if col == TEAL else 0.95) * a))
    x1, y0 = W - 96, Y0 + 44
    x0 = x1 - 500
    panel(d, x0, y0, x1, y0 + (150 if learned else 112), a)
    if not learned:
        tracked(d, (x0 + 22, y0 + 16), "WRENCH-FEASIBLE WORKSPACE", font("Bold", 15), rgba(GREY, a), 3)
        d.text((x0 + 22, y0 + 42), f"{int(feas.sum()):,} of 15,625 positions hold the load", font=font("Semibold", 22), fill=rgba(WHITE, a))
        d.text((x0 + 22, y0 + 76), f"{_ws['mass']:.1f} kg, solved for cable tensions one by one", font=font("Regular", 18), fill=rgba(GREY, a))
    else:
        tracked(d, (x0 + 22, y0 + 16), "RANDOM FOREST SURROGATE", font("Bold", 15), rgba(GREY, a), 3)
        d.text((x0 + 22, y0 + 40), f"{_ws['acc'] * 100:.1f}%", font=font("Black", 46), fill=rgba(WHITE, a))
        d.text((x0 + 178, y0 + 50), f"correct on {_ws['n_test']:,} unseen positions", font=font("Regular", 19), fill=rgba(WHITE, a))
        d.text((x0 + 178, y0 + 76), f"trained on {_ws['n_train']:,}", font=font("Regular", 19), fill=rgba(GREY, a))
        d.text((x0 + 22, y0 + 108), f"about {int(round(_ws['speedup'], -1))}\u00d7 faster than the solver", font=font("Semibold", 20), fill=rgba(TEAL, a))
        d.ellipse((x1 - 150, y0 + 116, x1 - 140, y0 + 126), fill=rgba(ORANGE, a))
        d.text((x1 - 132, y0 + 110), "mispredicted", font=font("Regular", 16), fill=rgba(GREY, a))


def _trail(d, fr, P, col, a, width=4):
    n = len(P)
    for j in range(0, n - 1, 6):                           # fading tail, brightest at the head
        polyline(d, fr, P[j:j + 7], rgba(col, a * (0.15 + 0.85 * (j / max(n - 1, 1)) ** 1.5)), width)


def o_track(d, sh, fr, a):
    """Commanded path, the path actually flown, and the error between them."""
    rec = sh.rec
    k = fr["k"]
    t = rec["t"]
    t0 = next(tt for tt, _, st in rec["ev"] if st == "TRACK")
    t1 = next(tt for tt, _, st in rec["ev"] if st == "DONE")
    ka, kb = np.searchsorted(t, [t0, t1])
    polyline(d, fr, rec["ref"][ka:kb:3], rgba(WHITE, 0.55 * a), 2)
    ib = rec["paths"].index("/World/Platform/base")
    _trail(d, fr, rec["frames"][max(ka, k - 240):k + 1, ib, :3], ORANGE, a)
    x, y, z = project_n(fr, rec["ref"][k][None])[0]
    d.ellipse((x - 9, y - 9, x + 9, y + 9), outline=rgba(WHITE, a), width=2)
    err = rec["tele"][max(0, k - 4):k + 5, 17].mean() * 1000
    x1, y1 = W - 96, Y0 + PH - 40
    x0, y0 = x1 - 430, y1 - 218
    panel(d, x0, y0, x1, y1, a)
    tracked(d, (x0 + 22, y0 + 16), "TRACKING ERROR", font("Bold", 15), rgba(GREY, a), 3)
    d.text((x0 + 22, y0 + 38), f"{err:4.1f}", font=font("Black", 62), fill=rgba(WHITE, a))
    d.text((x0 + 22 + d.textlength(f"{err:4.1f}", font=font("Black", 62)) + 10, y0 + 70), "mm", font=font("Regular", 26), fill=rgba(GREY, a))
    tr = rec["tele"][ka:kb, 17] * 1000
    lx0, lx1, ly0, ly1 = x0 + 200, x1 - 24, y0 + 50, y0 + 112
    hi = max(10.0, float(tr.max()) * 1.1)
    n_now = int(np.clip(k - ka, 1, len(tr) - 1))
    pts = [(lx0 + (lx1 - lx0) * j / (len(tr) - 1), ly1 - (ly1 - ly0) * float(tr[j]) / hi) for j in range(0, n_now + 1, 4)]
    d.line((lx0, ly1, lx1, ly1), fill=(90, 96, 104, int(200 * a)), width=1)
    if len(pts) > 1:
        d.line(pts, fill=rgba(ORANGE, a), width=2)
    m = rec["meta"]
    d.text((x0 + 22, y0 + 128), f"mean {m['err_mean_mm']:.1f} mm   max {m['err_max_mm']:.1f} mm", font=font("Semibold", 21), fill=rgba(WHITE, a))
    d.text((x0 + 22, y0 + 158), f"over the whole {t1 - t0:.0f} s flight", font=font("Regular", 17), fill=rgba(GREY, a))
    d.line((x0 + 22, y0 + 196, x0 + 52, y0 + 196), fill=rgba(WHITE, 0.6 * a), width=2)
    d.text((x0 + 60, y0 + 185), "commanded", font=font("Regular", 15), fill=rgba(GREY, a))
    d.line((x0 + 170, y0 + 196, x0 + 200, y0 + 196), fill=rgba(ORANGE, a), width=4)
    d.text((x0 + 208, y0 + 185), "flown", font=font("Regular", 15), fill=rgba(GREY, a))


def _box_wire(d, fr, c, sz, fill, width=2):
    c, h = np.asarray(c, float), np.asarray(sz, float) / 2
    v = np.array([[sx, sy, sz_] for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]) * h + c
    for i, j in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
        polyline(d, fr, v[[i, j]], fill, width)


def o_plan(d, sh, fr, u, a):
    """The planner at work: RRT* tree grows, then the raw, pruned and smoothed paths."""
    rec = sh.rec
    N, par = rec["nodes"], rec["parent"]
    for _, c, sz in rec["obstacles"]:
        _box_wire(d, fr, c, sz, rgba((255, 80, 60), 0.9 * a))
    q = project_n(fr, N)
    n_show = int(len(N) * ease(u / 0.46))
    for i in range(1, n_show):
        if par[i] >= 0:
            d.line((q[i, 0], q[i, 1], q[par[i], 0], q[par[i], 1]), fill=rgba(WHITE, 0.30 * a * (1 - 0.6 * ease((u - 0.8) / 0.15))), width=1)
    stage = 0
    if u > 0.48:
        polyline(d, fr, rec["raw"], rgba(BLUE, a * (1 - 0.7 * ease((u - 0.64) / 0.1))), 3)
        stage = 1
    if u > 0.62:
        polyline(d, fr, rec["pruned"], rgba(WHITE, a * (1 - 0.6 * ease((u - 0.8) / 0.1))), 3)
        for x, y, z in project_n(fr, rec["pruned"]):
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=rgba(WHITE, a))
        stage = 2
    if u > 0.76:
        polyline(d, fr, rec["smooth"], rgba(ORANGE, a), 6)
        stage = 3
    for P, name in ((rec["smooth"][0], "START"), (rec["smooth"][-1], "GOAL")):
        x, y, z = project_n(fr, P[None])[0]
        d.ellipse((x - 9, y - 9, x + 9, y + 9), outline=rgba(WHITE, a), width=3)
        tracked(d, (x, y - 40), name, font("Bold", 16), rgba(WHITE, a), 3, "m")
    L = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())   # noqa: E731
    lines = [("RRT*", f"{n_show:,} samples, each checked for the platform, robots and cables"),
             ("RRT*", f"first path: {len(rec['raw'])} waypoints, {L(rec['raw']):.1f} m"),
             ("PRUNED", f"{len(rec['raw'])} waypoints down to {len(rec['pruned'])}"),
             ("B\u00c9ZIER-SMOOTHED", f"{L(rec['smooth']):.1f} m, bends rounded, straights kept")][stage]
    x0, y0 = 96, Y0 + 44
    panel(d, x0 - 20, y0 - 14, x0 + 600, y0 + 66, a)
    tracked(d, (x0, y0 - 2), lines[0], font("Bold", 16), rgba(ORANGE, a), 3)
    d.text((x0, y0 + 24), lines[1], font=font("Regular", 22), fill=rgba(WHITE, a))


def o_path(d, sh, fr, a):
    """While the team drives the plan: the planned path, the part already flown, and the ground robots' tracks."""
    rec = sh.rec
    k = fr["k"]
    for _, c, sz in rec["obstacles"]:
        _box_wire(d, fr, c, sz, rgba((255, 80, 60), 0.55 * a), 2)
    polyline(d, fr, rec["smooth"], rgba(WHITE, 0.6 * a), 2)
    ib = rec["paths"].index("/World/Platform/base")
    t0 = next(tt for tt, _, st in rec["ev"] if st == "TRACK")
    ka = int(np.searchsorted(rec["t"], t0))
    _trail(d, fr, rec["frames"][max(ka, k - 300):k + 1, ib, :3], ORANGE, a, 5)
    for u_ in range(4):
        iu = rec["paths"].index(f"/World/UGV_{u_}/chassis")
        _trail(d, fr, rec["frames"][max(ka, k - 420):k + 1:2, iu, :3] * [1, 1, 0] + [0, 0, 0.02], BLUE, 0.9 * a, 3)


# ------------------------------------------------------------------------------------------------ grade

_yy, _xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
VIGNETTE = (1.0 - 0.30 * np.clip((((_xx - PW / 2) / (PW / 2)) ** 2 + ((_yy - PH / 2) / (PH / 2)) ** 2) / 2.0, 0, 1) ** 1.5)[..., None]
_rng = np.random.default_rng(3)
GRAIN = [(_rng.normal(0, 1.6, (PH, PW, 1))).astype(np.float32) for _ in range(8)]
_lut_x = np.arange(256, dtype=np.float32) / 255.0
_s = _lut_x + 0.11 * np.sin(2 * np.pi * (_lut_x - 0.5)) * -0.5          # gentle S-curve: deeper shadows, cleaner highlights
LUT = np.stack([np.clip(_s * g + o, 0, 1) for g, o in ((1.035, 0.004), (1.0, 0.0), (0.955, -0.004))], -1) * 255.0   # warm


def grade(img, n):
    a = np.asarray(img, np.uint8)
    out = np.stack([LUT[a[..., c], c] for c in range(3)], -1)
    out = out * VIGNETTE + GRAIN[n % len(GRAIN)]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------------------------------------ overlays
# All overlays are drawn on an RGBA layer the size of the full 1920x1080 frame; picture pixels are offset by Y0.

def o_bars(d, chapter, number, a):
    """Letterbox furniture: the chapter label, top left."""
    if chapter:
        f = font("Bold", 19)
        w = tracked(d, (96, 58), f"{number:02d}", f, rgba(ORANGE, a), 4)
        d.line((96 + w + 14, 70, 96 + w + 54, 70), fill=rgba(GREY, a * 0.7), width=1)
        tracked(d, (96 + w + 68, 58), chapter, font("Semibold", 19), rgba(WHITE, a), 5)


def o_caption(layer, d, title, line, u, dur):
    a = ease(u * dur / 0.5) * ease((1 - u) * dur / 0.35)
    dx = (1 - ease(u * dur / 0.6)) * -24
    x, y = 96 + dx, Y0 + PH - 150
    g = Image.new("L", (W, 1))                               # soft dark wedge behind the text for legibility
    g.putdata([int(150 * a * max(0.0, 1 - i / 1150.0) ** 1.4) for i in range(W)])
    shade = Image.new("RGBA", (W, 190), (6, 8, 10, 0))
    shade.putalpha(g.resize((W, 190)))
    layer.alpha_composite(shade, (0, Y0 + PH - 190))
    d.rectangle((x, y + 6, x + 4, y + 96), fill=rgba(ORANGE, a))
    d.text((x + 24, y), title, font=font("Bold", 46), fill=rgba(WHITE, a))
    d.text((x + 25, y + 62), line, font=font("Regular", 25), fill=rgba((226, 228, 232), a))


def o_speed(d, speed, a):
    if speed < 1.5:
        return
    txt = ("TIME-LAPSE" if speed >= 6 else "SPEED") + f"  \u00d7{int(round(speed))}"
    f = font("Mono", 24)
    w = d.textlength(txt, font=f)
    x1, y = W - 96, Y0 + 40
    d.rounded_rectangle((x1 - w - 36, y, x1, y + 44), 22, fill=(8, 10, 12, int(150 * a)))
    d.text((x1 - w - 18, y + 9), txt, font=f, fill=rgba(ORANGE, a))


def o_labels(d, shot, fr, labels, u, dur):
    rec = shot.rec
    track = {"platform": "/World/Platform/base", **{f"drone_{k}": f"/World/Drone_{k}" for k in range(4)},
             **{f"ugv_{k}": f"/World/UGV_{k}/chassis" for k in range(4)}}
    for i, (anchor, text) in enumerate(labels):
        a = ease((u * dur - 0.5 - 0.35 * i) / 0.4) * ease((1 - u) * dur / 0.35)
        if a <= 0:
            continue
        x, y, z = project(fr, rec["frames"][fr["k"], rec["paths"].index(track[anchor]), :3])
        y += Y0
        if x < 1250 and y > Y0 + PH - 215:                   # under the caption: leave it out
            continue
        ex, ey = x + 70, y + (62 if y - Y0 < 130 else -62)   # leader goes down when the point is near the top edge
        d.ellipse((x - 7, y - 7, x + 7, y + 7), outline=rgba(WHITE, a), width=2)
        d.line((x + 5, y + (5 if ey > y else -5), ex, ey, ex + 26, ey), fill=rgba(WHITE, a), width=2)
        d.text((ex + 36, ey - 17), text, font=font("Semibold", 27), fill=rgba(WHITE, a))


def smooth_tele(rec, k, w=5):
    a, b = max(0, k - w), min(len(rec["t"]), k + w + 1)
    return rec["tele"][a:b].mean(0)


def o_cable_tags(d, shot, fr, a, right=470):
    """Live tension next to every cable that is in frame."""
    rec = shot.rec
    k = fr["k"]
    T = smooth_tele(rec, k)[:8]
    c = rec["cables"][k].reshape(2, 8, 3).astype(float)       # [0] platform ends, [1] robot ends
    placed = []
    for i in range(8):
        best = None
        for s in np.linspace(0.12, 0.9, 27):
            x, y, z = project(fr, c[0, i] + s * (c[1, i] - c[0, i]))
            if z > 0.2 and 120 < x < PW - right and 70 < y < PH - 210 and all(abs(x - px) > 120 or abs(y - py) > 40 for px, py in placed):
                best = (x, y)
                break
        if best is None:
            continue
        placed.append(best)
        x, y = best[0], best[1] + Y0
        col = ORANGE if i < 4 else BLUE
        txt = f"{T[i]:4.1f} N"
        f = font("Mono", 21)
        w = d.textlength(txt, font=f)
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=rgba(col, a))
        d.line((x, y, x + 22, y - 18), fill=rgba(col, a), width=2)
        d.rounded_rectangle((x + 22, y - 34, x + 22 + w + 20, y - 2), 6, fill=(8, 10, 12, int(175 * a)), outline=rgba(col, 0.9 * a), width=1)
        d.text((x + 32, y - 31), txt, font=f, fill=rgba(WHITE, a))


def o_hud(d, shot, fr, t_clip, a):
    """Physics panel: 8 cable tensions, wrist load cell trace over the clip, platform tracking error, drone tilt."""
    rec = shot.rec
    k = fr["k"]
    row = smooth_tele(rec, k)
    x0, y0, x1, y1 = W - 96 - 430, Y0 + PH - 330, W - 96, Y0 + PH - 40
    d.rounded_rectangle((x0, y0, x1, y1), 12, fill=(8, 10, 12, int(170 * a)))
    tracked(d, (x0 + 22, y0 + 16), "CABLE TENSION  [N]", font("Bold", 15), rgba(GREY, a), 3)
    bw, gap, gh, gy = 34, 14, 78, y0 + 132
    for i in range(8):
        x = x0 + 24 + i * (bw + gap)
        d.rectangle((x, gy - gh, x + bw, gy), fill=(46, 50, 58, int(200 * a)))
        h = gh * float(np.clip(row[i] / 40.0, 0, 1))
        d.rectangle((x, gy - h, x + bw, gy), fill=rgba(ORANGE if i < 4 else BLUE, a))
        d.text((x + bw / 2, gy + 5), f"{row[i]:.0f}", font=font("Mono", 16), fill=rgba(WHITE, a), anchor="ma")
    d.text((x0 + 24, y0 + 38), "quadrotors", font=font("Regular", 14), fill=rgba(ORANGE, a))
    d.text((x0 + 24 + 4 * (bw + gap), y0 + 38), "ground robots", font=font("Regular", 14), fill=rgba(BLUE, a))
    # load cell trace over the sim-time window of this clip, drawn up to "now"
    ta, tb = t_clip
    ka, kb = np.searchsorted(rec["t"], [ta, tb])
    kb = max(kb, ka + 2)
    tr = rec["tele"][ka:kb, 8]
    ly0, ly1, lx0, lx1 = y0 + 190, y1 - 44, x0 + 24, x1 - 24
    tracked(d, (x0 + 22, y0 + 166), "WRIST LOAD CELL  [N]", font("Bold", 15), rgba(GREY, a), 3)
    lo, hi = 0.0, max(40.0, float(tr.max()) * 1.1)
    d.line((lx0, ly1, lx1, ly1), fill=(90, 96, 104, int(200 * a)), width=1)
    n_now = int(np.clip(k - ka, 1, len(tr) - 1))
    step = max(1, len(tr) // 200)
    pts = [(lx0 + (lx1 - lx0) * j / (len(tr) - 1), ly1 - (ly1 - ly0) * (float(tr[j]) - lo) / (hi - lo)) for j in range(0, n_now + 1, step)]
    if len(pts) > 1:
        d.line(pts, fill=rgba(WHITE, a), width=2)
        d.ellipse((pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4), fill=rgba(ORANGE, a))
    d.text((lx1, y0 + 160), f"{row[8]:5.1f}", font=font("Mono", 24), fill=rgba(WHITE, a), anchor="ra")
    f = font("Regular", 16)
    d.text((x0 + 24, y1 - 32), f"platform error  {row[17] * 1000:4.1f} mm", font=f, fill=rgba(WHITE, a))
    d.text((x1 - 24, y1 - 32), f"drone tilt  {row[9:13].max():4.1f}°", font=f, fill=rgba(WHITE, a), anchor="ra")


def o_fact(d, text, sub, a):
    """One physical fact, top left of the picture (macro shots)."""
    x, y = 96, Y0 + 44
    wtxt = max(d.textlength(text, font=font("Bold", 30)), d.textlength(sub, font=font("Regular", 21)))
    d.rounded_rectangle((x - 20, y - 14, x + wtxt + 24, y + 78), 10, fill=(8, 10, 12, int(160 * a)))
    d.text((x, y - 4), text, font=font("Bold", 30), fill=rgba(WHITE, a))
    d.text((x, y + 40), sub, font=font("Regular", 21), fill=rgba(GREY, a))


def darken(layer, a, left=False):
    if left:
        g = Image.new("L", (W, 1))
        g.putdata([int(255 * a * max(0.0, 1 - i / 1500.0) ** 1.2) for i in range(W)])
        sh = Image.new("RGBA", (W, PH), (6, 8, 10, 0))
        sh.putalpha(g.resize((W, PH)))
    else:
        sh = Image.new("RGBA", (W, PH), (6, 8, 10, int(255 * a)))
    layer.alpha_composite(sh, (0, Y0))


def o_title(layer, d, t, dur):
    a_in = ease((t - 0.6) / 1.0)
    out = ease((dur - t) / 0.7)
    darken(layer, 0.5 * ease(t / 1.2) * out)
    cx, cy = W / 2, H / 2
    words = plan.TITLE.upper().split(" ")
    l1, l2 = " ".join(words[:2]), " ".join(words[2:])
    tracked(d, (cx, cy - 150), "A DOCTORAL THESIS", font("Semibold", 21), rgba(ORANGE, a_in * out), 9, "m")
    d.line((cx - 40, cy - 108, cx + 40, cy - 108), fill=rgba(ORANGE, a_in * out), width=2)
    for i, (txt, y, wgt, sz) in enumerate(((l1, cy - 88, "Light", 62), (l2, cy - 16, "Black", 62))):
        a = ease((t - 0.9 - 0.35 * i) / 0.9) * out
        tracked(d, (cx, y + (1 - a) * 14), txt, font(wgt, sz), rgba(WHITE, a), 5, "m")
    a = ease((t - 2.6) / 0.9) * out
    d.text((cx, cy + 92), f"A thesis by {plan.AUTHOR}", font=font("Medium", 30), fill=rgba(WHITE, a), anchor="ma")
    a = ease((t - 3.1) / 0.9) * out
    d.text((cx, cy + 138), f"Under the supervision of {plan.SUPERVISOR}", font=font("Regular", 26), fill=rgba(WHITE, a), anchor="ma")
    d.text((cx, cy + 176), plan.UNIVERSITY, font=font("Regular", 26), fill=rgba(WHITE, a), anchor="ma")


def o_advantage(layer, d, num, title, line, u, dur):
    a = ease(u * dur / 0.45) * ease((1 - u) * dur / 0.35)
    darken(layer, 0.28 * a)
    darken(layer, 0.80 * a, left=True)
    x, y = 150, H / 2 - 92
    dx = (1 - ease(u * dur / 0.6)) * -30
    d.text((x + dx, y - 6), f"{num:02d}", font=font("Light", 120), fill=rgba(ORANGE, a))
    d.rectangle((x + 190 + dx, y + 26, x + 193 + dx, y + 126), fill=rgba(GREY, 0.6 * a))
    d.text((x + 224 + dx, y + 12), title, font=font("Black", 62), fill=rgba(WHITE, a))
    d.text((x + 226 + dx, y + 92), line, font=font("Regular", 29), fill=rgba((230, 232, 236), a))


def o_credits(layer, d, t, dur):
    a0 = ease((t - 0.8) / 1.2)
    darken(layer, 0.72 * ease(t / 1.5))
    cx, cy = W / 2, H / 2
    words = plan.TITLE.upper().split(" ")
    tracked(d, (cx, cy - 150), " ".join(words[:2]), font("Light", 44), rgba(WHITE, a0), 5, "m")
    tracked(d, (cx, cy - 96), " ".join(words[2:]), font("Black", 44), rgba(WHITE, a0), 5, "m")
    d.line((cx - 40, cy - 22, cx + 40, cy - 22), fill=rgba(ORANGE, a0), width=2)
    a = ease((t - 1.8) / 1.0)
    tracked(d, (cx, cy + 4), "A THESIS BY", font("Semibold", 17), rgba(GREY, a), 6, "m")
    d.text((cx, cy + 30), plan.AUTHOR, font=font("Medium", 34), fill=rgba(WHITE, a), anchor="ma")
    a = ease((t - 2.5) / 1.0)
    tracked(d, (cx, cy + 96), "UNDER THE SUPERVISION OF", font("Semibold", 17), rgba(GREY, a), 6, "m")
    d.text((cx, cy + 122), plan.SUPERVISOR, font=font("Medium", 34), fill=rgba(WHITE, a), anchor="ma")
    a = ease((t - 3.2) / 1.0)
    d.text((cx, cy + 192), plan.UNIVERSITY, font=font("Light", 28), fill=rgba(WHITE, a), anchor="ma")
    if getattr(plan, "CLOSING", ""):
        a = ease((t - 3.8) / 1.0)
        d.text((cx, cy + 236), plan.CLOSING, font=font("Medium", 22), fill=rgba(ORANGE, a), anchor="ma")


# ------------------------------------------------------------------------------------------------ timeline

def build(src):
    have = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "renders", src, "*", "meta.json"))}
    clips, chapters, t0 = [], [], 0.0
    for c in plan.edit(have):
        sh = Shot(src, c["shot"])
        a, b = c["a"], c["b"] if c["b"] is not None else sh.dur
        ch = c.get("chapter")
        if ch and ch not in chapters:
            chapters.append(ch)
        clips.append(dict(c, sh=sh, a=a, b=b, t0=t0, n=int(round((b - a) * FPS)), num=chapters.index(ch) + 1 if ch else 0))
        t0 += (b - a)
    return clips, t0


def frame(clips, ci, j, n_global):
    c = clips[ci]
    sh = c["sh"]
    dur = c["b"] - c["a"]
    t = j / FPS                                             # time inside the clip
    u = j / max(c["n"] - 1, 1)
    fr, path = sh.at(c["a"] + t)
    pic = grade(Image.open(path).convert("RGB"), n_global)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    canvas.paste(pic, (0, Y0))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    card = c.get("card")
    if card and card[0] == "title":
        o_title(layer, d, t, dur)
    elif card and card[0] == "advantage":
        o_advantage(layer, d, card[1], card[2], card[3], u, dur)
    elif card and card[0] == "credits":
        o_credits(layer, d, t, dur)
    if c.get("hud"):
        ts = sh.meta["shot"]["t"]
        span = (ts[0] + (ts[1] - ts[0]) * c["a"] / sh.dur, ts[0] + (ts[1] - ts[0]) * c["b"] / sh.dur)
        a = ease(t / 0.4) * ease((dur - t) / 0.3)
        o_cable_tags(d, sh, fr, a)
        o_hud(d, sh, fr, span, a)
    fade = ease(t / 0.4) * ease((dur - t) / 0.3)
    if c.get("viz") == "workspace":
        o_workspace(d, fr, u, fade, sh)
    elif c.get("viz") == "track":
        o_track(d, sh, fr, fade)
    elif c.get("viz") == "plan":
        o_plan(d, sh, fr, u, fade)
    elif c.get("viz") == "path":
        o_path(d, sh, fr, fade)
    if c.get("tags") and not c.get("hud"):
        o_cable_tags(d, sh, fr, fade)
    if c.get("fact") and c.get("viz") != "plan":
        o_fact(d, *c["fact"], ease((t - 0.3) / 0.4) * ease((dur - t) / 0.3))
    if c.get("labels"):
        o_labels(d, sh, fr, c["labels"], u, dur)
    if c.get("caption"):
        o_caption(layer, d, *c["caption"], u, dur)
    if c.get("speed"):
        o_speed(d, sh.meta["speed"], ease(t / 0.3) * ease((dur - t) / 0.3))
    layer.paste((0, 0, 0, 255), (0, 0, W, Y0))               # overlays never spill into the bars...
    layer.paste((0, 0, 0, 255), (0, Y0 + PH, W, H))
    d = ImageDraw.Draw(layer)
    if not card or card[0] == "advantage":
        o_bars(d, c.get("chapter"), c["num"], 1.0)           # ...except the bar furniture itself
    out = Image.alpha_composite(canvas, layer).convert("RGB")
    # fades: in at the very start, through black between chapters, out at the end
    prev_ch = clips[ci - 1].get("chapter") if ci > 0 else None
    next_ch = clips[ci + 1].get("chapter") if ci + 1 < len(clips) else None
    k = 1.0
    if ci == 0:
        k = min(k, ease(t / 1.2))
    elif prev_ch != c.get("chapter"):
        k = min(k, ease(t / 0.35))
    if ci == len(clips) - 1:
        k = min(k, ease((dur - t) / 1.5))
    elif next_ch != c.get("chapter"):
        k = min(k, ease((dur - t) / 0.35))
    if k < 1.0:
        out = Image.blend(Image.new("RGB", out.size, (0, 0, 0)), out, k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="trailer")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="trailer")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--allow-stale", action="store_true")
    args = ap.parse_args()
    clips, total = build(args.src)
    print(f"[trailer] {len(clips)} clips, {total:.1f} s")
    if STALE and not args.allow_stale:
        sys.exit(f"[trailer] these shots were rendered from an older recording than the one on disk, so labels and readouts would "
                 f"not line up with the picture: {sorted(set(STALE))}. Re-render them (tools/render_shots.py --rec <rec> --only ...) "
                 f"or pass --allow-stale.")
    marks = []
    for c in clips:
        prev = marks[-1][1] if marks else None
        if c.get("chapter") != prev or not marks:
            marks.append((c["t0"], c.get("chapter")))
    json.dump(dict(total=total, chapters=marks, cuts=[c["t0"] for c in clips]), open(os.path.join(ROOT, "renders", f"{args.out}_timeline.json"), "w"))
    if args.sample:
        od = os.path.join(ROOT, "renders", f"{args.out}_samples")
        os.makedirs(od, exist_ok=True)
        for q, ci in enumerate(np.linspace(0, len(clips) - 1, min(args.sample, len(clips))).round().astype(int)):
            c = clips[ci]
            j = int(c["n"] * 0.62)
            frame(clips, ci, j, q).save(os.path.join(od, f"{q:02d}_{c['shot']}.jpg"), quality=90)
        print("[trailer] samples in", od)
        return
    out = os.path.join(ROOT, "renders", f"{args.out}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if args.audio:
        cmd += ["-i", args.audio, "-c:a", "aac", "-b:a", "256k", "-shortest"]
    # keyframe every second and a capped bitrate: seeks and plays smoothly in ordinary players and survives re-encoding on upload
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-g", str(FPS), "-maxrate", "16M", "-bufsize", "32M", "-profile:v", "high",
            "-level", "4.2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for ci, c in enumerate(clips):
        for j in range(c["n"]):
            ff.stdin.write(frame(clips, ci, j, n).tobytes())
            n += 1
        print(f"[trailer] {c['shot']} ({c['t0']:.1f} s)", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"[trailer] {out}  {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
