"""Edit rendered frames into a finished clip: title card, HUD (task, block counter, speed tag), live telemetry
panel from the recording, cross-fades at shot changes, fade-out.

    python3 tools/compose_video.py --name tower --title "Twisted tower" --subtitle "6 hollow blocks ..." --index 3
Reads renders/<name>/frames.json + PNGs and recordings/<name>.npz, writes renders/<name>.mp4 (H.264, CRF 16).
"""

import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FONT = "/usr/share/fonts/truetype/ubuntu/"
ORANGE, BLUE, WHITE = (255, 106, 19), (120, 180, 235), (245, 246, 248)


def font(w, size):
    return ImageFont.truetype(os.path.join(FONT, f"Ubuntu{w}.ttf"), size)


def mono(size):
    return ImageFont.truetype(os.path.join(FONT, "UbuntuMono-B.ttf"), size)


def spaced(d, xy, text, f, fill, tracking=2):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + tracking
    return x


def panel(img, box, radius=14, alpha=150):
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle(box, radius, fill=(12, 14, 18, alpha))
    return Image.alpha_composite(img, ov)


def hud(img, task_label, block, n_blocks, speed, tele):
    W, H = img.size
    s = W / 1920
    img = img.convert("RGBA")
    # top-left: task label + counter
    img = panel(img, (int(36 * s), int(34 * s), int(560 * s), int(132 * s)), alpha=120)
    d = ImageDraw.Draw(img)
    d.rectangle((int(36 * s), int(34 * s), int(42 * s), int(132 * s)), fill=ORANGE)
    spaced(d, (int(62 * s), int(46 * s)), task_label.upper(), font("-M", int(28 * s)), WHITE, int(3 * s))
    d.text((int(62 * s), int(88 * s)), f"Block {min(block, n_blocks)} / {n_blocks} placed", font=font("-L", int(28 * s)), fill=(210, 214, 220))
    # top-right: speed tag
    tag = "REAL TIME" if speed <= 1.01 else ("x%d TIME-LAPSE" % round(speed) if speed > 1.5 else "x2")
    if speed == 0:
        tag = "COMPLETE"
    f = mono(int(28 * s))
    tw = d.textlength(tag, font=f)
    x1 = W - int(40 * s)
    img = panel(img, (int(x1 - tw - 36 * s), int(40 * s), x1, int(88 * s)), radius=int(24 * s), alpha=140)
    d = ImageDraw.Draw(img)
    d.text((x1 - tw - 18 * s, 48 * s), tag, font=f, fill=ORANGE if speed > 1.5 else WHITE)
    # bottom-left: telemetry
    T, load, tilt, perr = tele["T"], tele["load"], tele["tilt"], tele["perr"]
    bx0, by0, bx1, by1 = int(36 * s), H - int(286 * s), int(520 * s), H - int(36 * s)
    img = panel(img, (bx0, by0, bx1, by1), alpha=150)
    d = ImageDraw.Draw(img)
    spaced(d, (bx0 + 20 * s, by0 + 14 * s), "CABLE TENSION  [N]", font("-M", int(19 * s)), (200, 205, 212), int(2 * s))
    gx0, gy1, gh = bx0 + 24 * s, by0 + 170 * s, 118 * s
    bw = 40 * s
    for i, v in enumerate(T):
        x = gx0 + i * (bw + 14 * s)
        d.rectangle((x, gy1 - gh, x + bw, gy1), fill=(40, 44, 52))
        h = gh * np.clip(v / 60.0, 0, 1)
        d.rectangle((x, gy1 - h, x + bw, gy1), fill=ORANGE if i < 4 else BLUE)
        d.text((x + bw / 2, gy1 + 6 * s), f"{v:4.0f}", font=mono(int(17 * s)), fill=WHITE, anchor="ma")
    d.text((gx0, by0 + 40 * s), "drones", font=font("-L", int(16 * s)), fill=ORANGE)
    d.text((gx0 + 4 * (bw + 14 * s), by0 + 40 * s), "ground robots", font=font("-L", int(16 * s)), fill=BLUE)
    ly = by1 - 40 * s
    f = font("-R", int(19 * s))
    d.text((bx0 + 20 * s, ly), f"load cell {load:5.1f} N", font=f, fill=WHITE)
    d.text((bx0 + 190 * s, ly), f"drone tilt {tilt:4.1f}°", font=f, fill=WHITE)
    d.text((bx0 + 350 * s, ly), f"pose err {perr * 1000:4.1f} mm", font=f, fill=WHITE)
    return img.convert("RGB")


def title_card(img, title, subtitle, index, a):
    """a in [0,1]: card opacity."""
    W, H = img.size
    s = W / 1920
    base = img.filter(ImageFilter.GaussianBlur(10 * s * a)).convert("RGBA")
    ov = Image.new("RGBA", img.size, (8, 10, 14, int(170 * a)))
    base = Image.alpha_composite(base, ov)
    d = ImageDraw.Draw(base)
    al = int(255 * a)
    spaced(d, (int(140 * s), int(H / 2 - 150 * s)), f"TASK {index}", font("-M", int(30 * s)), ORANGE + (al,), int(6 * s))
    d.rectangle((int(140 * s), int(H / 2 - 104 * s), int(220 * s), int(H / 2 - 98 * s)), fill=ORANGE + (al,))
    d.text((int(136 * s), int(H / 2 - 82 * s)), title, font=font("-B", int(96 * s)), fill=WHITE + (al,))
    d.text((int(140 * s), int(H / 2 + 40 * s)), subtitle, font=font("-L", int(36 * s)), fill=(215, 220, 228, al))
    return base.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--fade", type=int, default=12)
    args = ap.parse_args()

    rdir = os.path.join(ROOT, "renders", args.name)
    info = json.load(open(os.path.join(rdir, "frames.json")))
    rec = np.load(os.path.join(ROOT, "recordings", f"{args.name}.npz"))
    tele, tt = rec["tele"], rec["t"]
    fps = info["fps"]
    fr = info["frames"]
    first = Image.open(os.path.join(rdir, f"frame_{fr[0]['frame']:05d}.png"))
    W, H = first.size
    out = os.path.join(ROOT, "renders", f"{args.name}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n_title = int(3.5 * fps)                                   # title card over the opening frames
    total = len(fr)
    prev = None
    shot_start = 0
    for j, f in enumerate(fr):
        img = Image.open(os.path.join(rdir, f"frame_{f['frame']:05d}.png")).convert("RGB")
        k = int(np.clip(np.searchsorted(tt, f["t"]), 0, len(tt) - 1))
        row = tele[k]
        tl = dict(T=row[:8], load=float(row[8]), tilt=float(row[9:13].max()), perr=float(row[17]))
        placed = int(row[18])
        if j > 0 and f["shot"] != fr[j - 1]["shot"]:
            shot_start = j
        img = hud(img, args.title, placed, info["n_blocks"], f["speed"], tl)
        if prev is not None and j - shot_start < args.fade:      # cross-fade into a new shot
            a = (j - shot_start + 1) / (args.fade + 1)
            img = Image.blend(prev_last, img, a)
        else:
            prev_last = img
        if j < n_title:                                          # title over the opening establishing shot
            a = 1.0 if j < n_title - 18 else (n_title - j) / 18
            img = title_card(img, args.title, args.subtitle, args.index, a)
        if total - j < 20:                                       # fade to black at the end
            img = Image.blend(Image.new("RGB", img.size, (0, 0, 0)), img, (total - j) / 20)
        prev = img
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"[compose] {out} ({total / fps:.1f} s)")


if __name__ == "__main__":
    main()
