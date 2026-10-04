"""Export the trailer as web media for the portfolio page (/work/sastra): looping chapter clips without the
letterbox bars, their posters, and clean graded stills from the rendered shots.

    python3 tools/export_web.py ~/siddharth_website/nextjs/public/assets
Writes <assets>/videos/qcdpm/*.mp4 (H.264, muted, 1280x536) and <assets>/img/qcdpm/*.jpg (1920x804).
"""

import json
import os
import subprocess
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_trailer as ct  # noqa: E402

ROOT = ct.ROOT
dst = os.path.expanduser(sys.argv[1])
vid, img = os.path.join(dst, "videos", "qcdpm"), os.path.join(dst, "img", "qcdpm")
os.makedirs(vid, exist_ok=True)
os.makedirs(img, exist_ok=True)
src = os.path.join(ROOT, "renders", "trailer_v5.mp4")
marks = {ch: t for t, ch in json.load(open(os.path.join(ROOT, "renders", "trailer_v5_timeline.json")))["chapters"] if ch}
order = ["THE SYSTEM", "MODEL AND LEARNING", "CONTROL", "PLANNING", "APPLICATION: CONSTRUCTION"]
crop = f"crop={ct.PW}:{ct.PH}:0:{ct.Y0},scale=1280:536"


def clip(name, a, b, crf=25):
    out = os.path.join(vid, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", src, "-an", "-vf", crop, "-c:v", "libx264",
                    "-preset", "slow", "-crf", str(crf), "-g", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print(f"{out}  {os.path.getsize(out) / 1e6:.1f} MB")


def poster(name, t):
    out = os.path.join(img, f"{name}.jpg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1", "-vf", f"crop={ct.PW}:{ct.PH}:0:{ct.Y0}",
                    "-q:v", "3", out], check=True)


names = ["system", "model", "control", "planning"]         # the thesis parts; the applications get clean clips below
for i, n in enumerate(names):
    a, b = marks[order[i]], marks[order[i + 1]]
    clip(f"qcdpm-{n}", a + 0.2, b - 0.2)                 # trim the fades through black at the chapter edges
    poster(f"poster-{n}", a + 0.62 * (b - a))
clip("qcdpm-trailer", 0.0, 90.2, crf=27)
poster("poster-trailer", 3.6)

# clean stills (no overlays): shot id -> (fraction through the shot, file name)
stills = {"system": (0.5, "system-wide"), "drone": (0.5, "drone"), "ugv": (0.6, "ground-robot"), "platform": (0.5, "end-effector"),
          "lookup": (0.6, "cables-from-below"), "track_close": (0.5, "tracking-close"), "plan_go1": (0.55, "around-the-cabin"),
          "plan_go2": (0.6, "over-the-pallet"), "pick": (0.8, "descent"), "grasp_macro": (0.75, "grasp"), "seat_macro": (0.55, "seating"),
          "tower_hero": (0.5, "tower"), "pyr_hero": (0.5, "stepped-wall"), "wall_top": (0.97, "enclosure-top"), "wall_hero": (0.9, "enclosure"),
          "end": (0.15, "site")}
for shot, (u, name) in stills.items():
    d = os.path.join(ROOT, "renders", "trailer", shot)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    f = fs[int(u * (len(fs) - 1))]
    ct.grade(Image.open(os.path.join(d, f)).convert("RGB"), 0).save(os.path.join(img, f"{name}.jpg"), quality=88)
print(len(stills), "stills ->", img)


# ---------------------------------------------------------------- application clips: clean footage (no overlays), graded
def seq_clip(name, shots, crf=25):
    """Join rendered shots (renders/trailer/<shot>/f_*.png) into one looping clip, 1280x536."""
    out = os.path.join(vid, f"{name}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ct.PW}x{ct.PH}", "-r", str(ct.FPS),
                           "-i", "-", "-an", "-vf", "scale=1280:536", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-g", "30",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = 0
    for sh in shots:
        sh, first = sh if isinstance(sh, tuple) else (sh, 0)          # (shot, first frame) skips the start of a shot
        d = os.path.join(ROOT, "renders", "trailer", sh)
        for f in sorted(x for x in os.listdir(d) if x.endswith(".png"))[first:]:
            ff.stdin.write(ct.grade(Image.open(os.path.join(d, f)).convert("RGB"), n).tobytes())
            n += 1
    ff.stdin.close()
    ff.wait()
    print(f"{out}  {n / ct.FPS:.1f} s  {os.path.getsize(out) / 1e6:.1f} MB")


def still(shot, u, name):
    d = os.path.join(ROOT, "renders", "trailer", shot)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    ct.grade(Image.open(os.path.join(d, fs[int(u * (len(fs) - 1))])).convert("RGB"), 0).save(os.path.join(img, f"{name}.jpg"), quality=88)


have = set(os.listdir(os.path.join(ROOT, "renders", "trailer")))
apps = {"app-construction": ["pick", "tower_lapse", "tower_hero", "pyr_lapse", "pyr_hero", "wall_lapse", "wall_top"],
        "app-painting": ["paint_wide", ("paint_close", 60), "paint_top", "paint_hero"],   # a cable crosses the lens before frame 60
        "app-printing": ["print_lapse", "print_close", "print_hero"]}
for name, shots in apps.items():
    if all((sh[0] if isinstance(sh, tuple) else sh) in have for sh in shots):
        seq_clip(f"qcdpm-{name}", shots)
for shot, u, name in (("paint_hero", 0.6, "painting-finished"), ("paint_close", 0.8, "painting-spray"), ("paint_top", 0.98, "painting-from-above"),
                      ("paint_wide", 0.5, "painting-hangar"), ("print_hero", 0.5, "printing-finished"), ("print_close", 0.5, "printing-nozzle"),
                      ("print_lapse", 0.55, "printing-site"), ("tower_lapse", 0.7, "construction-tower"), ("wall_lapse", 0.9, "construction-enclosure")):
    if shot in have:
        still(shot, u, name)
        if name in ("painting-hangar", "printing-site", "construction-tower"):
            still(shot, u, "poster-app-" + name.split("-")[0])
