"""Poster / video thumbnail from the hero still (renders/poster_stills/poster/f_0000.png, rendered with
`render_shots.py --rec tower --stills --out poster --res 2560 1440 --spp 48 --only poster`).

    python3 tools/make_poster.py        -> renders/trailer_poster.jpg (1920x1080) and renders/trailer_poster_1280.jpg
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trailer_plan as plan  # noqa: E402

ROOT = plan.ROOT
LATO = "/usr/share/fonts/truetype/lato/Lato-%s.ttf"
W, H = 1920, 1080
ORANGE, WHITE = (255, 122, 26), (248, 248, 246)


def font(w, size):
    return ImageFont.truetype(LATO % w, int(size))


def tracked(d, xy, text, f, fill, tracking):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + tracking
    return x


im = Image.open(os.path.join(ROOT, "renders", "poster_stills", "poster", "f_0000.png")).convert("RGB").resize((W, H), Image.LANCZOS)
a = np.asarray(im, np.float32) / 255.0
# grade: gentle S-curve, warm highlights, cooler and deeper sky so the type reads
a = np.clip(a + 0.10 * np.sin(2 * np.pi * (a - 0.5)) * -0.5, 0, 1)
a *= np.array([1.04, 1.0, 0.95])
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
sky = np.clip(1 - yy / (0.62 * H), 0, 1)[..., None] ** 1.3                    # darken towards the top
a = a * (1 - 0.50 * sky) + np.array([0.02, 0.05, 0.11]) * 0.50 * sky
left = (np.clip((0.62 - xx / W) / 0.62, 0, 1) ** 1.1 * np.clip(1 - np.abs(yy / H - 0.47) / 0.5, 0, 1) ** 0.7)[..., None]
a = a * (1 - 0.48 * left) + np.array([0.02, 0.04, 0.08]) * 0.48 * left         # extra weight behind the title block (left)
vig = 1 - 0.28 * np.clip(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2, 0, 2)[..., None] ** 1.4 / 2
im = Image.fromarray((np.clip(a * vig, 0, 1) * 255).astype(np.uint8))

layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(layer)
x0, y0 = 92, 236                                         # the title sits in the empty sky on the left; the system on the right
tracked(d, (x0 + 3, y0), "A DOCTORAL THESIS", font("Bold", 24), ORANGE + (255,), 9)
d.rectangle((x0 + 3, y0 + 48, x0 + 100, y0 + 53), fill=ORANGE + (255,))
words = plan.TITLE.upper().split(" ")
tracked(d, (x0, y0 + 76), " ".join(words[:2]), font("Light", 45), WHITE + (255,), 3)          # QUADROTOR-BASED MOBILE
tracked(d, (x0 - 2, y0 + 134), words[2], font("Black", 92), WHITE + (255,), 2)                # CABLE-DRIVEN
tracked(d, (x0 - 2, y0 + 232), words[3], font("Black", 92), WHITE + (255,), 2)                # PARALLEL
tracked(d, (x0 - 2, y0 + 330), words[4], font("Black", 92), WHITE + (255,), 2)                # MANIPULATOR
d.text((x0 + 3, y0 + 452), plan.AUTHOR, font=font("Semibold", 34), fill=WHITE + (255,))
d.text((x0 + 3, y0 + 498), plan.UNIVERSITY, font=font("Regular", 25), fill=(230, 232, 236, 255))
shadow = layer.filter(ImageFilter.GaussianBlur(10))
sh = np.asarray(shadow).copy()
sh[..., :3] = 0
sh[..., 3] = (sh[..., 3] * 0.55).astype(np.uint8)
out = Image.alpha_composite(Image.alpha_composite(im.convert("RGBA"), Image.fromarray(sh)), layer).convert("RGB")
p = os.path.join(ROOT, "renders", "trailer_poster.jpg")
out.save(p, quality=93)
out.resize((1280, 720), Image.LANCZOS).save(p.replace(".jpg", "_1280.jpg"), quality=92)
print(p)
