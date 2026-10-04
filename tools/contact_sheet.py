"""Contact sheet of the trailer framing stills: python3 tools/contact_sheet.py [dir] [which: 0 first, 1 middle, 2 last]"""
import glob
import os
import sys

from PIL import Image, ImageDraw

d = sys.argv[1] if len(sys.argv) > 1 else "renders/trailer_stills"
which = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ids = sys.argv[3:] or sorted(os.listdir(d), key=lambda s: os.path.getmtime(os.path.join(d, s)))
ims = []
for s in ids:
    fs = sorted(glob.glob(os.path.join(d, s, "f_*.png")))
    if fs:
        im = Image.open(fs[min(which, len(fs) - 1)]).resize((960, 402))
        ImageDraw.Draw(im).text((10, 8), s, fill=(255, 255, 0))
        ims.append(im)
rows = (len(ims) + 1) // 2
W = Image.new("RGB", (1920, 402 * rows))
for i, im in enumerate(ims):
    W.paste(im, ((i % 2) * 960, (i // 2) * 402))
out = os.path.join(d, f"sheet_{which}.jpg")
W.save(out, quality=88)
print(out)
