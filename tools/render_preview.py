"""Render still previews of the scene (headless RTX) to results/preview_<view>.png.

    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tools/render_preview.py [--frames 120]
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--res", type=int, nargs=2, default=[1600, 900])
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from PIL import Image  # noqa: E402

from cdpr_sim.blocks import build_blocks, pattern  # noqa: E402
from cdpr_sim.mathutil import look_at_quat  # noqa: E402
from cdpr_sim.scene import build_scene  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
info = build_scene(stage, cfg)
build_blocks(stage, cfg, len(pattern(cfg)), info["textures"]["block"])
app.update()

VIEWS = {  # name: (eye, target)
    "overview": ((-11.0, -15.0, 7.5), (2.0, 2.0, 1.5)),
    "robots": ((-4.5, -6.5, 3.2), (1.6, 0.0, 1.5)),
    "building": ((4.0, -3.0, 2.2), (-1.0, 14.0, 4.0)),
    "yard": ((5.0, -2.0, 2.5), (11.5, 3.0, 0.8)),
}
W, H = args.res
sensors = {}
for name, (eye, tgt) in VIEWS.items():
    cam = RtxCamera(f"/World/Preview/{name}", translations=np.array([eye]),
                    orientations=np.array([look_at_quat(np.array(eye), np.array(tgt))]))
    cam.prims[0].GetAttribute("focalLength").Set(18.0)
    sensors[name] = CameraSensor(cam, resolution=(H, W), annotators=["rgb"])
for _ in range(args.frames):  # let textures, referenced assets and the sky stream in
    app.update()
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
for name, sensor in sensors.items():
    data, _ = sensor.get_data("rgb")
    if data is None:
        print("no image for", name)
        continue
    img = data.numpy()[..., :3]
    p = os.path.join(ROOT, "results", f"preview_{name}.png")
    Image.fromarray(img).save(p)
    print("PREVIEW", p, img.shape, "mean", img.mean().round(1))
app.close()
