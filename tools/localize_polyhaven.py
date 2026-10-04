"""Point Poly Haven model texture paths (absolute paths on their build server) at the local textures/ folder.
Run once after tools/fetch_assets.py, with Isaac's python (needs pxr):
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tools/localize_polyhaven.py
"""

import glob
import os

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from pxr import Sdf, Usd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "polyhaven", "models")
for f in sorted(glob.glob(os.path.join(ROOT, "*", "*.usdc"))):
    tex_dir = os.path.join(os.path.dirname(f), "textures")
    local = set(os.listdir(tex_dir)) if os.path.isdir(tex_dir) else set()
    layer = Sdf.Layer.FindOrOpen(f)
    stage = Usd.Stage.Open(layer)
    fixed = missing = 0
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            v = attr.Get()
            if not v or not v.path:
                continue
            name = os.path.basename(v.path)
            if v.path.startswith("./textures/"):
                continue
            if name in local:
                attr.Set(Sdf.AssetPath("./textures/" + name))
                fixed += 1
            else:
                missing += 1
    layer.Save()
    print(f"LOCALIZE {os.path.basename(f):38s} fixed {fixed:3d}  unresolved {missing}")
app.close()
