"""Download the CC0 Poly Haven assets used by the site scene into assets/polyhaven/ (idempotent).

    python3 tools/fetch_assets.py
Textures: diffuse (sRGB), OpenGL normal and roughness maps. Models: USD (usdc + its texture includes).
All Poly Haven assets are CC0 (public domain): https://polyhaven.com/license
"""

import json
import os
import urllib.request

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "polyhaven")
TEXTURES = {"concrete_floor_02": "2k", "dry_ground_01": "2k", "corrugated_iron": "1k", "concrete_slab_wall": "1k",
            "dense_sand": "2k", "rough_concrete": "1k"}
MODELS = ["concrete_road_barrier", "concrete_road_barrier_02", "modular_chainlink_fence", "cement_bag", "Barrel_01",
          "metal_toolbox", "ladder_sectioned_01", "portable_generator", "wooden_crate_01", "metal_jerrycan",
          "street_lamp_01", "plastic_crate_01"]


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "isaac-cdpr-asset-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
        f.write(r.read())
    os.replace(path + ".part", path)


def files(asset):
    req = urllib.request.Request(f"https://api.polyhaven.com/files/{asset}", headers={"User-Agent": "isaac-cdpr-asset-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


if __name__ == "__main__":
    for name, res in TEXTURES.items():
        f = files(name)
        for key, out in (("Diffuse", "diff"), ("nor_gl", "nor_gl"), ("Rough", "rough")):
            fmt = "jpg" if "jpg" in f[key][res] else "png"
            fetch(f[key][res][fmt]["url"], os.path.join(OUT, name, f"{name}_{out}_{res}.{fmt}"))
    for name in MODELS:
        u = files(name)["usd"]["1k"]["usd"]
        base = os.path.join(OUT, "models", name)
        fetch(u["url"], os.path.join(base, os.path.basename(u["url"])))
        for rel, inc in u.get("include", {}).items():
            fetch(inc["url"], os.path.join(base, rel))
    with open(os.path.join(OUT, "LICENSE.txt"), "w") as fh:
        fh.write("All files in this folder are from Poly Haven (https://polyhaven.com) and are CC0 (public domain).\n")
    print("ok")
