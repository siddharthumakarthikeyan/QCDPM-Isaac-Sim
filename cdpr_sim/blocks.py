"""Hollow masonry blocks, the staging stock and build patterns.

Block: outer L x W x H with two through-cores (open top and bottom), like a hollow concrete block. The collision shape
is the real hollow geometry built from convex boxes (2 face shells + 3 webs), so stacking contacts happen on the real
bearing areas.

Patterns return block poses (x, y, z_centre, yaw) in build order. Every block rests on the ground or on blocks
already placed.
  compound : square enclosure wall, courses alternating which pair of sides runs through the corners
  dome     : corbelled dome; rings of tangentially oriented blocks, each course stepped inward by `corbel`
  column   : single stack (quick tests)
  tower    : twisted tower, each course rotated by `twist` (shows the wrist yaw)
  pyramid  : stepped pyramid wall, n, n-1, ..., 1 blocks per course, running bond
Neighbouring blocks in a course get a `joint` gap (dry-stack placement tolerance).
"""

import numpy as np

from .scene import _bind_phys, _material, _phys_material, _rigid, _uv_box, _xform

BLOCK_ROOT = "/World/Blocks"


def block_parts(cfg):
    """(name, size, centre) of the convex pieces of one block in its own frame."""
    L, W, H = cfg["blocks"]["size"]
    t = cfg["blocks"]["shell"]
    inner_w = W - 2 * t
    parts = [("shell_p", (L, t, H), (0, (W - t) / 2, 0)), ("shell_n", (L, t, H), (0, -(W - t) / 2, 0))]
    for i, x in enumerate((-(L - t) / 2, 0.0, (L - t) / 2)):
        parts.append((f"web_{i}", (t, inner_w, H), (x, 0, 0)))
    return parts


def build_block(stage, cfg, path, pose, mat, pm):
    L, W, H = cfg["blocks"]["size"]
    m = cfg["blocks"]["mass"]
    x, y, z, yaw = pose
    root = _xform(stage, path, (x, y, z), (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
    # inertia of the hollow section ~ solid box scaled (cores remove material near the centre)
    _rigid(root.GetPrim(), m, (m * (W * W + H * H) / 12 * 1.1, m * (L * L + H * H) / 12, m * (L * L + W * W) / 12))
    for name, size, c in block_parts(cfg):
        b = _uv_box(stage, f"{path}/{name}", size, c, mat=mat, collide=True, tile=0.5)
        _bind_phys(b.GetPrim(), pm)
    return root


def staging_poses(cfg):
    """Stock of blocks stacked in a rows x cols grid of `layers`-high stacks, top blocks are picked first."""
    st = cfg["blocks"]["staging"]
    L, W, H = cfg["blocks"]["size"]
    cx, cy = st["center"]
    gx, gy = st["gap"]
    poses = []
    for layer in range(st["layers"]):
        for r in range(st["rows"]):
            for c in range(st["cols"]):
                x = cx + (c - (st["cols"] - 1) / 2) * (L + gx)
                y = cy + (r - (st["rows"] - 1) / 2) * (W + gy)
                poses.append((x, y, H / 2 + layer * H, 0.0))
    return poses


def pick_order(staging):
    """Highest layer first, then row-major -> always picks an exposed top block."""
    idx = sorted(range(len(staging)), key=lambda i: (-round(staging[i][2], 3), staging[i][1], staging[i][0]))
    return idx


def pattern(cfg):
    task = cfg["task"]
    L, W, H = cfg["blocks"]["size"]
    cx, cy = task["build_center"]
    name = task["pattern"]
    poses = []
    J = task.get("joint", 0.006)
    if name == "column":
        poses = [(cx, cy, H / 2 + k * H, (k % 2) * np.pi / 2) for k in range(task["column"]["courses"])]
    elif name == "tower":
        tw = np.deg2rad(task["tower"]["twist_deg"])
        poses = [(cx, cy, H / 2 + k * H, k * tw) for k in range(task["tower"]["courses"])]
    elif name == "pyramid":
        n0 = task["pyramid"]["base"]
        for k in range(n0):
            n = n0 - k
            for i in range(n):
                poses.append((cx + (i - (n - 1) / 2) * (L + J), cy, H / 2 + k * H, 0.0))
    elif name == "compound":
        n = task["compound"]["blocks_per_side"]
        L = L + J                            # pitch along a wall includes the joint
        side = n * L                         # outer length of the through-running walls
        half = (side - W) / 2                # centre line of each wall
        for k in range(task["compound"]["courses"]):
            z = H / 2 + k * H
            through_x = k % 2 == 0           # alternate which walls run through the corners (interlocking corners)
            for sgn in (1, -1):
                for i in range(n):           # through walls
                    s = -side / 2 + L / 2 + i * L
                    poses.append((cx + s, cy + sgn * half, z, 0.0) if through_x else (cx + sgn * half, cy + s, z, np.pi / 2))
                inner = side - 2 * W         # infill walls between the through walls
                m = int(np.floor(inner / L + 1e-9))
                gap = (inner - m * L) / max(m, 1)
                for i in range(m):
                    s = -inner / 2 + gap / 2 + L / 2 + i * (L + gap)
                    poses.append((cx + sgn * half, cy + s, z, np.pi / 2) if through_x else (cx + s, cy + sgn * half, z, 0.0))
    elif name == "dome":
        d = task["dome"]
        for k in range(d["courses"]):
            r = d["base_radius"] - k * d["corbel"]
            # tangential blocks touch first at their inner corners: angular pitch >= 2 atan((L/2 + c) / (r - W/2))
            n = int(np.floor(np.pi / np.arctan((L / 2 + 0.006) / (r - W / 2))))
            if n < 3:
                break
            phase = (k % 2) * np.pi / n                       # stagger joints between courses
            for i in range(n):
                th = phase + 2 * np.pi * i / n
                poses.append((cx + r * np.cos(th), cy + r * np.sin(th), H / 2 + k * H, th + np.pi / 2))
    else:
        raise ValueError(f"unknown pattern {name}")
    return poses


def build_blocks(stage, cfg, n_needed, texture):
    """Spawn the whole staging stock. The first n_needed (in pick order) are named block_XXX and used by the task,
    the rest are spare_XXX (lower layers). Returns (paths, staging poses) of the used blocks."""
    _xform(stage, BLOCK_ROOT)
    mat = _material(stage, "/World/Looks/block", (0.62, 0.62, 0.60), 0.85, texture=texture)
    pm = _phys_material(stage, "/World/PhysicsMaterials/block", **cfg["blocks"]["friction"])
    stock = staging_poses(cfg)
    if len(stock) < n_needed:
        raise ValueError(f"pattern needs {n_needed} blocks but staging holds {len(stock)}; enlarge blocks.staging")
    order = pick_order(stock)
    paths = []
    for j, i in enumerate(order):
        p = f"{BLOCK_ROOT}/block_{j:03d}" if j < n_needed else f"{BLOCK_ROOT}/spare_{j - n_needed:03d}"
        build_block(stage, cfg, p, stock[i], mat, pm)
        if j < n_needed:
            paths.append(p)
    return paths, [stock[i] for i in order[:n_needed]]
