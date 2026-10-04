"""Process end-effectors for the application demos, and the material they lay down.

The physics model keeps the same hanging tool (wrist, load cell, tube, palm, jaws); these replace what is DRAWN
below the palm with a spray gun or a print head and hide the jaws. The deposited material (paint film, printed
bead) is geometry drawn along the path the tool tip actually took. It is not simulated as a fluid.
"""

import numpy as np
from pxr import Gf, UsdGeom, Vt

from .dressup import _mat, box, lathe, rbox
from .scene import _bind
from .site import _tube

PLAT = "/World/Platform"
PAINT = (0.04, 0.20, 0.52)                  # livery blue


def _palm_bottom(cfg):
    t = cfg["gripper"]["tool"]
    return -(t["tube_length"] / 2 + t["palm"][2])        # in the tool link frame (origin at the tube centre)


def _hide_jaws(stage):
    for n in ("finger_l", "finger_r"):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{PLAT}/{n}")).MakeInvisible()


def tool_tip(frame_row, drop):
    """World position of the tool tip from a recorded / simulated tool-link pose (x, y, z, qx, qy, qz, qw)."""
    x, y, z, qx, qy, qz, qw = (float(v) for v in frame_row)
    ax = np.array([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)])   # link z axis in world
    return np.array([x, y, z]) - ax * drop


def build_spray_tool(stage, cfg):
    """Spray gun on a lance under the palm. Returns (tip drop below the tool-link origin, path of the spray cone)."""
    _hide_jaws(stage)
    P = PLAT + "/tool/process"
    UsdGeom.Xform.Define(stage, P)
    m = {k: _mat(stage, "tool_" + k, *v) for k, v in {"body": ((0.07, 0.075, 0.085), 0.35, 0.4), "alu": ((0.78, 0.79, 0.8), 0.3, 1.0),
                                                       "rubber": ((0.02, 0.02, 0.02), 0.9), "cup": (PAINT, 0.25)}.items()}
    pb = _palm_bottom(cfg)
    rbox(stage, P + "/gun", (0.07, 0.10, 0.085), 0.012, m["body"], (0, 0, pb - 0.045))
    lathe(stage, P + "/cup", [(0, 0), (0.042, 0), (0.042, 0.10), (0.03, 0.115), (0, 0.115)], m["cup"], pos=(0.075, 0, pb - 0.07))
    lathe(stage, P + "/cup_lid", [(0, 0.115), (0.032, 0.115), (0.032, 0.125), (0, 0.125)], m["alu"], pos=(0.075, 0, pb - 0.07))
    _tube(stage, P + "/cup_neck", (0.04, 0, pb - 0.05), (0.07, 0, pb - 0.06), 0.012, m["alu"])
    z0, z1 = pb - 0.085, pb - 0.425
    _tube(stage, P + "/lance", (0, 0, z0), (0, 0, z1), 0.011, m["alu"])
    lathe(stage, P + "/air_cap", [(0, z1 - 0.03), (0.012, z1 - 0.03), (0.026, z1 - 0.012), (0.026, z1), (0, z1)], m["body"])
    _tube(stage, P + "/air_hose", (-0.03, 0.03, pb - 0.02), (-0.03, 0.03, pb + 0.26), 0.007, m["rubber"])
    _tube(stage, P + "/paint_hose", (-0.03, -0.03, pb - 0.02), (-0.03, -0.03, pb + 0.26), 0.006, m["cup"])
    tip = -(z1 - 0.03)
    mist = _mat(stage, "tool_mist", (0.75, 0.85, 1.0), 1.0, opacity=0.07, ior=1.0)
    lathe(stage, P + "/spray", [(0.006, -tip), (0.12, -tip - 0.17)], mist)
    return tip, [P + "/spray"]


def build_print_head(stage, cfg):
    """Hopper-fed extruder under the palm. Returns the tip drop below the tool-link origin."""
    _hide_jaws(stage)
    P = PLAT + "/tool/process"
    UsdGeom.Xform.Define(stage, P)
    body = _mat(stage, "tool_body", (0.07, 0.075, 0.085), 0.35, 0.4)
    alu = _mat(stage, "tool_alu", (0.78, 0.79, 0.8), 0.3, 1.0)
    acc = _mat(stage, "tool_accent", (1.0, 0.33, 0.04), 0.35)
    pb = _palm_bottom(cfg)
    lathe(stage, P + "/hopper", [(0.075, pb - 0.005), (0.075, pb - 0.03), (0.035, pb - 0.085), (0.035, pb - 0.115)], alu)
    lathe(stage, P + "/barrel", [(0.035, pb - 0.085), (0.04, pb - 0.085), (0.04, pb - 0.12), (0.035, pb - 0.12)], acc)
    lathe(stage, P + "/nozzle", [(0.035, pb - 0.12), (0.022, pb - 0.145), (0.022, pb - 0.15), (0.0, pb - 0.15)], body)
    box(stage, P + "/drive", (0.05, 0.06, 0.07), (-0.075, 0, pb - 0.06), body)
    _tube(stage, P + "/feed_hose", (0.05, 0.03, pb - 0.01), (0.05, 0.03, pb + 0.26), 0.012, body)
    return -(pb - 0.15)


class Trail:
    """Material laid down along the tool-tip path. kind "film": a flat ribbon on a surface z = zfun(x); kind "bead":
    a rectangular bead under the nozzle. Points are added as the tip moves; `show(n)` draws the first n segments."""

    def __init__(self, stage, path, kind, width, mat, zfun=None, height=0.03, spacing=0.02, clip=None):
        self.kind, self.w, self.h, self.zfun, self.spacing, self.clip = kind, width, height, zfun, spacing, clip
        self.pts, self.rows = [], []
        self.mesh = UsdGeom.Mesh.Define(stage, path)
        self.mesh.CreateSubdivisionSchemeAttr("none")
        self.mesh.CreateDoubleSidedAttr(True)
        _bind(self.mesh.GetPrim(), mat)
        self.per = 7 if kind == "film" else 4
        self.shown = -1

    def add(self, tip):
        """Append the tip position if it moved far enough. Returns True when a point was added."""
        tip = np.asarray(tip, float)
        if self.pts and np.linalg.norm(tip - self.pts[-1]) < self.spacing:
            return False
        if self.clip is not None and not (self.clip[0] <= tip[0] <= self.clip[1] and self.clip[2] <= tip[1] <= self.clip[3]):
            return False
        self.pts.append(tip)
        if len(self.pts) >= 2:
            a, b = self.pts[-2], self.pts[-1]
            d = b[:2] - a[:2]
            n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9) * self.w / 2
            if len(self.pts) == 2:
                self.rows.append(self._row(a, n))
            self.rows.append(self._row(b, n))
        return True

    def _row(self, p, n):
        l, r = p[:2] + n, p[:2] - n
        if self.kind == "film":                          # several points across the width so the film follows a curved skin
            q = [l + (r - l) * u for u in np.linspace(0, 1, self.per)]
            return [(v[0], v[1], float(self.zfun(v[0])) + 0.003) for v in q]
        zt = p[2] - 0.008
        return [(l[0], l[1], zt), (r[0], r[1], zt), (r[0], r[1], zt - self.h), (l[0], l[1], zt - self.h)]

    def show(self, n_rows=None):
        n = len(self.rows) if n_rows is None else min(n_rows, len(self.rows))
        if n == self.shown or n < 2:
            return
        self.shown = n
        P = np.array(self.rows[:n], np.float32).reshape(-1, 3)
        k, idx = self.per, []
        for i in range(n - 1):
            a, b = i * k, (i + 1) * k
            if self.kind == "film":
                for j in range(k - 1):
                    idx += [a + j, a + j + 1, b + j + 1, b + j]
            else:
                for j in (0, 1, 3):                       # top, right side, left side (the underside is never seen)
                    j2 = (j + 1) % 4
                    idx += [a + j, a + j2, b + j2, b + j]
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P)) if self.mesh.GetPointsAttr() else self.mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P))
        self.mesh.CreateFaceVertexCountsAttr(Vt.IntArray(len(idx) // 4, 4))
        self.mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        ext = [Gf.Vec3f(*map(float, P.min(0))), Gf.Vec3f(*map(float, P.max(0)))]
        self.mesh.CreateExtentAttr(ext)


class Hose:
    """Supply hose from a fixed outlet to the tool: down from the outlet, along the ground to a touchdown point, then
    an arch up to the feed port that follows the tool. Drawn as a tube and updated every frame; it carries no load
    in the physics."""

    def __init__(self, stage, path, outlet, touchdown, mat, radius=0.038, sides=10):
        self.r, self.n = radius, sides
        o, t = np.asarray(outlet, float), np.asarray(touchdown, float)
        sx = np.sign(t[0] - o[0])
        g = np.array([o[0] + 0.55 * sx, o[1], radius])    # where the drop from the outlet lands
        self.fixed = np.vstack([self._bez(o, o + [0.35 * sx, 0, 0.05], g + [0, 0, 0.5], g, 10),
                                self._bez(g, g + (t - g) * 0.4 + [0, 0.5, 0], t - (t - g) * 0.3 + [0, -0.4, 0], t, 26)[1:]])
        self.t = t
        self.mesh = UsdGeom.Mesh.Define(stage, path)
        self.mesh.CreateSubdivisionSchemeAttr("none")
        _bind(self.mesh.GetPrim(), mat)
        self.topo = False

    @staticmethod
    def _bez(p0, p1, p2, p3, n):
        s = np.linspace(0, 1, n)[:, None]
        return (1 - s) ** 3 * p0 + 3 * s * (1 - s) ** 2 * p1 + 3 * s**2 * (1 - s) * p2 + s**3 * p3

    def update(self, feed):
        feed = np.asarray(feed, float)
        d = self.t[:2] - feed[:2]
        d = np.r_[d / max(np.linalg.norm(d), 1e-6), 0.0]
        arch = self._bez(self.t, self.t - d * 0.9 + [0, 0, 0.05], feed + d * 0.55 + [0, 0, 0.55], feed, 28)
        c = np.vstack([self.fixed, arch[1:]])
        tan = np.gradient(c, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        a = np.cross(tan, [0.0, 0.0, 1.0])
        bad = np.linalg.norm(a, axis=1) < 1e-3
        a[bad] = np.cross(tan[bad], [1.0, 0, 0])
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = np.cross(a, tan)
        ang = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        ring = np.cos(ang)[None, :, None] * a[:, None, :] + np.sin(ang)[None, :, None] * b[:, None, :]
        P = (c[:, None, :] + self.r * ring).reshape(-1, 3).astype(np.float32)
        N = ring.reshape(-1, 3).astype(np.float32)
        if not self.topo:
            m, n = len(c), self.n
            idx = []
            for i in range(m - 1):
                for j in range(n):
                    j2 = (j + 1) % n
                    idx += [i * n + j, i * n + j2, (i + 1) * n + j2, (i + 1) * n + j]
            self.mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P))
            self.mesh.CreateFaceVertexCountsAttr(Vt.IntArray(len(idx) // 4, 4))
            self.mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
            self.mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N))
            self.mesh.SetNormalsInterpolation("vertex")
            self.mesh.CreateExtentAttr([Gf.Vec3f(-20, -20, -1), Gf.Vec3f(20, 20, 6)])
            self.topo = True
        else:
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P))
            self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(N))


def feed_point(frame_row, cfg):
    """World position of the print head's feed port from the tool-link pose."""
    return tool_tip(frame_row, -(_palm_bottom(cfg) + 0.27)) + np.array([0.05, 0.03, 0.0])


TANKER_POSE = ((6.3, 0.0, 0.0), 90.0)       # tanker for the printing demo: parked beside the work area, pump towards it
HOSE_TOUCHDOWN = (2.35, -0.5, 0.04)
