"""Platform-level CDPR control: structure matrix, tension distribution, inverse kinematics, formation targets.

Platform wrench balance (world frame, about the platform CoM):
    W(x) t + w_g = [m a ; I alpha + omega x I omega],  W = [u_i ; (R b_i) x u_i]
with u_i the unit vector from platform corner i towards its robot anchor and t_i >= 0 the cable tensions.
8 cables / 6 DOF -> 2-dimensional null space, resolved by the closed-form method with bound handling
(Pott, Bruckmann & Mikelsons 2009): t = t_m + W^+ (w - W t_m), clamping violated tensions and re-solving.
"""

import numpy as np

from .mathutil import cross, rot_error


def structure_matrix(p, R, b_local, anchors):
    rb = b_local @ R.T
    d = anchors - (p + rb)
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.vstack([u.T, cross(rb, u).T]), u, np.linalg.norm(d, axis=1)


def tension_distribution(W, w, t_min, t_max, t_ref=None, damping=1e-9):
    """Returns (t, feasible). Solves W t = w with t_min <= t <= t_max closest to t_ref (default: mid tension).
    t_max may be a per-cable vector (e.g. drone thrust/tilt limits mapped to tension bounds)."""
    n = W.shape[1]
    t_max = np.broadcast_to(np.asarray(t_max, float), (n,))
    t_ref = 0.5 * (t_min + t_max) if t_ref is None else t_ref
    t = np.full(n, t_ref)
    free = np.ones(n, bool)
    t_fixed = np.zeros(n)
    for _ in range(n - 6 + 1):
        Wf = W[:, free]
        w_res = w - W[:, ~free] @ t_fixed[~free]
        t_m = np.minimum(np.full(free.sum(), t_ref), 0.5 * (t_min + t_max[free]))
        # damped minimum-norm correction W^T (W W^T + lambda I)^-1 r. Wrench directions with singular value
        # sigma << sqrt(lambda) are attenuated by sigma^2 / (sigma^2 + lambda) instead of being forced through tiny
        # moment arms (yaw just off the straight layout); they are then held by the cable geometry/pretension.
        tf = t_m + Wf.T @ np.linalg.solve(Wf @ Wf.T + damping * np.eye(6), w_res - Wf @ t_m)
        t[free] = tf
        t[~free] = t_fixed[~free]
        viol = np.maximum(t_min - t, t - t_max) * free
        if viol.max() <= 1e-6:
            return t, True
        k = int(np.argmax(viol))
        free[k] = False
        t_fixed[k] = t_min if t[k] < t_min else t_max[k]
        if free.sum() < 6:
            break
    return np.clip(t, t_min, t_max), False


def drone_tension_caps(u, m, g, theta_max, f_max):
    """Largest tension each drone cable may carry (u: unit vectors platform -> drone, shape (k, 3)).
    The drone must produce f = m g e3 + t u.
      tilt  <= theta_max :  t (|u_xy| - u_z tan theta) <= m g tan theta   (binding only if elevation < 90 - theta)
      |f|   <= f_max     :  t <= -m g u_z + sqrt((m g u_z)^2 - (m g)^2 + f_max^2)"""
    mg = m * g
    uxy, uz = np.linalg.norm(u[:, :2], axis=1), u[:, 2]
    tan = np.tan(theta_max)
    den = uxy - uz * tan
    t_tilt = np.where(den > 1e-9, mg * tan / np.maximum(den, 1e-9), np.inf)
    t_thrust = -mg * uz + np.sqrt(np.maximum((mg * uz) ** 2 - mg**2 + f_max**2, 0.0))
    return np.minimum(t_tilt, t_thrust)


class PlatformController:
    def __init__(self, cfg, b_local, g):
        pc = cfg["platform"]
        self.m = pc["mass"]
        s = pc["side"]
        self.I = np.eye(3) * self.m * s * s / 6.0
        self.b = b_local
        self.g = g
        w = cfg["winch"]
        self.t_min, self.t_max, self.t_ref = w["tension_min"], w["tension_max"], w["tension_ref"]
        self.damping = w["tension_damping"]
        d = cfg["drone"]
        caps = d["control"]["tension_caps"]
        self.drone_caps = caps["enabled"]   # drone-aware tension distribution (cables 0-3 go to drones)
        self.cap_args = (d["mass"], g, np.deg2rad(d["control"]["max_tilt_deg"] - caps["tilt_margin_deg"]),
                         caps["thrust_fraction"] * 4 * d["rotor_max_thrust"])
        # optional wrench-level PD on top of the kinematic (length) control; 0 = pure IK + gravity feed-forward
        self.kp_lin, self.kd_lin = 0.0, 0.0
        self.kp_rot, self.kd_rot = 0.0, 0.0
        self.ki_lin, self.ki_rot = 0.0, 0.0     # integral action (used when not all cables are length-controlled)
        self.i_lim = 2.0                        # integral authority [m/s^2 | rad/s^2]
        self.e_int = np.zeros(6)
        self.w_ext = np.zeros(6)                # measured external wrench on the platform (hanging tool + payload)
        dmp = w.get("platform_damping", {"lin": 0.0, "rot": 0.0})
        self.c_lin, self.c_rot = dmp["lin"], dmp["rot"]   # active damping [N s/m, N m s/rad]
        self.p_ref = np.array(pc["start_position"], float)
        self.R_ref = np.eye(3)
        self.v_ref = np.zeros(3)
        self.a_ref = np.zeros(3)
        self.feasible = True
        self.t_des = np.full(len(b_local), self.t_min)

    def required_wrench(self, p, R, v, om, dt=0.0):
        e_p = self.p_ref - p
        e_r = -(R @ rot_error(self.R_ref, R))
        if dt > 0 and (self.ki_lin or self.ki_rot):
            self.e_int += dt * np.r_[e_p, e_r]
            lim = self.i_lim / np.maximum(np.r_[[self.ki_lin] * 3, [self.ki_rot] * 3], 1e-9)
            self.e_int = np.clip(self.e_int, -lim, lim)
        a = self.a_ref + self.kp_lin * e_p + self.kd_lin * (self.v_ref - v) + self.ki_lin * self.e_int[:3]
        alpha = self.kp_rot * e_r - self.kd_rot * om + self.ki_rot * self.e_int[3:]
        f = self.m * (a + np.array([0.0, 0.0, self.g])) - self.w_ext[:3] - self.c_lin * (v - self.v_ref)
        tau = self.I @ alpha + cross(om, self.I @ om) - self.w_ext[3:] - self.c_rot * om
        return np.r_[f, tau]

    def solve(self, anchors, p=None, R=None, v=None, om=None, dt=0.0):
        """Tensions + unstretched cable lengths that hold the *reference* pose given the current anchor positions."""
        p = self.p_ref if p is None else p
        R = self.R_ref if R is None else R
        v = np.zeros(3) if v is None else v
        om = np.zeros(3) if om is None else om
        w = self.required_wrench(p, R, v, om, dt)
        W, u, d = structure_matrix(self.p_ref, self.R_ref, self.b, anchors)
        t_max = np.full(len(self.b), float(self.t_max))
        if self.drone_caps:
            t_max[:4] = np.clip(drone_tension_caps(u[:4], *self.cap_args), self.t_min, self.t_max)
        self.t_cap = t_max
        self.t_des, self.feasible = tension_distribution(W, w, self.t_min, t_max, self.t_ref, self.damping)
        return self.t_des, d


def corner_points(side):
    h = side / 2.0
    az = np.deg2rad(45.0 + 90.0 * np.arange(4))
    xy = np.sqrt(2.0) * h * np.column_stack([np.cos(az), np.sin(az)])
    top = np.column_stack([xy, np.full(4, h)])
    bot = np.column_stack([xy, np.full(4, -h)])
    return np.vstack([top, bot])  # cables 0-3 -> drones 0-3, cables 4-7 -> UGVs 0-3


def formation(cfg, center_xy, cross_top=None, cross_bot=None):
    """Nominal anchor (fairlead) positions for a platform centred above `center_xy`.
    cross_top / cross_bot [rad] override the configured crossing angles (used for live reconfiguration)."""
    L = cfg["layout"]
    az = np.deg2rad(45.0 + 90.0 * np.arange(4))
    azd = az + (np.deg2rad(L["drone_cross_angle_deg"]) if cross_top is None else cross_top)
    azg = az + (np.deg2rad(L["ugv_cross_angle_deg"]) if cross_bot is None else cross_bot)
    c = np.r_[center_xy, 0.0]
    drones = c + np.column_stack([L["drone_radius"] * np.cos(azd), L["drone_radius"] * np.sin(azd), np.full(4, L["drone_altitude"])])
    ugvs = c + np.column_stack([L["ugv_radius"] * np.cos(azg), L["ugv_radius"] * np.sin(azg), np.full(4, cfg["ugv"]["attach_height"])])
    return drones, ugvs


def hold_table(cfg, fraction=None):
    """Hold radius R(z): how far the platform may be from the formation centre before the robots have to move.

    For a fixed formation and a level platform, the tension distribution is tested at increasing horizontal offsets
    in 16 directions, for the bare tool and for the tool with one block, with the drone cables capped by the drone
    tilt limit (minus its margin) and 85% thrust. The smallest feasible offset over directions and loads, times
    `fraction` (layout.hold_fraction), is the hold radius at that height. Returns (zs, R)."""
    L, g, w_, d = cfg["layout"], cfg["gripper"], cfg["winch"], cfg["drone"]
    frac = L.get("hold_fraction", 0.0) if fraction is None else fraction
    zs = np.arange(0.5, 2.31, 0.1)
    if frac <= 0:
        return zs, np.zeros(len(zs))
    b = corner_points(cfg["platform"]["side"])
    dr, ug = formation(cfg, np.zeros(2))
    A = np.vstack([dr, ug])
    grav = cfg["sim"]["gravity"]
    m_tool = cfg["platform"]["mass"] + g["wrist"]["mass"] + g["tool"]["mass"] + 2 * g["finger"]["mass"]
    th = np.deg2rad(d["control"]["max_tilt_deg"] - d["control"]["tension_caps"]["tilt_margin_deg"])
    f_max = 0.85 * 4 * d["rotor_max_thrust"]

    def feasible(p, m):
        W, u, _ = structure_matrix(p, np.eye(3), b, A)
        tb = np.full(8, float(w_["tension_max"]))
        tb[:4] = np.minimum(tb[:4], drone_tension_caps(u[:4], d["mass"], grav, th, f_max))
        return tension_distribution(W, np.array([0, 0, m * grav, 0, 0, 0]), w_["tension_min"], tb, w_["tension_ref"])[1]

    R = []
    for z in zs:
        r_min = 9.0
        for m in (m_tool, m_tool + cfg["blocks"]["mass"]):
            for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                r = 0.0
                while r < r_min and feasible(np.array([(r + 0.04) * np.cos(a), (r + 0.04) * np.sin(a), z]), m):
                    r += 0.04
                r_min = min(r_min, r)
        R.append(frac * r_min)
    return zs, np.array(R)


def lazy_center(c, p, radius):
    """Move the formation centre c the least distance that brings the platform p (xy) back within `radius` of it."""
    d = np.asarray(p, float)[:2] - c
    n = np.linalg.norm(d)
    return c if n <= radius else np.asarray(p, float)[:2] - d / n * radius
