"""Physical models and low-level controllers, vectorised over robots. Pure NumPy, no Isaac imports.

Cable  : massless, tension-only Kelvin-Voigt element between two body-fixed points.
         T = max(0, EA/L * (d - L) + c/L * (d_dot - L_dot)) when d > L, else 0.
Winch  : drum with reflected inertia, viscous + Coulomb friction and a linear torque-speed motor envelope.
         Written in cable coordinates: m_eq L_ddot = F_motor + T - c_v L_dot - F_c sgn(L_dot), m_eq = J / r^2.
Drone  : rigid body, four rotors with first-order speed dynamics (thrust = kf w^2, yaw drag = km/kf * thrust),
         rotor-induced linear drag + quadratic body drag. Geometric SE(3) controller + inverse mixer.
UGV    : wheel velocity servo is the PhysX joint drive; here only the unicycle command shaping.
"""

import numpy as np

from .mathutil import cross, quat_to_rot, rot_error, yaw_of

E3 = np.array([0.0, 0.0, 1.0])


class CableWinchModel:
    def __init__(self, cable, winch, n):
        self.n = n
        self.EA = cable["EA"]
        self.cEA = cable["damping_EA"]
        self.rho = cable["linear_density"]
        self.Lmin, self.Lmax = cable["min_length"], cable["max_length"]
        r = winch["drum_radius"]
        self.r = r
        self.m_eq = winch["inertia"] / r**2
        self.c_v = winch["viscous"] / r**2
        self.F_c = winch["coulomb"] / r
        self.F_max = winch["max_torque"] / r
        self.v_max = winch["max_speed"] * r
        self.wn, self.zeta = winch["wn"], winch["zeta"]
        self.L = np.full(n, 1.0)
        self.L_dot = np.zeros(n)
        self.T = np.zeros(n)
        self.F_motor = np.zeros(n)
        # command state (length mode with tension feed-forward, or pure tension mode)
        self.mode = np.array(["length"] * n, dtype=object)
        self.L_cmd = self.L.copy()
        self.Ld_cmd = np.zeros(n)
        self.T_ff = np.zeros(n)

    def tension(self, pa, va, pb, vb):
        """pa/va: platform-side points & velocities, pb/vb: robot-side. Returns (T, u, d) with u pointing a->b."""
        d_vec = pb - pa
        d = np.linalg.norm(d_vec, axis=1)
        u = d_vec / np.maximum(d, 1e-9)[:, None]
        d_dot = np.einsum("ij,ij->i", u, vb - va)
        stretch = d - self.L
        T = self.EA / self.L * stretch + self.cEA / self.L * (d_dot - self.L_dot)
        self.T = np.where(stretch > 0.0, np.maximum(T, 0.0), 0.0)
        return self.T, u, d

    def step(self, dt):
        e = self.L_cmd - self.L
        e_dot = self.Ld_cmd - self.L_dot
        servo = self.m_eq * (self.wn**2 * e + 2.0 * self.zeta * self.wn * e_dot)
        # tension mode: constant-force winch with damping on cable speed (the drum servo damps, doesn't hold length)
        damp = self.m_eq * 2.0 * self.zeta * self.wn * (self.Ld_cmd - self.L_dot)
        F = np.where(self.mode == "length", -self.T_ff + servo, -self.T_ff + damp)
        # motor envelope: available force drops linearly to zero at no-load speed when driving along the motion
        driving = np.sign(F) == np.sign(self.L_dot)
        F_avail = np.where(driving, self.F_max * np.clip(1.0 - np.abs(self.L_dot) / self.v_max, 0.0, 1.0), self.F_max)
        self.F_motor = np.clip(F, -F_avail, F_avail)
        acc = (self.F_motor + self.T - self.c_v * self.L_dot - self.F_c * np.tanh(self.L_dot / 1e-3)) / self.m_eq
        self.L_dot += acc * dt
        self.L += self.L_dot * dt
        hit = (self.L < self.Lmin) | (self.L > self.Lmax)
        self.L = np.clip(self.L, self.Lmin, self.Lmax)
        self.L_dot[hit] = 0.0

    def unstretched_for(self, d, T):
        return d / (1.0 + T / self.EA)


class QuadrotorModel:
    """Four identical X quadrotors. Rotor order: front-left, rear-left, rear-right, front-right (CCW from +x+y)."""

    def __init__(self, p, n, g):
        self.n, self.g = n, g
        self.m = p["mass"]
        self.J = np.diag(p["inertia"])
        self.kf = p["rotor_kf"]
        self.km = p["rotor_km_over_kf"]
        self.f_max = p["rotor_max_thrust"]
        self.w_max = np.sqrt(self.f_max / self.kf)
        self.tau = p["motor_tau"]
        self.drag_lin = p["drag_linear"]
        self.drag_quad = p["drag_quadratic"]
        a = p["arm_length"] / np.sqrt(2.0)
        xs = np.array([a, -a, -a, a])
        ys = np.array([a, a, -a, -a])
        spin = np.array([1.0, -1.0, 1.0, -1.0])  # +1: CCW rotor -> reaction torque -z... sign folded into km
        self.mix = np.vstack([np.ones(4), ys, -xs, -spin * self.km])  # [f, Mx, My, Mz] = mix @ f_i
        self.mix_inv = np.linalg.inv(self.mix)
        self.w = np.full((n, 4), np.sqrt(self.m * g / 4 / self.kf))
        c = p["control"]
        self.kp, self.kd, self.ki = (np.array(c[k]) * self.m for k in ("kp", "kd", "ki"))
        self.i_lim = c["i_limit"]
        self.kR, self.kW = np.array(c["kR"]), np.array(c["kW"])
        self.cable_ff = c["cable_feedforward"]
        self.r_att = np.array(p["attach_offset"], float)
        self.max_tilt = np.deg2rad(c["max_tilt_deg"])
        self.e_int = np.zeros((n, 3))
        # references
        self.p_ref = np.zeros((n, 3))
        self.v_ref = np.zeros((n, 3))
        self.yaw_ref = np.zeros(n)
        self.f_cmd = np.zeros((n, 4))

    def init_hover(self, extra_down_force):
        f = (self.m * self.g + extra_down_force) / 4.0
        self.w[:] = np.sqrt(np.clip(f, 0, self.f_max) / self.kf)[:, None]

    def control(self, pos, quat, vel, omega_w, F_ext_est, dt, R=None):
        R = quat_to_rot(quat) if R is None else R
        W = np.einsum("nji,nj->ni", R, omega_w)  # body rates
        ep = pos - self.p_ref
        ev = vel - self.v_ref
        self.e_int = np.clip(self.e_int + ep * dt, -self.i_lim / np.maximum(self.ki, 1e-9), self.i_lim / np.maximum(self.ki, 1e-9))
        F = -self.kp * ep - self.kd * ev - self.ki * self.e_int + self.m * self.g * E3
        if self.cable_ff:
            F = F - F_ext_est
        # tilt limit: keep the vertical component, shrink the horizontal one
        Fz = np.maximum(F[:, 2], 0.2 * self.m * self.g)
        h = F[:, :2]
        h_max = Fz * np.tan(self.max_tilt)
        h_norm = np.linalg.norm(h, axis=1)
        h *= np.minimum(1.0, h_max / np.maximum(h_norm, 1e-9))[:, None]
        F = np.column_stack([h, Fz])
        b3 = F / np.linalg.norm(F, axis=1, keepdims=True)
        b1c = np.column_stack([np.cos(self.yaw_ref), np.sin(self.yaw_ref), np.zeros(self.n)])
        b2 = cross(b3, b1c)
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        b1 = cross(b2, b3)
        Rd = np.stack([b1, b2, b3], axis=2)
        thrust = np.einsum("ni,ni->n", F, R[:, :, 2])
        eR = rot_error(Rd, R)
        M = -self.kR * eR - self.kW * W + cross(W, W @ self.J.T)
        if self.cable_ff:
            # cancel the moment of the cable force about the CoM (fairlead is offset from the CoM)
            F_ext_b = np.einsum("nji,nj->ni", R, F_ext_est)
            M = M - cross(self.r_att, F_ext_b)
        u = np.column_stack([thrust, M])
        f = u @ self.mix_inv.T
        self.f_cmd = np.clip(f, 0.0, self.f_max)

    def actuate(self, quat, vel, dt, R=None):
        """Rotor dynamics -> world force & torque on the body CoM."""
        w_cmd = np.sqrt(self.f_cmd / self.kf)
        self.w += (w_cmd - self.w) * (dt / self.tau)
        self.w = np.clip(self.w, 0.0, self.w_max)
        f = self.kf * self.w**2
        u = f @ self.mix.T  # [thrust, Mx, My, Mz] body
        R = quat_to_rot(quat) if R is None else R
        F_body = np.zeros((self.n, 3))
        F_body[:, 2] = u[:, 0]
        v_body = np.einsum("nji,nj->ni", R, vel)
        F_body[:, :2] -= self.drag_lin * v_body[:, :2]
        F = np.einsum("nij,nj->ni", R, F_body)
        F -= self.drag_quad * np.linalg.norm(vel, axis=1, keepdims=True) * vel
        tau = np.einsum("nij,nj->ni", R, u[:, 1:])
        return F, tau


class DiffDriveCommand:
    """Unicycle command shaping + go-to-pose controller for the UGVs. Output: wheel speed targets [rad/s]."""

    def __init__(self, p, n):
        self.n = n
        self.r = p["wheel_radius"]
        self.b = p["track_width"]
        self.w_max = p["wheel_max_speed"]
        self.a_lin, self.a_ang = p["max_lin_acc"], p["max_ang_acc"]
        self.v = np.zeros(n)
        self.om = np.zeros(n)
        # "pose": formation-driven go-to-goal, "goal": user go-to-goal (not overwritten), "velocity": twist command
        self.mode = np.array(["pose"] * n, dtype=object)
        self.goal = np.zeros((n, 3))  # x, y, yaw (yaw = nan -> keep current heading on arrival)
        self.goal_vel = np.zeros((n, 2))  # goal velocity feed-forward (moving formation)
        self.cmd_v = np.zeros(n)
        self.cmd_om = np.zeros(n)
        self.pos_tol = 0.03

    def step(self, pos, quat, dt, R=None):
        R = quat_to_rot(quat) if R is None else R
        yaw = yaw_of(R)
        v_t, om_t = self.cmd_v.copy(), self.cmd_om.copy()
        pose_mode = (self.mode == "pose") | (self.mode == "goal")
        if pose_mode.any():
            # vector-field go-to-goal: desired planar velocity = goal velocity feed-forward + P on position
            e = self.goal[:, :2] - pos[:, :2]
            vd = self.goal_vel + 1.0 * e
            sp = np.linalg.norm(vd, axis=1)
            vd *= np.minimum(1.0, 0.6 / np.maximum(sp, 1e-9))[:, None]
            sp = np.minimum(sp, 0.6)
            e_head = np.angle(np.exp(1j * (np.arctan2(vd[:, 1], vd[:, 0]) - yaw)))
            back = np.abs(e_head) > np.pi / 2  # drive backwards if that needs less turning
            e_head = np.where(back, np.angle(np.exp(1j * (e_head + np.pi))), e_head)
            v_goal = sp * np.cos(e_head) * np.where(back, -1.0, 1.0)
            om_goal = 2.0 * e_head
            moving = np.linalg.norm(self.goal_vel, axis=1) > 1e-3
            arrived = (np.linalg.norm(e, axis=1) < self.pos_tol) & ~moving
            yaw_goal = self.goal[:, 2]
            e_yaw = np.where(np.isnan(yaw_goal), 0.0, np.angle(np.exp(1j * (np.nan_to_num(yaw_goal) - yaw))))
            v_goal = np.where(arrived, 0.0, v_goal)
            om_goal = np.where(arrived, 1.5 * e_yaw, om_goal)
            v_t = np.where(pose_mode, v_goal, v_t)
            om_t = np.where(pose_mode, om_goal, om_t)
        self.v += np.clip(v_t - self.v, -self.a_lin * dt, self.a_lin * dt)
        self.om += np.clip(om_t - self.om, -self.a_ang * dt, self.a_ang * dt)
        wl = (self.v - 0.5 * self.b * self.om) / self.r
        wr = (self.v + 0.5 * self.b * self.om) / self.r
        return np.clip(np.column_stack([wl, wr]), -self.w_max, self.w_max)
