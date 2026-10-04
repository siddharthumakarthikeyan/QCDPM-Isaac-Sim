"""Bridge between the simulation and the C++ core (core/, built by core/build.sh into cdpr_core*.so).

params_from_cfg  config/cdpr.yaml -> cdpr_core.Params (no Isaac imports; also used by the Isaac-free tests)
CoreBridge       per physics step: true PhysX state -> simulated sensors -> estimators -> controllers -> winch,
                 rotor and wheel commands. Enabled with `core.enabled: true` in the config; otherwise the Python
                 controllers in cdpr_control.py / models.py run on the true state as before.
"""

import numpy as np

from . import cdpr_core
from .cdpr_control import corner_points
from .mathutil import yaw_of


def params_from_cfg(cfg):
    P = cdpr_core.Params()
    P.g, P.dt = cfg["sim"]["gravity"], cfg["sim"]["physics_dt"]
    rates = cfg["sim"]["control_rates"]
    P.div_drone, P.div_platform, P.div_ugv = (max(1, int(round(1.0 / (rates[k] * P.dt)))) for k in ("drone", "platform", "ugv"))
    cc = cfg.get("core", {})
    P.hybrid = cfg["winch"]["drone_cable_mode"] == "tension"
    P.use_truth = bool(cc.get("use_truth", False))
    P.dob_ff = bool(cc.get("disturbance_feedforward", True))

    pl = P.platform
    pl.m = cfg["platform"]["mass"]
    pl.J = np.eye(3) * pl.m * cfg["platform"]["side"] ** 2 / 6.0
    pl.b = corner_points(cfg["platform"]["side"])
    P.platform = pl

    c, cb = P.cable, cfg["cable"]
    c.EA, c.cEA, c.rho, c.Lmin, c.Lmax = cb["EA"], cb["damping_EA"], cb["linear_density"], cb["min_length"], cb["max_length"]
    P.cable = c

    w, wc = P.winch, cfg["winch"]
    r = wc["drum_radius"]
    w.m_eq, w.c_v, w.F_c = wc["inertia"] / r**2, wc["viscous"] / r**2, wc["coulomb"] / r
    w.F_max, w.v_max, w.wn, w.zeta = wc["max_torque"] / r, wc["max_speed"] * r, wc["wn"], wc["zeta"]
    P.winch = w

    d, dc = P.drone, cfg["drone"]
    d.m, d.J = dc["mass"], np.diag(dc["inertia"])
    d.kf, d.km, d.f_max, d.tau = dc["rotor_kf"], dc["rotor_km_over_kf"], dc["rotor_max_thrust"], dc["motor_tau"]
    d.drag_lin, d.drag_quad, d.arm = dc["drag_linear"], dc["drag_quadratic"], dc["arm_length"]
    d.r_att = dc["attach_offset"]
    ctl = dc["control"]
    d.kp, d.kd, d.ki, d.i_lim = ctl["kp"], ctl["kd"], ctl["ki"], ctl["i_limit"]
    d.kR, d.kW = ctl["kR"], ctl["kW"]
    d.max_tilt, d.cable_ff = np.deg2rad(ctl["max_tilt_deg"]), ctl["cable_feedforward"]
    P.drone = d

    u, uc = P.ugv, cfg["ugv"]
    cs = uc["chassis_size"]
    half_track, caster = uc["track_width"] / 2, uc["caster_offset_x"]
    u.m = uc["chassis_mass"] + 2 * uc["wheel_mass"]
    u.Iz = uc["chassis_mass"] * (cs[0] ** 2 + cs[1] ** 2) / 12 + 2 * uc["wheel_mass"] * half_track**2
    u.wheel_r, u.track, u.wheel_J = uc["wheel_radius"], uc["track_width"], 0.5 * uc["wheel_mass"] * uc["wheel_radius"] ** 2
    u.wheel_kd, u.wheel_tau_max, u.wheel_w_max = uc["wheel_drive_damping"], uc["wheel_max_torque"], uc["wheel_max_speed"]
    u.a_lin, u.a_ang, u.h_att, u.mu = uc["max_lin_acc"], uc["max_ang_acc"], uc["attach_height"], uc["friction"]["dynamic"]
    u.tip_radius = half_track * caster / np.hypot(half_track, caster)   # support polygon: wheels and casters (a rhombus)
    P.ugv = u

    t = P.tension
    caps = ctl["tension_caps"]
    t.t_min, t.t_max, t.t_ref, t.lam = wc["tension_min"], wc["tension_max"], wc["tension_ref"], wc["tension_damping"]
    t.w_rate = cc.get("tension_rate_weight", 0.0)
    t.t_min_length = cc.get("t_min_ground", t.t_min)
    t.t_ref = cc.get("t_ref", t.t_ref)
    t.lambda_min = float(cc.get("lambda_min", t.lambda_min))
    t.drone_caps, t.ugv_caps = bool(cc.get("drone_caps", True)), bool(cc.get("ground_caps", True))
    t.tilt_cap = np.deg2rad(ctl["max_tilt_deg"] - caps["tilt_margin_deg"])
    t.thrust_cap = caps["thrust_fraction"] * 4 * dc["rotor_max_thrust"]
    t.ground_margin = cc.get("ground_margin", 0.8)
    P.tension = t

    g, pd = P.gains, wc["hybrid_pd"]
    wn, z = pd["wn"], pd["zeta"]
    g.kp_lin = g.kp_rot = wn**2
    g.kd_lin = g.kd_rot = 2 * z * wn
    g.ki_lin = g.ki_rot = pd["ki_ratio"] * wn**3
    g.k_twist, g.twist_max, g.k_ik = cc.get("k_twist", 3.5), cc.get("twist_max", 0.05), cc.get("k_ik", 1.0)
    # the admittance exists for estimated anchors; with the true state it only chases cable dynamics
    g.k_adm = 0.0 if P.use_truth else cc.get("k_adm", 1.0e-3)
    g.adm_max, g.adm_band = cc.get("adm_max", 0.02), cc.get("adm_band", 0.0)
    g.t_guard, g.k_guard = (0.0, 0.0) if P.use_truth else (cc.get("t_guard", 4.0), cc.get("k_guard", 0.02))
    g.a_max, g.alpha_max, g.k_gov = cc.get("ref_acc", 2.0), cc.get("ref_ang_acc", 1.5), cc.get("ref_gain", 10.0)
    P.gains = g

    s, sc = P.sensors, cc.get("sensors", {})
    s.noise, s.seed = bool(sc.get("noise", True)), int(sc.get("seed", 7))
    for k in ("gyro_nd", "acc_nd", "drone_fix_sigma", "drone_yaw_sigma", "wheel_sigma", "fair_sigma", "ugv_fix_sigma", "ugv_yaw_sigma",
              "enc_sigma", "enc_rate_sigma", "load_sigma", "tag_sigma_p", "tag_sigma_th"):
        if k in sc:
            setattr(s, k, float(sc[k]))
    for k, key in (("div_drone_fix", "drone_fix_hz"), ("div_ugv_fix", "ugv_fix_hz"), ("div_tag", "tag_hz")):
        if key in sc:
            setattr(s, k, max(1, int(round(1.0 / (sc[key] * P.dt)))))
    P.sensors = s

    o, oc = P.observer, cc.get("observer", {})
    for k in ("q_acc", "q_alpha", "q_df", "q_dtau", "sigma_cable_drone", "sigma_cable_ugv", "sigma_tag_p", "sigma_tag_th",
              "t_slack", "gate"):
        if k in oc:
            setattr(o, k, float(oc[k]))
    o.sag = bool(oc.get("sag", False))
    P.observer = o
    return P


class CoreBridge:
    def __init__(self, rt):
        self.P = params_from_cfg(rt.cfg)
        self.core = cdpr_core.Core(self.P)
        self.truth = cdpr_core.Truth()
        self.refs = cdpr_core.Refs()
        self.wheel = np.zeros((4, 2))
        self._wheel_q = None
        self._R_ref = np.eye(3)
        self._w_ref = np.zeros(3)

    def _fill_truth(self, rt, T, acc):
        s, t = rt.state, self.truth
        R = s["R"]
        t.p, t.v, t.R, t.w = s["pos"][0], s["v"][0], R[0], R[0].T @ s["w"][0]
        t.set_drones(s["pos"][1:5], s["v"][1:5], list(R[1:5]), np.einsum("nji,nj->ni", R[1:5], s["w"][1:5]))
        t.drone_acc = acc[1:5]
        t.ugv = np.column_stack([s["pos"][5:9, :2], yaw_of(R[5:9])])
        t.wheel = self.wheel
        fair = np.einsum("nij,j->ni", R[5:9], rt.r_ugv)        # the mast top moves as the base rocks on its casters
        fair[:, 2] += s["pos"][5:9, 2]
        t.ugv_fair = fair
        t.L, t.Ldot, t.T = rt.cables.L, rt.cables.L_dot, T

    def reset(self, rt, T):
        self._fill_truth(rt, T, np.zeros((9, 3)))
        self.core.reset(self.truth)
        pc = self.core.platform
        pc.t_prev = rt.ctrl.t_des
        self.core.platform = pc
        self._R_ref = rt.ctrl.R_ref.copy()

    def step(self, rt, T, acc, run_platform, dt_p):
        core, P = self.core, self.P
        ugv_due = core.k % P.div_ugv == 0
        if ugv_due:   # wheel encoders: counts over the base-controller period (not a sampled speed)
            q = rt.art.get_dof_positions().reshape(4, -1)[:, rt.wheel_dofs].astype(np.float64)
            if self._wheel_q is not None:
                self.wheel = np.angle(np.exp(1j * (q - self._wheel_q))) / (P.dt * P.div_ugv)   # joint angles wrap
            self._wheel_q = q
        self._fill_truth(rt, T, acc)

        c, r = rt.ctrl, self.refs
        if run_platform:  # angular velocity of the (rate-limited) reference, in the reference frame
            E = self._R_ref.T @ c.R_ref
            self._w_ref = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / dt_p
            self._R_ref = c.R_ref.copy()
        pr = r.platform
        pr.p, pr.v, pr.a, pr.R, pr.w = c.p_ref, c.v_ref, c.a_ref, c.R_ref, self._w_ref
        r.platform = pr
        q, ug = rt.quads, rt.ugv_cmd
        r.drone_p, r.drone_v, r.drone_yaw = q.p_ref, q.v_ref, q.yaw_ref
        r.ugv_goal, r.ugv_goal_vel, r.ugv_yaw = ug.goal[:, :2], ug.goal_vel, ug.goal[:, 2]
        r.ugv_velocity_mode = (ug.mode == "velocity").astype(np.int32)
        r.ugv_cmd_v, r.ugv_cmd_om = ug.cmd_v, ug.cmd_om

        core.step(self.truth, r, c.w_ext)

        cmd = core.cmd
        if rt.winch_auto:
            w = cmd.winch
            rt.cables.mode[:] = np.where(w.tension_mode > 0, "tension", "length")
            rt.cables.L_cmd[:], rt.cables.Ld_cmd[:], rt.cables.T_ff[:] = w.L_cmd, w.Ld_cmd, w.T_ff
            c.t_des, c.feasible = core.platform.t_des, bool(core.platform.td.feasible)
        q.f_cmd = cmd.rotor_f
        if ugv_due:
            vt = np.zeros((4, rt.art.max_dofs), np.float32)
            vt[:, rt.wheel_dofs] = cmd.wheel_w
            rt.art.set_dof_velocity_targets(vt, rt.art_idx)
