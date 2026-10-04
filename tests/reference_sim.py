"""Isaac-free reference simulation (UGVs held fixed) used to sanity-check models and to cross-validate Isaac.

Rigid bodies integrated with semi-implicit Euler at the same dt as PhysX. Run:  python3 tests/reference_sim.py
"""

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cdpr_sim.cdpr_control import PlatformController, corner_points, formation  # noqa: E402
from cdpr_sim.mathutil import quat_to_rot, rot_to_quat, rpy_to_rot  # noqa: E402
from cdpr_sim.models import CableWinchModel, QuadrotorModel  # noqa: E402


def integrate(p, q, v, om, F, tau, m, I_body, dt):
    R = quat_to_rot(q)
    I_w = R @ I_body @ R.T
    v = v + dt * F / m
    om = om + dt * np.linalg.solve(I_w, tau - np.cross(om, I_w @ om))
    p = p + dt * v
    th = np.linalg.norm(om) * dt
    if th > 0:
        a = om / np.linalg.norm(om)
        dq = np.r_[np.cos(th / 2), np.sin(th / 2) * a]
        w0, x0, y0, z0 = dq
        w1, x1, y1, z1 = q
        q = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                      w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
        q /= np.linalg.norm(q)
    return p, q, v, om


def main(T_end=12.0, cfg=None, verbose=True):
    cfg = cfg or yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "cdpr.yaml")))
    dt, g = cfg["sim"]["physics_dt"], cfg["sim"]["gravity"]
    b = corner_points(cfg["platform"]["side"])
    p0 = np.array(cfg["platform"]["start_position"])
    d_anch, g_anch = formation(cfg, p0[:2])
    att = np.array(cfg["drone"]["attach_offset"])

    cables = CableWinchModel(cfg["cable"], cfg["winch"], 8)
    quads = QuadrotorModel(cfg["drone"], 4, g)
    ctrl = PlatformController(cfg, b, g)

    # state
    P = dict(p=p0.copy(), q=np.array([1.0, 0, 0, 0]), v=np.zeros(3), om=np.zeros(3))
    m_p = cfg["platform"]["mass"]
    I_p = ctrl.I
    D = [dict(p=d_anch[k] - att, q=np.array([1.0, 0, 0, 0]), v=np.zeros(3), om=np.zeros(3)) for k in range(4)]
    quads.p_ref = d_anch - att

    t_des, dist = ctrl.solve(np.vstack([d_anch, g_anch]))
    cables.L[:] = cables.unstretched_for(dist, t_des)
    cables.L_cmd[:] = cables.L
    cables.T_ff[:] = t_des
    quads.init_hover(t_des[:4] * 0.9)
    print("initial tensions", np.round(t_des, 2), "feasible", ctrl.feasible)

    log = []
    steps = int(T_end / dt)
    for i in range(steps):
        t = i * dt
        # reference: hold, then 0.3 m lift (+ 10 deg roll, 15 deg yaw for the crossed layout) from t=3 s
        s = np.clip((t - 3.0) / 3.0, 0, 1)
        s = s * s * (3 - 2 * s)
        ctrl.p_ref = p0 + np.array([0.0, 0.0, 0.3]) * s
        crossed = abs(cfg["layout"]["drone_cross_angle_deg"]) > 1e-6  # straight cables can only hold it level
        ctrl.R_ref = rpy_to_rot([np.deg2rad(10) * s, 0.0, np.deg2rad(15) * s]) if crossed else np.eye(3)

        Rp = quat_to_rot(P["q"])
        pa = P["p"] + b @ Rp.T
        va = P["v"] + np.cross(P["om"], b @ Rp.T)
        Rd = np.array([quat_to_rot(d["q"]) for d in D])
        pb_d = np.array([d["p"] for d in D]) + np.einsum("nij,j->ni", Rd, att)
        vb_d = np.array([d["v"] for d in D]) + np.cross(np.array([d["om"] for d in D]), np.einsum("nij,j->ni", Rd, att))
        pb = np.vstack([pb_d, g_anch])
        vb = np.vstack([vb_d, np.zeros((4, 3))])
        T, u, dist = cables.tension(pa, va, pb, vb)

        # platform-level: IK lengths + tension feed-forward using measured anchors
        t_des, d_ref = ctrl.solve(pb)
        cables.L_cmd[:] = cables.unstretched_for(d_ref, t_des)
        cables.T_ff[:] = t_des
        cables.step(dt)

        # platform dynamics
        Fc = T[:, None] * u
        F_p = Fc.sum(0) - m_p * g * np.array([0, 0, 1.0])
        tau_p = np.cross(pa - P["p"], Fc).sum(0)
        P["p"], P["q"], P["v"], P["om"] = integrate(P["p"], P["q"], P["v"], P["om"], F_p, tau_p, m_p, I_p, dt)

        # drones
        F_on_drone = -Fc[:4]
        pos = np.array([d["p"] for d in D])
        quat = np.array([d["q"] for d in D])
        vel = np.array([d["v"] for d in D])
        omw = np.array([d["om"] for d in D])
        quads.control(pos, quat, vel, omw, F_on_drone, dt)
        F_r, tau_r = quads.actuate(quat, vel, dt)
        for k, d in enumerate(D):
            Ftot = F_r[k] + F_on_drone[k] - quads.m * g * np.array([0, 0, 1.0])
            ttot = tau_r[k] + np.cross(pb_d[k] - d["p"], F_on_drone[k])
            d["p"], d["q"], d["v"], d["om"] = integrate(d["p"], d["q"], d["v"], d["om"], Ftot, ttot, quads.m, quads.J, dt)

        if i % 100 == 0:
            Rp = quat_to_rot(P["q"])
            e_rot = np.rad2deg(np.linalg.norm(rot_to_quat(ctrl.R_ref.T @ Rp)[1:]) * 2)
            log.append((t, *(P["p"] - ctrl.p_ref), e_rot, T.min(), T.max(),
                        max(np.linalg.norm(pos - quads.p_ref, axis=1)), quads.f_cmd.max()))
    log = np.array(log)
    np.set_printoptions(suppress=True, linewidth=160)
    print("   t     ex     ey     ez   erot   Tmin   Tmax  drone_err  rotor_fmax")
    for row in log[::10]:
        print("%5.1f %6.3f %6.3f %6.3f %6.2f %6.2f %6.2f %8.3f %8.2f" % tuple(row))
    final = log[-1]
    ok = np.all(np.abs(final[1:4]) < 0.02) and final[4] < 1.0 and np.all(np.isfinite(log))
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
