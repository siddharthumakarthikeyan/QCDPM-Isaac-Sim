"""Isaac-free checks of the C++ core (core/, module cdpr_sim.cdpr_core). Run:  python3 tests/core_test.py

  1. model: structure matrix, tension distribution and actuator caps against the Python reference; the box QP against
     its optimality conditions; forward kinematics round trip
  2. closed loop in the C++ plant: lift + rotate, then a 22 N payload. With the true state the payload is announced
     (wrist load cell, as in the simulation); with the simulated sensors and estimators in the loop it is not, and
     the platform observer has to find it
  3. stability: eigenvalues of the linearised closed-loop map over one control period
  4. speed
"""

import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from cdpr_sim import cdpr_core as cc  # noqa: E402
from cdpr_sim.cdpr_control import (corner_points, drone_tension_caps, formation, structure_matrix,  # noqa: E402
                                   tension_distribution)
from cdpr_sim.core_bridge import params_from_cfg  # noqa: E402
from cdpr_sim.mathutil import rpy_to_rot  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
np.set_printoptions(suppress=True, linewidth=170, precision=4)
results = {}


def check(name, ok, detail=""):
    results[name] = bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


g_ = cfg["gripper"]
TOOL = np.array([0, 0, -(g_["wrist"]["mass"] + g_["tool"]["mass"] + 2 * g_["finger"]["mass"]) * cfg["sim"]["gravity"], 0, 0, 0])


def make_sim(use_truth, noise=True, freeze_ugv=False, admittance=None):
    P = params_from_cfg(cfg)
    P.use_truth = use_truth
    if use_truth and not admittance:          # as in the bridge: the load-cell admittance is for estimated anchors
        g = P.gains
        g.k_adm = 0.0
        P.gains = g
    s = P.sensors
    s.noise = noise
    P.sensors = s
    p0 = np.array(cfg["platform"]["start_position"], float)
    dr, ug = formation(cfg, p0[:2])
    sim = cc.Simulator(P, p0, np.vstack([dr, ug]), np.zeros(4))
    sim.freeze_ugv = freeze_ugv
    sim.w_known = TOOL.copy()
    sim.initialise()
    return sim, p0


# ------------------------------------------------------------------------------------------------ 1. model
print("1. model")
P = params_from_cfg(cfg)
b = corner_points(cfg["platform"]["side"])
rng = np.random.default_rng(0)
dr, ug = formation(cfg, np.zeros(2))
A = np.vstack([dr, ug])
err_W = err_t = 0.0
kkt = 0.0
for _ in range(300):
    p = np.array([0, 0, 1.3]) + rng.uniform(-0.5, 0.5, 3)
    R = rpy_to_rot(rng.uniform(-0.3, 0.3, 3))
    W, u, d = cc.structure_matrix(p, R, b, A)
    W_py, _, _ = structure_matrix(p, R, b, A)
    err_W = max(err_W, np.abs(W - W_py).max())
    w = np.r_[rng.uniform(-8, 8, 2), 29.4 + rng.uniform(-10, 30), rng.uniform(-1.5, 1.5, 3)]
    lo, hi = np.full(8, 5.0), np.full(8, 60.0)
    td = cc.tension_distribution(W, w, lo, hi, 15.0, np.full(8, 15.0), 2.5e-3, 0.0)
    t_py, feas_py = tension_distribution(W_py, w, 5.0, 60.0, 15.0, 2.5e-3)
    if td.n_active == 0 and feas_py:
        err_t = max(err_t, np.abs(td.t - t_py).max())
    # optimality of the bounded solution: the projected gradient must vanish
    H = W.T @ W + 2.5e-3 * np.eye(8)
    g = -W.T @ w - 2.5e-3 * 15.0
    grad = H @ td.t + g
    pg = np.where(td.t <= lo + 1e-9, np.minimum(grad, 0), np.where(td.t >= hi - 1e-9, np.maximum(grad, 0), grad))
    kkt = max(kkt, np.abs(pg).max())
check("structure matrix = Python reference", err_W < 1e-12, f"max diff {err_W:.1e}")
check("tension distribution = closed form when no bound is active", err_t < 1e-6, f"max diff {err_t:.1e} N")
check("box QP optimality (projected gradient)", kkt < 1e-7, f"max {kkt:.1e}")
W, u, d = cc.structure_matrix(np.array([0.4, 0.2, 1.3]), np.eye(3), b, A)
cap_py = drone_tension_caps(u[:4], P.drone.m, P.g, P.tension.tilt_cap, P.tension.thrust_cap)
cap = [cc.drone_tension_cap(u[i], P.drone.m, P.g, P.tension.tilt_cap, P.tension.thrust_cap) for i in range(4)]
check("drone tension caps = Python reference", np.allclose(cap, cap_py), f"{np.round(cap, 1)} N")
print("     ground-robot caps (slide / tip, 80% margin):", np.round([cc.ugv_tension_cap(u[i], P.ugv, P.g, 0.8) for i in range(4, 8)], 1), "N")
p_t, R_t = np.array([0.3, -0.2, 1.5]), rpy_to_rot([0.15, -0.1, 0.25])
_, _, d_t = cc.structure_matrix(p_t, R_t, b, A)
p_fk, R_fk, rms = cc.forward_kinematics(d_t, b, A, np.array([0, 0, 1.3]), np.eye(3))
check("forward kinematics round trip", np.abs(p_fk - p_t).max() < 1e-9 and np.abs(R_fk - R_t).max() < 1e-9, f"rms {rms:.1e} m")


# ------------------------------------------------------------------------------------------------ 2. closed loop
def run_scenario(use_truth, label):
    sim, p0 = make_sim(use_truth)
    dt, div = sim.P.dt, sim.P.div_platform
    log = []
    T_END, n = 16.0, 0
    p_prev = p0.copy()
    while sim.time < T_END - 1e-9:
        t = sim.time
        s = np.clip((t - 2.0) / 3.0, 0, 1)
        s = s * s * (3 - 2 * s)
        refs = sim.refs
        pr = refs.platform
        p_ref = p0 + np.array([0.0, 0.0, 0.3]) * s
        pr.v, pr.p = (p_ref - p_prev) / (dt * div), p_ref
        pr.R = rpy_to_rot([np.deg2rad(10) * s, 0.0, np.deg2rad(15) * s])
        p_prev = p_ref
        refs.platform = pr
        sim.refs = refs
        if t >= 9.0:
            load = np.array([0, 0, -22.0, 0, 0, 0])
            if use_truth:
                sim.w_known = TOOL + load
            else:
                sim.ext_wrench = load                              # a block the controller is not told about
        sim.run(div)
        e_p = sim.platform.p - p_ref
        e_r = np.rad2deg(np.linalg.norm(cc_log(pr.R.T @ sim.platform.R)))
        est = sim.core.est
        log.append((sim.time, np.linalg.norm(e_p), e_r, sim.T.min(), sim.T.max(), np.linalg.norm(est.p - sim.platform.p),
                    np.rad2deg(np.linalg.norm(cc_log(est.R.T @ sim.platform.R))), est.d_hat[2],
                    max(np.linalg.norm(est.drone[k].p - sim.drone[k].p) for k in range(4)),
                    max(np.linalg.norm(sim.drone[k].p - sim.refs.drone_p[k]) for k in range(4)), sim.traction_use.max()))
    log = np.array(log)
    print(f"  [{label}]   t   |e_p| mm  e_R deg  Tmin   Tmax  est mm est deg  d_hat_z  drone_est mm  drone_err mm")
    for row in log[::250]:
        print("        %5.1f %8.2f %8.3f %6.2f %6.2f %6.2f %7.3f %8.2f %10.2f %12.2f" %
              (row[0], row[1] * 1e3, row[2], row[3], row[4], row[5] * 1e3, row[6], row[7], row[8] * 1e3, row[9] * 1e3))
    return log, sim


def cc_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * v if th < 1e-7 else th / (2 * np.sin(th)) * v


print("2. closed loop (C++ plant)")
for use_truth, label in ((True, "true state"), (False, "sensors  ")):
    log, sim = run_scenario(use_truth, label)
    hold = log[(log[:, 0] > 7.0) & (log[:, 0] < 9.0)]
    load = log[log[:, 0] > 14.0]
    tag = "true state" if use_truth else "sensors in the loop"
    lim = 0.003 if use_truth else 0.004      # winch friction (1 N) leaves a slowly decaying mm-level error after a move
    check(f"{tag}: pose held after the move", hold[:, 1].max() < lim and hold[:, 2].max() < 0.75,
          f"{hold[:, 1].mean() * 1e3:.2f} mm mean, {hold[:, 1].max() * 1e3:.2f} mm max, {hold[:, 2].max():.3f} deg")
    check(f"{tag}: pose recovered under the 22 N payload", load[:, 1].max() < lim and load[:, 2].max() < 0.75,
          f"{load[:, 1].mean() * 1e3:.2f} mm mean, {load[:, 1].max() * 1e3:.2f} mm max; peak {log[log[:, 0] > 9][:, 1].max() * 1e3:.1f} mm")
    check(f"{tag}: cables taut, tensions bounded", log[:, 3].min() > 1.0 and log[:, 4].max() < 61.0,
          f"{log[:, 3].min():.1f} .. {log[:, 4].max():.1f} N")
    check(f"{tag}: drones on station", log[:, 9].max() < 0.05, f"max {log[:, 9].max() * 1e3:.1f} mm")
    if not use_truth:
        check("platform estimate (cable lengths + tensions + tags)", log[500:, 5].max() < 0.01,
              f"{log[500:, 5].mean() * 1e3:.2f} mm mean, {log[500:, 5].max() * 1e3:.2f} mm max, {log[500:, 6].max():.3f} deg max")
        check("payload estimated by the observer", abs(load[:, 7].mean() + 22.0) < 2.0, f"{load[:, 7].mean():.2f} N (true -22)")
        check("drone estimate (IMU + fix)", log[:, 8].max() < 0.02, f"max {log[:, 8].max() * 1e3:.2f} mm")

# ------------------------------------------------------------------------------------------------ 3. stability
print("3. stability of the closed loop (linearised map over one 20 ms control period, ground robots parked)")
sim, p0 = make_sim(True, noise=False, freeze_ugv=True, admittance=True)    # every loop closed, as deployed
sim.run(8000)
period = int(np.lcm.reduce([sim.P.div_drone, sim.P.div_platform, sim.P.div_ugv]))
A = sim.monodromy(period, 1e-6)
ev = np.linalg.eigvals(A)
rho = np.abs(ev).max()
Tp = period * sim.P.dt
with np.errstate(divide="ignore"):
    s_cont = np.log(ev[np.abs(ev) > 1e-9]) / Tp                          # equivalent continuous-time poles
slow = s_cont[np.argmax(s_cont.real)]
osc = s_cont[np.abs(s_cont.imag) > 1e-6]
zeta = -osc.real / np.abs(osc)
print(f"     {A.shape[0]} states; spectral radius {rho:.5f}; slowest mode {slow.real:.3f} 1/s (time constant {-1 / slow.real:.2f} s)")
k = np.argmin(zeta)
print(f"     least damped mode: {np.abs(osc[k]) / 2 / np.pi:.2f} Hz, damping ratio {zeta[k]:.3f}")
check("all eigenvalues inside the unit circle", rho < 1.0, f"rho = {rho:.5f}")
np.save(os.path.join(ROOT, "results", "core_monodromy_eigs.npy"), ev)

# ------------------------------------------------------------------------------------------------ 4. speed
print("4. speed")
sim, _ = make_sim(False)
t0 = time.perf_counter()
sim.run(20000)
wall = time.perf_counter() - t0
print(f"     20 s of the full loop (plant + sensors + estimators + controllers, 1 kHz) in {wall:.2f} s: "
      f"{20 / wall:.0f}x real time, {wall / 20000 * 1e6:.1f} us per step")

print("\n" + ("ALL PASS" if all(results.values()) else "FAILED: " + ", ".join(k for k, v in results.items() if not v)))
sys.exit(0 if all(results.values()) else 1)
