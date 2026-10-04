"""Analytical checks of the three layout claims, with numbers the Isaac experiments must reproduce.

Claim 1  Crossing gives full 6-DOF control; straight (corner-to-adjacent-robot) cables cannot control yaw and
         cannot hold any tilt.
Claim 2  Because the anchors are robots, the crossing angle phi can be changed live; yaw/orientation authority
         grows continuously with phi (m_z = r rho sin(phi) / L) while the level pose stays feasible throughout.
Claim 3  Drone thrust and tilt limits act as direction-dependent upper bounds on drone-cable tension and shrink
         the wrench-feasible workspace.

Run: python3 analysis/theory.py      -> prints the derivations' numbers, writes results/theory.json + figures
"""

import itertools
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cdpr_sim.cdpr_control import corner_points, drone_tension_caps, structure_matrix, tension_distribution  # noqa: E402
from cdpr_sim.mathutil import rpy_to_rot  # noqa: E402

OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "cdpr.yaml")))
G = cfg["sim"]["gravity"]
L_ = cfg["layout"]
B = corner_points(cfg["platform"]["side"])
H = cfg["platform"]["side"] / 2
R_C = H * np.sqrt(2)                      # horizontal radius of a platform corner
P0 = np.array(cfg["platform"]["start_position"], float)
W_G = np.array([0, 0, cfg["platform"]["mass"] * G, 0, 0, 0])
T_MIN, T_MAX, T_REF = cfg["winch"]["tension_min"], cfg["winch"]["tension_max"], cfg["winch"]["tension_ref"]
M_D = cfg["drone"]["mass"]
F_MAX = 4 * cfg["drone"]["rotor_max_thrust"]
AZ = np.deg2rad(45 + 90 * np.arange(4))
RHO = np.r_[[L_["drone_radius"]] * 4, [L_["ugv_radius"]] * 4]
STRAIGHT, CROSSED = (0.0, 0.0), (np.pi / 2, -np.pi / 2)


def anchors(phi_top, phi_bot, center=np.zeros(2), drone_alt=None):
    z_d = L_["drone_altitude"] if drone_alt is None else drone_alt
    top = np.c_[L_["drone_radius"] * np.cos(AZ + phi_top), L_["drone_radius"] * np.sin(AZ + phi_top), np.full(4, z_d)]
    bot = np.c_[L_["ugv_radius"] * np.cos(AZ + phi_bot), L_["ugv_radius"] * np.sin(AZ + phi_bot), np.full(4, cfg["ugv"]["attach_height"])]
    A = np.vstack([top, bot])
    A[:, :2] += center
    return A


def W_at(A, p=P0, rpy=(0, 0, 0)):
    return structure_matrix(np.asarray(p, float), rpy_to_rot(np.deg2rad(rpy)), B, A)


def lp_feasible(W, w, t_max=T_MAX):
    """min max_i t_i  s.t.  W t = w, T_MIN <= t <= t_max. Returns (feasible, t)."""
    n = W.shape[1]
    ub = np.broadcast_to(np.asarray(t_max, float), (n,))
    res = linprog(np.r_[np.zeros(n), 1], A_ub=np.c_[np.eye(n), -np.ones(n)], b_ub=np.zeros(n),
                  A_eq=np.c_[W, np.zeros(6)], b_eq=w, bounds=[(T_MIN, u) for u in ub] + [(0, None)])
    return res.status == 0, (res.x[:n] if res.status == 0 else None)


def farkas(W, w, t_min=T_MIN):
    """Certificate y with W^T y >= 0 and y.(w - W t_min) < 0  =>  no t >= t_min solves W t = w."""
    wp = w - W @ np.full(W.shape[1], t_min)
    res = linprog(np.zeros(6), A_ub=np.r_[-W.T, wp[None]], b_ub=np.r_[np.zeros(W.shape[1]), -1.0], bounds=[(-1e3, 1e3)] * 6)
    return None if res.status != 0 else res.x / np.abs(res.x).max()


def max_angle(A, axis, lim=30.0, t_max_fn=None):
    """Largest |angle| about one axis (0 roll, 1 pitch, 2 yaw), both signs, that stays wrench-feasible."""
    def ok(a):
        for sgn in (1, -1):
            rpy = np.zeros(3)
            rpy[axis] = sgn * a
            W, u, _ = W_at(A, rpy=rpy)
            tm = T_MAX if t_max_fn is None else t_max_fn(u)
            if not lp_feasible(W, W_G, tm)[0]:
                return False
        return True
    if not ok(0.05):
        return 0.0
    lo, hi = 0.05, lim
    if ok(hi):
        return hi
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def stiffness(A, t, eps=1e-6):
    """6x6 static stiffness of the platform with winch length servos holding L (anchors fixed).
    Each cable is the cable EA/L in series with the winch servo m_eq wn^2; includes the geometric (pretension) term."""
    w_cfg = cfg["winch"]
    k_servo = w_cfg["inertia"] / w_cfg["drum_radius"] ** 2 * w_cfg["wn"] ** 2
    W0, u0, d0 = W_at(A)
    k = 1.0 / (d0 / cfg["cable"]["EA"] + 1.0 / k_servo)

    def wrench(dx):
        R = rpy_to_rot(dx[3:])  # small angles: rpy ~ rotation vector
        W, u, d = structure_matrix(P0 + dx[:3], R, B, A)
        return W @ (t + k * (d - d0))
    K = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = eps
        K[:, j] = -(wrench(e) - wrench(-e)) / (2 * eps)
    return K


def drone_tension_bounds(u, theta_max_deg=None, f_max=F_MAX):
    """Per-cable upper tension bounds implied by the drones (cables 0-3); UGV cables keep the winch bound.
    Drone must produce f = m g e3 + t u.  Tilt:  t (|u_xy| - u_z tan th) <= m g tan th.
    Thrust: |f| <= F  <=>  t <= -m g u_z + sqrt((m g u_z)^2 - (m g)^2 + F^2)."""
    tb = np.full(8, T_MAX, float)
    th = np.inf if theta_max_deg is None else np.deg2rad(theta_max_deg)
    tb[:4] = np.minimum(T_MAX, drone_tension_caps(u[:4], M_D, G, th if np.isfinite(th) else np.pi / 2 - 1e-6, f_max))
    return tb


def drone_tilt(t, u):
    mg = M_D * G
    return np.rad2deg(np.arctan2(t[:4] * np.linalg.norm(u[:4, :2], axis=1), mg + t[:4] * u[:4, 2]))


def main():
    res = {}
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    print("=" * 100)
    print("CLAIM 1  straight vs crossed")
    print("-" * 100)
    print("Yaw moment arm of cable i at the level pose (platform centred):  m_z,i = r * rho_i * sin(phi_i) / L_i")
    rows = []
    for phi in (0, 30, 60, 90):
        A = anchors(np.deg2rad(phi), -np.deg2rad(phi))
        W, _, d = W_at(A)
        closed = R_C * RHO * np.sin(np.deg2rad(phi) * np.r_[[1] * 4, [-1] * 4]) / d
        err = np.abs(W[5] - closed).max()
        rows.append(dict(phi=phi, top=float(closed[0]), bottom=float(closed[4]), max_err=float(err)))
        print(f"  phi={phi:3d} deg  m_z top={closed[0]:+.4f} m  bottom={closed[4]:+.4f} m   |numeric-closed| = {err:.1e}")
    res["yaw_arm"] = rows

    out1 = {}
    for name, (pt, pb) in (("straight", STRAIGHT), ("crossed", CROSSED)):
        A = anchors(pt, pb)
        W, u, _ = W_at(A)
        sv = np.linalg.svd(W, compute_uv=False)
        rank = int(np.linalg.matrix_rank(W, 1e-8))
        # force closure: a strictly positive null vector (t >= 1, W t = 0)
        closure = linprog(np.zeros(8), A_eq=W, b_eq=np.zeros(6), bounds=[(1, None)] * 8).status == 0
        ok, t = lp_feasible(W, W_G)
        _, t_ctrl_feas = tension_distribution(W, W_G, T_MIN, T_MAX, T_REF)
        t_ctrl, _ = tension_distribution(W, W_G, T_MIN, T_MAX, T_REF)
        K = stiffness(A, t_ctrl)
        tau = np.r_[0, 0, 0, 0, 0, 0.3]
        dx = np.linalg.lstsq(K, tau, rcond=None)[0]
        mx = [max_angle(A, ax) for ax in range(3)]
        out1[name] = dict(rank=rank, sv=sv.tolist(), closure=bool(closure), level_feasible=bool(ok),
                          max_roll=mx[0], max_pitch=mx[1], max_yaw=mx[2], K_yaw=float(K[5, 5]), K_diag=np.diag(K).tolist(),
                          yaw_deflection_deg_at_0p3Nm=float(np.rad2deg(dx[5])), t_level=t_ctrl.tolist())
        print(f"  {name:8s} rank W = {rank}  singular values {sv}")
        print(f"           pretension mode (t>0 with W t=0): {closure}  -> fully constrained (closure): {closure and rank == 6}"
              f"   level pose feasible under gravity: {ok}")
        print(f"           max feasible |roll| {mx[0]:.2f} deg  |pitch| {mx[1]:.2f} deg  |yaw| {mx[2]:.2f} deg  (search limit 30)")
        print(f"           yaw stiffness K_zz = {K[5, 5]:.3f} N m/rad  -> static yaw under 0.3 N m: {np.rad2deg(dx[5]):.2f} deg")
    for ang in (1.0, 2.0, 5.0):
        y = farkas(W_at(anchors(*STRAIGHT), rpy=(ang, 0, 0))[0], W_G)
        print(f"  straight, roll {ang} deg: Farkas certificate y = {y}" if y is not None else f"  straight roll {ang}: feasible")
        out1.setdefault("farkas", {})[str(ang)] = None if y is None else y.tolist()
    res["claim1"] = out1

    print("=" * 100)
    print("CLAIM 2  live reconfiguration: sweep the crossing angle (top +phi, bottom -phi)")
    print("-" * 100)
    sweep = []
    for phi in range(0, 91, 5):
        A = anchors(np.deg2rad(phi), -np.deg2rad(phi))
        W, _, _ = W_at(A)
        sv = np.linalg.svd(W, compute_uv=False)
        ok, t = lp_feasible(W, W_G)
        roll, yaw = max_angle(A, 0), max_angle(A, 2)
        sweep.append(dict(phi=phi, sigma_min=float(sv[-1]), level_feasible=bool(ok), t_level_max=float(t.max()) if ok else None,
                          max_roll=roll, max_yaw=yaw))
        print(f"  phi={phi:2d}: sigma_min={sv[-1]:.4f}  level feasible={ok}  max T(level)={t.max() if ok else float('nan'):5.1f} N"
              f"  max|roll|={roll:5.2f}  max|yaw|={yaw:5.2f} deg")
    res["claim2"] = sweep

    print("=" * 100)
    print("CLAIM 3  drone thrust/tilt limits as tension bounds (crossed layout)")
    print("-" * 100)
    print("  tilt cap  t <= m g tan(th) / (cos(beta) - sin(beta) tan(th)),  active only if beta < 90 - th (beta = cable elevation)")
    for beta in (40, 45, 50, 52, 55, 60):
        caps = []
        for th in (25, 35, 45):
            bb, tt = np.deg2rad(beta), np.tan(np.deg2rad(th))
            den = np.cos(bb) - np.sin(bb) * tt
            caps.append(M_D * G * tt / den if den > 1e-9 else np.inf)
        print(f"    beta={beta} deg: cap @25={caps[0]:7.1f} N  @35={caps[1]:7.1f} N  @45={caps[2]:7.1f} N")
    grid = list(itertools.product((-0.3, 0.0, 0.3), (-15, -7.5, 0, 7.5, 15), (-15, -7.5, 0, 7.5, 15), (-20, -10, 0, 10, 20)))
    frac = []
    for z_d in (3.5, 3.0, 2.75):
        A = anchors(*CROSSED, drone_alt=z_d)
        W0, u0, _ = W_at(A)
        beta = np.rad2deg(np.arcsin(u0[0, 2]))
        line = []
        for name, fmax, th in [("winch", None, None), ("thrust", F_MAX, None), ("tilt45", F_MAX, 45), ("tilt35", F_MAX, 35), ("tilt25", F_MAX, 25)]:
            n_ok = 0
            for dz, r, p, y in grid:
                W, u, _ = W_at(A, P0 + [0, 0, dz], (r, p, y))
                tm = T_MAX if fmax is None else drone_tension_bounds(u, th, fmax)
                n_ok += lp_feasible(W, W_G, tm)[0]
            frac.append(dict(drone_alt=z_d, beta=float(beta), case=name, feasible_fraction=n_ok / len(grid)))
            line.append(f"{name} {100 * n_ok / len(grid):5.1f}%")
        print(f"  drone altitude {z_d} m (drone-cable elevation {beta:.0f} deg at level): " + "  ".join(line))
    res["claim3_workspace"] = frac

    # experiment pose for E3 (drone altitude 3.0 m): the controller's own tensions need a drone tilt between 35 and 45 deg
    A = anchors(*CROSSED, drone_alt=3.0)
    best = None
    for dz, r, p, y in itertools.product((0.0, 0.2, 0.3), np.arange(-14, 15, 2), np.arange(-14, 15, 2), (-15, 0, 15)):
        W, u, _ = W_at(A, P0 + [0, 0, dz], (r, p, y))
        t, feas = tension_distribution(W, W_G, T_MIN, T_MAX, T_REF)
        if not feas:
            continue
        th = drone_tilt(t, u).max()
        thrust = np.linalg.norm(M_D * G * np.array([0, 0, 1]) + t[:4, None] * u[:4], axis=1).max()
        if 38.0 <= th <= 42.0 and thrust < 0.8 * F_MAX and (best is None or abs(r) + abs(p) < abs(best[1]) + abs(best[2])):
            best = (dz, float(r), float(p), float(y), float(th), float(thrust))
    if best:
        print(f"  E3 test pose (drone altitude 3.0 m): dz={best[0]} roll={best[1]} pitch={best[2]} yaw={best[3]} -> controller "
              f"needs drone tilt {best[4]:.1f} deg (thrust {best[5]:.1f} N): predicted FAIL with 35 deg limit, PASS with 45 deg")
    res["E3_pose"] = None if best is None else dict(drone_alt=3.0, dz=best[0], rpy=best[1:4], required_tilt=best[4], thrust=best[5])

    json.dump(res, open(os.path.join(OUT, "theory.json"), "w"), indent=1)
    plot(res)
    print(f"\nwrote {OUT}/theory.json and figures")


def plot(res):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    phis = np.linspace(0, 90, 91)
    A90 = anchors(*CROSSED)
    _, _, d = W_at(A90)
    ax[0].plot(phis, R_C * RHO[0] * np.sin(np.deg2rad(phis)) / d[0], label="top cable (closed form)")
    ax[0].plot(phis, -R_C * RHO[4] * np.sin(np.deg2rad(phis)) / d[4], label="bottom cable (closed form)")
    ax[0].scatter([r["phi"] for r in res["yaw_arm"]], [r["top"] for r in res["yaw_arm"]], c="k", zorder=3, label="numeric W")
    ax[0].scatter([r["phi"] for r in res["yaw_arm"]], [r["bottom"] for r in res["yaw_arm"]], c="k", zorder=3)
    ax[0].axhline(0, c="gray", lw=0.5)
    ax[0].set(xlabel="crossing angle phi [deg]", ylabel="yaw moment arm m_z [m]", title="Claim 1/2: m_z = r rho sin(phi) / L")
    ax[0].legend(fontsize=8)
    sw = res["claim2"]
    ax[1].plot([s["phi"] for s in sw], [s["max_roll"] for s in sw], "o-", label="max |roll|")
    ax[1].plot([s["phi"] for s in sw], [s["max_yaw"] for s in sw], "s-", label="max |yaw|")
    ax[1].set(xlabel="crossing angle phi [deg]", ylabel="feasible angle [deg] (limit 30)", title="Claim 2: orientation workspace vs phi")
    ax[1].legend(fontsize=8)
    fr = res["claim3_workspace"]
    cases = ["winch", "thrust", "tilt45", "tilt35", "tilt25"]
    alts = sorted({f["drone_alt"] for f in fr}, reverse=True)
    wdt = 0.8 / len(alts)
    for i, z in enumerate(alts):
        vals = [100 * next(f["feasible_fraction"] for f in fr if f["drone_alt"] == z and f["case"] == c) for c in cases]
        beta = next(f["beta"] for f in fr if f["drone_alt"] == z)
        ax[2].bar(np.arange(len(cases)) + i * wdt, vals, wdt, label=f"drones at {z} m (beta {beta:.0f} deg)")
    ax[2].set_xticks(np.arange(len(cases)) + wdt, ["winch only", "+thrust", "+tilt 45", "+tilt 35", "+tilt 25"], fontsize=8)
    ax[2].set(ylabel="feasible poses [%]", title="Claim 3: drone limits vs cable elevation")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "theory.png"), dpi=130)


if __name__ == "__main__":
    main()
