"""Compare the Isaac experiment logs (results/*.csv) with the analytical predictions (results/theory.json).
Run after experiments/run_all.sh:  python3 analysis/compare.py   -> results/comparison.json, results/experiments.png
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
theory = json.load(open(os.path.join(RES, "theory.json")))


def load(name):
    p = os.path.join(RES, f"{name}.csv")
    if not os.path.exists(p):
        return None
    d = np.genfromtxt(p, delimiter=",", names=True)
    return d


def win(d, col, t0, t1, fn=np.mean):
    m = (d["t"] >= t0) & (d["t"] <= t1)
    return float(fn(d[col][m]))


def tens(d, t0, t1):
    m = (d["t"] >= t0) & (d["t"] <= t1)
    T = np.vstack([d[f"T{i}"][m] for i in range(8)])
    return float(T.min()), float(T.max())


def wrap(a):
    return (a + 180) % 360 - 180


out = {}
D = {k: load(k) for k in ("E1_straight", "E1_crossed", "E2_reconfig", "E3_tilt35", "E3_tilt45")}

print("=" * 96)
print("CLAIM 1   prediction vs Isaac")
for k, name in (("E1_straight", "straight"), ("E1_crossed", "crossed")):
    d = D[k]
    if d is None:
        continue
    th = theory["claim1"][name]
    yaw_def = win(d, "yaw", 6.5, 8.0) - win(d, "yaw", 2.0, 3.0)
    roll = win(d, "roll", 19.0, 22.0)
    tmin, tmax = tens(d, 19.0, 22.0)
    out[k] = dict(yaw_defl_pred=th["yaw_deflection_deg_at_0p3Nm"], yaw_defl_sim=yaw_def, roll_cmd=10.0, roll_sim=roll,
                  roll_max_feasible_pred=th["max_roll"], tension_min_sim=tmin, tension_max_sim=tmax)
    print(f"  {name:8s} yaw under 0.3 N m: predicted {th['yaw_deflection_deg_at_0p3Nm']:.2f} deg   Isaac {yaw_def:.2f} deg")
    print(f"           10 deg roll command: predicted max feasible {th['max_roll']:.1f} deg   Isaac reached {roll:.2f} deg"
          f"   cable tensions [{tmin:.1f}, {tmax:.1f}] N")

d = D["E2_reconfig"]
if d is not None:
    print("CLAIM 2   live reconfiguration")
    roll_straight = win(d, "roll", 7.0, 9.0)
    m = (d["t"] >= 10) & (d["t"] <= 34)  # level hold + whole morph
    T = np.vstack([d[f"T{i}"][m] for i in range(8)])
    perr = np.sqrt((d["x"] - d["xr"]) ** 2 + (d["y"] - d["yr"]) ** 2 + (d["z"] - d["zr"]) ** 2)[m]
    tilt_morph = np.max(np.abs(np.c_[d["roll"][m], d["pitch"][m]]))
    roll_end, yaw_end = win(d, "roll", 43, 46), win(d, "yaw", 43, 46)
    out["E2_reconfig"] = dict(roll_while_straight=roll_straight, morph_tension_min=float(T.min()), morph_tension_max=float(T.max()),
                              morph_pos_err_max_mm=float(1000 * perr.max()), morph_tilt_max=float(tilt_morph),
                              roll_after=roll_end, yaw_after=yaw_end, cross_final=[float(d["cross_top"][-1]), float(d["cross_bot"][-1])])
    print(f"  straight phase, 10 deg roll command -> {roll_straight:.2f} deg (theory: <= {theory['claim1']['straight']['max_roll']:.1f})")
    print(f"  morph 0 -> +-90 deg: tensions stayed in [{T.min():.1f}, {T.max():.1f}] N, platform position error <= {1000 * perr.max():.1f} mm,"
          f" |tilt| <= {tilt_morph:.2f} deg")
    print(f"  after morph, command roll 10 / yaw 15 -> roll {roll_end:.2f} deg, yaw {yaw_end:.2f} deg")

print("CLAIM 3   drone tilt limit")
e3 = theory["E3_pose"]
for k in ("E3_tilt35", "E3_tilt45"):
    d = D[k]
    if d is None:
        continue
    t0, t1 = 18.0, 22.0
    ep = np.sqrt(win(d, "x", t0, t1) ** 2 + win(d, "y", t0, t1) ** 2 + (win(d, "z", t0, t1) - win(d, "zr", t0, t1)) ** 2)
    erpy = [win(d, a, t0, t1) - v for a, v in zip(("roll", "pitch", "yaw"), e3["rpy"])]
    tilt = max(win(d, f"tilt{i}", t0, t1, np.max) for i in range(4))
    derr = max(win(d, f"derr{i}", t0, t1, np.max) for i in range(4))
    out[k] = dict(required_tilt_pred=e3["required_tilt"], drone_tilt_max=tilt, drone_pos_err_max=derr, platform_pos_err=ep,
                  platform_rpy_err=erpy)
    print(f"  {k}: needs {e3['required_tilt']:.1f} deg (theory)  max drone tilt {tilt:.1f} deg  drone pos err {derr * 100:.1f} cm"
          f"  platform err {ep * 1000:.1f} mm, rpy err {np.round(erpy, 2)} deg")
json.dump(out, open(os.path.join(RES, "comparison.json"), "w"), indent=1)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for k, c in (("E1_straight", "C3"), ("E1_crossed", "C0")):
    d = D[k]
    if d is None:
        continue
    lab = k.split("_")[1]
    ax[0, 0].plot(d["t"], d["yaw"] - win(d, "yaw", 2, 3), c, label=f"{lab}")
    ax[0, 1].plot(d["t"], d["roll"], c, label=f"{lab}")
    T = np.vstack([d[f"T{i}"] for i in range(8)])
    ax[1, 0].fill_between(d["t"], T.min(0), T.max(0), color=c, alpha=0.3, label=f"{lab} tension range")
for k, name in (("E1_straight", "straight"), ("E1_crossed", "crossed")):
    ax[0, 0].axhline(theory["claim1"][name]["yaw_deflection_deg_at_0p3Nm"], ls="--", c="C3" if name == "straight" else "C0", lw=1)
ax[0, 0].axvspan(3, 8, color="gray", alpha=0.15)
ax[0, 0].set(title="E1: yaw under 0.3 N m (shaded); dashed = theory", xlabel="t [s]", ylabel="yaw change [deg]")
ax[0, 1].axhline(10, c="k", ls=":", lw=1)
ax[0, 1].axhline(theory["claim1"]["straight"]["max_roll"], c="C3", ls="--", lw=1)
ax[0, 1].set(title="E1: 10 deg roll command at t=10 s; dashed = straight limit", xlabel="t [s]", ylabel="roll [deg]")
ax[1, 0].set(title="E1: cable tension range", xlabel="t [s]", ylabel="T [N]")
for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.legend(fontsize=8)
d = D["E2_reconfig"]
if d is not None:
    ax[0, 2].plot(d["t"], d["cross_top"], "k--", label="crossing angle (top)")
    ax[0, 2].plot(d["t"], d["roll"], "C1", label="roll")
    ax[0, 2].plot(d["t"], d["yaw"], "C2", label="yaw")
    ax[0, 2].plot(d["t"], d["rollr"], "C1:", lw=1)
    ax[0, 2].plot(d["t"], d["yawr"], "C2:", lw=1)
    T = np.vstack([d[f"T{i}"] for i in range(8)])
    a2 = ax[0, 2].twinx()
    a2.fill_between(d["t"], T.min(0), T.max(0), color="gray", alpha=0.2)
    a2.set_ylabel("tension range [N]")
    ax[0, 2].set(title="E2: straight -> crossed live (dotted = command)", xlabel="t [s]", ylabel="deg")
    ax[0, 2].legend(fontsize=8, loc="center left")
for k, c in (("E3_tilt35", "C3"), ("E3_tilt45", "C0")):
    d = D[k]
    if d is None:
        continue
    tilt = np.vstack([d[f"tilt{i}"] for i in range(4)]).max(0)
    ax[1, 1].plot(d["t"], tilt, c, label=f"{k[3:]} limit: max drone tilt")
    ax[1, 2].plot(d["t"], 1000 * np.sqrt((d["x"] - d["xr"]) ** 2 + (d["y"] - d["yr"]) ** 2 + (d["z"] - d["zr"]) ** 2), c, label=k[3:])
ax[1, 1].axhline(theory["E3_pose"]["required_tilt"], c="k", ls="--", lw=1, label="theory: required")
ax[1, 1].set(title="E3: drone tilt (drones at 3.0 m)", xlabel="t [s]", ylabel="deg")
ax[1, 2].set(title="E3: platform position error", xlabel="t [s]", ylabel="mm")
ax[1, 1].legend(fontsize=8)
ax[1, 2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "experiments.png"), dpi=120)
print("wrote results/comparison.json, results/experiments.png")
