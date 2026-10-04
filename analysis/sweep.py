"""Summarise the E3 drone tilt-limit sweep (experiments/run_tilt_sweep.sh).
Steady-state window 15-22 s. Run: python3 analysis/sweep.py  -> table + results/tilt_sweep.png
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
e3 = json.load(open(os.path.join(RES, "theory.json")))["E3_pose"]

runs = []
for f in glob.glob(os.path.join(RES, "E3_tilt*.csv")):
    m = re.match(r"E3_tilt([\d.]+)(_aware)?(_hybrid)?\.csv", os.path.basename(f))
    if not m:
        continue
    d = np.genfromtxt(f, delimiter=",", names=True)
    if "sat0" not in d.dtype.names:
        continue  # older log format
    runs.append((float(m.group(1)), bool(m.group(2)), bool(m.group(3)), d))
runs.sort(key=lambda r: (r[2], r[1], r[0]))

print(f"E3 pose: dz={e3['dz']} rpy={e3['rpy']} deg, drones at 3.0 m; static theory: needs {e3['required_tilt']:.1f} deg drone tilt")
print(f"{'limit':>6} {'caps':>5} {'mode':>7} | {'tilt max':>8} {'T max':>6} {'T min':>6} {'sat %':>6} {'thrust max':>10} {'drone err':>9}"
      f" {'pos err':>8} {'rot err':>8} {'yaw std':>8}  verdict")
rows = []
for lim, aware, hyb, d in runs:
    w = (d["t"] >= 15) & (d["t"] <= 22)
    tilt = max(d[f"tilt{i}"][w].max() for i in range(4))
    T = np.vstack([d[f"T{i}"][w] for i in range(8)])
    sat = 100 * np.mean(np.vstack([d[f"sat{i}"][w] for i in range(4)]) > 0)
    thr = max(d[f"thrust{i}"][w].max() for i in range(4))
    derr = max(d[f"derr{i}"][w].max() for i in range(4))
    perr = 1000 * np.sqrt((d["x"] - d["xr"]) ** 2 + (d["y"] - d["yr"]) ** 2 + (d["z"] - d["zr"]) ** 2)[w].max()
    rerr = max(np.abs(d[a][w] - d[a + "r"][w]).max() for a in ("roll", "pitch", "yaw"))
    ystd = d["yaw"][w].std()
    # stability: no rotor saturation, drone tilt within its limit, cables within the winch rating
    stable = sat < 5.0 and tilt <= lim + 2.0 and T.max() <= 60.0
    rows.append(dict(limit=lim, aware=aware, hybrid=hyb, tilt_max=tilt, T_max=T.max(), T_min=T.min(), sat_pct=sat, thrust_max=thr,
                     drone_err=derr, pos_err_mm=perr, rot_err_deg=rerr, yaw_std=ystd, stable=bool(stable)))
    print(f"{lim:6.0f} {'yes' if aware else 'no':>5} {'hybrid' if hyb else 'length':>7} | {tilt:8.1f} {T.max():6.1f} {T.min():6.1f} {sat:6.1f} {thr:10.2f} {derr * 100:7.1f}cm"
          f" {perr:6.1f}mm {rerr:7.2f}° {ystd:7.2f}°  {'stable' if stable else 'RUNAWAY'}")
json.dump(rows, open(os.path.join(RES, "tilt_sweep.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for lim, aware, hyb, d in runs:
    ls = "-" if hyb else ("-." if aware else "--")
    lab = f"{lim:g}°" + (" caps" if aware else "") + (" hybrid" if hyb else "")
    ax[0].plot(d["t"], np.vstack([d[f"tilt{i}"] for i in range(4)]).max(0), ls, label=lab)
    ax[1].plot(d["t"], np.vstack([d[f"T{i}"] for i in range(4)]).max(0), ls, label=lab)
    ax[2].plot(d["t"], d["yaw"] - d["yawr"], ls, label=lab)
ax[0].axhline(e3["required_tilt"], c="k", lw=0.8, ls=":")
ax[0].set(title="max drone tilt (dotted: static requirement)", xlabel="t [s]", ylabel="deg")
ax[1].axhline(60, c="k", lw=0.8, ls=":")
ax[1].set(title="max drone-cable tension (dotted: winch limit)", xlabel="t [s]", ylabel="N")
ax[2].set(title="platform yaw error", xlabel="t [s]", ylabel="deg")
for a in ax:
    a.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(RES, "tilt_sweep.png"), dpi=120)
