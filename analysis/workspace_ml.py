"""Wrench-feasible workspace of the Isaac model and a Random Forest surrogate of it (the thesis method: evaluate
feasibility as a field, then learn a surrogate so the design search becomes affordable).

    python3 analysis/workspace_ml.py     -> results/workspace_ml.npz, prints the numbers used in the trailer
Feasibility of a platform position p (level, formation fixed and centred on the origin): there are cable tensions
T_MIN <= t <= T_MAX with W(p) t = gravity wrench of platform + tool + one 2.2 kg block (linear programme).
The surrogate sees 25% of the grid and is scored on the other 75%.
"""

import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory as th  # noqa: E402

g = th.cfg["gripper"]
m_tot = th.cfg["platform"]["mass"] + g["wrist"]["mass"] + g["tool"]["mass"] + 2 * g["finger"]["mass"] + th.cfg["blocks"]["mass"]
w = np.array([0, 0, m_tot * th.G, 0, 0, 0])
A = th.anchors(*th.CROSSED)
xs = ys = np.arange(-1.5, 1.501, 0.125)
zs = np.arange(0.4, 3.401, 0.125)
P = np.array([(x, y, z) for x in xs for y in ys for z in zs])
t0 = time.time()
feas = np.zeros(len(P), bool)
tmax = np.full(len(P), np.nan)
for i, p in enumerate(P):
    ok, t = th.lp_feasible(th.W_at(A, p)[0], w)
    feas[i] = ok
    if ok:
        tmax[i] = t.max()
t_lp = time.time() - t0
rng = np.random.default_rng(0)
idx = rng.permutation(len(P))
n_tr = len(P) // 4
tr, te = idx[:n_tr], idx[n_tr:]
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(P[tr], feas[tr])
pred = rf.predict(P)
t_rf = time.time() - t0
t0 = time.time()
rf.predict(P)
t_pred = time.time() - t0
acc = float((pred[te] == feas[te]).mean())
print(f"payload wrench: {m_tot:.2f} kg; grid {len(P)} points; feasible {feas.mean() * 100:.1f}%")
print(f"LP: {t_lp:.1f} s ({t_lp / len(P) * 1e3:.2f} ms/point); RF trained on {n_tr} points; held-out accuracy {acc * 100:.2f}% "
      f"on {len(te)} points; RF predict whole grid {t_pred * 1e3:.0f} ms ({t_lp / t_pred:.0f}x faster)")
np.savez(os.path.join(th.OUT, "workspace_ml.npz"), P=P, feas=feas, pred=pred, tmax=tmax, train=tr, test=te, acc=acc, mass=m_tot,
         t_lp=t_lp, t_pred=t_pred)
