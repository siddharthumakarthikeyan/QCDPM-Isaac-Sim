"""Sampling-based 3D path planning for the platform, following the thesis pipeline (plan for the platform, prune,
smooth only the bends with Bezier arcs, then hand the path to the formation through inverse kinematics):

    rrt_star -> prune -> bezier_smooth -> resample (constant speed with smooth start/stop)

Obstacles are axis-aligned boxes (centre, size). A platform position is valid when the whole team clears them:
the platform with its hanging tool, the four ground robots (which keep their formation around the platform) and
all eight cables. `Team` builds that check from the config; plain `inflate` checks the platform alone.
"""

import numpy as np

from .cdpr_control import corner_points, formation, hold_table, lazy_center


def _seg_hits_box(a, b, lo, hi):
    """Segment a-b against an axis-aligned box (slab test)."""
    d = b - a
    t0, t1 = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if a[i] < lo[i] or a[i] > hi[i]:
                return False
        else:
            ta, tb = (lo[i] - a[i]) / d[i], (hi[i] - a[i]) / d[i]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
            if t0 > t1:
                return False
    return True


class Team:
    """Collision model of the whole system for a platform position p (level platform, nominal formation)."""

    def __init__(self, cfg, boxes, robot_radius=0.42, cable_margin=0.12):
        self.cfg = cfg
        self.boxes = [(np.asarray(c, float) - np.asarray(s, float) / 2, np.asarray(c, float) + np.asarray(s, float) / 2) for c, s in boxes]
        self.body = inflate(boxes)
        self.rr, self.cm = robot_radius, cable_margin
        self.corners = corner_points(cfg["platform"]["side"])

    def anchors(self, p, c=None):
        drones, ugvs = formation(self.cfg, p[:2] if c is None else c)
        return np.vstack([drones, ugvs])

    def __call__(self, p, c=None):
        """c: formation centre (xy); default the platform itself (formation centred on it)."""
        if not free(p, self.body):
            return False
        A = self.anchors(p, c)
        for lo, hi in self.boxes:
            for g in A[4:]:                              # ground robots: disc against the box footprint
                dx = max(lo[0] - g[0], 0.0, g[0] - hi[0])
                dy = max(lo[1] - g[1], 0.0, g[1] - hi[1])
                if dx * dx + dy * dy < self.rr**2:
                    return False
            for k in range(8):                           # cables: platform corner -> fairlead
                if _seg_hits_box(p + self.corners[k], A[k], lo - self.cm, hi + self.cm):
                    return False
        return True


def inflate(boxes, xy=0.32, below=0.2, above=0.72):
    """(lo, hi) corners of each box grown by the platform half-width and the tool hanging below the platform."""
    out = []
    for c, s in boxes:
        c, s = np.asarray(c, float), np.asarray(s, float)
        out.append((c - s / 2 - [xy, xy, below], c + s / 2 + [xy, xy, above]))
    return out


def free(p, inflated):
    if callable(inflated):                               # a Team (or any predicate) instead of inflated boxes
        return inflated(p)
    return not any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in inflated)


def seg_free(a, b, inflated, res=0.03):
    n = max(2, int(np.linalg.norm(b - a) / res) + 1)
    return all(free(a + (b - a) * s, inflated) for s in np.linspace(0, 1, n))


def rrt_star(start, goal, inflated, lo, hi, rng, n_iter=2500, step=0.22, radius=0.55, goal_bias=0.08):
    """Returns (nodes, parent index per node, raw path start..goal). Keeps sampling after the first solution."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    nodes, parent, cost = [start], [-1], [0.0]
    goal_idx = None
    for _ in range(n_iter):
        q = goal if rng.random() < goal_bias else rng.uniform(lo, hi)
        N = np.array(nodes)
        i = int(np.argmin(np.linalg.norm(N - q, axis=1)))
        d = q - N[i]
        dist = np.linalg.norm(d)
        if dist < 1e-9:
            continue
        new = N[i] + d / dist * min(step, dist)
        if not seg_free(N[i], new, inflated):
            continue
        near = np.where(np.linalg.norm(N - new, axis=1) < radius)[0]
        best, best_c = i, cost[i] + np.linalg.norm(new - N[i])
        for j in near:                                   # choose the cheapest parent
            c = cost[j] + np.linalg.norm(new - N[j])
            if c < best_c and seg_free(N[j], new, inflated):
                best, best_c = int(j), c
        nodes.append(new)
        parent.append(best)
        cost.append(best_c)
        k = len(nodes) - 1
        for j in near:                                   # rewire
            c = best_c + np.linalg.norm(N[j] - new)
            if c < cost[j] and seg_free(new, N[j], inflated):
                parent[j], cost[j] = k, c
        if np.linalg.norm(new - goal) < step and seg_free(new, goal, inflated):
            c = best_c + np.linalg.norm(goal - new)
            if goal_idx is None:
                nodes.append(goal.copy())
                parent.append(k)
                cost.append(c)
                goal_idx = len(nodes) - 1
            elif c < cost[goal_idx]:
                parent[goal_idx], cost[goal_idx] = k, c
    if goal_idx is None:
        raise RuntimeError("RRT*: no path found")
    path, i = [], goal_idx
    while i != -1:
        path.append(nodes[i])
        i = parent[i]
    return np.array(nodes), np.array(parent), np.array(path[::-1])


def prune(path, inflated):
    """Greedy shortcutting: from each kept node jump to the farthest node still visible."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not seg_free(path[i], path[j], inflated):
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


def bezier_smooth(path, inflated, d=0.45, n=16):
    """Replace every corner by a quadratic Bezier arc between points d back along the two segments (shrinking d
    until the arc is collision-free). Straight parts stay straight."""
    out = [path[0]]
    for k in range(1, len(path) - 1):
        a, c, b = path[k - 1], path[k], path[k + 1]
        dk = min(d, 0.45 * np.linalg.norm(c - a), 0.45 * np.linalg.norm(b - c))
        while dk > 0.02:
            p0 = c + (a - c) / np.linalg.norm(a - c) * dk
            p2 = c + (b - c) / np.linalg.norm(b - c) * dk
            s = np.linspace(0, 1, n)[:, None]
            arc = (1 - s) ** 2 * p0 + 2 * s * (1 - s) * c + s ** 2 * p2
            if all(free(q, inflated) for q in arc):
                out.extend(arc)
                break
            dk *= 0.6
        else:
            out.append(c)
    out.append(path[-1])
    return np.array(out)


def resample(path, speed, dt, accel=0.25):
    """Constant-speed time parametrisation with a smooth start and stop. Returns positions at every dt."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    L = s[-1]
    t_acc = speed / accel
    d_acc = 0.5 * accel * t_acc**2
    if 2 * d_acc > L:
        t_acc = np.sqrt(L / accel)
        d_acc, speed = L / 2, accel * t_acc
    T = 2 * t_acc + (L - 2 * d_acc) / speed
    t = np.arange(0, T + dt, dt)
    dist = np.where(t < t_acc, 0.5 * accel * t**2,
                    np.where(t < T - t_acc, d_acc + speed * (t - t_acc), L - 0.5 * accel * np.clip(T - t, 0, None) ** 2))
    return np.stack([np.interp(dist, s, path[:, i]) for i in range(3)], 1)


def lazy_centers(traj, cfg):
    """Formation centre at every sample of a platform trajectory under the hold-region rule of the runtime."""
    hold = hold_table(cfg)
    c = np.asarray(traj[0][:2], float).copy()
    out = np.empty((len(traj), 2))
    for i, p in enumerate(traj):
        c = lazy_center(c, p, float(np.interp(p[2], *hold)))
        out[i] = c
    return out
