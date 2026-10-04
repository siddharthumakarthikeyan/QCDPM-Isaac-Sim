"""Batched rotation helpers. Quaternions are (w, x, y, z), matching Isaac Sim's experimental API."""

import numpy as np


def cross(a, b):
    """np.cross for (..., 3) arrays without np.cross's axis bookkeeping (it dominates small-array cost)."""
    a, b = np.broadcast_arrays(a, b)
    c = np.empty(a.shape)
    c[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    c[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    c[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return c


def quat_to_rot(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rot_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    R = R.reshape(-1, 3, 3)
    q = np.empty((R.shape[0], 4))
    for i, m in enumerate(R):
        t = np.trace(m)
        if t > 0:
            s = 2.0 * np.sqrt(t + 1.0)
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q *= np.sign(q[:, :1] + 1e-12)
    return q[0] if single else q


def rpy_to_rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def yaw_of(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def vee(S):
    return np.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], axis=-1)


def rot_error(R_des, R):
    """so(3) error 0.5 vee(Rd^T R - R^T Rd), expressed in the body frame of R."""
    E = np.swapaxes(R_des, -1, -2) @ R - np.swapaxes(R, -1, -2) @ R_des
    return 0.5 * vee(E)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Orientation of a USD camera (looks down its local -Z, +Y up) at `eye` aimed at `target`."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    if abs(f @ up) > 0.999:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return rot_to_quat(np.stack([x, y, -f], axis=1))
