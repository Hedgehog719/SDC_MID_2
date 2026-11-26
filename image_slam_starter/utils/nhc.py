"""Non-holonomic (car) motion constraint.

A car cannot move sideways: the point above the rear axle moves along the car's heading. The camera sits a
lever arm L ahead of that point, so in a turn the camera does move sideways, by exactly (yaw change) x L.
With the mounting direction of the camera (yaw a, pitch b) and L calibrated from the data itself, the
*direction* of every frame-to-frame translation is known from the rotation, and only its length has to
come from the images. PnP directions wander by 1-2 degrees, which is what bends an open lap sideways.

    rear-axle displacement   d = p_j - p_i - L (R_j f - R_i f)       f = travel direction in the camera frame
    constraint               l_mid . d = 0 ,  u_mid . d = 0          l/u = lateral / up axes, at the mean heading
"""
import gtsam
import numpy as np
from scipy.optimize import least_squares


def axes(a, b):
    f = np.array([np.tan(a), np.tan(b), 1.0]); f /= np.linalg.norm(f)
    l = np.cross([0.0, -1.0, 0.0], f); l /= np.linalg.norm(l)
    return f, l, np.cross(f, l)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def calibrate(samples, with_height=False):
    """samples: list of (R_ij, t_ij) relative motions (camera i frame, metric). Returns (a, b, L, h) and residual
    angles.  h = height of the camera above the constrained point (0 unless with_height): with h the point is
    on the ground under the rear axle, so that body roll / road camber changes (which swing the high-mounted
    camera sideways) are not mistaken for sideways driving."""
    Rs = np.array([s[0] for s in samples]); ts = np.array([s[1] for s in samples])
    d = np.linalg.norm(ts, axis=1)

    def res(p):
        f, l, u = axes(p[0], p[1])
        o = p[2] * f + (p[3] if with_height else 0.0) * u
        dp = ts - (Rs @ o - o)
        lm = l + Rs @ l; lm /= np.linalg.norm(lm, axis=1)[:, None]
        um = u + Rs @ u; um /= np.linalg.norm(um, axis=1)[:, None]
        return np.r_[(lm * dp).sum(1) / d, (um * dp).sum(1) / d]

    sol = least_squares(res, [0.0, 0.0, 1.5, 0.0], loss="soft_l1", f_scale=0.02)
    if not with_height:
        sol.x[3] = 0.0
    r = res(sol.x).reshape(2, -1)
    return sol.x, np.degrees(1.4826 * np.median(np.abs(r - np.median(r, axis=1)[:, None]), axis=1))


def error(Ti, Tj, f, l, u, L, jac=False, h=0.0):
    Ri, Rj, pi, pj = Ti[:3, :3], Tj[:3, :3], Ti[:3, 3], Tj[:3, 3]
    o = L * f + h * u                     # camera relative to the constrained point (car frame)
    d = pj - pi - (Rj @ o - Ri @ o)
    nl = Ri @ l + Rj @ l; nu = Ri @ u + Rj @ u
    sl, su = np.linalg.norm(nl), np.linalg.norm(nu)
    lm, um = nl / sl, nu / su
    e = np.array([lm @ d, um @ d])
    if not jac:
        return e
    # local coordinates of gtsam Pose3: [rotation (3), translation (3)], R <- R Exp(w), p <- p + R v
    Hi = np.zeros((2, 6)); Hj = np.zeros((2, 6))
    dd_wi = -Ri @ skew(o); dd_wj = Rj @ skew(o)
    for row, (m, s, ax) in enumerate(((lm, sl, l), (um, su, u))):
        Pm = (np.eye(3) - np.outer(m, m)) / s
        Hi[row, :3] = d @ Pm @ (-Ri @ skew(ax)) + m @ dd_wi
        Hj[row, :3] = d @ Pm @ (-Rj @ skew(ax)) + m @ dd_wj
        Hi[row, 3:] = -m @ Ri
        Hj[row, 3:] = m @ Rj
    return e, Hi, Hj


def factor(i, j, calib, sigmas):
    a, b, L = calib[:3]
    h = calib[3] if len(calib) > 3 else 0.0
    f, l, u = axes(a, b)
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                                           gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigmas, dtype=float)))

    def fn(this, v, H):
        Ti, Tj = v.atPose3(i).matrix(), v.atPose3(j).matrix()
        if H is None:
            return error(Ti, Tj, f, l, u, L, h=h)
        e, H[0], H[1] = error(Ti, Tj, f, l, u, L, jac=True, h=h)
        return e

    return gtsam.CustomFactor(noise, [i, j], fn)
