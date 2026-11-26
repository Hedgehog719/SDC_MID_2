"""Translation-direction factors from image geometry only (visual odometry style, no cloud depth).

With the relative rotation R_ab known (gyro-accurate in this pipeline), the epipolar constraint of every match
is linear in the translation:  t_ba . (R_ba x_a  x  x_b) = 0, so the direction of camera b seen from camera a
follows from all matched keypoints, also the far ones that have no usable cloud depth.  On Track01 this
direction scatters 2-3x less vertically and ~30% less sideways than the PnP + cloud-depth direction
(compared with the direction the car model predicts from the rotation alone; see NOTES.md).
Only the direction is used; the length still comes from the clouds.
"""
import cv2
import gtsam
import numpy as np


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def essential_inliers(K, pa, pb, thr_px=1.0, seed=0):
    """matches (pixel arrays) consistent with SOME relative pose: 5-point RANSAC. Returns a bool mask."""
    if len(pa) < 30:
        return np.zeros(len(pa), bool)
    cv2.setRNGSeed(int(seed))
    E, m = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, thr_px)
    if E is None or m is None:
        return np.zeros(len(pa), bool)
    return m[:, 0] > 0


def direction(Kinv, R_ab, pa, pb, iters=5):
    """unit direction of camera b's centre in camera a's frame, for a known rotation R_ab (camera b in a).
    Sign: not decided here (the caller orients it). Returns (dir, robust residual scale [rad])."""
    xa = np.c_[pa, np.ones(len(pa))] @ Kinv.T
    xb = np.c_[pb, np.ones(len(pb))] @ Kinv.T
    xa /= np.linalg.norm(xa, axis=1)[:, None]; xb /= np.linalg.norm(xb, axis=1)[:, None]
    A = np.cross(xa @ R_ab, xb)            # R_ba = R_ab^T  ->  R_ba x_a = (x_a^T R_ab)^T
    nA = np.linalg.norm(A, axis=1) + 1e-12
    w = np.ones(len(A)); s = np.nan
    for _ in range(iters):                 # IRLS, Huber-like
        t_ba = np.linalg.svd(A * w[:, None], full_matrices=False)[2][-1]
        res = np.abs(A @ t_ba) / nA
        s = 1.4826 * np.median(res) + 1e-9
        w = 1.0 / np.maximum(1.0, res / (2 * s))
    return -R_ab @ t_ba, s                 # c_b in frame a = -R_ab t_ba


def measure(K, poses, matches, min_inl=50, min_move=0.1, min_dist=None):
    """matches: {(a, b): (pa, pb)} essential-matrix inliers.  Returns {(a, b): (unit dir in frame a, n)} using
    the rotations of `poses`; the sign is taken from the current trajectory, pairs that hardly move are skipped
    (less than min_move per frame of b - a, or less than min_dist [m] when given: loop pairs)."""
    Kinv = np.linalg.inv(K)
    out = {}
    for (a, b), (pa, pb) in matches.items():
        if len(pa) < min_inl:
            continue
        T = np.linalg.inv(poses[a]) @ poses[b]
        if np.linalg.norm(T[:3, 3]) < (min_move * (b - a) if min_dist is None else min_dist):
            continue
        d, _ = direction(Kinv, T[:3, :3], pa.astype(np.float64), pb.astype(np.float64))
        if d @ T[:3, 3] < 0:
            d = -d
        out[(a, b)] = (d, len(pa))
    return out


def _angles(v):
    """(lateral, vertical) angle of a direction in the camera frame (x right, y down, z forward)"""
    return np.arctan2(v[..., 0], v[..., 2]), np.arctan2(v[..., 1], np.hypot(v[..., 0], v[..., 2]))


def debias(meas, poses, R=None):
    """Remove the constant angular offset between the VO directions and the current trajectory.
    A constant offset in the camera frame only rotates a (near) planar trajectory as a whole, but in the graph it
    would fight the PnP and car factors, so only the scatter of the VO directions is used.
    R: apply this correction (stats["R"] of an earlier call) instead of estimating it."""
    keys = list(meas)
    m = np.array([meas[p][0] for p in keys])
    t = np.array([(np.linalg.inv(poses[a]) @ poses[b])[:3, 3] for a, b in keys])
    t /= np.linalg.norm(t, axis=1)[:, None]
    (lm, vm), (lt, vt) = _angles(m), _angles(t)
    dl, dv = np.median(lt - lm), np.median(vt - vm)
    if R is None:
        R = cv2.Rodrigues(np.array([0.0, dl, 0.0]))[0] @ cv2.Rodrigues(np.array([-dv, 0.0, 0.0]))[0]
    # (rotation about y moves the direction sideways, about x up/down; small angles, order irrelevant)
    fixed = {p: (R @ meas[p][0], meas[p][1]) for p in keys}
    m2 = np.array([fixed[p][0] for p in keys])
    l2, v2 = _angles(m2)
    rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    stats = dict(offset_lat_deg=np.degrees(dl), offset_vert_deg=np.degrees(dv),
                 resid_lat_deg=np.degrees(rs(lt - l2)), resid_vert_deg=np.degrees(rs(vt - v2)),
                 bias_after_lat_deg=np.degrees(np.median(lt - l2)), bias_after_vert_deg=np.degrees(np.median(vt - v2)), R=R)
    return fixed, stats


def error(Ta, Tb, M, jac=False):
    """e = M t_hat, t = R_a^T (p_b - p_a); M (2x3) = lateral / vertical unit vectors perpendicular to the
    measured direction.  gtsam Pose3 local coordinates: R <- R Exp(w), p <- p + R v."""
    Ra, pa, Rb, pb = Ta[:3, :3], Ta[:3, 3], Tb[:3, :3], Tb[:3, 3]
    t = Ra.T @ (pb - pa)
    n = np.linalg.norm(t) + 1e-12
    th = t / n
    e = M @ th
    if not jac:
        return e
    D = M @ (np.eye(3) - np.outer(th, th)) / n     # de/dt
    Ha = np.zeros((2, 6)); Hb = np.zeros((2, 6))
    Ha[:, :3] = D @ skew(t)
    Ha[:, 3:] = -D
    Hb[:, 3:] = D @ (Ra.T @ Rb)
    return e, Ha, Hb


def basis(m):
    """lateral and vertical unit vectors perpendicular to the measured direction m (camera frame, y down)"""
    lat = np.cross([0.0, -1.0, 0.0], m); lat /= np.linalg.norm(lat)
    up = np.cross(m, lat); up /= np.linalg.norm(up)
    return np.vstack([lat, up])


def factor(a, b, m, sigmas):
    M = basis(m)
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                                           gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigmas, dtype=float)))

    def fn(this, v, H):
        Ta, Tb = v.atPose3(a).matrix(), v.atPose3(b).matrix()
        if H is None:
            return error(Ta, Tb, M)
        e, H[0], H[1] = error(Ta, Tb, M, jac=True)
        return e

    return gtsam.CustomFactor(noise, [a, b], fn)


# ---------------------------------------------------------------------------------------------- relative rotation
# Without a gyro (Track02) the rotations come only from PnP on the cloud depth (0.4 deg per 10 frames on Track01).
# The essential matrix of all matched keypoints gives the relative rotation 3-4x more precisely (Track01, against
# the gyro-constrained trajectory: median 0.06 / 0.08 / 0.11 deg for steps 3 / 5 / 8; see NOTES.md).

def _sampson(x, xa, xb):
    R = cv2.Rodrigues(x[:3])[0]
    t = x[3:] / np.linalg.norm(x[3:])
    E = skew(t) @ R                                  # x_b^T E x_a = 0 for x_b ~ R x_a + t (camera a -> camera b)
    Ex = xa @ E.T
    Etx = xb @ E
    num = np.einsum("ij,ij->i", xb, Ex)
    return num / (np.sqrt(Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2) + 1e-15)


def rotation_job(job):
    """job = (Kinv, f, pa, pb, T_ab): refine the relative pose of the pair on the Sampson error of all
    essential-matrix inliers (Huber), starting from the current estimate T_ab (camera b in camera a).
    Returns R_ab (camera b in a) and the robust residual scale [px]."""
    from scipy.optimize import least_squares
    Kinv, f, pa, pb, T = job
    xa = np.c_[pa, np.ones(len(pa))] @ Kinv.T
    xb = np.c_[pb, np.ones(len(pb))] @ Kinv.T
    R_ba, t_ba = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
    if np.linalg.norm(t_ba) < 1e-9:
        return None
    x0 = np.r_[cv2.Rodrigues(R_ba)[0][:, 0], t_ba / np.linalg.norm(t_ba)]
    r = least_squares(_sampson, x0, args=(xa, xb), loss="huber", f_scale=1.0 / f)
    return cv2.Rodrigues(r.x[:3])[0].T, 1.4826 * np.median(np.abs(r.fun)) * f


def rotations(K, poses, matches, procs=8, min_inl=50):
    """{(a, b): (R_ab, inliers)} relative rotations from image geometry only, in parallel processes"""
    import multiprocessing as mp
    Kinv = np.linalg.inv(K)
    keys = [p for p, (pa, _) in matches.items() if len(pa) >= min_inl]
    jobs = [(Kinv, K[0, 0], matches[p][0].astype(np.float64), matches[p][1].astype(np.float64),
             np.linalg.inv(poses[p[0]]) @ poses[p[1]]) for p in keys]
    with mp.get_context("fork").Pool(procs) as pool:
        res = pool.map(rotation_job, jobs, chunksize=32)
    return {p: (r[0], len(matches[p][0])) for p, r in zip(keys, res) if r is not None}


def rotation_factor(a, b, R_ab, sigma):
    """rotation-only between factor (translation free), robust"""
    T = np.eye(4); T[:3, :3] = R_ab
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.ones(3) * sigma, np.ones(3) * 1e3])
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), noise)
    return gtsam.BetweenFactorPose3(a, b, gtsam.Pose3(T), noise)
