"""Yellow dashed centre line as an along-track ruler (user's idea; detector from process/dash_ruler2.py).

The dashes repeat with one constant period along the roads (Track02: ~10 m, 4 m line + 6 m gap).  In every image the
road plane is sampled from above, the line's yellowness profile along the road is binarised and the dash edges are
located; an edge is followed from frame to frame and the fractional frame at which it passes a fixed distance x0 ahead
is interpolated.  Timing every edge at the same place in the image makes the result independent of the BEV scale
(camera height, pitch).  Consecutive crossings of the same edge type are one period apart along the road, so

    || p(t_k+1) - p(t_k) || = D        (D: one unknown period, a variable of the pose graph)

constrains the trajectory's local scale on every stretch with dashes, relative to all the others.
"""
import os
from multiprocessing import Pool

import cv2
import gtsam
import numpy as np

from utils import nhc

RES, XN, XF, YW = 0.02, 1.5, 11.0, 3.0
_G = {}


SFM_K = np.array([[1035.5, 0, 722.17], [0, 1035.5 * 1038.55506 / 1040.18078, 456.83], [0, 0, 1.0]])   # self-calibrated (sfm_ba.py)
SFM_D = np.array([-0.05833, 0.088413, 0, 0, 0.0])


def _setup(img_dir, K=SFM_K, D=SFM_D, mount=(-0.14, -1.88)):
    F, L, U = nhc.axes(*np.radians(mount))
    xs = np.arange(XF, XN, -RES); ys = np.arange(YW, -YW, -RES)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    P = gx[..., None] * F + gy[..., None] * L - U
    uv = cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)
    _G.update(uv=uv, ys=ys, files=sorted(os.listdir(img_dir)), dir=img_dir)


def _yellow(i):
    im = cv2.imread(os.path.join(_G["dir"], _G["files"][i])).astype(np.float32)
    b = cv2.remap(im, _G["uv"][..., 0], _G["uv"][..., 1], cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
    valid = b[..., 0] >= 0
    y = np.minimum(b[..., 1], b[..., 2]) - b[..., 0]
    y[~valid] = 0
    th = cv2.morphologyEx(y, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.35 / RES) | 1, 1)))
    th[~valid] = 0
    return th.astype(np.float16)


def profiles(img_dir, cache, procs=16, latlo=0.0, lathi=1.6, K=SFM_K, D=SFM_D, mount=(-0.14, -1.88)):
    """per frame: yellowness profile along the road in the dashed line's band, its lateral position.
    K, D, mount: the camera the trajectory is built with (one geometry for poses and ruler)"""
    if os.path.exists(cache):
        d = np.load(cache)
        return d["prof"], d["lat"]
    _setup(img_dir, K, D, mount)
    N = len(_G["files"]); ys = _G["ys"]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs, initializer=_setup, initargs=(img_dir, K, D, mount)) as pool:
        Ms = pool.map(_yellow, range(N), chunksize=8)
    r0, r1 = int((XF - 9) / RES), int((XF - 2.5) / RES)
    cs = np.array([m[r0:r1].astype(np.float32).mean(0) for m in Ms])
    cs = cv2.GaussianBlur(cs, (0, 0), sigmaX=2, sigmaY=0.001)
    cs[:, ~((ys > latlo) & (ys < lathi))] = 0                      # centre line: left of the car
    c_raw = cs.argmax(1).astype(float)
    c_med = np.array([np.median(c_raw[max(0, i - 25):i + 26]) for i in range(N)])
    lat_c = np.zeros(N, int)
    for i in range(N):
        lo = int(max(0, c_med[i] - 25)); hi = int(min(cs.shape[1], c_med[i] + 26))
        lat_c[i] = lo + cs[i, lo:hi].argmax()
    lat_c = np.round([np.median(lat_c[max(0, i - 7):i + 8]) for i in range(N)]).astype(int)
    prof = np.array([Ms[i][:, max(0, c - 3):c + 4].astype(np.float32).mean(1) for i, c in enumerate(lat_c)], np.float32)
    lat = ys[lat_c]
    tmp = cache + ".tmp.npz"; np.savez(tmp, prof=prof, lat=lat); os.replace(tmp, cache)
    return prof, lat


def _edges(p):
    lo, hi = np.percentile(p, 10), np.percentile(p, 95)
    if hi - lo < 6.0:
        return None
    mid = (lo + hi) / 2; on = p > mid
    near, far = [], []
    for k in np.where(on[:-1] != on[1:])[0]:
        v0, v1 = p[k], p[k + 1]
        x = XF - (k + (v0 - mid) / (v0 - v1 + 1e-9)) * RES
        (near if on[k] else far).append(x)
    return np.array(near), np.array(far)


def periods(prof, lat, poses, F, h=1.2, x0=5.0, plo=0.7, phi=1.35, cruise=0.0):
    """pairs of consecutive crossings (t_k, t_k+1) of the same edge type at x0 (fractional frames).
    The current trajectory is only used to match an edge between consecutive frames (its step) and to reject
    missed / doubled dashes (period outside plo..phi of the median)."""
    N = len(prof)
    E = [_edges(prof[i]) for i in range(N)]
    st = np.array([(np.linalg.inv(poses[i])[:3, :3] @ (poses[i + 1, :3, 3] - poses[i, :3, 3])) @ F / h for i in range(N - 1)] + [0.0])
    ev = {0: [], 1: []}
    ratios = []
    for i in range(N - 1):
        if E[i] is None or E[i + 1] is None or st[i] < 0.05 or abs(lat[i] - lat[i + 1]) > 0.2:
            continue
        for ty in (0, 1):
            e1 = E[i + 1][ty]
            for x in E[i][ty]:
                m = e1[np.abs(e1 - (x - st[i])) < 0.5 * st[i] + 0.05]
                if len(m) != 1:
                    continue
                ratios.append(((x - m[0]) / st[i], ty, i + (x - x0) / (x - m[0]) if m[0] < x0 <= x else None))
    rr = np.median([r[0] for r in ratios]) if ratios else 1.0
    for r_, ty, t in ratios:
        if t is not None and abs(r_ / rr - 1) < 0.35:
            ev[ty].append(t)
    C = poses[:, :3, 3]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    pairs = []
    stp = np.r_[np.linalg.norm(np.diff(C, axis=0), axis=1), 0.0]
    stp = np.array([np.median(stp[max(0, i - 2):i + 3]) for i in range(N)])         # speed per frame (smoothed)
    for ty in (0, 1):
        t = np.sort(ev[ty])
        for k in range(len(t) - 1):
            if cruise > 0:      # speed change over the period: braking / accelerating nods the car (pitch moves the ruler)
                va, vb = stp[int(t[k])], stp[min(int(t[k + 1]), N - 1)]
                if abs(vb / max(va, 1e-6) - 1) > cruise:
                    continue
            if 5 < t[k + 1] - t[k] < 40:
                pairs.append((t[k], t[k + 1], float(np.interp(t[k + 1], np.arange(N), s) - np.interp(t[k], np.arange(N), s))))
    if not pairs:
        return [], np.nan
    per = np.array([p[2] for p in pairs]); med = np.median(per)
    keep = [p for p in pairs if plo * med < p[2] < phi * med]
    return keep, float(med)


DKEY = 10 ** 7          # key of the unknown period (never a frame index)


def factor(ta, tb, sigma, robust=True):
    """|| p(tb) - p(ta) || - D ;  p(t) linear between frames floor(t) and floor(t)+1"""
    ia, ib = int(ta), int(tb); fa, fb = ta - ia, tb - ib
    keys = [ia, ia + 1, ib, ib + 1, DKEY]
    w = [-(1 - fa), -fa, 1 - fb, fb]

    def err(this, values, H):
        Ps = [values.atPose3(k) for k in keys[:4]]
        d = sum(wi * P_.translation() for wi, P_ in zip(w, Ps))
        n = max(np.linalg.norm(d), 1e-9); u = d / n
        Dv = values.atVector(DKEY)[0]
        if H is not None:
            for j in range(4):
                H[j] = np.hstack([np.zeros((1, 3)), (w[j] * u)[None] @ Ps[j].rotation().matrix()])
            H[4] = np.array([[-1.0]])
        return np.array([n - Dv])
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
    if robust:
        noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), noise)
    return gtsam.CustomFactor(noise, keys, err)


def line_points_factor(i, j, k, lat, F, L, up_w, h, x0, sigma):
    """the yellow centre line is straight: its points seen from frames i, j, k (x0 camera heights ahead, lat[.] to the
    left, in camera heights) lie on one line in the world -> horizontal distance of the middle point from the chord = 0.
    Numerical Jacobian (few hundred factors)."""
    keys = [i, j, k]
    off = [h * (x0 * F + lat[q] * L) for q in keys]

    def pts(Ps):
        return [P_.translation() + P_.rotation().matrix() @ o for P_, o in zip(Ps, off)]

    def e_of(Ps):
        Pa, Pb, Pc = pts(Ps)
        u = Pc - Pa; u /= max(np.linalg.norm(u), 1e-9)
        n = np.cross(u, up_w); n /= max(np.linalg.norm(n), 1e-9)
        return np.array([(Pb - Pa) @ n])

    def err(this, values, H):
        Ps = [values.atPose3(q) for q in keys]
        e = e_of(Ps)
        if H is not None:
            for a_ in range(3):
                J = np.zeros((1, 6))
                for d in range(6):
                    dx = np.zeros(6); dx[d] = 1e-6
                    Pp = list(Ps); Pp[a_] = Ps[a_].retract(dx)
                    Pm = list(Ps); Pm[a_] = Ps[a_].retract(-dx)
                    J[0, d] = (e_of(Pp)[0] - e_of(Pm)[0]) / 2e-6
                H[a_] = J
        return e
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), gtsam.noiseModel.Isotropic.Sigma(1, sigma))
    return gtsam.CustomFactor(noise, keys, err)


def line_triplets(lat, poses, U, span=20, max_turn=1.5, max_lat_jump=0.15):
    """frame triplets (i, i+span, i+2 span) on straight stretches with a tracked centre line: the line's lateral position
    changes smoothly (no jumps to another line) and the car turns less than max_turn degrees over the triplet"""
    from scipy.spatial.transform import Rotation as R_
    N = len(lat); out = []
    for i in range(0, N - 2 * span, span // 2):
        j, k = i + span, i + 2 * span
        seg = lat[i:k + 1]
        if np.max(np.abs(np.diff(seg))) > max_lat_jump:
            continue
        turn = abs(np.degrees(R_.from_matrix(poses[i, :3, :3].T @ poses[k, :3, :3]).as_rotvec() @ U))
        if turn > max_turn:
            continue
        out.append((i, j, k))
    return out
