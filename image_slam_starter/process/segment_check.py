"""Segment-by-segment accuracy check of an estimated trajectory WITHOUT ground truth.

Independent image measurements (cached in <target>/result/segcheck_cache.npz, ~1 min the first time) against what the
trajectory says, per route segment (~170 frames) and for the two public frame ranges:

 1. ROAD SLOPE.  Vanishing point of the road direction (RANSAC over LSD line segments of the undistorted lower image:
    lane lines, curbs, parked-car / building edges parallel to the road).  Its elevation e_vp is the pitch of the road
    ahead relative to the camera's vehicle-forward axis.  The trajectory predicts the same angle: the elevation of
    its own travel direction `LK` camera heights ahead, seen in the current camera frame (e_pred).  Fit
    e_vp = a + beta * e_pred over the whole route (a = constant mount-pitch error); the per-segment mean residual is the
    trajectory's missing grade [deg] and, times the segment length / camera height, a vertical error [camera heights].
    Also: trajectory grade (travel direction against the mean camera-up axis) of opposite passes of the same place must
    cancel (grade_a + grade_b) and the heights of the two passes must agree.
 2. ROAD CURVATURE.  Dominant lane line / curb in the bird's-eye view (road plane 1 camera height below the camera), robust
    quadratic y(x) over 2.5..13 camera heights ahead (constrained to run along the vanishing-point azimuth); its heading
    change over that range against the trajectory's heading change over the same arc length.  NOTE: measured too noisy /
    uncorrelated on Track2 (see output) - printed for completeness, not used for flags.
 3. REVISIT CONSISTENCY.  Same direction loop pairs of pairs_*.pkl (PnP relative poses, scaled by the cloud scale):
    rotation and translation-direction disagreement with the trajectory; opposite direction pairs: road-plane
    correction (forward, lateral, yaw) from utils/bevloop (marking BEV registration).
 4. START-END closure against the direct image measurement of pairs_*.pkl.

usage (from image_slam_starter):  python process/segment_check.py Track2 traj1.csv [traj2.csv ...] [--nseg 25]
"""
import argparse
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, ".")
from utils import nhc

CAM = (1035.5, 722.17, 456.83, -0.05833, 0.088413)
FYR = 1038.55506 / 1040.18078
MOUNT = (-0.14, -1.88)
K0 = np.array([[CAM[0], 0, CAM[1]], [0, CAM[0] * FYR, CAM[2]], [0, 0, 1.0]])
D0 = np.array([CAM[3], CAM[4], 0, 0, 0.0])
FV, LV, UV = nhc.axes(*np.radians(MOUNT))
KINV = np.linalg.inv(K0)
PUBLIC = [(1141, 1344), (2731, 2947)]
LK_LIST = (2, 3, 4, 6, 8, 12, 16)                    # lookahead candidates [camera heights] for the slope comparison
CX0, CX1 = 2.5, 13.0                           # BEV range for the curvature fit [camera heights]


# ------------------------------------------------------------------------------------------ image measurements
def _vp(img):
    """vanishing point of the road direction in the undistorted image -> (u, v, n_in, total_len, sd_v, spread_deg) or None"""
    r0, r1 = 470, 870
    g = img[r0:r1]
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    seg = lsd.detect(g)[0]
    if seg is None:
        return None
    s = seg.reshape(-1, 4).astype(np.float64)
    s[:, [1, 3]] += r0
    d = s[:, 2:] - s[:, :2]
    ln = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    s, d, ln, ang = (x[(ln > 35) & (ang > 10) & (ang < 78)] for x in (s, d, ln, ang))
    if len(s) < 4:
        return None
    # line coefficients a u + b v + c = 0, normalised
    nrm = np.c_[d[:, 1], -d[:, 0]] / ln[:, None]
    c = -(nrm * s[:, :2]).sum(1)
    L = np.c_[nrm, c]
    # the line must pass through the VP window (road ahead: row ~ 423 +- 130, column cx +- 500)
    ur = -(L[:, 1] * 423 + L[:, 2]) / L[:, 0]
    k = np.abs(ur - CAM[1]) < 650
    L, s, ln, nrm, d = L[k], s[k], ln[k], nrm[k], d[k]
    n = len(L)
    if n < 4:
        return None
    mid = s[:, :2] + d / 2
    rng = np.random.RandomState(0)
    ia = rng.randint(0, n, 500); ib = rng.randint(0, n, 500)
    best, bestv = 0.0, None
    P = np.cross(L[ia], L[ib])
    ok = np.abs(P[:, 2]) > 1e-9
    P = P[ok]
    u, v = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    ok = (np.abs(v - 423) < 130) & (np.abs(u - CAM[1]) < 500)
    u, v = u[ok], v[ok]
    if len(u) == 0:
        return None
    # angular residual of every segment to every hypothesis: distance(VP, line) / distance(VP, mid)
    dist = np.abs(L[None, :, 0] * u[:, None] + L[None, :, 1] * v[:, None] + L[None, :, 2])
    dm = np.hypot(mid[None, :, 0] - u[:, None], mid[None, :, 1] - v[:, None]) + 1.0
    inl = dist / dm < np.tan(np.radians(1.0))
    sc = (inl * ln[None, :]).sum(1)
    h = int(np.argmax(sc))
    x = np.array([u[h], v[h]])
    for _ in range(4):
        dm = np.hypot(mid[:, 0] - x[0], mid[:, 1] - x[1]) + 1.0
        res = (L[:, 0] * x[0] + L[:, 1] * x[1] + L[:, 2]) / dm
        m = np.abs(res) < np.tan(np.radians(1.0))
        if m.sum() < 3:
            return None
        w = ln[m] / dm[m] ** 2
        A = L[m, :2]; bb = -L[m, 2]
        AtW = A.T * w
        x = np.linalg.solve(AtW @ A, AtW @ bb)
    dm = np.hypot(mid[:, 0] - x[0], mid[:, 1] - x[1]) + 1.0
    res = (L[:, 0] * x[0] + L[:, 1] * x[1] + L[:, 2]) / dm
    m = np.abs(res) < np.tan(np.radians(1.0))
    A = L[m, :2]; w = ln[m] / dm[m] ** 2
    cov = np.linalg.inv((A.T * w) @ A)
    r = (A @ x + L[m, 2])
    sig2 = (w * r * r).sum() / max(m.sum() - 2, 1)
    sd_v = float(np.sqrt(sig2 * cov[1, 1]))
    ao = np.degrees(np.arctan2(nrm[m, 0], -nrm[m, 1])) % 180
    spread = float(ao.max() - ao.min())
    return x[0], x[1], int(m.sum()), float(ln[m].sum()), sd_v, spread


_BEV = {}


def _bev_map():
    if not _BEV:
        RES = 0.02
        xs = np.arange(CX1, CX0, -RES); ys = np.arange(2.3, -2.3, -RES)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        P = gx[..., None] * FV + gy[..., None] * LV - 1.0 * UV
        _BEV["uv"] = cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), K0, D0)[0].reshape(gx.shape + (2,)).astype(np.float32)
        _BEV["res"] = RES
    return _BEV


def _marking_points(gray):
    bm = _bev_map(); RES = bm["res"]
    b = cv2.remap(gray.astype(np.float32), bm["uv"][..., 0], bm["uv"][..., 1], cv2.INTER_LINEAR, borderValue=-1)
    valid = b >= 0
    b[~valid] = np.median(b[valid])
    valid = cv2.erode(valid.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)     # no edge response at the image border
    ks = int(0.5 / RES) | 1
    th = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
    sd = np.sqrt(np.maximum(cv2.blur(b * b, (61, 61)) - cv2.blur(b, (61, 61)) ** 2, 1.0))
    r = np.clip(th / sd, 0, 4)
    r[(r < 1.5) | ~valid] = 0
    rr, cc = np.nonzero(r[::2, ::2])
    return np.c_[CX1 - 2 * rr * RES, 2.3 - 2 * cc * RES], r[::2, ::2][rr, cc]


def _lane_quad(p, w, seed=0, tan_az=None):
    """dominant lane line: robust quadratic y = a + b x + c x^2 -> (a, b, c, span, n, rms) or None"""
    if len(p) < 300:
        return None
    rng = np.random.RandomState(seed)
    if len(p) > 4000:
        sel = rng.choice(len(p), 4000, replace=False); p, w = p[sel], w[sel]
    n = len(p)
    i = rng.randint(0, n, (700, 3))
    X = p[i, 0]; Y = p[i, 1]
    V = np.stack([np.ones_like(X), X, X * X], -1)
    okm = (X.max(1) - X.min(1) > 3.0) & (np.abs(np.linalg.det(V)) > 0.05)
    coef = np.full((700, 3), np.nan)
    try:
        coef[okm] = np.linalg.solve(V[okm], Y[okm][..., None])[..., 0]
    except np.linalg.LinAlgError:
        return None
    okc = np.isfinite(coef).all(1) & (np.abs(coef[:, 1]) < 0.45) & (np.abs(coef[:, 2]) < 0.015) & (np.abs(coef[:, 0]) < 3)
    if tan_az is not None:                      # lane lines run along the road: slope at 7 heights ~ the vanishing-point azimuth
        okc &= np.abs(coef[:, 1] + 14 * coef[:, 2] - tan_az) < 0.07
    coef = coef[okc]
    if len(coef) == 0:
        return None
    pred = coef[:, 0:1] + coef[:, 1:2] * p[None, :, 0] + coef[:, 2:3] * p[None, :, 0] ** 2
    sc = ((np.abs(pred - p[None, :, 1]) < 0.05) * w[None, :]).sum(1)
    c = coef[int(np.argmax(sc))]
    for tol in (0.08, 0.05):
        r = c[0] + c[1] * p[:, 0] + c[2] * p[:, 0] ** 2 - p[:, 1]
        m = np.abs(r) < tol
        if m.sum() < 30:
            return None
        c = np.polyfit(p[m, 0], p[m, 1], 2, w=w[m])[::-1]
    r = c[0] + c[1] * p[:, 0] + c[2] * p[:, 0] ** 2 - p[:, 1]
    m = np.abs(r) < 0.05
    return c[0], c[1], c[2], float(np.ptp(p[m, 0])), int(m.sum()), float(np.sqrt((r[m] ** 2).mean()))


def _measure(job):
    path = job
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    out = np.full(14, np.nan)
    try:
        und = cv2.undistort(im, K0, D0)
        v = _vp(und)
        if v is not None:
            out[0:6] = v
        ta = None
        if v is not None and v[5] > 8 and v[4] < 6 and abs(v[0] - CAM[1]) < 450:
            r_ = KINV @ np.array([v[0], v[1], 1.0])
            ta = float((r_ @ LV) / (r_ @ FV))
        q = _lane_quad(*_marking_points(im), tan_az=ta)
        if q is not None:
            out[6:12] = q
    except Exception:       # a single bad frame must not kill the run
        pass
    return out


def image_measurements(target, force=False):
    cache = f"{target}/result/segcheck_cache.npz"
    files = sorted(os.listdir(f"{target}/data/images"))
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        if len(z["m"]) == len(files):
            return z["m"]
    print(f"measuring {len(files)} images (vanishing point, lane curvature) ...", flush=True)
    with Pool(min(28, os.cpu_count())) as pool:
        m = np.array(pool.map(_measure, [f"{target}/data/images/{f}" for f in files], chunksize=8))
    np.savez(cache, m=m)
    return m


def vp_elevation(m):
    """elevation / azimuth [deg] of the vanishing point relative to the vehicle-forward axis, with a quality mask"""
    u, v = m[:, 0], m[:, 1]
    ray = np.c_[u, v, np.ones_like(u)] @ KINV.T
    el = np.degrees(np.arctan2(ray @ UV, ray @ FV))
    az = np.degrees(np.arctan2(ray @ LV, ray @ FV))
    good = np.isfinite(u) & (m[:, 4] < 6.0) & (m[:, 5] > 8.0) & (m[:, 2] >= 4) & (m[:, 3] > 250)
    return el, az, good


# ------------------------------------------------------------------------------------------ trajectory side
def load_poses(path):
    return np.genfromtxt(path, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)


def rot_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


class Traj:
    def __init__(self, path, target):
        self.path, self.name = path, os.path.basename(path)
        self.T = load_poses(path)
        self.N = len(self.T)
        self.C = self.T[:, :3, 3]
        Rm = self.T[:, :3, :3]
        self.F, self.L, self.U = Rm @ FV, Rm @ LV, Rm @ UV            # vehicle axes in the world
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.C, axis=0), axis=1))]
        self.g = self.U.mean(0); self.g /= np.linalg.norm(self.g)      # global up = mean camera-up axis
        k = 2
        i0, i1 = np.clip(np.arange(self.N) - k, 0, None), np.clip(np.arange(self.N) + k, None, self.N - 1)
        dv = self.C[i1] - self.C[i0]
        self.speed = np.linalg.norm(dv, axis=1) / np.maximum(i1 - i0, 1)           # traj units per frame
        self.d = dv / np.maximum(np.linalg.norm(dv, axis=1), 1e-12)[:, None]       # travel direction
        k = 8
        j0, j1_ = np.clip(np.arange(self.N) - k, 0, None), np.clip(np.arange(self.N) + k, None, self.N - 1)
        dg = self.C[j1_] - self.C[j0]
        self.grade = np.degrees(np.arcsin(np.clip((dg / np.maximum(np.linalg.norm(dg, axis=1), 1e-12)[:, None]) @ self.g, -1, 1)))   # +-8 frames
        self.extra = None

    def derive(self, target):
        """camera height [traj units] and the opposite-pass road-plane corrections (cached next to the trajectory)"""
        cp = f"{target}/result/segcheck_{self.name}.pkl"
        mt = os.path.getmtime(self.path)
        if os.path.exists(cp):
            z = pickle.load(open(cp, "rb"))
            if z.get("mtime") == mt:
                self.h, self.bev = z["h"], z["bev"]
                return
        from utils import bevloop
        mb = bevloop.MarkingBEV(f"{target}/data/images")
        h, nh = bevloop.unit_height(mb, self.T)
        print(f"  {self.name}: camera height {h} traj units ({nh} pairs); opposite-pass BEV registration ...", flush=True)
        corr, _ = bevloop.search(f"{target}/data/images", self.T, h, mb=mb, log=lambda *x: print(*x, flush=True))
        self.h, self.bev = h, corr
        pickle.dump(dict(mtime=mt, h=h, bev=corr), open(cp, "wb"))


def robust_fit(x, y, iters=8):
    """y = a + b x with Tukey-ish reweighting; returns a, b"""
    w = np.ones(len(x))
    for _ in range(iters):
        A = np.c_[np.ones(len(x)), x]
        sw = np.sqrt(w)
        a, b = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = y - a - b * x
        sd = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.clip(1 - (r / (4.0 * sd)) ** 2, 0, 1) ** 2
    return a, b


def seg_stats(vals, idx, edges):
    """per-segment (n, median, mean, p90 of abs) of vals[idx]"""
    out = []
    for lo, hi in edges:
        m = (idx >= lo) & (idx < hi)
        v = vals[m]
        v = v[np.isfinite(v)]
        out.append((len(v), np.median(v) if len(v) else np.nan, np.mean(v) if len(v) else np.nan,
                    np.percentile(np.abs(v), 90) if len(v) else np.nan))
    return out


def sem_blocks(x, idx, lo, hi, block=20):
    """standard error of the mean of x over frames lo..hi using 20-frame block means (frames are strongly correlated)"""
    bm = []
    for b0 in range(lo, hi, block):
        m = (idx >= b0) & (idx < b0 + block)
        if m.sum() >= 5:
            bm.append(np.mean(x[m]))
    return np.std(bm) / np.sqrt(len(bm)) if len(bm) >= 3 else np.nan


def f1(x, w=6, p=2):
    return " " * w if not np.isfinite(x) else f"{x:{w}.{p}f}"


def analyse(tr, m, pairs, scale, a, edges, names):
    N = tr.N
    t0 = np.arange(N)
    out = {}
    h = tr.h
    # ---------------------------------------------------------------- slope
    el, az, good = vp_elevation(m)
    pos = tr.speed / h > 0.04
    # heading change over the 3..11 height range ahead (straight-ish road only for the slope comparison)
    sn = tr.s
    def ahead(i, L):
        return np.minimum(np.searchsorted(sn, sn[i] + L * h), N - 1)
    ii = t0
    psi = lambda j: np.degrees(np.arctan2((tr.F[j] * tr.L[ii]).sum(1), (tr.F[j] * tr.F[ii]).sum(1)))
    j1, j2 = ahead(ii, CX0), ahead(ii, CX1)
    dpsi = psi(j2) - psi(j1)
    seglen = np.maximum(sn[j2] - sn[j1], 1e-9)
    kappa = np.radians(dpsi) / seglen                                   # 1 / traj unit
    valid_end = sn[ii] + CX1 * h < sn[-1]
    straight = np.abs(dpsi) < 5.0
    best = None
    for LK in LK_LIST:
        j = ahead(ii, LK)
        dj = tr.d[j]
        ep = np.degrees(np.arctan2((dj * tr.U[ii]).sum(1), (dj * tr.F[ii]).sum(1)))
        ok = good & pos & valid_end & straight & np.isfinite(ep)
        if ok.sum() < 200:
            continue
        a0, b0 = robust_fit(ep[ok], el[ok])
        r = el[ok] - a0 - b0 * ep[ok]
        sd = 1.4826 * np.median(np.abs(r - np.median(r)))
        if best is None or sd < best[0]:
            best = (sd, LK, a0, b0, ep, ok, np.corrcoef(ep[ok], el[ok])[0, 1])
    sd_s, LK, a0, b0, ep, ok, cor = best
    res = np.where(ok, el - a0 - b0 * ep, np.nan)
    e11 = np.where(good, el, np.nan)
    dd = [e11[i] - np.nanmedian(e11[i - 5:i + 6]) for i in range(5, N - 5) if np.isfinite(e11[i]) and np.isfinite(e11[i - 5:i + 6]).sum() >= 7]
    out["slope"] = dict(noise11=1.4826 * np.median(np.abs(dd)), LK=LK, a=a0, beta=b0, sd=sd_s, corr=cor, n=int(ok.sum()), res=res, ep=ep, el=el, ok=ok,
                        e_cur=np.degrees(np.arcsin(np.clip((tr.d * tr.U).sum(1), -1, 1))))
    # ---------------------------------------------------------------- curvature
    # image: heading change of the dominant lane line between 2.5 and 13 camera heights ahead (deg); trajectory: heading
    # change of the vehicle axis over the same arc range (deg).  Both median-filtered over 21 frames.
    ci = np.isfinite(m[:, 8]) & (m[:, 9] > 5) & (m[:, 10] > 60) & (m[:, 11] < 0.04) & pos & valid_end
    dh = np.degrees(np.arctan(m[:, 7] + 2 * m[:, 8] * CX1) - np.arctan(m[:, 7] + 2 * m[:, 8] * CX0))
    dh = np.where(ci, dh, np.nan)
    dps = np.where(valid_end & pos, dpsi, np.nan)
    def rmed(x, w=10):
        return np.array([np.nanmedian(x[max(0, i - w):i + w + 1]) if np.isfinite(x[max(0, i - w):i + w + 1]).sum() >= w else np.nan
                         for i in range(len(x))])
    dhs, dpss = rmed(dh), rmed(dps)
    sel = np.isfinite(dhs) & np.isfinite(dpss) & (np.abs(dpss) < 25)
    al = 1.0
    for _ in range(6):
        r = dhs - al * dpss
        mm = sel & (np.abs(r) < 4 * (1.4826 * np.median(np.abs(r[sel])) + 1e-6))
        al = (dhs[mm] * dpss[mm]).sum() / (dpss[mm] ** 2).sum()
    cres = np.where(sel, dhs - al * dpss, np.nan)
    gent = sel & (np.abs(dpss) > 3)
    out["curv"] = dict(alpha=al, n=int(sel.sum()), res=cres, dh=dhs, dpsi=dpss, sel=sel, n_gent=int(gent.sum()),
                       corr_all=np.corrcoef(dhs[sel], dpss[sel])[0, 1], corr_turn=np.corrcoef(dhs[gent], dpss[gent])[0, 1] if gent.sum() > 10 else np.nan,
                       sd=1.4826 * np.median(np.abs(cres[sel] - np.nanmedian(cres[sel]))), sd_straight=np.nanstd(dhs[sel & (np.abs(dpss) < 1.5)]),
                       iqr_frame=np.nanpercentile(dh[ci & (np.abs(dpsi) < 3)], [25, 75]))
    # ---------------------------------------------------------------- opposite passes of the same place: grade / height
    ia, ib = [], []
    for i in range(0, N, 3):
        if not pos[i]:
            continue
        far = np.abs(t0 - i) > 150
        cand = np.where(far & (tr.F @ tr.F[i] < -0.8) & pos)[0]
        if len(cand) == 0:
            continue
        dd = np.linalg.norm(tr.C[cand] - tr.C[i], axis=1)
        if dd.min() < 4 * h:
            ia.append(i); ib.append(int(cand[np.argmin(dd)]))
    ia, ib = np.array(ia, int), np.array(ib, int)
    dzp = (tr.C[ib] - tr.C[ia]) @ tr.g / h
    gsum = tr.grade[ia] + tr.grade[ib]
    out["opp"] = dict(ia=ia, ib=ib, dz=dzp, gsum=gsum)
    # ---------------------------------------------------------------- revisit consistency (PnP pairs)
    rows = []
    for (p, q), v in pairs.items():
        if v is None or q - p <= 150 or v[4] < 30 or v[6] < 30:
            continue
        tm = scale[p] * v[3]
        Tpq = np.linalg.inv(tr.T[p]) @ tr.T[q]
        rows.append((p, q, rot_deg(v[2].T @ Tpq[:3, :3]), tm, Tpq[:3, 3], np.linalg.norm(v[3])))
    P = np.array([r[0] for r in rows]); Q = np.array([r[1] for r in rows])
    rotd = np.array([r[2] for r in rows])
    tmv = np.array([r[3] for r in rows]); ttv = np.array([r[4] for r in rows])
    ok2 = np.linalg.norm(tmv, axis=1) > 0.3 * h
    sg = np.median(np.linalg.norm(ttv[ok2], axis=1) / np.linalg.norm(tmv[ok2], axis=1))
    tmv = tmv * sg
    cosd = (tmv * ttv).sum(1) / (np.linalg.norm(tmv, axis=1) * np.linalg.norm(ttv, axis=1) + 1e-12)
    dird = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    terr = np.linalg.norm(tmv - ttv, axis=1) / h
    out["rev"] = dict(P=P, Q=Q, rot=rotd, dir=dird, terr=terr, ok=ok2, sg=sg)
    # ---------------------------------------------------------------- start / end
    cl = []
    for p in range(0, 8):
        for q in range(N - 8, N):
            v = pairs.get((p, q))
            if v is None or v[4] < 30:
                continue
            Tpq = np.linalg.inv(tr.T[p]) @ tr.T[q]
            tm = sg * scale[p] * v[3]
            cl.append((Tpq[:3, 3], tm, rot_deg(v[2].T @ Tpq[:3, :3])))
    out["close"] = cl
    return out


def report(tr, o, edges, names):
    h = tr.h
    sl, cu, op, rv = o["slope"], o["curv"], o["opp"], o["rev"]
    print(f"\n==================== {tr.name}  (camera height {h:.3f} traj units, {tr.N} frames)")
    print(f"[1] ROAD SLOPE: VP elevation vs trajectory (lookahead {sl['LK']} heights): {sl['n']} straight-road frames, corr {sl['corr']:.2f}, "
          f"beta {sl['beta']:.2f} (1 = same), offset {sl['a']:+.2f} deg, residual sd (robust) {sl['sd']:.2f} deg; "
          f"trajectory e_pred sd {np.nanstd(sl['ep'][sl['ok']]):.2f} deg, VP sd {np.nanstd(sl['el'][sl['ok']]):.2f} deg")
    res_ = sl["res"]; bm = [np.nanmean(res_[i:i + 20]) for i in range(0, tr.N - 20, 20) if np.isfinite(res_[i:i + 20]).sum() >= 10]
    flat = sl["ok"] & (np.abs(sl["ep"]) < 0.3)
    print(f"    noise: single-frame VP elevation vs 11-frame median {sl['noise11']:.3f} deg; residual 20-frame block means sd {np.std(bm):.2f} deg ({len(bm)} blocks); "
          f"on flat straight frames (|e_pred| < 0.3, {flat.sum()}) the VP elevation is {np.median(sl['el'][flat]) - sl['a']:+.2f} deg from the fit offset, robust sd {1.4826 * np.median(np.abs(sl['el'][flat] - np.median(sl['el'][flat]))):.2f}")
    print(f"    travel direction vs camera forward axis in the trajectory: median {np.nanmedian(sl['e_cur']):+.2f}, sd {np.nanstd(sl['e_cur']):.2f} deg")
    print(f"[2] ROAD CURVATURE (lane-line heading change over 2.5..13 heights vs trajectory heading change, 21-frame medians): {cu['n']} frames, "
          f"fitted ratio {cu['alpha']:.2f} (1 = same), corr {cu['corr_all']:.2f} all / {cu['corr_turn']:.2f} where |trajectory change| > 3 deg ({cu['n_gent']} frames), "
          f"residual sd {cu['sd']:.1f} deg; image-only sd on trajectory-straight frames {cu['sd_straight']:.1f} deg, single-frame quartiles {cu['iqr_frame'][0]:+.1f}/{cu['iqr_frame'][1]:+.1f} deg")
    g = tr.grade
    idx = np.arange(tr.N)
    hdr = ("seg  frames        |  slope: n  mean_res(+-se) sd  e_pred_sd grade(deg)| curv: n  res_sd(deg) bias | "
           "same-dir revisit: n  rot med/p90  dir med/p90  terr[h] med/p90 | opposite pass: n  dz[h] med  gradesum med p90")
    print(hdr)
    rows = list(zip(names, edges))
    flags = []
    for nm, (lo, hi) in rows:
        m = (idx >= lo) & (idx < hi)
        r = sl["res"][m]; r = r[np.isfinite(r)]
        fin = np.isfinite(sl["res"])
        se = sem_blocks(sl["res"][fin], idx[fin], lo, hi)
        ep = sl["ep"][m & sl["ok"]]
        gr = np.nanmedian(g[m])
        c = cu["res"][m]; c = c[np.isfinite(c)]
        pm = (rv["P"] >= lo) & (rv["P"] < hi) | (rv["Q"] >= lo) & (rv["Q"] < hi)
        pm &= rv["ok"]
        om = (op["ia"] >= lo) & (op["ia"] < hi)
        s = f"{nm:>7} {lo:4d}-{hi:4d} | {len(r):4d} {f1(np.mean(r) if len(r) else np.nan)}+-{f1(se,4)} {f1(np.std(r) if len(r) else np.nan,5)} {f1(np.std(ep) if len(ep) else np.nan,6)} {f1(gr,7)}  | " \
            f"{len(c):4d} {f1(np.std(c) if len(c) else np.nan,6,1)} {f1(np.median(c) if len(c) else np.nan,6,1)} | " \
            f"{pm.sum():5d} {f1(np.median(rv['rot'][pm]) if pm.sum() else np.nan,5,2)}/{f1(np.percentile(rv['rot'][pm],90) if pm.sum() else np.nan,5,2)} " \
            f"{f1(np.median(rv['dir'][pm]) if pm.sum() else np.nan,5,2)}/{f1(np.percentile(rv['dir'][pm],90) if pm.sum() else np.nan,5,2)} " \
            f"{f1(np.median(rv['terr'][pm]) if pm.sum() else np.nan,5,2)}/{f1(np.percentile(rv['terr'][pm],90) if pm.sum() else np.nan,5,2)} | " \
            f"{om.sum():4d} {f1(np.median(op['dz'][om]) if om.sum() else np.nan,6,2)} {f1(np.median(op['gsum'][om]) if om.sum() else np.nan,6,2)} " \
            f"{f1(np.percentile(np.abs(op['gsum'][om]),90) if om.sum() else np.nan,6,2)}"
        print(s)
        if len(r) >= 30 and np.isfinite(se) and abs(np.mean(r)) > max(0.25, 3 * se):
            flags.append(f"slope {nm} {lo}-{hi}: mean residual {np.mean(r):+.2f} deg (se {se:.2f})")
        if pm.sum() >= 100 and (np.median(rv["terr"][pm]) > 0.25 or np.median(rv["dir"][pm]) > 1.5 or np.median(rv["rot"][pm]) > 0.3):
            flags.append(f"same-direction revisit {nm} {lo}-{hi}: median rot {np.median(rv['rot'][pm]):.2f} deg, direction {np.median(rv['dir'][pm]):.2f} deg, "
                         f"translation {np.median(rv['terr'][pm]):.2f} h ({pm.sum()} pairs)")
        if om.sum() >= 10 and abs(np.median(op["dz"][om])) > 0.15:
            flags.append(f"opposite-pass height {nm} {lo}-{hi}: dz {np.median(op['dz'][om]):+.2f} h")
    # opposite-pass BEV corrections
    print("[3b] OPPOSITE PASSES, road-plane correction from the marking BEV (units: camera heights; yaw deg); "
          "forward is weakly determined (road markings repeat)")
    print("      seg frames    |   n  fwd med  lat med  lat p90|abs|  yaw med  yaw p90|abs|")
    bev = tr.bev
    ba = np.array([k[0] for k in bev]); vals = np.array([v[:3] for v in bev.values()]) if bev else np.zeros((0, 3))
    for nm, (lo, hi) in rows:
        mk = (ba >= lo) & (ba < hi)
        if mk.sum() == 0:
            print(f"{nm:>9} {lo:4d}-{hi:4d} |   0"); continue
        v = vals[mk]
        print(f"{nm:>9} {lo:4d}-{hi:4d} | {mk.sum():3d} {v[:, 0].__class__ and f1(np.median(v[:, 0]) / h, 8, 2)} {f1(np.median(v[:, 1]) / h, 8, 2)} "
              f"{f1(np.percentile(np.abs(v[:, 1]) / h, 90), 8, 2)} {f1(np.degrees(np.median(v[:, 2])), 8, 2)} {f1(np.degrees(np.percentile(np.abs(v[:, 2]), 90)), 8, 2)}")
        if mk.sum() >= 4 and (abs(np.median(v[:, 1]) / h) > 0.2 or abs(np.degrees(np.median(v[:, 2]))) > 0.8):
            flags.append(f"opposite-pass BEV {nm} {lo}-{hi}: lateral {np.median(v[:, 1]) / h:+.2f} h, yaw {np.degrees(np.median(v[:, 2])):+.2f} deg")
    # overall
    allr = sl["res"][np.isfinite(sl["res"])]
    pm = rv["ok"]
    print(f"[overall] slope residual sd {np.std(allr):.2f} deg over {len(allr)} frames; same-direction revisit pairs {pm.sum()}: "
          f"rot median {np.median(rv['rot'][pm]):.2f} / p90 {np.percentile(rv['rot'][pm], 90):.2f} deg, direction {np.median(rv['dir'][pm]):.2f} / "
          f"{np.percentile(rv['dir'][pm], 90):.2f} deg, translation {np.median(rv['terr'][pm]):.2f} / {np.percentile(rv['terr'][pm], 90):.2f} h "
          f"(scale traj/meas {rv['sg']:.3f}); opposite-pass pairs {len(op['ia'])}: |dz| median {np.median(np.abs(op['dz'])):.2f} h, "
          f"p90 {np.percentile(np.abs(op['dz']), 90):.2f} h, grade sum median {np.median(op['gsum']):+.2f} deg, abs p90 {np.percentile(np.abs(op['gsum']), 90):.2f} deg")
    print(f"          total vertical range of the trajectory (along mean camera-up): {np.ptp(tr.C @ tr.g) / h:.1f} camera heights")
    # start / end
    cl = o["close"]
    if cl:
        tt = np.median([c[0] for c in cl], axis=0); tm = np.median([c[1] for c in cl], axis=0)
        print(f"[4] START-END: trajectory frame end in start camera frame (x right, y down, z fwd; heights) = {np.round(tt / h, 2)}; "
              f"measured (direct matching, {len(cl)} pairs, scaled) = {np.round(tm / h, 2)}; difference {np.round((tt - tm) / h, 2)} "
              f"(|diff| {np.linalg.norm(tt - tm) / h:.2f} h, {np.linalg.norm(tt - tm) / max(np.linalg.norm(tm), 1e-9) * 100:.0f} % of the gap); "
              f"rotation disagreement {np.median([c[2] for c in cl]):.2f} deg")
    else:
        print("[4] START-END: no direct pairs")
    print("FLAGS:" + ("" if flags else " none") )
    for f in flags:
        print("   " + f)
    cl = o["close"]
    se_ = (np.linalg.norm(np.median([c[0] for c in cl], axis=0) - np.median([c[1] for c in cl], axis=0)) / h) if cl else np.nan
    return dict(name=tr.name, h=h, beta=sl["beta"], corr=sl["corr"], sd=np.std(allr), blk=np.std(bm), rot=np.median(rv["rot"][pm]), dire=np.median(rv["dir"][pm]),
                terr=np.median(rv["terr"][pm]), terr90=np.percentile(rv["terr"][pm], 90), dz=np.median(np.abs(op["dz"])), dz90=np.percentile(np.abs(op["dz"]), 90),
                gs=np.percentile(np.abs(op["gsum"]), 90), vr=np.ptp(tr.C @ tr.g) / h, close=se_, nb=len(bev),
                bl=np.median(np.abs(vals[:, 1])) / h if len(vals) else np.nan, nfl=len(flags))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("trajs", nargs="*")
    ap.add_argument("--nseg", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--scale", default=None)
    a = ap.parse_args()
    m = image_measurements(a.target, a.force)
    pf = a.pairs or sorted(glob.glob(f"{a.target}/result/pairs_*.pkl"))[0]
    pairs = pickle.load(open(pf, "rb"))
    sf = a.scale or f"{a.target}/result/t2_lr.csv.scale.npy"
    scale = np.load(sf)
    N = len(m)
    eb = np.linspace(0, N, a.nseg + 1).astype(int)
    edges = [(int(eb[i]), int(eb[i + 1])) for i in range(a.nseg)] + PUBLIC
    names = [f"s{i:02d}" for i in range(a.nseg)] + ["pub1", "pub2"]
    summ = []
    for t in a.trajs:
        tr = Traj(t, a.target)
        tr.derive(a.target)
        o = analyse(tr, m, pairs, scale, a, edges, names)
        summ.append(report(tr, o, edges, names))
    if len(summ) > 1:
        print("\n==================== SUMMARY (one row per trajectory)")
        print("trajectory            h     | slope: resid sd, 20-frame-block sd, beta, corr | revisit med: rot deg, dir deg, transl h (p90) | opposite: |dz| h med/p90, grade-sum p90 deg, vertical range h | start-end diff h | BEV pairs, med |lat| h | flags")
        for s in summ:
            print(f"{s['name']:20s} {s['h']:.3f} | {s['sd']:.2f} {s['blk']:.2f} {s['beta']:.2f} {s['corr']:.2f} | {s['rot']:.2f} {s['dire']:.2f} {s['terr']:.2f} ({s['terr90']:.2f}) | "
                  f"{s['dz']:.2f}/{s['dz90']:.2f} {s['gs']:.2f} {s['vr']:.1f} | {s['close']:.2f} | {s['nb']} {s['bl']:.2f} | {s['nfl']}")
