"""Loops between OPPOSITE passes from road markings seen from above (user's idea, after Midterm01's BEV matching).

The point-cloud registration of opposite passes (utils/revloop.py) slides along the road (LoGeR compresses the far
depths where the two views overlap), so only its lateral offset is used.  Road markings (stop lines, crosswalks, text,
dashes, curbs) are the same ground pattern from both directions: every image is mapped onto the road plane near the
car (inverse perspective with the self-calibrated lens, 3..11 camera heights ahead, +-2 sideways), thin bright
structures are kept, and the marking points of pass b are placed in pass a's road frame with the current estimate and
aligned there by a search over yaw and a 2D shift.  The result is the relative road-plane pose (forward, lateral, yaw)
of b in a, in units of the camera height h (converted to trajectory units with h).

Reliability: the correlation peak must be clearly above the second best peak (dashes / crosswalk stripes repeat), and
neighbouring candidates (a +- 5 frames) must give the same correction; otherwise the pair is dropped.
"""
import os

import cv2
import gtsam
import numpy as np

from utils import nhc

RES, X0, X1, Y1 = 0.02, 3.0, 11.0, 2.0


def _roll_one(args):
    f, rmap, omap, K, mount = args
    from utils.io import load_ply
    F, L, U = nhc.axes(*np.radians(mount))
    p = load_ply(f)[:, :3]; p = p[np.isfinite(p).all(1) & (p[:, 2] > 0.5)]
    uv = p @ K.T; uv = uv[:, :2] / uv[:, 2:3]
    r = np.clip((uv[:, 1] / 4).astype(int), 0, rmap.shape[0] - 1); c = np.clip((uv[:, 0] / 4).astype(int), 0, rmap.shape[1] - 1)
    q = p[(rmap[r, c] > 200) & (omap[r, c] < 50)]
    x, y, z = q @ F, q @ L, q @ U
    s = (x > 3.5) & (x < 8) & (y > -1.2) & (y < 1.0)
    if s.sum() < 200:
        return np.nan
    A = np.c_[np.ones(s.sum()), x[s], y[s]]
    return float(np.degrees(np.arctan(np.linalg.lstsq(A, z[s], rcond=None)[0][2])))


def road_roll(files, seg, K, cache, procs=16, smooth=31, mount=(-0.14, -1.88)):
    """roll of the road plane under / ahead of the car in the camera frame [deg] (+ = road rises to the left), per
    frame, from the cloud points the segmentation calls road; median over `smooth` frames.  Contains the camera's mounting
    roll (Track02 about -0.9 deg) and the road's cross slope (Track02 top road about +-0.8 deg)."""
    if os.path.exists(cache):
        return np.load(cache)
    z = np.load(seg)
    from concurrent.futures import ThreadPoolExecutor       # (a forked pool crashes inside open3d)
    road_, obj_ = z["road"], z["obj"]
    with ThreadPoolExecutor(procs) as ex:
        r = np.array(list(ex.map(lambda i: _roll_one((files[i], road_[i], obj_[i], K, mount)), range(len(files)))))
    ok = np.isfinite(r)
    r = np.interp(np.arange(len(r)), np.where(ok)[0], r[ok])
    h = smooth // 2
    r = np.array([np.median(r[max(0, i - h):i + h + 1]) for i in range(len(r))])
    np.save(cache, r)
    return r


class MarkingBEV:
    def __init__(self, img_dir, cam=(1035.5, 722.17, 456.83, -0.05833, 0.088413), fy_ratio=1038.55506 / 1040.18078,
                 mount=(-0.14, -1.88), seg=None, roll=None, K=None, D=None):
        """K, D: the camera the trajectory is built with (OpenCV); default: the self-calibrated one of sfm_ba.py"""
        if K is None:
            f_, cx, cy, k1, k2 = cam
            K = np.array([[f_, 0, cx], [0, f_ * fy_ratio, cy], [0, 0, 1.0]])
            D = np.array([k1, k2, 0, 0, 0.0])
        self.F, self.L, self.U = nhc.axes(*np.radians(mount))
        xs = np.arange(X1, X0, -RES); ys = np.arange(Y1, -Y1, -RES)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self._K, self._D, self._gx, self._gy = K, D, gx, gy
        self.uv = self._uv(0.0)
        self.roll = roll                                   # per-frame road roll [deg] (road_roll) or None
        self._uvc = {0.0: self.uv}
        self.files = sorted(os.listdir(img_dir)); self.dir = img_dir
        self.cache = {}
        self.min_pts = 1500 if seg is None else 700
        # optional SegFormer probabilities (<track>/seg.npz, quarter resolution): markings only on the road, not on cars / people
        self.seg = None
        if seg is not None and os.path.exists(seg):
            z = np.load(seg)
            self.seg = (z["road"], z["obj"])
            self.suv = (self.uv / 4.0).astype(np.float32)

    def _uv(self, phi):
        """image coordinates of the road grid when the road rises to the left by phi degrees (camera frame)"""
        P = self._gx[..., None] * self.F + self._gy[..., None] * self.L + (self._gy[..., None] * np.tan(np.radians(phi)) - 1.0) * self.U
        return cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), self._K, self._D)[0].reshape(self._gx.shape + (2,)).astype(np.float32)

    def uv_for(self, i):
        if self.roll is None:
            return self.uv
        phi = round(float(self.roll[i]) * 4) / 4.0
        if phi not in self._uvc:
            self._uvc[phi] = self._uv(phi)
        return self._uvc[phi]

    def points(self, i):
        """marking cells of image i: (n, 2) road-frame points (forward, left) in camera heights, weights"""
        if i in self.cache:
            return self.cache[i]
        uvi = self.uv_for(i); suvi = uvi / 4.0
        g = cv2.imread(os.path.join(self.dir, self.files[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        b = cv2.remap(g, uvi[..., 0], uvi[..., 1], cv2.INTER_LINEAR, borderValue=-1)
        valid = b >= 0
        b[~valid] = np.median(b[valid])
        ks = int(0.5 / RES) | 1
        th = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
        sd = np.sqrt(np.maximum(cv2.blur(b * b, (61, 61)) - cv2.blur(b, (61, 61)) ** 2, 1.0))
        r = np.clip(th / sd, 0, 4)
        r[(r < 1.5) | ~valid] = 0
        if self.seg is not None:
            road = cv2.remap(self.seg[0][i], suvi[..., 0], suvi[..., 1], cv2.INTER_NEAREST, borderValue=0)
            obj = cv2.remap(self.seg[1][i], suvi[..., 0], suvi[..., 1], cv2.INTER_NEAREST, borderValue=255)
            r[(road < 128) | (obj > 76)] = 0
        rr, cc = np.nonzero(r[::2, ::2])                   # 4 cm is enough for the correlation
        out = (np.c_[X1 - 2 * rr * RES, Y1 - 2 * cc * RES], r[::2, ::2][rr, cc])
        self.cache[i] = out
        return out


def calibrate_pitch(img_dir, poses, seg=None, mount=(-0.14, -1.88), offsets=(-1.0, -0.5, 0.0, 0.5, 1.0), log=print):
    """BEV pitch from lane-line parallelism: on straight stretches the lateral spacing of the markings 9..11 camera heights
    ahead must equal the spacing 3.5..6 ahead; a pitch error scales the far band (Track02: +2 % at the car-constraint pitch).
    Returns the mount (yaw, pitch) [deg] for which the far / near spacing ratio is 1 (linear interpolation)."""
    from scipy.spatial.transform import Rotation as R_
    U_ = nhc.axes(*np.radians(mount))[2]
    yaw = np.array([abs(R_.from_matrix(poses[i, :3, :3].T @ poses[min(i + 15, len(poses) - 1), :3, :3]).as_rotvec() @ U_) for i in range(len(poses))])
    straight = [i for i in range(20, len(poses) - 20, 15) if yaw[i] < np.radians(1.0) and yaw[i - 15] < np.radians(1.0)]
    bins = np.arange(-2, 2.001, 0.02)

    def ratio(mb, i):
        p, w = mb.points(i)
        near = (p[:, 0] > 3.5) & (p[:, 0] < 6); far = (p[:, 0] > 8.5) & (p[:, 0] < 11)
        if near.sum() < 150 or far.sum() < 150:
            return None
        hn = np.histogram(p[near, 1], bins, weights=w[near])[0]
        best = None
        for s_ in np.arange(0.85, 1.151, 0.005):
            hs = np.histogram(p[far, 1] * s_, bins, weights=w[far])[0]
            for sh in range(-5, 6):
                c = np.dot(np.roll(hs, sh), hn) / (np.linalg.norm(hs) * np.linalg.norm(hn) + 1e-9)
                if best is None or c > best[0]:
                    best = (c, s_)
        return best[1] if best[0] > 0.5 else None

    med = []
    for dp in offsets:
        mb = MarkingBEV(img_dir, mount=(mount[0], mount[1] + dp), seg=seg)
        r = [x for x in (ratio(mb, i) for i in straight) if x is not None]
        med.append(np.median(r) if len(r) > 20 else np.nan)
    med = np.array(med); ok = np.isfinite(med)
    if ok.sum() < 3:
        log("  BEV pitch calibration: too few straight frames with lane lines, keeping the car-constraint pitch")
        return mount
    dp = float(np.interp(1.0, med[ok][::-1], np.array(offsets)[ok][::-1]))
    log(f"  BEV pitch calibration ({len(straight)} straight frames): far/near spacing " + ", ".join(f"{o:+.1f}:{m:.3f}" for o, m in zip(offsets, med))
        + f" -> pitch offset {dp:+.2f} deg")
    return (mount[0], mount[1] + dp)


def road_pose(Tab, F, L):
    """(forward, left, yaw) of camera b in camera a's road frame"""
    t = Tab[:3, 3]
    hb = np.arctan2((Tab[:3, :3] @ F) @ L, (Tab[:3, :3] @ F) @ F)
    return t @ F, t @ L, hb


G, EXT = 0.04, (-1.0, 16.0, -6.0, 6.0)       # raster cell and extent (forward, lateral) in camera heights


def _raster(p, w):
    im = np.zeros((int((EXT[1] - EXT[0]) / G), int((EXT[3] - EXT[2]) / G)), np.float32)
    ix = ((p[:, 0] - EXT[0]) / G).astype(int); iy = ((p[:, 1] - EXT[2]) / G).astype(int)
    m = (ix >= 0) & (ix < im.shape[0]) & (iy >= 0) & (iy < im.shape[1])
    np.add.at(im, (ix[m], iy[m]), w[m])
    return cv2.GaussianBlur(im, (0, 0), 1.5)


CROSS = {}                                               # (a, b) -> (margin within the best yaw, margin against other yaws)


def register(mb, a, b, Tab, h, yaw_range=4.0, shift=3.0):
    """correction (d_forward, d_left [trajectory units], d_yaw [rad]) of b's pose in a, peak, margin; or None"""
    pa, wa = mb.points(a); pb, wb = mb.points(b)
    if len(pa) < mb.min_pts or len(pb) < mb.min_pts:
        return None
    xb, yb, hb = road_pose(Tab, mb.F, mb.L)
    xb, yb = xb / h, yb / h
    A = _raster(pa, wa)
    sh = int(shift / G)
    best = None
    peaks = []                                             # (score, yaw, location) of every yaw's best shift (diagnostic CROSS)
    for dy in np.radians(np.arange(-yaw_range, yaw_range + 0.01, 0.25)):
        c, s = np.cos(hb + dy), np.sin(hb + dy)
        q = np.c_[xb + c * pb[:, 0] - s * pb[:, 1], yb + s * pb[:, 0] + c * pb[:, 1]]
        res = cv2.matchTemplate(np.pad(_raster(q, wb), sh), A, cv2.TM_CCORR_NORMED)
        _, mx, _, lc = cv2.minMaxLoc(res)
        peaks.append((mx, dy, lc))
        if best is None or mx > best[0]:
            r2 = res.copy(); r2[max(0, lc[1] - 12):lc[1] + 13, max(0, lc[0] - 12):lc[0] + 13] = 0
            best = (mx, dy, lc, r2.max())
    mx, dy, lc, second = best
    # competing solution at any yaw: best peak whose shift differs by > 12 cells (same shift at a nearby yaw = the same solution)
    alt = max([p_[0] for p_ in peaks if max(abs(p_[2][0] - lc[0]), abs(p_[2][1] - lc[1])) > 12] + [second])
    CROSS[(a, b)] = (mx - second, mx - alt)
    # the template A sits at (lc - sh) in padded B: B's content must move by -(lc - sh) to overlap A
    dfx, dfy = -(lc[1] - sh) * G, -(lc[0] - sh) * G
    return dfx * h, dfy * h, dy, mx, mx - second


def unit_height(mb, poses, step=6, every=4):
    """camera height in trajectory units: marking motion between frames i, i + step (road plane, camera heights) against
    the trajectory's own step, median over the route (self-calibrates the BEV unit to the trajectory)"""
    r_ = []
    for i in range(0, len(poses) - step, every):
        Tij = np.linalg.inv(poses[i]) @ poses[i + step]
        xb, _, _ = road_pose(Tij, mb.F, mb.L)
        if xb <= 0.05 * step:
            continue
        r = register(mb, i, i + step, Tij, 1.0, yaw_range=0.5, shift=1.5)
        if r is None or r[3] < 0.5 or r[4] < 0.05:
            continue
        if xb + r[0] > 0.05:
            r_.append(xb / (xb + r[0]))
    return float(np.median(r_)) if len(r_) > 50 else None, len(r_)


def search(img_dir, poses, h, every=5, ahead=(14.0,), min_gap=150, log=print, mb=None, max_yaw=2.5):
    """corrections of opposite-pass pairs; keeps a pair only when its neighbours (a+-5) agree"""
    mb = mb or MarkingBEV(img_dir)
    C, Z = poses[:, :3, 3], poses[:, :3, 2]
    raw = {}
    for ah in np.atleast_1d(ahead):
        for a in range(0, len(poses), every):
            opp = np.where((Z @ Z[a] < -0.95) & (np.abs(np.arange(len(C)) - a) > min_gap))[0]
            if len(opp) == 0:
                continue
            loc = np.einsum("ij,nj->ni", poses[a, :3, :3].T, C[opp] - C[a])
            cost = np.abs(loc[:, 2] - ah * h) + np.abs(loc[:, 0])
            if cost.min() > 4 * h:
                continue
            b = int(opp[np.argmin(cost)])
            if (a, b) in raw:
                continue
            r = register(mb, a, b, np.linalg.inv(poses[a]) @ poses[b], h)
            if r is not None and r[3] > 0.35 and r[4] > 0.04 and abs(np.degrees(r[2])) < max_yaw:
                raw[(a, b)] = r + (float(ah),)
    global LAST_RAW
    LAST_RAW = raw                                       # every registration that passed peak / margin / yaw (for audit)
    keys = sorted(raw)
    out = {}
    for n, (a, b) in enumerate(keys):
        nb = [raw[k] for k in keys if k != (a, b) and abs(k[0] - a) <= every and raw[k][5] == raw[(a, b)][5]]
        r = raw[(a, b)]
        if any(abs(x[0] - r[0]) < 0.4 * h and abs(x[1] - r[1]) < 0.3 * h and abs(x[2] - r[2]) < np.radians(1.0) for x in nb):
            out[(a, b)] = r[:5]
    if out:
        v = np.array([x[:3] for x in out.values()])
        log(f"  marking loops (opposite passes): {len(raw)} registered, {len(out)} consistent with a neighbour; correction of the estimate "
            f"median forward {np.median(v[:, 0]):+.2f}, left {np.median(v[:, 1]):+.2f}, yaw {np.degrees(np.median(v[:, 2])):+.2f} deg")
    else:
        log(f"  marking loops (opposite passes): {len(raw)} registered, none consistent")
    return out, mb


LAST_RAW = {}
PUBLIC = ((1141, 1345), (2731, 2948))                    # Track02 public road (both passes), frame ranges of a


def audit(mb, h, hist, final, out, log=print, peak=0.35, margin=0.04, max_yaw=2.5):
    """The BEV stage is a sequence of rounds in which the current trajectory decides which pairs are found / kept and the
    kept pairs decide the next trajectory.  A shrinking median correction can mean the trajectory got better OR that the
    sample changed (only the easy pairs survive).  Here:
      - per round: pairs in the graph, new / kept / removed against the previous round
      - fixed pair sets (accepted in round 0; accepted in every round; union of all rounds) re-registered under the poses
        before every round and after the last one: forward / left correction, robust sd, fraction still passing
    hist: [(poses before the round, accepted {(a, b): r}, raw {(a, b): r + (ahead,)})], final: poses after the last round"""
    acc = [set(c) for _, c, _ in hist]
    rawk = [set(r) for _, _, r in hist]
    for r, s in enumerate(acc):
        line = f"  bev audit round {r}: raw {len(rawk[r])}, in graph {len(s)}"
        if r:
            line += (f"; graph pairs new {len(s - acc[r - 1])}, kept {len(s & acc[r - 1])}, removed {len(acc[r - 1] - s)}"
                     f" (removed but still registered raw {len((acc[r - 1] - s) & rawk[r])})")
        log(line)
    sets = {"round0": sorted(acc[0]), "every round": sorted(set.intersection(*acc)), "union": sorted(set.union(*acc))}
    P_ = [p for p, _, _ in hist] + [final]
    names = [f"before r{r}" for r in range(len(hist))] + ["after last"]
    ah = {}
    for _, _, raw in hist:
        ah.update({k: v[5] for k, v in raw.items()})
    pairs = sets["union"]
    M = np.full((len(P_), len(pairs), 5), np.nan)
    for j, P in enumerate(P_):
        for n, (a, b) in enumerate(pairs):
            r = register(mb, a, b, np.linalg.inv(P[a]) @ P[b], h)
            if r is not None:
                M[j, n] = r
    ok = (M[..., 3] > peak) & (M[..., 4] > margin) & (np.abs(np.degrees(M[..., 2])) < max_yaw)
    A = np.array([p[0] for p in pairs])
    pub = np.zeros(len(pairs), bool)
    for f0, f1 in PUBLIC:
        pub |= (A >= f0) & (A < f1)
    rsd = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan
    for nm, ks in sets.items():
        idx = np.array([pairs.index(k) for k in ks], int)
        for sub, m in (("all", np.ones(len(pairs), bool)), ("public", pub)):
            ii = idx[m[idx]]
            if len(ii) == 0:
                continue
            log(f"  bev audit fixed set '{nm}' ({sub}, {len(ii)} pairs): re-registered under")
            for j, name in enumerate(names):
                g = ii[ok[j, ii]]
                log(f"     {name:10s}: pass {len(g):3d}/{len(ii)}  forward median {np.median(M[j, g, 0]):+.3f} (|.| {np.median(np.abs(M[j, g, 0])):.3f}, "
                    f"sd {rsd(M[j, g, 0]):.3f})  left {np.median(M[j, g, 1]):+.3f} (|.| {np.median(np.abs(M[j, g, 1])):.3f}, sd {rsd(M[j, g, 1]):.3f})  "
                    f"yaw {np.degrees(np.median(M[j, g, 2])):+.2f} deg")
    np.savez(out, pairs=np.array(pairs), ahead=np.array([ah.get(k, np.nan) for k in pairs]), meas=M, ok=ok, names=np.array(names),
             in_round=np.array([[k in s for k in pairs] for s in acc]), raw_round=np.array([[k in s for k in pairs] for s in rawk]))
    log(f"  bev audit saved {out}")


def factors(poses, corr, F, L, sig=(0.3, 0.15, 0.5), center_fix=False):
    """between factors b in a: current relative pose with the road-plane correction applied (forward, left, yaw),
    sigmas forward / lateral [traj units] and yaw [deg]; height / roll / pitch left free.
    register() turns b's marking points about b's own centre and then shifts them, so the corrected position of b is
    t + dfx F + dfy L.  center_fix=False (before 11-20) also turned the baseline t about a's centre (R(dyaw) t), which adds a
    spurious sideways shift of |t| * dyaw (14 h * 1 deg = 0.24 h)."""
    out = []
    Uc = np.cross(F, L)
    for (a, b), (dfx, dfy, dyaw, _, _) in corr.items():
        Tab = np.linalg.inv(poses[a]) @ poses[b]
        Tm = Tab.copy()
        Rz = cv2.Rodrigues(Uc * dyaw)[0]
        Tm[:3, :3] = Rz @ Tab[:3, :3]
        Tm[:3, 3] = (Tab[:3, 3] if center_fix else Rz @ Tab[:3, 3]) + dfx * F + dfy * L
        # translation error of a Pose3 between factor is in frame b (b faces the other way: forward / left swap sign only)
        sg = np.r_[np.radians([5.0, 5.0, 5.0]), 10.0, 10.0, 10.0]
        Fb, Lb = Tm[:3, :3].T @ F, Tm[:3, :3].T @ L
        info = np.zeros((6, 6))
        info[:3, :3] = np.eye(3) / np.radians(5.0) ** 2
        Ub = Tm[:3, :3].T @ Uc
        info[:3, :3] += np.outer(Ub, Ub) * (1 / np.radians(sig[2]) ** 2 - 1 / np.radians(5.0) ** 2)
        info[3:, 3:] = np.eye(3) / 10.0 ** 2 + np.outer(Fb, Fb) / sig[0] ** 2 + np.outer(Lb, Lb) / sig[1] ** 2
        noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), gtsam.noiseModel.Gaussian.Information(info))
        out.append(gtsam.BetweenFactorPose3(a, b, gtsam.Pose3(Tm), noise))
    return out
