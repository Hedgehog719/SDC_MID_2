"""Forward motion from road markings (idea from Midterm01 bev_vo.py): an independent ruler for the local scale.

Every image is mapped onto the road plane (inverse perspective with the self-calibrated lens; road 1 unit below the
camera, mount pitch / yaw of the car constraint; 2 cm cells, 4..11 units ahead, +-2 units sideways = own lane), and
thin bright markings are kept (white top-hat / local contrast, as bev.lane_response).  For a pair (i, i + step) the
marking image of i (6..9 units ahead) is rotated by the relative yaw of the trajectory and searched in the image of
i + step: the shift is the forward / lateral motion in road-plane units (camera height = 1).  Pairs without enough
marking pixels, or with a weak / ambiguous peak, are skipped.

The ratio (marking motion) / (trajectory step) along the route is the trajectory's local scale error up to one constant
(unknown camera height, constant mount pitch); its variation is what the private score sees.  Errors: the car's own
pitch relative to the road (brake / slope change; ~9 % per degree at 7.5 units), non-planar road.
usage: python process/marking_odo.py Track2 traj.csv [traj2.csv ...] [--step 3]     (from image_slam_starter)"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, ".")
from utils import nhc

ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("trajs", nargs="+")
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--mount", default="-0.14,-1.88", help="mount yaw, pitch [deg] (car constraint calibration)")
ap.add_argument("--cam", default="1035.5,722.17,456.83,-0.05833,0.088413", help="RADIAL f, cx, cy, k1, k2 (process/sfm_ba.py)")
ap.add_argument("--fy-ratio", type=float, default=1038.55506 / 1040.18078)
ap.add_argument("--min-mark", type=int, default=400, help="marking pixels needed in the template")
ap.add_argument("--out", default=None)
ap.add_argument("--near", type=float, default=4.0, help="nearest road distance in the BEV")
ap.add_argument("--tpl", default="6,9", help="template range ahead in frame i")
a = ap.parse_args()

RES, X0, X1, Y1 = 0.02, a.near, 11.0, 2.0
TX0, TX1 = (float(v) for v in a.tpl.split(","))
f_, cx, cy, k1, k2 = (float(v) for v in a.cam.split(","))
K = np.array([[f_, 0, cx], [0, f_ * a.fy_ratio, cy], [0, 0, 1.0]])
D = np.array([k1, k2, 0, 0, 0.0])
yaw_m, pitch_m = (np.radians(float(v)) for v in a.mount.split(","))
F, L, U = nhc.axes(yaw_m, pitch_m)               # vehicle forward / left / up in camera coordinates
xs = np.arange(X1, X0, -RES)                     # rows: far -> near
ys = np.arange(Y1, -Y1, -RES)                    # cols: left -> right
gx, gy = np.meshgrid(xs, ys, indexing="ij")
P = gx[..., None] * F + gy[..., None] * L - 1.0 * U
uv = cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)
files = sorted(os.listdir(f"{a.target}/data/images"))


def bev(i):
    g = cv2.imread(f"{a.target}/data/images/{files[i]}", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    b = cv2.remap(g, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=-1)
    valid = b >= 0
    b[~valid] = np.median(b[valid])
    ks = int(0.5 / RES) | 1
    th = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
    sd = np.sqrt(np.maximum(cv2.blur(b * b, (61, 61)) - cv2.blur(b, (61, 61)) ** 2, 1.0))
    r = np.clip(th / sd, 0, 4)
    r[r < 1.5] = 0
    r[~valid] = 0
    return r.astype(np.float32)


def pair(job):
    i, j, dyaw = job
    A, B = bev(i), bev(j)
    r0, r1 = int((X1 - TX1) / RES), int((X1 - TX0) / RES)
    c0, c1 = int(0.6 / RES), A.shape[1] - int(0.6 / RES)
    # rotate the template about the vehicle origin (row of x = 0, centre column) by the yaw between the frames
    M = cv2.getRotationMatrix2D((A.shape[1] / 2, X1 / RES), -np.degrees(dyaw), 1.0)
    Ar = cv2.warpAffine(A, M, (A.shape[1], A.shape[0]))
    tpl = Ar[r0:r1, c0:c1]
    if (tpl > 0).sum() < a.min_mark:
        return None
    res = cv2.matchTemplate(B, tpl, cv2.TM_CCORR_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    # forward motion = rows the markings moved towards the car; lateral = columns
    y_, x_ = loc[1], loc[0]
    sub = 0.0
    if 0 < y_ < res.shape[0] - 1:          # parabolic sub-cell peak along the road
        c_, l_, h_ = res[y_, x_], res[y_ - 1, x_], res[y_ + 1, x_]
        den = l_ - 2 * c_ + h_
        sub = 0.5 * (l_ - h_) / den if den < 0 else 0.0
    dx = (loc[1] + sub - r0) * RES
    dy = (loc[0] - c0) * RES
    # ambiguity: second peak away from the first (dashes repeat)
    r2 = res.copy(); y_, x_ = loc[1], loc[0]
    r2[max(0, y_ - 25):y_ + 25, max(0, x_ - 25):x_ + 25] = 0
    second = r2.max()
    return i, j, dx, dy, float(mx), float(second), int((tpl > 0).sum())


if __name__ == "__main__":
    T0 = np.genfromtxt(a.trajs[0], delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
    N = len(T0)
    jobs = []
    for i in range(0, N - a.step, a.every):
        Rr = T0[i, :3, :3].T @ T0[i + a.step, :3, :3]
        jobs.append((i, i + a.step, float(R.from_matrix(Rr).as_rotvec() @ U)))
    out_f = a.out or f"{a.target}/result/marking_odo_s{a.step}_{a.near:g}_{a.tpl.replace(',', '-')}_v3.npy"
    if os.path.exists(out_f):
        res = np.load(out_f)
    else:
        with Pool(16) as pool:
            res = [r for r in pool.map(pair, jobs, chunksize=8) if r is not None]
        res = np.array(res)
        np.save(out_f, res)
    i, j, dx, dy, mx, sec, nm = res.T
    ok = (mx > 0.5) & (mx - sec > 0.05) & (dx > 0.05)
    print(f"{len(jobs)} pairs, {len(res)} with markings, {ok.sum()} accepted (peak > 0.5, unambiguous, moving)")
    i, j, dx = i[ok].astype(int), j[ok].astype(int), dx[ok]
    from scipy.ndimage import uniform_filter1d
    for t in a.trajs:
        T = np.genfromtxt(t, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
        st = np.array([(np.linalg.inv(T[p])[:3, :3] @ (T[q, :3, 3] - T[p, :3, 3])) @ F for p, q in zip(i, j)])
        lr = np.log(dx / np.maximum(st, 1e-6))
        g = np.isfinite(lr) & (st > 0.05)
        lr = lr - np.median(lr[g])
        prof = np.full(N, np.nan); prof[i[g]] = lr[g]
        okp = np.isfinite(prof)
        pi = np.interp(np.arange(N), np.where(okp)[0], prof[okp])
        # robust 101-frame moving median where there are markings
        sm = np.array([np.nanmedian(prof[max(0, k - 50):k + 51]) if np.isfinite(prof[max(0, k - 50):k + 51]).sum() >= 15 else np.nan for k in range(N)])
        gs = np.isfinite(sm)
        print(f"{os.path.basename(t):22s} per pair robust sd {100*1.4826*np.median(np.abs(lr[g])):.2f} %, 101-frame median: sd {100*np.nanstd(sm):.2f} % "
              f"({gs.sum()} frames covered)\n   every 300 [%]: " + " ".join("  nan" if not np.isfinite(sm[k]) else f"{100*sm[k]:+5.1f}" for k in range(0, N, 300)))
