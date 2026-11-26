"""Gravity direction of every image from vertical structures (no IMU): line segments (LSD) on the k1-undistorted image,
near-vertical ones -> vertical vanishing direction by RANSAC + least squares (each segment's interpretation-plane
normal n must be orthogonal to it).  Output <track>/gravity.npz: up (N, 3) in camera coordinates (NaN when too few
lines), number of inlier lines, residual.  Compared with v10 when available.
usage: python process/gravity_vp.py Track1 [--k1 -0.03] [--procs 16]"""
import argparse
import glob
import json
import multiprocessing as mp
import os

import cv2
import numpy as np

cv2.setNumThreads(1)
ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("--k1", type=float, default=-0.03)
ap.add_argument("--procs", type=int, default=16)
ap.add_argument("--min-len", type=float, default=40.0, help="minimum segment length [px]")
ap.add_argument("--max-tilt", type=float, default=20.0, help="segments within this angle of the image vertical [deg]")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, a.target, "data")
IMG = sorted(glob.glob(f"{D}/images/*.jpg"))
cam = json.load(open(f"{D}/camera_info.json"))
K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
DIST = np.array([a.k1, 0, 0, 0, 0.0])


def frame(i):
    g = cv2.imread(IMG[i], cv2.IMREAD_GRAYSCALE)
    lsd = cv2.createLineSegmentDetector()
    L = lsd.detect(g)[0]
    if L is None:
        return i, None
    L = np.asarray(L).reshape(-1, 4)
    d = L[:, 2:] - L[:, :2]
    ln = np.linalg.norm(d, axis=1)
    tilt = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    s = (ln > a.min_len) & (tilt < a.max_tilt)
    if s.sum() < 6:
        return i, None
    L, ln = L[s], ln[s]
    p = cv2.undistortPoints(L.reshape(-1, 1, 2).astype(np.float64), K, DIST).reshape(-1, 2, 2)
    x1 = np.c_[p[:, 0], np.ones(len(p))]; x2 = np.c_[p[:, 1], np.ones(len(p))]
    n = np.cross(x1, x2); n /= np.linalg.norm(n, axis=1)[:, None]
    w = ln
    rng = np.random.default_rng(i)
    best, bi = None, None
    for _ in range(300):
        j = rng.choice(len(n), 2, replace=False)
        v = np.cross(n[j[0]], n[j[1]])
        if np.linalg.norm(v) < 1e-6:
            continue
        v /= np.linalg.norm(v)
        r = np.abs(n @ v)                                  # sin of the angle between the segment plane and v
        inl = r < np.sin(np.radians(0.5))
        sc = w[inl].sum()
        if best is None or sc > best:
            best, bi = sc, inl
    if bi is None or bi.sum() < 6:
        return i, None
    for _ in range(3):                                     # refine: weighted least squares, re-select inliers
        M = (n[bi] * w[bi, None]).T @ n[bi]
        v = np.linalg.eigh(M)[1][:, 0]
        bi = np.abs(n @ v) < np.sin(np.radians(0.3))
        if bi.sum() < 6:
            return i, None
    v = v * np.sign(-v[1])                                 # up = -y (camera y points down)
    return i, (v, int(bi.sum()), float(np.degrees(np.median(np.abs(n[bi] @ v)))))


with mp.get_context("fork").Pool(a.procs) as pool:
    res = dict(pool.map(frame, range(len(IMG)), chunksize=8))
N = len(IMG)
up = np.full((N, 3), np.nan); nl = np.zeros(N, int); rr = np.full(N, np.nan)
for i, r in res.items():
    if r is not None:
        up[i], nl[i], rr[i] = r
np.savez(os.path.join(ROOT, a.target, "gravity.npz"), up=up, n_lines=nl, resid_deg=rr)
ok = np.isfinite(up[:, 0])
print(f"{ok.sum()} / {N} frames with a vertical vanishing direction, inlier lines median {np.median(nl[ok]):.0f}")
ref = os.path.join(ROOT, a.target, "result", "v10_vo_shared_it5.csv")
if os.path.exists(ref):
    import pandas as pd
    P = pd.read_csv(ref).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
    Vt = np.linalg.svd(P[:, :3, 3] - P[:, :3, 3].mean(0))[2]
    upw = Vt[2] * np.sign(Vt[2] @ -P[:, :3, 1].mean(0))
    upc = np.einsum("nji,j->ni", P[:, :3, :3], upw)        # v10 'up' (trajectory plane normal) in each camera
    ang = lambda a_, b_: np.degrees(np.arctan2(np.linalg.norm(np.cross(a_, b_), axis=1), np.einsum("ij,ij->i", a_, b_)))
    e = ang(up[ok], upc[ok])
    # pitch / roll components of the difference (camera x = right, z = forward)
    dp = np.degrees(np.arcsin(up[ok, 2]) - np.arcsin(upc[ok, 2])); dr = np.degrees(np.arcsin(up[ok, 0]) - np.arcsin(upc[ok, 0]))
    print(f"vs v10 up: angle median {np.median(e):.2f} deg, p90 {np.percentile(e, 90):.2f} | pitch diff median {np.median(dp):+.2f} (robust sd {1.4826*np.median(np.abs(dp-np.median(dp))):.2f}), "
          f"roll diff median {np.median(dr):+.2f} (robust sd {1.4826*np.median(np.abs(dr-np.median(dr))):.2f})")
    for a0, b0 in ((0, 473), (473, 830), (830, N)):
        o = ok[a0:b0]
        print(f"  frames {a0}-{b0}: {o.sum()} frames, pitch diff median {np.median(dp[(np.where(ok)[0] >= a0) & (np.where(ok)[0] < b0)]):+.2f}, "
              f"roll diff median {np.median(dr[(np.where(ok)[0] >= a0) & (np.where(ok)[0] < b0)]):+.2f}")
