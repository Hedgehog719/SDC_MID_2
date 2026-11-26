"""Diagnostics (run from image_slam_starter/): camera height above the road along the route, as a local scale gauge.

The camera height is physically constant; in a bundle adjustment the triangulated road and the poses share the local
scale, so a segment the BA made too short has a smaller camera height there.  Per frame: robust plane through the
static points 2..8 units ahead in the own lane (as cam_height.py); per window: median.  A trajectory with fewer
local scale errors has a flatter profile.
usage: python process/diag/height_profile.py NAME=PTS.npz,TRAJ.csv ... [--win 100]"""
import argparse
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--win", type=int, default=100); ap.add_argument("--step", type=int, default=2)
a = ap.parse_args()


def heights(pts, traj):
    z = np.load(pts); X = z["xyz"].astype(float)
    if "obj" in z.files:
        X = X[z["obj"] < 0.2]
    T = np.genfromtxt(traj, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4); P = T[:, :3, 3]; N = len(P)
    tree = cKDTree(X); hs = np.full(N, np.nan)
    for i in range(0, N, a.step):
        R = T[i, :3, :3]; fwd = R[:, 2]; dn = R[:, 1]
        fh = fwd - (fwd @ dn) * dn; fh /= np.linalg.norm(fh); lft = np.cross(fh, dn)
        nb = tree.query_ball_point(P[i] + 5.0 * fh + 1.2 * dn, 4.0)
        if len(nb) < 30:
            continue
        Q = X[nb] - P[i]; f, l, d = Q @ fh, Q @ lft, Q @ dn
        m = (f > 2) & (f < 8) & (np.abs(l) < 0.7) & (d > 0.6) & (d < 1.8)
        if m.sum() < 25:
            continue
        A = np.c_[f[m], l[m], np.ones(m.sum())]; y = d[m]; w = np.ones(len(y))
        for _ in range(5):
            coef = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
            r = y - A @ coef; s = max(1.4826 * np.median(np.abs(r)), 0.01); w = 1 / np.maximum(1, np.abs(r) / (2 * s))
        hs[i] = coef[2] / np.linalg.norm([-coef[0], -coef[1], 1.0])
    return hs


prof = {}
for run in a.runs:
    name, files = run.split("="); pts, traj = files.split(",")
    prof[name] = heights(pts, traj)
N = len(next(iter(prof.values())))
edges = np.arange(0, N + a.win, a.win)
print("window   " + "  ".join(f"{n:>22s}" for n in prof))
rows = []
for e0, e1 in zip(edges[:-1], edges[1:]):
    vals = []
    for n, h in prof.items():
        seg = h[e0:e1]; seg = seg[~np.isnan(seg)]
        vals.append((np.median(seg), len(seg)) if len(seg) >= 8 else (np.nan, len(seg)))
    rows.append([v[0] for v in vals])
    print(f"{e0:4d}-{e1:4d} " + "  ".join(f"{v[0]:16.4f} ({v[1]:3d})" for v in vals))
M = np.array(rows); ok = ~np.isnan(M).any(1)
print("windows with all runs:", ok.sum())
for k, n in enumerate(prof):
    v = M[ok, k] / np.median(M[ok, k])
    print(f"{n:24s} median h {np.median(M[ok, k]):.4f}; window spread (robust sd) {1.4826 * np.median(np.abs(v - 1)) * 100:.2f} %, rms {np.sqrt(((v - 1) ** 2).mean()) * 100:.2f} %")
