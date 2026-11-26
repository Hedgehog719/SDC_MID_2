"""Diagnostics of 2026-11-20 (run from image_slam_starter/): camera height above the road in trajectory units, from points that
the images alone triangulated with the trajectory's poses (sfm_ba --fix-poses --dump-tracks), per route segment.

Per frame: static points (obj < 0.2) in the own lane ahead (2..8 units forward, |lateral| < 0.7) and 0.6..1.8 units below the
camera; a robust plane through them; height = distance camera -> plane.  Independent of the centre line and of the BEV unit.
usage: python process/diag/cam_height.py PTS.npz TRAJ.csv"""
import sys
import numpy as np
from scipy.spatial import cKDTree

z = np.load(sys.argv[1]); X = z["xyz"].astype(float); obj = z["obj"] if "obj" in z.files else np.zeros(len(X))
X = X[obj < 0.2]
T = np.genfromtxt(sys.argv[2], delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4); P = T[:, :3, 3]; N = len(P)
tree = cKDTree(X)
hs = np.full(N, np.nan)
for i in range(0, N, 3):
    R = T[i, :3, :3]; fwd = R[:, 2]; dn = R[:, 1]                       # camera axes in the world (z forward, y down)
    fwd_h = fwd - (fwd @ dn) * dn; fwd_h /= np.linalg.norm(fwd_h); lft = np.cross(fwd_h, dn)
    nb = tree.query_ball_point(P[i] + 5.0 * fwd_h + 1.2 * dn, 4.0)
    if len(nb) < 30:
        continue
    Q = X[nb] - P[i]
    f, l, d = Q @ fwd_h, Q @ lft, Q @ dn
    m = (f > 2) & (f < 8) & (np.abs(l) < 0.7) & (d > 0.6) & (d < 1.8)
    if m.sum() < 25:
        continue
    A = np.c_[f[m], l[m], np.ones(m.sum())]; y = d[m]
    w = np.ones(len(y))
    for _ in range(5):                                                  # robust plane d = a f + b l + c
        coef = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        r = y - A @ coef; s = max(1.4826 * np.median(np.abs(r)), 0.01); w = 1 / np.maximum(1, np.abs(r) / (2 * s))
    nrm = np.array([-coef[0], -coef[1], 1.0]); hs[i] = coef[2] / np.linalg.norm(nrm)  # distance from the camera (origin) to the plane
SEG = [(0, 200), (200, 800), (800, 1004), (1004, 1076), (1076, 1136), (1136, 1206), (1206, 1700), (1700, 2400), (2400, 2881), (2881, 2941), (2941, 3025), (3025, 3100), (3100, 3450), (3450, 4206)]
ok = ~np.isnan(hs)
print(f"frames with a road plane: {ok.sum()}; camera height median {np.nanmedian(hs):.4f} units (IQR {np.nanpercentile(hs,25):.3f}-{np.nanpercentile(hs,75):.3f})")
print("   " + "  ".join(f"{a}-{b}: {np.nanmedian(hs[a:b]):.3f}({np.sum(ok[a:b])})" for a, b in SEG if np.sum(ok[a:b]) >= 5))
