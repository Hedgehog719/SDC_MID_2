"""Diagnostics of 2026-11-20 (run from image_slam_starter/): camera height in trajectory units from same-direction revisits only.

Two frames of the same direction at the same place: lateral offset between them = h * (y0_i - y0_j) (both measure the
same centre line on the same side), no assumption about which line an opposite pass sees.  Least squares with a free
offset per route segment would hide nothing here: report h with a bootstrap interval, overall and for the pairs with a
large y0 difference (where h is determined)."""
import sys
import numpy as np
from scipy.spatial import cKDTree

z = np.load("Track2/result/centerline2_0-4206_-0.21_-1.49.npz"); y0 = z["y0"] / np.sqrt(1 + z["k"] ** 2); ok = ~np.isnan(y0)
for fn in sys.argv[1:]:
    T = np.genfromtxt(fn, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4); P = T[:, :3, 3]; N = len(P)
    tree = cKDTree(P)
    up = np.zeros_like(P)
    for i in range(0, N, 5):
        nb = tree.query_ball_point(P[i], 40.0); Q = P[nb] - P[nb].mean(0); n = np.linalg.svd(Q, full_matrices=False)[2][2]
        up[i:i + 5] = n if n @ (-T[i, :3, 1]) > 0 else -n
    fw = np.gradient(P, axis=0); fw = np.array([fw[max(0, i - 3):i + 4].mean(0) for i in range(N)])
    fw -= (fw * up).sum(1)[:, None] * up; fw /= np.linalg.norm(fw, axis=1, keepdims=True) + 1e-12; lf = np.cross(up, fw)
    I, J = [], []
    for i in range(0, N, 2):
        for j in tree.query_ball_point(P[i], 2.0):
            if j - i > 150 and ok[i] and ok[j] and fw[i] @ fw[j] > np.cos(np.radians(10)):
                I.append(i); J.append(j)
    I, J = np.array(I), np.array(J)
    lat = ((P[J] - P[I]) * lf[I]).sum(1); dy = y0[I] - y0[J]
    rng = np.random.default_rng(0)
    for nm, m in (("all", np.ones(len(I), bool)), ("|dy| > 0.15 h", np.abs(dy) > 0.15), ("|dy| > 0.25 h", np.abs(dy) > 0.25)):
        if m.sum() < 20:
            continue
        hs = []
        idx = np.where(m)[0]
        # bootstrap over whole revisit blocks (pairs are strongly correlated along a pass): resample frame blocks of 20
        blocks = np.unique(I[idx] // 20)
        for _ in range(300):
            bb = rng.choice(blocks, len(blocks)); sel = np.concatenate([idx[I[idx] // 20 == b] for b in bb])
            hs.append((lat[sel] * dy[sel]).sum() / (dy[sel] ** 2).sum())
        h = (lat[idx] * dy[idx]).sum() / (dy[idx] ** 2).sum()
        print(f"{fn.split('/')[-1]:28s} {nm:14s}: n {m.sum():5d} ({len(blocks)} blocks), h = {h:.3f} (block bootstrap 5-95%: {np.percentile(hs,5):.3f}-{np.percentile(hs,95):.3f})")
