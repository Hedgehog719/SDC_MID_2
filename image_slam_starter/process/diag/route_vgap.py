"""Height gap between two passes at the same place over the whole Track02 route (SLAM stage vs BA stage).

Pairs: frames i < j with j - i > 150 and horizontal distance < 2.6 units (same road, either direction).
Height = component along the local up (mean camera -y of the two frames, corrected for the mount pitch by using the
plane of positions within 40 units).  Reports the gap per route segment of i.
"""
import sys
import numpy as np
from scipy.spatial import cKDTree

files = {"SLAM cz16_bu1035cf": "Track2/result/t2_cz16_bu1035cf.csv",
         "BA sfm_bu1035cf": "../submissions/track2/t2_sfm_bu1035cf.csv",
         "BA sfm_pp06z (rot fixed)": "../submissions/track2/t2_sfm_pp06z.csv",
         "SLAM dash1r": "../submissions/track2/t2_dash1r.csv"}
SEG = [(0, 200), (200, 800), (800, 1004), (1004, 1206), (1206, 1700), (1700, 2400), (2400, 2881), (2881, 3100), (3100, 3450), (3450, 4000), (4000, 4206)]
for name, fn in files.items():
    T = np.genfromtxt(fn, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4); P = T[:, :3, 3]
    # local up: normal of the plane through positions within 40 units (oriented like camera -y)
    tree = cKDTree(P)
    ups = np.zeros_like(P)
    for i in range(0, len(P), 10):
        nb = tree.query_ball_point(P[i], 40.0); Q = P[nb] - P[nb].mean(0)
        n = np.linalg.svd(Q, full_matrices=False)[2][2]
        if n @ (-T[i, :3, 1]) < 0:
            n = -n
        ups[i:i + 10] = n
    pairs = tree.query_pairs(2.6, output_type="ndarray")
    pairs = pairs[np.abs(pairs[:, 1] - pairs[:, 0]) > 150]
    i, j = pairs.min(1), pairs.max(1)
    u = ups[i] + ups[j]; u /= np.linalg.norm(u, axis=1, keepdims=True)
    d = P[j] - P[i]; h = (d * u).sum(1); horiz = np.linalg.norm(d - h[:, None] * u, axis=1)
    keep = horiz < 2.6
    i, j, h = i[keep], j[keep], h[keep]
    # one value per frame i: the median over its partners
    out = []
    for a, b in SEG:
        m = (i >= a) & (i < b)
        if m.sum() < 20:
            out.append(f"{a}-{b}: -"); continue
        out.append(f"{a}-{b}: {np.median(h[m]):+.3f} (|.| {np.median(np.abs(h[m])):.3f}, n {m.sum()})")
    allabs = np.median(np.abs(h))
    print(f"== {name}: {len(h)} same-place pairs, median |height gap| {allabs:.3f} u, rms {np.sqrt((h**2).mean()):.3f} u")
    print("   " + "\n   ".join(out))
