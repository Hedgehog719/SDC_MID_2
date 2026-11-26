"""Cross-pass correspondences with RoMa (dense matcher, large viewpoint changes): for frame pairs of the two passes
(heading difference lo..hi, distance < dmax on the trajectory), RoMa outdoor matches, sampled, undistorted with the
calibrated lens, 5-point RANSAC (1 px); inliers vs the floor of unrelated pairs.
usage: python image_slam_starter/process/roma_cross.py TRAJ.csv A0 A1 B0 B1 [--step 10] [--lo 30] [--hi 150] [--dmax 25] [--n 3000] [--save out.npz]"""
import argparse, os
import cv2
import numpy as np
import torch
from romatch import roma_outdoor

ap = argparse.ArgumentParser()
ap.add_argument("traj"); ap.add_argument("a0", type=int); ap.add_argument("a1", type=int); ap.add_argument("b0", type=int); ap.add_argument("b1", type=int)
ap.add_argument("--step", type=int, default=10); ap.add_argument("--lo", type=float, default=30); ap.add_argument("--hi", type=float, default=150)
ap.add_argument("--dmax", type=float, default=25.0); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--save", default=None)
a = ap.parse_args()
T2 = "data/Track2"
CAM = (1037.0, 1035.38, 723.93, 463.29, -0.055911, 0.085266, 0, 0)
Kc = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]]); DIST = np.array(CAM[4:8] + (0.0,))
names = sorted(os.listdir(f"{T2}/data/images"))
SEG = np.load(f"{T2}/seg.npz")


def static_ok(i, kp):
    """not on the hood (bottom rows), the road, cars / people (quarter-res segmentation), not the sky"""
    u = np.clip((kp[:, 0] / 4).astype(int), 0, 359); v = np.clip((kp[:, 1] / 4).astype(int), 0, 231)
    return (kp[:, 1] < 895) & (SEG["road"][i][v, u] < 128) & (SEG["obj"][i][v, u] < 128) & (kp[:, 1] > 15)
P = np.loadtxt(a.traj, delimiter=",", skiprows=1)[:, 1:].reshape(-1, 4, 4)
up = -P[:, :3, 1].mean(0); up /= np.linalg.norm(up)
dev = "cuda"
model = roma_outdoor(device=dev, coarse_res=560, upsample_res=(616, 952), use_custom_corr=False)
W, H = 1440, 928


def heading(i):
    z = P[i][:3, 2]; z = z - (z @ up) * up; return z / np.linalg.norm(z)


def match(i, j):
    fa, fb = f"{T2}/data/images/{names[i]}", f"{T2}/data/images/{names[j]}"
    warp, cert = model.match(fa, fb, device=dev)
    m, c = model.sample(warp, cert, num=a.n)
    ka, kb = model.to_pixel_coordinates(m, H, W, H, W)
    ka, kb = ka.cpu().numpy().astype(np.float64), kb.cpu().numpy().astype(np.float64)
    f_ = static_ok(i, ka) & static_ok(j, kb) & (np.linalg.norm(ka - kb, axis=1) > 20)
    ka, kb = ka[f_], kb[f_]
    if len(ka) < 30:
        return 0, float(c.mean()), ka[:0], kb[:0]
    ra = cv2.undistortPoints(ka.reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2); rb = cv2.undistortPoints(kb.reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2)
    E, ok = cv2.findEssentialMat(ra, rb, np.eye(3), cv2.RANSAC, 0.9999, 1.0 / CAM[0])
    n = int(ok.sum()) if ok is not None else 0
    return n, float(c.mean()), ka[ok[:, 0] > 0] if n else ka[:0], kb[ok[:, 0] > 0] if n else kb[:0]


rows, keep = [], {}
for i in range(a.a0, a.a1, a.step):
    for j in range(a.b0, a.b1, a.step):
        dpsi = np.degrees(np.arccos(np.clip(heading(i) @ heading(j), -1, 1)))
        d = np.linalg.norm(P[i][:3, 3] - P[j][:3, 3])
        if a.lo <= dpsi <= a.hi and d < a.dmax:
            n, c, ka, kb = match(i, j)
            rows.append((i, j, dpsi, d, n, c)); keep[(i, j)] = (ka, kb)
            print(f"  {i}-{j}: heading {dpsi:5.1f}, dist {d:5.1f}, inliers {n:4d} / {a.n}, certainty {c:.3f}", flush=True)
rng = np.random.default_rng(0)
fl = [match(int(i), int(j))[:2] for i, j in zip(rng.integers(0, 900, 6), rng.integers(3300, 4200, 6))]
print(f"floor (unrelated pairs): inliers {[f[0] for f in fl]}, certainty {[round(f[1], 3) for f in fl]}")
if a.save:
    np.savez(a.save, rows=np.array(rows), **{f"{i}_{j}_a": v[0] for (i, j), v in keep.items()}, **{f"{i}_{j}_b": v[1] for (i, j), v in keep.items()})
