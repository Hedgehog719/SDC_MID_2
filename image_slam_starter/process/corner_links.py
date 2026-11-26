"""Do the two passes share image features where their headings differ by 30-150 deg (corners, intersections)?
Cached RootSIFT (result/feat_*), keypoints on road / cars / people dropped (seg.npz), mutual NN + ratio, RANSAC essential
matrix on undistorted rays.  For every frame pair of the two given frame ranges (step), with heading difference (camera
optical axes projected on the ground) in [lo, hi] and camera distance < dmax: inliers.  Also reports the inliers that a
random pair would give (frames far apart) as the floor.
usage: python image_slam_starter/process/corner_links.py TRAJ.csv A0 A1 B0 B1 [--step 5] [--lo 30] [--hi 150] [--dmax 25]"""
import argparse, glob, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("traj"); ap.add_argument("a0", type=int); ap.add_argument("a1", type=int); ap.add_argument("b0", type=int); ap.add_argument("b1", type=int)
ap.add_argument("--step", type=int, default=5); ap.add_argument("--lo", type=float, default=30); ap.add_argument("--hi", type=float, default=150)
ap.add_argument("--dmax", type=float, default=25.0)
a = ap.parse_args()
T2 = "data/Track2"
CAM = (1037.0, 1035.38, 723.93, 463.29, -0.055911, 0.085266, 0, 0)
Kc = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]]); DIST = np.array(CAM[4:8] + (0.0,))
names = sorted(os.listdir(f"{T2}/data/images")); FEAT = glob.glob(f"{T2}/result/feat_*")[0]; SEG = np.load(f"{T2}/seg.npz")
P = np.loadtxt(a.traj, delimiter=",", skiprows=1)[:, 1:].reshape(-1, 4, 4)
up = -P[:, :3, 1].mean(0); up /= np.linalg.norm(up)
_cache = {}


def feats(i):
    if i not in _cache:
        z = np.load(f"{FEAT}/{names[i][:-4]}.npz"); kp, d = z["kp"].astype(np.float64), z["desc"].astype(np.float32)
        u = np.clip((kp[:, 0] / 4).astype(int), 0, 359); v = np.clip((kp[:, 1] / 4).astype(int), 0, 231)
        keep = (SEG["road"][i][v, u] < 128) & (SEG["obj"][i][v, u] < 128)
        _cache[i] = (kp[keep], d[keep], cv2.undistortPoints(kp[keep].reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2))
    return _cache[i]


BF = cv2.BFMatcher(cv2.NORM_L2)


def inliers(i, j, ratio=0.8):
    ka, da, ra = feats(i); kb, db, rb = feats(j)
    m1 = BF.knnMatch(da, db, k=2); back = {m.queryIdx: m.trainIdx for m in BF.match(db, da)}
    pr = np.array([(m.queryIdx, m.trainIdx) for m, n in m1 if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx]).reshape(-1, 2)
    if len(pr) < 8:
        return 0, len(pr)
    E, ok = cv2.findEssentialMat(ra[pr[:, 0]], rb[pr[:, 1]], np.eye(3), cv2.RANSAC, 0.9999, 1.0 / CAM[0])
    return (int(ok.sum()) if ok is not None else 0), len(pr)


def heading(i):
    z = P[i][:3, 2]; z = z - (z @ up) * up; return z / np.linalg.norm(z)


rows = []
for i in range(a.a0, a.a1, a.step):
    for j in range(a.b0, a.b1, a.step):
        dpsi = np.degrees(np.arccos(np.clip(heading(i) @ heading(j), -1, 1)))
        d = np.linalg.norm(P[i][:3, 3] - P[j][:3, 3])
        if a.lo <= dpsi <= a.hi and d < a.dmax:
            n, nr = inliers(i, j)
            rows.append((i, j, dpsi, d, n, nr))
rows = np.array(rows)
rng = np.random.default_rng(0)
floor = [inliers(int(i), int(j))[0] for i, j in zip(rng.integers(0, 1000, 20), rng.integers(2000, 4200, 20))]
print(f"{len(rows)} pairs with heading difference {a.lo}-{a.hi} deg and distance < {a.dmax}; random-pair inlier floor: median {np.median(floor):.0f}, max {max(floor)}")
if len(rows):
    for lo, hi in ((30, 60), (60, 90), (90, 120), (120, 150)):
        m = (rows[:, 2] >= lo) & (rows[:, 2] < hi)
        if m.any():
            print(f"  heading {lo:3d}-{hi:3d}: {m.sum():3d} pairs, inliers median {np.median(rows[m, 4]):.0f}, p90 {np.percentile(rows[m, 4], 90):.0f}, max {rows[m, 4].max():.0f}, "
                  f">= 40: {(rows[m, 4] >= 40).sum()}")
    best = rows[np.argsort(-rows[:, 4])[:12]]
    for r in best:
        print(f"    {int(r[0])} - {int(r[1])}: heading {r[2]:5.1f} deg, distance {r[3]:5.1f}, inliers {int(r[4])} / {int(r[5])}")
