"""Image points (walls, doors, corners; not the road) between two cameras of opposite passes, and the along-road
residual of a trajectory at them.

1. cross-pass matches a <-> b: cached RootSIFT (result/feat_*), keypoints on road / cars / people dropped (seg.npz),
   mutual NN + ratio, RANSAC essential matrix on undistorted rays (calibrated camera). Pixels saved.
2. each pass triangulates the matched points on its own: a with a + da, b with b + db (same pass, the frame the car
   reaches next), using the trajectory's poses -> P1 (pass of a), P2 (pass of b) in world coordinates.
3. residual P2 - P1 in a's camera axes: forward (= along the road of a), lateral, up.  The median forward residual is
   how far the trajectory has slid the two passes apart along the road at this place (+ : pass-b points lie further
   along a's forward than pass-a points).

usage: python process/opp_pair_points.py TRAJ.csv a b [da db] [out.npz]
"""
import glob
import os
import sys
import cv2
import numpy as np

T2 = "data/Track2"
CAM = (1037.0, 1035.38, 723.93, 463.29, -0.055911, 0.085266, 0, 0)
Kc = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]])
DIST = np.array(CAM[4:8] + (0.0,))
H = 1.178

names = sorted(os.listdir(f"{T2}/data/images"))
FEAT = glob.glob(f"{T2}/result/feat_*")[0]
SEG = np.load(f"{T2}/seg.npz")


def feats(i):
    z = np.load(f"{FEAT}/{names[i][:-4]}.npz")
    kp, d = z["kp"].astype(np.float64), z["desc"].astype(np.float32)
    u = np.clip((kp[:, 0] / 4).astype(int), 0, 359); v = np.clip((kp[:, 1] / 4).astype(int), 0, 231)
    keep = (SEG["road"][i][v, u] < 128) & (SEG["obj"][i][v, u] < 128)
    return kp[keep], d[keep]


def rays(kp):
    return cv2.undistortPoints(kp.reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2)


def match(fa, fb, ratio=0.8, thr_px=1.0):
    (ka, da), (kb, db) = fa, fb
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m1 = bf.knnMatch(da, db, k=2); m2 = bf.match(db, da)
    back = {m.queryIdx: m.trainIdx for m in m2}
    pr = np.array([(m.queryIdx, m.trainIdx) for m, n in m1 if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx])
    if len(pr) < 8:
        return pr.reshape(-1, 2), 0
    ra, rb = rays(ka[pr[:, 0]]), rays(kb[pr[:, 1]])
    E, ok = cv2.findEssentialMat(ra, rb, np.eye(3), cv2.RANSAC, 0.9999, thr_px / CAM[0])
    n_raw = len(pr)
    return (pr[ok[:, 0] > 0] if ok is not None else pr[:0]), n_raw


def tri(P, i, j, xi, xj):
    """DLT with world-from-camera poses; returns world points and the parallax angle [deg]"""
    out, par = [], []
    for u, v in zip(xi, xj):
        A = []
        for T, x in ((P[i], u), (P[j], v)):
            Tcw = np.linalg.inv(T)[:3]
            A += [x[0] * Tcw[2] - Tcw[0], x[1] * Tcw[2] - Tcw[1]]
        X = np.linalg.svd(np.array(A))[2][-1]; X = X[:3] / X[3]
        r1, r2 = X - P[i][:3, 3], X - P[j][:3, 3]
        out.append(X); par.append(np.degrees(np.arccos(r1 @ r2 / np.linalg.norm(r1) / np.linalg.norm(r2))))
    return np.array(out), np.array(par)


def main(traj, a, b, da=8, db=8, out=None):
    a, b, da, db = int(a), int(b), int(da), int(db)
    P = np.loadtxt(traj, delimiter=",", skiprows=1)[:, 1:].reshape(-1, 4, 4)
    F = {i: feats(i) for i in (a, b, a + da, b + db)}
    mab, nraw = match(F[a], F[b])
    d = P[b][:3, 3] - P[a][:3, 3]
    print(f"{os.path.basename(traj)}  a {a}  b {b}: b in a forward {d @ P[a][:3, 2] / H:+.1f} h, lateral {d @ P[a][:3, 0] / H:+.2f} h, "
          f"heading off antiparallel {180 - np.degrees(np.arccos(P[a][:3, 2] @ P[b][:3, 2])):.1f} deg")
    print(f"   cross-pass: {len(F[a][0])} / {len(F[b][0])} keypoints off road / cars, {nraw} mutual-ratio matches, {len(mab)} essential inliers")
    if len(mab) < 8:
        return
    res = {}
    for side, (i, j, col) in (("a", (a, a + da, 0)), ("b", (b, b + db, 1))):
        mij, _ = match(F[i], F[j])
        lut = dict(zip(mij[:, 0], mij[:, 1]))
        sel = [(n, lut[k]) for n, k in enumerate(mab[:, col]) if k in lut]
        if not sel:
            res[side] = ({}, None)
            continue
        n_, kj = np.array(sel).T
        X, par = tri(P, i, j, rays(F[i][0][mab[n_, col]]), rays(F[j][0][kj]))
        z = (X - P[i][:3, 3]) @ P[i][:3, 2]
        good = (par > 1.0) & (z > 0) & (z < 40 * H)
        res[side] = (dict(zip(n_[good], X[good])), (len(sel), good.sum()))
        print(f"   pass of {side} ({i} + {j}): tracked {len(sel)}, triangulated (parallax > 1 deg, 0 < depth < 40 h) {good.sum()}")
    common = sorted(set(res["a"][0]) & set(res["b"][0]))
    print(f"   points triangulated by both passes: {len(common)}")
    if len(common) >= 3:
        R = np.array([res["b"][0][n] - res["a"][0][n] for n in common])
        f_, l_, u_ = R @ P[a][:3, 2], R @ P[a][:3, 0], R @ -P[a][:3, 1]
        rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
        print(f"   residual P_b - P_a in a's axes [traj units]: forward median {np.median(f_):+.3f} (robust sd {rs(f_):.3f}, "
              f"sem {rs(f_) / np.sqrt(len(f_)):.3f}), lateral {np.median(l_):+.3f} (sd {rs(l_):.3f}), up {np.median(u_):+.3f} (sd {rs(u_):.3f})")
        Xa = np.array([res["a"][0][n] for n in common])
        print(f"   points from a: forward {np.percentile((Xa - P[a][:3, 3]) @ P[a][:3, 2] / H, [10, 50, 90]).round(1)} h, "
              f"lateral {np.percentile((Xa - P[a][:3, 3]) @ P[a][:3, 0] / H, [10, 50, 90]).round(1)} h")
    if out:
        np.savez(out, a=a, b=b, px_a=F[a][0][mab[:, 0]], px_b=F[b][0][mab[:, 1]], common=np.array(common, int),
                 Xa=np.array([res["a"][0][n] for n in common]).reshape(-1, 3), Xb=np.array([res["b"][0][n] for n in common]).reshape(-1, 3))
        print("   saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
