"""Opposite-pass image points on a wall beside the road: each image is re-rendered by a virtual camera turned by
+-90 deg yaw about its own centre to face the wall (pure rotation, no road plane, no trajectory), so the wall is
near fronto-parallel in both views and the two views differ by a similarity; SIFT + ratio + RANSAC homography.
Matched points go back to rays / pixels of the real cameras.  Each pass then triangulates them on its own
(frame i with i + d of the same pass, trajectory poses) and P_b - P_a in a's axes is the trajectory's residual.

usage: python process/wall_rectify.py TRAJ.csv a side_a b side_b [d] [out_prefix]     side = R | L (wall to the right / left)
(d is unused since the multi-partner triangulation; kept for the command line)
"""
import os
import sys
import cv2
import numpy as np

T2 = "data/Track2"
CAM = (1037.0, 1035.38, 723.93, 463.29, -0.055911, 0.085266, 0, 0)
Kc = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]]); DIST = np.array(CAM[4:8] + (0.0,))
H = 1.178
FV, W, HH = 400.0, 2000, 1100                     # virtual camera; the wall the real camera sees lies 55-85 deg off the
OFF = 450                                         # virtual axis, so the principal point sits off-canvas (OFF px beyond the edge)


def KV(side):
    return np.array([[FV, 0, W + OFF if side == "R" else -OFF], [0, FV, HH / 2], [0, 0, 1.0]])
names = sorted(os.listdir(f"{T2}/data/images"))
SEG = np.load(f"{T2}/seg.npz")


def Rv(side):
    """real-from-virtual rotation: virtual z = real +x (R) or -x (L)"""
    s = 1 if side == "R" else -1
    return np.array([[0, 0, s], [0, 1, 0], [-s, 0, 0]], float)


def rectify(i, side):
    u, v = np.meshgrid(np.arange(W), np.arange(HH))
    r = np.linalg.inv(KV(side)) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    rr = Rv(side) @ r
    ok = rr[2] > 0.05
    px = np.full((2, u.size), -1.0)
    p, _ = cv2.projectPoints(rr[:, ok].T.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc, DIST)
    px[:, ok] = p.reshape(-1, 2).T
    mx, my = px[0].reshape(HH, W).astype(np.float32), px[1].reshape(HH, W).astype(np.float32)
    img = cv2.imread(f"{T2}/data/images/{names[i]}", cv2.IMREAD_GRAYSCALE)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0)
    # mask: inside the real image, not road / cars / people (quarter-res seg)
    seg_bad = ((SEG["road"][i] > 127) | (SEG["obj"][i] > 127)).astype(np.uint8) * 255
    bad = cv2.remap(cv2.resize(seg_bad, (1440, 928), interpolation=cv2.INTER_NEAREST), mx, my, cv2.INTER_NEAREST, borderValue=255)
    inside = (mx >= 0) & (mx < 1440) & (my >= 0) & (my < 928)
    mask = (inside & (bad < 128)).astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    return out, mask


SIFT = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.005, edgeThreshold=20)   # low-texture, blurred wall


def kpd(i, side):
    im, m = rectify(i, side)
    k, d = SIFT.detectAndCompute(im, m)
    d = d / (d.sum(1, keepdims=True) + 1e-9); d = np.sqrt(d).astype(np.float32)          # RootSIFT
    return np.array([x.pt for x in k]), d, im


def match(A, B, ratio=0.9, thr=6.0):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m1 = bf.knnMatch(A[1], B[1], k=2); back = {m.queryIdx: m.trainIdx for m in bf.match(B[1], A[1])}
    pr = np.array([(m.queryIdx, m.trainIdx) for m, n in m1 if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx]).reshape(-1, 2)
    if len(pr) < 8:
        return pr[:0], len(pr), None
    Hm, ok = cv2.findHomography(A[0][pr[:, 0]], B[0][pr[:, 1]], cv2.RANSAC, thr)
    return pr[ok[:, 0] > 0], len(pr), Hm


def match_E(A, B, side, ratio=0.85, thr_px=1.5):
    """same-pass pair: mutual NN + ratio, RANSAC essential matrix on the real-camera rays (all depths, not one plane)"""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m1 = bf.knnMatch(A[1], B[1], k=2); back = {m.queryIdx: m.trainIdx for m in bf.match(B[1], A[1])}
    pr = np.array([(m.queryIdx, m.trainIdx) for m, n in m1 if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx]).reshape(-1, 2)
    if len(pr) < 8:
        return pr[:0]
    E, ok = cv2.findEssentialMat(ray(A[0][pr[:, 0]], side)[:, :2], ray(B[0][pr[:, 1]], side)[:, :2], np.eye(3), cv2.RANSAC, 0.9999, thr_px / CAM[0])
    return pr[ok[:, 0] > 0]


def ray(pt, side):
    r = Rv(side) @ (np.linalg.inv(KV(side)) @ np.c_[pt, np.ones(len(pt))].T)
    return (r / r[2]).T                               # normalised real-camera rays (x/z, y/z, 1)


def pix(pt, side):
    r = Rv(side) @ (np.linalg.inv(KV(side)) @ np.c_[pt, np.ones(len(pt))].T)
    p, _ = cv2.projectPoints(r.T.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc, DIST)
    return p.reshape(-1, 2)


def tri(P, i, j, ri, rj):
    X, par = [], []
    for u, v in zip(ri, rj):
        A = []
        for T, x in ((P[i], u), (P[j], v)):
            Tcw = np.linalg.inv(T)[:3]
            A += [x[0] * Tcw[2] - Tcw[0], x[1] * Tcw[2] - Tcw[1]]
        x = np.linalg.svd(np.array(A))[2][-1]; x = x[:3] / x[3]
        a1, a2 = x - P[i][:3, 3], x - P[j][:3, 3]
        X.append(x); par.append(np.degrees(np.arccos(a1 @ a2 / np.linalg.norm(a1) / np.linalg.norm(a2))))
    return np.array(X), np.array(par)


def main(traj, a, sa, b, sb, d=6, out=None):
    a, b, d = int(a), int(b), int(d)
    P = np.loadtxt(traj, delimiter=",", skiprows=1)[:, 1:].reshape(-1, 4, 4)
    fa, fb = kpd(a, sa), kpd(b, sb)
    mab, nraw, Hab = match(fa, fb)
    s = np.linalg.svd(Hab[:2, :2])[1] if Hab is not None else (np.nan, np.nan)
    print(f"{os.path.basename(traj)} a {a}{sa} b {b}{sb}: {len(fa[0])} / {len(fb[0])} keypoints, {nraw} mutual-ratio, {len(mab)} homography inliers "
          f"(H scale {s[0]:.2f} / {s[1]:.2f})")
    if out and len(mab):
        vis = cv2.cvtColor(np.hstack([fa[2], fb[2]]), cv2.COLOR_GRAY2BGR)
        for p, q in zip(fa[0][mab[:, 0]], fb[0][mab[:, 1]]):
            cv2.line(vis, tuple(int(x) for x in p), (int(q[0]) + W, int(q[1])), (0, 255, 0), 1)
        cv2.imwrite(f"{out}_rect.jpg", cv2.resize(vis, (2000, 550)))
    if len(mab) < 8:
        return
    res = {}
    for nm, i, side, F, col in (("a", a, sa, fa, 0), ("b", b, sb, fb, 1)):
        acc = {}
        for j in [i + k for k in (-8, -6, -4, -2, 2, 4, 6, 8, 10)]:                  # same-pass partners
            fj = kpd(j, side)
            mij = match_E(F, fj, side)
            lut = dict(zip(mij[:, 0], mij[:, 1]))
            sel = [(n, lut[k]) for n, k in enumerate(mab[:, col]) if k in lut]
            if not sel:
                continue
            n_, kj = np.array(sel).T
            X, par = tri(P, i, j, ray(F[0][mab[n_, col]], side), ray(fj[0][kj], side))
            lat = np.abs((X - P[i][:3, 3]) @ P[i][:3, 0])
            for n, x, ok in zip(n_, X, (par > 1.0) & (lat > 0.5 * H) & (lat < 30 * H)):
                if ok:
                    acc.setdefault(n, []).append(x)
        res[nm] = {n: np.median(np.array(v), 0) for n, v in acc.items()}
        nv = [len(v) for v in acc.values()]
        print(f"   pass of {nm} ({i}, partners -8..+10): {len(acc)} of {len(mab)} cross-pass points triangulated, views per point median {np.median(nv) if nv else 0:.0f}")
    common = sorted(set(res["a"]) & set(res["b"]))
    print(f"   points triangulated by both passes: {len(common)}")
    if len(common) >= 3:
        Xa = np.array([res["a"][n] for n in common]); Xb = np.array([res["b"][n] for n in common])
        R = Xb - Xa
        fw, la, up = R @ P[a][:3, 2], R @ P[a][:3, 0], R @ -P[a][:3, 1]
        rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
        print(f"   P_b - P_a in a's axes [traj units]: forward median {np.median(fw):+.3f} (robust sd {rs(fw):.3f}, sem {rs(fw) / np.sqrt(len(fw)):.3f}), "
              f"lateral {np.median(la):+.3f} (sd {rs(la):.3f}), up {np.median(up):+.3f} (sd {rs(up):.3f})")
        rel = Xa - P[a][:3, 3]
        print(f"   points (from a): forward {np.percentile(rel @ P[a][:3, 2] / H, [10, 50, 90]).round(1)} h, lateral {np.percentile(rel @ P[a][:3, 0] / H, [10, 50, 90]).round(1)} h, "
              f"up {np.percentile(rel @ -P[a][:3, 1] / H, [10, 50, 90]).round(1)} h")
    if out:
        np.savez(f"{out}.npz", a=a, b=b, px_a=pix(fa[0][mab[:, 0]], sa), px_b=pix(fb[0][mab[:, 1]], sb), common=np.array(common, int),
                 Xa=np.array([res["a"][n] for n in common]).reshape(-1, 3), Xb=np.array([res["b"][n] for n in common]).reshape(-1, 3))
        print("   saved", f"{out}.npz")


if __name__ == "__main__":
    main(*sys.argv[1:])
