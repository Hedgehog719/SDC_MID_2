"""Pole landmarks (process/segment_poles.py) triangulated from bearings only, separately by each pass, then compared.

A pole looks the same from both directions, and its position on the road plane follows from the bearings of one pass
alone (the trajectory's positions + the undistorted base pixel's horizontal direction; no cloud depth, no road
markings, no appearance matching between passes).  Per pass: detections -> rays in the road plane; rough range from the
base pixel's angle below the horizon groups the detections of one pole; the pole = least-squares intersection of its
rays (>= min_obs detections, bearing spread >= min_spread deg).  Poles of the two passes within `match` units are the
same pole; their difference E - W in the leg's road frame is the passes' relative offset there.
Check: the same with odd / even frames of one pass (must be ~0).
usage: python process/pole_landmarks.py TRAJ.csv [TRAJ2.csv ...]"""
import argparse, os, sys
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("trajs", nargs="+"); ap.add_argument("--min-obs", type=int, default=5); ap.add_argument("--min-spread", type=float, default=10.0)
ap.add_argument("--group", type=float, default=2.0, help="grouping radius of the rough positions [units]")
ap.add_argument("--match", type=float, default=1.5); ap.add_argument("--h", type=float, default=1.179)
a = ap.parse_args()
CAM = (1035.5, 1033.88, 722.17, 456.84, -0.05833, 0.088413, 0, 0)
Kc = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]]); DIST = np.array(CAM[4:8] + (0.0,))
z = np.load("Track2/result/poles.npz")
PASSES = {"W": (990, 1220), "E": (2870, 3110)}
LEGS = {"NS": ((1004, 1076), (3025, 3100)), "corner": ((1076, 1136), (2941, 3025)), "EW": ((1136, 1206), (2881, 2941))}


def rays(P, frames):
    m = np.isin(z["frame"], frames)
    f = z["frame"][m]; px = np.c_[z["xb"][m], z["yb"][m]]
    n = cv2.undistortPoints(px.reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2)
    up = -P[f, :3, 1]                                          # camera up in the world (per frame)
    rc = np.c_[n, np.ones(len(n))]                             # camera-frame ray of the base pixel
    rw = np.einsum("nij,nj->ni", P[f, :3, :3], rc)
    upm = up.mean(0); upm /= np.linalg.norm(upm)
    hor = rw - np.outer(rw @ upm, upm); d = hor / np.linalg.norm(hor, axis=1, keepdims=True)
    below = -(rw @ upm) / np.linalg.norm(hor, axis=1)          # tan(angle below the horizon)
    rng = np.where(below > 0.02, a.h / np.maximum(below, 1e-3), np.nan)
    C = P[f, :3, 3]
    return f, C, d, rng, upm


def triangulate(P, frames):
    """image-space tracking (x moves smoothly, similar height, consecutive frames up to 2 apart), then the least-squares
    intersection of each track's bearing rays in the road plane"""
    f, C, d, rng, upm = rays(P, frames)
    m = np.isin(z["frame"], frames)
    xb, hh = z["xb"][m], z["height"][m]
    order = np.argsort(f, kind="stable"); f, C, d, xb, hh = f[order], C[order], d[order], xb[order], hh[order]
    tracks, active = [], []                                    # active: (track index, last idx)
    for i in range(len(f)):
        best, bd = None, 1e9
        for t, j in active:
            if 0 < f[i] - f[j] <= 2 and abs(xb[i] - xb[j]) < 40 * (f[i] - f[j]) and 0.6 < hh[i] / hh[j] < 1.6:
                dd = abs(xb[i] - xb[j])
                if dd < bd:
                    best, bd = t, dd
        if best is None:
            tracks.append([i]); active.append((len(tracks) - 1, i))
        else:
            tracks[best].append(i); active = [(t, (i if t == best else j)) for t, j in active]
        active = [(t, j) for t, j in active if f[i] - f[j] <= 2]
    poles = []
    for s in tracks:
        s = np.array(s)
        if len(np.unique(f[s])) < a.min_obs:
            continue
        e1 = d[s][0]; e2 = np.cross(upm, e1)
        ang = np.degrees(np.arctan2(d[s] @ e2, d[s] @ e1))
        if ang.max() - ang.min() < a.min_spread:
            continue
        A = np.zeros((3, 3)); bvec = np.zeros(3)
        for c_, d_ in zip(C[s], d[s]):
            M = np.eye(3) - np.outer(d_, d_); M = M - np.outer(M @ upm, upm) - np.outer(upm, M @ upm) + np.outer(upm, upm) * (upm @ M @ upm)
            A += M; bvec += M @ c_
        A += np.outer(upm, upm) * 1e-6
        X = np.linalg.solve(A, bvec)
        fwd = [(X - c_) @ d_ for c_, d_ in zip(C[s], d[s])]
        if min(fwd) <= 0:
            continue
        res = [np.linalg.norm((np.eye(3) - np.outer(d_, d_)) @ (X - c_) - ((X - c_) @ upm) * upm) for c_, d_ in zip(C[s], d[s])]
        poles.append((X, len(s), float(np.median(res))))
    return poles, upm


def compare(PW, PE, P, legs=LEGS):
    out = {}
    for leg, ((w0, w1), _) in legs.items():
        Cw = P[w0:w1, :3, 3]; o = Cw.mean(0); upm = -P[w0:w1, :3, 1].mean(0); upm /= np.linalg.norm(upm)
        u = Cw[-1] - Cw[0]; u -= (u @ upm) * upm; u /= np.linalg.norm(u); l = np.cross(upm, u)
        dd = []
        for Xw, nw, rw in PW:
            if abs((Xw - o) @ u) > np.linalg.norm(Cw[-1] - Cw[0]) / 2 + 3 or abs((Xw - o) @ l) > 15:
                continue
            if not PE:
                continue
            hd = lambda v: np.linalg.norm(v - (v @ upm) * upm)          # horizontal distance (bearings fix no height)
            j = int(np.argmin([hd(Xe - Xw) for Xe, _, _ in PE]))
            if hd(PE[j][0] - Xw) < a.match:
                dd.append(((PE[j][0] - Xw) @ u, (PE[j][0] - Xw) @ l))
        out[leg] = np.array(dd).reshape(-1, 2)
    return out


for tf in a.trajs:
    P = np.genfromtxt(tf, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
    PW, _ = triangulate(P, np.arange(*PASSES["W"])); PE, _ = triangulate(P, np.arange(*PASSES["E"]))
    PWo, _ = triangulate(P, np.arange(*PASSES["W"])[0::2]); PWe, _ = triangulate(P, np.arange(*PASSES["W"])[1::2])
    print(f"{os.path.basename(tf)}: poles W {len(PW)}, E {len(PE)}; ray residual median W {np.median([r for _, _, r in PW]):.3f}, E {np.median([r for _, _, r in PE]):.3f} units")
    cx = compare(PW, PE, P); ch = compare(PWo, PWe, P)
    for leg in LEGS:
        d = cx[leg]; e = ch[leg]
        s = (f"  {leg:6s}: {len(d):2d} matched poles, E - W along median {np.median(d[:, 0]):+.3f}, lateral {np.median(d[:, 1]):+.3f} "
             f"(spread {1.4826 * np.median(np.abs(d[:, 0] - np.median(d[:, 0]))):.3f} / {1.4826 * np.median(np.abs(d[:, 1] - np.median(d[:, 1]))):.3f})") if len(d) else f"  {leg:6s}: no matched poles"
        s += (f" | W odd - even: {len(e)} poles, along {np.median(e[:, 0]):+.3f}, lateral {np.median(e[:, 1]):+.3f}" if len(e) else " | W odd - even: none")
        print(s)
