"""Offset between the two passes of a road from multi-view triangulated points (sfm_ba.py --fix-poses --dump-tracks).

Every 3D point is triangulated from the images of ONE pass (its observing frames all lie in that pass), with the poses
of the trajectory, so its precision comes from the image geometry (parallax), not from a depth network.  Per leg, the
points of pass W and of pass E above the road (parked cars, trees, poles, walls, fences) give two top-view occupancy maps
in the road frame; their cross-correlation peak = where E's copy of the street sits relative to W's.
Check: the points of one pass split by index parity (must give 0).
usage: python process/pass_points_offset.py PTS.npz TRAJ.csv [--cars] [--smax 0.75]"""
import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("pts"); ap.add_argument("traj")
ap.add_argument("--cars", action="store_true", help="only points on road users (P > 0.5)")
ap.add_argument("--cell", type=float, default=0.1); ap.add_argument("--smax", type=float, default=0.75); ap.add_argument("--lmax", type=float, default=0.6)
ap.add_argument("--hmin", type=float, default=0.3, help="minimum height above the camera's road plane [units]")
a = ap.parse_args()
z = np.load(a.pts); X = z["xyz"].astype(np.float64); fmin, fmax, obj = z["fmin"], z["fmax"], z["obj"]
P = np.genfromtxt(a.traj, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
PASS = {"W": (990, 1220), "E": (2870, 3110)}
LEGS = {"NS": ((1004, 1076), (3025, 3100)), "corner": ((1076, 1136), (2941, 3025)), "EW": ((1136, 1206), (2881, 2941))}
inW = (fmin >= PASS["W"][0]) & (fmax < PASS["W"][1]); inE = (fmin >= PASS["E"][0]) & (fmax < PASS["E"][1])
sel = (obj > 0.5) if a.cars else np.ones(len(X), bool)


def frame_axes(w0, w1):
    C = P[w0:w1, :3, 3]; up = -P[w0:w1, :3, 1].mean(0); up /= np.linalg.norm(up)
    u = C[-1] - C[0]; u -= (u @ up) * up; u /= np.linalg.norm(u)
    return C, up, u, np.cross(up, u)


def occ(Q, o, u, l, ext):
    x, y = (Q - o) @ u, (Q - o) @ l
    nx, ny = int((ext[1] - ext[0]) / a.cell), int((ext[3] - ext[2]) / a.cell)
    g, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    return cv2.GaussianBlur(np.log1p(g).astype(np.float32), (0, 0), 1.0)


def corr(A, B):
    sx, sy = int(a.smax / a.cell), int(a.lmax / a.cell)
    res = cv2.matchTemplate(np.pad(B, ((sx, sx), (sy, sy))), A, cv2.TM_CCOEFF_NORMED)
    _, mx, _, lc = cv2.minMaxLoc(res)
    r2 = res.copy(); r2[max(0, lc[1] - 3):lc[1] + 4, max(0, lc[0] - 3):lc[0] + 4] = -1
    # sub-cell peak (parabola along both axes)
    def sub(v0, v1, v2):
        d = v0 - 2 * v1 + v2
        return 0.0 if abs(d) < 1e-9 else 0.5 * (v0 - v2) / d
    i, j = lc[1], lc[0]
    di = sub(res[i - 1, j], res[i, j], res[i + 1, j]) if 0 < i < res.shape[0] - 1 else 0.0
    dj = sub(res[i, j - 1], res[i, j], res[i, j + 1]) if 0 < j < res.shape[1] - 1 else 0.0
    return -(i + di - sx) * a.cell, -(j + dj - sy) * a.cell, mx, mx - r2.max()


print(f"{a.pts}: {'road users only' if a.cars else 'all points above the road'}; points of pass W {int((inW & sel).sum())}, E {int((inE & sel).sum())}")
for leg, ((w0, w1), _) in LEGS.items():
    C, up, u, l = frame_axes(w0, w1); o = C.mean(0)
    L = np.linalg.norm(C[-1] - C[0]) / 2 + 6
    ext = (-L, L, -10.0, 10.0)
    h = lambda Q: (Q - o) @ up - (C - o).mean(0) @ up + 1.18                  # height above the road (camera height 1.18)
    near = lambda m: m & sel & (np.abs((X - o) @ u) < L) & (np.abs((X - o) @ l) < 10)
    mW, mE = near(inW), near(inE)
    mW &= h(X) > a.hmin; mE &= h(X) > a.hmin
    QW, QE = X[mW], X[mE]
    da, dl, pk, mg = corr(occ(QW, o, u, l, ext), occ(QE, o, u, l, ext))
    idx = np.where(mW)[0]
    sa, sl, spk, smg = corr(occ(X[idx[0::2]], o, u, l, ext), occ(X[idx[1::2]], o, u, l, ext))
    print(f"  {leg:6s}: W {len(QW):5d} pts, E {len(QE):5d} pts; E->W shift along {da:+.3f}, lateral {dl:+.3f} units (peak {pk:.2f}, margin {mg:.2f})"
          f" | W parity {sa:+.3f} / {sl:+.3f} (peak {spk:.2f})")
