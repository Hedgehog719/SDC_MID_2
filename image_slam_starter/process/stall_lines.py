"""Transverse road lines (parking-stall lines, stop lines, crosswalk bars) crossing a fixed distance ahead of the car:
an along-track ruler that does not depend on the BEV scale (user's idea: parking-stall grid on the road).

Every image is mapped onto the road plane (2 cm cells, 2.5..9 camera heights ahead, +-6 sideways).  Thin bright
structures that run ACROSS the road (horizontal in the BEV: long along y, thin along x) are kept, in three lateral
bands (left curb side, own lane, right curb side), and summed into a profile along x.  A line is tracked from frame to
frame; the fractional frame at which it passes x0 (default 4.5 camera heights) is interpolated.  Because every line is
timed at the same place in the image, a pitch / height error of the BEV only shifts all crossings together.

For a trajectory, the arc length at every crossing gives the spacing of consecutive lines; in a row of parking stalls
the true spacing is constant, so the spacing along the row is the trajectory's local scale.  Rows driven in both
directions must give the same spacing.
usage: python process/stall_lines.py Track2 traj.csv [traj2.csv ...] [--x0 4.5]      (from image_slam_starter)
writes Track2/result/stall_crossings.npz (cached image measurement)"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, ".")
from utils import nhc

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("trajs", nargs="*")
ap.add_argument("--x0", type=float, default=4.5)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
RES, XN, XF, YW = 0.02, 2.5, 9.0, 6.0
BANDS = {"left": (1.8, 6.0), "lane": (-1.8, 1.8), "right": (-6.0, -1.8)}       # y > 0 = left
cam = (1035.5, 722.17, 456.83, -0.05833, 0.088413)
K = np.array([[cam[0], 0, cam[1]], [0, cam[0] * 1038.55506 / 1040.18078, cam[2]], [0, 0, 1.0]]); D = np.array([cam[3], cam[4], 0, 0, 0.0])
F, L, U = nhc.axes(*np.radians((-0.14, -1.88)))
xs = np.arange(XF, XN, -RES); ys = np.arange(YW, -YW, -RES)
gx, gy = np.meshgrid(xs, ys, indexing="ij")
P = gx[..., None] * F + gy[..., None] * L - U
uv = cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)
files = sorted(os.listdir(f"{a.target}/data/images"))
SEG = None


def profile(i):
    global SEG
    if SEG is None and os.path.exists(f"{a.target}/seg.npz"):
        z = np.load(f"{a.target}/seg.npz"); SEG = (z["road"], z["obj"])
    g = cv2.imread(f"{a.target}/data/images/{files[i]}", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    b = cv2.remap(g, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=-1)
    valid = b >= 0
    b[~valid] = np.median(b[valid])
    # bright, thin along x (rows), long along y: top-hat with a tall kernel, then an opening with a wide kernel
    th = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.6 / RES) | 1)))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.5 / RES), 1)))
    sd = np.sqrt(np.maximum(cv2.blur(b * b, (61, 61)) - cv2.blur(b, (61, 61)) ** 2, 1.0))
    r = np.clip(th / sd, 0, 4); r[(r < 1.0) | ~valid] = 0
    if SEG is not None:
        suv = uv / 4.0
        obj = cv2.remap(SEG[1][i], suv[..., 0], suv[..., 1], cv2.INTER_NEAREST, borderValue=255)
        r[obj > 76] = 0                                       # not on cars / people
    out = {}
    for k, (y0, y1) in BANDS.items():
        c0, c1 = int((YW - y1) / RES), int((YW - y0) / RES)
        out[k] = cv2.GaussianBlur(r[:, c0:c1].sum(1)[:, None], (1, 0), 2.0)[:, 0].astype(np.float32)
    return out


def peaks(p, thr):
    m = (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > thr)
    idx = np.where(m)[0] + 1
    sub = []
    for j in idx:                                            # parabolic sub-cell
        l_, c_, h_ = p[j - 1], p[j], p[j + 1]; d_ = l_ - 2 * c_ + h_
        sub.append(j + (0.5 * (l_ - h_) / d_ if d_ < 0 else 0.0))
    return XF - np.array(sub) * RES                         # distance ahead


if __name__ == "__main__":
    cache = f"{a.target}/result/stall_profiles.npz"
    if os.path.exists(cache) and not a.force:
        z = np.load(cache); PR = {k: z[k] for k in BANDS}
    else:
        with Pool(16) as pool:
            res = pool.map(profile, range(len(files)), chunksize=8)
        PR = {k: np.array([r_[k] for r_ in res]) for k in BANDS}
        np.savez_compressed(cache, **PR)
    N = len(files)
    T0 = np.genfromtxt(a.trajs[0], delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4) if a.trajs else None
    # crossings: a line at x_i >= x0 in frame i that is at x_{i+1} < x0 in frame i+1 (moved by about the expected step)
    events = []
    for k, prof in PR.items():
        thr = 4.0 * np.median(prof[prof > 0]) if (prof > 0).any() else 1.0
        pk = [peaks(prof[i], thr) for i in range(N)]
        for i in range(N - 1):
            if T0 is not None:
                st = (np.linalg.inv(T0[i])[:3, :3] @ (T0[i + 1, :3, 3] - T0[i, :3, 3])) @ F / 1.15
            else:
                st = 0.35
            if st < 0.05:
                continue
            for xi in pk[i]:
                if not (a.x0 <= xi < a.x0 + 1.5 * st + 0.1):
                    continue
                cand = pk[i + 1][(pk[i + 1] < a.x0) & (np.abs(pk[i + 1] - (xi - st)) < 0.5 * st + 0.05)]
                if len(cand) == 1:
                    xj = cand[0]
                    events.append((k, i + (xi - a.x0) / (xi - xj), xi - xj))
    print(f"{len(events)} line crossings at {a.x0} camera heights: " + ", ".join(f"{k} {sum(e[0] == k for e in events)}" for k in BANDS))
    np.save(f"{a.target}/result/stall_events.npy", np.array([(list(BANDS).index(e[0]), e[1], e[2]) for e in events]))
    for t in a.trajs:
        T = np.genfromtxt(t, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4); C = T[:, :3, 3]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        print(os.path.basename(t))
        for bi, k in enumerate(BANDS):
            ev = sorted(e[1] for e in events if e[0] == k)
            sv = np.interp(ev, np.arange(N), s)
            sp = np.diff(sv); fr = np.array(ev[:-1])
            ok = (sp > 0.8) & (sp < 8.0)
            print(f"  {k:5s}: {ok.sum()} spacings, median {np.median(sp[ok]):.3f} units")
