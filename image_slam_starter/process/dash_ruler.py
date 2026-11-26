"""Dashed lane lines as an along-track ruler (user's idea: use the painted grid on the road to judge the straights).

A dashed line repeats with a constant period (dash + gap) along a road.  Every image is mapped onto the road plane
(2 cm cells, 2.5..9 camera heights ahead, +-2.5 sideways); the dashed line is found as the column band with the
strongest thin bright line that is interrupted (on/off along x).  The intensity along that band is binarised; a dash
END (the near edge of a dash, where dash turns into gap going towards the car) is tracked from frame to frame and the
fractional frame at which it passes x0 is interpolated.  Every edge is timed at the same place in the image, so a
pitch / height error of the BEV moves all crossings together and cancels.

For a trajectory, the arc length between consecutive crossings is one dash period.  The true period is constant along
a road, so period(s) / median period of that road = the trajectory's local scale error; the same road driven in the
other direction must give the same period.
usage: python process/dash_ruler.py Track2 traj.csv [traj2.csv ...]      (from image_slam_starter)"""
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
ap.add_argument("--x0", type=float, default=4.0)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
RES, XN, XF, YW = 0.02, 2.5, 9.0, 2.5
cam = (1035.5, 722.17, 456.83, -0.05833, 0.088413)
K = np.array([[cam[0], 0, cam[1]], [0, cam[0] * 1038.55506 / 1040.18078, cam[2]], [0, 0, 1.0]]); D = np.array([cam[3], cam[4], 0, 0, 0.0])
F, L, U = nhc.axes(*np.radians((-0.14, -1.88)))
xs = np.arange(XF, XN, -RES); ys = np.arange(YW, -YW, -RES)
gx, gy = np.meshgrid(xs, ys, indexing="ij")
P = gx[..., None] * F + gy[..., None] * L - U
uv = cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)
files = sorted(os.listdir(f"{a.target}/data/images"))


def dash_profile(i):
    """(lateral position of the dashed line, on/off profile along x) or None"""
    g = cv2.imread(f"{a.target}/data/images/{files[i]}", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    b = cv2.remap(g, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=-1)
    valid = b >= 0
    b[~valid] = np.median(b[valid])
    th = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.4 / RES) | 1, 1)))   # thin across y
    sd = np.sqrt(np.maximum(cv2.blur(b * b, (61, 61)) - cv2.blur(b, (61, 61)) ** 2, 1.0))
    r = np.clip(th / sd, 0, 4); r[~valid] = 0
    on = (r > 1.5).astype(np.float32)
    colfrac = on.mean(0)                                     # how much of every column is line
    best = None
    for c in range(5, on.shape[1] - 5, 2):
        band = on[:, c - 3:c + 4].max(1)
        frac = band.mean()
        if not (0.2 < frac < 0.8):                            # interrupted: dashed, not solid, not empty
            continue
        trans = np.abs(np.diff(band)).sum()
        if 2 <= trans <= 12 and (best is None or frac * trans > best[0]):
            best = (frac * trans, c, band)
    if best is None:
        return None
    _, c, band = best
    return float(ys[c]), cv2.GaussianBlur(band[:, None], (1, 0), 1.0)[:, 0].astype(np.float32)


def edges(band):
    """x of the near ends of dashes (on -> off when going from far to near, i.e. along increasing row)"""
    on = band > 0.5
    j = np.where(on[:-1] & ~on[1:])[0]
    out = []
    for k in j:                                              # linear sub-cell
        v0, v1 = band[k], band[k + 1]
        out.append(XF - (k + (v0 - 0.5) / max(v0 - v1, 1e-6)) * RES)
    return np.array(out)


if __name__ == "__main__":
    cache = f"{a.target}/result/dash_profiles.npy"
    if os.path.exists(cache) and not a.force:
        prof = list(np.load(cache, allow_pickle=True))
    else:
        with Pool(16) as pool:
            prof = pool.map(dash_profile, range(len(files)), chunksize=8)
        np.save(cache, np.array(prof, dtype=object), allow_pickle=True)
    N = len(files)
    T0 = np.genfromtxt(a.trajs[0], delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
    have = np.array([p is not None for p in prof])
    print(f"dashed line found in {have.sum()} of {N} frames")
    E = [edges(p[1]) if p is not None else np.zeros(0) for p in prof]
    cross = []
    for i in range(N - 1):
        if prof[i] is None or prof[i + 1] is None or abs(prof[i][0] - prof[i + 1][0]) > 0.3:
            continue
        st = (np.linalg.inv(T0[i])[:3, :3] @ (T0[i + 1, :3, 3] - T0[i, :3, 3])) @ F / 1.2
        if st < 0.05:
            continue
        for xi in E[i]:
            if not (a.x0 <= xi < a.x0 + 2 * st):
                continue
            c = E[i + 1][(E[i + 1] < a.x0) & (np.abs(E[i + 1] - (xi - st)) < 0.6 * st + 0.05)]
            if len(c) == 1:
                cross.append(i + (xi - a.x0) / (xi - c[0]))
    cross = np.array(sorted(cross))
    print(f"{len(cross)} dash ends crossing {a.x0} camera heights")
    np.save(f"{a.target}/result/dash_crossings.npy", cross)
    for t in a.trajs:
        T = np.genfromtxt(t, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4); C = T[:, :3, 3]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        sv = np.interp(cross, np.arange(N), s); per = np.diff(sv); fr = cross[:-1]
        ok = (np.diff(cross) < 40) & (per > 0.5)
        print(f"{os.path.basename(t)}: {ok.sum()} periods, median {np.median(per[ok]):.3f} units")
        np.save(f"{a.target}/result/dash_periods_{os.path.basename(t)[:-4]}.npy", np.c_[fr[ok], per[ok]])
