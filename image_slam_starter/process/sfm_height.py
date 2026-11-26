"""Camera height above the road in an image-only reconstruction (process/sfm_ba.py): the triangulated points that lie
0.9-1.9 m below the nearest camera are taken as road, a plane is fitted around every camera (utils/ba.camera_heights).
A constant real camera height means: height profile = local scale error of the reconstruction.
usage: python process/sfm_height.py Track1/result/<name> [...]      (reads <name>.csv, <name>.pts.npz)"""
import sys
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from utils import ba, nhc

up_c = nhc.axes(np.radians(-0.02), np.radians(-1.79))[2]
LAPS = [0, 473, 830]
for name in sys.argv[1:]:
    T = np.genfromtxt(name + ".csv", delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
    z = np.load(name + ".pts.npz"); X = z["xyz"].astype(float)
    C = T[:, :3, 3]
    d, j = cKDTree(C).query(X)
    up = np.einsum("nij,j->ni", T[j, :3, :3], up_c)
    hgt = -np.einsum("ni,ni->n", X - C[j], up)
    road = (hgt > 0.9) & (hgt < 1.9) & (d < 6.0)
    pts = [(x, None, None, True) for x in X[road]]
    hh, hn = ba.camera_heights(T, pts, up_c, radius=4.0)
    g = np.isfinite(hh); N = len(T)
    sm = median_filter(np.interp(np.arange(N), np.where(g)[0], hh[g]), size=41, mode="nearest")
    ls = np.log(sm / np.median(hh[g]))
    lap = np.searchsorted(LAPS, np.arange(N), side="right") - 1
    acc, cnt = ls.copy(), np.ones(N)
    for L in range(3):
        idx = np.where(lap == L)[0]
        dd, jj = cKDTree(C[idx]).query(C, distance_upper_bound=3.0)
        q = idx[np.minimum(jj, len(idx) - 1)]; ok = (lap != L) & np.isfinite(dd)
        acc[ok] += ls[q[ok]]; cnt[ok] += 1
    sh = acc / cnt
    np.save(name + ".sfmh.npy", hh)
    print(f"{name}: {road.sum()} road points of {len(X)}; heights for {g.sum()} frames, median {np.median(hh[g]):.3f} m, robust spread "
          f"{100*1.4826*np.median(np.abs(hh[g]-np.median(hh[g]))):.1f} cm; smoothed profile: std {100*np.std(ls):.2f} %, shared by the passes "
          f"{100*np.std(sh):.2f} %, own {100*np.std(ls-sh):.2f} %\n   every 60 frames [%]: " + " ".join(f"{100*ls[i]:+.1f}" for i in range(0, N, 60)))
