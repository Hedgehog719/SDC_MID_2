"""Build a small web viewer (map + trajectory + camera image per frame) from a submission.
Usage: python process/make_viewer.py [Track1] [--csv submission.csv] [--out ../viewer]"""
import argparse, json, os, sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.io import load_ply

ap = argparse.ArgumentParser()
ap.add_argument("target", nargs="?", default="Track1")
ap.add_argument("--csv", default="submission.csv")
ap.add_argument("--out", default="../viewer")
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--voxel", type=float, default=0.2)
args = ap.parse_args()

df = pd.read_csv(f"{args.target}/result/{args.csv}", dtype={"timestamp": str})
T = df.iloc[:, 1:].to_numpy(np.float64).reshape(-1, 4, 4)
H = np.load(f"{args.target}/result/ground_h.npy"); sc = np.median(H) / H

# display frame: z up, from the plane of the route
P = T[:, :3, 3]; c0 = P.mean(0)
Vt = np.linalg.svd(P - c0, full_matrices=False)[2]
if (T[:, :3, 1] @ Vt[2]).mean() > 0:      # camera y axis points down -> flip so that z is up
    Vt[2] *= -1
if np.linalg.det(Vt) < 0:
    Vt[1] *= -1
disp = lambda X: (X - c0) @ Vt.T

pts, rgb = [], []
for i in range(0, len(T), args.every):
    p, c = load_ply(f"{args.target}/data/raw_ply/{df.timestamp[i]}.ply", color=True)
    p = p * sc[i]
    m = (p[:, 2] > 1) & (p[:, 2] < 12) & (p[:, 1] > -3)
    pts.append(p[m] @ T[i, :3, :3].T + T[i, :3, 3]); rgb.append(c[m])
pts = disp(np.concatenate(pts)); rgb = np.concatenate(rgb)
key = np.floor(pts / args.voxel).astype(np.int64)
_, idx = np.unique(key, axis=0, return_index=True)
if len(idx) > 250000:
    idx = np.random.default_rng(0).choice(idx, 250000, replace=False)
pts, rgb = pts[idx], (rgb[idx] * 255).astype(int)

os.makedirs(f"{args.out}/img", exist_ok=True)
for i, t in enumerate(df.timestamp):
    dst = f"{args.out}/img/{i:04d}.jpg"
    if not os.path.exists(dst):
        im = cv2.imread(f"{args.target}/data/images/{t}.jpg")
        cv2.imwrite(dst, cv2.resize(im, (720, 464)), [cv2.IMWRITE_JPEG_QUALITY, 75])

tr = disp(P); fwd = T[:, :3, 2] @ Vt.T
json.dump({"pts": np.round(pts, 2).T.tolist(), "rgb": ["rgb(%d,%d,%d)" % tuple(c) for c in rgb],
           "traj": np.round(tr, 3).T.tolist(), "fwd": np.round(fwd, 3).T.tolist()},
          open(f"{args.out}/data.json", "w"))
print("viewer data:", len(pts), "points ->", args.out)
