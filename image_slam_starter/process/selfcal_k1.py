"""Self-calibration of the radial lens distortion k1 from the images alone (no IMU, no reference trajectory).

The essential-matrix inliers of the VO pairs are undistorted with each candidate k1; every pair's relative pose is
estimated (5-point RANSAC + Sampson refinement with Huber loss) and the median Sampson residual of all pairs is
reported.  The right distortion makes the matches fit one pinhole two-view geometry best (Track01: minimum near
k1 = -0.03, the same value the gyro gives; see NOTES "Track02 準備").

usage: python process/selfcal_k1.py <run.csv.run.json of a run with k1 = 0 and --vo> [--step 8]
       e.g.  python slam_v2.py Track2 --out k1zero.csv && python process/selfcal_k1.py Track2/result/k1zero.csv.run.json
"""
import json
import multiprocessing as mp
import os
import pickle
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import vodir  # noqa: E402
from utils.io import load_camera_info  # noqa: E402

run = json.load(open(sys.argv[1]))
STEP = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 8
if run["args"].get("k1"):
    sys.exit("the run must use k1 = 0 (its VO matches are the raw keypoints)")
vo_file = [c for c in run["caches"] if os.path.basename(c).startswith("vo_")][0]
cam = os.path.join(ROOT, run["args"]["target"], "data", "camera_info.json")
K, _ = load_camera_info(cam)
Kinv = np.linalg.inv(K)
vo = pickle.load(open(os.path.join(ROOT, vo_file), "rb"))
vo = {p: m for p, m in vo.items() if p[1] - p[0] == STEP and p[0] % 2 == 0 and len(m[0]) >= 50}
print(f"camera {cam}\nVO matches {vo_file}: {len(vo)} pairs of step {STEP}")


def job(a):
    from scipy.optimize import least_squares
    pa, pb = a
    E, m = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        return None
    _, Rm, t, _ = cv2.recoverPose(E[:3], pa, pb, K)
    xa = np.c_[pa, np.ones(len(pa))] @ Kinv.T
    xb = np.c_[pb, np.ones(len(pb))] @ Kinv.T
    r = least_squares(vodir._sampson, np.r_[cv2.Rodrigues(Rm)[0][:, 0], t[:, 0]], args=(xa, xb), loss="huber", f_scale=1.0 / K[0, 0])
    return np.median(np.abs(r.fun)) * K[0, 0]


def und(x, k1):
    return cv2.undistortPoints(x.reshape(-1, 1, 2).astype(np.float64), K, np.array([k1, 0, 0, 0, 0.0]), P=K).reshape(-1, 2)


grid = np.round(np.arange(0.0, -0.0601, -0.01), 3)
res = []
with mp.get_context("fork").Pool(min(32, os.cpu_count())) as pool:
    for k1 in grid:
        cv2.setRNGSeed(0)
        r = [x for x in pool.map(job, [(und(pa, k1), und(pb, k1)) for pa, pb in vo.values()], chunksize=16) if x is not None]
        res.append(np.median(r))
        print(f"  k1 {k1:+.3f}: median Sampson residual {res[-1]:.4f} px", flush=True)
i = int(np.argmin(res))
j = np.clip(i, 1, len(grid) - 2)
c = np.polyfit(grid[j - 1:j + 2], res[j - 1:j + 2], 2)
best = -c[1] / (2 * c[0]) if c[0] > 0 else grid[i]
print(f"best k1 = {best:+.4f}  (grid minimum {grid[i]:+.3f}; use --k1 {best:.3f})")
