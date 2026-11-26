"""Local lever-arm estimate along a trajectory: an absolute scale check that uses neither the cloud depth
nor the road.  The camera sits L in front of the rear axle, so in a turn it moves sideways by L * yaw.
L is a constant of the car, so (estimate / true L) is the local scale error of the trajectory.
usage: python process/lever_profile.py a.csv [b.csv ...]   (run from image_slam_starter)"""
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, ".")
from utils import nhc


def profile(T, yaw=np.radians(-0.02), pitch=np.radians(-1.79), s=3, blk=40):
    f, l, u = nhc.axes(yaw, pitch)
    lat, dth = [], []
    for i in range(len(T) - s):
        rv = R.from_matrix(T[i, :3, :3].T @ T[i + s, :3, :3]).as_rotvec()
        Rm = T[i, :3, :3] @ R.from_rotvec(0.5 * rv).as_matrix()
        lat.append((Rm.T @ (T[i + s, :3, 3] - T[i, :3, 3])) @ l); dth.append(rv @ u)
    lat, dth = np.array(lat), np.array(dth)
    out = []
    for a in range(0, len(lat), blk):
        x, y = dth[a:a + blk], lat[a:a + blk]
        out.append(abs((x * y).sum() / (x * x).sum()) if (x ** 2).sum() > 0.02 else np.nan)
    return np.array(out)


if __name__ == "__main__":
    P = {n: profile(pd.read_csv(n).values[:, 1:].astype(float).reshape(-1, 4, 4)) for n in sys.argv[1:]}
    for n, p in P.items():
        g = np.isfinite(p)
        print(f"{n}: median {np.median(p[g]):.3f} m, spread (std) {100*np.std(p[g])/np.median(p[g]):.1f} %, range {p[g].min():.2f}..{p[g].max():.2f}")
        print("   " + " ".join("  -- " if not np.isfinite(x) else f"{x:5.2f}" for x in p))
