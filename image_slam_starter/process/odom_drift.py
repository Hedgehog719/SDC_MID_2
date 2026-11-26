"""Odometry quality without loops (proxy for a route without revisits, e.g. Track02): Sim(3)-aligned position RMSE of
a run against a reference trajectory (the best full solution), per lap and per window of W frames.
Each lap / window is aligned on its own, like the leaderboard aligns the Public and the Private part separately.
Usage: python process/odom_drift.py ref.csv run.csv [more.csv ...] [--win 132]"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import ate  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
W = int(sys.argv[sys.argv.index("--win") + 1]) if "--win" in sys.argv else 132
if "--win" in sys.argv:
    args.remove(str(W))
load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)


def laps(P):
    """lap number of every frame from the accumulated heading (same rule as slam_v2.lap_index)"""
    Z = P[:, :3, 2]
    Vt = np.linalg.svd(Z - Z.mean(0))[2]
    yaw = np.unwrap(np.arctan2(Z @ np.cross(Vt[2], Vt[0]), Z @ Vt[0]))
    lap = np.maximum.accumulate(np.floor(np.abs(yaw - yaw[0]) / (2 * np.pi)).astype(int))
    if (lap == lap.max()).sum() < 0.5 * len(P) / max(lap.max(), 1):
        lap[lap == lap.max()] -= 1
    return lap


ref = load(args[0])
lap = laps(ref)
R = ref[:, :3, 3]
print(f"reference {os.path.basename(args[0])}: laps start at {[int(np.argmax(lap == l)) for l in range(lap.max() + 1)]}, "
      f"windows of {W} frames")
for p in args[1:]:
    X = load(p)[:, :3, 3]
    whole = ate(X, R)[0]
    per_lap = [ate(X[lap == l], R[lap == l])[0] for l in range(lap.max() + 1)]
    win = [ate(X[a:a + W], R[a:a + W])[0] for a in range(0, len(X) - W + 1, W // 2)]
    print(f"{os.path.basename(p)}: whole {whole:.3f} | per lap " + " / ".join(f"{e:.3f}" for e in per_lap)
          + f" (mean {np.mean(per_lap):.3f}) | {W}-frame windows median {np.median(win):.3f}, max {np.max(win):.3f}")
