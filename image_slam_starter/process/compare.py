"""Sim(3)-aligned position RMSE between trajectories (same metric as the leaderboard, but against another
estimate instead of the ground truth): tells how much two versions really differ.
Usage: python process/compare.py ref.csv other.csv [more.csv ...]"""
import sys

import numpy as np
import pandas as pd


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    X, Y = src - ms, dst - md
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src))
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    Rm = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / X.var(0).sum()
    return s, Rm, md - s * Rm @ ms


def ate(src, dst):
    s, Rm, t = umeyama(src, dst)
    return np.sqrt((np.linalg.norm(s * src @ Rm.T + t - dst, axis=1) ** 2).mean()), s


load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)[:, :3, 3]
if __name__ == "__main__":
    ref = load(sys.argv[1])
    for p in sys.argv[2:]:
        x = load(p)
        e, s = ate(x, ref)
        win = [ate(x[i:i + 132], ref[i:i + 132])[0] for i in range(0, len(ref) - 132, 20)]
        print(f"{p.split('/')[-1]:>24} vs {sys.argv[1].split('/')[-1]}: ATE {e:.3f} m (scale {s:.3f}) | "
              f"132-frame windows: median {np.median(win):.3f}, max {np.max(win):.3f} m")
