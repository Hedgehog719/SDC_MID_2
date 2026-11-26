"""Top view of one or more trajectories (no ground truth needed): laps of the same route should overlap.
Usage: python process/plot_traj.py Track1/result/a.csv [b.csv ...] -o out.png"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("-o", default="traj.png")
args = ap.parse_args()

fig, axes = plt.subplots(2, len(args.csv), figsize=(6 * len(args.csv), 9), squeeze=False,
                         gridspec_kw={"height_ratios": [3, 1]})
for k, path in enumerate(args.csv):
    P = pd.read_csv(path).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)[:, :3, 3]
    c = P - P.mean(0)
    # ground plane = the two main directions of the route
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    q = c @ Vt.T
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    ax = axes[0, k]
    sc = ax.scatter(q[:, 0], q[:, 1], c=np.arange(len(q)), s=3, cmap="turbo")
    ax.plot(q[0, 0], q[0, 1], "k*", ms=12)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"{path.split('/')[-1]}  length {step.sum():.1f} m  height range {np.ptp(q[:, 2]):.2f} m")
    plt.colorbar(sc, ax=ax, label="frame")
    axes[1, k].plot(step, lw=.6); axes[1, k].set_ylabel("step [m]"); axes[1, k].set_xlabel("frame"); axes[1, k].grid(alpha=.3)
plt.tight_layout(); plt.savefig(args.o, dpi=90); print("saved", args.o)
