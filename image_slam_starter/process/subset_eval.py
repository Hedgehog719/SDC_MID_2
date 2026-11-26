"""Single-pass simulation (slam_v2.py --frames N, no revisits): Sim(3) ATE of each run against the first N frames of v10,
plus 132-frame windows and the along / cross / vertical split.
usage: python process/subset_eval.py run.csv [run.csv ...]"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
REF = load(os.path.join(ROOT, "Track1", "result", "v10_vo_shared_it5.csv"))


def sim3(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rr = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((X - mx) ** 2).sum()
    return (s * (Rr @ (X - mx).T)).T + my


for p in sys.argv[1:]:
    X = load(p)[:, :3, 3]
    n = len(X)
    C = REF[:n, :3, 3]
    Xa = sim3(X, C)
    e = Xa - C
    up = np.linalg.svd(C - C.mean(0))[2][2]
    fwd = np.gradient(C, axis=0); fwd -= np.outer(fwd @ up, up); fwd /= np.linalg.norm(fwd, axis=1)[:, None] + 1e-12
    al, cr, ve = np.einsum("ij,ij->i", e, fwd), np.einsum("ij,ij->i", e, np.cross(up, fwd)), e @ up
    win = [np.sqrt(((sim3(X[a:a + 132], C[a:a + 132]) - C[a:a + 132]) ** 2).sum(1).mean()) for a in range(0, n - 131, 66)]
    rms = lambda v: np.sqrt(np.mean(v ** 2))
    print(f"{os.path.basename(p):28s} ATE {rms(np.linalg.norm(e, axis=1)):.3f} m | along {rms(al):.3f} cross {rms(cr):.3f} vert {rms(ve):.3f} | "
          f"132-frame windows median {np.median(win):.3f} max {np.max(win):.3f}")
