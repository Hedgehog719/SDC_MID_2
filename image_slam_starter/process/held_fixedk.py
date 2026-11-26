"""Holdout metric with a FIXED measurement scale: the held-out loop pairs (touching the held lap) are converted to metres
with v10's per-frame cloud scales for every run, so a configuration that changes the cloud scales (height correction,
depth prior) cannot also change the ruler it is judged with.  Laps = v10's split [0, 473, 830].
usage: python process/held_fixedk.py <pairs.pkl> name [name ...]    (runs Track1/result/_vp_<name>_<lap>.csv)"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "Track1", "result")
load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
K = np.load(f"{R}/v10_vo_shared_it5.csv.scale.npy")
pairs = pickle.load(open(os.path.join(R, sys.argv[1]), "rb"))
N = len(K)
lap = np.searchsorted([0, 473, 830], np.arange(N), side="right") - 1
loops = {p: r for p, r in pairs.items() if r is not None and p[1] - p[0] > 150}
print(f"{len(loops)} verified loop pairs in {sys.argv[1]}")
for name in sys.argv[2:]:
    out = []
    for k in (2, 1, 0):
        f = f"{R}/_vp_{name}_{k}.csv"
        if not os.path.exists(f):
            out.append(None); continue
        P = load(f)
        res = []
        for (a, b), r in loops.items():
            if k not in (lap[a], lap[b]):
                continue
            t = (r[4] * K[a] * r[3] + r[6] * K[b] * r[5]) / (r[4] + r[6])
            Tt = np.linalg.inv(P[a]) @ P[b]
            res.append(np.linalg.norm(Tt[:3, 3] - t))
        out.append(np.sqrt(np.mean(np.square(res))))
    ok = [o for o in out if o is not None]
    print(f"{name:16s} " + " / ".join("  -  " if o is None else f"{o:.3f}" for o in out) + (f"   mean {np.mean(ok):.3f}" if len(ok) == 3 else ""))
