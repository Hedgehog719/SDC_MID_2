"""How good are LoGeR's own camera poses (process/loger_run.py chunks)?  Against v10 (gyro) and a no-gyro run:
per chunk Sim(3) ATE, relative yaw errors (per-metre bias, scatter) and the scale stability inside a chunk (step-length
ratio over 30-frame windows).
usage: python process/loger_eval.py Track1 [nogyro_run.csv]"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = sys.argv[1]
load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
REF = load(os.path.join(ROOT, T, "result", "v10_vo_shared_it5.csv"))
OTH = load(sys.argv[2]) if len(sys.argv) > 2 else None
Vt = np.linalg.svd(REF[:, :3, 3] - REF[:, :3, 3].mean(0))[2]
up = Vt[2] * np.sign(Vt[2] @ -REF[:, :3, 1].mean(0))
rs = lambda x: 1.4826 * np.median(np.abs(np.asarray(x) - np.median(x)))


def sim3(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt_ = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt_))
    Rr = U @ D @ Vt_
    s = np.trace(np.diag(S) @ D) / ((X - mx) ** 2).sum()
    return s, Rr, lambda Z: (s * (Rr @ (Z - mx).T)).T + my


def stats(P, fr, name):
    C = P[:, :3, 3]; Cr = REF[fr, :3, 3]
    s, Rr, f = sim3(C, Cr)
    ate = np.sqrt(((f(C) - Cr) ** 2).sum(1).mean())
    ye, d, tu = [], [], []
    for st in (3, 8):
        for i in range(len(fr) - st):
            a, b = fr[i], fr[i + st]
            Rm = P[i, :3, :3].T @ P[i + st, :3, :3]
            Rf = REF[a, :3, :3].T @ REF[b, :3, :3]
            e = Rot.from_matrix(Rf.T @ Rm).as_rotvec(); upc = REF[a, :3, :3].T @ up
            ye.append(np.degrees(e @ upc)); d.append(np.linalg.norm(Cr[i + st] - Cr[i])); tu.append(np.degrees(Rot.from_matrix(Rf).as_rotvec() @ upc))
    ye, d, tu = map(np.array, (ye, d, tu))
    mv = d > 0.3
    A = np.c_[np.ones(mv.sum()), d[mv], tu[mv]]
    c = np.linalg.lstsq(A, ye[mv], rcond=None)[0]
    w = [np.linalg.norm(C[i + 30] - C[i]) * s / np.linalg.norm(Cr[i + 30] - Cr[i]) for i in range(0, len(fr) - 30, 10)
         if np.linalg.norm(Cr[i + 30] - Cr[i]) > 3]
    return ate, rs(ye[mv]) * 1000, c[1] * 1000, 100 * np.std(w) / np.mean(w) if len(w) > 2 else np.nan


rows = []
for f in sorted(glob.glob(os.path.join(ROOT, T, os.environ.get("LOGER_NAME", "loger"), "chunk_*.npz"))):
    z = np.load(f)
    fr = z["frames"]
    r = stats(z["camera_poses"], fr, "loger")
    o = stats(OTH[fr], fr, "other") if OTH is not None else (np.nan,) * 4
    rows.append((fr[0], fr[-1]) + r + o)
    print(f"frames {fr[0]:4d}-{fr[-1]:4d}: LoGeR ATE {r[0]:.3f} m, yaw sd {r[1]:.0f} mdeg, yaw bias {r[2]:+.1f} mdeg/m, 30-frame scale sd {r[3]:.2f} % | "
          f"no-gyro run: ATE {o[0]:.3f}, yaw sd {o[1]:.0f}, bias {o[2]:+.1f}, scale sd {o[3]:.2f} %", flush=True)
d = np.array(rows, float)
print(f"median over {len(d)} chunks: LoGeR ATE {np.nanmedian(d[:, 2]):.3f} m, yaw sd {np.nanmedian(d[:, 3]):.0f} mdeg, bias {np.nanmedian(d[:, 4]):+.1f} mdeg/m, "
      f"scale sd {np.nanmedian(d[:, 5]):.2f} % | no-gyro run: ATE {np.nanmedian(d[:, 6]):.3f}, yaw sd {np.nanmedian(d[:, 7]):.0f}, bias {np.nanmedian(d[:, 8]):+.1f}, scale sd {np.nanmedian(d[:, 9]):.2f} %")
