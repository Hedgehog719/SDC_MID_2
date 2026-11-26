"""Independent metric for the no-gyro holdout runs: position RMSE of the held lap against v10 (gyro), after a Sim(3)
fit on the other two laps; laps = v10's split [0, 473, 830].  Prints per lap and the mean."""
import sys, os, numpy as np, pandas as pd
load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
R = "Track1/result/"
C = load(R + "v10_vo_shared_it5.csv")[:, :3, 3]
segs = {0: (0, 473), 1: (473, 830), 2: (830, 1314)}
def sim3(X, Y):
    mx, my = X.mean(0), Y.mean(0); U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); Rr = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((X - mx) ** 2).sum(); return lambda Z: (s * (Rr @ (Z - mx).T)).T + my
for name in sys.argv[1:]:
    out = []
    for k in (2, 1, 0):
        p = f"{R}_vp_{name}_{k}.csv"
        if not os.path.exists(p):
            out.append(None); continue
        X = load(p)[:, :3, 3]; a, b = segs[k]; held = np.zeros(len(X), bool); held[a:b] = True
        f = sim3(X[~held], C[~held]); out.append(np.sqrt(((f(X[held]) - C[held]) ** 2).sum(1).mean()))
    ok = [o for o in out if o is not None]
    print(f"{name:22s} " + " / ".join("  -  " if o is None else f"{o:.3f}" for o in out) + (f"   mean {np.mean(ok):.3f}" if len(ok) == 3 else ""))
