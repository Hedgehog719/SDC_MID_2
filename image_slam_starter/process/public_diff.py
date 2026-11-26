"""Where do two Track02 trajectories differ on the public road?  TRAJ is Sim(3)-aligned to REF on the public frames only
(westbound 1141-1344, eastbound 2731-2947, the stretch the public score uses); the remaining difference of every
public frame is split into along the road / lateral / vertical in REF's local road frame (tangent of REF, up = REF's
camera -Y), for the two directions separately.  Reported: RMS of each component per direction, the mean and the linear /
quadratic parts along the pass (a shift, a stretch, a bend), and a plot.
usage: python process/public_diff.py REF TRAJ [--png out.png]   (names in Track2/result, without .csv)"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("traj")
ap.add_argument("--png", default=None)
a = ap.parse_args()
PASSES = {"W": (1141, 1345), "E": (2731, 2948)}


def load(t):
    return np.genfromtxt(f"Track2/result/{t}.csv", delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)


def umeyama(X, Y):
    """s, R, t with s R X + t ~ Y"""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


Tr, Tt = load(a.ref), load(a.traj)
idx = np.r_[tuple(np.arange(*p) for p in PASSES.values())]
s, R, t = umeyama(Tt[idx, :3, 3], Tr[idx, :3, 3])
Pt = (s * (R @ Tt[:, :3, 3].T)).T + t
Pr = Tr[:, :3, 3]
print(f"{a.traj} -> {a.ref}: Sim(3) on {len(idx)} public frames, scale {s:.4f}; public RMSE {np.sqrt(((Pt[idx] - Pr[idx]) ** 2).sum(1).mean()):.3f}")
out = {}
for p, (f0, f1) in PASSES.items():
    fr = np.arange(f0, f1)
    C = Pr[fr]
    tang = np.gradient(C, axis=0)
    tang = np.array([tang[max(0, i - 5):i + 6].mean(0) for i in range(len(tang))])
    up = -Tr[fr, :3, 1]
    fw = tang - (tang * up).sum(1, keepdims=True) * up; fw /= np.linalg.norm(fw, axis=1, keepdims=True)
    lt = np.cross(up, fw)
    d = Pt[fr] - C
    comp = np.c_[(d * fw).sum(1), (d * lt).sum(1), (d * up).sum(1)]
    sd = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    x = (sd - sd.mean()) / (sd.max() / 2)                                   # -1 .. 1 along the pass
    out[p] = (sd, comp)
    print(f"  {p} ({f0}-{f1 - 1}, {sd[-1]:.1f} units): RMS along {np.sqrt((comp[:,0]**2).mean()):.3f}, lateral {np.sqrt((comp[:,1]**2).mean()):.3f}, "
          f"vertical {np.sqrt((comp[:,2]**2).mean()):.3f}")
    for k, nm in enumerate(("along", "lateral", "vertical")):
        c2, c1, c0 = np.polyfit(x, comp[:, k], 2)
        res = comp[:, k] - np.polyval([c2, c1, c0], x)
        print(f"     {nm:8s} mean {comp[:, k].mean():+.3f}  linear (end-to-end) {2 * c1:+.3f}  bend (mid vs ends) {-c2:+.3f}  "
              f"rest RMS {np.sqrt((res ** 2).mean()):.3f}")

if a.png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"along": "#2a78d6", "lateral": "#eb6834", "vertical": "#1baf7a"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (p, (sd, comp)) in zip(axs, out.items()):
        for k, nm in enumerate(col):
            ax.plot(sd, comp[:, k], color=col[nm], lw=2, label=nm)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_title(f"{'westbound' if p == 'W' else 'eastbound'} {PASSES[p][0]}-{PASSES[p][1] - 1}", fontsize=11, loc="left")
        ax.set_xlabel(f"distance along the pass [traj units] (driving direction)")
        ax.grid(color="#e5e5e5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel(f"{a.traj} minus {a.ref} [traj units]")
    axs[0].legend(frameon=False, loc="best")
    fig.suptitle(f"{a.traj} vs {a.ref}, Sim(3) on the public frames", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.png, dpi=130)
    print("saved", a.png)
