"""Diagnostics of 2026-11-20 (run from image_slam_starter/): Model-free read-out of the scored Track02 submissions around X0 = t2_sfm_bu1035cf.

For a trajectory X_k Sim(3)-aligned to X0 on the public frames, d_k = X_k - X0 (the Sim(3) part is removed to first order):
  f_k = score_k^2 = f0 + 2 sigma <e, d_k> + sigma^2 |d_k|^2      (|.|^2 = mean over the 421 frames of the squared norm)
so every score gives one exact linear measurement b_k = <e, d_k> of the unknown error e (metres, in X0's frame).
The part of e inside span{d_k}: |e_span|^2 = b^T G^-1 b (G = Gram matrix) -> how much of the error the explored directions hold.
"""
import numpy as np

S = "../submissions/track2/"
PASSES = {"W": (1004, 1206), "E": (2881, 3100)}
IDX = np.r_[tuple(np.arange(*p) for p in PASSES.values())]
SC = dict(t2_sfm_bu1035cf=0.203, t2_sfm_bu1035=0.203, t2_sfm_bu105=0.206, t2_sfm_bu1035_sf=0.210, t2_sfm_bu1035_rl_r0=0.249,
          t2_sfm_pp06zr=0.218, t2_sfm_pp06zrl=0.225, t2_sfm_pp06zr3=0.213, t2_sfm_pp06z=0.330, t2_sfm_pp10z=0.333,
          t2_probe_slide=0.213, t2_probe_pts=0.280, t2_dash1r=0.421, t2_once=0.432, t2_seg=0.431, t2_dash05r=0.436,
          t2_once_cars=0.434, t2_cam_z16_it3=0.506, t2_cam=0.577, t2_seg_tight=0.484, t2_pw1s104=0.477)
SIG = 1.5


def load(t):
    d = np.genfromtxt(S + t + ".csv", delimiter=",", skip_header=1)
    return d[:, 1:].reshape(-1, 4, 4)[IDX, :3, 3]


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, Sv, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    s = np.trace(np.diag(Sv) @ D) / (Xc ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


X0 = load("t2_sfm_bu1035cf"); f0 = SC["t2_sfm_bu1035cf"] ** 2
rows = []
for k, sc in SC.items():
    if k == "t2_sfm_bu1035cf":
        continue
    X = load(k); s, R, t = umeyama(X, X0); d = (s * X @ R.T + t) - X0
    nd = np.sqrt((d ** 2).sum(1).mean())
    b = (sc ** 2 - f0 - SIG ** 2 * nd ** 2) / (2 * SIG)          # <e, d> in m*unit
    c = b / nd                                                  # component of e along d_hat [m]
    rows.append((k, sc, nd, c, d))
    print(f"{k:22s} score {sc:.3f}  |Pd| {nd:.3f} u ({SIG*nd:.3f} m)  e along d_hat {c:+.3f} m   (rounding +-{0.0005*sc/(SIG*nd):.4f})")

# span analysis on the near (linear) family
for lim in (0.15, 0.25, 0.40):
    sel = [r for r in rows if r[2] <= lim and r[2] >= 0.02]
    D = np.array([r[4].ravel() / np.sqrt(len(IDX)) for r in sel])            # m x (3N), rows normalised so <a,b> = mean
    b = np.array([r[3] * r[2] for r in sel])
    G = D @ D.T
    w, V = np.linalg.eigh(G)
    keep = w > 1e-4 * w.max()
    Ginv = (V[:, keep] / w[keep]) @ V[:, keep].T
    e2 = b @ Ginv @ b
    print(f"\n|Pd| <= {lim}: {len(sel)} trajectories, rank {keep.sum()}; error inside their span: {np.sqrt(max(e2,0)):.3f} m of {np.sqrt(f0):.3f} m "
          f"-> best blend would score {np.sqrt(max(f0 - e2, 0)):.3f}")
    print("   ", ", ".join(r[0].replace("t2_", "") for r in sel))
