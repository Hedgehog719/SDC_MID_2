"""Diagnostics of 2026-11-20 (run from image_slam_starter/): error read-out around the current best X0 (default t2_pc3_bu1035_sf).

Same exact relation as span_proj.py: f_k = f0 + 2 sigma <e, d_k> + sigma^2 |d_k|^2 for every scored trajectory X_k Sim(3)-aligned to X0.
Prints |e_span| for growing sets of near trajectories, a leave-one-out check, and the shape of e_span per pass / leg."""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--best", default="t2_pc3_bu1035_sf"); ap.add_argument("--score", type=float, default=0.177)
ap.add_argument("--lim", type=float, default=0.15, help="use trajectories with |Pd| below this [units]")
a = ap.parse_args()
S = "../submissions/track2/"
PASSES = {"W": (1004, 1206), "E": (2881, 3100)}
LEGS = {"W": ((1004, 1076, "NS"), (1076, 1136, "corner"), (1136, 1206, "EW")), "E": ((2881, 2941, "EW"), (2941, 3025, "corner"), (3025, 3100, "NS"))}
IDX = np.r_[tuple(np.arange(*p) for p in PASSES.values())]
N = len(IDX); SIG = 1.5
SC = dict(t2_sfm_bu1035cf=0.203, t2_sfm_bu1035=0.203, t2_sfm_bu105=0.206, t2_sfm_bu1035_sf=0.210, t2_sfm_bu1035_rl_r0=0.249,
          t2_sfm_pp06zr=0.218, t2_sfm_pp06zrl=0.225, t2_sfm_pp06zr3=0.213, t2_probe_slide=0.213, t2_probe_pts=0.280,
          t2_pch_bu1035cf=0.185, t2_pc3_bu1035_sf=0.177, t2_pcm_bu1035_sf=0.154, t2_sfm_pp06z=0.330, t2_sfm_pp10z=0.333)
F0 = a.score


def load(t):
    return np.genfromtxt(S + t + ".csv", delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, Sv, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    s = np.trace(np.diag(Sv) @ D) / (Xc ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


T0 = load(a.best); X0 = T0[IDX, :3, 3]
names, Dl, sc = [], [], []
for k, v in SC.items():
    if k == a.best:
        continue
    X = load(k)[IDX, :3, 3]; s, R, t = umeyama(X, X0); d = (s * X @ R.T + t) - X0
    nd = np.sqrt((d ** 2).sum(1).mean())
    c = (v ** 2 - F0 ** 2 - SIG ** 2 * nd ** 2) / (2 * SIG * nd)
    print(f"{k:22s} score {v:.3f} |Pd| {nd:.3f} u  error component along d_hat {c:+.3f} m")
    if 0.02 <= nd <= a.lim:
        names.append(k); Dl.append(d); sc.append(v)
Dl = np.array(Dl); sc = np.array(sc)


def espan(sel, scores=None, sig=SIG):
    scores = sc[sel] if scores is None else scores
    D = Dl[sel].reshape(len(sel), -1) / np.sqrt(N); nd2 = (D ** 2).sum(1)
    b = (scores ** 2 - F0 ** 2 - sig ** 2 * nd2) / (2 * sig)
    G = D @ D.T; w, V = np.linalg.eigh(G); keep = w > 1e-6 * w.max()
    return ((V[:, keep] / w[keep]) @ V[:, keep].T @ b) @ D * np.sqrt(N)


sel = list(range(len(names)))
rng = np.random.default_rng(1)
mags = [np.sqrt((espan(sel, sc + rng.uniform(-5e-4, 5e-4, len(sc)), rng.uniform(1.45, 1.56)) ** 2).sum() / N) for _ in range(300)]
e = espan(sel).reshape(N, 3)
print(f"\nX0 {a.best} ({F0}); {len(names)} trajectories with |Pd| <= {a.lim}: {', '.join(n.replace('t2_', '') for n in names)}")
print(f"|e_span| {np.sqrt((e**2).sum(1).mean()):.3f} m (noise 5-95%: {np.percentile(mags,5):.3f}-{np.percentile(mags,95):.3f}) of {F0} -> unexplored {np.sqrt(max(F0**2 - (e**2).sum(1).mean(), 0)):.3f} m")
for j in sel:
    rest = [i for i in sel if i != j]
    ej = espan(rest).ravel(); d = Dl[j].ravel()
    p = np.sqrt(max(F0 ** 2 + 2 * SIG * (ej @ d) / N + SIG ** 2 * (d @ d) / N, 0))
    print(f"   LOO {names[j]:22s} actual {sc[j]:.3f} predicted {p:.3f}")
up = -T0[IDX, :3, 1].mean(0); up /= np.linalg.norm(up)
o = 0
print("shape (metres, + = estimate ahead / left / above the truth):")
for p, (f0_, f1_) in PASSES.items():
    P = T0[f0_:f1_, :3, 3]; tg = np.gradient(P, axis=0); tg -= np.outer(tg @ up, up); tg /= np.linalg.norm(tg, axis=1, keepdims=True); lf = np.cross(up, tg)
    ee = e[o:o + f1_ - f0_]
    for a_, b_, nm in LEGS[p]:
        m = slice(a_ - f0_, b_ - f0_)
        al, la, vu = (ee[m] * tg[m]).sum(1), (ee[m] * lf[m]).sum(1), ee[m] @ up
        print(f"   {p} {nm:6s}: along {al.mean():+.3f} (rms {np.sqrt((al**2).mean()):.3f})  left {la.mean():+.3f} (rms {np.sqrt((la**2).mean()):.3f})  "
              f"up {vu.mean():+.3f} (rms {np.sqrt((vu**2).mean()):.3f})")
    o += f1_ - f0_
np.save(f"Track2/result/e_span_{a.best}.npy", e)
