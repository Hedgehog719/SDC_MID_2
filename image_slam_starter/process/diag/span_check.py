"""Robustness of the span read-out: rounding Monte Carlo, leave-one-out score prediction, and the shape of e_span."""
import numpy as np

S = "../submissions/track2/"
PASSES = {"W": (1004, 1206), "E": (2881, 3100)}
LEGS = {"W": ((1004, 1076, "NS"), (1076, 1136, "corner"), (1136, 1206, "EW")),
        "E": ((2881, 2941, "EW"), (2941, 3025, "corner"), (3025, 3100, "NS"))}
IDX = np.r_[tuple(np.arange(*p) for p in PASSES.values())]
SC = dict(t2_sfm_bu1035_sf=0.210, t2_sfm_bu1035_rl_r0=0.249, t2_sfm_pp06zr=0.218, t2_sfm_pp06zrl=0.225, t2_sfm_pp06zr3=0.213,
          t2_probe_slide=0.213, t2_probe_pts=0.280, t2_sfm_pp06z=0.330, t2_sfm_pp10z=0.333, t2_dash1r=0.421, t2_once=0.432,
          t2_seg=0.431, t2_once_cars=0.434)
F0 = 0.203
rng = np.random.default_rng(0)


def load(t):
    d = np.genfromtxt(S + t + ".csv", delimiter=",", skip_header=1)
    return d[:, 1:].reshape(-1, 4, 4)


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, Sv, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    s = np.trace(np.diag(Sv) @ D) / (Xc ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


T0 = load("t2_sfm_bu1035cf"); X0 = T0[IDX, :3, 3]
names = list(SC); Dl = []
for k in names:
    X = load(k)[IDX, :3, 3]; s, R, t = umeyama(X, X0); Dl.append((s * X @ R.T + t) - X0)
Dl = np.array(Dl)                                     # m x N x 3
N = len(IDX)


def solve(sel, scores, sig):
    D = Dl[sel].reshape(len(sel), -1) / np.sqrt(N)
    nd2 = (D ** 2).sum(1)
    b = (scores ** 2 - F0 ** 2 - sig ** 2 * nd2) / (2 * sig)
    G = D @ D.T; w, V = np.linalg.eigh(G); keep = w > 1e-6 * w.max()
    alpha = (V[:, keep] / w[keep]) @ V[:, keep].T @ b
    return alpha @ D * np.sqrt(N)                     # e_span as N*3 vector (metres), mean-normalised inner product


def predict(e, k, sig):
    d = Dl[k].ravel(); return np.sqrt(F0 ** 2 + 2 * sig * (e @ d) / N + sig ** 2 * (d @ d) / N)


sc = np.array([SC[k] for k in names])
for fam, sel in (("near7", [0, 1, 2, 3, 4, 5, 6]), ("all13", list(range(len(names))))):
    print(f"== {fam}: {[names[i].replace('t2_', '') for i in sel]}")
    # Monte Carlo on rounding and sigma
    mags = []
    for _ in range(400):
        sig = rng.uniform(1.45, 1.56); s_ = sc[sel] + rng.uniform(-0.0005, 0.0005, len(sel))
        e = solve(sel, s_, sig); mags.append(np.sqrt((e ** 2).sum() / N))
    mags = np.array(mags)
    print(f"   |e_span| over rounding/sigma noise: median {np.median(mags):.3f}, 5-95% {np.percentile(mags, 5):.3f}-{np.percentile(mags, 95):.3f} m")
    # leave-one-out prediction
    for j in sel:
        rest = [i for i in sel if i != j]
        e = solve(rest, sc[rest], 1.5)
        print(f"   LOO {names[j]:22s} actual {sc[j]:.3f}  predicted {predict(e, j, 1.5):.3f}  (no-info {predict(np.zeros(N*3), j, 1.5):.3f})")

# shape of e_span (near7), in X0's local road frame
sel = [0, 1, 2, 3, 4, 5, 6]
e = solve(sel, sc[sel], 1.5).reshape(N, 3)
up = -T0[IDX, :3, 1].mean(0); up /= np.linalg.norm(up)
o = 0
print("\nshape of e_span (near7), metres; along = X0's direction of travel, left = up x along")
for p, (f0_, f1_) in PASSES.items():
    P = T0[f0_:f1_, :3, 3]; tg = np.gradient(P, axis=0); tg -= np.outer(tg @ up, up); tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    lf = np.cross(up, tg); ee = e[o:o + f1_ - f0_]
    al, la, vu = (ee * tg).sum(1), (ee * lf).sum(1), ee @ up
    for a_, b_, nm in LEGS[p]:
        m = slice(a_ - f0_, b_ - f0_)
        print(f"   {p} {nm:6s} {a_}-{b_}: along mean {al[m].mean():+.3f} (rms {np.sqrt((al[m]**2).mean()):.3f}), "
              f"left mean {la[m].mean():+.3f} (rms {np.sqrt((la[m]**2).mean()):.3f}), up mean {vu[m].mean():+.3f} (rms {np.sqrt((vu[m]**2).mean()):.3f})")
    o += f1_ - f0_
tot = (e ** 2).sum(1)
print(f"   energy shares: along/left/up computed above; total rms {np.sqrt(tot.mean()):.3f} m")
np.save("Track2/result/e_span_near7.npy", e)
