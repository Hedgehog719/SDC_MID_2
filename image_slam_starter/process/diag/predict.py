"""Predict the Track02 public score of trajectories from the e_span read-out around t2_sfm_bu1035cf (near7 and all13)."""
import sys
import numpy as np

S = "../submissions/track2/"
PASSES = {"W": (1004, 1206), "E": (2881, 3100)}
IDX = np.r_[tuple(np.arange(*p) for p in PASSES.values())]
N = len(IDX); SIG = 1.5; F0 = 0.203
SC = dict(t2_sfm_bu1035_sf=0.210, t2_sfm_bu1035_rl_r0=0.249, t2_sfm_pp06zr=0.218, t2_sfm_pp06zrl=0.225, t2_sfm_pp06zr3=0.213,
          t2_probe_slide=0.213, t2_probe_pts=0.280, t2_sfm_pp06z=0.330, t2_sfm_pp10z=0.333, t2_dash1r=0.421, t2_once=0.432,
          t2_seg=0.431, t2_once_cars=0.434)
EXTRA = {"t2_pch_bu1035cf": ("../submissions/track2/t2_pch_bu1035cf.csv", 0.185),      # scored 2026-11-20 19:07
         "t2_pc3_bu1035_sf": ("../submissions/track2/t2_pc3_bu1035_sf.csv", 0.177),
         "t2_pcm_bu1035_sf": ("../submissions/track2/t2_pcm_bu1035_sf.csv", 0.154),
         "t2_sfmx_q_r0": ("../submissions/track2/t2_sfmx_q_r0.csv", 0.186)}      # scored 2026-11-20 19:40


def load(p):
    return np.genfromtxt(p, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, Sv, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    s = np.trace(np.diag(Sv) @ D) / (Xc ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


X0 = load(S + "t2_sfm_bu1035cf.csv")[IDX, :3, 3]


def delta(X):
    s, R, t = umeyama(X, X0); return (s * X @ R.T + t) - X0


names = list(SC) + list(EXTRA)
paths = [S + k + ".csv" for k in SC] + [v[0] for v in EXTRA.values()]
sc = np.array(list(SC.values()) + [v[1] for v in EXTRA.values()])
Dl = np.array([delta(load(p)[IDX, :3, 3]) for p in paths])


def espan(sel):
    D = Dl[sel].reshape(len(sel), -1) / np.sqrt(N); nd2 = (D ** 2).sum(1)
    b = (sc[sel] ** 2 - F0 ** 2 - SIG ** 2 * nd2) / (2 * SIG)
    G = D @ D.T; w, V = np.linalg.eigh(G); keep = w > 1e-6 * w.max()
    return ((V[:, keep] / w[keep]) @ V[:, keep].T @ b) @ D * np.sqrt(N)


E7 = espan(list(range(7))).reshape(N, 3); E13 = espan(list(range(13))).reshape(N, 3)
EX = espan(list(range(7)) + list(range(13, 13 + len(EXTRA)))).reshape(N, 3)          # near7 + the new pipeline scores


def pred(E, d):
    return np.sqrt(max(F0 ** 2 + 2 * SIG * (E * d).sum(1).mean() + SIG ** 2 * (d ** 2).sum(1).mean(), 0))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = delta(load(p)[IDX, :3, 3])
        print(f"{p.split('/')[-1]:40s} |Pd| {np.sqrt((d**2).sum(1).mean()):.3f} u   predicted near7 {pred(E7, d):.3f}  all13 {pred(E13, d):.3f}  near7+new {pred(EX, d):.3f}")
