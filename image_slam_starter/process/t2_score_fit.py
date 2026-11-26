"""What the Track02 public scores already paid for say about the truth on the public frames (no new submission).

score_k = sigma * RMSE( Sim3(T_k) -> G ) on the 421 public frames (westbound 1141-1344, eastbound 2731-2947), G unknown.
G is written as REF + Delta, Delta along REF's direction of travel only (the submissions differ almost only along the road,
NOTES 11-19), and models of Delta are compared by leave-one-out prediction of the scores:
  zero    : Delta = 0                                    (REF is the truth up to an error e0 no submission explores)
  slide   : one constant along-road shift per pass         (2)
  latency : shift = speed * dt in the direction of travel (1; a truth-vs-image time offset)
  scale   : per pass a shift + a stretch along the pass    (4)
  knots   : per pass a piecewise-linear along-road profile, knots every K frames, smoothness lambda
Every model also fits sigma [m / unit] and e0 [m]: score^2 = sigma^2 RMSE^2 + e0^2.
usage (from image_slam_starter): python process/t2_score_fit.py [--ref t2_dash1r] [--max 1.0] [--knot 25] [--lam 3]
"""
import argparse
import numpy as np
from scipy.optimize import least_squares

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="t2_dash1r")
ap.add_argument("--max", type=float, default=1.0, help="use scores below this")
ap.add_argument("--knot", type=int, default=25)
ap.add_argument("--lam", type=float, default=3.0, help="knots: second-difference penalty [per unit]")
ap.add_argument("--sigma", type=float, default=0, help="fix sigma [m/unit] (0 = fit)")
ap.add_argument("--models", default="zero,slide,latency,scale,knots")
ap.add_argument("--predict", nargs="*", default=[], help="trajectory csv files to score with each fitted model")
ap.add_argument("--save", default=None, help="save the knots model's along-road profile (npz)")
a = ap.parse_args()

SCORES = dict(t2_rev_hs0=0.638, t2_ret_hs0=0.645, t2_ret=0.967, t2_bev=0.549, t2_seg=0.431, t2_seg_tight=0.484, t2_pc=0.557,
              t2_once=0.432, t2_seg_hs2=0.673, t2_seg_da05=0.462, t2_once_da05=0.461, t2_once_cars=0.434, t2_dash1r=0.421,
              t2_dash1r_da05=0.461, t2_dash05r=0.436, t2_cam=0.577, t2_cam_z16_it3=0.506,
              t2_v1=4.227, t2_lr=4.369, t2_sfm_fr=2.094, t2_pw2s104=0.421, t2_pw1s104=0.477, t2_sfm_pp06z=0.330, t2_sfm_pp10z=0.333, t2_sfm_pp06zr=0.218, t2_sfm_pp06zrl=0.225, t2_sfm_pp06zr3=0.213, t2_sfm_bu1035=0.203, t2_sfm_bu1035_rl_r0=0.249, t2_sfm_bu1035_sf=0.210, t2_sfm_bu105=0.206, t2_sfm_bu1035cf=0.203)
PASSES = {"W": (1004, 1206), "E": (2881, 3100)}       # public set refitted 11-19 (t2_public_set.py)
CUTS = {"W": (1076, 1136), "E": (2941, 3025)}        # north-south leg | corner | east-west leg
IDX = np.r_[tuple(np.arange(*p) for p in PASSES.values())]


def load(t):
    d = np.genfromtxt(f"../submissions/track2/{t}.csv", delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1:].reshape(-1, 4, 4)


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


def rmse(X, G):
    s, R, t = umeyama(X, G)
    return np.sqrt(((s * X @ R.T + t - G) ** 2).sum(1).mean())


ts, Tref = load(a.ref)
G0 = Tref[IDX, :3, 3]
# direction of travel and speed of REF on the public frames
tang = np.zeros_like(G0); speed = np.zeros(len(IDX))
o = 0
for f0, f1 in PASSES.values():
    n = f1 - f0
    P = Tref[f0:f1, :3, 3]; tt = ts[f0:f1] * 1e-9
    d = np.gradient(P, axis=0); tang[o:o + n] = d / np.linalg.norm(d, axis=1, keepdims=True)
    speed[o:o + n] = np.linalg.norm(np.gradient(P, tt, axis=0), axis=1)
    o += n
nW = PASSES["W"][1] - PASSES["W"][0]
names = [k for k, v in SCORES.items() if v < a.max]
S = np.array([SCORES[k] for k in names])
X = [load(k)[1][IDX, :3, 3] for k in names]
print(f"REF {a.ref}; {len(names)} scores < {a.max}; REF speed on public: W median {np.median(speed[:nW]):.2f}, E {np.median(speed[nW:]):.2f} units/s")


def basis(model):
    """along-road displacement basis, (len(IDX), p)"""
    if model == "zero":
        return np.zeros((len(IDX), 0))
    if model == "slide":
        B = np.zeros((len(IDX), 2)); B[:nW, 0] = 1; B[nW:, 1] = 1; return B
    if model == "latency":
        return speed[:, None].copy()
    if model == "scale":
        B = np.zeros((len(IDX), 4))
        for j, (sl, n) in enumerate(((slice(0, nW), nW), (slice(nW, None), len(IDX) - nW))):
            B[sl, 2 * j] = 1; B[sl, 2 * j + 1] = np.linspace(-1, 1, n)
        return B
    if model == "legslide":
        # one along-road slide per leg and pass (W NS, W EW, E EW, E NS); corner frames interpolate between the two legs
        cols = []
        for (nm, (f0, f1)), o in zip(PASSES.items(), (0, nW)):
            n = f1 - f0; c0, c1 = CUTS[nm][0] - f0, CUTS[nm][1] - f0
            w = np.clip((np.arange(n) - c0) / (c1 - c0), 0, 1)
            a_ = np.zeros(len(IDX)); a_[o:o + n] = 1 - w; b_ = np.zeros(len(IDX)); b_[o:o + n] = w
            cols += [a_, b_]
        return np.array(cols).T
    if model == "seg":
        # per pass a slide + a stretch of each of its three parts (displacement ramps 0 -> 1 across the part, stays after)
        cols = []
        for (nm, (f0, f1)), o in zip(PASSES.items(), (0, nW)):
            n = f1 - f0; c = np.zeros(len(IDX)); c[o:o + n] = 1; cols.append(c)
            b = [0, CUTS[nm][0] - f0, CUTS[nm][1] - f0, n]
            for lo, hi in zip(b[:-1], b[1:]):
                c = np.zeros(len(IDX)); c[o:o + n] = np.clip((np.arange(n) - lo) / (hi - lo), 0, 1); cols.append(c)
        return np.array(cols).T
    if model == "knots":
        cols = []
        for sl, n in ((slice(0, nW), nW), (slice(nW, None), len(IDX) - nW)):
            k = np.linspace(0, n - 1, max(2, int(round(n / a.knot)) + 1))
            for i in range(len(k)):
                e = np.zeros(len(k)); e[i] = 1
                c = np.zeros(len(IDX)); c[sl] = np.interp(np.arange(n), k, e); cols.append(c)
        return np.array(cols).T


def blocks(model, B):
    """second-difference penalty rows for the knots model (within each pass)"""
    if model != "knots":
        return np.zeros((0, B.shape[1]))
    rows = []
    nk = B.shape[1] // 2 if B.shape[1] % 2 == 0 else None
    kW = int((B[:nW] != 0).any(0).sum())
    for start, cnt in ((0, kW), (kW, B.shape[1] - kW)):
        for i in range(start, start + cnt - 2):
            r = np.zeros(B.shape[1]); r[i], r[i + 1], r[i + 2] = 1, -2, 1; rows.append(r)
    return np.array(rows) * a.lam


# road axes of REF on the public frames: u along the road (principal axis), v lateral, w vertical (REF camera -Y mean)
_c = G0.mean(0); _U = np.linalg.svd(G0 - _c)[2]; AX_U = _U[0]
_up = -Tref[IDX, :3, 1].mean(0); AX_W = _up - (_up @ AX_U) * AX_U; AX_W /= np.linalg.norm(AX_W); AX_V = np.cross(AX_W, AX_U)
LEFT = np.cross(np.repeat(AX_W[None], len(IDX), 0), tang); LEFT /= np.linalg.norm(LEFT, axis=1, keepdims=True)
def _rot(v, ang):
    k = AX_W; return v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + np.outer(v @ k, k) * (1 - np.cos(ang))


def legyaw(c):
    G = G0.copy()
    for j, (nm, (f0, f1)) in enumerate(PASSES.items()):
        o = 0 if nm == "W" else nW
        piv = Tref[CUTS[nm][0] if nm == "W" else CUTS[nm][1], :3, 3]
        fr = np.arange(f0, CUTS[nm][0]) if nm == "W" else np.arange(CUTS[nm][1], f1)
        ii = o + fr - f0
        G[ii] = piv + _rot(G0[ii] - piv, np.radians(c[j]))
    return G


GEN = {"sep": (1, lambda c: G0 + c[0] / 2 * LEFT),
       "aniso": (2, lambda c: _c + np.outer((G0 - _c) @ AX_U, AX_U) + (1 + c[0]) * np.outer((G0 - _c) @ AX_V, AX_V)
                 + (1 + c[1]) * np.outer((G0 - _c) @ AX_W, AX_W)),
       "legyaw": (2, lambda c: legyaw(c)),
       "latscale": (1, lambda c: _c + np.outer((G0 - _c) @ AX_U, AX_U) + (1 + c[0]) * np.outer((G0 - _c) @ AX_V, AX_V)
                    + np.outer((G0 - _c) @ AX_W, AX_W))}


def fit(model, use):
    if model in GEN:
        p, gfun = GEN[model]; B = np.zeros((len(IDX), p)); L = np.zeros((0, p))
    else:
        B = basis(model); L = blocks(model, B); p = B.shape[1]; gfun = None

    def pred(x, sel):
        c, sig, e0 = x[:p], x[p], x[p + 1]
        G = gfun(c) if gfun else G0 + (B @ c)[:, None] * tang
        return np.sqrt(sig ** 2 * np.array([rmse(X[i], G) for i in sel]) ** 2 + e0 ** 2)

    def res(x):
        return np.r_[(pred(x, use) - S[use]) / 0.005, L @ x[:p]]
    x0 = np.r_[np.zeros(p), a.sigma or 1.56, 0.3]
    lo_s, hi_s = (a.sigma - 1e-6, a.sigma + 1e-6) if a.sigma else (0.5, 4.0)
    r = least_squares(res, x0, bounds=(np.r_[np.full(p, -np.inf), lo_s, 0.0], np.r_[np.full(p, np.inf), hi_s, 2.0]))
    return r.x, pred, B


for model in a.models.split(","):
    x, pred, B = fit(model, np.arange(len(names)))
    p = B.shape[1]
    fitres = pred(x, range(len(names))) - S
    loo = []
    for k in range(len(names)):
        use = np.array([i for i in range(len(names)) if i != k])
        xk, pk, _ = fit(model, use)
        loo.append(pk(xk, [k])[0] - S[k])
    loo = np.array(loo)
    extra = ""
    if model == "legslide":
        extra = f"  W: NS {x[0]:+.3f}, EW {x[1]:+.3f};  E: EW {x[2]:+.3f}, NS {x[3]:+.3f}  units along each frame's travel"
    if model == "slide":
        extra = f"  shift W {x[0]:+.3f}, E {x[1]:+.3f} units along each pass's travel"
    if model == "latency":
        extra = f"  dt {x[0] * 1000:+.1f} ms (shift W {x[0] * np.median(speed[:nW]):+.3f}, E {x[0] * np.median(speed[nW:]):+.3f} units)"
    if model in GEN:
        extra = "  params " + ", ".join(f"{v:+.4f}" for v in x[:p])
    if model == "seg":
        extra = (f"  W slide {x[0]:+.3f}, NS/corner/EW stretch {x[1]:+.3f} {x[2]:+.3f} {x[3]:+.3f}; "
                 f"E slide {x[4]:+.3f}, EW/corner/NS stretch {x[5]:+.3f} {x[6]:+.3f} {x[7]:+.3f}  [units of extra length]")
    if model == "scale":
        extra = f"  W shift {x[0]:+.3f} stretch {x[1]:+.3f}; E shift {x[2]:+.3f} stretch {x[3]:+.3f}"
    print(f"{model:8s} p={p:2d}: sigma {x[p]:.3f} m/unit, e0 {x[p + 1]:.3f} m, fit rms {np.sqrt((fitres ** 2).mean()):.4f}, "
          f"LOO rms {np.sqrt((loo ** 2).mean()):.4f}, LOO max |{names[np.abs(loo).argmax()]}| {np.abs(loo).max():.3f}{extra}")
    if model == "knots":
        prof = B @ x[:p]
        for nm, sl in (("W", slice(0, nW)), ("E", slice(nW, None))):
            fr = np.arange(*PASSES[nm])
            pts = list(range(0, len(fr), a.knot)) + [len(fr) - 1]
            print(f"   along-road correction to {a.ref}, {nm}: " + "  ".join(f"{fr[i]}:{prof[sl][i]:+.2f}" for i in pts))
        print("   best predicted score if the truth were REF + profile: e0 =", round(x[p + 1], 3))
        if a.save:
            np.savez(a.save, idx=IDX, prof=prof, tang=tang, sigma=x[p], e0=x[p + 1], names=names, S=S)
    if a.predict:
        G = G0 + (B @ x[:p])[:, None] * tang if model not in GEN else GEN[model][1](x[:p])
        out = []
        for f in a.predict:
            Pp = np.genfromtxt(f, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)[IDX, :3, 3]
            out.append((np.sqrt((x[p] * rmse(Pp, G)) ** 2 + x[p + 1] ** 2), f.split("/")[-1][:-4]))
        print("   predicted: " + ", ".join(f"{n} {v:.3f}" for v, n in sorted(out)))
    print("   LOO residuals: " + ", ".join(f"{n[3:]} {r:+.3f}" for n, r in zip(names, loo)))
