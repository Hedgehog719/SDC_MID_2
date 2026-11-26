"""Yellow dashed centre line as an along-track ruler for Track2 (second attempt).
The centre line is yellow, dashes ~2.5 camera heights, period ~5.4 heights.  Per frame: BEV (road plane, 0.02 h cells),
yellowness map (R+G)/2-B, thin-line top-hat, lateral position of the line tracked with a temporal median, 1D profile
along x inside a +-3 cell band.  Profile is binarised (mid-level between its own low/high, contrast required) and the
near end (on->off going to the car) and far start of each dash are located to sub-cell accuracy.  Edges are matched
between consecutive frames, and the fractional frame at which an edge crosses x0 (several x0) is interpolated; the arc
length between consecutive crossings of the same edge type = one period (trajectory units).  Crossing at a fixed x0
means the BEV scale (pitch/height) error cancels to first order.
usage: python process/dash_ruler2.py Track2 traj1.csv traj2.csv ... [--force] [--out DIR]"""
import argparse, os, sys
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, ".")
from utils import nhc

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("trajs", nargs="*")
ap.add_argument("--force", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--latlo", type=float, default=0.0); ap.add_argument("--lathi", type=float, default=1.6)
ap.add_argument("--x0s", default="3,4,5,6,7")
a = ap.parse_args()
RES, XN, XF, YW = 0.02, 1.5, 11.0, 3.0
K = np.array([[1035.5, 0, 722.17], [0, 1035.5 * 1038.55506 / 1040.18078, 456.83], [0, 0, 1.0]]); D = np.array([-0.05833, 0.088413, 0, 0, 0.])
F, L, U = nhc.axes(*np.radians((-0.14, -1.88)))
xs = np.arange(XF, XN, -RES); ys = np.arange(YW, -YW, -RES)
gx, gy = np.meshgrid(xs, ys, indexing="ij")
P = gx[..., None] * F + gy[..., None] * L - U
uv = cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)
files = sorted(os.listdir(f"{a.target}/data/images"))
OUT = a.out or f"{a.target}/result"


def yellow_cols(i):
    """returns column-sum of line yellowness (rows 2.5..9), and the full per-frame yellowness map (float16) restricted to later band extraction"""
    im = cv2.imread(f"{a.target}/data/images/{files[i]}").astype(np.float32)
    b = cv2.remap(im, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
    valid = b[..., 0] >= 0
    y = np.minimum(b[..., 1], b[..., 2]) - b[..., 0]
    y[~valid] = 0
    th = cv2.morphologyEx(y, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.35 / RES) | 1, 1)))
    th[~valid] = 0
    return th.astype(np.float16)


def work(i):
    return yellow_cols(i)


def edges_of(p, x_lo, x_hi):
    """near-end (on->off toward car) and far-start edges, x in heights; p: profile indexed far(row0)->near"""
    lo, hi = np.percentile(p, 10), np.percentile(p, 95)
    if hi - lo < 6.0:
        return None
    mid = (lo + hi) / 2
    on = p > mid
    near, far = [], []
    for k in np.where(on[:-1] != on[1:])[0]:
        v0, v1 = p[k], p[k + 1]
        x = XF - (k + (v0 - mid) / (v0 - v1 + 1e-9)) * RES
        (near if on[k] else far).append(x)
    return np.array(near), np.array(far)


if __name__ == "__main__":
    N = len(files)
    cache = f"{OUT}/dash2_prof.npz"
    if os.path.exists(cache) and not a.force:
        d = np.load(cache); prof = d["prof"]; lat = d["lat"]; qual = d["qual"]
    else:
        with Pool(16) as pool:
            Ms = pool.map(work, range(N), chunksize=8)
        # lateral position: column sum of the top-hat map over x in [2.5,9]
        r0, r1 = int((XF - 9) / RES), int((XF - 2.5) / RES)
        cs = np.array([m[r0:r1].astype(np.float32).mean(0) for m in Ms])          # N x cols
        cs = cv2.GaussianBlur(cs, (0, 0), sigmaX=2, sigmaY=0.001)
        cmask = (ys > a.latlo) & (ys < a.lathi)                                  # centre line is left of the car (y>0)
        cs[:, ~cmask] = 0
        c_raw = cs.argmax(1).astype(float)
        c_med = np.array([np.median(c_raw[max(0, i - 25):i + 26]) for i in range(N)])
        # temporal continuity: pick strongest column within +-25 cells of running median
        lat_c = np.zeros(N, int)
        for i in range(N):
            lo = int(max(0, c_med[i] - 25)); hi = int(min(cs.shape[1], c_med[i] + 26))
            lat_c[i] = lo + cs[i, lo:hi].argmax()
        lat_c = np.round([np.median(lat_c[max(0, i - 7):i + 8]) for i in range(N)]).astype(int)
        prof = np.zeros((N, len(xs)), np.float32)
        for i in range(N):
            c = lat_c[i]
            prof[i] = Ms[i][:, max(0, c - 3):c + 4].astype(np.float32).mean(1)
        lat = ys[lat_c]; qual = cs[np.arange(N), lat_c]
        os.makedirs(OUT, exist_ok=True)
        np.savez(cache, prof=prof, lat=lat, qual=qual)
    x0s = [float(v) for v in a.x0s.split(",")]
    E = [edges_of(prof[i], 0, 0) for i in range(N)]
    Tr = {t: np.genfromtxt(t, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4) for t in a.trajs}
    T0 = next(iter(Tr.values()))
    st = np.zeros(N)
    for i in range(N - 1):
        st[i] = (np.linalg.inv(T0[i])[:3, :3] @ (T0[i + 1, :3, 3] - T0[i, :3, 3])) @ F / 1.2
    ev = []     # (type 0 near/1 far, x0, fractional frame, frame i)
    ratios = []
    for i in range(N - 1):
        if E[i] is None or E[i + 1] is None or st[i] < 0.05 or abs(lat[i] - lat[i + 1]) > 0.2:
            continue
        for ty in (0, 1):
            e0, e1 = E[i][ty], E[i + 1][ty]
            for x in e0:
                m = e1[np.abs(e1 - (x - st[i])) < 0.5 * st[i] + 0.05]
                if len(m) != 1:
                    continue
                ratios.append((m[0] - x) / -st[i])
                for x0 in x0s:
                    if m[0] < x0 <= x:
                        ev.append((ty, x0, i + (x - x0) / (x - m[0]), i, (x - m[0]) / st[i]))
    ev = np.array(ev)
    rr = np.array(ratios)
    print(f"BEV step / trajectory step ratio: median {np.median(rr):.3f}, MAD {1.48*np.median(abs(rr-np.median(rr))):.3f}")
    rmed = np.median(rr)
    ev = ev[np.abs(ev[:, 4] / rmed - 1) < 0.35]              # edge motion consistent with the vehicle step
    print(f"{len(ev)} edge crossings")
    np.save(f"{OUT}/dash2_events.npy", ev)

    # ---- periods: consecutive crossings of the same edge type at the same x0 --------------------------------------
    SEG = [("out1", 190, 350), ("out2", 450, 580), ("out3", 680, 760), ("back1", 3330, 3440), ("back2", 3550, 3660),
           ("back3", 3770, 4000), ("topW", 1141, 1344), ("topE", 2731, 2947)]
    PLO, PHI = 4.5, 8.5          # accepted period range, trajectory units (missed dash -> ~12.7 and double count -> <3 rejected)
    rng = np.random.default_rng(0)
    for name, T in Tr.items():
        C = T[:, :3, 3]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        rows = []                                           # frame, period, edge type, x0, frames between crossings
        for ty in (0, 1):
            for x0 in x0s:
                e = ev[(ev[:, 0] == ty) & (ev[:, 1] == x0)]
                e = e[np.argsort(e[:, 2])]
                t_ = e[:, 2]; sv = np.interp(t_, np.arange(N), s)
                for k in range(len(t_) - 1):
                    per = sv[k + 1] - sv[k]
                    if PLO < per < PHI and 5 < t_[k + 1] - t_[k] < 40:
                        rows.append((t_[k], per, ty, x0, t_[k + 1] - t_[k]))
        rows = np.array(rows)
        np.save(f"{OUT}/dash2_periods_{os.path.basename(name)[:-4]}.npy", rows)
        print(f"== {os.path.basename(name)}  (period in trajectory units; n_dash = periods of one edge type at one x0)")
        res = {}
        for sn, a0, a1 in SEG:
            r = rows[(rows[:, 0] >= a0) & (rows[:, 0] < a1)]
            if len(r) < 3:
                print(f"{sn:6s} {a0}-{a1}: too few ({len(r)})"); continue
            # per-dash average over the 10 (type, x0) combinations: group by the frame of the dash (round to 4 frames)
            key = np.round(r[:, 0] / 6).astype(int)
            ks = np.unique(key); pd_ = np.array([np.median(r[key == k, 1]) for k in ks])
            med = np.median(pd_)
            bs = [np.median(rng.choice(pd_, len(pd_))) for _ in range(500)]
            sc = 1.4826 * np.median(abs(pd_ - med))
            res[sn] = (med, np.std(bs), len(pd_))
            print(f"{sn:6s} {a0}-{a1}: dashes {len(pd_):3d}  median {med:.3f} +-{np.std(bs):.3f} (boot)  per-period robust sd {sc:.3f} ({100*sc/med:.1f}%)  "
                  f"speed {np.median(1/r[:,4]*r[:,1]):.2f} u/frame")
        if "out2" in res:
            def comb(names):
                v = np.array([res[n][0] for n in names if n in res]); w = np.array([res[n][2] for n in names if n in res])
                return np.average(v, weights=w) if len(v) else np.nan
            out, back = comb(["out1", "out2", "out3"]), comb(["back1", "back2", "back3"])
            print(f"  out-leg {out:.3f}  back-leg {back:.3f}  ratio back/out {back/out:.3f};  topE/topW {res.get('topE',(np.nan,))[0]/res.get('topW',(np.nan,))[0]:.3f}")
