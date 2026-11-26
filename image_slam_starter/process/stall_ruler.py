"""Right-side parking-stall separator lines as an along-track ruler (Track2).
Per frame: BEV of the road plane (x 3..11 ahead, y -1..-6 right, 0.02 h cells; road plane with the per-frame road roll),
luminance top-hat along x (thin line) averaged along y over 0.35 h windows (separator is long in y), vehicles/people from
seg.npz 'obj' (projected, dilated) excluded; per row the best window in the stall band gives a profile along x.  Peaks are
the separators; a peak is followed to the next frame (shift = trajectory step) and the fractional frame at which it passes
x0 (several x0) is interpolated.  Spacing of consecutive separators = arc length between consecutive crossings (trajectory
units; a fixed x0 cancels the BEV scale).  Only rows where stalls are on the right are measured (STALL segments below).
usage: python process/stall_ruler.py Track2 traj1.csv traj2.csv ... [--force] [--out DIR]   (from image_slam_starter)"""
import argparse, os, sys
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, ".")
from utils import nhc

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("trajs", nargs="*")
ap.add_argument("--force", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--x0s", default="5,6,7,8")
ap.add_argument("--ylo", type=float, default=-5.0); ap.add_argument("--yhi", type=float, default=-2.0)
ap.add_argument("--thr", type=float, default=5.0); ap.add_argument("--minlen", type=int, default=8)
a = ap.parse_args()
RES, XN, XF, Y0, Y1 = 0.02, 3.0, 11.0, -1.0, -6.0
K = np.array([[1035.5, 0, 722.17], [0, 1035.5 * 1038.55506 / 1040.18078, 456.83], [0, 0, 1.0]]); D = np.array([-0.05833, 0.088413, 0, 0, 0.])
F, L, U = nhc.axes(*np.radians((-0.14, -1.88)))
xs = np.arange(XF, XN, -RES); ys = np.arange(Y0, Y1, -RES)
gx, gy = np.meshgrid(xs, ys, indexing="ij")
files = sorted(os.listdir(f"{a.target}/data/images"))
OUT = a.out or f"{a.target}/result"
roll_all = np.load(f"{a.target}/result/road_roll.npy")
SEGOBJ = None


def uv_for(roll_deg):
    P = gx[..., None] * F + gy[..., None] * L + (gy * np.tan(np.radians(roll_deg)) - 1.0)[..., None] * U
    return cv2.projectPoints(P.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)


def work(i):
    global SEGOBJ
    if SEGOBJ is None:
        SEGOBJ = np.load(f"{a.target}/seg.npz")["obj"]
    uv = uv_for(roll_all[i])
    im = cv2.imread(f"{a.target}/data/images/{files[i]}")
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.remap(g, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=-1)
    ob = cv2.remap(SEGOBJ[i].astype(np.float32), uv[..., 0] / 4, uv[..., 1] / 4, cv2.INTER_LINEAR, borderValue=255)
    bad = (b < 0) | (ob > 60)
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((int(0.15 / RES) | 1,) * 2)) > 0     # cars / people / outside image
    b[b < 0] = 0
    b = cv2.GaussianBlur(b, (0, 0), sigmaX=0.6, sigmaY=1.2)
    th = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.22 / RES) | 1)))   # thin along x
    th[bad] = 0
    w = int(0.35 / RES)
    ok = (~bad).astype(np.float32)
    num = cv2.boxFilter(th, -1, (w, 1), normalize=True); den = cv2.boxFilter(ok, -1, (w, 1), normalize=True)
    sc = np.where(den > 0.8, num / np.maximum(den, 1e-3), 0)
    c0, c1 = int((Y0 - a.yhi) / RES), int((Y0 - a.ylo) / RES)
    s = sc[:, c0:c1]
    return s.max(1).astype(np.float16), (c0 + s.argmax(1)).astype(np.int16), (den[:, c0:c1] > 0.8).mean(1).astype(np.float16)


if __name__ == "__main__":
    N = len(files)
    cache = f"{OUT}/stall_prof.npz"
    if os.path.exists(cache) and not a.force:
        d = np.load(cache); prof, ypos, vis = d["prof"], d["ypos"], d["vis"]
    else:
        with Pool(24) as pool:
            R = pool.map(work, range(N), chunksize=8)
        prof = np.array([r[0] for r in R], np.float32); ypos = np.array([r[1] for r in R]); vis = np.array([r[2] for r in R], np.float32)
        np.savez(cache, prof=prof, ypos=ypos, vis=vis)
    print("profiles", prof.shape)

    # ---- peaks per frame --------------------------------------------------------------------------------------------
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    ps = gaussian_filter1d(prof, 1.5, axis=1)
    PK = []
    for i in range(N):
        p = ps[i]
        k, pr = find_peaks(p, height=a.thr, prominence=a.thr * 0.6, distance=int(0.5 / RES))
        k = k[(k > 3) & (k < len(p) - 4)]
        sub = k + 0.5 * (p[k - 1] - p[k + 1]) / (p[k - 1] - 2 * p[k] + p[k + 1] - 1e-9)
        PK.append((XF - sub * RES, p[k]))
    Tr = {t: np.genfromtxt(t, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4) for t in a.trajs}
    T0 = next(iter(Tr.values()))
    st = np.zeros(N)
    for i in range(N - 1):
        st[i] = (np.linalg.inv(T0[i])[:3, :3] @ (T0[i + 1, :3, 3] - T0[i, :3, 3])) @ F / 1.2
    x0s = [float(v) for v in a.x0s.split(",")]
    # ---- chains of matched peaks = separator tracks -------------------------------------------------------------------
    nxt = {}
    for i in range(N - 1):
        if st[i] < 0.05:
            continue
        e0, h0 = PK[i]; e1, h1 = PK[i + 1]
        for j, x in enumerate(e0):
            c = np.where(np.abs(e1 - (x - st[i])) < 0.25 * st[i] + 0.04)[0]
            if len(c) == 1:
                nxt[(i, j)] = (i + 1, c[0])
    prv = {v: k for k, v in nxt.items()}
    tracks = []
    for k in nxt:
        if k in prv:
            continue
        ch = [k]
        while ch[-1] in nxt:
            ch.append(nxt[ch[-1]])
        if len(ch) >= a.minlen:
            tracks.append(ch)
    print(len(tracks), "tracks with >= ", a.minlen, "frames")
    TR = []                      # per track: frames, x, height
    for ch in tracks:
        fr = np.array([c[0] for c in ch]); x = np.array([PK[c[0]][0][c[1]] for c in ch]); h = np.array([PK[c[0]][1][c[1]] for c in ch])
        TR.append((fr, x, h))
    Tr = {t: np.genfromtxt(t, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4) for t in a.trajs}
    SEG = [(s.split(":")[0], *map(int, s.split(":")[1].split("-"))) for s in
           ("A:880-1130", "B:1380-1500", "C:2000-2080", "D:2560-2800", "E:2840-2930", "F:3080-3270")]
    rng = np.random.default_rng(0)
    # step-consistency of the tracks: BEV x per unit trajectory step
    sl = []
    for fr, x, h in TR:
        A = np.polyfit(fr, x, 1); res = x - np.polyval(A, fr)
        sl.append((A[0], res.std()))
    sl = np.array(sl)
    stm = np.array([np.mean(st[fr]) for fr, x, h in TR])
    ratio = -sl[:, 0] / stm
    good = (np.abs(ratio - np.median(ratio)) < 0.25) & (sl[:, 1] < 0.12)
    print(f"tracks: BEV/traj step ratio median {np.median(ratio):.3f}, kept {good.sum()}/{len(good)} (linear, residual < 0.12)")
    for name, T in Tr.items():
        C = T[:, :3, 3]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        cross = []                           # per track: dict x0 -> arc length s at crossing, plus mean frame
        for (fr, x, h), g in zip(TR, good):
            if not g:
                continue
            sv = s[fr]
            A = np.polyfit(sv, x, 1); d = {}
            for x0 in x0s:
                if x.min() - 0.1 <= x0 <= x.max() + 0.1:
                    d[x0] = (x0 - A[1]) / A[0]
            if d:
                cross.append((fr.mean(), d, h.mean()))
        cross.sort(key=lambda c: c[0])
        rows = []
        for k in range(len(cross) - 1):
            common = [x0 for x0 in cross[k][1] if x0 in cross[k + 1][1]]
            if not common or cross[k + 1][0] - cross[k][0] > 80:
                continue
            rows.append((0.5 * (cross[k][0] + cross[k + 1][0]), np.median([cross[k + 1][1][x0] - cross[k][1][x0] for x0 in common]), len(common),
                         min(cross[k][2], cross[k + 1][2])))
        rows = np.array(rows)
        np.save(f"{OUT}/stall_spacings_{os.path.basename(name)[:-4]}.npy", rows)
        print("==", os.path.basename(name), f"{len(cross)} tracks")
        for sn, a0, a1 in SEG:
            r = rows[(rows[:, 0] >= a0) & (rows[:, 0] < a1)]
            if len(r) < 2:
                print(sn, a0, a1, "few", len(r)); continue
            h, b = np.histogram(r[:, 1], bins=np.arange(0.4, 4.6, 0.2))
            print(f"{sn} {a0}-{a1}: n {len(r)}  spacing hist (0.2 bins from 0.4):", " ".join(map(str, h)))
