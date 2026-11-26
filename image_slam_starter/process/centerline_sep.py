"""Lateral separation of two passes from the yellow centre line, independent of the trajectory.

Per frame (true lens, mount angles of the SLAM log): bird's-eye view of the road (0.02 h cells), yellow top-hat,
robust line fit y = y0 + k x to the centre line on the car's left (x in [2, 9] h) -> y0 = distance from the camera to
the line at the camera, k = heading of the car against the line.  Two passes driving opposite ways with the line between
them are y0_a + y0_b apart (camera heights); the trajectory says |lateral(P_b - P_a)| (units).  Their ratio is the
camera height in trajectory units; it should be the same on every leg if the trajectory's lateral spacing is right.
usage: python process/centerline_sep.py Track2 TRAJ.csv [TRAJ2.csv ...]
"""
import argparse, os, sys
import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import nhc

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("trajs", nargs="+")
ap.add_argument("--cam", default="1035.5,1033.88,722.17,456.84,-0.05833,0.088413,0,0")
ap.add_argument("--mount", default="-0.21,-1.49", help="mount yaw, pitch [deg] (vehicle constraint of the SLAM run)")
ap.add_argument("--frames", default="1004:1206,2881:3100")
a = ap.parse_args()
c = [float(x) for x in a.cam.split(",")]
K = np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1.0]]); D = np.array(c[4:8] + [0.0])
F, L, U = nhc.axes(*np.radians([float(x) for x in a.mount.split(",")]))
RES, XN, XF, YW = 0.02, 2.0, 9.0, 2.0
xs = np.arange(XF, XN, -RES); ys = np.arange(YW, -0.4, -RES)
gx, gy = np.meshgrid(xs, ys, indexing="ij")
Pg = gx[..., None] * F + gy[..., None] * L - U
uv = cv2.projectPoints(Pg.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)
files = sorted(os.listdir(f"{a.target}/data/images"))
frames = [f for r in a.frames.split(",") for f in range(*(int(x) for x in r.split(":")))]


def line_of(f):
    im = cv2.imread(f"{a.target}/data/images/{files[f]}").astype(np.float32)
    b = cv2.remap(im, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
    valid = b[..., 0] >= 0
    yel = np.minimum(b[..., 1], b[..., 2]) - b[..., 0]; yel[~valid] = 0
    th = cv2.morphologyEx(yel, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.35 / RES) | 1, 1)))
    ii, jj = np.where(th > max(12.0, np.percentile(th[valid], 99.5)))
    if len(ii) < 40:
        return None
    X, Y = xs[ii], ys[jj]
    best = None
    rng = np.random.default_rng(f)
    for _ in range(200):                                  # RANSAC line in the BEV
        p, q = rng.choice(len(X), 2, replace=False)
        if abs(X[p] - X[q]) < 1.0:
            continue
        k = (Y[q] - Y[p]) / (X[q] - X[p]); y0 = Y[p] - k * X[p]
        inl = np.abs(Y - (y0 + k * X)) < 0.06
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best is None or best[0] < 40:
        return None
    inl = best[1]
    k, y0 = np.polyfit(X[inl], Y[inl], 1)
    if np.ptp(X[inl]) < 2.0 or not (0.15 < y0 < 1.9) or abs(k) > 0.15:
        return None
    # second, parallel yellow line (double centre line): offsets of the other yellow pixels from the fitted line
    off = Y - (y0 + k * X)
    far = (np.abs(off) > 0.08) & (np.abs(off) < 0.6)
    dbl = 0.0
    if far.sum() >= 30:
        hb, e = np.histogram(off[far], bins=np.arange(-0.6, 0.62, 0.04))
        if hb.max() >= 30:
            dbl = float((e[hb.argmax()] + e[hb.argmax() + 1]) / 2)
    # occupancy along the line (dashed ~ 0.3-0.6, solid ~ 1): 0.1 h bins over the visible length
    xb = np.arange(XN, XF + 1e-9, 0.1)
    occ = float((np.histogram(X[inl], bins=xb)[0] > 0).mean())
    return y0, k, int(best[0]), dbl, occ


cache = f"{a.target}/result/centerline2_{a.frames.replace(':', '-').replace(',', '_')}_{a.mount.replace(',', '_')}.npz"
if os.path.exists(cache):
    z = np.load(cache); Y0, KK, NI = z["y0"], z["k"], z["n"]
else:
    Y0 = np.full(len(files), np.nan); KK = np.full(len(files), np.nan); NI = np.zeros(len(files), int)
    DBL = np.zeros(len(files)); OCC = np.full(len(files), np.nan)
    for f in frames:
        r = line_of(f)
        if r is not None:
            Y0[f], KK[f], NI[f], DBL[f], OCC[f] = r
    np.savez(cache, y0=Y0, k=KK, n=NI, double=DBL, occ=OCC)
ok = ~np.isnan(Y0)
print(f"centre line found in {ok[frames].sum()} / {len(frames)} frames")

LEGS = {"W": ((1004, 1076, "NS"), (1076, 1136, "corner"), (1136, 1206, "EW")), "E": ((2881, 2941, "EW"), (2941, 3025, "corner"), (3025, 3100, "NS"))}
for t in a.trajs:
    T = np.genfromtxt(t, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4); P = T[:, :3, 3]
    up = -T[1004:1206, :3, 1].mean(0); up /= np.linalg.norm(up)
    W = np.arange(1004, 1206); E = np.arange(2881, 3100)
    H = lambda Q: Q - np.outer(Q @ up, up)
    d, j = cKDTree(H(P[E])).query(H(P[W]))
    out = []
    for a_, b_, nm in LEGS["W"]:
        rat = []
        for i in range(a_, b_):
            jj = E[j[i - 1004]]
            if d[i - 1004] > 3.5 or not (ok[i] and ok[jj]):
                continue
            tg = P[min(i + 3, 1205)] - P[max(i - 3, 1004)]; tg = H(tg[None])[0]; tg /= np.linalg.norm(tg); n = np.cross(up, tg)
            # line offsets at the camera, corrected to the perpendicular distance: y0 * cos(atan k)
            sep_img = Y0[i] / np.sqrt(1 + KK[i] ** 2) + Y0[jj] / np.sqrt(1 + KK[jj] ** 2)
            sep_tr = abs((P[jj] - P[i]) @ n)
            rat.append((sep_tr, sep_img))
        if rat:
            r = np.array(rat)
            out.append(f"{nm}: n {len(r)}, traj {np.median(r[:,0]):.3f} u, image {np.median(r[:,1]):.3f} h, ratio {np.median(r[:,0]/r[:,1]):.3f} u/h "
                       f"(IQR {np.percentile(r[:,0]/r[:,1],25):.3f}-{np.percentile(r[:,0]/r[:,1],75):.3f})")
        else:
            out.append(f"{nm}: -")
    print(f"{os.path.basename(t)}\n   " + "\n   ".join(out))
