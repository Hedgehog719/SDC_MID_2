"""Parked vehicles / bicycles as landmarks between the two passes of a road (feasibility test).

Every frame of a pass contributes the NEAR points (depth 1..zmax of the given cloud) that the segmentation calls a road
user (vehicles, bicycles, people), scaled to trajectory units by the camera height (h_units / cloud ground height of the
frame, median over +-5 frames) and placed in the world with the trajectory.  Each pass gives a top-view occupancy map in
the road frame of a leg (along x lateral, 0.1 units); the two passes' maps are cross-correlated: the peak is where pass
E's parked cars sit relative to pass W's -> an along-road / lateral offset between the passes that uses no far depth,
no road markings and no appearance matching.  Checks: the two halves of one pass (offset must be 0), and a second
trajectory (the offset must change by the known difference).
usage: python process/car_occupancy.py TRAJ.csv [TRAJ2.csv ...] [--zmax 10]"""
import argparse, os, sys
import numpy as np
import cv2
sys.path.insert(0, ".")
from utils.io import load_ply, load_camera_info

ap = argparse.ArgumentParser()
ap.add_argument("trajs", nargs="+"); ap.add_argument("--zmax", type=float, default=10.0); ap.add_argument("--h", type=float, default=1.179)
ap.add_argument("--cell", type=float, default=0.1)
ap.add_argument("--smax", type=float, default=2.0, help="along-road search [units]; perpendicular parking repeats every stall (~1.6 units), so keep it below half a stall")
ap.add_argument("--lmax", type=float, default=1.0)
ap.add_argument("--edges", action="store_true", help="correlate the along-road gradient of the occupancy (car ends) instead of the occupancy")
a = ap.parse_args()
T2 = "Track2"
files = sorted(os.listdir(f"{T2}/data/raw_ply")); K, _ = load_camera_info(f"{T2}/data/camera_info.json")
SEG = np.load(f"{T2}/seg.npz"); FEAT = [d for d in os.listdir(f"{T2}/result") if d.startswith("feat_")][0]
W_FR, E_FR = list(range(1004, 1206)), list(range(2881, 3100))
LEGS = {"NS": ((1004, 1076), (3025, 3100)), "EW": ((1136, 1206), (2881, 2941))}
Hc = np.array([float(np.load(f"{T2}/result/{FEAT}/{f[:-4]}.npz")["h"]) for f in files])
Hs = np.array([np.nanmedian(Hc[max(0, i - 5):i + 6]) for i in range(len(Hc))])
CACHE = {}


def road_user_points(i):
    if i in CACHE:
        return CACHE[i]
    p = load_ply(f"{T2}/data/raw_ply/{files[i]}")
    p = p[np.isfinite(p).all(1) & (p[:, 2] > 1.0) & (p[:, 2] < a.zmax)]
    uv = p @ K.T; uv = uv[:, :2] / uv[:, 2:3]
    r = np.clip((uv[:, 1] / 4).astype(int), 0, 231); c = np.clip((uv[:, 0] / 4).astype(int), 0, 359)
    m = (SEG["obj"][i][r, c] > 128) & (p[:, 1] < Hc[i] - 0.15)          # road users, above the road
    CACHE[i] = p[m] * (a.h / Hs[i])
    return CACHE[i]


def leg_axes(P, f0, f1):
    C = P[f0:f1, :3, 3]; up = -P[f0:f1, :3, 1].mean(0); up /= np.linalg.norm(up)
    u = C[-1] - C[0]; u -= (u @ up) * up; u /= np.linalg.norm(u)
    return C.mean(0), u, np.cross(up, u)


def occupancy(P, frames, o, u, l, ext):
    xs, ys = [], []
    for i in frames:
        q = road_user_points(i)
        if len(q) == 0:
            continue
        X = q @ P[i][:3, :3].T + P[i][:3, 3] - o
        xs.append(X @ u); ys.append(X @ l)
    x, y = np.concatenate(xs), np.concatenate(ys)
    nx, ny = int((ext[1] - ext[0]) / a.cell), int((ext[3] - ext[2]) / a.cell)
    g, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    g = cv2.GaussianBlur(np.log1p(g).astype(np.float32), (0, 0), 1.0)
    if a.edges:                                          # along-road gradient: car ends and gaps
        g = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) if False else np.gradient(g, axis=0).astype(np.float32)
    return g


def correlate(A, B, smax=2.0, lmax=1.0):
    """shift (along, lateral) [units] that moves B onto A; peak and margin"""
    sx, sy = int(smax / a.cell), int(lmax / a.cell)
    res = cv2.matchTemplate(np.pad(B, ((sx, sx), (sy, sy))), A, cv2.TM_CCOEFF_NORMED)
    _, mx, _, lc = cv2.minMaxLoc(res)
    r2 = res.copy(); r2[max(0, lc[1] - 3):lc[1] + 4, max(0, lc[0] - 3):lc[0] + 4] = -1
    return -(lc[1] - sx) * a.cell, -(lc[0] - sy) * a.cell, mx, mx - r2.max()


for tf in a.trajs:
    P = np.genfromtxt(tf, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
    print(os.path.basename(tf))
    for leg, ((w0, w1), (e0, e1)) in LEGS.items():
        o, u, l = leg_axes(P, w0, w1)
        ext = (-25.0, 25.0, -8.0, 8.0)
        Wf, Ef = [f for f in W_FR], [f for f in E_FR]
        # only frames whose camera is near this leg (within the map, +-12 units along)
        near = lambda fr: [i for i in fr if abs((P[i][:3, 3] - o) @ u) < 30 and abs((P[i][:3, 3] - o) @ l) < 8]
        Wf, Ef = near(Wf), near(Ef)
        MW, ME = occupancy(P, Wf, o, u, l, ext), occupancy(P, Ef, o, u, l, ext)
        da, dl, pk, mg = correlate(MW, ME, a.smax, a.lmax)
        h1, h2 = Wf[0::2], Wf[1::2]                       # interleaved halves cover the same road
        sa, sl, spk, smg = correlate(occupancy(P, h1, o, u, l, ext), occupancy(P, h2, o, u, l, ext), a.smax, a.lmax)
        print(f"  {leg}: W {len(Wf)} frames, E {len(Ef)} frames; E->W shift along {da:+.2f}, lateral {dl:+.2f} units (peak {pk:.2f}, margin {mg:.2f}) | "
              f"W odd/even {sa:+.2f} / {sl:+.2f} (peak {spk:.2f}, margin {smg:.2f})", flush=True)
