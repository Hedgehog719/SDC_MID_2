"""Road-marking points seen by both passes of a road (opposite directions) as 2-view tracks for process/sfm_ba.py --links.

sfm_ba.py only pairs frames that face the same way (within 30 deg), so nothing ties the two directions of a road
together and BA moves them independently.  Here every frame a with a frame b of the other pass ~ahead*h in front:
  1. coarse: utils/bevloop.register of the two marking rasters (road plane, a's frame)
  2. corners of a's WHITE markings (stop lines, crosswalk bars, arrows, text, stall-line ends: 2D structure, not
     periodic at the search range).  The yellow centre dashes are left out (periodic; they stay a lateral constraint
     elsewhere).
  3. every corner refined locally (patch NCC in the road plane, +-0.5 h): the identity of the painted feature is right
     even where the flat-road / pitch model of the BEV is off, because only the image positions are kept
  4. the image pixel of the feature in a and in b (through the same road-plane grids) -> one 2-view track
BA then places these points with the real lens / pose geometry.
usage: python process/road_links.py Track2 traj.csv --out Track2/result/links_<name>.npz"""
import argparse
import os
import sys
from multiprocessing import get_context

import cv2
import numpy as np

sys.path.insert(0, ".")
from utils import bevloop

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("traj")
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--ahead", default="10,14")
ap.add_argument("--cam", default="1035.5,1033.88,722.17,456.84,-0.05833,0.088413,0,0")
ap.add_argument("--mount", default="-0.22,-1.48")
ap.add_argument("--roll", default=None)
ap.add_argument("--h", type=float, default=1.156)
ap.add_argument("--per-pair", type=int, default=40)
ap.add_argument("--ncc", type=float, default=0.75)
ap.add_argument("--procs", type=int, default=24)
ap.add_argument("--out", required=True)
a = ap.parse_args()

T = np.genfromtxt(a.traj, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
c_ = [float(x) for x in a.cam.split(",")]
K = np.array([[c_[0], 0, c_[2]], [0, c_[1], c_[3]], [0, 0, 1.0]]); D = np.array(c_[4:8] + [0.0])
MOUNT = tuple(float(x) for x in a.mount.split(","))
ROLL = np.load(a.roll) if a.roll else None
G = bevloop.G; E0, E2 = bevloop.EXT[0], bevloop.EXT[2]
MB = None


def init():
    global MB
    cv2.setNumThreads(1)
    MB = bevloop.MarkingBEV(f"{a.target}/data/images", seg=f"{a.target}/seg.npz", mount=MOUNT, roll=ROLL, K=K, D=D)


def pix(i, x, y):
    """image pixel of road-plane point (forward x, left y) [camera heights] of frame i (same model as the BEV grid)"""
    phi = 0.0 if ROLL is None else round(float(ROLL[i]) * 4) / 4.0
    P = x[:, None] * MB.F + y[:, None] * MB.L + (y[:, None] * np.tan(np.radians(phi)) - 1.0) * MB.U
    return cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


def paint(g, uv, r=4, R=24, thr=30.0):
    """a bright painted pixel within r px of uv, against the asphalt around it"""
    x, y = int(round(uv[0])), int(round(uv[1]))
    near = g[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    ring = g[max(0, y - R):y + R + 1, max(0, x - R):x + R + 1]
    return near.size > 0 and near.max() - np.median(ring) > thr


def work(job):
    i, ah = job
    h = a.h
    C, Z = T[:, :3, 3], T[:, :3, 2]
    opp = np.where((Z @ Z[i] < -0.95) & (np.abs(np.arange(len(C)) - i) > 150))[0]
    if len(opp) == 0:
        return []
    loc = np.einsum("ij,nj->ni", T[i, :3, :3].T, C[opp] - C[i])
    cost = np.abs(loc[:, 2] - ah * h) + np.abs(loc[:, 0])
    if cost.min() > 4 * h:
        return []
    b = int(opp[np.argmin(cost)])
    Tab = np.linalg.inv(T[i]) @ T[b]
    r = bevloop.register(MB, i, b, Tab, h)
    if r is None or r[3] < 0.35 or r[4] < 0.04 or abs(np.degrees(r[2])) > 2.5:
        return []
    xb, yb, hb = bevloop.road_pose(Tab, MB.F, MB.L)
    ox, oy, th = xb / h + r[0] / h, yb / h + r[1] / h, hb + r[2]
    pa, wa = MB.points(i); pb, wb = MB.points(b)
    c, s = np.cos(th), np.sin(th)
    qa = np.c_[ox + c * pb[:, 0] - s * pb[:, 1], oy + s * pb[:, 0] + c * pb[:, 1]]
    RA, RB = bevloop._raster(pa, wa), bevloop._raster(qa, wb)
    # white only: drop cells whose image colour is yellow
    im = cv2.imread(os.path.join(MB.dir, MB.files[i]))
    n8 = (RA / (RA.max() + 1e-9) * 255).astype(np.uint8)
    cor = cv2.goodFeaturesToTrack(n8, maxCorners=4 * a.per_pair, qualityLevel=0.15, minDistance=12, blockSize=7)
    if cor is None:
        return []
    out = []
    ga, imb = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float), None
    P_, S_ = 15, 12
    for (cx, cy) in cor[:, 0]:
        ix, iy = int(round(cy)), int(round(cx))                    # raster row = forward, column = lateral
        if not (P_ + S_ <= ix < RA.shape[0] - P_ - S_ and P_ + S_ <= iy < RA.shape[1] - P_ - S_):
            continue
        x, y = E0 + (ix + 0.5) * G, E2 + (iy + 0.5) * G
        if not (2.5 < x < 11.0 and abs(y) < 4.0):
            continue
        u = pix(i, np.array([x]), np.array([y]))[0]
        if not (0 <= u[0] < im.shape[1] - 1 and 0 <= u[1] < im.shape[0] - 1):
            continue
        if RA[ix - 3:ix + 4, iy - 3:iy + 4].max() < 0.35 * RA.max():   # on / at the edge of a strong marking
            continue
        pt = im[max(0, int(u[1]) - 6):int(u[1]) + 7, max(0, int(u[0]) - 6):int(u[0]) + 7].astype(float)
        if (np.minimum(pt[..., 1], pt[..., 2]) - pt[..., 0]).max() > 12:   # yellow centre line nearby
            continue
        tpl = RA[ix - P_:ix + P_ + 1, iy - P_:iy + P_ + 1]
        win = RB[ix - P_ - S_:ix + P_ + S_ + 1, iy - P_ - S_:iy + P_ + S_ + 1]
        if tpl.std() < 1e-6 or win.std() < 1e-6:
            continue
        # 2D structure inside the patch (not a straight line: aperture)
        gx, gy = cv2.Sobel(tpl, cv2.CV_32F, 1, 0), cv2.Sobel(tpl, cv2.CV_32F, 0, 1)
        ev = np.linalg.eigvalsh(np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]]))
        if ev[0] < 0.3 * ev[1]:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, lc = cv2.minMaxLoc(res)
        if mx < a.ncc or not (0 < lc[0] < res.shape[1] - 1 and 0 < lc[1] < res.shape[0] - 1):
            continue
        r2 = res.copy(); r2[max(0, lc[1] - 4):lc[1] + 5, max(0, lc[0] - 4):lc[0] + 5] = -1
        if r2.max() > mx - 0.1:
            continue

        def sub(v0, v1, v2):
            d_ = v0 - 2 * v1 + v2
            return 0.5 * (v0 - v2) / d_ if d_ < 0 else 0.0
        dr = lc[1] - S_ + sub(res[lc[1] - 1, lc[0]], mx, res[lc[1] + 1, lc[0]])
        dc = lc[0] - S_ + sub(res[lc[1], lc[0] - 1], mx, res[lc[1], lc[0] + 1])
        xq, yq = x + dr * G, y + dc * G                                 # the feature in b's content, a's frame
        dx, dy = xq - ox, yq - oy                                       # -> b's road frame
        xbq, ybq = c * dx + s * dy, -s * dx + c * dy
        if not (2.5 < xbq < 11.0 and abs(ybq) < 4.0):
            continue
        v = pix(b, np.array([xbq]), np.array([ybq]))[0]
        if not (0 <= v[0] < im.shape[1] and 0 <= v[1] < im.shape[0]):
            continue
        if imb is None:
            imb = cv2.imread(os.path.join(MB.dir, MB.files[b]), cv2.IMREAD_GRAYSCALE).astype(float)
        if not (paint(ga, u) and paint(imb, v)):                        # real paint at the pixel in both images
            continue
        out.append((i, b, u[0], u[1], v[0], v[1], mx, ah, dr * G * h, dc * G * h))
        if len(out) >= a.per_pair:
            break
    return out


if __name__ == "__main__":
    jobs = [(i, float(ah)) for ah in a.ahead.split(",") for i in range(0, len(T), a.every)]
    with get_context("spawn").Pool(a.procs, initializer=init) as pool:
        res = pool.map(work, jobs, chunksize=4)
    L = np.array([r for rr in res for r in rr])
    np.savez(a.out, fa=L[:, 0].astype(int), fb=L[:, 1].astype(int), uva=L[:, 2:4], uvb=L[:, 4:6], ncc=L[:, 6], ahead=L[:, 7],
             refine=L[:, 8:10])
    pairs = {(int(p), int(q)) for p, q in L[:, :2]}
    print(f"{len(L)} road-marking links in {len(pairs)} opposite-pass pairs; local refinement (traj units) forward median "
          f"{np.median(L[:,8]):+.3f} robust sd {1.4826*np.median(np.abs(L[:,8]-np.median(L[:,8]))):.3f}, lateral {np.median(L[:,9]):+.3f}")
    for ah in sorted(set(L[:, 7])):
        m = L[:, 7] == ah
        print(f"   ahead {ah:g}: {m.sum()} links, refinement forward {np.median(L[m,8]):+.3f}")
    fr = L[:, 0]
    for nm, (p, q) in dict(topW=(1141, 1345), topE=(2731, 2948)).items():
        m = (fr >= p) & (fr < q)
        print(f"   {nm}: {m.sum()} links")
