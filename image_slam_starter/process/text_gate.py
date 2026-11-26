"""Gate: the yellow road text "速限 30" on the Track02 public road, seen by both passes (westbound reads it upright in its
own lane, eastbound sees it upside down across the centre line).  For every trajectory, where does each pass put it?

Per frame: road-plane view (self-calibrated lens, mount of the car constraint), yellow paint, the centre-line dashes
removed (long thin components), the remaining text split along the road into blocks ("速", "限", "30"); landmark = paint
centroid of the "30" block and of the "速限" pair.  The road plane is only used to FIND the centroid; its image pixel
(through the same grid) is what is kept, so a pitch / height error of the plane model only distorts the 1-m glyph, not
where it is.
  (a) triangulation: rays of the centroid pixel from all frames of one pass, least-squares intersection with that
      pass's poses (no flat-road assumption)
  (b) road plane: per frame C + R h (x F + y L - U), median over the frames of each character (NO x^2 fit / x -> 0
      extrapolation is done here)
Reported: eastbound minus westbound landmark, along the road / lateral / vertical [trajectory units].
Known limits (NOTES, 2026-11-19 02:00):
  - CRUISE only rejects speed changes; the eastbound pitch nod at 2849-2856 is at constant speed, so it is NOT removed.
  - H and the lens are fixed (self-calibrated lens, H 1.156): numbers for trajectories built with another lens / unit
    (e.g. t2_dash1r, split lens) are not in the same geometry and should not be compared with the others.
usage: python process/text_gate.py traj1 traj2 ...   (names in Track2/result, without .csv)"""
import os
import sys

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from utils import nhc

K = np.array([[1035.5, 0, 722.17], [0, 1033.88, 456.84], [0, 0, 1.0]]); D = np.array([-0.05833, 0.088413, 0, 0, 0.0])
F, L, U = nhc.axes(*np.radians((-0.22, -1.48)))
TH = 22.0
CRUISE = 0.10
PROPAGATE = 0           # frames with all three characters only (the eastbound car slows down: no fixed step)
H = 1.156                                            # camera height [traj units] (BEV unit of the consistent lens)
RES = 0.02
XS = np.arange(14.0, 2.0, -RES); YS = np.arange(4.0, -4.0, -RES)
gx, gy = np.meshgrid(XS, YS, indexing="ij")
UV = cv2.projectPoints((gx[..., None] * F + gy[..., None] * L - U).reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0]
UV = UV.reshape(gx.shape + (2,)).astype(np.float32)
FILES = sorted(os.listdir("Track2/data/images"))
PASSES = {"W": (range(1170, 1200), (-0.9, 0.05)), "E": (range(2836, 2876), (0.8, 1.65))}   # frames, lateral window of the text


def blocks(i, ywin):
    """[(name, x, y, n)] paint centroids (camera heights, road plane) of the "30" block and the "速限" pair"""
    im = cv2.imread("Track2/data/images/" + FILES[i]).astype(np.float32)
    b = cv2.remap(im, UV[..., 0], UV[..., 1], cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
    valid = b[..., 0] >= 0
    g = cv2.cvtColor(np.clip(b, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.5 / RES) | 1, int(0.5 / RES) | 1)))
    m = (th > TH) & valid
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    m &= ((gy > ywin[0]) & (gy < ywin[1]) & (gx > 2.4))
    lab, n = ndimage.label(m)
    keep = np.zeros_like(m)
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        hx, hy = (sl[0].stop - sl[0].start) * RES, (sl[1].stop - sl[1].start) * RES
        cnt = (lab[sl] == k).sum()
        if cnt < 30 or (hx > 2.0 and hy < 0.25):                   # specks; a lane line reaching into the window
            continue
        keep[sl] |= lab[sl] == k
    if keep.sum() < 300:
        return None
    rows = np.where(keep.any(1))[0]
    # touching the window border (partly outside the view): unusable
    win = (YS > ywin[0]) & (YS < ywin[1])
    if XS[rows.min()] > 13.7 or XS[rows.max()] < 2.6 or not valid[rows.min():rows.max() + 1][:, win].all():
        return None
    gaps = np.where(np.diff(rows) > int(0.3 / RES))[0]
    segs = [s_ for s_ in np.split(rows, gaps + 1) if 1.0 < (s_[-1] - s_[0] + 1) * RES < 1.75]   # complete characters
    out = []
    for s_ in segs:
        if XS[s_[0]] > 13.7 or XS[s_[-1]] < 2.6:
            continue
        mm = np.zeros_like(keep); mm[s_] = keep[s_]
        w = mm.astype(float)
        out.append(((gx * w).sum() / w.sum(), (gy * w).sum() / w.sum(), int(w.sum())))
    return sorted(out) or None


def identify(dets, far_is_30):
    """character identity by continuity: frames with all three characters fix the order; the others are matched to the
    previous frame's characters shifted by the median step along x"""
    full = {i: d for i, d in dets.items() if len(d) == 3}
    if not full:
        return {}
    fr = sorted(full)
    xs = np.array([[c[0] for c in full[i]] for i in fr])
    step = np.median(np.diff(xs, axis=0) / np.diff(fr)[:, None]) if len(fr) > 1 else -0.4
    names = ["SU", "XIAN", "30"] if far_is_30 else ["30", "XIAN", "SU"]       # near -> far
    lab = {i: dict(zip(names, full[i])) for i in fr
           if 1.5 < full[i][1][0] - full[i][0][0] < 2.4 and 1.5 < full[i][2][0] - full[i][1][0] < 2.4}   # character spacing
    if PROPAGATE == 0:
        return lab
    for direction in (1, -1):
        prev = fr[0] if direction == 1 else fr[-1]
        i = prev + direction
        while i in dets or (i - prev) * direction <= 3:
            if i in dets and i not in lab:
                k_ = (i - prev)
                pred = {n: c[0] + step * k_ for n, c in lab[prev].items()}
                got = {}
                for c in dets[i]:
                    n = min(pred, key=lambda n: abs(pred[n] - c[0]))
                    if abs(pred[n] - c[0]) < 0.25:
                        got[n] = c
                if got:
                    # names not seen: carry their prediction for the next match
                    lab[i] = {**{n: (pred[n],) for n in pred if n not in got}, **got}
                    prev = i
            elif i in lab:
                prev = i
            i += direction
            if i < min(dets) - 1 or i > max(dets) + 1:
                break
    return {i: {n: c for n, c in d.items() if len(c) == 3} for i, d in lab.items()}


def pix(x, y):
    """image pixel of road-plane point (x, y) via the grid (bilinear in the grid index)"""
    r = (XS[0] - x) / RES; c = (YS[0] - y) / RES
    return np.array([cv2.getRectSubPix(UV[..., k], (1, 1), (float(c), float(r)))[0, 0] for k in (0, 1)])


def triangulate(Ts, uvs):
    und = cv2.undistortPoints(np.array(uvs, np.float64).reshape(-1, 1, 2), K, D).reshape(-1, 2)
    A = np.zeros((3, 3)); bb = np.zeros(3)
    for T, u in zip(Ts, und):
        d = T[:3, :3] @ np.r_[u, 1.0]; d /= np.linalg.norm(d)
        Pm = np.eye(3) - np.outer(d, d)
        A += Pm; bb += Pm @ T[:3, 3]
    X = np.linalg.solve(A, bb)
    ang = max(np.degrees(np.arccos(np.clip(
        (Ts[0][:3, :3] @ np.r_[und[0], 1]) @ (Ts[-1][:3, :3] @ np.r_[und[-1], 1]) /
        np.linalg.norm(np.r_[und[0], 1]) / np.linalg.norm(np.r_[und[-1], 1]), -1, 1))), 0)
    res = [np.linalg.norm((np.eye(3) - np.outer(d := T[:3, :3] @ np.r_[u, 1] / np.linalg.norm(np.r_[u, 1]), d)) @ (X - T[:3, 3]))
           for T, u in zip(Ts, und)]
    return X, ang, float(np.median(res))


if __name__ == "__main__":
    det, lab3 = {}, {}
    for p, (fr, ywin) in PASSES.items():
        raw = {i: b for i in fr if (b := blocks(i, ywin)) is not None}
        lab = identify(raw, far_is_30=(p == "E"))
        lab3[p] = lab
        det[p] = {}
        for i, d in lab.items():
            e = {}
            if "30" in d:
                e["30"] = d["30"]
            if "SU" in d and "XIAN" in d:
                e["SX"] = tuple((np.array(d["SU"]) * d["SU"][2] + np.array(d["XIAN"]) * d["XIAN"][2]) / (d["SU"][2] + d["XIAN"][2]))
            if e:
                det[p][i] = e
        for lm in ("30", "SX"):
            f_ = sorted(i for i in det[p] if lm in det[p][i])
            print(f"{p} {lm}: {len(f_)} frames ({f_[0] if f_ else '-'}..{f_[-1] if f_ else '-'}), x " + " ".join(f"{det[p][i][lm][0]:.1f}" for i in f_))
    print("road plane, cruising frames only (speed change < CRUISE over +-3 frames: braking nods the car and moves the")
    print("flat-road projection); per character: eastbound minus westbound [traj units], along = westbound direction")
    print(f"{'trajectory':18s} | along  SU     XIAN   30    mean  | lateral mean | W/E frames | triangulated 30 / SX along (low parallax)")
    for t in sys.argv[1:]:
        T = np.genfromtxt(f"Track2/result/{t}.csv", delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
        C = T[:, :3, 3]
        fw = C[1199] - C[1176]; fw /= np.linalg.norm(fw)
        up = -T[1176:1200, :3, 1].mean(0); up -= (up @ fw) * fw; up /= np.linalg.norm(up)
        lt = np.cross(up, fw)
        sp = np.r_[np.linalg.norm(np.diff(C, axis=0), axis=1), 0]
        cruise = lambda i: abs(sp[i + 3] / max(sp[i - 3], 1e-9) - 1) < CRUISE
        pos = {}
        nfr = {}
        for p in ("W", "E"):
            fr = [i for i in sorted(lab3[p]) if cruise(i)]
            nfr[p] = len(fr)
            for n in ("SU", "XIAN", "30"):
                Pw = np.array([T[i, :3, 3] + T[i, :3, :3] @ (H * (lab3[p][i][n][0] * F + lab3[p][i][n][1] * L - U)) for i in fr])
                pos[p, n] = np.array([np.median(Pw @ fw), np.median(Pw @ lt)])
        al = [pos["E", n][0] - pos["W", n][0] for n in ("SU", "XIAN", "30")]
        la = [pos["E", n][1] - pos["W", n][1] for n in ("SU", "XIAN", "30")]
        tri = []
        for lm in ("30", "SX"):
            Xs = {}
            for p in ("W", "E"):
                fr = sorted(i for i in det[p] if lm in det[p][i])
                Xs[p] = triangulate([T[i] for i in fr], [pix(*det[p][i][lm][:2]) for i in fr])[0]
            tri.append((Xs["E"] - Xs["W"]) @ fw)
        print(f"{t:18s} | {al[0]:+.3f} {al[1]:+.3f} {al[2]:+.3f} {np.mean(al):+.3f} | {np.mean(la):+.3f}       | {nfr['W']}/{nfr['E']}      | {tri[0]:+.2f} / {tri[1]:+.2f}")
