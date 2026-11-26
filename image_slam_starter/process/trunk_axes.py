"""Tree trunks as view-independent landmarks between the two passes of a road.

A trunk is close to a cylinder: from any direction the middle of its silhouette points at its axis, so bearings to the
silhouette centre, triangulated by ONE pass, give the axis without the visible-side bias of surface points.
  1. trunk candidates: vertical columns of the multi-view triangulated points of each pass (sfm_ba --dump-tracks), kept
     where both passes have a column within 1 unit (rough position = their mean)
  2. per frame of a pass: the candidate is projected (lens model) at heights hlo..hhi above the road (above parked cars,
     below the canopy); in that image strip the column profile's strongest pair of opposite edges within the expected
     trunk width is the silhouette; its middle -> bearing in the road plane
  3. per pass and trunk: robust least-squares intersection of the bearings (>= min_obs, spread >= min_spread deg)
  4. E - W per trunk, by leg and by side of the road (an unbiased landmark gives the same offset on both sides)
usage: python process/trunk_axes.py PTS.npz TRAJ.csv"""
import argparse, os
import cv2
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("pts"); ap.add_argument("traj")
ap.add_argument("--hlo", type=float, default=1.1); ap.add_argument("--hhi", type=float, default=1.9)
ap.add_argument("--radius", type=float, default=0.12, help="expected trunk radius [units]")
ap.add_argument("--min-obs", type=int, default=6); ap.add_argument("--min-spread", type=float, default=8.0)
ap.add_argument("--seg", default="edges", choices=["edges", "sam2"],
                help="silhouette from the strongest edge pair (default) or from a SAM 2.1 mask prompted at the projected candidate")
ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
ap.add_argument("--dump", default="", help="save per-observation silhouettes (npz) for inspection")
ap.add_argument("--load", default="", help="reuse silhouettes from a --dump file instead of running SAM")
ap.add_argument("--max-wvar", type=float, default=2.0, help="sam2: max (ptp of row widths) / median width")
ap.add_argument("--min-iou", type=float, default=0.0, help="sam2: min SAM iou score")
ap.add_argument("--per-trunk", action="store_true")
a = ap.parse_args()
T2 = "Track2"
CAM = (1035.5, 1033.88, 722.17, 456.84, -0.05833, 0.088413, 0, 0)
Kc = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]]); DIST = np.array(CAM[4:8] + (0.0,))
names = sorted(os.listdir(f"{T2}/data/images"))
z = np.load(a.pts); X = z["xyz"].astype(float); fmin, fmax, obj = z["fmin"], z["fmax"], z["obj"]
P = np.genfromtxt(a.traj, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
PASS = {"W": (990, 1220), "E": (2870, 3110)}
LEGS = {"NS": (1004, 1076), "corner": (1076, 1136), "EW": (1136, 1206)}
up = -P[1004:1206, :3, 1].mean(0); up /= np.linalg.norm(up)
road = (P[np.r_[1004:1206, 2881:3100], :3, 3] @ up).mean() - 1.18


def columns(m):
    Q = X[m]; hgt = Q @ up - road; sel = (hgt > 0.3) & (hgt < 2.5) & (obj[m] < 0.3); Q = Q[sel]; h2 = hgt[sel]
    Qh = Q - np.outer(Q @ up, up); t = cKDTree(Qh); c = []; used = np.zeros(len(Q), bool)
    for i in np.argsort(-h2):
        if used[i]:
            continue
        nb = np.array(t.query_ball_point(Qh[i], 0.25)); nb = nb[~used[nb]]
        if len(nb) >= 8 and np.ptp(h2[nb]) > 1.0:
            c.append(Qh[nb].mean(0)); used[nb] = True
    return np.array(c)


CW = columns((fmin >= PASS["W"][0]) & (fmax < PASS["W"][1])); CE = columns((fmin >= PASS["E"][0]) & (fmax < PASS["E"][1]))
d, j = cKDTree(CE).query(CW)
TR = [(CW[i] + CE[j[i]]) / 2 for i in range(len(CW)) if d[i] < 1.0]
print(f"trunk candidates seen by both passes: {len(TR)}")
IMG = {}


def gray(f):
    if f not in IMG:
        IMG[f] = cv2.imread(f"{T2}/data/images/{names[f]}", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return IMG[f]


def project(f, Th):
    """projected strip of candidate Th in frame f: (x0, y0, y1, half [px]) or None"""
    R, C = P[f][:3, :3], P[f][:3, 3]
    hs = np.linspace(a.hlo, a.hhi, 5)
    Xw = np.array([Th + (road + h) * up for h in hs])
    Xc = (Xw - C) @ R
    if (Xc[:, 2] < 2.0).any():
        return None
    rng = np.linalg.norm((Th - C) - ((Th - C) @ up) * up)
    if rng > 25:
        return None
    uv = cv2.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc, DIST)[0].reshape(-1, 2)
    if (uv[:, 0] < 30).any() or (uv[:, 0] > 1410).any() or (uv[:, 1] < 0).any() or (uv[:, 1] > 900).any():
        return None
    half = max(3.0, a.radius / rng * CAM[0])
    win = int(3 * half + 25)
    x0 = int(uv[:, 0].mean()); y0, y1 = int(uv[:, 1].min()), int(uv[:, 1].max()) + 1
    if y1 - y0 < 4 or x0 - win < 0 or x0 + win >= 1440:
        return None
    return x0, y0, y1, half, win


def to_bearing(f, uc, vc):
    R, C = P[f][:3, :3], P[f][:3, 3]
    n = cv2.undistortPoints(np.array([[[uc, vc]]]), Kc, DIST).reshape(2)
    rw = R @ np.r_[n, 1.0]; rw -= (rw @ up) * up
    return C, rw / np.linalg.norm(rw)


SAM = {}          # (frame, trunk index) -> (uc, vc, width px, iou score)


def sam_silhouettes(jobs):
    """jobs: frame -> [(trunk index, x0, y0, y1, half, win)]; one SAM 2.1 forward per frame, one object per candidate"""
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor
    proc = Sam2Processor.from_pretrained(a.sam_model); model = Sam2Model.from_pretrained(a.sam_model).to("cuda").eval()
    for n, (f, js) in enumerate(sorted(jobs.items())):
        img = Image.open(f"{T2}/data/images/{names[f]}").convert("RGB")
        pts = [[[[x0, y0 + 0.25 * (y1 - y0)], [x0, y0 + 0.75 * (y1 - y0)]] for (_, x0, y0, y1, _, _) in js]]
        inp = proc(images=img, input_points=pts, input_labels=[[[1, 1]] * len(js)], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inp, multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0].numpy()     # objects x 3 x H x W
        iou = out.iou_scores[0].float().cpu().numpy()
        for o, (ti, x0, y0, y1, half, win) in enumerate(js):
            best = None
            for k in np.argsort(-iou[o]):                  # best-scoring mask whose strip run looks like a trunk
                m = masks[o, k, y0:y1, :]
                mids, ws = [], []
                for row in m:
                    if not row[x0]:
                        continue
                    l = x0
                    while l > 0 and row[l - 1]:
                        l -= 1
                    r = x0
                    while r < len(row) - 1 and row[r + 1]:
                        r += 1
                    mids.append((l + r) / 2.0); ws.append(r - l + 1)
                if len(mids) < 0.6 * (y1 - y0):
                    continue
                w = np.median(ws)
                if not (0.8 * half <= w <= 5.0 * half) or np.ptp(ws) > 2.0 * w:      # canopy / car merged in, or a sliver
                    continue
                uc = float(np.median(mids))
                if abs(uc - x0) > max(1.5 * half, 6.0):
                    continue
                best = (uc, (y0 + y1) / 2.0, float(w), float(iou[o, k]), float(np.ptp(ws) / w), float(w / half)); break
            if best is not None:
                SAM[(f, ti)] = best
        if n % 50 == 0:
            print(f"  sam2: {n}/{len(jobs)} frames, {len(SAM)} silhouettes", flush=True)


def silhouette_bearing(f, Th, ti=None):
    """bearing (unit horizontal vector in the world) of the trunk's silhouette centre in frame f, or None"""
    pr = project(f, Th)
    if pr is None:
        return None
    if a.seg == "sam2":
        o = SAM.get((f, ti))
        if o is None or o[4] > a.max_wvar or o[3] < a.min_iou:
            return None
        return to_bearing(f, *SAM[(f, ti)][:2])
    x0, y0, y1, half, win = pr
    strip = gray(f)[y0:y1, x0 - win:x0 + win + 1]
    prof = cv2.GaussianBlur(strip.mean(0)[None], (0, 0), 1.5)[0]
    g = np.gradient(prof)
    best, bs = None, 0.0
    lo, hi = max(2, int(1.0 * half)), int(4.0 * half) + 2           # silhouette width range [px]
    for i in range(len(g)):
        for k in range(i + lo, min(len(g), i + hi)):
            if g[i] * g[k] < 0:
                s = abs(g[i]) + abs(g[k]) - 0.3 * abs(abs(g[i]) - abs(g[k]))
                if s > bs:
                    bs, best = s, (i + k) / 2.0
    if best is None or bs < 6.0 or abs(best - win) > max(1.5 * half, 6.0):     # must stay near the rough position
        return None
    return to_bearing(f, x0 - win + best, (y0 + y1) / 2.0)


def axis(frames, Th, ti=None):
    obs = [o for o in (silhouette_bearing(f, Th, ti) for f in frames) if o is not None]
    if len(obs) < a.min_obs:
        return None
    C = np.array([o[0] for o in obs]); D = np.array([o[1] for o in obs])
    e1 = D[0]; e2 = np.cross(up, e1); ang = np.degrees(np.arctan2(D @ e2, D @ e1))
    if np.ptp(ang) < a.min_spread:
        return None
    w = np.ones(len(C))
    for _ in range(5):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for c_, d_, w_ in zip(C, D, w):
            M = np.eye(3) - np.outer(d_, d_) - np.outer(up, up)
            A += w_ * M; b += w_ * M @ c_
        Xa = np.linalg.solve(A + np.outer(up, up), b + up * (up @ Th))
        r = np.array([np.linalg.norm((np.eye(3) - np.outer(d_, d_) - np.outer(up, up)) @ (Xa - c_)) for c_, d_ in zip(C, D)])
        s = max(1.4826 * np.median(r), 0.01); w = 1.0 / np.maximum(1.0, r / (2 * s))
    return Xa - (Xa @ up) * up, len(obs), float(np.median(r))


if a.seg == "sam2":
    jobs = {}
    for ti, Th in enumerate(TR):
        for f in [*range(*PASS["W"]), *range(*PASS["E"])]:
            pr = project(f, Th)
            if pr is not None:
                jobs.setdefault(f, []).append((ti,) + pr)
    print(f"sam2 prompts: {sum(len(v) for v in jobs.values())} in {len(jobs)} frames")
    if a.load:
        zl = np.load(a.load)
        SAM.update({(int(f), int(t)): tuple(v) for f, t, v in zip(zl["frame"], zl["trunk"], zl["sil"])})
    else:
        sam_silhouettes(jobs)
    if a.dump and not a.load:
        k = sorted(SAM); np.savez(a.dump, frame=[x[0] for x in k], trunk=[x[1] for x in k], sil=np.array([SAM[x] for x in k]).reshape(-1, 6),
                                  cand=np.array(TR))
rows = []
for ti, Th in enumerate(TR):
    aw = axis(range(*PASS["W"]), Th, ti); ae = axis(range(*PASS["E"]), Th, ti)
    if aw is None or ae is None:
        continue
    rows.append((Th, aw[0], ae[0], aw[1], ae[1], aw[2], ae[2]))
print(f"trunks with an axis from both passes: {len(rows)}")
for leg, (f0, f1) in LEGS.items():
    C = P[f0:f1, :3, 3]; o = C.mean(0); u = C[-1] - C[0]; u -= (u @ up) * up; u /= np.linalg.norm(u); l = np.cross(up, u)
    L = np.linalg.norm(C[-1] - C[0]) / 2 + 3
    sel = [r for r in rows if abs((r[0] - o) @ u) < L and abs((r[0] - o) @ l) < 12]
    if not sel:
        print(f"  {leg}: none"); continue
    side = np.array([(r[0] - o) @ l for r in sel]); dd = np.array([((r[2] - r[1]) @ u, (r[2] - r[1]) @ l) for r in sel])
    if a.per_trunk:
        for r_, s_, d_ in zip(sel, side, dd):
            print(f"    {leg} side {s_:+5.1f} at {(r_[0] - o) @ u:+6.1f}: E-W along {d_[0]:+.3f} lat {d_[1]:+.3f}  obs W/E {r_[3]}/{r_[4]}  res {r_[5]:.3f}/{r_[6]:.3f}")
    res = np.array([(r[5], r[6]) for r in sel])
    for nm, m in (("all", np.ones(len(sel), bool)), ("left of W", side > 0), ("right of W", side < 0)):
        if m.sum():
            print(f"  {leg:6s} {nm:11s}: {m.sum():2d} trunks, E - W along median {np.median(dd[m, 0]):+.3f}, lateral {np.median(dd[m, 1]):+.3f} "
                  f"(spread {1.4826 * np.median(np.abs(dd[m, 0] - np.median(dd[m, 0]))):.3f}); ray residual W/E {np.median(res[m, 0]):.3f}/{np.median(res[m, 1]):.3f}")
