"""Along-road camera steps from static features OFF the road (wall, doors, trees, parked cars), by their HORIZONTAL
bearing only.  Question it answers: on the eastbound public road (2849-2856) the road text moves 20-30 % less per frame
in the image than the trajectory steps.  Pitch (a nod) or real scale?

The road-plane projection mixes the two: a 1 deg nod moves a road point at 5 h by ~10 %.  The horizontal bearing of a
point (atan2(lateral, forward) in the road frame) barely depends on pitch (1 deg changes it by < 0.4 %), but it does
depend on where along the road the camera is.  So:
  - KLT tracks (forward-backward checked) of corners outside the road / the objects of seg.npz, in [f0, f1]
  - ray of each observation with the trajectory rotation -> horizontal bearing in a road frame (forward fw, left lt)
  - unknowns: every track's (X, Y) on the road plane, every frame's along-road position a_k and a small yaw correction;
    gauge: a_k = trajectory at f0 and f1 only; camera lateral position = trajectory
  - residual: measured - predicted bearing [px at the focal length], soft-L1
Reported per frame: trajectory step vs wall step.  Walls slower by 20 % in the window -> the trajectory step is wrong
(real scale / along error); walls equal to the trajectory -> the road text slow-down is pitch.
usage: python process/wall_speed.py Track2/result/traj.csv [--frames 2836:2877] [--window 2849:2857] [--set wall|left|all]"""
import argparse
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ap = argparse.ArgumentParser()
ap.add_argument("traj", nargs="+")
ap.add_argument("--frames", default="2836:2877")
ap.add_argument("--window", default="2849:2857")
ap.add_argument("--set", default="wall", help="wall: off road/objects, left half; left: left half incl. objects; all: off road")
ap.add_argument("--cam", default="1035.5,1033.88,722.17,456.84,-0.05833,0.088413,0,0")
ap.add_argument("--free-lateral", action="store_true", help="also solve every frame's lateral position (gauge: both ends)")
ap.add_argument("--yaw-prior", type=float, default=1.0, help="prior sigma of the per-frame yaw correction [deg]")
ap.add_argument("--save", default=None, help="npz prefix for the per-frame profile (one file per trajectory)")
ap.add_argument("--draw", default=None)
a = ap.parse_args()

c_ = [float(x) for x in a.cam.split(",")]
K = np.array([[c_[0], 0, c_[2]], [0, c_[1], c_[3]], [0, 0, 1.0]]); D = np.array(c_[4:8] + [0.0])
f0, f1 = (int(x) for x in a.frames.split(":"))
w0, w1 = (int(x) for x in a.window.split(":"))
FILES = sorted(os.listdir("Track2/data/images"))
SEG = np.load("Track2/seg.npz")


def mask(i, shape):
    road = cv2.resize(SEG["road"][i], (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    obj = cv2.resize(SEG["obj"][i], (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    m = ~road
    if a.set in ("wall", "all"):
        m &= ~cv2.dilate(obj.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if a.set in ("wall", "left"):
        side = np.zeros(shape, bool); side[:, : shape[1] * 9 // 20] = True
        m &= side
    m[: shape[0] // 10] = False                       # sky / top border
    m[shape[0] * 85 // 100:] = False                  # bonnet
    return cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8))


# ---- KLT tracks
lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
tracks, active = [], []
prev = None
for i in range(f0, f1):
    g = cv2.cvtColor(cv2.imread("Track2/data/images/" + FILES[i]), cv2.COLOR_BGR2GRAY)
    m = mask(i, g.shape)
    if prev is not None and active:
        p0 = np.array([tracks[t][-1][1:] for t in active], np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **lk)
        fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
        keep = []
        for t, q, s1, s2, e in zip(active, p1.reshape(-1, 2), st.ravel(), st2.ravel(), fb):
            x, y = int(q[0]), int(q[1])
            if s1 and s2 and e < 0.5 and 0 <= x < g.shape[1] and 0 <= y < g.shape[0] and m[y, x]:
                tracks[t].append((i, float(q[0]), float(q[1]))); keep.append(t)
        active = keep
    occ = m.copy()
    for t in active:
        cv2.circle(occ, (int(tracks[t][-1][1]), int(tracks[t][-1][2])), 12, 0, -1)
    new = cv2.goodFeaturesToTrack(g, maxCorners=600, qualityLevel=0.01, minDistance=12, mask=occ, blockSize=7)
    for q in ([] if new is None else new.reshape(-1, 2)):
        tracks.append([(i, float(q[0]), float(q[1]))]); active.append(len(tracks) - 1)
    prev = g
tracks = [t for t in tracks if len(t) >= 5]
print(f"{len(tracks)} tracks (>= 5 frames, {a.set}), frames {f0}-{f1 - 1}, observations {sum(map(len, tracks))}")

if a.draw:
    im = cv2.imread("Track2/data/images/" + FILES[(w0 + w1) // 2])
    for t in tracks:
        p = np.array([o[1:] for o in t], np.int32)
        cv2.polylines(im, [p], False, (0, 255, 255), 1)
        cv2.circle(im, tuple(p[-1]), 2, (0, 0, 255), -1)
    cv2.imwrite(a.draw, im)

obs = np.array([(k, o[0], o[1], o[2]) for k, t in enumerate(tracks) for o in t])      # track, frame, u, v
und = cv2.undistortPoints(obs[:, 2:4].reshape(-1, 1, 2), K, D).reshape(-1, 2)
FOC = K[0, 0]

for traj in a.traj:
    T = np.genfromtxt(traj, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
    C = T[:, :3, 3]
    fw = C[f1 - 1] - C[f0]; fw /= np.linalg.norm(fw)
    up = -T[f0:f1, :3, 1].mean(0); up -= (up @ fw) * fw; up /= np.linalg.norm(up)
    lt = np.cross(up, fw)
    s, l = C @ fw, C @ lt
    fr = obs[:, 1].astype(int); tk = obs[:, 0].astype(int)
    rays = np.einsum("nij,nj->ni", T[fr, :3, :3], np.c_[und, np.ones(len(und))])
    th = np.arctan2(rays @ lt, rays @ fw)
    # initial (X, Y) per track: linear intersection of the horizontal bearings with the trajectory positions
    nt = len(tracks)
    XY = np.full((nt, 2), np.nan); good = np.zeros(nt, bool)
    for k in range(nt):
        j = tk == k
        par = np.degrees(np.ptp(th[j]))
        if par < 2.0:
            continue
        A = np.c_[np.sin(th[j]), -np.cos(th[j])]
        b = s[fr[j]] * np.sin(th[j]) - l[fr[j]] * np.cos(th[j])
        xy = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.all(xy[0] - s[fr[j]] > 0.5):
            XY[k] = xy; good[k] = True
    sel = good[tk]
    o_tk, o_fr, o_th = tk[sel], fr[sel], th[sel]
    tid = {k: n for n, k in enumerate(np.where(good)[0])}
    o_t = np.array([tid[k] for k in o_tk])
    nT = len(tid)
    frames = np.arange(f0, f1)
    free = frames[1:-1]                                                    # a_k (and b_k) free inside, fixed at the two ends
    fidx = {f: n for n, f in enumerate(free)}
    nF = len(free)
    nL = nF if a.free_lateral else 0
    Y0 = 2 * nT + nF + nL                                                  # first yaw variable

    def unpack(x):
        P = x[:2 * nT].reshape(-1, 2)
        av = s.copy(); av[free] = x[2 * nT:2 * nT + nF]
        bv = l.copy()
        if nL:
            bv[free] = x[2 * nT + nF:Y0]
        yaw = np.zeros(len(C)); yaw[frames] = x[Y0:]
        return P, av, bv, yaw

    def fun(x):
        P, av, bv, yaw = unpack(x)
        pred = np.arctan2(P[o_t, 1] - bv[o_fr], P[o_t, 0] - av[o_fr])
        r = np.angle(np.exp(1j * (o_th + yaw[o_fr] - pred))) * FOC
        return np.r_[r, x[Y0:] / np.radians(a.yaw_prior)]

    nobs = len(o_th)
    J = lil_matrix((nobs + len(frames), Y0 + len(frames)), dtype=int)
    for n in range(nobs):
        J[n, 2 * o_t[n]] = 1; J[n, 2 * o_t[n] + 1] = 1
        if o_fr[n] in fidx:
            J[n, 2 * nT + fidx[o_fr[n]]] = 1
            if nL:
                J[n, 2 * nT + nF + fidx[o_fr[n]]] = 1
        J[n, Y0 + (o_fr[n] - f0)] = 1
    for n in range(len(frames)):
        J[nobs + n, Y0 + n] = 1
    x0 = np.r_[XY[good].ravel(), s[free], l[free] if nL else [], np.zeros(len(frames))]
    res = least_squares(fun, x0, jac_sparsity=J, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=200)
    P, av, bv, yaw = unpack(res.x)
    r = fun(res.x)[:nobs]
    print(f"\n== {os.path.basename(traj)}: {nT} tracks with parallax >= 2 deg, {nobs} bearings; residual median {np.median(np.abs(r)):.2f} px, "
          f"p90 {np.percentile(np.abs(r), 90):.2f} px; yaw corrections sd {np.degrees(np.std(yaw[frames])):.3f} deg"
          f"{' (lateral free)' if nL else ''}, yaw prior {a.yaw_prior} deg")
    if a.save:
        np.savez(f"{a.save}_{os.path.basename(traj)[:-4]}.npz", frames=frames, s=s[frames], l=l[frames], a=av[frames], b=bv[frames],
                 yaw=yaw[frames], nobs=np.bincount(o_fr - f0, minlength=len(frames)))
    if len(frames) > 60:
        # long range: walls minus trajectory, position along / lateral (zero at both ends by the gauge), in bins
        da, db = av[frames] - s[frames], bv[frames] - l[frames]
        x = (s[frames] - s[frames].mean()) / (np.ptp(s[frames]) / 2)
        print("  walls minus trajectory (gauge: both ends):  frame   along   lateral   yaw corr [deg]")
        for q in range(0, len(frames), 10):
            print(f"     {frames[q]:5d}  {da[q]:+.3f}  {db[q]:+.3f}  {np.degrees(yaw[frames[q]]):+.3f}")
        for nm, v in (("along", da), ("lateral", db)):
            c2, c1, c0 = np.polyfit(x, v, 2)
            print(f"  {nm:8s}: RMS {np.sqrt((v ** 2).mean()):.3f}, max |.| {np.abs(v).max():.3f}, bend (mid vs ends) {-c2:+.3f}")
        yd = np.polyfit(s[frames], np.degrees(yaw[frames]) * 1000, 1)[0]
        print(f"  yaw correction trend {yd:+.2f} mdeg per traj unit ({yd / 1.56:+.2f} mdeg/m at 1.56 m/unit)")
        continue
    # how many tracks constrain each step: tracks seen in both k-1 and k with lateral offset > 1 unit
    print(" frame | traj step | wall step | ratio | tracks k-1,k (|Y-l|>1)")
    rows = []
    for k in frames[1:]:
        ts, ws = s[k] - s[k - 1], av[k] - av[k - 1]
        both = np.intersect1d(o_t[o_fr == k - 1], o_t[o_fr == k])
        nb = int(np.sum(np.abs(P[both, 1] - l[k]) > 1.0))
        rows.append((k, ts, ws))
        flag = " <- window" if w0 <= k < w1 else ""
        print(f" {k:5d} | {ts:9.3f} | {ws:9.3f} | {ws / ts:5.2f} | {nb:4d}{flag}")
    rows = np.array(rows)
    inw = (rows[:, 0] >= w0) & (rows[:, 0] < w1)
    for nm, m in (("window", inw), ("before", rows[:, 0] < w0), ("after", rows[:, 0] >= w1)):
        print(f"   {nm:7s} frames {int(rows[m,0].min())}-{int(rows[m,0].max())}: trajectory {rows[m,1].sum():.3f}, walls {rows[m,2].sum():.3f}, "
              f"ratio {rows[m,2].sum() / rows[m,1].sum():.3f}")
