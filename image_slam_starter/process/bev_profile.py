"""Along-road profile of the opposite-pass BEV marking registrations (all raw candidates, before the neighbour filter).

For every frame a of one pass on a road driven both ways, the frame b of the other pass ~ahead*h in front is registered
(bevloop.register, same camera / mount / roll / seg as slam_v2 --bev-loops) and the correction (forward, left, yaw) of
b's pose in a is written against a's arc length.  A smooth non-zero slope of the forward correction along the road is a
shear between the two passes that Sim(3) cannot absorb, and which the slam_v2 neighbour filter (0.4 h) partly removes.
usage: python process/bev_profile.py Track2 traj.csv [--frames 1141:1344] [--every 2] [--out prof.npz]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from utils import bevloop

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("traj")
ap.add_argument("--frames", default="1141:1344")
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--ahead", default="10,14,18")
ap.add_argument("--cam", default="1035.5,1033.88,722.17,456.84,-0.05833,0.088413,0,0")
ap.add_argument("--mount", default="-0.22,-1.48")
ap.add_argument("--roll", default=None, help="road_roll_<key>.npy")
ap.add_argument("--h", type=float, default=1.156, help="camera height [traj units]; 0: calibrate on the profiled frames (unit_height)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

T = np.genfromtxt(a.traj, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
c_ = [float(x) for x in a.cam.split(",")]
K = np.array([[c_[0], 0, c_[2]], [0, c_[1], c_[3]], [0, 0, 1.0]]); D = np.array(c_[4:8] + [0.0])
roll = np.load(a.roll) if a.roll else None
mb = bevloop.MarkingBEV(f"{a.target}/data/images", seg=f"{a.target}/seg.npz", mount=tuple(float(x) for x in a.mount.split(",")),
                        roll=roll, K=K, D=D)
h = a.h
f0, f1 = (int(x) for x in a.frames.split(":"))
if h <= 0:
    r_ = []
    for i in range(f0, f1 - 6, 2):
        Tij = np.linalg.inv(T[i]) @ T[i + 6]
        xb, _, _ = bevloop.road_pose(Tij, mb.F, mb.L)
        r = bevloop.register(mb, i, i + 6, Tij, 1.0, yaw_range=0.5, shift=1.5) if xb > 0.3 else None
        if r is not None and r[3] >= 0.5 and r[4] >= 0.05 and xb + r[0] > 0.05:
            r_.append(xb / (xb + r[0]))
    h = float(np.median(r_)); print(f"camera height {h:.4f} units ({len(r_)} pairs)", flush=True)
C, Z = T[:, :3, 3], T[:, :3, 2]
s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
rows = []
for ah in [float(x) for x in a.ahead.split(",")]:
    for i in range(f0, f1, a.every):
        opp = np.where((Z @ Z[i] < -0.95) & (np.abs(np.arange(len(C)) - i) > 150))[0]
        if len(opp) == 0:
            continue
        loc = np.einsum("ij,nj->ni", T[i, :3, :3].T, C[opp] - C[i])
        cost = np.abs(loc[:, 2] - ah * h) + np.abs(loc[:, 0])
        if cost.min() > 4 * h:
            continue
        b = int(opp[np.argmin(cost)])
        r = bevloop.register(mb, i, b, np.linalg.inv(T[i]) @ T[b], h)
        if r is None:
            continue
        rows.append((i, b, ah, s[i], r[0], r[1], np.degrees(r[2]), r[3], r[4]))
    print(f"ahead {ah}: {sum(1 for r in rows if r[2] == ah)} registrations", flush=True)
rows = np.array(rows)
out = a.out or f"{a.target}/result/bevprof_{os.path.basename(a.traj)[:-4]}_{f0}.npy"
np.save(out, rows)
ok = (rows[:, 7] > 0.35) & (rows[:, 8] > 0.04) & (np.abs(rows[:, 6]) < 2.5)
r = rows[ok]
p = np.polyfit(r[:, 3] - r[:, 3].mean(), r[:, 4], 1)
res = r[:, 4] - np.polyval(p, r[:, 3] - r[:, 3].mean())
print(f"{os.path.basename(a.traj)} frames {f0}-{f1}: {len(rows)} raw, {ok.sum()} pass peak/margin/yaw; forward correction "
      f"median {np.median(r[:,4]):+.3f}, slope {p[0]*100:+.3f} units per 100 units of road (span {np.ptp(r[:,3]):.1f}), "
      f"residual robust sd {1.4826*np.median(np.abs(res-np.median(res))):.3f}; left median {np.median(r[:,5]):+.3f}, yaw {np.median(r[:,6]):+.2f} deg")
for q in np.array_split(np.argsort(r[:, 3]), 6):
    print(f"   s {r[q,3].mean():7.1f}  n {len(q):3d}  forward {np.median(r[q,4]):+.3f}  left {np.median(r[q,5]):+.3f}  yaw {np.median(r[q,6]):+.2f}")
