"""Single-pass local scale from road markings: for frame pairs (i, i + step) of ONE pass, the BEV registration of the
markings gives the forward motion in camera heights; the trajectory gives it in units.  ratio = traj / (marking * h)
- 1 is the local along-road scale error of the trajectory (+ = trajectory too long), up to the BEV unit h (a constant)
and the frame's pitch (nodding).  No pairing between passes.  Printed per block of frames.
usage: python process/step_ruler.py TRAJ.csv f0:f1 [--step 4] [--block 20] [--cam ...]"""
import argparse, sys
import numpy as np
sys.path.insert(0, ".")
from utils import bevloop

ap = argparse.ArgumentParser()
ap.add_argument("traj"); ap.add_argument("frames"); ap.add_argument("--step", type=int, default=4); ap.add_argument("--block", type=int, default=20)
ap.add_argument("--cam", default="1037.0,1035.38,723.93,463.29,-0.055911,0.085266,0,0")
ap.add_argument("--h", type=float, default=0.0, help="camera height in units (0: median over all pairs of the frames given)")
a = ap.parse_args()
c_ = [float(x) for x in a.cam.split(",")]
K = np.array([[c_[0], 0, c_[2]], [0, c_[1], c_[3]], [0, 0, 1.0]]); D = np.array(c_[4:8] + [0.0])
mb = bevloop.MarkingBEV("Track2/data/images", seg="Track2/seg.npz", K=K, D=D)
T = np.genfromtxt(a.traj, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
rows = []
for rng in a.frames.split(","):
    f0, f1 = (int(x) for x in rng.split(":"))
    for i in range(f0, f1 - a.step):
        Tij = np.linalg.inv(T[i]) @ T[i + a.step]
        xb, _, _ = bevloop.road_pose(Tij, mb.F, mb.L)
        if xb < 0.3:
            continue
        r = bevloop.register(mb, i, i + a.step, Tij, 1.0, yaw_range=0.5, shift=1.5)
        if r is None or r[3] < 0.5 or r[4] < 0.05 or xb + r[0] <= 0.05:
            continue
        rows.append((i, xb, xb + r[0], r[3]))           # traj units, marking camera heights, peak
rows = np.array(rows)
h = a.h or float(np.median(rows[:, 1] / rows[:, 2]))
print(f"{a.traj.split('/')[-1]}: {len(rows)} pairs, h = {h:.4f} units{' (median over these frames)' if not a.h else ''}")
for rng in a.frames.split(","):
    f0, f1 = (int(x) for x in rng.split(":"))
    for b in range(f0, f1, a.block):
        m = (rows[:, 0] >= b) & (rows[:, 0] < b + a.block)
        if m.sum() < 3:
            print(f"   {b}-{b + a.block - 1}: {m.sum()} pairs"); continue
        e = rows[m, 1] / (rows[m, 2] * h) - 1
        print(f"   {b}-{b + a.block - 1}: {m.sum():3d} pairs, traj step {np.median(rows[m, 1]):.3f}, scale error median {np.median(e) * 100:+5.1f} % "
              f"(IQR {np.percentile(e, 25) * 100:+.1f} .. {np.percentile(e, 75) * 100:+.1f})")
np.save(f"Track2/result/stepruler_{a.traj.split('/')[-1][:-4]}.npy", np.c_[rows, np.full(len(rows), h)])
