"""Poses at the times of the 10 Hz reference clock instead of the image times.

The images come from a ~27.8 Hz camera (36 ms); the dataset keeps, for every tick of a 10 Hz clock, the nearest image:
image time - tick is a sawtooth within +-18 ms (std 10.4 ms).  If the ground-truth pose of an image is the pose at
the tick (the lidar scan), a trajectory given at the image times is off along the track by speed x sawtooth
(1.7 cm rms at 1.67 m/s).  This script fits the 10 Hz grid to the timestamps and interpolates the trajectory
(cubic spline for the positions, slerp for the rotations) at   t_image - sign * (t_image - t_tick).
usage: python process/time_resample.py in.csv out.csv [--sign 1] [--shift ms]"""
import sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

src, dst = sys.argv[1], sys.argv[2]
sign = float(sys.argv[sys.argv.index("--sign") + 1]) if "--sign" in sys.argv else 1.0
shift = float(sys.argv[sys.argv.index("--shift") + 1]) * 1e-3 if "--shift" in sys.argv else 0.0
lines = open(src).read().strip().split("\n")
stamps = [l.split(",")[0] for l in lines[1:]]
T = np.array([[float(x) for x in l.split(",")[1:]] for l in lines[1:]]).reshape(-1, 4, 4)
t = (np.array([int(s) for s in stamps]) - int(stamps[0])) * 1e-9
i = np.arange(len(t))
per, t0 = np.linalg.lstsq(np.c_[i, np.ones(len(t))], t, rcond=None)[0]
saw = t - (per * i + t0)                                  # image time - tick
tq_raw = t - sign * saw + shift
tq = np.clip(tq_raw, t[0], t[-1])
P = CubicSpline(t, T[:, :3, 3], extrapolate=True)(tq_raw)
Rq = Slerp(t, R.from_matrix(T[:, :3, :3]))(tq).as_matrix()
out = T.copy(); out[:, :3, 3] = P; out[:, :3, :3] = Rq
moved = np.linalg.norm(P - T[:, :3, 3], axis=1)
print(f"grid period {per*1e3:.4f} ms, sawtooth std {saw.std()*1e3:.2f} ms (min {saw.min()*1e3:.1f}, max {saw.max()*1e3:.1f}); "
      f"positions moved rms {np.sqrt((moved**2).mean())*100:.2f} cm, max {moved.max()*100:.2f} cm")
with open(dst, "w") as f:
    f.write(lines[0] + "\n")
    for s, M in zip(stamps, out):
        f.write(s + "," + ",".join(repr(float(x)) for x in M.reshape(-1)) + "\n")
