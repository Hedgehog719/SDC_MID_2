"""Move the reported point of every pose by a constant offset in the car frame (forward, left, up) [m].
Purpose: test whether the ground truth refers to the camera centre or to another point of the car
(a constant offset is not absorbed by the Sim(3) alignment when the car turns).  Rotations are unchanged.
usage: python process/ref_point.py in.csv out.csv fwd left up"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from utils import nhc

src, dst, fwd, left, up = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:6])
f, l, u = nhc.axes(np.radians(-0.02), np.radians(-1.79))
df = pd.read_csv(src)
T = df.iloc[:, 1:].to_numpy(float).reshape(-1, 4, 4)
T[:, :3, 3] += np.einsum("nij,j->ni", T[:, :3, :3], fwd * f + left * l + up * u)
df.iloc[:, 1:] = T.reshape(len(T), -1)
df.to_csv(dst, index=False)
print(f"{dst}: point moved by ({fwd:+.2f}, {left:+.2f}, {up:+.2f}) m (forward, left, up)")
