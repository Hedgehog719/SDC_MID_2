"""Fuse trajectories from independent pipelines: every input is Sim(3)-aligned to the first one, positions are averaged
with weights, rotations averaged on SO(3) (chordal mean).  Two estimates whose errors come from different sources
(cloud scale + camera-height correction vs multi-view bundle adjustment) average to a smaller error.
usage: python process/fuse.py out.csv a.csv b.csv [--w 0.5,0.5]"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, ".")
from process.compare import umeyama

args = sys.argv[1:]
w = None
if "--w" in args:
    i = args.index("--w"); w = np.array([float(x) for x in args[i + 1].split(",")]); args = args[:i] + args[i + 2:]
out, ins = args[0], args[1:]
lines = [open(f).read().strip().split("\n") for f in ins]
T = [np.array([[float(x) for x in l.split(",")[1:]] for l in L[1:]]).reshape(-1, 4, 4) for L in lines]
w = np.ones(len(T)) / len(T) if w is None else w / w.sum()
ref = T[0]
P = np.zeros((len(ref), 3)); M = np.zeros((len(ref), 3, 3))
for Ti, wi in zip(T, w):
    s, Rm, t = umeyama(Ti[:, :3, 3], ref[:, :3, 3])
    P += wi * (s * Ti[:, :3, 3] @ Rm.T + t)
    M += wi * np.einsum("ij,njk->nik", Rm, Ti[:, :3, :3])
U, _, Vt = np.linalg.svd(M)
D = np.ones((len(ref), 3)); D[:, 2] = np.sign(np.linalg.det(U @ Vt))
Rf = np.einsum("nij,nj,njk->nik", U, D, Vt)
F = np.tile(np.eye(4), (len(ref), 1, 1)); F[:, :3, :3] = Rf; F[:, :3, 3] = P
with open(out, "w") as f:
    f.write(lines[0][0] + "\n")
    for l, M_ in zip(lines[0][1:], F):
        f.write(l.split(",")[0] + "," + ",".join(repr(float(x)) for x in M_.reshape(-1)) + "\n")
print(f"saved {out}: {len(F)} poses, weights {w}")
