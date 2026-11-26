"""Compare trajectories with a reference without ground truth:
  - Sim(3) ATE (all frames; median over 132-frame windows = size of the public leaderboard part)
  - relative-rotation difference over 10 / 100 frames
  - local scale: step length ratio to the reference in blocks of 40 frames (spread = how differently the scale varies
    along the route), and the same ratio averaged over the passes of a place (shared) vs each pass's own deviation
  - lever arm of the car (process/lever_profile.py): spread of the local estimate, an absolute scale check
usage: python process/traj_diag.py ref.csv a.csv [b.csv ...]      (from image_slam_starter)"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from process.compare import ate
from process.lever_profile import profile

load = lambda p: np.genfromtxt(p, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
LAPS = [0, 473, 830]


def rotdiff(A, B, w):
    return np.median([np.degrees(np.linalg.norm(R.from_matrix((A[i, :3, :3].T @ A[i + w, :3, :3]).T @ (B[i, :3, :3].T @ B[i + w, :3, :3])).as_rotvec()))
                      for i in range(0, len(A) - w, 5)])


def local_scale(T, Tr, blk=40):
    """log ratio of path length to the reference, per frame (centred window of blk frames)"""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1))]
    sr = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Tr[:, :3, 3], axis=0), axis=1))]
    n = len(T); h = blk // 2
    i0 = np.clip(np.arange(n) - h, 0, n - 1); i1 = np.clip(np.arange(n) + h, 0, n - 1)
    d, dr = s[i1] - s[i0], sr[i1] - sr[i0]
    ok = dr > 0.3 * np.median(dr)
    l = np.where(ok, np.log(np.maximum(d, 1e-9) / np.maximum(dr, 1e-9)), np.nan)
    return l - np.nanmedian(l)


def shared_own(Tr, l):
    """split a per-frame value into the mean over the passes of a place and the rest"""
    lap = np.searchsorted(LAPS, np.arange(len(Tr)), side="right") - 1
    C = Tr[:, :3, 3]
    acc, cnt = np.nan_to_num(l).copy(), np.isfinite(l).astype(float)
    for L in range(len(LAPS)):
        idx = np.where(lap == L)[0]
        d, j = cKDTree(C[idx]).query(C, distance_upper_bound=3.0)
        jj = idx[np.minimum(j, len(idx) - 1)]
        ok = (lap != L) & np.isfinite(d) & np.isfinite(l[jj])
        acc[ok] += l[jj[ok]]; cnt[ok] += 1
    sh = acc / np.maximum(cnt, 1)
    g = np.isfinite(l) & (cnt >= 3)
    return 100 * np.nanstd(sh[g]), 100 * np.nanstd((l - sh)[g])


if __name__ == "__main__":
    ref = load(sys.argv[1])
    for p in sys.argv[2:]:
        T = load(p)
        e = ate(T[:, :3, 3], ref[:, :3, 3])[0]
        w = [ate(T[s:s + 132, :3, 3], ref[s:s + 132, :3, 3])[0] for s in range(0, len(T) - 132, 10)]
        l = local_scale(T, ref)
        sh, own = shared_own(ref, l)
        lv = profile(T); g = np.isfinite(lv)
        print(f"{p}\n   ATE {e:.3f} m (132-frame windows: median {np.median(w):.3f}, max {np.max(w):.3f}) | rotation diff 10 / 100 frames "
              f"{rotdiff(T, ref, 10):.3f} / {rotdiff(T, ref, 100):.3f} deg | local scale vs ref: std {100*np.nanstd(l):.2f} % "
              f"(shared by the passes {sh:.2f} %, own {own:.2f} %) | lever arm {np.median(lv[g]):.3f} m spread {100*np.std(lv[g])/np.median(lv[g]):.1f} %")
