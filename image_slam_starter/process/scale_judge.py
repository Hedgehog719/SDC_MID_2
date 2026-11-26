"""Independent judge of the local scale of a trajectory: metric depth networks.

A trajectory implies a scale for every cloud:  k_i = (distance between two frames in the trajectory) / (length of the
PnP translation in the raw units of cloud i), median over the odometry pairs of frame i.  Depth Anything V2 / ZoeDepth
give a per-frame cloud scale too (process/depth_scale_check.py), noisy (4-5 %) but independent of everything in the
pipeline and of the bundle adjustment.  The trajectory whose implied scales agree better with the networks has the
better local scale.   usage: python process/scale_judge.py pairs.pkl a.csv [b.csv ...]   (from image_slam_starter)"""
import pickle, sys
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

pairs = pickle.load(open(sys.argv[1], "rb"))
odo = [(p, r) for p, r in pairs.items() if r is not None and p[1] - p[0] <= 3]
nets = {}
for nm in ("da2", "zoe", "da2b", "hood"):
    try:
        z = np.load(f"Track1/depth_{nm}.npz"); l = np.full(1314, np.nan); l[z["frames"]] = np.log(z["s_all"]); nets[nm] = l
    except Exception:
        pass
nets["da2+zoe"] = 0.5 * (nets["da2"] + nets["zoe"])
print(f"{len(odo)} odometry pairs; residual std of log(implied cloud scale) - log(network scale) [%]: per frame / 31-frame mean")
print(" " * 34 + "".join(f"{n:>22}" for n in nets))
for f in sys.argv[2:]:
    C = np.genfromtxt(f, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)[:, :3, 3]
    est = [[] for _ in range(len(C))]
    for (a, b), r in odo:
        d = np.linalg.norm(C[b] - C[a])
        if d < 0.05 * (b - a):
            continue
        est[a].append(np.log(d / np.linalg.norm(r[3]))); est[b].append(np.log(d / np.linalg.norm(r[5])))
    k = np.array([np.median(e) if len(e) >= 2 else np.nan for e in est])
    line = f"{f.split('/')[-1][:32]:34s}"
    for n, l in nets.items():
        g = np.isfinite(k) & np.isfinite(l)
        res = np.where(g, k - l, np.nan); res -= np.nanmedian(res)
        ri = np.interp(np.arange(len(res)), np.where(g)[0], res[g])
        sm = uniform_filter1d(ri, 31, mode="nearest")
        line += f"  {100*1.4826*np.nanmedian(np.abs(res)):5.2f} / {100*np.std(sm[g]):4.2f} c{np.corrcoef(k[g], l[g])[0,1]:+.2f}"
    print(line)
