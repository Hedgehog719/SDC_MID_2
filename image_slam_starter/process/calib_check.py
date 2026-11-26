"""Check camera models against the gyro-constrained trajectory with image-only relative rotations.
For every VO pair (essential-matrix inliers of the VO cache) the relative pose is refined on the Sampson error with the
keypoints undistorted by the model under test; the rotation is compared with the reference trajectory:
  yaw error = c0 + c1 * distance + c2 * yaw turned   (c1: the per-metre yaw bias of NOTES.md, c2: scale error of the yaw)
  pitch/roll: scale error against the reference, median error, Sampson residual.
usage: python process/calib_check.py ref.csv vo_cache.pkl "name:fx,fy,cx,cy,k1,k2,p1,p2" ...   (from image_slam_starter)"""
import pickle, sys, json
import multiprocessing as mp
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, ".")
from utils import vodir

ref = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
vo = pickle.load(open(sys.argv[2], "rb"))
keys = [p for p, (pa, _) in vo.items() if len(pa) >= 50]
C = ref[:, :3, 3]
up_w = np.linalg.svd(C - C.mean(0))[2][2]; up_w *= np.sign(up_w @ -ref[:, :3, 1].mean(0))
print(f"{len(keys)} pairs")
for spec in sys.argv[3:]:
    name, p = spec.split(":"); p = [float(x) for x in p.split(",")]
    Kc = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]]); dist = np.array(p[4:8] + [0.0])
    und = lambda x: cv2.undistortPoints(x.reshape(-1, 1, 2).astype(np.float64), Kc, dist).reshape(-1, 2)
    jobs = [(np.eye(3), p[0], und(vo[k][0]), und(vo[k][1]), np.linalg.inv(ref[k[0]]) @ ref[k[1]]) for k in keys]
    with mp.get_context("fork").Pool(28) as pool:
        res = pool.map(vodir.rotation_job, jobs, chunksize=32)
    for step in (3, 5, 8):
        yaw, pit, d, turn, tp, err, samp = [], [], [], [], [], [], []
        for k, r in zip(keys, res):
            if r is None or k[1] - k[0] != step:
                continue
            Rr = ref[k[0], :3, :3].T @ ref[k[1], :3, :3]
            u = ref[k[0], :3, :3].T @ up_w
            e = R.from_matrix(Rr.T @ r[0]).as_rotvec(); rv = R.from_matrix(Rr).as_rotvec()
            lat = np.cross(u, [0, 0, 1.0]); lat /= np.linalg.norm(lat)
            yaw.append(np.degrees(e @ u)); pit.append(np.degrees(e @ lat)); turn.append(np.degrees(rv @ u)); tp.append(np.degrees(rv @ lat))
            d.append(np.linalg.norm(C[k[1]] - C[k[0]])); err.append(np.degrees(np.linalg.norm(e))); samp.append(r[1])
        yaw, pit, d, turn, tp = map(np.array, (yaw, pit, d, turn, tp))
        A = np.c_[np.ones(len(d)), d, turn]
        c = np.linalg.lstsq(A, yaw, rcond=None)[0]
        big = np.abs(tp) > 0.05
        ps = np.linalg.lstsq(tp[big, None], pit[big], rcond=None)[0][0] if big.sum() > 20 else np.nan
        print(f"  {name:>10} step {step}: yaw per metre {1000*c[1]:+6.1f} mdeg/m, yaw scale {100*c[2]:+.2f} %, pitch scale {100*ps:+.2f} %, "
              f"error median {np.median(err):.3f} deg, Sampson {np.median(samp):.4f} px")
