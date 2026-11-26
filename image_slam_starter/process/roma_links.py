"""Cross-pass image links from RoMa for the bundle adjustment (sfm_ba.py --links format: fa, fb, uva, uvb).
For every candidate pair (two passes of a place with a heading difference, see the caller), RoMa outdoor matches are
sampled, points on the hood / road / cars / sky-edge and near-identical pixels dropped, 5-point RANSAC on the lens-
undistorted rays; the pair is kept only if its relative rotation agrees with the trajectory (which is accurate to
<0.1 deg) and its translation direction too -> real co-visibility, not RoMa's smooth warp between unrelated views.
usage: python image_slam_starter/process/roma_links.py TRAJ.csv PAIRS.npy part nparts out.npz [--keep 150]"""
import argparse, os, time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from romatch import roma_outdoor

ap = argparse.ArgumentParser()
ap.add_argument("traj"); ap.add_argument("pairs"); ap.add_argument("part", type=int); ap.add_argument("nparts", type=int); ap.add_argument("out")
ap.add_argument("--keep", type=int, default=150); ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--min-inl", type=int, default=150); ap.add_argument("--max-rot", type=float, default=3.0); ap.add_argument("--max-dir", type=float, default=5.0)
ap.add_argument("--upsample", default="616,952")
a = ap.parse_args()
T2 = "data/Track2"
CAM = (1037.0, 1035.38, 723.93, 463.29, -0.055911, 0.085266, 0, 0)
Kc = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]]); DIST = np.array(CAM[4:8] + (0.0,))
names = sorted(os.listdir(f"{T2}/data/images")); SEG = np.load(f"{T2}/seg.npz")
P = np.loadtxt(a.traj, delimiter=",", skiprows=1)[:, 1:].reshape(-1, 4, 4)
pairs = np.load(a.pairs)[a.part::a.nparts]
up_ = tuple(int(x) for x in a.upsample.split(","))
model = roma_outdoor(device="cuda", coarse_res=560, upsample_res=up_, use_custom_corr=False)
H, W = 928, 1440
rng = np.random.default_rng(a.part)


def static_ok(i, kp):
    u = np.clip((kp[:, 0] / 4).astype(int), 0, 359); v = np.clip((kp[:, 1] / 4).astype(int), 0, 231)
    return (kp[:, 1] < 895) & (kp[:, 1] > 15) & (SEG["road"][i][v, u] < 128) & (SEG["obj"][i][v, u] < 128)


FA, FB, UA, UB, LOG = [], [], [], [], []
t0 = time.time()
for n_, (i, j) in enumerate(pairs):
    i, j = int(i), int(j)
    warp, cert = model.match(f"{T2}/data/images/{names[i]}", f"{T2}/data/images/{names[j]}", device="cuda")
    m, c = model.sample(warp, cert, num=a.n)
    ka, kb = model.to_pixel_coordinates(m, H, W, H, W)
    ka, kb = ka.cpu().numpy().astype(np.float64), kb.cpu().numpy().astype(np.float64)
    f_ = static_ok(i, ka) & static_ok(j, kb) & (np.linalg.norm(ka - kb, axis=1) > 20)
    ka, kb = ka[f_], kb[f_]
    rec = [i, j, len(ka), 0, np.nan, np.nan, 0]
    if len(ka) >= 50:
        ra = cv2.undistortPoints(ka.reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2); rb = cv2.undistortPoints(kb.reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2)
        E, ok = cv2.findEssentialMat(ra, rb, np.eye(3), cv2.RANSAC, 0.9999, 1.0 / CAM[0])
        if E is not None and E.shape == (3, 3) and ok.sum() >= a.min_inl:
            _, Re, te, ok2 = cv2.recoverPose(E, ra, rb, np.eye(3), mask=ok.copy())
            Tji = np.linalg.inv(P[j]) @ P[i]
            er = np.degrees(np.linalg.norm(Rot.from_matrix(Re @ Tji[:3, :3].T).as_rotvec()))
            tt = Tji[:3, 3] / np.linalg.norm(Tji[:3, 3])
            ed = np.degrees(np.arccos(np.clip(float(te.ravel() @ tt), -1, 1)))
            inl = np.where(ok2.ravel() > 0)[0]
            rec[3:6] = [len(inl), er, ed]
            if er <= a.max_rot and ed <= a.max_dir and len(inl) >= a.min_inl:
                sel = rng.choice(inl, min(a.keep, len(inl)), replace=False)
                FA += [i] * len(sel); FB += [j] * len(sel); UA.append(ka[sel]); UB.append(kb[sel]); rec[6] = len(sel)
    LOG.append(rec)
    if n_ % 50 == 0:
        print(f"{n_}/{len(pairs)} pairs, kept {sum(r[6] > 0 for r in LOG)}, links {len(FA)}  ({time.time() - t0:.0f}s)", flush=True)
np.savez(a.out, fa=np.array(FA, int), fb=np.array(FB, int), uva=np.concatenate(UA) if UA else np.zeros((0, 2)),
         uvb=np.concatenate(UB) if UB else np.zeros((0, 2)), log=np.array(LOG, float))
print(f"done: {len(pairs)} pairs, kept {sum(r[6] > 0 for r in LOG)}, links {len(FA)} -> {a.out}  ({time.time() - t0:.0f}s)")
