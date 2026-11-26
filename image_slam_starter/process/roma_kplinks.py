"""Keypoint-to-keypoint cross-pass links from RoMa, to be merged into the multi-view tracks of sfm_ba.py (--links-kp).

RoMa's dense warp is sampled AT the existing SIFT keypoints of image a (feature cache), the warped position is checked
by the reverse warp (cycle < 2 px) and snapped to a SIFT keypoint of image b (< 2.5 px).  Keypoints on the hood / road /
cars are dropped.  The surviving keypoint pairs must pass a 5-point RANSAC on lens-undistorted rays, and the pair's
relative rotation / translation direction must agree with the trajectory (as in roma_links.py).  Output: fa, fb, ia, ib
(keypoint indices of the feature cache), so that sfm_ba joins the two passes' existing tracks instead of adding weak
2-view tracks.
usage: python image_slam_starter/process/roma_kplinks.py TRAJ.csv PAIRS.npy out.npz [--upsample 616,952]"""
import argparse, glob, os, time
import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from romatch import roma_outdoor

ap = argparse.ArgumentParser()
ap.add_argument("traj"); ap.add_argument("pairs"); ap.add_argument("out")
ap.add_argument("--upsample", default="616,952"); ap.add_argument("--cert", type=float, default=0.6)
ap.add_argument("--cycle", type=float, default=2.0); ap.add_argument("--snap", type=float, default=2.5)
ap.add_argument("--max-rot", type=float, default=3.0); ap.add_argument("--max-dir", type=float, default=5.0); ap.add_argument("--min-inl", type=int, default=40)
a = ap.parse_args()
T2 = "data/Track2"
CAM = (1037.0, 1035.38, 723.93, 463.29, -0.055911, 0.085266, 0, 0)
Kc = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]]); DIST = np.array(CAM[4:8] + (0.0,))
names = sorted(os.listdir(f"{T2}/data/images")); SEG = np.load(f"{T2}/seg.npz")
FEAT = glob.glob(f"{T2}/result/feat_*")[0]
P = np.loadtxt(a.traj, delimiter=",", skiprows=1)[:, 1:].reshape(-1, 4, 4)
pairs = np.load(a.pairs)
model = roma_outdoor(device="cuda", coarse_res=560, upsample_res=tuple(int(x) for x in a.upsample.split(",")), use_custom_corr=False)
H, W = 928, 1440


def kps(i):
    return np.load(f"{FEAT}/{names[i][:-4]}.npz")["kp"].astype(np.float64)


def static_ok(i, kp):
    u = np.clip((kp[:, 0] / 4).astype(int), 0, 359); v = np.clip((kp[:, 1] / 4).astype(int), 0, 231)
    return (kp[:, 1] < 895) & (kp[:, 1] > 15) & (SEG["road"][i][v, u] < 128) & (SEG["obj"][i][v, u] < 128)


def sample(field, px):
    """bilinear sample of a (h, w, C) field at pixel positions of the full image -> (n, C)"""
    g = torch.tensor(np.c_[2 * px[:, 0] / W - 1, 2 * px[:, 1] / H - 1], dtype=torch.float32, device=field.device)[None, None]
    return Fn.grid_sample(field.permute(2, 0, 1)[None].float(), g, mode="bilinear", align_corners=False)[0, :, 0].T.cpu().numpy()


FA, FB, IA, IB, LOG = [], [], [], [], []
t0 = time.time()
for n_, (i, j) in enumerate(pairs):
    i, j = int(i), int(j)
    warp, cert = model.match(f"{T2}/data/images/{names[i]}", f"{T2}/data/images/{names[j]}", device="cuda")
    if warp.dim() == 4:
        warp, cert = warp[0], cert[0]
    ws = warp.shape[1] // 2
    left = torch.cat([warp[:, :ws], cert[:, :ws, None]], -1)        # A -> B
    right = torch.cat([warp[:, ws:], cert[:, ws:, None]], -1)       # B -> A
    ka, kb_all = kps(i), kps(j)
    sa = np.where(static_ok(i, ka))[0]
    s1 = sample(left, ka[sa])
    xb = np.c_[(s1[:, 2] + 1) * W / 2, (s1[:, 3] + 1) * H / 2]
    ok = s1[:, 4] > a.cert
    s2 = sample(right, xb)
    xa2 = np.c_[(s2[:, 0] + 1) * W / 2, (s2[:, 1] + 1) * H / 2]
    ok &= np.linalg.norm(xa2 - ka[sa], axis=1) < a.cycle
    d_, jb = cKDTree(kb_all).query(xb)
    ok &= (d_ < a.snap) & static_ok(j, kb_all[jb]) & (np.linalg.norm(ka[sa] - kb_all[jb], axis=1) > 20)
    ia_, ib_ = sa[ok], jb[ok]
    if len(ib_):                                                    # one keypoint of b for one keypoint of a
        _, first = np.unique(ib_, return_index=True); ia_, ib_ = ia_[first], ib_[first]
    rec = [i, j, len(ia_), 0, np.nan, np.nan, 0]
    if len(ia_) >= a.min_inl:
        ra = cv2.undistortPoints(ka[ia_].reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2)
        rb = cv2.undistortPoints(kb_all[ib_].reshape(-1, 1, 2), Kc, DIST).reshape(-1, 2)
        E, m = cv2.findEssentialMat(ra, rb, np.eye(3), cv2.RANSAC, 0.9999, 1.0 / CAM[0])
        if E is not None and E.shape == (3, 3) and m.sum() >= a.min_inl:
            _, Re, te, m2 = cv2.recoverPose(E, ra, rb, np.eye(3), mask=m.copy())
            Tji = np.linalg.inv(P[j]) @ P[i]
            er = np.degrees(np.linalg.norm(Rot.from_matrix(Re @ Tji[:3, :3].T).as_rotvec()))
            ed = np.degrees(np.arccos(np.clip(float(te.ravel() @ (Tji[:3, 3] / np.linalg.norm(Tji[:3, 3]))), -1, 1)))
            inl = np.where(m2.ravel() > 0)[0]
            rec[3:6] = [len(inl), er, ed]
            if er <= a.max_rot and ed <= a.max_dir and len(inl) >= a.min_inl:
                FA += [i] * len(inl); FB += [j] * len(inl); IA += list(ia_[inl]); IB += list(ib_[inl]); rec[6] = len(inl)
    LOG.append(rec)
    if n_ % 50 == 0:
        print(f"{n_}/{len(pairs)} pairs, kept {sum(r[6] > 0 for r in LOG)}, keypoint links {len(FA)}  ({time.time() - t0:.0f}s)", flush=True)
np.savez(a.out, fa=np.array(FA, int), fb=np.array(FB, int), ia=np.array(IA, int), ib=np.array(IB, int), log=np.array(LOG, float))
print(f"done: {len(pairs)} pairs, kept {sum(r[6] > 0 for r in LOG)}, keypoint links {len(FA)} -> {a.out}  ({time.time() - t0:.0f}s)")
