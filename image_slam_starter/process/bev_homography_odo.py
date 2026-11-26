"""(Midterm02, 2026-11-14) Dense road-plane homography odometry: ECC (MOTION_HOMOGRAPHY) on the road region of k1-undistorted, band-passed
half-resolution images, initialised from a no-gyro trajectory; decomposed into R, t/h, plane normal.  vs v10.
usage: ecc_homog.py <init.csv> <step> [rows r0:r1]"""
import os, sys, glob, multiprocessing as mp
import cv2, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
cv2.setNumThreads(1)
sys.path.insert(0, ".")
from utils.io import load_camera_info
D = "Track1/data"; RES = "Track1/result/"
S = "Track1/result"
init_f, STEP = sys.argv[1], int(sys.argv[2])
R0, R1 = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "540:880").split(":"))
K, _ = load_camera_info(f"{D}/camera_info.json"); Kinv = np.linalg.inv(K)
IMG = sorted(glob.glob(f"{D}/images/*.jpg"))
load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
REF, P0 = load(RES + "v10_vo_shared_it5.csv"), load(init_f)
N = len(REF)
C0 = P0[:, :3, 3]
up_w = np.linalg.svd(C0 - C0.mean(0))[2][2]; up_w *= np.sign(up_w @ -P0[:, :3, 1].mean(0))
H_CAM = 1.42
MX, MY = cv2.initUndistortRectifyMap(K, np.array([-0.03, 0, 0, 0, 0.0]), None, K, (1440, 928), cv2.CV_32FC1)
Sh = np.diag([0.5, 0.5, 1.0])
MASK = np.zeros((464, 720), np.uint8); MASK[R0 // 2:R1 // 2, 80:640] = 1

def prep(i):
    img = cv2.remap(cv2.imread(IMG[i], cv2.IMREAD_GRAYSCALE), MX, MY, cv2.INTER_LINEAR).astype(np.float32)
    img = cv2.resize(img, (720, 464), interpolation=cv2.INTER_AREA)
    hp = cv2.GaussianBlur(img, (0, 0), 0.7) - cv2.GaussianBlur(img, (0, 0), 4.0)
    return (hp / np.sqrt(np.maximum(cv2.GaussianBlur(hp * hp, (0, 0), 10), 1.0))).astype(np.float32)

def job(a):
    b = a + STEP
    T = np.linalg.inv(P0[b]) @ P0[a]                     # a -> b: X_b = R X_a + t
    n = -(P0[a, :3, :3].T @ up_w)                          # ground normal (down) in camera a
    Hm = K @ (T[:3, :3] + np.outer(T[:3, 3], n) / H_CAM) @ Kinv
    W = (Sh @ Hm @ np.linalg.inv(Sh)).astype(np.float32); W /= W[2, 2]
    A, B = prep(a), prep(b)
    try:
        cc, W = cv2.findTransformECC(A, B, W, cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), MASK, 3)
    except cv2.error:
        return a, None
    Hf = np.linalg.inv(Sh) @ W.astype(np.float64) @ Sh
    _, Rs, ts, ns = cv2.decomposeHomographyMat(Hf, K)
    k = int(np.argmax([float(nn[:, 0] @ n) for nn in ns]))
    return a, (cc, Rs[k], ts[k][:, 0], ns[k][:, 0], n)

if __name__ == "__main__":
    with mp.get_context("fork").Pool(int(os.environ.get("NP", 14))) as pool:
        res = dict(pool.map(job, range(0, N - STEP), chunksize=8))
    rows = []
    for a, r in res.items():
        if r is None:
            continue
        cc, Rb, tb, nb, n0 = r
        b = a + STEP
        Tr = np.linalg.inv(REF[b]) @ REF[a]                # a -> b in v10
        upc = REF[a, :3, :3].T @ up_w
        e = Rot.from_matrix(Rb @ Tr[:3, :3].T).as_rotvec()  # measured vs reference (a -> b)
        ne = -(REF[a, :3, :3].T @ up_w)
        rows.append((a, cc, np.degrees(-e @ upc), np.degrees(Rot.from_matrix(Tr[:3, :3]).as_rotvec() @ -upc), np.linalg.norm(Tr[:3, 3]),
                     np.linalg.norm(Tr[:3, 3]) / np.linalg.norm(tb), np.degrees(np.arccos(np.clip(nb @ ne, -1, 1))),
                     np.degrees(np.arcsin(np.cross(ne, nb) @ np.array([1.0, 0, 0])))))
    d = pd.DataFrame(rows, columns=["a", "cc", "yerr", "turn", "dist", "h", "n_ang", "n_pitch_err"])
    d.to_csv(f"{S}/ecc_homog_s{STEP}.csv", index=False)
    g = d[(d.dist > 0.08 * STEP) & (d.cc > 0.5)]
    rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    A = np.c_[np.ones(len(g)), g.dist, g.turn]; ok = np.abs(g.yerr - g.yerr.median()) < 0.5
    c, *_ = np.linalg.lstsq(A[ok], g.yerr[ok], rcond=None)
    print(f"step {STEP} rows {R0}:{R1}: {len(d)} pairs, {len(g)} moving with cc > 0.5 (cc median {d.cc.median():.2f}) | yaw err median {g.yerr.median()*1000:+.1f} mdeg, "
          f"robust sd {rs(g.yerr)*1000:.1f} | fit {c[1]*1000:+.1f} mdeg/m + {c[2]*100:+.2f}% turn | implied camera height median {g.h.median():.3f}, robust sd {100*rs(g.h)/g.h.median():.2f} % | "
          f"normal vs v10 vertical median {g.n_ang.median():.2f} deg")
