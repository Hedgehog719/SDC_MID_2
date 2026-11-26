"""(Midterm02, 2026-11-14) Prototype: BEV (road-plane) odometry for Track01, rotation + translation, compared with v10 (gyro).
The front image (k1-undistorted model) is mapped onto the road plane h below the camera (level frame from the camera's
'up' vector), band-passed in the image first (as in Midterm01 bev_vo.py), then consecutive BEVs are registered with
ECC (Euclidean: rotation + translation), initialised from a no-gyro trajectory.
usage: bev_odo.py <init.csv> <step> [k1]"""
import os, sys, glob
import multiprocessing as mp
import cv2, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
cv2.setNumThreads(1)
sys.path.insert(0, ".")
from utils.io import load_camera_info

D = "Track1/data"
RES = "Track1/result/"
S = "Track1/result"
init_f, STEP = sys.argv[1], int(sys.argv[2])
K1 = float(sys.argv[3]) if len(sys.argv) > 3 else -0.03
K, _ = load_camera_info(f"{D}/camera_info.json")
IMG = sorted(glob.glob(f"{D}/images/*.jpg"))
load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
REF = load(RES + "v10_vo_shared_it5.csv")
P0 = load(init_f)
N = len(REF)

# level frame in camera coordinates from the trajectory plane normal of the INIT trajectory (no gyro information)
C0 = P0[:, :3, 3]
up_w = np.linalg.svd(C0 - C0.mean(0))[2][2]; up_w *= np.sign(up_w @ -P0[:, :3, 1].mean(0))
u = np.einsum("nji,j->ni", P0[:, :3, :3], up_w).mean(0); u /= np.linalg.norm(u)
yl = -u                                     # level 'down' in camera coordinates
zl = np.array([0, 0, 1.0]) - yl[2] * yl; zl /= np.linalg.norm(zl)
xl = np.cross(yl, zl)
H = 1.42
RESM, X0, X1, Z0, Z1 = 0.02, -2.5, 2.5, 4.0, 11.0
xs = np.arange(X0, X1, RESM); zs = np.arange(Z1, Z0, -RESM)        # cols: left -> right, rows: far -> near
GX, GZ = np.meshgrid(xs, zs)
pc = GX[..., None] * xl + H * yl + GZ[..., None] * zl
xn, yn = pc[..., 0] / pc[..., 2], pc[..., 1] / pc[..., 2]
f = 1 + K1 * (xn ** 2 + yn ** 2)
MAPX = (K[0, 0] * xn * f + K[0, 2]).astype(np.float32); MAPY = (K[1, 1] * yn * f + K[1, 2]).astype(np.float32)


def bev(i):
    img = cv2.imread(IMG[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    hp = cv2.GaussianBlur(img, (0, 0), 1.0) - cv2.GaussianBlur(img, (0, 0), 6.0)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(hp * hp, (0, 0), 15), 1.0))
    return cv2.GaussianBlur(cv2.remap(hp / sd, MAPX, MAPY, cv2.INTER_LINEAR, borderValue=0), (0, 0), 1.0)


def pix2g(c, r):
    return np.c_[X0 + c * RESM, Z1 - r * RESM]


def g2pix(g):
    return np.c_[(g[:, 0] - X0) / RESM, (Z1 - g[:, 1]) / RESM]


def rel2d(P, a, b):
    """2-D relative motion b in a, in a's level frame (x right, z forward): rotation angle (+ = towards +x? see sign fit) and translation"""
    T = np.linalg.inv(P[a]) @ P[b]
    Rl = np.c_[xl, yl, zl]                       # level -> camera columns
    Tl = Rl.T @ T[:3, :3] @ Rl
    t = Rl.T @ T[:3, 3]
    th = np.arctan2(Tl[0, 2], Tl[2, 2])          # heading change of the forward axis in the level x-z plane
    return th, np.array([t[0], t[2]])


def job(a):
    b = a + STEP
    A, B = bev(a), bev(b)
    th, t = rel2d(P0, a, b)
    # world point fixed: g_a = R(th) g_b + t  ->  g_b = R(th)^T (g_a - t)
    Rm = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])      # rotation in (x, z): forward axis turns by th
    pts_a = pix2g(np.array([0.0, 249, 0, 249]), np.array([0.0, 0, 349, 349]))
    pts_b = (Rm.T @ (pts_a - t).T).T
    W0 = cv2.estimateAffinePartial2D(g2pix(pts_a).astype(np.float32), g2pix(pts_b).astype(np.float32))[0]
    if W0 is None:
        return a, None
    # ECC: find W with B(W p) ~ A(p); MOTION_EUCLIDEAN wants a pure rotation, project W0 onto it
    ang = np.arctan2(W0[1, 0], W0[0, 0])
    W = np.array([[np.cos(ang), -np.sin(ang), W0[0, 2]], [np.sin(ang), np.cos(ang), W0[1, 2]]], np.float32)
    mask = ((A != 0) & (cv2.warpAffine(np.ones_like(B), W, (B.shape[1], B.shape[0]), flags=cv2.WARP_INVERSE_MAP) > 0)).astype(np.uint8)
    try:
        cc, W = cv2.findTransformECC(A, B, W, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), mask, 5)
    except cv2.error:
        return a, None
    pb = cv2.transform(g2pix(pts_a)[None].astype(np.float32), W)[0]
    gb = pix2g(pb[:, 0], pb[:, 1])
    # fit g_b = Rm^T (g_a - t)  ->  g_a = Rm g_b + t
    ma, mb = pts_a.mean(0), gb.mean(0)
    U, _, Vt = np.linalg.svd((pts_a - ma).T @ (gb - mb))
    Rf = U @ np.diag([1, np.linalg.det(U @ Vt)]) @ Vt
    thf = np.arctan2(Rf[0, 1], Rf[0, 0])
    tf = ma - Rf @ mb
    return a, (cc, thf, tf[0], tf[1], th)


if __name__ == "__main__":
    with mp.get_context("fork").Pool(int(os.environ.get("NP", 12))) as pool:
        res = dict(pool.map(job, range(0, N - STEP), chunksize=8))
    rows = []
    for a, r in res.items():
        if r is None:
            continue
        thr, tr = rel2d(REF, a, a + STEP)
        rows.append((a, r[0], np.degrees(r[1]), r[2], r[3], np.degrees(r[4]), np.degrees(thr), tr[0], tr[1]))
    d = pd.DataFrame(rows, columns=["a", "cc", "th", "tx", "tz", "th_init", "th_ref", "tx_ref", "tz_ref"])
    d.to_csv(f"{S}/bev_odo_s{STEP}.csv", index=False)
    g = d[(d.cc > 0.3) & (d.tz_ref > 0.05 * STEP)]
    print(f"step {STEP}: {len(d)} pairs, {len(g)} with cc > 0.3 and moving; cc median {d.cc.median():.2f}")
    for name, col in (("BEV", "th"), ("init (no-gyro run)", "th_init")):
        e = (g[col] - g.th_ref) * 1000
        A = np.c_[np.ones(len(g)), g.tz_ref, g.th_ref]
        c, *_ = np.linalg.lstsq(A, g[col] - g.th_ref, rcond=None)
        print(f"  {name:20s} yaw err vs v10: median {np.median(e):+.1f} mdeg, robust sd {1.4826*np.median(np.abs(e-np.median(e))):.1f} mdeg | "
              f"fit {c[0]*1000:+.1f} mdeg + {c[1]*1000:+.1f} mdeg/m + {c[2]*100:+.2f}% turn")
    r = g.tz / g.tz_ref
    print(f"  BEV forward / v10 forward: median {r.median():.4f}, robust sd {100*1.4826*np.median(np.abs(r-r.median()))/r.median():.2f} %;  lateral err median {1000*np.median(g.tx-g.tx_ref):+.1f} mm")
    ri = d.a.isin(g.a)
