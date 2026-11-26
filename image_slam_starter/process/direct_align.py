"""Direct (photometric) two-frame alignment with the LoGeR cloud of the first frame (prototype, Track01).

The cloud of frame a (LoGeR, pinhole back-projection of the distorted pixels) is placed on the k1-undistorted rays
(depth kept), moved by the candidate pose T_ba, projected into frame b with the same lens model and compared with the
intensity of image b (affine brightness a*I + b, Huber).  The pose is found by batched Gauss-Newton / Levenberg-
Marquardt on the GPU (analytic Jacobian, coarse-to-fine image blur 4 / 2 / 1 px), started from a no-gyro trajectory.
Compared with v10 (gyro): yaw of the relative rotation and the metric step length (cloud scale k of frame a).

usage: python process/direct_align.py <init.csv> <step> [--k1 -0.03] [--pairs-every 1]
"""
import argparse
import glob
import os
import pickle
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from scipy.spatial.transform import Rotation as Rot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.io import load_camera_info, load_ply  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("init")
ap.add_argument("step", type=int)
ap.add_argument("--k1", type=float, default=-0.03)
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--zmax", type=float, default=30.0)
ap.add_argument("--out", default=None)
ap.add_argument("--edge", type=float, default=0.0, help="drop points whose 8 image neighbours differ in depth by more than this fraction")
args = ap.parse_args()
D = os.path.join(ROOT, "Track1", "data")
RES = os.path.join(ROOT, "Track1", "result")
dev = "cuda"
K, _ = load_camera_info(f"{D}/camera_info.json")
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
IMG = sorted(glob.glob(f"{D}/images/*.jpg"))
PLY = sorted(glob.glob(f"{D}/raw_ply/*.ply"))
load = lambda p: pd.read_csv(p).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
REF = load(f"{RES}/v10_vo_shared_it5.csv")
KREF = np.load(f"{RES}/v10_vo_shared_it5.csv.scale.npy")
P0 = load(args.init)
N = len(REF)
W_, H_ = 1440, 928
SIG = (4.0, 2.0, 1.0)


def undist_norm(u, v):
    """pixel (distorted) -> undistorted normalised coordinates (k1 model, as cv2.undistortPoints)"""
    pts = cv2.undistortPoints(np.c_[u, v].reshape(-1, 1, 2).astype(np.float64), K, np.array([args.k1, 0, 0, 0, 0.0])).reshape(-1, 2)
    return pts[:, 0], pts[:, 1]


def cloud(i):
    """3-D points of frame i on the undistorted rays (depth z kept) + their pixel in image i"""
    xyz = np.asarray(load_ply(PLY[i]), np.float64)[:, :3]
    z = xyz[:, 2]
    ok = (z > 1.0) & (z < args.zmax)
    xyz, z = xyz[ok], z[ok]
    u, v = fx * xyz[:, 0] / z + cx, fy * xyz[:, 1] / z + cy          # the distorted pixel the point came from
    if args.edge > 0:              # depth discontinuities: the cloud's depth there is blurred between the two surfaces
        from scipy.spatial import cKDTree
        _, nb = cKDTree(np.c_[u, v]).query(np.c_[u, v], k=9)
        keep = np.abs(z[nb] - z[:, None]).max(1) < args.edge * z
        xyz, z, u, v = xyz[keep], z[keep], u[keep], v[keep]
    xn, yn = undist_norm(u, v)
    return np.c_[xn * z, yn * z, z].astype(np.float32), np.c_[u, v].astype(np.float32)


def pyramid(i):
    """blurred intensity + gradients of image i for every level: (L, 3, H, W)"""
    g = cv2.imread(IMG[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    out = []
    for s in SIG:
        b = cv2.GaussianBlur(g, (0, 0), s)
        out.append(np.stack([b, cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) / 8.0]))
    return np.stack(out)


def sample(img, uv):
    """img (B, C, H, W), uv (B, N, 2) pixels -> (B, C, N), bilinear"""
    gx = uv[..., 0] / (W_ - 1) * 2 - 1
    gy = uv[..., 1] / (H_ - 1) * 2 - 1
    return Fnn.grid_sample(img, torch.stack([gx, gy], -1)[:, None], align_corners=True, padding_mode="zeros")[:, :, 0]


def skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1), torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def expso3(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)[..., None]
    Kx = skew(w / th[..., 0])
    I = torch.eye(3, device=w.device).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def align(batch):
    """batch of (a, b): returns T_ba (4x4, cloud-a units), brightness, final robust residual, inlier fraction"""
    B = len(batch)
    X, I_a, M = [], [], []
    for a, _ in batch:
        xyz, uv = cloud(a)
        X.append(xyz); M.append(uv)
    n = max(len(x) for x in X)
    Xt = torch.zeros(B, n, 3, device=dev); valid0 = torch.zeros(B, n, dtype=torch.bool, device=dev)
    for j, x in enumerate(X):
        Xt[j, :len(x)] = torch.from_numpy(x); valid0[j, :len(x)] = True
    UVa = torch.zeros(B, n, 2, device=dev)
    for j, m in enumerate(M):
        UVa[j, :len(m)] = torch.from_numpy(m)
    PA = torch.from_numpy(np.stack([pyramid(a) for a, _ in batch])).to(dev)        # (B, L, 3, H, W)
    PB = torch.from_numpy(np.stack([pyramid(b) for _, b in batch])).to(dev)
    T0 = np.stack([np.linalg.inv(P0[b]) @ P0[a] for a, b in batch])               # a -> b in the init trajectory
    T0[:, :3, 3] /= np.array([KINIT[a] for a, _ in batch])[:, None]             # metric -> cloud-a units
    Rm = torch.from_numpy(T0[:, :3, :3]).float().to(dev)
    tv = torch.from_numpy(T0[:, :3, 3]).float().to(dev)
    ab = torch.tensor([[1.0, 0.0]], device=dev).repeat(B, 1)
    for L in range(len(SIG)):
        Ia = sample(PA[:, L, :1], UVa)[:, 0]                                       # (B, n) reference intensity
        gA = sample(PA[:, L, 1:], UVa).norm(dim=1)
        lam = torch.full((B,), 1e-3, device=dev)
        for it in range(25):
            Xb = (Rm @ Xt.transpose(1, 2)).transpose(1, 2) + tv[:, None]
            Z = Xb[..., 2].clamp_min(0.3)
            xn, yn = Xb[..., 0] / Z, Xb[..., 1] / Z
            r2 = xn * xn + yn * yn
            dd = 1 + args.k1 * r2
            u, v = fx * xn * dd + cx, fy * yn * dd + cy
            Sb = sample(PB[:, L], torch.stack([u, v], -1))                       # (B, 3, n)
            inside = (u > 2) & (u < W_ - 3) & (v > 2) & (v < H_ - 3) & valid0 & (Xb[..., 2] > 0.3) & (gA > 1e-3)
            res = Sb[:, 0] - (ab[:, :1] * Ia + ab[:, 1:])
            # Jacobian: d u / d xn (distortion), d xn / d Xb, d Xb / d xi = [-[Xb]x | I]  (left perturbation)
            dux = fx * (dd + 2 * args.k1 * xn * xn); duy = fx * 2 * args.k1 * xn * yn
            dvx = fy * 2 * args.k1 * xn * yn; dvy = fy * (dd + 2 * args.k1 * yn * yn)
            gu = Sb[:, 1] * dux + Sb[:, 2] * dvx                                   # dI/dxn
            gv = Sb[:, 1] * duy + Sb[:, 2] * dvy                                   # dI/dyn
            iz = 1 / Z
            dX = gu * iz; dY = gv * iz; dZ = -(gu * xn + gv * yn) * iz             # dI/dXb
            Xx, Xy, Xz = Xb[..., 0], Xb[..., 1], Xb[..., 2]
            jw = torch.stack([dY * (-Xz) + dZ * Xy, dX * Xz + dZ * (-Xx), dX * (-Xy) + dY * Xx], -1)   # dI/dw (rotation part)
            J = torch.cat([jw, torch.stack([dX, dY, dZ], -1), -Ia[..., None], -torch.ones_like(Ia)[..., None]], -1)  # (B, n, 8)
            sc = 1.4826 * torch.stack([torch.median(res[j][inside[j]].abs()) if inside[j].any() else torch.tensor(1.0, device=dev) for j in range(B)]) + 1e-4
            k = 1.345 * sc[:, None]
            w = torch.where(res.abs() <= k, torch.ones_like(res), k / res.abs()) * inside
            H = torch.einsum("bni,bn,bnj->bij", J, w, J)
            g = torch.einsum("bni,bn,bn->bi", J, w, res)
            cost = (w * res * res).sum(1)
            Hd = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2) + 1e-6)
            delta = -torch.linalg.solve(Hd, g)
            Rn = expso3(delta[:, :3]) @ Rm
            tn = (expso3(delta[:, :3]) @ tv[..., None])[..., 0] + delta[:, 3:6]
            abn = ab + delta[:, 6:]
            # accept per pair if the robust cost does not increase (evaluated with the same weights)
            Xb2 = (Rn @ Xt.transpose(1, 2)).transpose(1, 2) + tn[:, None]
            Z2 = Xb2[..., 2].clamp_min(0.3)
            x2, y2 = Xb2[..., 0] / Z2, Xb2[..., 1] / Z2
            d2 = 1 + args.k1 * (x2 * x2 + y2 * y2)
            S2 = sample(PB[:, L, :1], torch.stack([fx * x2 * d2 + cx, fy * y2 * d2 + cy], -1))[:, 0]
            r2_ = S2 - (abn[:, :1] * Ia + abn[:, 1:])
            ok2 = inside & (fx * x2 * d2 + cx > 2) & (fx * x2 * d2 + cx < W_ - 3) & (fy * y2 * d2 + cy > 2) & (fy * y2 * d2 + cy < H_ - 3)
            cost2 = (w * ok2 * r2_ * r2_).sum(1) + (w * (~ok2) * res * res).sum(1)
            acc = cost2 <= cost
            Rm = torch.where(acc[:, None, None], Rn, Rm); tv = torch.where(acc[:, None], tn, tv); ab = torch.where(acc[:, None], abn, ab)
            lam = torch.where(acc, lam * 0.3, lam * 10).clamp(1e-6, 1e4)
            if delta[:, :6].abs().max() < 1e-6:
                break
    T = np.tile(np.eye(4), (B, 1, 1)); T[:, :3, :3] = Rm.cpu().numpy(); T[:, :3, 3] = tv.cpu().numpy()
    inl = ((w > 0.99) & inside).float().sum(1) / inside.float().sum(1).clamp_min(1)
    return T, ab.cpu().numpy(), sc.cpu().numpy(), inl.cpu().numpy()


KINIT = np.load(args.init + ".scale.npy") if os.path.exists(args.init + ".scale.npy") else KREF
pairs = [(a, a + args.step) for a in range(0, N - args.step, args.every)]
t0 = time.time()
rows = []
Vt = np.linalg.svd(REF[:, :3, 3] - REF[:, :3, 3].mean(0))[2]
up = Vt[2] * np.sign(Vt[2] @ -REF[:, :3, 1].mean(0))
for s in range(0, len(pairs), args.batch):
    bt = pairs[s:s + args.batch]
    T, ab, sc, inl = align(bt)
    for (a, b), Tb, scj, inj in zip(bt, T, sc, inl):
        Tr = np.linalg.inv(REF[b]) @ REF[a]
        e = Rot.from_matrix(Tb[:3, :3] @ Tr[:3, :3].T).as_rotvec()
        upc = REF[b, :3, :3].T @ up
        Ti = np.linalg.inv(P0[b]) @ P0[a]
        ei = Rot.from_matrix(Ti[:3, :3] @ Tr[:3, :3].T).as_rotvec()
        rows.append((a, b, scj, inj, np.degrees(-e @ upc), np.degrees(-ei @ upc), np.degrees(Rot.from_matrix(Tr[:3, :3]).as_rotvec() @ -upc),
                     np.linalg.norm(Tr[:3, 3]), KREF[a] * np.linalg.norm(Tb[:3, 3]), np.linalg.norm(Ti[:3, 3])))
    if s == 0:
        print(f"  first batch {time.time()-t0:.1f}s", flush=True)
d = pd.DataFrame(rows, columns=["a", "b", "res", "inl", "yerr", "yerr_init", "turn", "dist", "dist_direct", "dist_init"])
out = args.out or f"{RES}/direct_s{args.step}.csv"
d.to_csv(out, index=False)
g = d[d.dist > 0.08 * args.step]
rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
print(f"step {args.step}: {len(d)} pairs in {time.time()-t0:.0f}s, residual median {d.res.median():.4f}, inliers median {d.inl.median():.2f}")
for name, col in (("direct", "yerr"), ("init (no-gyro run)", "yerr_init")):
    A = np.c_[np.ones(len(g)), g.dist, g.turn]; ok = np.abs(g[col] - g[col].median()) < 0.5
    c, *_ = np.linalg.lstsq(A[ok], g[col][ok], rcond=None)
    print(f"  {name:20s} yaw err vs v10: robust sd {rs(g[col])*1000:.1f} mdeg | fit {c[0]*1000:+.1f} mdeg + {c[1]*1000:+.1f} mdeg/m + {c[2]*100:+.2f}% turn")
r = g.dist_direct / g.dist
print(f"  direct step length (v10 cloud scale) / v10: median {r.median():.4f}, robust sd {100*rs(r)/r.median():.2f} %")
