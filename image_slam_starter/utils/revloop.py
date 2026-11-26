"""Loops between passes in OPPOSITE directions (the same road driven both ways), from the point clouds.

A front camera driving the other way sees other sides of everything, so the image (SIFT + PnP) loops never find these
revisits; on Track02 most of the route is driven both ways (the long road out and back, sides of the block) and the
two directions drifted ~9 m apart sideways.  When the two cameras face each other, both clouds contain the road
between them, its curbs, walls, parked cars and trees: GICP of two small submaps (5 frames each, metric with the
current cloud scales) gives their relative pose.  Along the road such a corridor is degenerate (the registration
slides), so the factor constrains the lateral / vertical offset and the rotation tightly and the along-track offset
only loosely.  A pair is accepted when GICP from several lateral starts (+-4 m) converges to the same pose.
"""
import os

import gtsam
import numpy as np
import small_gicp as sg

from utils.ICP import preprocess
from utils.io import load_ply


class CloudCache:
    def __init__(self, files, max_range=22.0, max_up=3.0, voxel=0.1, seg=None, K=None):
        """seg: <track>/seg.npz (SegFormer): points on cars / people (P > 0.5) are dropped - they move between passes"""
        self.files, self.kw, self.c = files, dict(max_range=max_range, max_up=max_up, voxel=voxel), {}
        self.seg_path, self.K, self.obj = seg, K, None

    def get(self, i):
        if i not in self.c:
            p = load_ply(self.files[i])[:, :3]
            if self.seg_path is not None:
                if self.obj is None:
                    self.obj = np.load(self.seg_path)["obj"]
                ok = np.isfinite(p).all(1) & (p[:, 2] > 0.1)
                p = p[ok]
                uv = p @ self.K.T; uv = uv[:, :2] / uv[:, 2:3]
                r = np.clip((uv[:, 1] / 4).astype(int), 0, self.obj.shape[1] - 1); c = np.clip((uv[:, 0] / 4).astype(int), 0, self.obj.shape[2] - 1)
                p = p[self.obj[i][r, c] < 128]
            self.c[i] = preprocess(p, **self.kw)
        return self.c[i]


def submap(cc, poses, k, i, half=2):
    out = []
    for j in range(max(0, i - half), min(len(poses), i + half + 1)):
        Tij = np.linalg.inv(poses[i]) @ poses[j]
        out.append((cc.get(j) * k[j]) @ Tij[:3, :3].T + Tij[:3, 3])
    return np.ascontiguousarray(np.vstack(out))


def candidates(poses, every=5, min_gap=150, radius=25.0, ahead=(6.0, 22.0), min_turn=150.0):
    """(a, b): b drives the other way and, by the current estimate, is ahead of a (they face each other)"""
    C, Z = poses[:, :3, 3], poses[:, :3, 2]
    from scipy.spatial import cKDTree
    tree = cKDTree(C)
    out = []
    for a in range(0, len(poses), every):
        nb = np.array([b for b in tree.query_ball_point(C[a], radius) if abs(b - a) > min_gap], dtype=int)
        if len(nb) == 0:
            continue
        nb = nb[Z[nb] @ Z[a] < np.cos(np.radians(min_turn))]
        if len(nb) == 0:
            continue
        loc = np.einsum("ij,nj->ni", poses[a, :3, :3].T, C[nb] - C[a])        # b in camera a
        mid = 0.5 * (ahead[0] + ahead[1])
        ok = (loc[:, 2] > ahead[0] - 4) & (loc[:, 2] < ahead[1] + 4)
        if not ok.any():
            continue
        b = nb[ok][np.argmin(np.abs(loc[ok, 2] - mid))]
        out.append((a, int(b)))
    return out


def register(cc, poses, k, a, b, starts=(-4.0, 0.0, 4.0), threads=16, agree=1.0, min_fit=0.6):
    A, ta = sg.preprocess_points(submap(cc, poses, k, a), downsampling_resolution=0.2, num_threads=threads)
    B, _ = sg.preprocess_points(submap(cc, poses, k, b), downsampling_resolution=0.2, num_threads=threads)
    T0 = np.linalg.inv(poses[a]) @ poses[b]
    res = []
    for dx in starts:
        T = T0.copy(); T[0, 3] += dx
        for mc in (3.0, 1.5, 0.7):
            r = sg.align(A, B, ta, T, registration_type="GICP", max_correspondence_distance=mc, num_threads=threads, max_iterations=60)
            T = np.asarray(r.T_target_source)
        res.append((T, r.num_inliers / max(B.size(), 1), r.error / max(r.num_inliers, 1)))
    good = [x for x in res if x[1] >= min_fit]
    if len(good) < 2:
        return None
    lat = np.array([x[0][0, 3] for x in good]); up = np.array([x[0][1, 3] for x in good])
    if lat.max() - lat.min() > agree or up.max() - up.min() > agree:
        return None
    best = min(good, key=lambda x: x[2])
    return best[0], best[1], float(lat.max() - lat.min())


_JOB = {}


def _init(files, poses, k, max_range, max_up, voxel, seg, K):
    _JOB.update(cc=CloudCache(files, max_range, max_up, voxel, seg, K), poses=poses, k=k)


def _one(ab):
    a, b = ab
    return ab, register(_JOB["cc"], _JOB["poses"], _JOB["k"], a, b, threads=2)


def search(files, poses, k, cc=None, log=print, procs=12, **kw):
    """candidates registered in parallel processes (spawned: fork after OpenMP deadlocks)"""
    import multiprocessing as mp
    cc = cc or CloudCache(files)
    cand = candidates(poses, **kw)
    with mp.get_context("spawn").Pool(procs, initializer=_init,
                                      initargs=(list(files), poses, k, cc.kw["max_range"], cc.kw["max_up"], cc.kw["voxel"], cc.seg_path, cc.K)) as pool:
        res = pool.map(_one, cand, chunksize=4)
    out = {ab: r for ab, r in res if r is not None}
    if out:
        d = np.array([np.linalg.inv(np.linalg.inv(poses[a]) @ poses[b])[:3, :3] @ ((np.linalg.inv(poses[a]) @ poses[b])[:3, 3] - r[0][:3, 3])
                      for (a, b), r in out.items()])
        log(f"  opposite-direction loops: {len(cand)} candidates, {len(out)} accepted; current estimate minus registration "
            f"(in b, lateral / vertical / along) median {np.median(d[:, 0]):+.2f} / {np.median(d[:, 1]):+.2f} / {np.median(d[:, 2]):+.2f} m")
    else:
        log(f"  opposite-direction loops: {len(cand)} candidates, none accepted")
    return out, cc


def factors(rev, sig_rot_deg=1.5, sig_lat=0.4, sig_up=0.4, sig_along=8.0):
    """between factors (b in a); the translation error of a gtsam Pose3 between factor is in frame b:
    x lateral, y vertical, z along the road"""
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.ones(3) * np.radians(sig_rot_deg), sig_lat, sig_up, sig_along]))
    return [gtsam.BetweenFactorPose3(a, b, gtsam.Pose3(T), noise) for (a, b), (T, _, _) in rev.items()]
