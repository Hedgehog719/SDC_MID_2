"""Bundle adjustment on feature tracks.

The pose graph gets its translation lengths from the predicted clouds, whose scale changes from place to
place; loop closures only fix what differs between laps, not a scale error that all laps share at the same
place. Here every image keypoint is used as a bearing only: matches are chained into tracks, each track is
one 3D point triangulated from many images (also far points, whose cloud depth is useless), and poses and
points are optimised on the reprojection error. Shape then comes from multi-view geometry; the cloud-based
factors stay in the graph and only have to fix the overall scale.
"""
import gtsam
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def epipolar_inliers(K, Ta, Tb, xa, xb, thr=2.0):
    """matches consistent with the current relative pose (Sampson distance in pixels)"""
    T = np.linalg.inv(Tb) @ Ta                       # X_b = R X_a + t
    Rm, t = T[:3, :3], T[:3, 3]
    Ki = np.linalg.inv(K)
    xah = np.c_[xa, np.ones(len(xa))]; xbh = np.c_[xb, np.ones(len(xb))]
    if np.linalg.norm(t) < 0.03:                     # (almost) no translation: points only rotate
        p = xah @ (K @ Rm @ Ki).T
        return np.linalg.norm(p[:, :2] / p[:, 2:3] - xb, axis=1) < 1.5 * thr
    Fm = Ki.T @ skew(t) @ Rm @ Ki
    l = xah @ Fm.T; lt = xbh @ Fm
    s = (xbh * l).sum(1) ** 2 / (l[:, 0] ** 2 + l[:, 1] ** 2 + lt[:, 0] ** 2 + lt[:, 1] ** 2 + 1e-12)
    return s < thr ** 2


def build_tracks(edges, offsets, n_nodes, min_len=3):
    """edges: (E, 2) global keypoint ids (offsets[frame] + keypoint index). Returns list of arrays of
    global ids; tracks that contain two keypoints of the same image are dropped (wrong match inside)."""
    g = coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
    _, lab = connected_components(g, directed=False)
    used = np.unique(edges)
    order = used[np.argsort(lab[used], kind="stable")]
    cuts = np.where(np.diff(lab[order]))[0] + 1
    tracks = []
    for tr in np.split(order, cuts):
        if len(tr) < min_len:
            continue
        fr = np.searchsorted(offsets, tr, side="right") - 1
        if len(np.unique(fr)) == len(fr):
            tracks.append(tr)
    return tracks


def triangulate(K, Pw, obs_frames, obs_xy):
    """Linear triangulation from all views. Pw: world->pixel 3x4 projection of every frame."""
    A = np.zeros((2 * len(obs_frames), 4))
    for n, (f, xy) in enumerate(zip(obs_frames, obs_xy)):
        A[2 * n] = xy[0] * Pw[f, 2] - Pw[f, 0]
        A[2 * n + 1] = xy[1] * Pw[f, 2] - Pw[f, 1]
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else None


def make_points(K, poses, tracks, offsets, kps, ground_kp=None, max_reproj=4.0, min_parallax_deg=1.0):
    """Triangulate every track with the current poses; keep the well-conditioned ones.
    returns list of (X, frames, xy, is_ground)"""
    Tinv = np.linalg.inv(poses)
    Pw = K @ Tinv[:, :3, :]
    C = poses[:, :3, 3]
    out = []
    stats = dict(tracks=len(tracks), behind=0, reproj=0, parallax=0)
    for tr in tracks:
        fr = np.searchsorted(offsets, tr, side="right") - 1
        xy = np.array([kps[f][i - offsets[f]] for f, i in zip(fr, tr)], dtype=np.float64)
        X = triangulate(K, Pw, fr, xy)
        if X is None:
            stats["behind"] += 1; continue
        Xc = (Tinv[fr, :3, :3] @ X) + Tinv[fr, :3, 3]
        if (Xc[:, 2] <= 0.2).any():
            stats["behind"] += 1; continue
        pr = Xc @ K.T
        err = np.linalg.norm(pr[:, :2] / pr[:, 2:3] - xy, axis=1)
        keep = err < max_reproj
        if keep.sum() < 3 or keep.mean() < 0.6:
            stats["reproj"] += 1; continue
        fr, xy = fr[keep], xy[keep]
        rays = X - C[fr]; rays /= np.linalg.norm(rays, axis=1)[:, None]
        cosmin = (rays @ rays.T).min()
        if cosmin > np.cos(np.radians(min_parallax_deg)):
            stats["parallax"] += 1; continue
        out.append((X, fr, xy, bool(ground_kp is not None and ground_kp[tr].mean() > 0.5)))
    stats["points"] = len(out); stats["obs"] = int(sum(len(p[1]) for p in out))
    return out, stats


def add_factors(graph, values, K, points, sigma_px=1.0, huber=1.345):
    cal = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(huber),
                                           gtsam.noiseModel.Isotropic.Sigma(2, sigma_px))
    L = gtsam.symbol("l", 0)
    for j, (X, fr, xy, _) in enumerate(points):
        values.insert(L + j, gtsam.Point3(*X))
        for f, p in zip(fr, xy):
            graph.add(gtsam.GenericProjectionFactorCal3_S2(gtsam.Point2(*p), noise, int(f), L + j, cal))
    return L


def reprojection_stats(K, poses, points):
    Tinv = np.linalg.inv(poses)
    errs, rad, radial = [], [], []
    c = np.array([K[0, 2], K[1, 2]])
    for X, fr, xy, _ in points:
        Xc = (Tinv[fr, :3, :3] @ X) + Tinv[fr, :3, 3]
        pr = Xc @ K.T; pr = pr[:, :2] / pr[:, 2:3]
        d = pr - xy
        errs.append(np.linalg.norm(d, axis=1))
        r = xy - c; rn = np.linalg.norm(r, axis=1) + 1e-9
        rad.append(rn); radial.append((d * r).sum(1) / rn)      # >0: predicted further out than observed
    return np.concatenate(errs), np.concatenate(rad), np.concatenate(radial)


def camera_heights(poses, points, up_c, radius=4.0, min_pts=8, lat=None):
    """Height of every camera above the road, from the *triangulated* road points around it (plane fit).
    The real camera height is constant, so its variation along the route measures the local scale error of
    the reconstruction - independently of the predicted clouds."""
    from scipy.spatial import cKDTree
    G = np.array([p[0] for p in points if p[3]])
    h = np.full(len(poses), np.nan); n = np.zeros(len(poses), int)
    if len(G) < min_pts:
        return h, n
    tree = cKDTree(G)
    for i, T in enumerate(poses):
        up = T[:3, :3] @ up_c
        C = T[:3, 3]
        idx = tree.query_ball_point(C, radius + 3.0)
        if len(idx) < min_pts:
            continue
        d = G[idx] - C
        z = d @ up
        xy = d - np.outer(z, up)
        e1 = T[:3, 2] - (T[:3, 2] @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
        x, y = xy @ e1, xy @ e2
        m = (np.hypot(x, y) < radius) & (z < -0.3)
        if lat is not None:         # only the road of the own lane (no crown / camber of the other side)
            m &= np.abs(y) < lat
        if m.sum() < min_pts:
            continue
        A = np.c_[x[m], y[m], np.ones(m.sum())]; zz = z[m]
        w = np.ones(len(zz))
        for _ in range(4):      # robust plane z = a x + b y + c ; the camera is at the origin
            c = np.linalg.lstsq(A * w[:, None], zz * w, rcond=None)[0]
            r = zz - A @ c
            sgm = 1.4826 * np.median(np.abs(r)) + 1e-3
            w = np.where(np.abs(r) < 2 * sgm, 1.0, 0.0)
            if w.sum() < min_pts:
                break
        else:
            h[i] = -c[2] / np.sqrt(1 + c[0] ** 2 + c[1] ** 2); n[i] = int(w.sum())
    return h, n


# ---------------------------------------------------------------- camera height above the road
def height_pairs(poses, points, up_c, h0, fwd_range=2.5, lat_range=1.5, tol=0.35, per_frame=8, seed=0):
    """(frame, point index) pairs: triangulated road points inside the footprint of the car at that frame
    (the car drove over them, so they are road for sure and close enough for the road to be flat)."""
    from scipy.spatial import cKDTree
    gi = np.array([j for j, p in enumerate(points) if p[3]], dtype=int)
    if len(gi) == 0:
        return []
    G = np.array([points[j][0] for j in gi])
    tree = cKDTree(G)
    rng = np.random.default_rng(seed)
    out = []
    for i, T in enumerate(poses):
        up = T[:3, :3] @ up_c
        C = T[:3, 3]
        idx = np.array(tree.query_ball_point(C, np.hypot(fwd_range, lat_range) + h0 + tol), dtype=int)
        if len(idx) == 0:
            continue
        d = G[idx] - C
        z = d @ up
        e1 = T[:3, 2] - (T[:3, 2] @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
        m = (np.abs(d @ e1) < fwd_range) & (np.abs(d @ e2) < lat_range) & (np.abs(z + h0) < tol)
        sel = gi[idx[m]]
        if per_frame and len(sel) > per_frame:     # Python factors are slow; a few points per frame are enough
            sel = rng.choice(sel, per_frame, replace=False)
        out += [(i, int(j)) for j in sel]
    return out


def height_factor(i, key_point, up_c, h0, sigma):
    """(X - C_i) . (R_i up) = -h0 : the camera is h0 above the road point X"""
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                                           gtsam.noiseModel.Isotropic.Sigma(1, sigma))
    S = skew(up_c)

    def fn(this, v, H):
        T = v.atPose3(i).matrix(); X = v.atPoint3(key_point)
        Rm = T[:3, :3]; d = X - T[:3, 3]; up = Rm @ up_c
        if H is not None:
            Hp = np.zeros((1, 6)); Hp[0, :3] = -d @ Rm @ S; Hp[0, 3:] = -up_c
            H[0] = Hp; H[1] = up.reshape(1, 3)
        return np.array([d @ up + h0])

    return gtsam.CustomFactor(noise, [i, key_point], fn)
