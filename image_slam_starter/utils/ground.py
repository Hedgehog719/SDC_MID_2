import numpy as np


def ground_height(pts, x_half=2.5, z_min=2.0, z_max=9.0, iters=5, th=0.08):
    """Camera height above the road, from the point cloud itself (camera frame, y down).

    Fits a plane to the road patch right in front of the car (iteratively re-fitted on its
    inliers) and returns (height, unit normal pointing up to the camera, inlier count).
    The real camera height is constant, so height changes between images = scale changes
    of the predicted cloud.
    """
    m = (np.abs(pts[:, 0]) < x_half) & (pts[:, 2] > z_min) & (pts[:, 2] < z_max) & (pts[:, 1] > 0.3)
    p = pts[m]
    if len(p) < 200:
        return np.nan, None, 0
    # start from the lowest points (largest y): the road
    q = p[p[:, 1] > np.percentile(p[:, 1], 50)]
    n = np.array([0.0, -1.0, 0.0]); c = q.mean(0)
    for _ in range(iters):
        c = q.mean(0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][2]
        if n[1] > 0:
            n = -n
        d = (p - c) @ n
        q = p[np.abs(d) < th * max(1.0, abs(c @ n))]
        if len(q) < 100:
            return np.nan, None, 0
    return float(-(c @ n)), n, len(q)
