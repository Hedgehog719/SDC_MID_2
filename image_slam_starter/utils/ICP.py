import small_gicp as sg
import numpy as np


def preprocess(pts, max_range=25.0, max_up=4.0, voxel=0.15):
    """Clean one image point cloud (camera frame: x right, y down, z forward).

    The LoGeR clouds contain the sky and far points whose depth is unreliable, so keep
    only points closer than `max_range` and lower than `max_up` metres above the camera,
    then voxel-downsample (also bounds memory for submaps / loop closure).
    """
    pts = np.asarray(pts, dtype=np.float64)[:, :3]
    m = np.isfinite(pts).all(1) & (pts[:, 2] > 0.5) & (pts[:, 2] < max_range) & (pts[:, 1] > -max_up)
    pts = pts[m]
    if voxel > 0 and len(pts):
        key = np.floor(pts / voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        pts = pts[idx]
    return pts


class ICP:
    def __init__(self, voxel_size=0.25, max_corr=1.0, num_threads=8, max_iterations=30):
        self.voxel_size = voxel_size
        self.max_corr = max_corr
        self.num_threads = num_threads
        self.max_iterations = max_iterations
        self.last = None  # last small_gicp result (H, error, ...) for diagnostics

    def align(self, src_pts, tgt_pts, init_T=np.eye(4), max_corr=None):
        """GICP: source (sensor frame) -> target (world frame), starting from init_T.

        Returns (T_target_source, score). score = RMS-like residual per inlier weighted by
        the inlier ratio: lower is better; `self.fitness` holds the inlier ratio.
        """
        max_corr = self.max_corr if max_corr is None else max_corr
        tgt, tgt_tree = sg.preprocess_points(
            np.ascontiguousarray(tgt_pts[:, :3], dtype=np.float64),
            downsampling_resolution=self.voxel_size, num_threads=self.num_threads)
        src, _ = sg.preprocess_points(
            np.ascontiguousarray(src_pts[:, :3], dtype=np.float64),
            downsampling_resolution=self.voxel_size, num_threads=self.num_threads)

        res = sg.align(tgt, src, tgt_tree, np.asarray(init_T, dtype=np.float64),
                       registration_type="GICP",
                       max_correspondence_distance=max_corr,
                       num_threads=self.num_threads,
                       max_iterations=self.max_iterations)
        self.last = res
        n = max(src.size(), 1)
        self.fitness = res.num_inliers / n
        score = res.error / max(res.num_inliers, 1)

        return np.asarray(res.T_target_source), score
