"""Image features with 3D points taken from the per-image point cloud, and PnP between two images.

The clouds in raw_ply are a regular grid over the image (one point every ~4.57 px), so every image
keypoint can be given a 3D point. Matching keypoints between two images and solving PnP
(3D of image A, 2D of image B) gives the relative pose without relying on the shape of the noisy clouds;
the translation scale comes from the cloud of A.
"""
import cv2
import numpy as np


class Frame:
    __slots__ = ("kp", "desc", "xyz")

    def __init__(self, kp, desc, xyz):
        self.kp, self.desc, self.xyz = kp, desc, xyz   # (N,2) px, (N,128), (N,3) camera frame


class VisualMatcher:
    def __init__(self, K, n_features=6000, contrast=0.02, max_depth=30.0, ratio=0.75, reproj_px=3.0):
        self.K = np.asarray(K, dtype=np.float64)
        self.sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.max_depth = max_depth
        self.ratio = ratio
        self.reproj_px = reproj_px

    def depth_grid(self, pts):
        """Organise the cloud as a (rows, cols, 3) grid indexed by pixel; NaN where there is no point."""
        uv = pts @ self.K.T
        uv = uv[:, :2] / uv[:, 2:3]
        if not hasattr(self, "grid"):
            u = np.unique(np.round(uv[:, 0], 1)); v = np.unique(np.round(uv[:, 1], 1))
            self.grid = (u[0], (u[-1] - u[0]) / (len(u) - 1), len(u), v[0], (v[-1] - v[0]) / (len(v) - 1), len(v))
        u0, du, nu, v0, dv, nv = self.grid
        c = np.clip(np.round((uv[:, 0] - u0) / du).astype(int), 0, nu - 1)
        r = np.clip(np.round((uv[:, 1] - v0) / dv).astype(int), 0, nv - 1)
        G = np.full((nv, nu, 3), np.nan)
        G[r, c] = pts
        return G

    def extract(self, image_path, pts, scale=1.0, keep_all=False):
        """SIFT keypoints of the image + their 3D points (raw cloud `pts`, multiplied by `scale`)."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        kps, desc = self.sift.detectAndCompute(img, None)
        if desc is None or len(kps) == 0:
            return Frame(np.zeros((0, 2)), np.zeros((0, 128), np.float32), np.zeros((0, 3)))
        kp = np.array([k.pt for k in kps])
        G = self.depth_grid(np.asarray(pts, dtype=np.float64))
        u0, du, nu, v0, dv, nv = self.grid
        c = np.clip(np.round((kp[:, 0] - u0) / du).astype(int), 0, nu - 1)
        r = np.clip(np.round((kp[:, 1] - v0) / dv).astype(int), 0, nv - 1)
        xyz = G[r, c] * scale
        ok = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0.5) & (xyz[:, 2] < self.max_depth)
        # RootSIFT
        desc = desc / (np.abs(desc).sum(1, keepdims=True) + 1e-9)
        desc = np.sqrt(desc).astype(np.float32)
        if keep_all:    # every keypoint, NaN where the cloud gives no usable depth (bundle adjustment uses bearings)
            xyz[~ok] = np.nan
            return Frame(kp, desc, xyz)
        return Frame(kp[ok], desc[ok], xyz[ok])

    def match(self, A, B):
        if len(A.desc) < 8 or len(B.desc) < 8:
            return np.zeros((0, 2), int)
        ab = self.bf.knnMatch(A.desc, B.desc, k=2)
        ba = self.bf.knnMatch(B.desc, A.desc, k=1)
        back = {m[0].queryIdx: m[0].trainIdx for m in ba if m}
        out = [(m.queryIdx, m.trainIdx) for m, n in (x for x in ab if len(x) == 2)
               if m.distance < self.ratio * n.distance and back.get(m.trainIdx) == m.queryIdx]
        return np.array(out, dtype=int).reshape(-1, 2)

    def relative_pose(self, A, B, min_inliers=25):
        """Pose of camera B in camera A (T_A_B) from 3D points of A and 2D keypoints of B.

        Returns (T_A_B, n_inliers) or (None, n_inliers).
        """
        m = self.match(A, B)
        if len(m) < min_inliers:
            return None, len(m)
        obj = A.xyz[m[:, 0]].astype(np.float64)
        img = B.kp[m[:, 1]].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, reprojectionError=self.reproj_px,
                                                 iterationsCount=300, confidence=0.999,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < min_inliers:
            return None, 0 if inl is None else len(inl)
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
        T_B_A = np.eye(4)
        T_B_A[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_B_A[:3, 3] = tvec[:, 0]
        return np.linalg.inv(T_B_A), len(inl)

    def relative_pose_sym(self, A, B, min_inliers=25):
        """T_A_B from both directions (3D of A + 2D of B, and 3D of B + 2D of A): the translation is
        averaged so that the depth-scale error of a single cloud matters less."""
        T1, n1 = self.relative_pose(A, B, min_inliers)
        T2, n2 = self.relative_pose(B, A, min_inliers)
        if T1 is None and T2 is None:
            return None, max(n1, n2)
        if T2 is None:
            return T1, n1
        T2 = np.linalg.inv(T2)
        if T1 is None:
            return T2, n2
        T = T1.copy() if n1 >= n2 else T2.copy()
        T[:3, 3] = (n1 * T1[:3, 3] + n2 * T2[:3, 3]) / (n1 + n2)
        return T, min(n1, n2)
