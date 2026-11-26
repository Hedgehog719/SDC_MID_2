import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class ImuRotation:
    """Camera orientation from the IMU attitude (Track1 only), used as a rotation prior.

    R_world_cam(t) = R_world_imu(t) @ R_cam_imu^T. Only *relative* rotations between two
    image times are used, so the arbitrary IMU world frame does not matter.
    """

    def __init__(self, imu_df, R_cam_imu):
        t = imu_df["timestamp"].to_numpy(np.int64)
        q = imu_df[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy()
        t, keep = np.unique(t, return_index=True)
        self.t0 = t[0]
        self.ts = (t - self.t0) * 1e-9
        self.slerp = Slerp(self.ts, R.from_quat(q[keep]))
        self.R_imu_cam = np.asarray(R_cam_imu).T

    def R_world_cam(self, stamp_ns):
        s = np.clip((int(stamp_ns) - self.t0) * 1e-9, self.ts[0], self.ts[-1])
        return self.slerp(s).as_matrix() @ self.R_imu_cam

    def delta(self, stamp_a, stamp_b):
        """Rotation of camera b expressed in camera a: R_a_b."""
        return self.R_world_cam(stamp_a).T @ self.R_world_cam(stamp_b)
