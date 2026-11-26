"""Relative camera rotations from the raw gyroscope (Track1).

The `orientation` field of imu.csv drifts by several degrees in yaw, but the raw angular velocity,
integrated with its real timestamps and with the bias measured while the car stands still, follows the
visual rotations within about 1 degree over the whole drive. It is therefore used as rotation-only
constraints between images.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


class Gyro:
    def __init__(self, imu_df, R_cam_imu, time_offset=0.0):
        self.t = imu_df["timestamp"].to_numpy(np.int64) * 1e-9 + time_offset
        self.w = imu_df[["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]].to_numpy(np.float64)
        self.R_ci = np.asarray(R_cam_imu, dtype=np.float64)
        self.bias0 = np.zeros(3); self.bias1 = np.zeros(3)
        self.t0, self.t1 = self.t[0], self.t[-1]

    def set_bias_from_still(self, stamps_s, still):
        """stamps_s: image times [s]; still: bool per image (camera not moving, from the visual odometry).
        Bias = mean gyro of the still period at the start and at the end, linearly interpolated in between."""
        idx = np.where(still)[0]
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []
        est = []
        for r in runs:
            # keep a margin to the moment the car starts / stops; before the first and after the last
            # image the car is assumed to stand as well
            ta = self.t[0] if r[0] == 0 else stamps_s[r[0]] + 0.3
            tb = self.t[-1] if r[-1] == len(still) - 1 else stamps_s[r[-1]] - 0.3
            m = (self.t >= ta) & (self.t <= tb)
            if tb - ta > 1.0 and m.sum() > 50:
                est.append((0.5 * (ta + tb), self.w[m].mean(0), m.sum()))
        if not est:
            return False
        self.t0, self.bias0 = est[0][0], est[0][1]
        self.t1, self.bias1 = est[-1][0], est[-1][1]
        return [(round(e[0] - self.t[0], 1), e[1].round(5).tolist(), int(e[2])) for e in est]

    def bias(self, t):
        a = np.clip((t - self.t0) / max(self.t1 - self.t0, 1e-6), 0, 1)[:, None]
        return (1 - a) * self.bias0 + a * self.bias1

    def camera_rotations(self, stamps_s):
        """R_g_cam at every image time (g = arbitrary gyro start frame); only differences are meaningful."""
        dt = np.diff(self.t)
        w = self.w - self.bias(self.t)
        inc = R.from_rotvec(w[:-1] * dt[:, None])
        cum = [R.identity()]
        for r in inc:
            cum.append(cum[-1] * r)
        out = []
        for ts in stamps_s:
            j = int(np.clip(np.searchsorted(self.t, ts) - 1, 0, len(self.t) - 2))
            frac = R.from_rotvec(w[j] * np.clip(ts - self.t[j], 0, dt[j]))
            out.append((cum[j] * frac).as_matrix() @ self.R_ci.T)
        return np.array(out)
