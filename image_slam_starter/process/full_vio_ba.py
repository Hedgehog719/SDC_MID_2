"""Full-Information 6-DOF Visual-Inertial-Kinematic SLAM & Bundle Adjustment (GTSAM 4.2 C++).

Jointly optimizes in an exact metric, gravity-aligned world frame (+Z = UP):
  - X(i) : 6-DOF IMU Pose3 at every frame i (camera pose = X(i) * T_imu_cam)
  - V(i) : 3-DOF metric IMU velocity [m/s]
  - B(i) : 6-DOF time-varying accelerometer & gyroscope bias (imuBias.ConstantBias)
  - L(j) : 3-DOF triangulated visual landmarks (in --ba-mode generic)

Factors jointly unified in ONE factor graph:
  1. Exact 100 Hz 6-DOF IMU Preintegration (gtsam.CombinedImuFactor + bias random walk)
     with exact boundary sub-interval interpolation at t_img + imu_dt_ms, self-calibrated
     3-DOF R_cam_imu AND 3-DOF IMU-to-camera lever arm r_c = p_cam - p_imu, plus ZUPT
     zero-velocity priors on stationary frames.
  2. Multi-View Visual Bundle Adjustment (C++ GenericProjectionFactorCal3_S2 with Huber loss
     or SmartProjectionPose3Factor, both natively using body_P_sensor = T_imu_cam) on
     RS34-deskewed, hood-masked, static-parked-car-filtered, undistorted 3-lap SIFT tracks,
     plus optional Round-2 exact 6-DOF RS deskew (rotational + close-wall translational v*dt/Z).
  3. Car Non-Holonomic Kinematic Constraint (NHC, exact Adjoint Jacobian on X_imu * T_imu_cam).
  4. Pure-Image Epipolar VO Directions (sequential + cross-lap --vo-loops debias).
  5. Optional Triangulated Road-Height Factors (constant camera height above triangulated road)
     and/or Anisotropic 6x6 PnP BetweenFactors (--pnp-cov) for hybrid VIO-PGO ablations.

Usage (from image_slam_starter, using .venv with gtsam + torch):
  ../.venv/bin/python process/full_vio_ba.py ../data/Track1 ../data/Track1/result/g_cam_fullrs34_pnpcov80_spd03_it5.csv --out t1_fullvio_ba
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import gtsam
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ba, nhc, vodir
from utils.gpu_match import GpuMatcher, HAS_CUDA
from utils.io import load_camera_info, load_folder, load_imu

X_KEY = lambda i: gtsam.symbol("x", int(i))
V_KEY = lambda i: gtsam.symbol("v", int(i))
B_KEY = lambda i: gtsam.symbol("b", int(i))
L_KEY = lambda j: gtsam.symbol("l", int(j))


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def gtsam_pose(T):
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), T[:3, 3])


def load_traj(path):
    df = pd.read_csv(path)
    stamps = [str(int(s)) for s in df["timestamp"].values]
    poses = df.iloc[:, 1:].values.reshape(-1, 4, 4)
    return stamps, poses


def save_traj(path, stamps, poses):
    header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
    with open(path, "w") as f:
        f.write(header + "\n")
        for s, M in zip(stamps, poses):
            f.write(s + "," + ",".join(repr(float(x)) for x in M.reshape(-1)) + "\n")


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean()
    return s, Rm, my - s * Rm @ mx


def sim3_ate(P, Q):
    s, Rm, t = umeyama(P[:, :3, 3], Q[:, :3, 3])
    err = np.linalg.norm((s * P[:, :3, 3] @ Rm.T + t) - Q[:, :3, 3], axis=1)
    return float(np.sqrt(np.mean(err ** 2))), float(s)


# ---------------------------------------------------------------- IMU Calibration & Exact Preintegration
def calibrate_imu_and_gravity(ts, P_init, imu_df, R_ci0, imu_dt_s=-0.019, lever_mode="horiz", lever_fixed=None, scale_fixed=0.0):
    """Calibrate exact R_cam_imu, stationary biases, gravity vector, metric scale, and 3-DOF lever arm r_c = p_cam - p_imu."""
    t_i = (imu_df["timestamp"].values.astype(np.int64) - int(round(ts[0] * 1e9))) * 1e-9 if False else None
    # ts is already relative seconds starting at 0; align imu timestamps with exact int64 difference outside
    raise NotImplementedError


def calibrate_imu_system(ts_rel, P_init, t_i, w_i, a_i, R_ci0, imu_dt_s=-0.019, lever_mode="horiz", lever_fixed=None, scale_fixed=0.0):
    N = len(ts_rel)
    steps = np.r_[np.linalg.norm(np.diff(P_init[:, :3, 3], axis=0), axis=1), 0.0]
    med_step = np.median(steps[steps > 0.2 * np.median(steps)])
    still_f = median_filter(steps, size=5, mode="nearest") < 0.017 * med_step
    stat_m = np.zeros(len(t_i), bool)
    for fi in np.where(still_f)[0]:
        stat_m |= np.abs(t_i - (ts_rel[fi] + imu_dt_s)) < 0.055
    if stat_m.sum() < 100:
        stat_m = (t_i < 2.5) | (t_i > t_i[-1] - 2.5)
    wb0 = w_i[stat_m].mean(0)
    a_stat_mean = a_i[stat_m].mean(0)
    g_mag_stat = float(np.linalg.norm(a_stat_mean))

    # Exact boundary-interpolated 5-frame gyro integrals (no 2% endpoint truncation!)
    _, uniq = np.unique(t_i, return_index=True)
    t_u, w_u, a_u = t_i[uniq], w_i[uniq] - wb0, a_i[uniq]
    w_vis = np.array([R.from_matrix(P_init[k, :3, :3].T @ P_init[k + 5, :3, :3]).as_rotvec() for k in range(N - 5)])
    w_int = np.zeros_like(w_vis)
    for k in range(N - 5):
        g_grid = np.linspace(ts_rel[k] + imu_dt_s, ts_rel[k + 5] + imu_dt_s, 32)
        wg = np.column_stack([np.interp(g_grid, t_u, w_u[:, c]) for c in range(3)])
        w_int[k] = np.trapz(wg, g_grid, axis=0)

    # Smooth visual accel and frame-centered IMU accel/gyro for joint 3-DOF R_cam_imu + lever-arm calibration
    cs = CubicSpline(ts_rel, P_init[:, :3, 3], axis=0)
    a_cam_w = gaussian_filter1d(cs(ts_rel, 2), sigma=1.5, axis=0)
    a_imu_f = np.zeros((N, 3))
    w_imu_f = np.zeros((N, 3))
    for k in range(N):
        tq = ts_rel[k] + imu_dt_s
        m = np.abs(t_i - tq) < 0.035
        if m.sum() > 0:
            a_imu_f[k] = a_i[m].mean(0)
            w_imu_f[k] = w_i[m].mean(0) - wb0
        else:
            idx = int(np.clip(np.searchsorted(t_i, tq), 0, len(t_i) - 1))
            a_imu_f[k] = a_i[idx]
            w_imu_f[k] = w_i[idx] - wb0
    wdot_imu_f = np.gradient(gaussian_filter1d(w_imu_f, sigma=1.5, axis=0), ts_rel, axis=0)
    Rk_all = P_init[5:N - 5, :3, :3]

    def solve_linear_accel(R_ci_try):
        a_c = a_imu_f[5:N - 5] @ R_ci_try.T
        w_c = w_imu_f[5:N - 5] @ R_ci_try.T
        wd_c = wdot_imu_f[5:N - 5] @ R_ci_try.T
        rA, ry = [], []
        for idx in range(len(Rk_all)):
            Rk = Rk_all[idx]
            M_r = -(Rk @ (skew(wd_c[idx]) + skew(w_c[idx]) @ skew(w_c[idx])))
            rA.append(np.hstack([a_cam_w[idx + 5:idx + 6].T, np.eye(3), Rk, M_r]))
            ry.append(Rk @ a_c[idx])
        A = np.vstack(rA)
        y = np.concatenate(ry)
        if lever_fixed is not None:
            r_c_ = np.asarray(lever_fixed, dtype=np.float64)
            y_adj = y - A[:, 7:10] @ r_c_
            th = np.linalg.lstsq(A[:, :7], y_adj, rcond=None)[0]
            s_, g_w_, ba_c_ = float(th[0]), th[1:4], th[4:7]
            res_ = (A[:, :7] @ th - y_adj).reshape(-1, 3)
        elif lever_mode == "full":
            th = np.linalg.lstsq(A, y, rcond=None)[0]
            s_, g_w_, ba_c_, r_c_ = float(th[0]), th[1:4], th[4:7], th[7:10]
            res_ = (A @ th - y).reshape(-1, 3)
        elif lever_mode == "horiz":
            cols = [0, 1, 2, 3, 4, 5, 6, 7, 9]
            th = np.linalg.lstsq(A[:, cols], y, rcond=None)[0]
            s_, g_w_, ba_c_ = float(th[0]), th[1:4], th[4:7]
            r_c_ = np.array([th[7], 0.0, th[8]])
            res_ = (A[:, cols] @ th - y).reshape(-1, 3)
        else:
            r_c_ = np.zeros(3)
            th = np.linalg.lstsq(A[:, :7], y, rcond=None)[0]
            s_, g_w_, ba_c_ = float(th[0]), th[1:4], th[4:7]
            res_ = (A[:, :7] @ th - y).reshape(-1, 3)
        return s_, g_w_, ba_c_, r_c_, res_

    # Joint 3-DOF R_cam_imu optimization using BOTH 5-step gyro AND 3-axis accelerometer (eliminates rank-1 yaw nullspace!)
    def joint_res(rv):
        R_try = R_ci0 @ R.from_rotvec(rv).as_matrix()
        _, _, _, _, res_a = solve_linear_accel(R_try)
        res_g = (w_int @ R_try.T) - w_vis
        return np.concatenate([res_a.ravel() / 0.15, res_g.ravel() / 0.005])

    opt_r = least_squares(joint_res, np.zeros(3), method="lm", max_nfev=30)
    R_ci = R_ci0 @ R.from_rotvec(opt_r.x).as_matrix()
    d_ext = float(np.degrees(np.linalg.norm(opt_r.x)))
    s_est, g_w, ba_c, r_c, res = solve_linear_accel(R_ci)

    if scale_fixed > 0.0:
        s_est = float(scale_fixed)
    ba0_imu = R_ci.T @ ba_c
    # Remove vertical collinearity between |g| and ba_z: anchor |g| to exact stationary gravity magnitude!
    g_dir_imu = a_stat_mean / g_mag_stat
    ba0_imu = ba0_imu - (ba0_imu @ g_dir_imu) * g_dir_imu
    # Estimate world UP direction cleanly from stationary IMU accel minus horizontal accel bias
    g_up_w_list = [P_init[fi, :3, :3] @ (R_ci @ (a_i[np.abs(t_i - (ts_rel[fi] + imu_dt_s)) < 0.05].mean(0) - ba0_imu))
                   for fi in np.where(still_f)[0] if (np.abs(t_i - (ts_rel[fi] + imu_dt_s)) < 0.05).sum() > 0]
    g_up_w = np.mean(g_up_w_list, axis=0) if g_up_w_list else g_w
    g_mag_cal = float(np.linalg.norm(g_up_w))
    up_w_init = g_up_w / g_mag_cal
    rms_a = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))

    print(f"IMU calib (dt={imu_dt_s*1e3:+.1f} ms, lever={lever_mode}): still={int(still_f.sum())} fr ({int(stat_m.sum())} samp), "
          f"R_ci_joint={d_ext:.2f} deg, |g|={g_mag_cal:.4f} m/s^2, s={s_est:.4f} m/u", flush=True)
    print(f"  wb0_imu={np.round(wb0, 5)}, ba0_imu={np.round(ba0_imu, 4)}, r_c(p_cam-p_imu)={np.round(r_c, 3)} m, accel fit RMS={rms_a:.4f} m/s^2", flush=True)
    return dict(
        R_ci=R_ci, wb0=wb0, ba0=ba0_imu, r_c=r_c, s=s_est,
        g_mag=g_mag_cal, up_w_init=up_w_init, still_f=still_f,
    )


def align_to_gravity_world(P_cam_init, up_w_init, scale_m):
    """Rotate & scale P_cam_init into exact metric world where +Z = up_w_init."""
    z_axis = up_w_init / np.linalg.norm(up_w_init)
    x_tmp = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = x_tmp - (x_tmp @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_gw = np.vstack([x_axis, y_axis, z_axis])  # maps old world -> gravity-aligned world (+Z = UP)
    P_out = np.zeros_like(P_cam_init)
    for i in range(len(P_cam_init)):
        P_out[i, :3, :3] = R_gw @ P_cam_init[i, :3, :3]
        P_out[i, :3, 3] = scale_m * (R_gw @ (P_cam_init[i, :3, 3] - P_cam_init[0, :3, 3]))
        P_out[i, 3, 3] = 1.0
    return P_out, R_gw


def build_imu_preintegrations(ts_rel, t_i, w_i, a_i, imu_dt_s, pim_params, bias0):
    """Exact sub-interval preintegration of 100 Hz IMU between consecutive frames [ts[k]+dt, ts[k+1]+dt]."""
    N = len(ts_rel)
    pims = []
    # Ensure strictly increasing timestamps for linear interpolation
    _, uniq = np.unique(t_i, return_index=True)
    t_u, w_u, a_u = t_i[uniq], w_i[uniq], a_i[uniq]

    def interp_imu(tq):
        j = int(np.clip(np.searchsorted(t_u, tq) - 1, 0, len(t_u) - 2))
        alpha = float(np.clip((tq - t_u[j]) / max(t_u[j + 1] - t_u[j], 1e-9), 0.0, 1.0))
        return (1.0 - alpha) * a_u[j] + alpha * a_u[j + 1], (1.0 - alpha) * w_u[j] + alpha * w_u[j + 1]

    for k in range(N - 1):
        t0 = ts_rel[k] + imu_dt_s
        t1 = ts_rel[k + 1] + imu_dt_s
        pim = gtsam.PreintegratedCombinedMeasurements(pim_params, bias0)
        # Collect exact grid: t0, all interior t_u in (t0, t1), t1
        i0 = int(np.searchsorted(t_u, t0, side="right"))
        i1 = int(np.searchsorted(t_u, t1, side="left"))
        t_pts = [t0]
        a_pts, w_pts = [], []
        a0, w0 = interp_imu(t0)
        a_pts.append(a0); w_pts.append(w0)
        for idx in range(i0, i1):
            t_pts.append(t_u[idx]); a_pts.append(a_u[idx]); w_pts.append(w_u[idx])
        a1, w1 = interp_imu(t1)
        t_pts.append(t1); a_pts.append(a1); w_pts.append(w1)
        for m in range(len(t_pts) - 1):
            dt = float(t_pts[m + 1] - t_pts[m])
            if dt > 1e-6:
                am = 0.5 * (a_pts[m] + a_pts[m + 1])
                wm = 0.5 * (w_pts[m] + w_pts[m + 1])
                pim.integrateMeasurement(am, wm, dt)
        pims.append(pim)
    return pims


# ---------------------------------------------------------------- Custom Camera-Frame Factors on X_imu
def nhc_on_imu_factor(ki, kj, T_ic_mat, Ad_ci, calib, sigmas):
    """Exact NHC factor applied to X_cam = X_imu * T_imu_cam, with Adjoint Jacobian wrt X_imu."""
    a, b, L = calib[:3]
    h = calib[3] if len(calib) > 3 else 0.0
    f, l, u = nhc.axes(a, b)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigmas, dtype=float)),
    )

    def fn(this, v, H):
        Ti = v.atPose3(ki).matrix() @ T_ic_mat
        Tj = v.atPose3(kj).matrix() @ T_ic_mat
        if H is None:
            return nhc.error(Ti, Tj, f, l, u, L, h=h)
        e, Hi_c, Hj_c = nhc.error(Ti, Tj, f, l, u, L, jac=True, h=h)
        H[0] = Hi_c @ Ad_ci
        H[1] = Hj_c @ Ad_ci
        return e

    return gtsam.CustomFactor(noise, [ki, kj], fn)


def vodir_on_imu_factor(ka, kb, T_ic_mat, Ad_ci, m_dir, sigmas):
    """Exact pure-image epipolar VO direction factor on X_cam = X_imu * T_imu_cam."""
    M = vodir.basis(m_dir)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigmas, dtype=float)),
    )

    def fn(this, v, H):
        Ta = v.atPose3(ka).matrix() @ T_ic_mat
        Tb = v.atPose3(kb).matrix() @ T_ic_mat
        if H is None:
            return vodir.error(Ta, Tb, M)
        e, Ha_c, Hb_c = vodir.error(Ta, Tb, M, jac=True)
        H[0] = Ha_c @ Ad_ci
        H[1] = Hb_c @ Ad_ci
        return e

    return gtsam.CustomFactor(noise, [ka, kb], fn)


def height_on_imu_factor(ki, kl, T_ic_mat, Ad_ci, up_c, h0, sigma):
    """Constant camera height above triangulated road landmark L: (L - C_cam) . (R_cam up_c) = -h0."""
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(1, sigma),
    )
    S = skew(up_c)

    def fn(this, v, H):
        Tc = v.atPose3(ki).matrix() @ T_ic_mat
        X = v.atPoint3(kl)
        Rm = Tc[:3, :3]
        d = X - Tc[:3, 3]
        up = Rm @ up_c
        if H is not None:
            Hp_c = np.zeros((1, 6))
            Hp_c[0, :3] = -d @ Rm @ S
            Hp_c[0, 3:] = -up_c
            H[0] = Hp_c @ Ad_ci
            H[1] = up.reshape(1, 3)
        return np.array([d @ up + h0])

    return gtsam.CustomFactor(noise, [ki, kl], fn)


def road_normal_pair_on_imu_factor(ki, kj, T_ic_mat, Ad_ci, d_vec, target, sigma):
    """Cross-lap road-normal height pair ((C_cam_j - C_cam_i) . d_vec - target) = 0."""
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(2.0),
        gtsam.noiseModel.Isotropic.Sigma(1, sigma),
    )
    dv = np.asarray(d_vec, dtype=np.float64)

    def fn(this, v, H):
        Ti = v.atPose3(ki).matrix() @ T_ic_mat
        Tj = v.atPose3(kj).matrix() @ T_ic_mat
        r = float((Tj[:3, 3] - Ti[:3, 3]) @ dv - target)
        if H is not None:
            Hi_c = np.zeros((1, 6)); Hi_c[0, 3:] = -(dv @ Ti[:3, :3])
            Hj_c = np.zeros((1, 6)); Hj_c[0, 3:] = +(dv @ Tj[:3, :3])
            H[0] = Hi_c @ Ad_ci
            H[1] = Hj_c @ Ad_ci
        return np.array([r])

    return gtsam.CustomFactor(noise, [ki, kj], fn)


def vp_gravity_on_imu_factor(ki, T_ic_mat, Ad_ci, up_vp_c, sigma_rad):
    """Visual vanishing-point verticality factor: R_cam^T @ e_z = up_vp_c (world +Z = UP)."""
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(3, sigma_rad),
    )
    uvp = np.asarray(up_vp_c, dtype=np.float64)
    uvp = uvp / (np.linalg.norm(uvp) + 1e-12)
    ez = np.array([0.0, 0.0, 1.0])

    def fn(this, v, H):
        Tc = v.atPose3(ki).matrix() @ T_ic_mat
        u_pred = Tc[:3, :3].T @ ez
        if H is not None:
            Hc = np.zeros((3, 6))
            Hc[:, :3] = skew(u_pred)
            H[0] = Hc @ Ad_ci
        return u_pred - uvp

    return gtsam.CustomFactor(noise, [ki], fn)


def wall_plane_factor(kl, normal_w, d_w, sigma_m):
    """Manhattan vertical building-wall coplanarity factor on landmark L: n . L + d = 0."""
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(1, sigma_m),
    )
    nw = np.asarray(normal_w, dtype=np.float64)
    H0 = nw.reshape(1, 3)

    def fn(this, v, H):
        X = v.atPoint3(kl)
        if H is not None:
            H[0] = H0
        return np.array([float(nw @ X + d_w)])

    return gtsam.CustomFactor(noise, [kl], fn)


def detect_manhattan_walls(points, P_cam, min_inliers=35, inlier_thr=0.08, max_walls=8, seed=0):
    """Detect strictly vertical planar building facades (n_z=0) among triangulated non-road landmarks."""
    z_cam_med = float(np.median(P_cam[:, 2, 3]))
    # Candidate facade landmarks: above road surface, not road, >= 4 observations
    cand_idx = [j for j, (X, fr, _, is_rd, _) in enumerate(points)
                if (not is_rd) and len(fr) >= 4 and (X[2] > z_cam_med - 1.2) and (X[2] < z_cam_med + 14.0)]
    if len(cand_idx) < min_inliers:
        return []
    cand_idx = np.array(cand_idx, dtype=int)
    XY = np.array([points[j][0][:2] for j in cand_idx], dtype=np.float64)
    avail = np.ones(len(cand_idx), bool)
    rng = np.random.default_rng(seed)
    walls = []
    for _ in range(max_walls):
        pool = np.where(avail)[0]
        if len(pool) < min_inliers:
            break
        P = XY[pool]
        ptree = cKDTree(P)
        best_in, best_nd = None, None
        for _it in range(800):
            i1 = int(rng.integers(0, len(P)))
            nb = ptree.query_ball_point(P[i1], 12.0)
            if len(nb) < 8:
                continue
            i2 = int(nb[rng.integers(0, len(nb))])
            if i1 == i2:
                continue
            dv = P[i2] - P[i1]
            ln = float(np.hypot(dv[0], dv[1]))
            if ln < 1.0:
                continue
            n2 = np.array([-dv[1] / ln, dv[0] / ln])
            d = -float(n2 @ P[i1])
            dist = np.abs(P @ n2 + d)
            inl = dist < inlier_thr
            if best_in is None or inl.sum() > best_in.sum():
                best_in = inl
                best_nd = (n2, d)
        if best_in is None or int(best_in.sum()) < min_inliers:
            break
        # Refine line via SVD on inliers and verify span >= 4.0 m (true building facade, not a narrow pole!)
        Pin = P[best_in]
        mu = Pin.mean(0)
        _, svals, Vt = np.linalg.svd(Pin - mu, full_matrices=False)
        n2 = Vt[1]
        d = -float(n2 @ mu)
        dist = np.abs(P @ n2 + d)
        inl_ref = dist < inlier_thr
        Pin = P[inl_ref]
        proj = (Pin - mu) @ Vt[0]
        span = float(proj.max() - proj.min())
        if len(Pin) < min_inliers or span < 4.0:
            avail[pool[inl_ref]] = False
            continue
        rms = float(np.sqrt(np.mean((Pin @ n2 + d) ** 2)))
        walls.append((np.array([n2[0], n2[1], 0.0]), d, cand_idx[pool[inl_ref]], span, rms))
        avail[pool[inl_ref]] = False
    return walls


def compute_bev_marking_loops(target, stamps, P_cam, cam_params, nhc_cal, h0_m, every=3,
                              sig_fwd=0.12, sig_lat=0.06, sig_yaw_deg=0.4):
    """Cross-lap BEV registration using Track 1 red curbs + white/yellow lane/zebra/'慢' markings."""
    t0 = time.time()
    data = f"{target}/data"
    img_dir = f"{data}/images"
    c = [float(x) for x in cam_params.split(",")]
    K = np.array([[c[0], 0.0, c[2]], [0.0, c[1], c[3]], [0.0, 0.0, 1.0]])
    D = np.array(c[4:8] + [0.0], dtype=np.float64)
    yaw_m, pitch_m = nhc_cal[:2] if nhc_cal is not None else (0.0, -0.03)
    F_ax, L_ax, U_ax = nhc.axes(yaw_m, pitch_m)

    # Wide BEV grid covering BOTH left & right red curbs + zebra + yellow dashes: X in [2.5, 10.0] h, Y in [-3.2, +3.2] h
    RES = 0.04  # in camera heights (~7.2 cm/px -> super fast & high SNR)
    X0, X1, Y1 = 2.5, 10.0, 3.2
    xs = np.arange(X1, X0, -RES)
    ys = np.arange(Y1, -Y1, -RES)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    P_grid = gx[..., None] * F_ax + gy[..., None] * L_ax - 1.0 * U_ax
    uv = cv2.projectPoints(P_grid.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)
    suv = (uv / 4.0).astype(np.float32)

    seg_npz = np.load(f"{target}/seg.npz") if os.path.exists(f"{target}/seg.npz") else None
    seg_obj = seg_npz["obj"] if seg_npz is not None else None
    H_img = 928.0
    in_bounds = (uv[..., 0] >= 0) & (uv[..., 0] < 1440) & (uv[..., 1] >= 0) & (uv[..., 1] <= 845)

    bev_cache = {}
    def get_bev(fi):
        if fi in bev_cache:
            return bev_cache[fi]
        bgr = cv2.imread(f"{img_dir}/{stamps[fi]}.jpg")
        if bgr is None:
            return None
        # Red-curb chromaticity + bright white/yellow lane/zebra top-hat
        b_ch, g_ch, r_ch = bgr[..., 0].astype(np.float32), bgr[..., 1].astype(np.float32), bgr[..., 2].astype(np.float32)
        red = np.clip(r_ch - 0.5 * (g_ch + b_ch) - 18.0, 0.0, 80.0) / 40.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        red_bev = cv2.remap(red, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=0.0)
        g_bev = cv2.remap(gray, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=-1.0)
        valid = in_bounds & (g_bev >= 0)
        if valid.sum() < 200:
            bev_cache[fi] = None
            return None
        g_bev[~valid] = np.median(g_bev[valid])
        ks = int(0.5 / RES) | 1
        th = cv2.morphologyEx(g_bev, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
        sd = np.sqrt(np.maximum(cv2.blur(g_bev * g_bev, (31, 31)) - cv2.blur(g_bev, (31, 31)) ** 2, 4.0))
        wh = np.clip(th / sd - 1.4, 0.0, 3.0) / 1.5
        comb = np.maximum(red_bev * 1.4, wh)
        comb[~valid] = 0.0
        if seg_obj is not None:
            om = cv2.remap(seg_obj[fi], suv[..., 0], suv[..., 1], cv2.INTER_NEAREST, borderValue=255)
            comb[om > 76] = 0.0
        # Require enough marking signal
        if (comb > 0.35).sum() < 120:
            bev_cache[fi] = None
            return None
        bev_cache[fi] = comb.astype(np.float32)
        return bev_cache[fi]

    # Same-direction cross-lap candidates
    C, Z = P_cam[:, :3, 3], P_cam[:, :3, 2]
    N = len(P_cam)
    tree = cKDTree(C)
    raw = {}
    sh = int(round(0.65 / RES))  # ±0.65 h (~±1.15 m) search window
    H_g, W_g = gx.shape
    cx_g, cy_g = W_g / 2.0, H_g / 2.0

    for a in range(0, N, every):
        nb = [b for b in tree.query_ball_point(C[a], 2.0) if (b - a) > 150 and (Z[b] @ Z[a]) > np.cos(np.radians(20.0))]
        if not nb:
            continue
        b = min(nb, key=lambda j: np.linalg.norm(C[j] - C[a]))
        ba_img = get_bev(a)
        bb_img = get_bev(b)
        if ba_img is None or bb_img is None:
            continue
        # Initial relative pose of b in a's road frame (in camera heights h0_m)
        Tab = np.linalg.inv(P_cam[a]) @ P_cam[b]
        t_ab = Tab[:3, 3] / h0_m
        xb0 = float(t_ab @ F_ax)
        yb0 = float(t_ab @ L_ax)
        f_b_in_a = Tab[:3, :3] @ F_ax
        yaw0 = float(np.arctan2(f_b_in_a @ L_ax, f_b_in_a @ F_ax))
        # Warp bb_img into a's BEV frame using (xb0, yb0, yaw0) + small yaw search dyaw in [-2, +2] deg
        # In BEV image: row r -> x = X1 - r*RES, col c -> y = Y1 - c*RES
        best = None
        for dy_deg in (-1.5, -0.75, 0.0, 0.75, 1.5):
            dy = np.radians(dy_deg)
            ang = yaw0 + dy
            ca, sa = np.cos(ang), np.sin(ang)
            # Point (xa, ya) in a corresponds to (xb, yb) in b:
            # [xa; ya] = [xb0; yb0] + R(ang) [xb; yb] => [xb; yb] = R(-ang) ([xa - xb0; ya - yb0])
            # Pad target canvas by sh on each side for template matching!
            xa_p = np.arange(X1 + sh * RES, X0 - sh * RES + 1e-6, -RES)[:H_g + 2 * sh]
            ya_p = np.arange(Y1 + sh * RES, -Y1 - sh * RES + 1e-6, -RES)[:W_g + 2 * sh]
            gxa, gya = np.meshgrid(xa_p, ya_p, indexing="ij")
            dx_a = gxa - xb0
            dy_a = gya - yb0
            xb_m = ca * dx_a + sa * dy_a
            yb_m = -sa * dx_a + ca * dy_a
            rb_m = ((X1 - xb_m) / RES).astype(np.float32)
            cb_m = ((Y1 - yb_m) / RES).astype(np.float32)
            warped_b = cv2.remap(bb_img, cb_m, rb_m, cv2.INTER_LINEAR, borderValue=0.0)
            res = cv2.matchTemplate(warped_b, ba_img, cv2.TM_CCOEFF_NORMED)
            _, mx, _, lc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                r2 = res.copy()
                r2[max(0, lc[1] - 6):lc[1] + 7, max(0, lc[0] - 6):lc[0] + 7] = -1.0
                best = (float(mx), float(dy), lc, float(r2.max()))
        mx, dy, lc, second = best
        if mx > 0.42 and (mx - second) > 0.05:
            # Template ba sits at (lc[1], lc[0]) in padded warped_b (zero shift = sh, sh):
            # moving b by -(lc - sh)*RES in (x, y) aligns b onto a!
            dfx = -(lc[1] - sh) * RES * h0_m
            dfy = -(lc[0] - sh) * RES * h0_m
            if abs(dfx) < 0.55 and abs(dfy) < 0.45:
                raw[(a, b)] = (dfx, dfy, dy, mx, mx - second)

    # Neighbor consistency filter (a +- 2*every must agree within 0.15 m / 1.0 deg)
    keys = sorted(raw)
    kept = {}
    for a, b in keys:
        r = raw[(a, b)]
        nb = [raw[k] for k in keys if k != (a, b) and abs(k[0] - a) <= 2 * every]
        if any(abs(x[0] - r[0]) < 0.15 and abs(x[1] - r[1]) < 0.12 and abs(x[2] - r[2]) < np.radians(1.0) for x in nb):
            kept[(a, b)] = r

    if kept:
        v = np.array([x[:3] for x in kept.values()])
        print(f"BEV red-curb/marking cross-lap loops: {len(raw)} matched -> {len(kept)} neighbor-consistent ({time.time()-t0:.1f}s) | "
              f"med dfx={np.median(v[:,0])*100:+.2f}cm, dfy={np.median(v[:,1])*100:+.2f}cm, dyaw={np.degrees(np.median(v[:,2])):+.2f}deg", flush=True)
    else:
        print(f"BEV red-curb/marking cross-lap loops: {len(raw)} matched -> 0 neighbor-consistent ({time.time()-t0:.1f}s)", flush=True)
    return kept, (F_ax, L_ax, U_ax)


def verify_custom_jacobians(T_ic):
    """Numerically verify all custom Adjoint Jacobians to < 1e-6."""
    T_ic_mat = T_ic.matrix()
    Ad_ci = T_ic.inverse().AdjointMap()
    rng = np.random.default_rng(42)
    Xi = gtsam.Pose3(gtsam.Rot3.Expmap(rng.normal(0, 0.3, 3)), rng.normal(0, 2.0, 3))
    Xj = gtsam.Pose3(gtsam.Rot3.Expmap(rng.normal(0, 0.3, 3)), rng.normal(0, 2.0, 3) + np.array([0.0, 0.0, 1.5]))
    Lpt = np.array([0.5, -1.0, 3.0])
    vals = gtsam.Values()
    vals.insert(X_KEY(0), Xi); vals.insert(X_KEY(1), Xj); vals.insert(L_KEY(0), Lpt)

    eps = 1e-6
    def check(name, fac, keys):
        # Compute analytical H via linearize / custom call
        e0 = fac.unwhitenedError(vals)
        # Use GTSAM's own numerical check via unwhitenedError perturbation vs factor.linearize
        # Since unwhitenedError is pre-whitened inside GTSAM, check via GaussianFactor Jacobian with sigma=1 no-Huber!
        return e0

    # Direct check of nhc, vodir, height, road_normal with isotropic sigma=1 (no Huber)
    f_n, l_n, u_n = nhc.axes(0.01, -0.03)
    Ti_c = Xi.matrix() @ T_ic_mat; Tj_c = Xj.matrix() @ T_ic_mat
    e0, Hi_c, Hj_c = nhc.error(Ti_c, Tj_c, f_n, l_n, u_n, 2.0, jac=True)
    Hi_an = Hi_c @ Ad_ci; Hj_an = Hj_c @ Ad_ci
    Hi_num = np.zeros_like(Hi_an); Hj_num = np.zeros_like(Hj_an)
    for k in range(6):
        d = np.zeros(6); d[k] = eps
        ep = nhc.error(Xi.retract(d).matrix() @ T_ic_mat, Tj_c, f_n, l_n, u_n, 2.0)
        em = nhc.error(Xi.retract(-d).matrix() @ T_ic_mat, Tj_c, f_n, l_n, u_n, 2.0)
        Hi_num[:, k] = (ep - em) / (2 * eps)
        ep2 = nhc.error(Ti_c, Xj.retract(d).matrix() @ T_ic_mat, f_n, l_n, u_n, 2.0)
        em2 = nhc.error(Ti_c, Xj.retract(-d).matrix() @ T_ic_mat, f_n, l_n, u_n, 2.0)
        Hj_num[:, k] = (ep2 - em2) / (2 * eps)
    err_nhc = max(np.max(np.abs(Hi_an - Hi_num)), np.max(np.abs(Hj_an - Hj_num)))

    M_vo = vodir.basis(np.array([0.05, -0.02, 0.998]) / np.linalg.norm([0.05, -0.02, 0.998]))
    _, Va_c, Vb_c = vodir.error(Ti_c, Tj_c, M_vo, jac=True)
    Va_an = Va_c @ Ad_ci; Vb_an = Vb_c @ Ad_ci
    Va_num = np.zeros_like(Va_an); Vb_num = np.zeros_like(Vb_an)
    for k in range(6):
        d = np.zeros(6); d[k] = eps
        Va_num[:, k] = (vodir.error(Xi.retract(d).matrix() @ T_ic_mat, Tj_c, M_vo) - vodir.error(Xi.retract(-d).matrix() @ T_ic_mat, Tj_c, M_vo)) / (2 * eps)
        Vb_num[:, k] = (vodir.error(Ti_c, Xj.retract(d).matrix() @ T_ic_mat, M_vo) - vodir.error(Ti_c, Xj.retract(-d).matrix() @ T_ic_mat, M_vo)) / (2 * eps)
    err_vo = max(np.max(np.abs(Va_an - Va_num)), np.max(np.abs(Vb_an - Vb_num)))

    # Height factor Jacobian check
    up_c = np.array([0.0, -0.999, -0.03]); up_c /= np.linalg.norm(up_c)
    Rm = Ti_c[:3, :3]; diff_h = Lpt - Ti_c[:3, 3]
    Hh_c = np.zeros((1, 6)); Hh_c[0, :3] = -diff_h @ Rm @ skew(up_c); Hh_c[0, 3:] = -up_c
    Hh_an = Hh_c @ Ad_ci
    Hh_num = np.zeros_like(Hh_an)
    for k in range(6):
        d = np.zeros(6); d[k] = eps
        Tp = Xi.retract(d).matrix() @ T_ic_mat; Tm = Xi.retract(-d).matrix() @ T_ic_mat
        ep = (Lpt - Tp[:3, 3]) @ (Tp[:3, :3] @ up_c)
        em = (Lpt - Tm[:3, 3]) @ (Tm[:3, :3] @ up_c)
        Hh_num[0, k] = (ep - em) / (2 * eps)
    err_h = float(np.max(np.abs(Hh_an - Hh_num)))

    # Verify exact BetweenFactorPose3 camera -> IMU frame & covariance transform
    Ad_ic = T_ic.AdjointMap()
    M_cam = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.05, 0.01])), np.array([0.1, -0.02, 1.4]))
    A_rand = rng.normal(0, 0.05, (6, 6))
    C_cam = A_rand @ A_rand.T + np.diag([1e-4]*3 + [1e-3]*3)
    M_imu = T_ic.compose(M_cam).compose(T_ic.inverse())
    C_imu = Ad_ic @ C_cam @ Ad_ic.T
    C_imu = 0.5 * (C_imu + C_imu.T)
    f_cam = gtsam.BetweenFactorPose3(X_KEY(0), X_KEY(1), M_cam, gtsam.noiseModel.Gaussian.Covariance(C_cam))
    f_imu = gtsam.BetweenFactorPose3(X_KEY(0), X_KEY(1), M_imu, gtsam.noiseModel.Gaussian.Covariance(C_imu))
    v_cam = gtsam.Values()
    v_cam.insert(X_KEY(0), Xi.compose(T_ic)); v_cam.insert(X_KEY(1), Xj.compose(T_ic))
    err_btw = abs(f_cam.error(v_cam) - f_imu.error(vals)) / max(1.0, f_cam.error(v_cam))

    # Verify VP gravity Jacobian wrt X_imu
    ez = np.array([0.0, 0.0, 1.0])
    u_pred0 = Ti_c[:3, :3].T @ ez
    Hvp_an = np.zeros((3, 6)); Hvp_an[:, :3] = skew(u_pred0); Hvp_an = Hvp_an @ Ad_ci
    Hvp_num = np.zeros((3, 6))
    for k in range(6):
        d = np.zeros(6); d[k] = eps
        up_p = (Xi.retract(d).matrix() @ T_ic_mat)[:3, :3].T @ ez
        up_m = (Xi.retract(-d).matrix() @ T_ic_mat)[:3, :3].T @ ez
        Hvp_num[:, k] = (up_p - up_m) / (2 * eps)
    err_vp = float(np.max(np.abs(Hvp_an - Hvp_num)))

    print(f"Jacobian verification: NHC={err_nhc:.2e}, VO={err_vo:.2e}, Height={err_h:.2e}, BtwCov={err_btw:.2e}, VPGrav={err_vp:.2e}", flush=True)
    assert max(err_nhc, err_vo, err_h, err_btw, err_vp) < 1e-6, "Jacobian/BetweenCov check failed!"


# ---------------------------------------------------------------- Visual Features, Deskew & Track Building
def prepare_visual_tracks(
    target, stamps, P_cam_init, ts_rel, t_i, w_i, wb0, R_ci, imu_dt_s,
    cam_params, mask_v_max=850.0, rs_ms=34.0, mask_obj=128, keep_static_obj=True,
    min_parallax=0.7, max_reproj=2.5, max_obs=450000, loop_radius=3.5, loop_per_frame=6,
    V_cam_w=None, rs_6dof_depths=None, prev_raw_m=None
):
    """Extract/load SIFT features, apply hood mask, undistortion, RS34 (or 6-DOF RS), match & triangulate tracks."""
    t0 = time.time()
    res_dir = f"{target}/result"
    data = f"{target}/data"
    N = len(stamps)
    K_raw, _ = load_camera_info(f"{data}/camera_info.json")
    H_img = float(cv2.imread(f"{data}/images/{stamps[0]}.jpg").shape[0])

    # Self-calibrated camera K_true, dist -> undistort to virtual pinhole K_pin
    c = [float(x) for x in cam_params.split(",")]
    K_true = np.array([[c[0], 0.0, c[2]], [0.0, c[1], c[3]], [0.0, 0.0, 1.0]])
    dist = np.array(c[4:8] + [0.0], dtype=np.float64)
    K_pin = K_true.copy()  # exact undistorted pinhole intrinsics

    feat_dir = [f"{res_dir}/{d}" for d in sorted(os.listdir(res_dir)) if d.startswith("feat_")][0]
    seg_npz = np.load(f"{target}/seg.npz") if os.path.exists(f"{target}/seg.npz") else None
    seg_obj = seg_npz["obj"] if (seg_npz is not None and mask_obj > 0) else None
    seg_road = seg_npz["road"] if (seg_npz is not None and "road" in seg_npz) else None

    kps, descs, obj_flags, road_flags, v_raw_list = [], [], [], [], []
    n_hood, n_obj = 0, 0
    for fi, s in enumerate(stamps):
        z = np.load(f"{feat_dir}/{s}.npz")
        kp = z["kp"].astype(np.float64)
        dc = z["desc"].astype(np.float32)
        xyz = z["xyz"].astype(np.float64)
        h_g = float(z["h"])
        if mask_v_max > 0:
            km = kp[:, 1] <= mask_v_max
            n_hood += int((~km).sum())
            kp, dc, xyz = kp[km], dc[km], xyz[km]
        v_raw = kp[:, 1].copy()

        # Tag road & object keypoints on raw pixel coordinates
        is_obj = np.zeros(len(kp), bool)
        if seg_obj is not None:
            rr = np.clip((kp[:, 1] / 4).astype(int), 0, seg_obj.shape[1] - 1)
            cc = np.clip((kp[:, 0] / 4).astype(int), 0, seg_obj.shape[2] - 1)
            is_obj = seg_obj[fi, rr, cc] >= mask_obj
            if not keep_static_obj:
                km2 = ~is_obj
                n_obj += int(is_obj.sum())
                kp, dc, xyz, v_raw, is_obj = kp[km2], dc[km2], xyz[km2], v_raw[km2], is_obj[km2]
            else:
                n_obj += int(is_obj.sum())
        is_road = np.isfinite(xyz[:, 2]) & (np.abs(xyz[:, 1] - h_g) < 0.1 * h_g) & (np.abs(xyz[:, 0]) < 4.0) & (xyz[:, 2] < 12.0)
        if seg_road is not None:
            rr = np.clip((kp[:, 1] / 4).astype(int), 0, seg_road.shape[1] - 1)
            cc = np.clip((kp[:, 0] / 4).astype(int), 0, seg_road.shape[2] - 1)
            is_road &= (seg_road[fi, rr, cc] > 128)

        # Undistort to exact pinhole K_pin
        kp_u = cv2.undistortPoints(kp.reshape(-1, 1, 2), K_true, dist, P=K_pin).reshape(-1, 2)

        # Exact 100 Hz IMU rotational rolling-shutter deskew
        if rs_ms != 0.0 and len(kp_u) > 0:
            tq = ts_rel[fi] + imu_dt_s
            m = np.abs(t_i - tq) < 0.025
            w_imu = w_i[m].mean(0) if m.sum() else w_i[int(np.clip(np.searchsorted(t_i, tq), 0, len(t_i) - 1))]
            w_c = R_ci @ (w_imu - wb0)
            dt_v = (rs_ms * 1e-3) * (v_raw - K_true[1, 2]) / H_img
            rays = np.c_[(kp_u[:, 0] - K_pin[0, 2]) / K_pin[0, 0], (kp_u[:, 1] - K_pin[1, 2]) / K_pin[1, 1], np.ones(len(kp_u))]
            r0 = R.from_rotvec(w_c[None, :] * dt_v[:, None]).apply(rays)
            u0 = r0[:, :2] / r0[:, 2:3]
            kp_u = np.c_[u0[:, 0] * K_pin[0, 0] + K_pin[0, 2], u0[:, 1] * K_pin[1, 1] + K_pin[1, 2]]

        kps.append(kp_u)
        descs.append(dc)
        obj_flags.append(is_obj)
        road_flags.append(is_road)
        v_raw_list.append(v_raw)

    offsets = np.r_[0, np.cumsum([len(k) for k in kps])]
    obj_all = np.concatenate(obj_flags)
    road_all = np.concatenate(road_flags)
    print(f"Visual prep: {offsets[-1]} kps (hood -{n_hood}, obj tagged {n_obj}, road {int(road_all.sum())}), K_pin f={K_pin[0,0]:.2f}/{K_pin[1,1]:.2f}", flush=True)

    # Match pairs (reuse prev_raw_m if supplied in Round 2!)
    if prev_raw_m is None:
        pairs = {(i, i + s) for s in (1, 2, 3, 5, 8, 13, 21) for i in range(N - s)}
        C, Z = P_cam_init[:, :3, 3], P_cam_init[:, :3, 2]
        tree = cKDTree(C)
        for a in range(N):
            nb = np.array([b for b in tree.query_ball_point(C[a], loop_radius) if abs(b - a) > 150], dtype=int)
            if len(nb) == 0:
                continue
            nb = nb[Z[nb] @ Z[a] > np.cos(np.radians(30.0))]
            nb = nb[np.argsort(np.linalg.norm(C[nb] - C[a], axis=1))]
            chosen = []
            for b in nb:
                if all(abs(b - c_) >= 4 for c_ in chosen):
                    chosen.append(b)
                if len(chosen) >= loop_per_frame:
                    break
            pairs.update((min(a, b), max(a, b)) for b in chosen)
        pairs = sorted(pairs)
        dev = "cuda" if HAS_CUDA else "cpu"
        gm = GpuMatcher(descs, ratio=0.75, half=True, device=dev)
        raw_m = {}
        with ThreadPoolExecutor(8) as ex:
            for pq, m in zip(pairs, ex.map(lambda pq: gm.match(*pq), pairs)):
                if len(m) >= 20:
                    raw_m[pq] = m
        del gm
        if dev == "cuda":
            import torch
            torch.cuda.empty_cache()
        print(f"  matched {len(raw_m)}/{len(pairs)} pairs on {dev} ({time.time()-t0:.1f}s)", flush=True)
    else:
        raw_m = prev_raw_m

    # Epipolar verification using current metric poses
    edges = []
    for (a, b), m in raw_m.items():
        ok = ba.epipolar_inliers(K_pin, P_cam_init[a], P_cam_init[b], kps[a][m[:, 0]], kps[b][m[:, 1]], thr=2.0)
        if ok.sum() >= 15:
            edges.append(np.c_[offsets[a] + m[ok, 0], offsets[b] + m[ok, 1]])
    edges = np.concatenate(edges)

    # Conflict-splitting tracks (keep clean non-conflicting observations!)
    g = coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])), shape=(offsets[-1], offsets[-1]))
    _, lab = connected_components(g, directed=False)
    used = np.unique(edges)
    order = used[np.argsort(lab[used], kind="stable")]
    cuts = np.where(np.diff(lab[order]))[0] + 1
    tracks = []
    for tr in np.split(order, cuts):
        if len(tr) < 3:
            continue
        fr = np.searchsorted(offsets, tr, side="right") - 1
        u_fr, cnts = np.unique(fr, return_counts=True)
        if (cnts == 1).all():
            tracks.append(tr)
        else:
            keep_fr = set(u_fr[cnts == 1].tolist())
            sub = tr[[f in keep_fr for f in fr]]
            if len(sub) >= 3:
                tracks.append(sub)

    # Triangulate & filter (with static parked-car rigidity test!)
    Tinv = np.linalg.inv(P_cam_init)
    Pw = K_pin @ Tinv[:, :3, :]
    C_init = P_cam_init[:, :3, 3]
    points = []
    n_kept_obj, n_drop_obj = 0, 0
    for tr in tracks:
        fr = np.searchsorted(offsets, tr, side="right") - 1
        loc_idx = tr - offsets[fr]
        xy = np.array([kps[f][li] for f, li in zip(fr, loc_idx)], dtype=np.float64)
        X = ba.triangulate(K_pin, Pw, fr, xy)
        if X is None:
            continue
        Xc = (Tinv[fr, :3, :3] @ X) + Tinv[fr, :3, 3]
        if (Xc[:, 2] <= 0.3).any() or (Xc[:, 2] > 300.0).any():
            continue
        # Optional 6-DOF translational RS adjustment on triangulated exact depth Z!
        if V_cam_w is not None and rs_ms != 0.0:
            xy_adj = xy.copy()
            for idx_k, (f_, li_) in enumerate(zip(fr, loc_idx)):
                dt_v = (rs_ms * 1e-3) * (v_raw_list[f_][li_] - K_true[1, 2]) / H_img
                v_c = P_cam_init[f_, :3, :3].T @ V_cam_w[f_]
                # Camera moves by +v_c * dt_v -> point in cam frame moves by -v_c * dt_v -> undo by +v_c * dt_v!
                Xc_0 = Xc[idx_k] + v_c * dt_v
                if Xc_0[2] > 0.3:
                    uv_t = K_pin @ Xc_0
                    uv_t = uv_t[:2] / uv_t[2]
                    uv_c = K_pin @ Xc[idx_k]
                    uv_c = uv_c[:2] / uv_c[2]
                    xy_adj[idx_k] = xy[idx_k] + (uv_t - uv_c)
            xy = xy_adj
        pr = Xc @ K_pin.T
        err = np.linalg.norm(pr[:, :2] / pr[:, 2:3] - xy, axis=1)
        is_o = obj_all[tr].mean() > 0.35
        if is_o:
            keep = err < 1.2
            if keep.sum() < 6 or keep.mean() < 0.85:
                n_drop_obj += 1
                continue
            n_kept_obj += 1
        else:
            keep = err < max_reproj
            if keep.sum() < 3 or keep.mean() < 0.65:
                continue
        fr, xy = fr[keep], xy[keep]
        tr_k = tr[keep]
        rays = X - C_init[fr]
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        if (rays @ rays.T).min() > np.cos(np.radians(min_parallax)):
            continue
        is_rd = bool(road_all[tr_k].mean() > 0.5)
        span = int(fr[-1] - fr[0])
        points.append((X, fr, xy, is_rd, span))

    # Prioritize road points + cross-lap revisits (span > 150) + longest tracks up to max_obs
    total_obs = sum(len(p[1]) for p in points)
    if max_obs > 0 and total_obs > max_obs:
        points.sort(key=lambda p: (not p[3], not (p[4] > 150), -len(p[1])))
        acc = np.cumsum([len(p[1]) for p in points])
        points = points[:int(np.searchsorted(acc, max_obs))]
    used_obs = sum(len(p[1]) for p in points)
    n_rd = sum(1 for p in points if p[3])
    n_rev = sum(1 for p in points if p[4] > 150)
    print(f"  triangulated {len(points)} pts / {used_obs} obs (of {total_obs}; road={n_rd}, revisits={n_rev}, static_obj={n_kept_obj}, drop_mov={n_drop_obj}, {time.time()-t0:.1f}s)", flush=True)
    return K_pin, points, raw_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="Track directory, e.g. ../data/Track1")
    ap.add_argument("init", help="Initial camera trajectory CSV")
    ap.add_argument("--out", required=True, help="Output filename inside <target>/result/")
    ap.add_argument("--cam", default="1037.0,1035.38,723.93,463.29,-0.055911,0.085266,0,0")
    ap.add_argument("--imu-dt-ms", type=float, default=-19.0, help="t_imu = t_img + imu_dt_ms [ms]")
    ap.add_argument("--rs-readout-ms", type=float, default=34.0)
    ap.add_argument("--rs-6dof", action="store_true", help="Round-2 exact 6-DOF translational RS using triangulated depth Z")
    ap.add_argument("--mask-v-max", type=float, default=850.0)
    ap.add_argument("--lever-mode", choices=["horiz", "full", "zero"], default="horiz", help="IMU-to-camera lever arm calibration")
    ap.add_argument("--lever-fixed", default=None, help="Override lever arm r_c=p_cam-p_imu in cam frame 'rx,ry,rz' [m]")
    ap.add_argument("--scale-fixed", type=float, default=0.0, help="Override metric scale [m/u] (0 = auto from IMU)")
    # IMU noise tuning
    ap.add_argument("--acc-sigma", type=float, default=0.15, help="Accel white noise [m/s^2 / sqrt(Hz)]")
    ap.add_argument("--gyro-sigma", type=float, default=0.008, help="Gyro white noise [rad/s / sqrt(Hz)]")
    ap.add_argument("--acc-rw", type=float, default=0.002, help="Accel bias random walk [m/s^3 / sqrt(Hz)]")
    ap.add_argument("--gyro-rw", type=float, default=0.0002, help="Gyro bias random walk [rad/s^2 / sqrt(Hz)]")
    ap.add_argument("--zupt-sigma", type=float, default=0.01, help="Stationary velocity prior [m/s]")
    ap.add_argument("--no-imu", action="store_true", help="Disable CombinedImuFactor (ablation)")
    # Visual BA & geometric factors
    ap.add_argument("--ba-mode", choices=["generic", "smart", "none"], default="generic")
    ap.add_argument("--ba-sigma", type=float, default=1.0, help="Reprojection pixel sigma [px]")
    ap.add_argument("--ba-huber", type=float, default=1.345)
    ap.add_argument("--max-obs", type=int, default=450000)
    ap.add_argument("--min-parallax", type=float, default=0.7)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--nhc-sigma", type=float, default=0.3, help="Car NHC angle sigma [deg] (0 = off)")
    ap.add_argument("--vo-cache", default=None, help="Optional vo_*.pkl cache for VO direction factors")
    ap.add_argument("--vol-cache", default=None, help="Optional vol_*.pkl cache for cross-lap VO-loop directions")
    ap.add_argument("--vo-sigma", type=float, default=0.5)
    ap.add_argument("--vo-sigma-up", type=float, default=0.3)
    ap.add_argument("--height-sigma", type=float, default=0.03, help="Triangulated road-height factor sigma [m] (0 = off)")
    ap.add_argument("--height-per-frame", type=int, default=6)
    ap.add_argument("--rn-pairs", default=None, help="Optional cross-lap road-normal height constraints (.npz)")
    ap.add_argument("--rn-stride", type=int, default=2)
    ap.add_argument("--pnp-pairs", default=None, help="Optional pairs_*.pkl cache for anisotropic 6x6 PnP BetweenFactors")
    ap.add_argument("--pnp-scales", default=None, help="Optional *.scale.npy per-frame cloud scales for --pnp-pairs")
    ap.add_argument("--pnp-cov", type=float, default=0.0, help="Anisotropic 6x6 PnP covariance multiplier (e.g. 8.0; 0 = off)")
    ap.add_argument("--pnp-min-inl", type=int, default=50, help="Min bidirectional PnP inliers for --pnp-pairs")
    ap.add_argument("--pnp-scale-sig", default="0.05,0.01", help="Depth scale/dir sigma along/perp translation")
    ap.add_argument("--pnp-floor", default="0.05,0.005", help="Floor [deg, m] for 6x6 PnP covariance")
    ap.add_argument("--wall-plane-sigma", type=float, default=0.0, help="Manhattan vertical wall coplanarity sigma [m] (0 = off)")
    ap.add_argument("--vp-grav-sigma", type=float, default=0.0, help="Visual VP gravity factor sigma [deg] using gravity.npz (0 = off)")
    ap.add_argument("--bev-marking-sigma", type=float, default=0.0, help="Cross-lap BEV red-curb/zebra lateral sigma [m] (0 = off)")
    ap.add_argument("--bev-every", type=int, default=3, help="Frame stride for BEV cross-lap marking loops")
    ap.add_argument("--pose-prior", type=float, default=0.0, help="Optional weak camera position prior to init [m] (0 = 100%% FREE)")
    ap.add_argument("--speed-sigma", type=float, default=0.0, help="Optional speed smoothness [m/s^2] (0 = off)")
    args = ap.parse_args()

    t_start = time.time()
    data = f"{args.target}/data"
    res_dir = f"{args.target}/result"
    stamps, P_init = load_traj(args.init)
    N = len(stamps)
    ts_ns = np.array([int(s) for s in stamps], dtype=np.int64)
    ts_rel = (ts_ns - ts_ns[0]) * 1e-9

    ci = json.load(open(f"{data}/camera_info.json"))
    R_ci0 = np.array(ci["R_cam_imu"], dtype=np.float64)
    imu_df = load_imu(f"{data}/imu.csv")
    t_i = (imu_df["timestamp"].values.astype(np.int64) - ts_ns[0]) * 1e-9
    w_i = imu_df[["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]].values.astype(np.float64)
    a_i = imu_df[["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]].values.astype(np.float64)
    imu_dt_s = args.imu_dt_ms * 1e-3

    # 1. Self-calibrate IMU extrinsic R_ci, lever arm r_c = p_cam - p_imu, scale s, biases & world UP
    lev_fix = [float(x) for x in args.lever_fixed.split(",")] if args.lever_fixed else None
    cal = calibrate_imu_system(ts_rel, P_init, t_i, w_i, a_i, R_ci0, imu_dt_s, args.lever_mode, lev_fix, args.scale_fixed)
    R_ci = cal["R_ci"]
    r_c = cal["r_c"]  # p_cam - p_imu expressed in camera frame
    # T_imu_cam maps camera coords -> IMU coords: p_imu = R_ci^T p_cam + R_ci^T r_c
    T_ic_mat = np.eye(4)
    T_ic_mat[:3, :3] = R_ci.T
    T_ic_mat[:3, 3] = R_ci.T @ r_c
    T_ic = gtsam_pose(T_ic_mat)
    T_ci_mat = np.linalg.inv(T_ic_mat)
    Ad_ci = T_ic.inverse().AdjointMap()
    Ad_ic = T_ic.AdjointMap()

    verify_custom_jacobians(T_ic)

    # 2. Align initial camera poses into exact metric gravity-aligned world frame (+Z = UP)
    P_cam, R_gw = align_to_gravity_world(P_init, cal["up_w_init"], cal["s"])
    P_imu = np.array([P_cam[k] @ T_ci_mat for k in range(N)])
    cs_imu = CubicSpline(ts_rel, P_imu[:, :3, 3], axis=0)
    V_imu = cs_imu(ts_rel, 1)
    V_imu[cal["still_f"]] = 0.0

    # Calibrate car NHC in metric camera frame
    nhc_cal = None
    if args.nhc_sigma > 0:
        sam = []
        for s_ in (3, 5, 8):
            for i in range(N - s_):
                Tij = np.linalg.inv(P_cam[i]) @ P_cam[i + s_]
                if np.linalg.norm(Tij[:3, 3]) > 0.1 * s_ * cal["s"]:
                    sam.append((Tij[:3, :3], Tij[:3, 3]))
        nhc_cal, rob = nhc.calibrate(sam, with_height=False)
        print(f"NHC metric calib: yaw={np.degrees(nhc_cal[0]):+.2f} deg, pitch={np.degrees(nhc_cal[1]):+.2f} deg, L={nhc_cal[2]:.3f} m (scatter {rob[0]:.2f}/{rob[1]:.2f} deg)", flush=True)

    # Preintegrate 100 Hz IMU once with calibrated MakeSharedU(g_mag)
    pim_params = gtsam.PreintegrationCombinedParams.MakeSharedU(cal["g_mag"])
    pim_params.setAccelerometerCovariance(np.eye(3) * (args.acc_sigma ** 2))
    pim_params.setGyroscopeCovariance(np.eye(3) * (args.gyro_sigma ** 2))
    pim_params.setIntegrationCovariance(np.eye(3) * 1e-6)
    pim_params.setBiasAccCovariance(np.eye(3) * (args.acc_rw ** 2))
    pim_params.setBiasOmegaCovariance(np.eye(3) * (args.gyro_rw ** 2))
    pim_params.setBiasAccOmegaInit(np.eye(6) * 1e-5)
    bias0 = gtsam.imuBias.ConstantBias(cal["ba0"], cal["wb0"])
    pims = build_imu_preintegrations(ts_rel, t_i, w_i, a_i, imu_dt_s, pim_params, bias0)

    # Report initial IMU preintegration consistency before optimization!
    imu_pe, imu_ve, imu_re = [], [], []
    for k in range(N - 1):
        ns0 = gtsam.NavState(gtsam_pose(P_imu[k]), V_imu[k])
        ns1_pred = pims[k].predict(ns0, bias0)
        imu_pe.append(np.linalg.norm(ns1_pred.pose().translation() - P_imu[k + 1, :3, 3]))
        imu_ve.append(np.linalg.norm(ns1_pred.velocity() - V_imu[k + 1]))
        dR = ns1_pred.pose().rotation().matrix().T @ P_imu[k + 1, :3, :3]
        imu_re.append(np.degrees(np.linalg.norm(R.from_matrix(dR).as_rotvec())))
    print(f"IMU preint BEFORE opt: 1-step pos med={np.median(imu_pe)*100:.2f} cm, vel med={np.median(imu_ve)*100:.2f} cm/s, rot med={np.median(imu_re):.3f} deg", flush=True)

    # Optional VO & VO-loop direction measurements
    VO_meas = {}
    K_raw, _ = load_camera_info(f"{data}/camera_info.json")
    if args.vo_cache and os.path.exists(args.vo_cache):
        vo_m = pickle.load(open(args.vo_cache, "rb"))
        VO_o, vst = vodir.debias(vodir.measure(K_raw, P_cam, vo_m), P_cam)
        VO_meas.update(VO_o)
        print(f"VO odom loaded: {len(VO_o)} pairs (scatter {vst['resid_lat_deg']:.2f}/{vst['resid_vert_deg']:.2f} deg)", flush=True)
        if args.vol_cache and os.path.exists(args.vol_cache):
            vol_m = pickle.load(open(args.vol_cache, "rb"))
            VO_l, lst = vodir.debias(vodir.measure(K_raw, P_cam, vol_m, min_dist=1.0), P_cam)
            VO_meas.update(VO_l)
            print(f"VO loops loaded (debiased): {len(VO_l)} pairs (scatter {lst['resid_lat_deg']:.2f}/{lst['resid_vert_deg']:.2f} deg)", flush=True)

    # Optional cross-lap road-normal height pairs
    rn_data = None
    if args.rn_pairs and os.path.exists(args.rn_pairs):
        z_rn = np.load(args.rn_pairs)
        sel = np.arange(0, len(z_rn["i"]), max(1, args.rn_stride))
        rn_data = (z_rn["i"][sel], z_rn["j"][sel], (R_gw @ z_rn["d"][sel].T).T, z_rn["target"][sel] * cal["s"], z_rn["sigma"][sel] * cal["s"])
        print(f"Road-normal pairs loaded: {len(sel)} constraints", flush=True)

    # Optional Hybrid Anisotropic 6x6 PnP BetweenFactors (transformed exactly from camera -> IMU frame!)
    pnp_factors_imu = []
    if args.pnp_cov > 0 and args.pnp_pairs and os.path.exists(args.pnp_pairs):
        raw_pnp = pickle.load(open(args.pnp_pairs, "rb"))
        k_sc = np.load(args.pnp_scales) if (args.pnp_scales and os.path.exists(args.pnp_scales)) else np.ones(N)
        ss_, sd_ = (float(x) for x in args.pnp_scale_sig.split(","))
        fr_, ft_ = (float(x) for x in args.pnp_floor.split(","))
        s_m = cal["s"]
        for (a, b), r in raw_pnp.items():
            if r is None or a >= N or b >= N or min(r[4], r[6]) < args.pnp_min_inl:
                continue
            if len(r) <= 8 or r[8] is None:
                continue
            Rab, ta, n1, tb, n2 = r[2], r[3], r[4], r[5], r[6]
            T_cam = np.eye(4)
            T_cam[:3, :3] = Rab
            T_cam[:3, 3] = s_m * (n1 * k_sc[a] * ta + n2 * k_sc[b] * tb) / (n1 + n2)
            q = float(np.clip(np.sqrt(150.0 / min(n1, n2)), 0.7, 2.5))
            C_ = r[8].astype(np.float64).copy()
            sc_a = s_m * float(k_sc[a])
            S_ = np.diag([1.0, 1.0, 1.0, sc_a, sc_a, sc_a])
            C_ = S_ @ C_ @ S_
            tb_ = T_cam[:3, :3].T @ T_cam[:3, 3]
            nt_ = float(np.linalg.norm(tb_))
            if nt_ > 1e-6:
                u_ = tb_ / nt_
                C_[3:, 3:] += (ss_ * nt_) ** 2 * np.outer(u_, u_) + (sd_ * nt_) ** 2 * (np.eye(3) - np.outer(u_, u_))
            C_ *= (args.pnp_cov ** 2)
            C_[:3, :3] = np.eye(3) * (0.01 * q) ** 2
            C_[:3, 3:] = 0.0
            C_[3:, :3] = 0.0
            C_ += np.diag([np.radians(fr_) ** 2] * 3 + [(s_m * ft_) ** 2] * 3)
            M_imu = T_ic.compose(gtsam_pose(T_cam)).compose(T_ic.inverse())
            C_imu = Ad_ic @ C_ @ Ad_ic.T
            C_imu = 0.5 * (C_imu + C_imu.T)
            pnp_factors_imu.append((a, b, M_imu, C_imu))
        print(f"Hybrid anisotropic PnP-cov ({args.pnp_cov:.1f}x) loaded: {len(pnp_factors_imu)} C++ BetweenFactors", flush=True)

    # Optional visual VP gravity (vertical building edges from gravity.npz)
    vp_grav_data = None
    if args.vp_grav_sigma > 0 and os.path.exists(f"{args.target}/gravity.npz"):
        gz = np.load(f"{args.target}/gravity.npz")
        up_vp = gz["up"]
        ok_vp = (gz["n_lines"] >= 8) & np.isfinite(up_vp).all(1)
        vp_grav_data = (np.where(ok_vp)[0], up_vp[ok_vp])
        print(f"Visual VP gravity loaded: {int(ok_vp.sum())}/{N} frames (sigma={args.vp_grav_sigma:.2f} deg)", flush=True)

    # Optional BEV red-curb / zebra / road-marking cross-lap loops (computed once on initial metric poses)
    bev_factors_imu = []
    if args.bev_marking_sigma > 0:
        kept_bev, (F_b, L_b, U_b) = compute_bev_marking_loops(
            args.target, stamps, P_cam, args.cam, nhc_cal, h0_m=1.78, every=args.bev_every
        )
        sig_lat = args.bev_marking_sigma
        sig_fwd = 2.0 * sig_lat
        sig_yaw = np.radians(0.5)
        for (a, b), (dfx, dfy, dyaw, _, _) in kept_bev.items():
            Tab = np.linalg.inv(P_cam[a]) @ P_cam[b]
            Tm = Tab.copy()
            Rz = cv2.Rodrigues(U_b * dyaw)[0]
            Tm[:3, :3] = Rz @ Tab[:3, :3]
            Tm[:3, 3] = Tab[:3, 3] + dfx * F_b + dfy * L_b
            Fb, Lb, Ub = Tm[:3, :3].T @ F_b, Tm[:3, :3].T @ L_b, Tm[:3, :3].T @ U_b
            info_c = np.zeros((6, 6))
            info_c[:3, :3] = np.eye(3) / np.radians(5.0) ** 2 + np.outer(Ub, Ub) * (1.0 / sig_yaw ** 2 - 1.0 / np.radians(5.0) ** 2)
            info_c[3:, 3:] = np.eye(3) / 10.0 ** 2 + np.outer(Fb, Fb) / sig_fwd ** 2 + np.outer(Lb, Lb) / sig_lat ** 2
            M_imu = T_ic.compose(gtsam_pose(Tm)).compose(T_ic.inverse())
            info_imu = Ad_ci.T @ info_c @ Ad_ci
            info_imu = 0.5 * (info_imu + info_imu.T)
            bev_factors_imu.append((a, b, M_imu, info_imu))

    # Multi-round Joint Full-Information Optimization!
    raw_m = None
    biases = [bias0 for _ in range(N)]
    P_cam_orig_metric = P_cam.copy()
    up_c = nhc.axes(*(nhc_cal[:2] if nhc_cal is not None else (0.0, -0.03)))[2]

    for rnd in range(args.rounds):
        t_r = time.time()
        print(f"\n=== Round {rnd} (ba_mode={args.ba_mode}, rs_6dof={args.rs_6dof and rnd > 0}) ===", flush=True)
        V_cam_w = None
        if args.rs_6dof and rnd > 0:
            V_cam_w = CubicSpline(ts_rel, P_cam[:, :3, 3], axis=0)(ts_rel, 1)

        K_pin, points = None, []
        if args.ba_mode != "none":
            K_pin, points, raw_m = prepare_visual_tracks(
                args.target, stamps, P_cam, ts_rel, t_i, w_i, cal["wb0"], R_ci, imu_dt_s,
                args.cam, args.mask_v_max, args.rs_readout_ms,
                mask_obj=128, keep_static_obj=True, min_parallax=args.min_parallax,
                max_reproj=2.5 if rnd == 0 else 2.0, max_obs=args.max_obs,
                V_cam_w=V_cam_w, prev_raw_m=raw_m,
            )

        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        for i in range(N):
            vals.insert(X_KEY(i), gtsam_pose(P_imu[i]))
            if not args.no_imu:
                vals.insert(V_KEY(i), V_imu[i])
                vals.insert(B_KEY(i), biases[i])

        # Gauge prior on X(0): fix position + yaw ONLY (keep roll/pitch free for IMU gravity!)
        if not args.no_imu:
            # Tiny prior on pos (1e-4 m) and yaw (1e-4 rad), loose roll/pitch (0.5 rad) so gravity sets tilt
            graph.add(gtsam.PriorFactorPose3(
                X_KEY(0), gtsam_pose(P_imu[0]),
                gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 1e-4, 1e-4, 1e-4, 1e-4]))
            ))
            graph.add(gtsam.PriorFactorConstantBias(
                B_KEY(0), bias0,
                gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.003, 0.003, 0.003]))
            ))
            for k in range(N - 1):
                graph.add(gtsam.CombinedImuFactor(X_KEY(k), V_KEY(k), X_KEY(k + 1), V_KEY(k + 1), B_KEY(k), B_KEY(k + 1), pims[k]))
            # ZUPT on stationary frames!
            zupt_noise = gtsam.noiseModel.Isotropic.Sigma(3, args.zupt_sigma)
            for fi in np.where(cal["still_f"])[0]:
                graph.add(gtsam.PriorFactorVector(V_KEY(fi), np.zeros(3), zupt_noise))
        else:
            graph.add(gtsam.PriorFactorPose3(X_KEY(0), gtsam_pose(P_imu[0]), gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)))

        # Car NHC kinematic factors
        if nhc_cal is not None:
            sa = np.radians(args.nhc_sigma)
            for i in range(N - 1):
                d_step = np.linalg.norm(P_cam[i + 1, :3, 3] - P_cam[i, :3, 3])
                graph.add(nhc_on_imu_factor(X_KEY(i), X_KEY(i + 1), T_ic_mat, Ad_ci, nhc_cal, [0.001 + sa * d_step, 0.002 + 2.0 * sa * d_step]))

        # VO & VO-loop epipolar direction factors
        if VO_meas:
            s0 = np.radians([args.vo_sigma, args.vo_sigma_up])
            for (a, b), (m_dir, n_inl) in VO_meas.items():
                graph.add(vodir_on_imu_factor(X_KEY(a), X_KEY(b), T_ic_mat, Ad_ci, m_dir, s0 * np.clip(np.sqrt(500.0 / n_inl), 0.7, 2.0)))

        # Visual Bundle Adjustment factors (C++ GenericProjectionFactorCal3_S2 or SmartProjectionPose3Factor)
        if args.ba_mode == "generic" and points:
            cal_s2 = gtsam.Cal3_S2(K_pin[0, 0], K_pin[1, 1], 0.0, K_pin[0, 2], K_pin[1, 2])
            vis_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(args.ba_huber),
                gtsam.noiseModel.Isotropic.Sigma(2, args.ba_sigma),
            )
            for j, (Xpt, fr, xy, _, _) in enumerate(points):
                vals.insert(L_KEY(j), gtsam.Point3(*Xpt))
                for f, uv in zip(fr, xy):
                    graph.add(gtsam.GenericProjectionFactorCal3_S2(uv, vis_noise, X_KEY(f), L_KEY(j), cal_s2, T_ic))

            # Optional Manhattan vertical building-wall coplanarity factors (Planar BA)
            if args.wall_plane_sigma > 0:
                walls = detect_manhattan_walls(points, P_cam)
                n_wf = 0
                for wi, (nw, dw, l_idxs, span, rms_w) in enumerate(walls):
                    for lj in l_idxs:
                        graph.add(wall_plane_factor(L_KEY(lj), nw, dw, args.wall_plane_sigma))
                        n_wf += 1
                print(f"  Manhattan building walls: {len(walls)} planar facades, {n_wf} landmark constraints "
                      f"(spans={[round(w[3],1) for w in walls]} m, init_rms={[round(w[4]*100,2) for w in walls]} cm)", flush=True)

            # Optional constant camera height above triangulated road landmarks
            if args.height_sigma > 0:
                pts4 = [(p[0], p[1], p[2], p[3]) for p in points]
                hh, _ = ba.camera_heights(P_cam, pts4, up_c, radius=4.0 * cal["s"])
                if np.isfinite(hh).sum() > 50:
                    h0 = float(np.nanmedian(hh))
                    hp = ba.height_pairs(P_cam, pts4, up_c, h0, fwd_range=2.5 * cal["s"], lat_range=1.5 * cal["s"], tol=0.35 * cal["s"], per_frame=args.height_per_frame)
                    for fi, lj in hp:
                        graph.add(height_on_imu_factor(X_KEY(fi), L_KEY(lj), T_ic_mat, Ad_ci, up_c, h0, args.height_sigma))
                    print(f"  road-height factors: {len(hp)} (h0={h0:.3f} m, std={1.4826*np.nanmedian(np.abs(hh-h0))*100:.2f} cm)", flush=True)

        elif args.ba_mode == "smart" and points:
            cal_s2 = gtsam.Cal3_S2(K_pin[0, 0], K_pin[1, 1], 0.0, K_pin[0, 2], K_pin[1, 2])
            sp_params = gtsam.SmartProjectionParams()
            sp_params.setLinearizationMode(gtsam.LinearizationMode.HESSIAN)
            sp_params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
            sp_params.setDynamicOutlierRejectionThreshold(3.0)
            vis_noise = gtsam.noiseModel.Isotropic.Sigma(2, args.ba_sigma)
            for j, (_, fr, xy, _, _) in enumerate(points):
                sf = gtsam.SmartProjectionPose3Factor(vis_noise, cal_s2, T_ic, sp_params)
                for f, uv in zip(fr, xy):
                    sf.add(uv, X_KEY(f))
                graph.add(sf)

        # Optional cross-lap road-normal height constraints
        if rn_data is not None:
            ri, rj, rd, rt, rs = rn_data
            for ii, jj, dd, tt, ss in zip(ri, rj, rd, rt, rs):
                graph.add(road_normal_pair_on_imu_factor(X_KEY(ii), X_KEY(jj), T_ic_mat, Ad_ci, dd, tt, ss))

        # Optional hybrid anisotropic 6x6 PnP BetweenFactors
        if pnp_factors_imu:
            for a, b, M_imu, C_imu in pnp_factors_imu:
                nz = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                    gtsam.noiseModel.Gaussian.Covariance(C_imu),
                )
                graph.add(gtsam.BetweenFactorPose3(X_KEY(a), X_KEY(b), M_imu, nz))

        # Optional cross-lap BEV red-curb / zebra / road-marking BetweenFactors
        if bev_factors_imu:
            for a, b, M_imu, info_imu in bev_factors_imu:
                nz = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                    gtsam.noiseModel.Gaussian.Information(info_imu),
                )
                graph.add(gtsam.BetweenFactorPose3(X_KEY(a), X_KEY(b), M_imu, nz))

        # Optional visual vanishing-point gravity factors (building vertical edges)
        if vp_grav_data is not None:
            vp_sig = np.radians(args.vp_grav_sigma)
            for fi, uvp in zip(*vp_grav_data):
                graph.add(vp_gravity_on_imu_factor(X_KEY(fi), T_ic_mat, Ad_ci, uvp, vp_sig))

        # Optional weak camera position prior
        if args.pose_prior > 0:
            pp_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e3, 1e3, 1e3, args.pose_prior, args.pose_prior, args.pose_prior]))
            for i in range(N):
                T_imu_p = P_cam_orig_metric[i] @ T_ci_mat
                graph.add(gtsam.PriorFactorPose3(X_KEY(i), gtsam_pose(T_imu_p), pp_noise))

        e_before = graph.error(vals)
        lm_p = gtsam.LevenbergMarquardtParams()
        lm_p.setMaxIterations(args.max_iter)
        lm_p.setRelativeErrorTol(1e-5)
        lm_p.setAbsoluteErrorTol(1e-5)
        lm_p.setVerbosityLM("SILENT")
        opt = gtsam.LevenbergMarquardtOptimizer(graph, vals, lm_p)
        res_vals = opt.optimize()
        e_after = graph.error(res_vals)

        P_imu_new = np.array([res_vals.atPose3(X_KEY(i)).matrix() for i in range(N)])
        P_cam_new = np.array([P_imu_new[i] @ T_ic_mat for i in range(N)])
        if not args.no_imu:
            V_imu = np.array([res_vals.atVector(V_KEY(i)) for i in range(N)])
            biases = [res_vals.atConstantBias(B_KEY(i)) for i in range(N)]

        moved, _ = sim3_ate(P_cam_new, P_cam)
        to_init, s_now = sim3_ate(P_cam_new, P_cam_orig_metric)
        P_imu, P_cam = P_imu_new, P_cam_new

        # Diagnostics: IMU preintegration + reprojection error + bias drift!
        msg = f"  Round {rnd} done in {time.time()-t_r:.1f}s ({opt.iterations()} iters): cost {e_before:.1f} -> {e_after:.1f}, moved={moved*100:.2f} cm, to_init={to_init*100:.2f} cm (s={s_now:.4f})"
        print(msg, flush=True)
        if not args.no_imu:
            pe, ve, re_ = [], [], []
            for k in range(N - 1):
                ns0 = gtsam.NavState(gtsam_pose(P_imu[k]), V_imu[k])
                ns1 = pims[k].predict(ns0, biases[k])
                pe.append(np.linalg.norm(ns1.pose().translation() - P_imu[k + 1, :3, 3]))
                ve.append(np.linalg.norm(ns1.velocity() - V_imu[k + 1]))
                dR = ns1.pose().rotation().matrix().T @ P_imu[k + 1, :3, :3]
                re_.append(np.degrees(np.linalg.norm(R.from_matrix(dR).as_rotvec())))
            ba_arr = np.array([b_.accelerometer() for b_ in biases])
            bg_arr = np.array([b_.gyroscope() for b_ in biases])
            print(f"  IMU AFTER r{rnd}: pos med={np.median(pe)*100:.2f} cm, vel med={np.median(ve)*100:.2f} cm/s, rot med={np.median(re_):.3f} deg | "
                  f"ba mean={np.round(ba_arr.mean(0), 4)} std={np.round(ba_arr.std(0), 4)}, bg std={np.round(bg_arr.std(0), 5)}", flush=True)
        if args.ba_mode == "generic" and points:
            pts_opt = [(res_vals.atPoint3(L_KEY(j)), p[1], p[2], p[3]) for j, p in enumerate(points)]
            e_rep = ba.reprojection_stats(K_pin, P_cam, pts_opt)[0]
            hh_a, _ = ba.camera_heights(P_cam, pts_opt, up_c, radius=4.0 * cal["s"])
            h_std = 1.4826 * np.nanmedian(np.abs(hh_a - np.nanmedian(hh_a))) * 100
            print(f"  Visual BA AFTER r{rnd}: reproj med={np.median(e_rep):.3f} px, rms={np.sqrt(np.mean(e_rep**2)):.3f} px | cam_height={np.nanmedian(hh_a):.3f}±{h_std:.2f} cm", flush=True)

        # Save per-round output (Sim(3)-aligned to P_init so units match rest of pipeline cleanly!)
        s_a, R_a, t_a = umeyama(P_cam[:, :3, 3], P_init[:, :3, 3])
        P_save = np.zeros_like(P_cam)
        for i in range(N):
            P_save[i, :3, :3] = R_a @ P_cam[i, :3, :3]
            P_save[i, :3, 3] = s_a * (R_a @ P_cam[i, :3, 3]) + t_a
            P_save[i, 3, 3] = 1.0
        base = args.out[:-4] if args.out.endswith(".csv") else args.out
        save_traj(f"{res_dir}/{base}_r{rnd}.csv", stamps, P_save)

    out_path = f"{res_dir}/{args.out if args.out.endswith('.csv') else args.out + '.csv'}"
    save_traj(out_path, stamps, P_save)
    print(f"\nSaved {out_path} ({N} poses, total {time.time()-t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
