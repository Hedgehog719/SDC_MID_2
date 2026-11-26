"""Image-only bundle adjustment with a self-calibrated lens (Ceres through pycolmap), started from a trajectory.

Why: the GTSAM bundle adjustment of slam_v2.py (--ba) used the given pinhole K.  The lens has barrel distortion
(k1 about -0.03, NOTES.md), i.e. up to ~14 px at the image corners, which a pinhole BA can only absorb in the
structure and the translations.  Here the reprojection error is minimised over poses, points AND the camera
(fx, fy, cx, cy, k1, k2, p1, p2), with nothing else in the problem (no clouds, no gyro, no car model): an
independent reference for the shape of the trajectory, and a calibration of the camera from the images alone.

  1. SIFT keypoints of the feature cache (all keypoints, no cloud depth needed)
  2. pairs: frame steps 1,2,3,5,8,13,21 + revisits of the start trajectory (other pass within --loop-radius)
  3. mutual-NN matches -> 5-point RANSAC inliers -> tracks (union-find, tracks with two keypoints of one image dropped)
  4. rounds of: undistort with the current camera, triangulate, filter (reprojection / parallax), bundle adjustment

usage (from image_slam_starter, in the env that has pycolmap + torch):
  ../.venv-loger/bin/python process/sfm_ba.py Track1 Track1/result/v10_vo_shared_it5.csv --out sfm_v10
writes Track1/result/<out>.csv (poses, Sim(3)-aligned to the start trajectory), <out>.json (camera, statistics)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpu_match import GpuMatcher, HAS_CUDA

ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("init", help="start trajectory (submission csv)")
ap.add_argument("--out", default="sfm")
ap.add_argument("--feat", default=None, help="feature cache directory (default: the only <target>/result/feat_* directory)")
ap.add_argument("--steps", default="1,2,3,5,8,13,21")
ap.add_argument("--loop-radius", type=float, default=3.0)
ap.add_argument("--loop-per-frame", type=int, default=6)
ap.add_argument("--no-loops", action="store_true")
ap.add_argument("--ratio", type=float, default=0.75)
ap.add_argument("--e-thr", type=float, default=2.0, help="5-point RANSAC threshold of the pair verification [px]")
ap.add_argument("--rounds", default="8,4,2,2", help="reprojection threshold [px] of the point filter before every bundle adjustment")
ap.add_argument("--min-parallax", type=float, default=1.5)
ap.add_argument("--min-track", type=int, default=3)
ap.add_argument("--max-obs", type=int, default=0, help="keep only the longest tracks up to this many observations (0 = all)")
ap.add_argument("--k1", type=float, default=0.0, help="start value of k1")
ap.add_argument("--fix-intrinsics", action="store_true", help="keep the camera (with --model PINHOLE: K as given, what slam_v2.py --ba does)")
ap.add_argument("--camera", default=None, help="start parameters of the camera model (comma separated, colmap order), e.g. from <out>.json")
ap.add_argument("--prior-aniso", default="",
                help="along,lateral,vertical sigmas [units] of the position prior in each frame's road axes (direction of travel of the "
                     "start trajectory, camera up): e.g. 5,0.15,0.15 keeps the two passes of a road together sideways / in height "
                     "and leaves the BA free along the road (long-straight scale); implies the pose-prior BA")
ap.add_argument("--pose-prior", type=float, default=0.0,
                help="position prior on every camera at the start trajectory with this sigma [units] (pose-prior BA): the BA refines the "
                     "local shape of every pass while the start trajectory keeps what images cannot link (opposite passes, start = end)")
ap.add_argument("--anchors", default="",
                help="frames whose poses stay constant (fixes the gauge: with --fix-rot the translation and scale are otherwise free; "
                     "Track02 diverged without it), e.g. 0,2100")
ap.add_argument("--fix-poses", action="store_true", help="only triangulate / refine the points for the start trajectory (e.g. to measure its camera height)")
ap.add_argument("--no-pp", action="store_true", help="do not refine the principal point")
ap.add_argument("--fix-rot", action="store_true",
                help="keep the rotations of the start trajectory (gyro-constrained): only positions, points and the camera are "
                     "optimised, so the camera is calibrated against the gyro rotations")
ap.add_argument("--fix-f", action="store_true", help="do not refine the focal length (distortion / principal point only)")
ap.add_argument("--fscale", type=float, default=1.0, help="start focal length = given * this")
ap.add_argument("--square", action="store_true",
                help="rescale the keypoints' y about cy by fx/fy first, so that one focal length describes the given camera "
                     "(for RADIAL / SIMPLE_RADIAL: with a car on a flat road fy is not observable, a vertical stretch of the "
                     "world compensates it)")
ap.add_argument("--model", default="OPENCV", choices=["OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "RADIAL", "SIMPLE_RADIAL", "PINHOLE"])
ap.add_argument("--loss-scale", type=float, default=1.0, help="Cauchy loss scale [px]")
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--max-iter", type=int, default=300, help="Ceres iterations per bundle adjustment (300 stopped every Track02 round with NO_CONVERGENCE)")
ap.add_argument("--solver", default="auto", choices=["auto", "sparse"],
                help="auto: pycolmap picks ITERATIVE_SCHUR + JACOBI above 1000 images (Track02 4206: never converged); "
                     "sparse: exact SPARSE_SCHUR (SuiteSparse) for any number of images")
ap.add_argument("--split-conflicts", action="store_true",
                help="a track that has two keypoints in one image keeps its other observations (only the conflicting images are "
                     "dropped) instead of being dropped as a whole")
ap.add_argument("--repeat-last", type=int, default=0,
                help="after the listed rounds, repeat the last threshold up to this many times until the BA converges: Ceres reports "
                     "CONVERGENCE and the cameras move less than --tol (0 = off, the rounds as listed)")
ap.add_argument("--tol", type=float, default=0.002, help="... camera movement [units, rms] below which a repeated round counts as unchanged")
ap.add_argument("--verify-undistort", action="store_true",
                help="verify the pairs (5-point RANSAC) on keypoints undistorted with the start camera (--camera) instead of the raw pixels "
                     "with the given K: the lens moves mid-field points by ~5 px, beyond the 2 px threshold")
ap.add_argument("--dump-tracks", action="store_true",
                help="also save, per 3D point, the first / last frame that observes it and the mean P(road user) of the segmentation at "
                     "its observations (<out>.pts.npz: fmin, fmax, obj) -> per-pass point clouds of parked cars etc.")
ap.add_argument("--links-merge", type=float, default=0.0,
                help="instead of adding --links as separate 2-view tracks: snap both pixels of every link to the nearest existing keypoint "
                     "of its image within this radius [px] and add the pair as a match edge, so that the link joins the two passes' "
                     "multi-view tracks (0 = off)")
ap.add_argument("--links", default=None,
                help="npz of process/road_links.py: road-marking points seen by both passes of a road (opposite directions), "
                     "added as 2-view tracks (the revisit pairs above only join frames facing the same way)")
ap.add_argument("--cross-pass", default=None,
                help="npz of process/pass_consensus.py --dump-pairs: constraints between camera centres of two passes over the same "
                     "road, ((c_j - c_i) . dir - target) / sigma (equal camera height, yellow centre line), added to every bundle "
                     "adjustment next to the reprojection errors (needs --pose-prior / --prior-aniso or --anchors for the gauge)")
ap.add_argument("--cross-loss", type=float, default=2.0, help="Huber threshold of the cross-pass residuals [sigmas]")
ap.add_argument("--cross-stride", type=int, default=1, help="use every k-th cross-pass constraint")
ap.add_argument("--mask-obj", type=int, default=0, help="drop keypoints where seg.npz['obj'] >= this threshold (0 = off, e.g. 128)")
ap.add_argument("--keep-static-obj", action="store_true",
                help="with --mask-obj: keep rigid stationary objects (parked cars) that triangulate cleanly (<1.2 px across >=6 frames) while dropping moving vehicles/pedestrians")
ap.add_argument("--mask-v-max", type=float, default=0.0, help="drop keypoints with pixel row v > this value (e.g. 850 to mask car hood; 0 = off)")
ap.add_argument("--rs-readout-ms", type=float, default=0.0, help="top-to-bottom CMOS rolling-shutter readout time [ms] to deskew keypoints using camera angular velocity (0 = off)")
ap.add_argument("--rs-6dof", action="store_true", help="with --rs-readout-ms: also deskew 3D translational rolling shutter (v*dt_v/Z) per BA round using triangulated depths and camera velocity")
ap.add_argument("--imu-gyro", type=float, default=0.0, help="tightly-coupled 100 Hz IMU gyro relative-rotation sigma [deg] for step 1 (scaled by sqrt(step) for steps 1,5,20; 0 = off)")
ap.add_argument("--imu-gravity", type=float, default=0.0, help="tightly-coupled IMU accelerometer gravity-tilt prior sigma [deg] after subtracting centripetal/linear accel (0 = off)")
ap.add_argument("--imu-dt-ms", type=float, default=-19.0, help="camera-to-IMU exposure time offset [ms] (default -19.0 ms)")
ap.add_argument("--imu-loss", type=float, default=3.0, help="Huber threshold [sigmas] for IMU gyro & gravity factors")
ap.add_argument("--road-height", type=float, default=0.0, help="joint BA triangulated-road camera-height ruler sigma [units] (e.g. 0.04; 0 = off)")
ap.add_argument("--road-height-per-frame", type=int, default=6, help="max triangulated ego-lane road 3D points constrained per frame for --road-height")
ap.add_argument("--road-height-loss", type=float, default=2.5, help="Cauchy loss scale [sigmas] for --road-height")
ap.add_argument("--nhc", type=float, default=0.0, help="vehicle non-holonomic constraint (midpoint rear-axle zero lateral/vertical slip) sigma [units] (e.g. 0.015; 0 = off)")
ap.add_argument("--nhc-loss", type=float, default=2.0, help="Huber loss scale [sigmas] for --nhc")
ap.add_argument("--step-prior", default=None, help="npz from process/road_homography.py: dense asphalt homography step-length scale factors (||c_j - c_i|| = h0 * s_obs)")
ap.add_argument("--step-sigma", type=float, default=0.025, help="relative step-length sigma for --step-prior (e.g. 0.025 = 2.5%%; auto-scaled by sqrt(step) for overlapping pairs)")
ap.add_argument("--step-loss", type=float, default=2.0, help="Huber loss scale [sigmas] for --step-prior")
ap.add_argument("--step-kappa", type=float, default=0.0, help="fixed acceleration scale-bias correction for --step-prior: d_tgt = h0 * (1 + kappa * a_fwd) * s_obs [s^2/m] (e.g. 0.0053; 0 = off)")
args = ap.parse_args()

import pycolmap
USE_PYCERES = bool(args.cross_pass or args.imu_gyro > 0 or args.imu_gravity > 0 or args.road_height > 0 or args.nhc > 0 or args.step_prior)
if USE_PYCERES:
    import pyceres
    from numpy.lib.stride_tricks import as_strided

    def pose_block(rec_, image_id):
        """the 7-double pose parameter block (quaternion x y z w, translation) of an image's frame, as a writable view"""
        q_ = rec_.frame(rec_.image(image_id).frame_id).rig_from_world.rotation.quat
        return as_strided(q_, shape=(7,), strides=(q_.strides[0],), writeable=True)

    def _qmul(a, b):
        ax, ay, az, aw = a[0], a[1], a[2], a[3]
        bx, by, bz, bw = b[0], b[1], b[2], b[3]
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    def _qconj(q):
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def _qL(a):
        ax, ay, az, aw = a
        return np.array([[aw, -az, ay, ax], [az, aw, -ax, ay], [-ay, ax, aw, az], [-ax, -ay, -az, aw]])

    def _qR(b):
        bx, by, bz, bw = b
        return np.array([[bw, bz, -by, bx], [-bz, bw, bx, by], [by, -bx, bw, bz], [-bx, -by, -bz, bw]])

    _QCONJ_DIAG = np.diag([-1.0, -1.0, -1.0, 1.0])

    class GyroPairCost(pyceres.CostFunction):
        """Tightly-coupled relative SO(3) rotation factor: R(q_meas)^T R(q_i) R(q_j)^T = I (analytic Jacobian)."""
        def __init__(self, q_meas_inv, sigma_rad):
            super().__init__()
            self.set_num_residuals(3)
            self.set_parameter_block_sizes([7, 7])
            self.q_minv = np.asarray(q_meas_inv, dtype=np.float64)
            self.L_minv = _qL(self.q_minv)
            self.scale = 2.0 / float(sigma_rad)

        def Evaluate(self, parameters, residuals, jacobians):
            qi = np.asarray(parameters[0][:4], dtype=np.float64)
            qj = np.asarray(parameters[1][:4], dtype=np.float64)
            qj_c = _qconj(qj)
            qe = _qmul(self.q_minv, _qmul(qi, qj_c))
            sgn = self.scale if qe[3] >= 0.0 else -self.scale
            residuals[:] = sgn * qe[:3]
            if jacobians is not None:
                if jacobians[0] is not None:
                    J0 = np.zeros((3, 7))
                    J0[:, :4] = sgn * (self.L_minv @ _qR(qj_c))[:3, :]
                    jacobians[0][:] = J0.ravel()
                if jacobians[1] is not None:
                    J1 = np.zeros((3, 7))
                    J1[:, :4] = sgn * (_qL(_qmul(self.q_minv, qi)) @ _QCONJ_DIAG)[:3, :]
                    jacobians[1][:] = J1.ravel()
            return True

    class GravityCost(pyceres.CostFunction):
        """Tightly-coupled gravity tilt factor: (R(q_i) g_world - g_cam_meas) / sigma_rad (analytic Jacobian)."""
        def __init__(self, g_world, g_cam_meas, sigma_rad):
            super().__init__()
            self.set_num_residuals(3)
            self.set_parameter_block_sizes([7])
            self.u = np.asarray(g_world, dtype=np.float64)
            self.g_cam = np.asarray(g_cam_meas, dtype=np.float64)
            self.inv_s = 1.0 / float(sigma_rad)
            u = self.u
            self.ux = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])

        def Evaluate(self, parameters, residuals, jacobians):
            q = np.asarray(parameters[0][:4], dtype=np.float64)
            v, w = q[:3], q[3]
            u = self.u
            vu = float(v @ u)
            y = (w * w - float(v @ v)) * u + 2.0 * v * vu + 2.0 * w * np.cross(v, u)
            residuals[:] = (y - self.g_cam) * self.inv_s
            if jacobians is not None and jacobians[0] is not None:
                J = np.zeros((3, 7))
                dv = -2.0 * np.outer(u, v) + 2.0 * vu * np.eye(3) + 2.0 * np.outer(v, u) - 2.0 * w * self.ux
                dw = 2.0 * w * u + 2.0 * np.cross(v, u)
                J[:, :3] = dv * self.inv_s
                J[:, 3] = dw * self.inv_s
                jacobians[0][:] = J.ravel()
            return True

    def _rot_q(q, u):
        """Rotate 3-vector u by unit quaternion q=(vx,vy,vz,w): R(q) u, plus d(R(q)u)/dq (3x4) and R(q) (3x3)."""
        v, w = q[:3], q[3]
        vu = float(v @ u)
        vv = float(v @ v)
        y = (w * w - vv) * u + 2.0 * v * vu + 2.0 * w * np.cross(v, u)
        ux = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
        dv = -2.0 * np.outer(u, v) + 2.0 * vu * np.eye(3) + 2.0 * np.outer(v, u) - 2.0 * w * ux
        dw = 2.0 * w * u + 2.0 * np.cross(v, u)
        Jq = np.empty((3, 4))
        Jq[:, :3] = dv
        Jq[:, 3] = dw
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        Rm = (w * w - vv) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * vx
        return y, Jq, Rm

    def _rot_qt(q, u):
        """Rotate 3-vector u by R(q)^T: y = R(q)^T u, plus dy/dq (3x4) and R(q)^T (3x3)."""
        v, w = q[:3], q[3]
        vu = float(v @ u)
        vv = float(v @ v)
        y = (w * w - vv) * u + 2.0 * v * vu - 2.0 * w * np.cross(v, u)
        ux = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
        dv = -2.0 * np.outer(u, v) + 2.0 * vu * np.eye(3) + 2.0 * np.outer(v, u) + 2.0 * w * ux
        dw = 2.0 * w * u - 2.0 * np.cross(v, u)
        Jq = np.empty((3, 4))
        Jq[:, :3] = dv
        Jq[:, 3] = dw
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        Rt = (w * w - vv) * np.eye(3) + 2.0 * np.outer(v, v) - 2.0 * w * vx
        return y, Jq, Rt

    class RoadHeightCost(pyceres.CostFunction):
        """Joint 2-body BA road-height scale ruler: (n_c . (R(q_i) X_j + t_i) - h0) / sigma_h.
        Touches 1 pose [7] and 1 point3D [3] -> preserves exact SPARSE_SCHUR sparsity."""
        def __init__(self, n_c, h0, sigma_h):
            super().__init__()
            self.set_num_residuals(1)
            self.set_parameter_block_sizes([7, 3])
            self.n = np.asarray(n_c, dtype=np.float64)
            self.h0 = float(h0)
            self.inv_s = 1.0 / float(sigma_h)

        def Evaluate(self, parameters, residuals, jacobians):
            p = np.asarray(parameters[0], dtype=np.float64)
            X = np.asarray(parameters[1], dtype=np.float64)
            q, t = p[:4], p[4:]
            y, Jq, Rm = _rot_q(q, X)
            residuals[0] = (float(self.n @ (y + t)) - self.h0) * self.inv_s
            if jacobians is not None:
                if jacobians[0] is not None:
                    J0 = np.empty(7)
                    J0[:4] = (self.n @ Jq) * self.inv_s
                    J0[4:] = self.n * self.inv_s
                    jacobians[0][:] = J0
                if jacobians[1] is not None:
                    jacobians[1][:] = (self.n @ Rm) * self.inv_s
            return True

    class NHCPairCost(pyceres.CostFunction):
        """Midpoint vehicle non-holonomic constraint between consecutive poses (i, j):
        rear-axle world displacement dp_w = p_rear_j - p_rear_i, where p_rear_k = R(q_k)^T (b_rear - t_k).
        Projected onto exact midpoint axes m = 0.5*(R(q_i)^T e + R(q_j)^T e): EXACTLY zero on any circular arc."""
        def __init__(self, e_lat, e_up, b_rear, sigma_lat, sigma_up):
            super().__init__()
            self.set_num_residuals(2)
            self.set_parameter_block_sizes([7, 7])
            self.e_lat = np.asarray(e_lat, dtype=np.float64) / float(sigma_lat)
            self.e_up = np.asarray(e_up, dtype=np.float64) / float(sigma_up)
            self.b = np.asarray(b_rear, dtype=np.float64)

        def Evaluate(self, parameters, residuals, jacobians):
            pi = np.asarray(parameters[0], dtype=np.float64)
            pj = np.asarray(parameters[1], dtype=np.float64)
            qi, ti = pi[:4], pi[4:]
            qj, tj = pj[:4], pj[4:]
            pri, Jpi_q, Rit = _rot_qt(qi, self.b - ti)
            prj, Jpj_q, Rjt = _rot_qt(qj, self.b - tj)
            dp_w = prj - pri
            li, Jli_q, _ = _rot_qt(qi, self.e_lat)
            lj, Jlj_q, _ = _rot_qt(qj, self.e_lat)
            ui, Jui_q, _ = _rot_qt(qi, self.e_up)
            uj, Juj_q, _ = _rot_qt(qj, self.e_up)
            ml = 0.5 * (li + lj)
            mu = 0.5 * (ui + uj)
            residuals[0] = float(ml @ dp_w)
            residuals[1] = float(mu @ dp_w)
            if jacobians is not None:
                if jacobians[0] is not None:
                    J0 = np.empty((2, 7))
                    J0[0, :4] = 0.5 * (dp_w @ Jli_q) - (ml @ Jpi_q)
                    J0[0, 4:] = ml @ Rit
                    J0[1, :4] = 0.5 * (dp_w @ Jui_q) - (mu @ Jpi_q)
                    J0[1, 4:] = mu @ Rit
                    jacobians[0][:] = J0.ravel()
                if jacobians[1] is not None:
                    J1 = np.empty((2, 7))
                    J1[0, :4] = 0.5 * (dp_w @ Jlj_q) + (ml @ Jpj_q)
                    J1[0, 4:] = -(ml @ Rjt)
                    J1[1, :4] = 0.5 * (dp_w @ Juj_q) + (mu @ Jpj_q)
                    J1[1, 4:] = -(mu @ Rjt)
                    jacobians[1][:] = J1.ravel()
            return True

    def _fit_road_plane(xc_arr):
        """Robust IRLS road plane fit n_c . X_c = h in camera frame + quadratic curvature sag; returns (n_c, h, mad, sag)."""
        Ap = np.c_[xc_arr[:, 0], xc_arr[:, 2], -np.ones(len(xc_arr))]
        w = np.ones(len(xc_arr))
        for _ in range(8):
            p_, _, _, _ = np.linalg.lstsq(Ap * w[:, None], -xc_arr[:, 1] * w, rcond=None)
            w = 1.0 / np.maximum(np.abs(xc_arr[:, 1] + Ap @ p_), 0.02)
        n_raw = np.array([p_[0], 1.0, p_[1]], dtype=np.float64)
        n_norm = np.linalg.norm(n_raw)
        n_c = n_raw / n_norm
        h_ = float(p_[2] / n_norm)
        r_lin = xc_arr @ n_c - h_
        mad = float(np.median(np.abs(r_lin)) * 1.4826)
        zm = xc_arr[:, 2] - np.median(xc_arr[:, 2])
        Aq = np.c_[np.ones(len(zm)), zm, zm ** 2]
        cq, _, _, _ = np.linalg.lstsq(Aq * w[:, None], r_lin * w, rcond=None)
        sag = abs(float(cq[2])) * 4.0  # vertical sag [u] over +/-2u Z span
        return n_c, h_, mad, sag

    def add_road_height_factors(adj_, rec_, P_cur):
        """Per-frame ego-lane road-plane normal n_i (via cKDTree over same-pass triangulated road pts) + scale-free outlier filter."""
        from scipy.spatial.transform import Rotation as _RotRH
        seg_path = f"{args.target}/seg.npz"
        seg_road = np.load(seg_path)["road"] if os.path.exists(seg_path) and "road" in np.load(seg_path).files else None
        # IMPORTANT: read poses directly from rec_ because create_pose_prior_ceres_bundle_adjuster centers rec_ internally!
        R_wc = np.empty((N, 3, 3), dtype=np.float64)
        C_w = np.empty((N, 3), dtype=np.float64)
        for i_ in range(N):
            M_ = rec_.image(i_ + 1).cam_from_world().matrix()
            R_wc[i_] = M_[:3, :3].T
            C_w[i_] = -R_wc[i_] @ M_[:3, 3]
        # Compute yaw rate [deg/s] for scale-free U-turn exclusion
        ts_rh = (np.array([int(s) for s in stamps], dtype=np.int64) - int(stamps[0])) * 1e-9
        wy_deg = np.zeros(N)
        for i_ in range(1, N - 1):
            dt_ = ts_rh[i_ + 1] - ts_rh[i_ - 1]
            if dt_ > 1e-6:
                wy_deg[i_] = np.degrees(_RotRH.from_matrix(R_wc[i_ - 1].T @ R_wc[i_ + 1]).as_rotvec()[1] / dt_)
        # Collect all triangulated road points (tlen >= 4) and per-frame observed road candidates
        # Clean ego-lane core: x_c in [-0.4, +0.7] (avoids E/W centerline crown at -0.4 and right curb/gutter > +1.0)
        road_pids, road_xyz, road_tlen, road_fset = [], [], [], []
        per_frame_obs = [[] for _ in range(N)]
        for pid, pt in rec_.points3D.items():
            tlen = pt.track.length()
            if tlen < 4:
                continue
            xw = pt.xyz.copy()
            is_road_pt = False
            fset = []
            for elem in pt.track.elements:
                fi = elem.image_id - 1
                fset.append(fi)
                xc = R_wc[fi].T @ (xw - C_w[fi])
                if not (1.8 < xc[2] < 8.5 and -0.4 < xc[0] < 0.7 and 0.6 < xc[1] < 1.8):
                    continue
                if seg_road is not None:
                    uv = rec_.image(elem.image_id).points2D[elem.point2D_idx].xy
                    r_ = int(np.clip(uv[1] / 4.0, 0, seg_road.shape[1] - 1))
                    c_ = int(np.clip(uv[0] / 4.0, 0, seg_road.shape[2] - 1))
                    if seg_road[fi, r_, c_] < 128:
                        continue
                is_road_pt = True
                per_frame_obs[fi].append((pid, xc, tlen))
            if is_road_pt:
                road_pids.append(pid)
                road_xyz.append(xw)
                road_tlen.append(tlen)
                road_fset.append(np.array(fset, dtype=np.int32))
        if len(road_xyz) < 200:
            print(f"   road-height: too few road 3D points ({len(road_xyz)}), skipping", flush=True)
            return []
        road_xyz_arr = np.array(road_xyz, dtype=np.float64)
        tree_r = cKDTree(road_xyz_arr)
        # Fit per-frame local ego-lane road plane using same-pass triangulated road points (|frame - fi| <= 50) within 9.5u
        frame_planes = {}
        pitch_deg = np.full(N, np.nan)
        for fi in range(N):
            raw_idx = tree_r.query_ball_point(C_w[fi], 9.5)
            if len(raw_idx) < 25:
                continue
            idx = [q_ for q_ in raw_idx if np.any(np.abs(road_fset[q_] - fi) <= 50)]
            if len(idx) < 25:
                continue
            xc = (road_xyz_arr[idx] - C_w[fi]) @ R_wc[fi]
            m = (xc[:, 2] > 1.8) & (xc[:, 2] < 8.5) & (xc[:, 0] > -0.4) & (xc[:, 0] < 0.7) & (xc[:, 1] > 0.6) & (xc[:, 1] < 1.8)
            if m.sum() < 25:
                continue
            n_i, h_i, mad_i, sag_i = _fit_road_plane(xc[m])
            if n_i[1] < np.cos(np.radians(8.0)) or not (0.5 < h_i < 2.2):
                continue
            frame_planes[fi] = (n_i, h_i, mad_i, sag_i, idx)
            pitch_deg[fi] = np.degrees(np.arcsin(np.clip(n_i[2], -1.0, 1.0)))
        ok_idx = np.where(np.isfinite(pitch_deg))[0]
        if len(ok_idx) < 50:
            print(f"   road-height: too few valid per-frame planes ({len(ok_idx)}), skipping", flush=True)
            return []
        # 100% scale-free outlier exclusion: mad > 0.04 | sag > 0.04 | pitch_jump > 0.6 deg | |wy| > 15 deg/s
        valid_frames = {}
        n_excl = 0
        for fi in ok_idx:
            n_i, h_i, mad_i, sag_i, idx = frame_planes[fi]
            nb = ok_idx[(np.abs(ok_idx - fi) <= 10) & (ok_idx != fi)]
            p_jump = abs(pitch_deg[fi] - float(np.median(pitch_deg[nb]))) if len(nb) >= 2 else 0.0
            if mad_i > 0.04 or sag_i > 0.04 or p_jump > 0.6 or abs(wy_deg[fi]) > 15.0:
                n_excl += 1
                continue
            valid_frames[fi] = (n_i, h_i, mad_i, idx)
        if len(valid_frames) < 50:
            print(f"   road-height: too few frames after scale-free filter ({len(valid_frames)}), skipping", flush=True)
            return []
        h_vals = np.array([v[1] for v in valid_frames.values()])
        h0 = float(np.median(h_vals))
        prob_ = adj_.problem
        blocks_ = [pose_block(rec_, i + 1) for i in range(N)]
        loss_h = pyceres.CauchyLoss(args.road_height_loss)
        keep_ = [blocks_, loss_h]
        n_added = 0
        n_frames_cov = 0
        k_max = max(1, int(args.road_height_per_frame))
        for fi, (n_i, h_i, mad_i, idx) in valid_frames.items():
            # Prefer road points directly observed by frame fi; fallback to nearby same-pass ego-core road points
            cands = per_frame_obs[fi]
            if len(cands) < k_max:
                xc_nb = (road_xyz_arr[idx] - C_w[fi]) @ R_wc[fi]
                m_nb = (xc_nb[:, 2] > 1.8) & (xc_nb[:, 2] < 5.5) & (xc_nb[:, 0] > -0.4) & (xc_nb[:, 0] < 0.7) & (xc_nb[:, 1] > 0.6) & (xc_nb[:, 1] < 1.8)
                seen_p = {c[0] for c in cands}
                cands = list(cands)
                for q_ in np.where(m_nb)[0]:
                    p_id = road_pids[idx[q_]]
                    if p_id not in seen_p:
                        cands.append((p_id, xc_nb[q_], road_tlen[idx[q_]]))
            good_c = [c for c in cands if abs(float(c[1] @ n_i) - h_i) < 0.05 and c[1][2] <= 5.5]
            if not good_c:
                continue
            good_c.sort(key=lambda c: (abs(c[1][2] - 3.2), -min(c[2], 12)))
            # Collect valid parameter blocks first so sigma scales by sqrt(k) -> exact per-frame info = 1/sigma^2
            sel_pts = []
            for pid, xc, tlen in good_c[:k_max]:
                pt_xyz = rec_.point3D(pid).xyz
                if prob_.has_parameter_block(pt_xyz):
                    sel_pts.append(pt_xyz)
            if not sel_pts:
                continue
            sig_eff = args.road_height * np.sqrt(len(sel_pts))
            for pt_xyz in sel_pts:
                cf = RoadHeightCost(n_i, h0, sig_eff)
                keep_.append(cf)
                keep_.append(pt_xyz)
                prob_.add_residual_block(cf, loss_h, [blocks_[fi], pt_xyz])
                n_added += 1
            n_frames_cov += 1
        spread = float(np.median(np.abs(h_vals - h0)) * 1.4826 / h0 * 100.0)
        print(f"   road-height ruler (ego-core [-0.4,+0.7], sqrt(k)-normalized): h0={h0:.4f}, frame-h mad={spread:.2f}%, "
              f"excluded {n_excl}/{len(ok_idx)} non-planar/turn frames; added {n_added} factors across {n_frames_cov}/{N} frames "
              f"(per-frame sigma={args.road_height}, Cauchy={args.road_height_loss})", flush=True)
        return keep_

    def add_nhc_factors(adj_, rec_, P_cur):
        """Self-calibrate (yaw_0, pitch_0, physical lever_arm L) using exact midpoint axes and attach NHCPairCost."""
        R_wc, C_w = P_cur[:, :3, :3], P_cur[:, :3, 3]
        dC_w = C_w[1:] - C_w[:-1]
        step = np.linalg.norm(dC_w, axis=1)
        mov = step > 0.05
        if mov.sum() < 50:
            print("   NHC: too few moving frames, skipping", flush=True)
            return []
        mx = 0.5 * (R_wc[:-1, :, 0] + R_wc[1:, :, 0])
        my = 0.5 * (R_wc[:-1, :, 1] + R_wc[1:, :, 1])
        mz = 0.5 * (R_wc[:-1, :, 2] + R_wc[1:, :, 2])
        dz_w = R_wc[1:, :, 2] - R_wc[:-1, :, 2]
        # Exact midpoint lateral equation: mx . (dC_w - L*dz_w) - yaw_0 * (mz . dC_w) = 0
        # => [mz . dC_w, mx . dz_w] @ [yaw_0, L] = mx . dC_w
        rhs_l = np.einsum("ni,ni->n", mx[mov], dC_w[mov])
        A_lat = np.c_[np.einsum("ni,ni->n", mz[mov], dC_w[mov]), np.einsum("ni,ni->n", mx[mov], dz_w[mov])]
        coef_lat, _, _, _ = np.linalg.lstsq(A_lat, rhs_l, rcond=None)
        yaw_0 = float(np.clip(coef_lat[0], -0.1, 0.1))
        L_est = float(np.clip(coef_lat[1], 0.0, 3.0))
        dP_rear = dC_w[mov] - L_est * dz_w[mov]
        num_p = float(np.einsum("ni,ni->n", my[mov], dP_rear) @ np.einsum("ni,ni->n", mz[mov], dP_rear))
        den_p = float(np.einsum("ni,ni->n", mz[mov], dP_rear) @ np.einsum("ni,ni->n", mz[mov], dP_rear)) + 1e-12
        pitch_0 = float(np.clip(num_p / den_p, -0.15, 0.15))
        e_lat = np.array([np.cos(yaw_0), 0.0, -np.sin(yaw_0)], dtype=np.float64)
        e_up = np.array([0.0, np.cos(pitch_0), -np.sin(pitch_0)], dtype=np.float64)
        b_rear = np.array([0.0, 0.0, -L_est], dtype=np.float64)
        prob_ = adj_.problem
        blocks_ = [pose_block(rec_, i + 1) for i in range(N)]
        loss_n = pyceres.HuberLoss(args.nhc_loss)
        keep_ = [blocks_, loss_n]
        n_added = 0
        for i in range(N - 1):
            if not mov[i]:
                continue
            cf = NHCPairCost(e_lat, e_up, b_rear, args.nhc, args.nhc)
            keep_.append(cf)
            prob_.add_residual_block(cf, loss_n, [blocks_[i], blocks_[i + 1]])
            n_added += 1
        print(f"   NHC factors (midpoint): yaw0={np.degrees(yaw_0):+.2f} deg, pitch0={np.degrees(pitch_0):+.2f} deg, L={L_est:.3f} u; "
              f"added {n_added} consecutive pairs (sigma={args.nhc}, Huber={args.nhc_loss})", flush=True)
        return keep_

    class StepLengthCost(pyceres.CostFunction):
        """Dense asphalt homography step-length factor: (||c_j - c_i|| - d_target) / sigma_d.
        Touches only (pose_i [7], pose_j [7]) -> zero SPARSE_SCHUR fill-in, zero 3D-point/cy coupling."""
        def __init__(self, d_target, sigma_d):
            super().__init__()
            self.set_num_residuals(1)
            self.set_parameter_block_sizes([7, 7])
            self.d_tgt = float(d_target)
            self.inv_s = 1.0 / float(sigma_d)

        def Evaluate(self, parameters, residuals, jacobians):
            pi = np.asarray(parameters[0], dtype=np.float64)
            pj = np.asarray(parameters[1], dtype=np.float64)
            ci, Jci_q, Rit = _rot_qt(pi[:4], -pi[4:])
            cj, Jcj_q, Rjt = _rot_qt(pj[:4], -pj[4:])
            delta = cj - ci
            dist = float(np.linalg.norm(delta))
            residuals[0] = (dist - self.d_tgt) * self.inv_s
            if jacobians is not None:
                u = (delta / dist) if dist > 1e-9 else np.zeros(3)
                if jacobians[0] is not None:
                    J0 = np.empty(7)
                    J0[:4] = -(u @ Jci_q) * self.inv_s
                    J0[4:] = (u @ Rit) * self.inv_s
                    jacobians[0][:] = J0
                if jacobians[1] is not None:
                    J1 = np.empty(7)
                    J1[:4] = (u @ Jcj_q) * self.inv_s
                    J1[4:] = -(u @ Rjt) * self.inv_s
                    jacobians[1][:] = J1
            return True

    def add_step_prior_factors(adj_, rec_, P_cur):
        """Attach dense-homography StepLengthCost factors calibrated to current trajectory's median height h0."""
        z = np.load(args.step_prior)
        fi_a, fj_a, sobs_a = z["i"].astype(int), z["j"].astype(int), z["s_obs"].astype(np.float64)
        step_k = int(z["step"]) if "step" in z.files else 2
        ok_m = (fi_a >= 0) & (fj_a < N) & np.isfinite(sobs_a) & (sobs_a > 0.05)
        n_fp_drop = 0
        if "used_fp" in z.files:
            ufp = z["used_fp"].astype(bool)
            frp_on = bool(z["frp_enabled"]) if "frp_enabled" in z.files else bool(ufp.any())
            if frp_on:
                n_fp_drop = int((ok_m & ~ufp).sum())
                ok_m &= ufp
        fi_a, fj_a, sobs_a = fi_a[ok_m], fj_a[ok_m], sobs_a[ok_m]
        C_w = P_cur[:, :3, 3]
        R_wc = P_cur[:, :3, :3]
        d_cur = np.linalg.norm(C_w[fj_a] - C_w[fi_a], axis=1)
        # Optional fixed acceleration correction: d_tgt = h0 * (1 + kappa * a_fwd_m_s2) * s_obs
        k_mult = np.ones(len(fi_a), dtype=np.float64)
        if args.step_kappa != 0.0:
            from scipy.interpolate import CubicSpline as _CS_st
            ts_st = (np.array([int(s) for s in stamps], dtype=np.int64) - int(stamps[0])) * 1e-9
            # Estimate meters per unit from typical car camera height ~1.70 m / h_med
            h_tmp = float(np.median(d_cur / sobs_a))
            m_per_u = 1.70 / max(h_tmp, 0.2)
            a_w_st = _CS_st(ts_st, C_w, axis=0)(ts_st, 2) * m_per_u
            a_fwd_st = np.convolve(np.einsum("ni,ni->n", R_wc[:, :, 2], a_w_st), np.ones(5) / 5.0, mode="same")
            a_p = np.clip(0.5 * (a_fwd_st[fi_a] + a_fwd_st[fj_a]), -4.0, 4.0)
            k_mult = 1.0 + float(args.step_kappa) * a_p
        sobs_eff = sobs_a * k_mult
        h_vals = d_cur / sobs_eff
        h0 = float(np.median(h_vals))
        rel_mad = float(np.median(np.abs(h_vals - h0)) * 1.4826 / h0 * 100.0)
        # Scale sigma by sqrt(step_k) to account for overlapping stride=1 pairs of span step_k
        sig_rel = float(args.step_sigma) * np.sqrt(float(max(1, step_k)))
        prob_ = adj_.problem
        blocks_ = [pose_block(rec_, i + 1) for i in range(N)]
        loss_s = pyceres.HuberLoss(args.step_loss)
        keep_ = [blocks_, loss_s]
        for ii, jj, so in zip(fi_a, fj_a, sobs_eff):
            d_tgt = h0 * so
            cf = StepLengthCost(d_tgt, sig_rel * d_tgt)
            keep_.append(cf)
            prob_.add_residual_block(cf, loss_s, [blocks_[ii], blocks_[jj]])
        print(f"   step-prior (dense homography): h0={h0:.4f}, before rel-MAD={rel_mad:.2f}%, kappa={args.step_kappa:+.4f}; "
              f"added {len(fi_a)} pairs (dropped {n_fp_drop} fp-fallback, step={step_k}, eff rel-sigma={sig_rel*100:.2f}%, Huber={args.step_loss})", flush=True)
        return keep_

    def solve_cross(adj_, opt_):
        """solve the adjuster's problem with pyceres.solve (releases GIL so C++ worker threads never deadlock)."""
        so = pyceres.SolverOptions(opt_.ceres.solver_options)
        if args.solver == "sparse":
            so.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        else:
            so.linear_solver_type = pyceres.LinearSolverType.ITERATIVE_SCHUR
            so.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI
        so.num_threads = os.cpu_count() or 8
        sm = pyceres.SolverSummary()
        pyceres.solve(so, adj_.problem, sm)
        term = str(sm.termination_type).split(".")[-1]
        return f"termination_type=TerminationType.{term}, " + sm.BriefReport().replace("\n", " ")

if args.cross_pass:

    def centre_and_jac(p):
        """camera centre c = -R(q)^T t and dc/dp (3 x 7) for p = (x, y, z, w, tx, ty, tz), unit quaternion"""
        v, w, t = p[:3], p[3], p[4:]
        vt = v @ t
        c = -((w * w - v @ v) * t + 2 * v * vt - 2 * w * np.cross(v, t))
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        dv = -(-2 * np.outer(t, v) + 2 * vt * np.eye(3) + 2 * np.outer(v, t) + 2 * w * tx)
        dw = -(2 * w * t - 2 * np.cross(v, t))
        dt = -((w * w - v @ v) * np.eye(3) + 2 * np.outer(v, v) - 2 * w * vx)
        return c, np.c_[dv, dw[:, None], dt]

    class CentreDirCost(pyceres.CostFunction):
        """((c_j - c_i) . d - target) / sigma on the pose blocks of images i and j (analytic Jacobian, checked numerically)"""
        def __init__(self, d, target, sigma):
            super().__init__()
            self.set_num_residuals(1); self.set_parameter_block_sizes([7, 7])
            self.d, self.target, self.sigma = np.asarray(d, float), float(target), float(sigma)

        def Evaluate(self, parameters, residuals, jacobians):
            ci_, Ji = centre_and_jac(parameters[0]); cj_, Jj = centre_and_jac(parameters[1])
            residuals[0] = ((cj_ - ci_) @ self.d - self.target) / self.sigma
            if jacobians is not None:
                if jacobians[0] is not None:
                    jacobians[0][:] = -(self.d @ Ji) / self.sigma
                if jacobians[1] is not None:
                    jacobians[1][:] = (self.d @ Jj) / self.sigma
            return True

    XP = np.load(args.cross_pass)
    XP_i, XP_j, XP_d, XP_t, XP_s, XP_k = (XP[k][::args.cross_stride] for k in ("i", "j", "dir", "target", "sigma", "kind"))
    print(f"cross-pass constraints: {len(XP_i)} ({int((XP_k == 0).sum())} height, {int((XP_k == 1).sum())} centre line, "
          f"{int((XP_k == 2).sum())} dash along), Huber {args.cross_loss} sigma", flush=True)

    def cross_stats(P_):
        r = (((P_[XP_j, :3, 3] - P_[XP_i, :3, 3]) * XP_d).sum(1) - XP_t)
        return "; ".join(f"{nm} rms {np.sqrt((r[XP_k == k] ** 2).mean()):.4f}"
                         for k, nm in ((0, "height"), (1, "centre line"), (2, "dash along")) if (XP_k == k).any())

    def centres_and_jacs(Pm):
        """vectorised centre_and_jac for poses Pm (n, 7): centres (n, 3), Jacobians (n, 3, 7)"""
        v, w, t = Pm[:, :3], Pm[:, 3], Pm[:, 4:]
        vt = (v * t).sum(1); vv = (v * v).sum(1); vxt = np.cross(v, t)
        c = -((w * w - vv)[:, None] * t + 2 * v * vt[:, None] - 2 * w[:, None] * vxt)
        n = len(Pm); I3 = np.eye(3)[None]
        tx = np.zeros((n, 3, 3)); tx[:, 0, 1], tx[:, 0, 2], tx[:, 1, 0], tx[:, 1, 2], tx[:, 2, 0], tx[:, 2, 1] = -t[:, 2], t[:, 1], t[:, 2], -t[:, 0], -t[:, 1], t[:, 0]
        vx = np.zeros((n, 3, 3)); vx[:, 0, 1], vx[:, 0, 2], vx[:, 1, 0], vx[:, 1, 2], vx[:, 2, 0], vx[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
        dv = -(-2 * t[:, :, None] * v[:, None, :] + 2 * vt[:, None, None] * I3 + 2 * v[:, :, None] * t[:, None, :] + 2 * w[:, None, None] * tx)
        dw = -(2 * w[:, None] * t - 2 * vxt)
        dt = -((w * w - vv)[:, None, None] * I3 + 2 * v[:, :, None] * v[:, None, :] - 2 * w[:, None, None] * vx)
        return c, np.concatenate([dv, dw[:, :, None], dt], axis=2)

    class GroupCost(pyceres.CostFunction):
        """all constraints of one frame i: parameter blocks [p_i, p_u1 .. p_un] (unique partners), one residual per
        constraint ((c_j - c_i) . d - target) / sigma, each robustified on its own (square root of the Huber cost)"""
        def __init__(self, slot, d, target, sigma, n_partners):
            super().__init__()
            self.slot, self.d, self.target, self.sigma = np.asarray(slot), np.asarray(d, float), np.asarray(target, float), np.asarray(sigma, float)
            self.m = len(self.slot); self.k = args.cross_loss
            self.set_num_residuals(self.m); self.set_parameter_block_sizes([7] * (1 + n_partners))

        def Evaluate(self, parameters, residuals, jacobians):
            ci_, Ji = centres_and_jacs(np.asarray(parameters[0])[None])
            Pj = np.array([parameters[1 + s] for s in range(len(parameters) - 1)])
            cj, Jj = centres_and_jacs(Pj)
            r = (((cj[self.slot] - ci_[0]) * self.d).sum(1) - self.target) / self.sigma
            a = np.abs(r); big = a > self.k
            rr = r.copy(); scale = np.ones(self.m)
            rr[big] = np.sign(r[big]) * np.sqrt(2 * self.k * a[big] - self.k ** 2)       # rho(r^2) = 2k|r| - k^2 beyond k
            scale[big] = self.k / np.abs(rr[big])                                          # d rr / d r
            residuals[:] = rr
            if jacobians is not None:
                g = (scale / self.sigma)[:, None]
                if jacobians[0] is not None:
                    jacobians[0][:] = (-(np.einsum("mk,kp->mp", self.d, Ji[0])) * g).ravel()
                for s in range(len(parameters) - 1):
                    if jacobians[1 + s] is not None:
                        Jm = np.zeros((self.m, 7)); rows = self.slot == s
                        Jm[rows] = np.einsum("mk,kp->mp", self.d[rows], Jj[s]) * g[rows]
                        jacobians[1 + s][:] = Jm.ravel()
            return True

    def add_cross(adj_, rec_, max_partners=24):
        """add the cross-pass residuals to the adjuster's Ceres problem, grouped per first frame (fewer Python calls);
        returns the objects that must outlive solve()"""
        keep_ = []
        prob_ = adj_.problem
        blocks_ = {}

        def blk(f_):
            if f_ not in blocks_:
                b_ = pose_block(rec_, int(f_) + 1)
                if not prob_.has_parameter_block(b_):
                    raise RuntimeError(f"pose of frame {f_} is not a parameter block of the bundle adjustment")
                blocks_[f_] = b_
            return blocks_[f_]
        order = np.argsort(XP_i, kind="stable")
        cuts = np.r_[0, np.where(np.diff(XP_i[order]))[0] + 1, len(order)]
        nblk = 0
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            idx = order[c0:c1]
            partners = list(dict.fromkeys(XP_j[idx].tolist()))
            for g0 in range(0, len(partners), max_partners):
                part = partners[g0:g0 + max_partners]
                pos = {p: s for s, p in enumerate(part)}
                sub = idx[np.isin(XP_j[idx], part)]
                c_ = GroupCost([pos[j] for j in XP_j[sub]], XP_d[sub], XP_t[sub], XP_s[sub], len(part)); keep_.append(c_)
                prob_.add_residual_block(c_, None, [blk(XP_i[idx[0]])] + [blk(p) for p in part])
                nblk += 1
        print(f"   cross-pass: {len(XP_i)} constraints in {nblk} residual blocks", flush=True)
        keep_.append(blocks_)
        return keep_

t0 = time.time()
res_dir = f"{args.target}/result"
data = f"{args.target}/data"
stamps = sorted(f[:-4] for f in os.listdir(f"{data}/raw_ply") if f.endswith(".ply"))
if args.frames:
    stamps = stamps[:args.frames]
N = len(stamps)
ci = json.load(open(f"{data}/camera_info.json"))
K0 = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1.0]])
W, H = ci["width"], ci["height"]
P_init = np.genfromtxt(args.init, delimiter=",", skip_header=1)[:N, 1:].reshape(-1, 4, 4)
assert len(P_init) == N

# Tightly-coupled IMU preparation (gyro relative rotations + accelerometer gravity tilt)
IMU_GYRO_PAIRS = []
IMU_A_CAM = None
IMU_G_NORM = 9.8
if (args.imu_gyro > 0 or args.imu_gravity > 0):
    import csv as _csv
    from scipy.spatial.transform import Rotation as _Rot
    imu_path = f"{data}/imu.csv"
    assert os.path.exists(imu_path), f"IMU requested but {imu_path} not found"
    ts_f = (np.array([int(s) for s in stamps], dtype=np.int64) - int(stamps[0])) * 1e-9
    t_i, w_i, a_i = [], [], []
    with open(imu_path) as _f:
        _r = _csv.DictReader(_f)
        for _row in _r:
            t_i.append(int(_row["timestamp"]))
            w_i.append([float(_row["angular_velocity_x"]), float(_row["angular_velocity_y"]), float(_row["angular_velocity_z"])])
            a_i.append([float(_row["linear_acceleration_x"]), float(_row["linear_acceleration_y"]), float(_row["linear_acceleration_z"])])
    t_imu = (np.array(t_i, dtype=np.int64) - int(stamps[0])) * 1e-9
    w_imu = np.array(w_i, dtype=np.float64)
    a_imu = np.array(a_i, dtype=np.float64)
    # Identify stationary IMU samples from start trajectory step lengths
    steps_p = np.r_[np.linalg.norm(np.diff(P_init[:, :3, 3], axis=0), axis=1), 0.0]
    med_step = np.median(steps_p[steps_p > 0.2 * np.median(steps_p)])
    still_fr = np.where(steps_p < 0.02 * med_step)[0]
    stat_mask = np.zeros(len(t_imu), bool)
    for fi_ in still_fr:
        stat_mask |= np.abs(t_imu - (ts_f[fi_] + args.imu_dt_ms * 1e-3)) < 0.06
    if stat_mask.sum() < 50:
        stat_mask = (t_imu < 3.0) | (t_imu > t_imu[-1] - 3.0)
    wb_imu = w_imu[stat_mask].mean(0)
    g_imu_stat = a_imu[stat_mask].mean(0)
    IMU_G_NORM = float(np.linalg.norm(g_imu_stat))
    # Refine 3-DOF R_cam_imu via Kabsch SVD against P_init frame-to-frame rotations
    dt_s = args.imu_dt_ms * 1e-3
    w_vis1 = np.array([_Rot.from_matrix(P_init[k, :3, :3].T @ P_init[k + 1, :3, :3]).as_rotvec() for k in range(N - 1)])
    w_int1 = np.zeros_like(w_vis1)
    for k in range(N - 1):
        m_ = (t_imu >= ts_f[k] + dt_s) & (t_imu <= ts_f[k + 1] + dt_s)
        if m_.sum() >= 2:
            w_int1[k] = np.trapezoid(w_imu[m_] - wb_imu, t_imu[m_], axis=0) if hasattr(np, "trapezoid") else np.trapz(w_imu[m_] - wb_imu, t_imu[m_], axis=0)
    U_, _, Vt_ = np.linalg.svd(w_vis1.T @ w_int1)
    D_ = np.eye(3); D_[2, 2] = np.sign(np.linalg.det(U_ @ Vt_))
    R_ci_ref = U_ @ D_ @ Vt_
    R_ci0 = np.array(ci["R_cam_imu"], dtype=np.float64)
    d_ext = np.degrees(np.linalg.norm(_Rot.from_matrix(R_ci0.T @ R_ci_ref).as_rotvec()))
    e_g1 = np.degrees(np.linalg.norm((R_ci_ref @ w_int1.T).T - w_vis1, axis=1))
    print(f"IMU init (dt={args.imu_dt_ms:+.1f} ms): still_samples={int(stat_mask.sum())}, |g|={IMU_G_NORM:.4f}, "
          f"R_cam_imu refined by {d_ext:.2f} deg, 1-step gyro-vs-init med={np.median(e_g1):.3f} deg", flush=True)
    # Integrate exact SO(3) camera-frame gyro rotations along full sequence
    w_cam_imu = (w_imu - wb_imu) @ R_ci_ref.T
    # Precompute cumulative R_cum at each frame timestamp (ts_f + dt_s)
    R_at = np.empty((N, 3, 3))
    R_cur = np.eye(3)
    k_imu = 0
    n_imu = len(t_imu)
    for fi_ in range(N):
        tq = ts_f[fi_] + dt_s
        while k_imu + 1 < n_imu and t_imu[k_imu + 1] <= tq:
            dtm = t_imu[k_imu + 1] - t_imu[k_imu]
            if 0.0 < dtm < 0.1:
                R_cur = R_cur @ _Rot.from_rotvec(0.5 * (w_cam_imu[k_imu] + w_cam_imu[k_imu + 1]) * dtm).as_matrix()
            k_imu += 1
        R_at[fi_] = R_cur
    if args.imu_gyro > 0:
        for stp_ in (1, 5, 20):
            sig_rad = np.radians(args.imu_gyro * np.sqrt(stp_))
            for i_ in range(0, N - stp_, 1 if stp_ <= 5 else 2):
                j_ = i_ + stp_
                R_ij = R_at[i_].T @ R_at[j_]
                q_m = _Rot.from_matrix(R_ij).as_quat()
                IMU_GYRO_PAIRS.append((i_, j_, _qconj(q_m), sig_rad))
        print(f"   IMU gyro factors prepared: {len(IMU_GYRO_PAIRS)} pairs (steps 1,5,20; base sigma {args.imu_gyro:.3f} deg, Huber {args.imu_loss})", flush=True)
    if args.imu_gravity > 0:
        IMU_A_CAM = np.zeros((N, 3))
        for fi_ in range(N):
            m_ = np.abs(t_imu - (ts_f[fi_] + dt_s)) < 0.04
            if m_.sum() > 0:
                IMU_A_CAM[fi_] = R_ci_ref @ a_imu[m_].mean(0)
            else:
                idx_n = int(np.clip(np.searchsorted(t_imu, ts_f[fi_] + dt_s), 0, n_imu - 1))
                IMU_A_CAM[fi_] = R_ci_ref @ a_imu[idx_n]
        print(f"   IMU gravity factors prepared: {N} frames (sigma {args.imu_gravity:.2f} deg, Huber {args.imu_loss})", flush=True)

    def imu_stats(P_):
        parts_ = []
        if IMU_GYRO_PAIRS:
            errs_ = []
            for i_, j_, q_minv, _ in IMU_GYRO_PAIRS[:N - 1]:
                R_p = P_[i_, :3, :3].T @ P_[j_, :3, :3]
                R_m = _Rot.from_quat(_qconj(q_minv)).as_matrix()
                errs_.append(np.degrees(np.linalg.norm(_Rot.from_matrix(R_m.T @ R_p).as_rotvec())))
            parts_.append(f"gyro1 med {np.median(errs_):.3f} deg")
        if IMU_A_CAM is not None:
            g_w_ = np.mean([P_[k, :3, :3] @ IMU_A_CAM[k] for k in range(N)], axis=0)
            g_w_ /= np.linalg.norm(g_w_)
            # Self-calibrated visual accel scale from centripetal acceleration on turning frames
            v_p = np.gradient(P_[:, :3, 3], ts_f, axis=0)
            a_p = np.gradient(v_p, ts_f, axis=0)
            a_imu_w = np.einsum("nij,nj->ni", P_[:, :3, :3], IMU_A_CAM) - g_w_ * IMU_G_NORM
            # Horizontal component dot product -> scale m/unit
            a_p_h = a_p - np.outer(a_p @ g_w_, g_w_)
            a_i_h = a_imu_w - np.outer(a_imu_w @ g_w_, g_w_)
            mag_h = np.linalg.norm(a_p_h, axis=1)
            m_t = mag_h > np.percentile(mag_h, 75)
            sig_m = float(np.clip(np.sum(a_p_h[m_t] * a_i_h[m_t]) / max(np.sum(a_p_h[m_t] ** 2), 1e-9), 0.2, 5.0))
            angs_ = []
            for k in range(N):
                gv = P_[k, :3, :3] @ IMU_A_CAM[k] - sig_m * a_p[k]
                gv /= np.linalg.norm(gv)
                angs_.append(np.degrees(np.arccos(np.clip(gv @ g_w_, -1.0, 1.0))))
            parts_.append(f"grav med {np.median(angs_):.2f} deg (scale {sig_m:.3f} m/u)")
        return "; ".join(parts_)

    def add_imu_factors(adj_, rec_, P_cur):
        keep_ = []
        prob_ = adj_.problem
        blocks_ = {}
        def blk(f_):
            if f_ not in blocks_:
                blocks_[f_] = pose_block(rec_, int(f_) + 1)
            return blocks_[f_]
        loss_g = pyceres.HuberLoss(args.imu_loss)
        keep_.append(loss_g)
        if IMU_GYRO_PAIRS:
            for i_, j_, q_minv, sig_rad in IMU_GYRO_PAIRS:
                c_ = GyroPairCost(q_minv, sig_rad)
                keep_.append(c_)
                prob_.add_residual_block(c_, loss_g, [blk(i_), blk(j_)])
        if IMU_A_CAM is not None:
            g_w_ = np.mean([P_cur[k, :3, :3] @ IMU_A_CAM[k] for k in range(N)], axis=0)
            g_w_ /= np.linalg.norm(g_w_)
            v_p = np.gradient(P_cur[:, :3, 3], ts_f, axis=0)
            a_p = np.gradient(v_p, ts_f, axis=0)
            a_imu_w = np.einsum("nij,nj->ni", P_cur[:, :3, :3], IMU_A_CAM) - g_w_ * IMU_G_NORM
            a_p_h = a_p - np.outer(a_p @ g_w_, g_w_)
            a_i_h = a_imu_w - np.outer(a_imu_w @ g_w_, g_w_)
            mag_h = np.linalg.norm(a_p_h, axis=1)
            m_t = mag_h > np.percentile(mag_h, 75)
            sig_m = float(np.clip(np.sum(a_p_h[m_t] * a_i_h[m_t]) / max(np.sum(a_p_h[m_t] ** 2), 1e-9), 0.2, 5.0))
            sig_grav_rad = np.radians(args.imu_gravity)
            loss_a = pyceres.HuberLoss(args.imu_loss)
            keep_.append(loss_a)
            for k in range(N):
                # Subtract visual acceleration transformed back to camera frame from IMU accel -> pure gravity vector in cam frame!
                g_cam_k = IMU_A_CAM[k] - P_cur[k, :3, :3].T @ (sig_m * a_p[k])
                gn = np.linalg.norm(g_cam_k)
                if gn > 5.0:
                    c_ = GravityCost(g_w_, g_cam_k / gn, sig_grav_rad)
                    keep_.append(c_)
                    prob_.add_residual_block(c_, loss_a, [blk(k)])
        keep_.append(blocks_)
        return keep_

feat_dir = args.feat or [f"{res_dir}/{d}" for d in sorted(os.listdir(res_dir)) if d.startswith("feat_")][0]
kps, descs, obj_kp, kps_v_raw = [], [], [], []
seg_obj = np.load(f"{args.target}/seg.npz")["obj"] if args.mask_obj > 0 else None
n_masked = 0
n_hood_masked = 0
for fi, s in enumerate(stamps):
    z = np.load(f"{feat_dir}/{s}.npz")
    kp_ = z["kp"].astype(np.float64); dc_ = z["desc"].astype(np.float32)
    if args.mask_v_max > 0:
        keep_v = kp_[:, 1] <= args.mask_v_max
        n_hood_masked += int((~keep_v).sum())
        kp_, dc_ = kp_[keep_v], dc_[keep_v]
    is_obj_ = np.zeros(len(kp_), bool)
    if seg_obj is not None:
        r_ = np.clip((kp_[:, 1] / 4).astype(int), 0, seg_obj.shape[1] - 1)
        c_ = np.clip((kp_[:, 0] / 4).astype(int), 0, seg_obj.shape[2] - 1)
        is_obj_ = seg_obj[fi, r_, c_] >= args.mask_obj
        if not args.keep_static_obj:
            keep_m = ~is_obj_
            n_masked += int(is_obj_.sum())
            kp_, dc_, is_obj_ = kp_[keep_m], dc_[keep_m], is_obj_[keep_m]
        else:
            n_masked += int(is_obj_.sum())
    kps.append(kp_); descs.append(dc_); obj_kp.append(is_obj_); kps_v_raw.append(kp_[:, 1].copy())
if args.mask_v_max > 0:
    print(f"hood/bottom mask (v > {args.mask_v_max:.1f}): dropped {n_hood_masked} keypoints", flush=True)
if seg_obj is not None:
    if args.keep_static_obj:
        print(f"static-object mode (seg['obj'] >= {args.mask_obj}): tagged {n_masked} object keypoints for rigidity filtering", flush=True)
    else:
        print(f"dynamic object mask (seg['obj'] >= {args.mask_obj}): dropped {n_masked} keypoints", flush=True)
if args.rs_readout_ms != 0.0:
    from scipy.spatial.transform import Rotation as _Rot
    ts_s = (np.array([int(s) for s in stamps], dtype=np.int64) - int(stamps[0])) * 1e-9
    R_in = P_init[:, :3, :3]
    # Central-difference camera angular velocity [rad/s] in camera frame
    w_cam = np.zeros((N, 3))
    for i_ in range(N):
        i0_, i1_ = max(0, i_ - 1), min(N - 1, i_ + 1)
        dt_ = ts_s[i1_] - ts_s[i0_]
        if dt_ > 1e-6:
            w_cam[i_] = _Rot.from_matrix(R_in[i0_].T @ R_in[i1_]).as_rotvec() / dt_
    # Use start camera (if given) or given K0 for ray undistortion/re-distortion during deskew
    if args.camera:
        cp_ = [float(x) for x in args.camera.split(",")]
        if args.model == "RADIAL":
            K_rs = np.array([[cp_[0], 0, cp_[1]], [0, cp_[0], cp_[2]], [0, 0, 1.0]])
            D_rs = np.array([cp_[3], cp_[4], 0.0, 0.0, 0.0])
        elif args.model == "OPENCV":
            K_rs = np.array([[cp_[0], 0, cp_[2]], [0, cp_[1], cp_[3]], [0, 0, 1.0]])
            D_rs = np.array([cp_[4], cp_[5], cp_[6], cp_[7], 0.0])
        else:
            K_rs, D_rs = K0.copy(), np.zeros(5)
    else:
        K_rs, D_rs = K0.copy(), np.zeros(5)
    cy_rs = K_rs[1, 2]
    max_shift = 0.0
    for fi in range(N):
        kp_ = kps[fi]
        if len(kp_) == 0:
            continue
        dt_v = (args.rs_readout_ms * 1e-3) * (kp_[:, 1] - cy_rs) / float(H)
        u_n = cv2.undistortPoints(kp_.reshape(-1, 1, 2), K_rs, D_rs).reshape(-1, 2)
        rays = np.c_[u_n, np.ones(len(u_n))]
        rot_v = w_cam[fi][None, :] * dt_v[:, None]
        rays0 = _Rot.from_rotvec(rot_v).apply(rays)
        u0 = rays0[:, :2] / rays0[:, 2:3]
        r2 = (u0 ** 2).sum(1)
        rad = 1.0 + D_rs[0] * r2 + D_rs[1] * (r2 ** 2)
        x_d = u0[:, 0] * rad + 2 * D_rs[2] * u0[:, 0] * u0[:, 1] + D_rs[3] * (r2 + 2 * u0[:, 0] ** 2)
        y_d = u0[:, 1] * rad + D_rs[2] * (r2 + 2 * u0[:, 1] ** 2) + 2 * D_rs[3] * u0[:, 0] * u0[:, 1]
        kp_new = np.c_[x_d * K_rs[0, 0] + K_rs[0, 2], y_d * K_rs[1, 1] + K_rs[1, 2]]
        max_shift = max(max_shift, float(np.linalg.norm(kp_new - kp_, axis=1).max()))
        kps[fi] = kp_new
    print(f"rolling-shutter rotational deskew (T_rs={args.rs_readout_ms:+.1f} ms): max pixel shift {max_shift:.2f} px", flush=True)
LINKS = []
LINK_EDGES = []
if args.links and args.links_merge > 0:
    zl = np.load(args.links)
    trees_ = {}
    def snap_(f, uv):
        if f not in trees_:
            trees_[f] = cKDTree(kps[f])
        d_, j_ = trees_[f].query(uv)
        return j_ if d_ <= args.links_merge else -1
    for fa_, fb_, ua_, ub_ in zip(zl["fa"], zl["fb"], zl["uva"], zl["uvb"]):
        if fa_ >= N or fb_ >= N:
            continue
        ja_, jb_ = snap_(fa_, ua_), snap_(fb_, ub_)
        if ja_ >= 0 and jb_ >= 0:
            LINK_EDGES.append((fa_, ja_, fb_, jb_))
    print(f"cross-pass links snapped to existing keypoints (<= {args.links_merge} px): {len(LINK_EDGES)} of {len(zl['fa'])}")
elif args.links:
    zl = np.load(args.links)
    for fa_, fb_, ua_, ub_ in zip(zl["fa"], zl["fb"], zl["uva"], zl["uvb"]):
        if fa_ >= N or fb_ >= N:
            continue
        ia_, ib_ = len(kps[fa_]), len(kps[fb_])
        kps[fa_] = np.vstack([kps[fa_], ua_[None]]); kps[fb_] = np.vstack([kps[fb_], ub_[None]])
        kps_v_raw[fa_] = np.r_[kps_v_raw[fa_], ua_[1]]; kps_v_raw[fb_] = np.r_[kps_v_raw[fb_], ub_[1]]
        LINKS.append((np.array([fa_, fb_]), np.array([ia_, ib_])))
    print(f"road-marking links between opposite passes: {len(LINKS)}")
if args.square:
    for k_ in kps:
        k_[:, 1] = ci["cy"] + (k_[:, 1] - ci["cy"]) * ci["fx"] / ci["fy"]
    ci["fy"] = ci["fx"]
ci["fx"] *= args.fscale; ci["fy"] *= args.fscale
K0 = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1.0]])
offsets = np.r_[0, np.cumsum([len(k) for k in kps])]
print(f"{N} images, {offsets[-1]} keypoints ({feat_dir})")

# ---------------------------------------------------------------- pairs
pairs = {(i, i + s) for s in (int(x) for x in args.steps.split(",")) for i in range(N - s)}
n_seq = len(pairs)
if not args.no_loops:
    C, Z = P_init[:, :3, 3], P_init[:, :3, 2]
    tree = cKDTree(C)
    for a in range(N):
        nb = np.array([b for b in tree.query_ball_point(C[a], args.loop_radius) if abs(b - a) > 150], dtype=int)
        if len(nb) == 0:
            continue
        nb = nb[Z[nb] @ Z[a] > np.cos(np.radians(30.0))]
        nb = nb[np.argsort(np.linalg.norm(C[nb] - C[a], axis=1))]
        chosen = []
        for b in nb:
            if all(abs(b - c) >= 4 for c in chosen):
                chosen.append(b)
            if len(chosen) >= args.loop_per_frame:
                break
        pairs.update((min(a, b), max(a, b)) for b in chosen)
pairs = sorted(pairs)
print(f"pairs: {n_seq} sequential + {len(pairs) - n_seq} revisits")

KPV = kps
if args.verify_undistort and args.camera:
    cp_ = [float(x) for x in args.camera.split(",")]
    if args.model == "RADIAL":
        Kv_ = np.array([[cp_[0], 0, cp_[1]], [0, cp_[0], cp_[2]], [0, 0, 1.0]]); dv_ = np.array([cp_[3], cp_[4], 0, 0, 0.0])
    elif args.model == "OPENCV":
        Kv_ = np.array([[cp_[0], 0, cp_[2]], [0, cp_[1], cp_[3]], [0, 0, 1.0]]); dv_ = np.array(cp_[4:8] + [0.0])
    else:
        raise SystemExit("--verify-undistort: RADIAL or OPENCV camera needed")
    KPV = [cv2.undistortPoints(k_.reshape(-1, 1, 2), Kv_, dv_, P=Kv_).reshape(-1, 2) for k_ in kps]
    K0 = Kv_
    print("pair verification on undistorted keypoints")

dev = "cuda" if (HAS_CUDA and not args.cpu) else "cpu"
gm = GpuMatcher(descs, ratio=args.ratio, half=True, device=dev)


def verified(pq):
    cv2.setNumThreads(1)
    a, b = pq
    m = gm.match(a, b)
    if len(m) < 30:
        return np.zeros((0, 2), int)
    cv2.setRNGSeed(a * 100003 + b)
    E, ok = cv2.findEssentialMat(KPV[a][m[:, 0]], KPV[b][m[:, 1]], K0, cv2.RANSAC, 0.999, args.e_thr)
    if E is None or ok is None or ok.sum() < 30:
        return np.zeros((0, 2), int)
    m = m[ok[:, 0] > 0]
    return np.c_[offsets[a] + m[:, 0], offsets[b] + m[:, 1]]


with ThreadPoolExecutor(os.cpu_count()) as ex:
    edges = [e for e in ex.map(verified, pairs) if len(e)]
del gm
try:                                                   # give the matcher's GPU cache back (parallel runs)
    import torch; torch.cuda.empty_cache()
except Exception:
    pass
print(f"verified pairs: {len(edges)}, matches {sum(len(e) for e in edges)}  ({time.time()-t0:.0f}s)")
if LINK_EDGES:
    edges.append(np.array([(offsets[fa_] + ja_, offsets[fb_] + jb_) for fa_, ja_, fb_, jb_ in LINK_EDGES], dtype=np.int64))
edges = np.concatenate(edges)

# ---------------------------------------------------------------- tracks
g = coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])), shape=(offsets[-1], offsets[-1]))
_, lab = connected_components(g, directed=False)
used = np.unique(edges)
order = used[np.argsort(lab[used], kind="stable")]
tracks = []
N_SPLIT = [0, 0]
for tr in np.split(order, np.where(np.diff(lab[order]))[0] + 1):
    if len(tr) < args.min_track:
        continue
    fr = np.searchsorted(offsets, tr, side="right") - 1
    if len(np.unique(fr)) == len(fr):
        tracks.append((fr, tr - offsets[fr]))
    elif args.split_conflicts:
        u_, c_ = np.unique(fr, return_counts=True)
        keep_ = np.isin(fr, u_[c_ == 1])
        N_SPLIT[0] += 1
        if keep_.sum() >= args.min_track:
            tracks.append((fr[keep_], (tr - offsets[fr])[keep_]))
    else:
        N_SPLIT[1] += 1
if args.max_obs:
    tracks.sort(key=lambda t: -len(t[0]))
    tot = np.cumsum([len(t[0]) for t in tracks])
    tracks = tracks[:int(np.searchsorted(tot, args.max_obs))]
is_link = np.r_[np.zeros(len(tracks), bool), np.ones(len(LINKS), bool)]
tracks += LINKS
tr_fr = np.concatenate([t[0] for t in tracks]); tr_kp = np.concatenate([t[1] for t in tracks])
tr_len = np.array([len(t[0]) for t in tracks]); tr_off = np.r_[0, np.cumsum(tr_len)]
tr_id = np.repeat(np.arange(len(tracks)), tr_len)
laps_span = np.array([t[0].max() - t[0].min() for t in tracks])
print(f"conflicting tracks: {N_SPLIT[0]} split, {N_SPLIT[1]} dropped")
print(f"tracks: {len(tracks)}, observations {len(tr_fr)}, mean length {tr_len.mean():.1f}, "
      f"spanning > 150 frames (revisits): {(laps_span > 150).sum()}  ({time.time()-t0:.0f}s)")
obs_xy = np.concatenate([kps[f] for f in range(N)])[offsets[tr_fr] + tr_kp]      # distorted pixels of every observation
obs_v_raw = np.concatenate(kps_v_raw)[offsets[tr_fr] + tr_kp]                    # raw image row before RS deskew & --square
obs_is_obj = np.concatenate(obj_kp)[offsets[tr_fr] + tr_kp] if (seg_obj is not None and args.keep_static_obj) else None
tr_has_obj = (np.add.reduceat(obs_is_obj.astype(int), tr_off[:-1]) > 0) if obs_is_obj is not None else None


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    X, Y = src - ms, dst - md
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src))
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    Rm = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / X.var(0).sum()
    return s, Rm, md - s * Rm @ ms


def _project_pts(Xc_arr, Kc, dist):
    """Project 3D camera-frame points (N,3) to distorted pixels (N,2) matching the active camera model."""
    if model == "OPENCV_FISHEYE":
        uv, _ = cv2.fisheye.projectPoints(Xc_arr.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc, dist[:4])
        return uv.reshape(-1, 2)
    if model == "FULL_OPENCV":
        uv, _ = cv2.projectPoints(Xc_arr.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc, dist)
        return uv.reshape(-1, 2)
    u = Xc_arr[:, :2] / Xc_arr[:, 2:3]
    r2 = (u ** 2).sum(1)
    k1 = dist[0] if len(dist) > 0 else 0.0
    k2 = dist[1] if len(dist) > 1 else 0.0
    p1 = dist[2] if len(dist) > 2 else 0.0
    p2 = dist[3] if len(dist) > 3 else 0.0
    rad = 1.0 + k1 * r2 + k2 * (r2 ** 2)
    xd = u[:, 0] * rad + 2.0 * p1 * u[:, 0] * u[:, 1] + p2 * (r2 + 2.0 * u[:, 0] ** 2)
    yd = u[:, 1] * rad + p1 * (r2 + 2.0 * u[:, 1] ** 2) + 2.0 * p2 * u[:, 0] * u[:, 1]
    return np.c_[xd * Kc[0, 0] + Kc[0, 2], yd * Kc[1, 1] + Kc[1, 2]]


def deskew_rs_6dof(poses_cur, X_cur, Kc, dist):
    """Apply 3D translational rolling-shutter deskew on top of rotational-deskewed obs_xy.
    Camera moves +v_c * dt_v at row exposure time t + dt_v, so point in cam frame is Xc - v_c * dt_v.
    Deskewed pixel = obs_xy + (proj(Xc) - proj(Xc - v_c * dt_v))."""
    from scipy.interpolate import CubicSpline
    pos_w = poses_cur[:, :3, 3]
    if np.all(np.diff(ts_s) > 0):
        V_cam_w = CubicSpline(ts_s, pos_w, axis=0)(ts_s, 1)
    else:
        V_cam_w = np.gradient(pos_w, ts_s, axis=0)
    v_cam = np.einsum("nji,nj->ni", poses_cur[:, :3, :3], V_cam_w)   # R_wc^T @ V_w -> camera frame [m/s]
    Tinv = np.linalg.inv(poses_cur)
    Xc = np.einsum("nij,nj->ni", Tinv[tr_fr, :3, :3], X_cur[tr_id]) + Tinv[tr_fr, :3, 3]
    dt_v = (args.rs_readout_ms * 1e-3) * (obs_v_raw - cy_rs) / float(H)
    Xc_rs = Xc - v_cam[tr_fr] * dt_v[:, None]
    valid = (Xc[:, 2] > 0.5) & (Xc_rs[:, 2] > 0.5)
    du = np.zeros_like(obs_xy)
    if valid.any():
        du_v = _project_pts(Xc[valid], Kc, dist) - _project_pts(Xc_rs[valid], Kc, dist)
        mag = np.linalg.norm(du_v, axis=1)
        ok_d = np.isfinite(mag) & (mag < 25.0)
        idx_v = np.where(valid)[0][ok_d]
        du[idx_v] = du_v[ok_d]
        if ok_d.any():
            m_ok = mag[ok_d]
            print(f"   RS 6-DOF translational deskew: {ok_d.sum()}/{len(obs_xy)} obs, "
                  f"median {np.median(m_ok):.2f} px, p95 {np.percentile(m_ok, 95):.2f} px, max {m_ok.max():.2f} px", flush=True)
    return obs_xy + du


def triangulate_all(poses, Kc, dist, max_reproj, min_par, obs_xy_=None):
    """DLT of every track from undistorted observations; returns (X, keep mask per observation, track ok)"""
    oxy = obs_xy if obs_xy_ is None else obs_xy_
    if model == "OPENCV_FISHEYE":
        und = cv2.fisheye.undistortPoints(oxy.reshape(-1, 1, 2), Kc, dist[:4], P=Kc).reshape(-1, 2)
    else:
        und = cv2.undistortPoints(oxy.reshape(-1, 1, 2), Kc, dist, P=Kc).reshape(-1, 2)
    Tinv = np.linalg.inv(poses)
    Pw = Kc @ Tinv[:, :3, :]
    rows = np.empty((len(tr_fr), 2, 4))
    rows[:, 0] = und[:, 0:1] * Pw[tr_fr, 2] - Pw[tr_fr, 0]
    rows[:, 1] = und[:, 1:2] * Pw[tr_fr, 2] - Pw[tr_fr, 1]
    # normal equations per track (4x4), smallest eigenvector
    AtA = np.einsum("nri,nrj->nij", rows, rows)
    M = np.add.reduceat(AtA, tr_off[:-1], axis=0)
    w, v = np.linalg.eigh(M)
    Xh = v[:, :, 0]
    X = Xh[:, :3] / np.where(np.abs(Xh[:, 3:4]) > 1e-12, Xh[:, 3:4], 1e-12)
    Xc = np.einsum("nij,nj->ni", Tinv[tr_fr, :3, :3], X[tr_id]) + Tinv[tr_fr, :3, 3]
    pr = Xc @ Kc.T
    err = np.linalg.norm(pr[:, :2] / pr[:, 2:3] - und, axis=1)
    front = Xc[:, 2] > 0.2
    keep = front & (err < max_reproj)
    nk = np.add.reduceat(keep.astype(int), tr_off[:-1])
    nfront = np.add.reduceat(front.astype(int), tr_off[:-1])
    ok = (nk >= np.where(is_link, 2, args.min_track)) & (nk >= 0.6 * tr_len) & (nfront == tr_len)
    if tr_has_obj is not None:
        # Strict rigidity for segmented objects: keeps stationary parked cars (<1.2 px over >=6 frames), drops moving traffic
        strict_keep = front & (err < min(max_reproj, 1.2))
        nk_strict = np.add.reduceat(strict_keep.astype(int), tr_off[:-1])
        rigid_obj = (nk_strict >= max(args.min_track, 6)) & (nk_strict >= 0.85 * tr_len)
        n_obj_tot = int((ok & tr_has_obj).sum())
        ok = np.where(tr_has_obj, ok & rigid_obj, ok)
        n_obj_kept = int((ok & tr_has_obj).sum())
        print(f"   static parked-object tracks kept: {n_obj_kept} / {n_obj_tot} (dropped {n_obj_tot - n_obj_kept} moving)", flush=True)
    # parallax: largest angle between the rays of the kept observations ~ angle between the two extreme frames' rays
    rays = X[tr_id] - poses[tr_fr, :3, 3]; rays /= np.linalg.norm(rays, axis=1)[:, None] + 1e-12
    mean = np.add.reduceat(rays, tr_off[:-1], axis=0); mean /= np.linalg.norm(mean, axis=1)[:, None] + 1e-12
    cosm = np.minimum.reduceat(np.einsum("ni,ni->n", rays, mean[tr_id]), tr_off[:-1])
    ok &= cosm < np.cos(np.radians(min_par / 2))
    return X, keep, ok, err


poses = P_init.copy()
PRIOR_COV = np.repeat(np.eye(3)[None] * args.pose_prior ** 2, N, axis=0)
if args.prior_aniso:
    sa, sl, su = (float(x) for x in args.prior_aniso.split(","))
    Cp = P_init[:, :3, 3]
    tang = np.gradient(Cp, axis=0)
    tang = np.array([np.mean(tang[max(0, i - 3):i + 4], 0) for i in range(N)])
    for i in range(N):
        up_ = -P_init[i, :3, 1]
        f_ = tang[i] - (tang[i] @ up_) * up_
        f_ = f_ / np.linalg.norm(f_) if np.linalg.norm(f_) > 1e-9 else P_init[i, :3, 2]
        l_ = np.cross(up_, f_); u_ = np.cross(f_, l_)
        Rr = np.c_[f_, l_, u_]
        PRIOR_COV[i] = Rr @ np.diag([sa ** 2, sl ** 2, su ** 2]) @ Rr.T
    print(f"anisotropic position prior: along {sa}, lateral {sl}, vertical {su}")
model = args.model
cam_params = {"OPENCV": [ci["fx"], ci["fy"], ci["cx"], ci["cy"], args.k1, 0, 0, 0],
              "FULL_OPENCV": [ci["fx"], ci["fy"], ci["cx"], ci["cy"], args.k1, 0, 0, 0, 0, 0, 0, 0],
              "OPENCV_FISHEYE": [ci["fx"], ci["fy"], ci["cx"], ci["cy"], 0, 0, 0, 0],
              "RADIAL": [(ci["fx"] + ci["fy"]) / 2, ci["cx"], ci["cy"], args.k1, 0],
              "SIMPLE_RADIAL": [(ci["fx"] + ci["fy"]) / 2, ci["cx"], ci["cy"], args.k1],
              "PINHOLE": [ci["fx"], ci["fy"], ci["cx"], ci["cy"]]}[model]
if args.camera:
    cam_params = [float(x) for x in args.camera.split(",")]


def cv_camera(p):
    """OpenCV K, dist of the current colmap camera parameters"""
    if model == "OPENCV":
        return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]]), np.array([p[4], p[5], p[6], p[7], 0.0])
    if model == "FULL_OPENCV":
        return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]]), np.array([p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]])
    if model == "OPENCV_FISHEYE":
        return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]]), np.array([p[4], p[5], p[6], p[7]])
    if model == "RADIAL":
        return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]]), np.array([p[3], p[4], 0, 0, 0.0])
    if model == "SIMPLE_RADIAL":
        return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]]), np.array([p[3], 0, 0, 0, 0.0])
    return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]]), np.zeros(5)


def write_csv(path, P_):
    header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
    with open(path + ".tmp", "w") as f:
        f.write(header + "\n")
        for s_, T in zip(stamps, P_):
            f.write(s_ + "," + ",".join(repr(float(x)) for x in T.reshape(-1)) + "\n")
    os.replace(path + ".tmp", path)


log = []
THR = [float(x) for x in args.rounds.split(",")]
THR += [THR[-1]] * args.repeat_last
TERM = [None]
for rnd, thr in enumerate(THR):
    if rnd >= len(THR) - args.repeat_last and args.repeat_last and log and log[-1]["moved"] < args.tol and (
            TERM[0] == "CONVERGENCE" or (len(log) >= 2 and log[-2]["moved"] < args.tol)):
        print(f"converged: moved {log[-1]['moved']:.4f} < {args.tol} (Ceres {TERM[0]}) -> stop", flush=True)
        break
    Kc, dist = cv_camera(cam_params)
    X, keep, ok, err = triangulate_all(poses, Kc, dist, thr, args.min_parallax)
    obs_xy_round = obs_xy
    if args.rs_6dof and args.rs_readout_ms != 0.0:
        obs_xy_round = deskew_rs_6dof(poses, X, Kc, dist)
        X, keep, ok, err = triangulate_all(poses, Kc, dist, thr, args.min_parallax, obs_xy_=obs_xy_round)
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model=model, width=W, height=H, params=np.array(cam_params, dtype=np.float64), camera_id=1)
    rec.add_camera_with_trivial_rig(cam)
    # per image: only the keypoints that are observations of a kept track
    sel = keep & ok[tr_id]
    o_idx = np.where(sel)[0]
    o_fr = tr_fr[o_idx]
    srt = np.argsort(o_fr, kind="stable")
    o_idx, o_fr = o_idx[srt], o_fr[srt]
    p2d = np.empty(len(o_idx), int)            # index of the observation among its image's points2D
    cnt = np.bincount(o_fr, minlength=N)
    st = np.r_[0, np.cumsum(cnt)]
    p2d[:] = np.arange(len(o_idx)) - st[o_fr]
    for i in range(N):
        im = pycolmap.Image(name=stamps[i], keypoints=obs_xy_round[o_idx[st[i]:st[i + 1]]], camera_id=1, image_id=i + 1)
        rec.add_image_with_trivial_frame(im, pycolmap.Rigid3d(np.linalg.inv(poses[i])[:3]))
    # observations grouped by track
    by_tr = np.argsort(tr_id[o_idx], kind="stable")
    t_of = tr_id[o_idx][by_tr]
    cuts = np.r_[0, np.where(np.diff(t_of))[0] + 1, len(t_of)]
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        j = by_tr[c0:c1]
        tr = pycolmap.Track([pycolmap.TrackElement(int(o_fr[q]) + 1, int(p2d[q])) for q in j])
        rec.add_point3D(X[t_of[c0]], tr, np.zeros(3, np.uint8))
    n_pts, n_obs = rec.num_points3D(), rec.compute_num_observations()
    e0 = rec.compute_mean_reprojection_error()
    opt = pycolmap.BundleAdjustmentOptions()
    opt.refine_focal_length = not (args.fix_intrinsics or args.fix_f)
    opt.refine_extra_params = not args.fix_intrinsics
    opt.refine_principal_point = not (args.fix_intrinsics or args.no_pp)
    opt.constant_rig_from_world_rotation = args.fix_rot
    opt.refine_rig_from_world = not args.fix_poses
    opt.print_summary = False
    opt.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opt.ceres.loss_function_scale = args.loss_scale
    opt.ceres.solver_options.max_num_iterations = args.max_iter
    if args.solver == "sparse":
        opt.ceres.max_num_images_direct_sparse_cpu_solver = 100000
    if args.pose_prior > 0 or args.prior_aniso:
        cfg = pycolmap.BundleAdjustmentConfig()
        for i in range(N):
            cfg.add_image(i + 1)
        if args.fix_intrinsics:
            cfg.set_constant_cam_intrinsics(1)
        for f in (int(x) for x in args.anchors.split(",") if x):
            cfg.set_constant_rig_from_world_pose(f + 1)
        priors = []
        for i in range(N):
            pp = pycolmap.PosePrior()
            pp.pose_prior_id = i + 1
            pp.corr_data_id = rec.image(i + 1).data_id
            pp.position = P_init[i, :3, 3].astype(np.float64)
            pp.position_covariance = PRIOR_COV[i]
            pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            priors.append(pp)
        popt = pycolmap.PosePriorBundleAdjustmentOptions()
        if USE_PYCERES:
            adj = pycolmap.create_pose_prior_ceres_bundle_adjuster(opt, popt, cfg, priors, rec)
            keep_all = []
            if args.cross_pass:
                print(f"   cross-pass before: {cross_stats(poses)}", flush=True)
                keep_all.append(add_cross(adj, rec))
            if args.imu_gyro > 0 or args.imu_gravity > 0:
                print(f"   IMU before:        {imu_stats(poses)}", flush=True)
                keep_all.append(add_imu_factors(adj, rec, poses))
            if args.road_height > 0:
                keep_all.append(add_road_height_factors(adj, rec, poses))
            if args.nhc > 0:
                keep_all.append(add_nhc_factors(adj, rec, poses))
            if args.step_prior:
                keep_all.append(add_step_prior_factors(adj, rec, poses))
            summ = solve_cross(adj, opt)
            del keep_all
        else:
            summ = pycolmap.create_pose_prior_bundle_adjuster(opt, popt, cfg, priors, rec).solve()
        TERM[0] = str(summ).split("termination_type=TerminationType.")[1].split(",")[0].split(")")[0] if "termination_type=" in str(summ) else None
        print("   BA (pose priors):", str(summ).split("\n")[0][:160], flush=True)
    elif args.anchors or USE_PYCERES:
        cfg = pycolmap.BundleAdjustmentConfig()
        for i in range(N):
            cfg.add_image(i + 1)
        if args.anchors:
            for f in (int(x) for x in args.anchors.split(",")):
                cfg.set_constant_rig_from_world_pose(f + 1)
        if args.fix_intrinsics:
            cfg.set_constant_cam_intrinsics(1)
        if USE_PYCERES:
            adj = pycolmap.create_default_ceres_bundle_adjuster(opt, cfg, rec)
            keep_all = []
            if args.cross_pass:
                print(f"   cross-pass before: {cross_stats(poses)}", flush=True)
                keep_all.append(add_cross(adj, rec))
            if args.imu_gyro > 0 or args.imu_gravity > 0:
                print(f"   IMU before:        {imu_stats(poses)}", flush=True)
                keep_all.append(add_imu_factors(adj, rec, poses))
            if args.road_height > 0:
                keep_all.append(add_road_height_factors(adj, rec, poses))
            if args.nhc > 0:
                keep_all.append(add_nhc_factors(adj, rec, poses))
            if args.step_prior:
                keep_all.append(add_step_prior_factors(adj, rec, poses))
            summ = solve_cross(adj, opt)
            del keep_all
        else:
            summ = pycolmap.create_default_bundle_adjuster(opt, cfg, rec).solve()
        TERM[0] = str(summ).split("termination_type=TerminationType.")[1].split(",")[0].split(")")[0] if "termination_type=" in str(summ) else None
        print("   BA:", str(summ).split("\n")[0][:160], flush=True)
    else:
        pycolmap.bundle_adjustment(rec, opt)
    cam_params = [float(x) for x in rec.camera(1).params]
    new = np.array([np.linalg.inv(np.vstack([rec.image(i + 1).cam_from_world().matrix(), [0, 0, 0, 1]])) for i in range(N)])
    s, Rm, t = umeyama(new[:, :3, 3], P_init[:, :3, 3])       # keep the frame / unit of the start trajectory
    new[:, :3, 3] = s * new[:, :3, 3] @ Rm.T + t
    new[:, :3, :3] = Rm @ new[:, :3, :3]
    moved = np.sqrt((np.linalg.norm(new[:, :3, 3] - poses[:, :3, 3], axis=1) ** 2).mean())
    ate0 = np.sqrt((np.linalg.norm(new[:, :3, 3] - P_init[:, :3, 3], axis=1) ** 2).mean())
    poses = new
    if args.cross_pass:
        print(f"   cross-pass after:  {cross_stats(poses)}", flush=True)
    if args.imu_gyro > 0 or args.imu_gravity > 0:
        print(f"   IMU after:         {imu_stats(poses)}", flush=True)
    PTS = np.array([p_.xyz for p_ in rec.points3D.values()]) * s @ Rm.T + t
    PLEN = np.array([p_.track.length() for p_ in rec.points3D.values()])
    e1 = rec.compute_mean_reprojection_error()
    if LINKS:
        Kc_, d_ = cv_camera(cam_params)
        X2, keep2, ok2, err2 = triangulate_all(poses, Kc_, d_, 1e9, 0.0)
        li = np.where(is_link[tr_id])[0]
        print(f"   links: {int(ok[is_link].sum())} of {len(LINKS)} used; reprojection after BA median {np.median(err2[li]):.2f} px, "
              f"p90 {np.percentile(err2[li], 90):.2f} px", flush=True)
    line = dict(round=rnd, thr=thr, termination=TERM[0], points=n_pts, obs=n_obs, reproj_before=e0, reproj_after=e1, camera=cam_params,
                moved=moved, ate_to_init=ate0, seconds=time.time() - t0)
    log.append(line)
    write_csv(f"{res_dir}/{args.out}_r{rnd}.csv", poses)          # every round: usable before the run ends
    print(f"round {rnd} (filter {thr:g} px): {n_pts} points / {n_obs} observations, mean reprojection {e0:.3f} -> {e1:.3f} px; "
          f"camera {model} " + " ".join(f"{x:.5g}" for x in cam_params) + f"; moved {moved:.4f} m, to start trajectory {ate0:.3f} m; Ceres {TERM[0]}  ({time.time()-t0:.0f}s)", flush=True)

write_csv(f"{res_dir}/{args.out}.csv", poses)
json.dump(dict(argv=sys.argv, model=model, camera=cam_params, given=[ci["fx"], ci["fy"], ci["cx"], ci["cy"]], rounds=log,
               pycolmap=pycolmap.__version__), open(f"{res_dir}/{args.out}.json", "w"), indent=1)
extra_ = {}
if args.dump_tracks:
    seg_ = np.load(f"{args.target}/seg.npz")["obj"]
    fmin, fmax, objp = [], [], []
    for p_ in rec.points3D.values():
        fr_ = [e.image_id - 1 for e in p_.track.elements]
        xy_ = np.array([rec.image(e.image_id).points2D[e.point2D_idx].xy for e in p_.track.elements])
        r_ = np.clip((xy_[:, 1] / 4).astype(int), 0, seg_.shape[1] - 1); c_ = np.clip((xy_[:, 0] / 4).astype(int), 0, seg_.shape[2] - 1)
        fmin.append(min(fr_)); fmax.append(max(fr_)); objp.append(float(seg_[fr_, r_, c_].mean()) / 255.0)
    extra_ = dict(fmin=np.array(fmin), fmax=np.array(fmax), obj=np.array(objp, np.float32))
    print(f"track dump: {len(fmin)} points", flush=True)
np.savez(f"{res_dir}/{args.out}.pts.npz", xyz=PTS.astype(np.float32), length=PLEN, **extra_)
print(f"saved {res_dir}/{args.out}.csv")
