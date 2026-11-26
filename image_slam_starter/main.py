import argparse
import os

import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.ICP import ICP, preprocess
from utils.submap import SubmapManager
from utils.keyframe import KeyframeManager
from utils.posegraph import PoseGraph
from utils.loopclosure import LoopClosure
from utils.io import load_folder, load_ply, load_camera_info, load_imu
from utils.imu import ImuRotation
from utils.visual import VisualMatcher
from utils.viz import Visualizer

def distribute_pose_corrections(poses, kf_indices, kf_opt_poses):
    """
    Interpolate pose graph corrections to all non-keyframe poses using SLERP.
    """
    new_poses = [p.copy() for p in poses]

    for k in range(len(kf_indices) - 1):
        idx_A = kf_indices[k]
        idx_B = kf_indices[k+1]

        orig_A, orig_B = poses[idx_A], poses[idx_B]
        opt_A, opt_B = kf_opt_poses[k], kf_opt_poses[k+1]

        # Calculate correction transforms
        corr_A = opt_A @ np.linalg.inv(orig_A)
        corr_B = opt_B @ np.linalg.inv(orig_B)

        rot_A = R.from_matrix(corr_A[:3, :3])
        rot_B = R.from_matrix(corr_B[:3, :3])
        trans_A = corr_A[:3, 3]
        trans_B = corr_B[:3, 3]

        # Setup SLERP
        key_rots = R.from_quat(np.vstack([rot_A.as_quat(), rot_B.as_quat()]))
        try:
            slerp = Slerp([0, 1], key_rots)
        except Exception:
            slerp = None

        # Apply interpolated correction to intermediate frames
        for j in range(idx_A, idx_B):
            t = (j - idx_A) / float(idx_B - idx_A)

            rot_t = slerp(t).as_matrix() if slerp else rot_A.as_matrix()
            trans_t = trans_A + t * (trans_B - trans_A)

            corr_j = np.eye(4)
            corr_j[:3, :3] = rot_t
            corr_j[:3, 3] = trans_t

            new_poses[j] = corr_j @ poses[j]

    # Apply the last correction to any trailing frames
    if len(kf_indices) > 0:
        last_kf_idx = kf_indices[-1]
        last_corr = kf_opt_poses[-1] @ np.linalg.inv(poses[last_kf_idx])
        for j in range(last_kf_idx, len(poses)):
            new_poses[j] = last_corr @ poses[j]

    return new_poses


# ==========================================
# Main SLAM Pipeline
# ==========================================
'''
Path setting
'''

ap = argparse.ArgumentParser()
ap.add_argument("target", nargs="?", default="Track1")
ap.add_argument("--no-viz", action="store_true", help="run without the Open3D window")
ap.add_argument("--no-imu", action="store_true", help="do not use the IMU rotation prior")
ap.add_argument("--no-loop", action="store_true", help="odometry only (for ablation)")
ap.add_argument("--no-ground-scale", action="store_true", help="do not rescale each cloud to a constant camera height")
ap.add_argument("--scale-win", type=int, default=9, help="median window [frames] of the camera-height estimate")
ap.add_argument("--odom", choices=["pnp", "gicp"], default="pnp",
                help="pnp: image features + cloud depth (GICP only as fallback); gicp: scan-to-submap only")
ap.add_argument("--loop", choices=["visual", "gicp"], default="visual")
ap.add_argument("--out", default="submission.csv")
args = ap.parse_args()

target = args.target
data = f"{target}/data"
files = load_folder(f"{data}/raw_ply/")
os.makedirs(f"{target}/result", exist_ok=True)

# Given data (camera intrinsics; the IMU in data/imu.csv is not used by the starter)
K, img_size = load_camera_info(f"{data}/camera_info.json")

# Initialization
icp = ICP()
submap_mgr = SubmapManager()
kf_manager = KeyframeManager()
pg = PoseGraph()
vm = VisualMatcher(K)
lc = LoopClosure(icp, matcher=vm if args.loop == "visual" else None)
use_visual = args.odom == "pnp" or args.loop == "visual"
image_of = lambda path: path.replace("raw_ply", "images").replace(".ply", ".jpg")
kf_frame = None      # image features of the last keyframe
n_pnp = n_gicp = 0
viz = None if args.no_viz else Visualizer()

# IMU attitude -> relative camera rotation between consecutive images (initial guess of the scan matching)
imu_rot = None
if not args.no_imu and os.path.exists(f"{data}/imu.csv"):
    import json
    R_cam_imu = json.load(open(f"{data}/camera_info.json")).get("R_cam_imu")
    if R_cam_imu is not None:
        imu_rot = ImuRotation(load_imu(f"{data}/imu.csv"), R_cam_imu)
# Per-image scale correction: the camera height above the road is constant in reality, so the
# height measured in each predicted cloud tells how much that cloud is scaled.
cloud_scale = np.ones(len(files))
if not args.no_ground_scale:
    from scipy.ndimage import median_filter
    from utils.ground import ground_height
    cache = f"{target}/result/ground_h.npy"
    if os.path.exists(cache) and len(np.load(cache)) == len(files):
        H = np.load(cache)
    else:
        H = np.array([ground_height(load_ply(p))[0] for p in files])
        np.save(cache, H)
    good = np.isfinite(H)
    H = np.interp(np.arange(len(H)), np.where(good)[0], H[good])
    H = median_filter(H, size=args.scale_win, mode="nearest")
    cloud_scale = np.median(H) / H
    print(f"ground scale: camera height {H.min():.2f}..{H.max():.2f} (median {np.median(H):.2f})")

stamp = lambda path: os.path.basename(path).replace(".ply", "")
prev_delta = np.eye(4)   # last frame-to-frame motion (constant-velocity model)
fit_log = []

pg.add_prior()

# The trajectory starts at the identity (the first camera); the score aligns it to GT with Sim(3)
prev_pose = np.eye(4)
poses = []
timestamps = []

# Visualization & Interpolation state
full_map = np.zeros((0, 3))
kf_indices = []
kf_poses_list = []
last_kf_pose = np.eye(4)
current_kf_idx = 0

# main loop
for i, f in enumerate(files):
    raw = load_ply(f)
    pts = preprocess(raw * cloud_scale[i])
    frame = vm.extract(image_of(f), raw, cloud_scale[i]) if use_visual else None

    # Initial frame
    if i == 0:
        poses.append(prev_pose)
        timestamps.append(os.path.basename(f).replace(".ply", ""))
        submap_mgr.add_keyframe(pts, prev_pose)
        lc.add_keyframe(pts, prev_pose, frame)
        kf_frame = frame

        kf_indices.append(i)
        kf_poses_list.append(prev_pose.copy())
        continue

    submap_pts = submap_mgr.get_latest_submap()
    if submap_pts is None:
        continue

    # ===== Scan-to-Submap (Odometry) =====
    curr_pose = None
    # initial guess: constant-velocity translation, rotation from the IMU when available
    delta = prev_delta.copy()
    if imu_rot is not None:
        delta[:3, :3] = imu_rot.delta(stamp(files[i - 1]), stamp(f))
    init_T = prev_pose @ delta
    T = None
    if args.odom == "pnp":
        # visual odometry: PnP between the last keyframe and this image
        T_kf_cur, n_inl = vm.relative_pose_sym(kf_frame, frame, min_inliers=30)
        if T_kf_cur is not None:
            T = last_kf_pose @ T_kf_cur
            # reject jumps that disagree with the motion model (wrong PnP solution)
            if np.linalg.norm(T[:3, 3] - init_T[:3, 3]) > 1.0:
                T = None
        fit_log.append(n_inl)
    if T is None:
        # scan matching (GICP) of the cloud to the local submap
        T, score = icp.align(pts, submap_pts, init_T)
        n_gicp += 1
        if args.odom == "gicp":
            fit_log.append(icp.fitness)
        if icp.fitness < 0.3:   # scan matching failed: keep the motion-model prediction
            print(f"[warn] frame {i}: low fitness {icp.fitness:.2f}, using the prediction")
            T = init_T
    else:
        n_pnp += 1
    curr_pose = T.copy()
    prev_delta = np.linalg.inv(prev_pose) @ curr_pose

    poses.append(curr_pose)
    timestamps.append(os.path.basename(f).replace(".ply", ""))

    # ===== Keyframe Processing =====
    if kf_manager.is_keyframe(curr_pose):
        current_kf_idx += 1

        # 1. Add Node to Managers
        submap_mgr.add_keyframe(pts, curr_pose)
        lc.add_keyframe(pts, curr_pose, frame)
        kf_frame = frame

        # 2. Add Odom Edge between Keyframes
        rel_pose = np.linalg.inv(last_kf_pose) @ curr_pose
        pg.add_node(current_kf_idx, curr_pose)
        pg.add_odom(current_kf_idx - 1, current_kf_idx, rel_pose, rot_sigma=0.02, trans_sigma=0.1)

        last_kf_pose = curr_pose.copy()
        kf_indices.append(i)

        # 3. Update Visual Map (Voxel downsampled)
        pts_w = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
        full_map = np.concatenate([full_map, pts_w], axis=0)
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(full_map)
        pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=0.5)
        full_map = np.asarray(pcd_tmp.points)

        # ===== Loop Closure =====
        loop_result = None if args.no_loop else lc.detect(submap_mgr)
        if loop_result is not None:
            loop_idx, T_loop, score = loop_result
            if args.loop == "visual":
                pg.add_loop(loop_idx, current_kf_idx, T_loop, rot_sigma=0.03,
                            trans_sigma=0.1 + 0.1 * np.linalg.norm(T_loop[:3, 3]))
            else:
                pg.add_loop(loop_idx, current_kf_idx, T_loop)

            # ===== Optimize & Interpolate =====
            kf_opt_dict = pg.optimize()
            kf_opt_poses = [kf_opt_dict[k] for k in sorted(kf_opt_dict.keys())]

            submap_mgr.update_poses(kf_opt_poses)
            lc.update_poses(kf_opt_poses)

            poses = distribute_pose_corrections(poses, kf_indices, kf_opt_poses)
            # Update states to reflect optimized poses
            curr_pose = kf_opt_poses[-1].copy()
            last_kf_pose = curr_pose.copy()
            kf_poses_list = kf_opt_poses

    prev_pose = curr_pose.copy()

    # ===== Visualization =====
    pts_curr_w = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
    if viz is not None:
        viz.update(pts_curr_w, full_map, poses, kf_poses_list)
    if i % 100 == 0:
        print(f"frame {i}/{len(files)}  keyframes {current_kf_idx}  loops {len(lc.loops)}  match {np.mean(fit_log[-100:]):.2f}  pnp/gicp {n_pnp}/{n_gicp}", flush=True)

# ===== Save Output CSV: camera-to-world pose of every image (the submission) =====
header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
rows = [[str(t)] + T.reshape(-1).tolist() for t, T in zip(timestamps, poses)]

df = pd.DataFrame(rows, columns=header.split(','))
df.to_csv(f"{target}/result/{args.out}", index=False)
print(f"loops: {len(lc.loops)}")
print(f"saved {target}/result/{args.out} ({len(rows)} poses); optional check: python process/build_map.py")
