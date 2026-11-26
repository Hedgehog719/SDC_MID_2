"""Visual SLAM v2 (offline, all frames in one graph).

  A. per image: SIFT keypoints + their 3D points from the image point cloud (regular grid), road-plane height
  B. pairs of images -> PnP in both directions (relative pose) + relative scale of the two clouds
     - odometry pairs: (i, i+1), (i, i+2), (i, i+3), (i, i+5), (i, i+8)
     - loop pairs: images of other laps near the current estimate, verified by PnP inliers
  C. scale graph: one log-scale per image from all pair ratios + the road-height prior (linear least squares)
  D. pose graph: every image is a node, every verified pair is a robust between factor (GTSAM, LM)
     + gyro rotation factors (raw rates integrated, bias from still periods, extrinsic refined)
     + car constraint (the rear axle does not slip sideways)
     + VO (default on, --no-vo): translation-direction factors from image geometry only (gyro-accurate rotation
       + epipolar constraint on all keypoints, no cloud depth), constant offset to the trajectory removed
  D/B are repeated once with the optimised poses to find more loop pairs.
  E. --height-scale N (default 5): triangulate image tracks, measure the camera height above the triangulated
     road, rescale the clouds so that it is constant, solve D again.  Only the part of the height deviation
     that all passes of a place share is corrected (--height-lap-gain 0): the loops already make the laps agree,
     a lap's own deviation is mostly noise.  The plain pose graph is --height-scale 0; before v10 the default
     was --height-scale 1 --height-lap-gain none --no-vo.
  F. --ba (default off): bundle adjustment with reprojection factors (made the public score worse).

The matching runs on the GPU when there is one and with the same code on the CPU otherwise (slower, float32
instead of float16 descriptors in stage E, so the result can differ slightly); <out>.run.json records the
device, the arguments, the stages that ran and the cache files used.  Caches are keyed by a hash of every
parameter they depend on.

Usage: python slam_v2.py [Track1] [--out submission.csv] [--holdout-lap k] [--height-scale 0] [--ba] [--cpu]
"""
import sys
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import cv2
import gtsam
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import median_filter
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from utils.ground import ground_height
from utils.io import load_camera_info, load_folder, load_ply
from utils.visual import Frame, VisualMatcher
from utils import nhc
from utils import ba
from utils import vodir
from utils.gpu_match import GpuMatcher, HAS_CUDA

ap = argparse.ArgumentParser()
ap.add_argument("target", nargs="?", default="Track1")
ap.add_argument("--out", default="submission.csv")
ap.add_argument("--holdout", type=float, default=0.0, help="fraction of loop pairs kept out of the optimisation (validation)")
ap.add_argument("--holdout-lap-starts", default=None,
                help="validation: first frame of every lap for --holdout-lap, e.g. 0,473,830 (default: from the heading of "
                     "the odometry, which then depends on the odometry's own drift)")
ap.add_argument("--holdout-lap", type=int, default=-1,
                help="validation: do not use any loop pair that touches this lap (0-based); the lap is then only "
                     "held by odometry and its loop pairs measure the drift over a whole lap")
ap.add_argument("--loop-keep", type=int, default=0, help="keep only the K best loop pairs (most inliers) of every image, 0 = all")
ap.add_argument("--gyro-steps", default="1,10,100", help="frame steps of the gyro rotation factors")
ap.add_argument("--gyro-growth", type=float, default=0.0, help="gyro sigma grows with step**growth")
ap.add_argument("--gyro-extrinsic", choices=["full", "tilt", "none"], default="full",
                help="refinement of R_cam_imu from the visual rotations: full rotation, only the part that is "
                     "observable (not about the mean turning axis), or none")
ap.add_argument("--no-loop", action="store_true")
ap.add_argument("--no-scale-graph", action="store_true", help="scale from the road height only")
ap.add_argument("--ground-sigma", type=float, default=0.05)
ap.add_argument("--zmax", type=float, default=20.0, help="only keypoints whose cloud depth is below this are used [m]")
ap.add_argument("--no-scale-bias", action="store_true", help="do not estimate the forward-motion bias of the scale ratios")
ap.add_argument("--loop-inliers", type=int, default=50)
ap.add_argument("--pnp-cov", type=float, default=0.0,
                help="use the PnP geometry for the noise of every cloud pair: 6x6 covariance of the relative pose from the inliers' "
                     "reprojection Jacobian (along-track loose where the scene is a wall ahead, tight with near points on both sides), "
                     "inflated by this factor because the steps 1,2,3,5,8 share their points; 0 = off (isotropic 0.02 + 0.05|t|)")
ap.add_argument("--pnp-cov-floor", default="0.02,0.005", help="... minimum sigmas added: rotation [deg], translation [units]")
ap.add_argument("--pnp-cov-scale", default="0.05,0.01",
                help="... cloud-depth part of the translation covariance: sigma along the translation (the cloud scale, fraction of |t|) "
                     "and perpendicular to it (direction, fraction of |t|); the pixel part comes from the PnP geometry")
ap.add_argument("--loop-radii", default="12,5",
                help="search radii [m] of the loop-candidate rounds (each round searches around the current estimate).  Track02: "
                     "the start and the end of the drive are the same place but 15 m apart in the loop-free estimate -> 30,12,5")
ap.add_argument("--rev-loops", type=int, default=0,
                help="rounds of loops between passes in OPPOSITE directions from point-cloud registration (utils/revloop.py; "
                     "Track02 drives most roads both ways), after the image loops; 0 = off")
ap.add_argument("--loop-seg", action="store_true",
                help="opposite-direction loops use the SegFormer segmentation (<track>/seg.npz): no cloud points on cars / people, "
                     "markings only on the road")
ap.add_argument("--rev-keep-cars", action="store_true",
                help="with --loop-seg: keep vehicles in the clouds of the opposite-direction GICP (parked cars are static landmarks); markings stay road-only")
ap.add_argument("--dash-ruler", type=float, default=0.0,
                help="yellow dashed centre line as a ruler (utils/dashruler.py): consecutive dash-edge crossings are one unknown "
                     "period apart; factor sigma as a fraction of the period (e.g. 0.03); 0 = off")
ap.add_argument("--dash-robust", action=argparse.BooleanOptionalAction, default=True,
                help="Huber kernel on the dash-ruler factors (--no-dash-robust: plain least squares, so that a 4-5 %% short stretch is not "
                     "down-weighted)")
ap.add_argument("--dash-cruise", type=float, default=0.0,
                help="use only dash periods whose speed changes by less than this fraction (braking nods the car; 0 = all)")
ap.add_argument("--line-straight", type=float, default=0.0,
                help="yellow centre line straight in the world on straight stretches (utils/dashruler.line_points_factor): sigma of "
                     "the middle point's distance from the chord [units]; removes a common bend of both directions; 0 = off")
ap.add_argument("--dash-scale", type=float, default=0.0,
                help="dash periods of the yellow centre line (utils/dashruler.py) as constraints of the SCALE graph: the cloud scales of "
                     "the steps of one period make one unknown length; sigma in log units (e.g. 0.02); 0 = off")
ap.add_argument("--bev-loops", type=int, default=0,
                help="rounds of loops between opposite passes from road markings seen from above (utils/bevloop.py): forward, "
                     "lateral and yaw in the road plane; after --rev-loops.  0 = off")
ap.add_argument("--bev-pitch-cal", action="store_true", help="... calibrate the BEV pitch from lane-line parallelism (utils/bevloop.calibrate_pitch)")
ap.add_argument("--bev-roll", action="store_true",
                help="... BEV on the road plane rolled per frame (camera mounting roll + road cross slope, from the cloud points the "
                     "segmentation calls road; bevloop.road_roll)")
ap.add_argument("--bev-ahead", default="14", help="... distances between the facing cameras [camera heights], e.g. 10,14,18")
ap.add_argument("--bev-sigma", default="0.5,0.2,0.7", help="... their sigmas: forward, lateral [traj units], yaw [deg]")
ap.add_argument("--bev-unit-scale", type=float, default=1.0,
                help="... multiply the self-calibrated BEV camera height [trajectory units] by this: the BEV pairs measure the distance between "
                     "facing cameras of opposite passes (~16 units) in camera heights, so the unit sets the along-road placement of the two "
                     "passes (1 %% of the unit = ~0.16 units of relative slide); 1.0 = unchanged")
ap.add_argument("--bev-center-fix", action="store_true",
                help="... BEV factors: apply the yaw correction about b's centre as register() measures it (default off = the old factor, "
                     "which also turned the baseline about a and added |t| * dyaw sideways)")
ap.add_argument("--bev-audit", action="store_true",
                help="... diagnostics only (the solution is unchanged): pairs that enter / stay / leave the graph per round, and fixed "
                     "pair sets re-registered under every round's poses (bevloop.audit -> result/bevaudit_<out>.npz)")
ap.add_argument("--rev-sigma", default="1.5,0.4,0.4,8.0", help="... their sigmas: rotation [deg], lateral, vertical, along the road [m]")
ap.add_argument("--loop-retrieval", type=int, default=0,
                help="place recognition independent of the estimate: every N-th image is matched (descriptors, GPU) against every "
                     "N-th image more than 150 frames away; pairs with many mutual matches (and their neighbours) go to the PnP "
                     "verification.  Finds revisits that drift put far apart (Track02: start = end).  0 = off")
ap.add_argument("--loop-retrieval-min", type=int, default=100, help="... mutual descriptor matches needed to try a pair")
ap.add_argument("--loop-after-height", type=float, default=0.0,
                help="after the height correction: search loop candidates once more within this radius [m] around the corrected "
                     "trajectory and solve again with the same cloud scales (0 = off)")
ap.add_argument("--no-gyro", action="store_true", help="do not use the gyroscope rotation constraints")
ap.add_argument("--gyro-offset", type=float, default=0.0,
                help="seconds added to the IMU timestamps (camera-IMU time offset; process/time_offset.py estimates it)")
ap.add_argument("--gyro-sigma", type=float, default=0.2, help="sigma of a gyro relative rotation [deg]")
ap.add_argument("--no-nhc", action="store_true", help="do not use the car (non-holonomic) motion constraint")
ap.add_argument("--nhc-height", type=float, default=0.0,
                help="height of the camera above the point that cannot move sideways [m]; negative = calibrate it")
ap.add_argument("--nhc-sigma", type=float, default=0.3, help="direction sigma of the car constraint [deg]")
ap.add_argument("--vo", action=argparse.BooleanOptionalAction, default=True,
                help="translation-direction factors from image geometry only (known rotation + epipolar "
                     "constraint on all keypoints, no cloud depth; see utils/vodir.py); --no-vo turns them off")
ap.add_argument("--vo-steps", default="3,5,8", help="frame steps of the VO direction pairs")
ap.add_argument("--vo-sigma", type=float, default=0.5, help="lateral sigma of a VO direction [deg]")
ap.add_argument("--vo-sigma-up", type=float, default=0.3, help="vertical sigma of a VO direction [deg]")
ap.add_argument("--vo-thr", type=float, default=1.0, help="essential-matrix RANSAC threshold [px]")
ap.add_argument("--k1", type=float, default=0.0,
                help="radial distortion of the images (OpenCV k1, given K): the keypoints are undistorted and their cloud points "
                     "re-cast along the undistorted rays with the same depth z (the clouds are pinhole back-projections of the "
                     "distorted pixels). Track01: about -0.03 (the image rotations turn 1.1 %% too little without it; NOTES.md)")
ap.add_argument("--cam", default=None,
                help="true camera 'fx,fy,cx,cy,k1,k2,p1,p2' (OpenCV; e.g. self-calibrated by process/sfm_ba.py): the keypoints are "
                     "undistorted with it and expressed as pixels of the given pinhole K, their cloud points re-cast along those rays "
                     "(same depth z).  Generalises --k1 (not both)")
ap.add_argument("--vo-rot", action=argparse.BooleanOptionalAction, default=None,
                help="with --vo: also relative-rotation factors from the essential matrix of the VO pairs. "
                     "Default: on without the gyro (Track02), off with it")
ap.add_argument("--vo-rot-fscale", type=float, default=1.0,
                help="focal length factor used only for the VO rotations (Track01: the yaw of the essential matrix is 1.1 %% short with the given fx; x0.987 removes it)")
ap.add_argument("--vo-rot-bias", type=lambda x: x if x == "auto" else float(x), default=None,
                help="yaw bias of the image rotations per metre of camera motion [deg/m], removed from the VO rotations "
                     "(Track01 with --k1 -0.03: -0.028 ~ -0.035 against the gyro, -0.032 from its loops).  'auto': "
                     "estimate it from the loop pairs (yaw drift of the loop-free trajectory per metre), then solve again.  "
                     "Default: auto without the gyro, 0 with it")
ap.add_argument("--vo-rot-bias-fallback", type=float, default=-0.032,
                help="--vo-rot-bias auto without enough loop pairs (a route without revisits): use this value [deg/m] "
                     "(Track01's camera; single-pass simulation on Track01: 0.134 -> 0.088 m)")
ap.add_argument("--speed-sigma", type=float, default=0.0,
                help="speed smoothness: the speed of consecutive frame steps (step length / timestamp difference) may change "
                     "by this acceleration sigma [m/s^2] (0 = off)")
ap.add_argument("--road-seg", action="store_true",
                help="road points for the height correction must also be road in the SegFormer segmentation (<track>/seg.npz, process/segment_road.py)")
ap.add_argument("--seg-objects", action="store_true", help="drop keypoints on road users (person, rider, vehicles) in that segmentation")
ap.add_argument("--depth-prior", default=None,
                help="per-frame cloud scale from an independent metric depth network (<track>/depth_<name>.npz of "
                     "process/depth_scale_check.py, e.g. da2 = Depth Anything V2 metric outdoor): prior on the log cloud scales")
ap.add_argument("--depth-prior-mode", choices=["add", "replace"], default="add", help="add to / replace the cloud's own camera-height prior")
ap.add_argument("--depth-robust", action="store_true", help="Huber weights (IRLS) on the depth-network scale prior too")
ap.add_argument("--depth-sigma", type=float, default=None,
                help="sigma of the depth-network scale prior (log).  Default: 0.05 with the gyro, 0.10 without (Track01 holdout sweeps)")
ap.add_argument("--planar-sigma", type=float, default=0.0,
                help="flat-route prior: every camera within this distance [m] of the best-fit plane of the current trajectory (0 = off)")
ap.add_argument("--gravity-sigma", type=float, default=0.0,
                help="gravity from vertical lines (<track>/gravity.npz of process/gravity_vp.py): attitude factor sigma [deg] (0 = off)")
ap.add_argument("--frames", type=int, default=0, help="use only the first N frames (simulation of a route without revisits)")
ap.add_argument("--vo-rot-sigma", type=float, default=0.04, help="sigma of a VO rotation for step 3 [deg per axis], grows with (step/3)^0.6")
ap.add_argument("--vo-loops", action="store_true", help="with --vo: also direction factors for the loop pairs (between laps)")
ap.add_argument("--vo-loop-debias", action="store_true", help="remove the constant offset of the loop directions separately")
ap.add_argument("--vo-loop-min", type=float, default=1.0, help="... only for loop pairs further apart than this [m]")
ap.add_argument("--ba", action="store_true", help="run the bundle adjustment (off by default: it made the public score worse)")
ap.add_argument("--ba-sigma", type=float, default=1.0, help="keypoint sigma of the bundle adjustment [px]")
ap.add_argument("--ba-rounds", type=int, default=3)
ap.add_argument("--no-height", action="store_true", help="no constant-camera-height factors in the bundle adjustment")
ap.add_argument("--height-scale", type=int, default=None,
                help="iterations of: rescale the clouds so that the camera height above the TRIANGULATED road is "
                     "constant, then re-solve the pose graph (no reprojection factors). Default 5 with the gyro, "
                     "4 without (with --height-gain 0.5: without the gyro full corrections diverge; NOTES.md)")
ap.add_argument("--height-scale-all", action="store_true", help="also use the heights of frames where the car hardly moves")
ap.add_argument("--height-min-step", type=float, default=0.5, help="'moving' = step above this fraction of the usual step")
ap.add_argument("--height-gain", type=float, default=None,
                help="exponent of the scale correction (1 = full).  Default: 1 with the gyro, 0.5 without (with 4 iterations)")
ap.add_argument("--height-lap-gain", type=lambda x: None if x.lower() == "none" else float(x), default=0.0,
                help="split the height correction into the part shared by all passes of the same place (gain "
                     "--height-gain) and each lap's own deviation from it (this gain; default 0 = only the shared "
                     "part); 'none' = not split, the whole deviation with --height-gain (before v10)")
ap.add_argument("--height-radius", type=float, default=4.0, help="road points within this distance of the camera for its height [m]")
ap.add_argument("--height-lat", type=float, default=None, help="... and only within this lateral distance [m] (own lane)")
ap.add_argument("--height-place-radius", type=float, default=3.0, help="passes of the same place: within this distance [m]")
ap.add_argument("--height-turn-detrend", action="store_true",
                help="with --height-lap-gain: do not correct the part of the shared height deviation that follows the "
                     "turn rate (in turns the triangulated road gives a lower camera, independent of the cloud scale)")
ap.add_argument("--height-freeze", type=float, default=None,
                help="with --height-lap-gain: from the 2nd iteration on, stop correcting places whose shared height "
                     "deviation did not shrink below this fraction of the previous one (it no longer reacts to the scale)")
ap.add_argument("--dump-height", action="store_true", help="save heights / scales / poses of every height-scale iteration (<out>.hs.npz)")
ap.add_argument("--height-scale-win", type=int, default=41, help="median window [frames] of that height profile")
ap.add_argument("--height-per-frame", type=int, default=8, help="camera-height factors per image")
ap.add_argument("--ba-between", type=float, default=1.0,
                help="in the bundle adjustment, multiply the translation sigma of the cloud (PnP) pair factors by this")
ap.add_argument("--height-sigma", type=float, default=0.03, help="sigma of one camera-height factor [m]")
ap.add_argument("--ba-max-obs", type=int, default=900000)
ap.add_argument("--cpu", action="store_true", help="match on the CPU (slow)")
ap.add_argument("--procs", type=int, default=os.cpu_count())
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mask-v-max", type=float, default=0.0, help="drop keypoints with raw pixel row v > this (e.g. 850 to mask car hood; 0 = off)")
ap.add_argument("--rs-readout-ms", type=float, default=0.0, help="top-to-bottom CMOS rolling-shutter deskew [ms] using 100 Hz IMU gyro (0 = off)")
ap.add_argument("--rs-6dof-traj", default="", help="optional trajectory CSV for 3D translational RS deskew (v_cam * dt_v) on top of --rs-readout-ms (empty = off)")
args = ap.parse_args()
args.no_ba = not args.ba      # internal name; the bundle adjustment only runs with --ba

target = args.target
data = f"{target}/data"
HAS_GYRO = not args.no_gyro and os.path.exists(f"{data}/imu.csv")
if args.vo_rot is None:
    args.vo_rot = not HAS_GYRO
if args.height_scale is None:
    args.height_scale = 5 if HAS_GYRO else 4
if args.height_gain is None:
    args.height_gain = 1.0 if HAS_GYRO else 0.5
if args.vo_rot_bias is None:
    args.vo_rot_bias = 0.0 if HAS_GYRO else "auto"
if args.depth_sigma is None:
    args.depth_sigma = 0.05 if HAS_GYRO else 0.10
res_dir = f"{target}/result"

def _key(**kw):
    """short hash of everything a cache depends on: a changed parameter can never reuse stale results"""
    import hashlib
    return hashlib.md5(repr(sorted(kw.items())).encode()).hexdigest()[:8]


FEAT = dict(n_features=6000, contrast=0.02, max_depth=30.0, version=1)                # bump version when extract() changes
PAIR = dict(ratio=0.75, pnp_px=3.0, pnp_iters=300, rot_check_deg=3.0, version=1)      # ... when pair() changes
feat_dir = f"{res_dir}/feat_{_key(**FEAT)}"
GEOM = {**(dict(k1=args.k1) if args.k1 else {}), **(dict(cam=args.cam) if args.cam else {}), **(dict(seg_obj=1) if args.seg_objects else {}), **(dict(vmax=args.mask_v_max) if args.mask_v_max > 0 else {}), **(dict(rs=args.rs_readout_ms, rso=round(args.gyro_offset, 4), rs6=os.path.basename(args.rs_6dof_traj) if args.rs_6dof_traj else "") if args.rs_readout_ms != 0.0 else {})}      # geometry of the keypoints (part of the pair / VO cache keys)
os.makedirs(feat_dir, exist_ok=True)
files = load_folder(f"{data}/raw_ply/")
if args.frames:
    files = files[:args.frames]
stamps = [os.path.basename(f)[:-4] for f in files]
TSEC = np.array([int(t) for t in stamps]) * 1e-9
N = len(files)
K, _ = load_camera_info(f"{data}/camera_info.json")
vm = VisualMatcher(K, n_features=FEAT["n_features"], contrast=FEAT["contrast"], max_depth=FEAT["max_depth"],
                   ratio=PAIR["ratio"], reproj_px=PAIR["pnp_px"])
F = [None] * N          # features with a usable cloud depth, for the PnP stages
F_ALL = [None] * N      # every keypoint (kp, desc float16), for the bundle adjustment
ODOM_STEPS = (1, 2, 3, 5, 8)


# ---------------------------------------------------------------- A. features
def extract(i):
    cv2.setNumThreads(1)
    cache = f"{feat_dir}/{stamps[i]}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return z["kp"], z["desc"], z["xyz"].astype(np.float64), float(z["h"])
    raw = load_ply(files[i])
    fr = vm.extract(f"{data}/images/{stamps[i]}.jpg", raw, 1.0, keep_all=True)
    h = ground_height(raw)[0]
    kp, desc, xyz = fr.kp.astype(np.float32), fr.desc.astype(np.float16), fr.xyz.astype(np.float32)
    np.savez(cache, kp=kp, desc=desc, xyz=xyz, h=h)
    return kp, desc, xyz.astype(np.float64), h


# ---------------------------------------------------------------- B. pairs
def pnp(obj, img):
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=PAIR["pnp_px"], iterationsCount=PAIR["pnp_iters"],
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 12:
        return None, None, np.zeros(0, int)
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
    return cv2.Rodrigues(rvec)[0], tvec[:, 0], inl


MATCHER = None   # GpuMatcher (set in main) or None -> OpenCV brute force


def pair(ab):
    """Relative pose of image b in image a from both clouds, in the raw units of each cloud.
    returns (a, b, R_ab, t_ab in a-units, n1, t_ab in b-units, n2, log(k_a / k_b)) or None"""
    a, b = ab
    A, B = F[a], F[b]
    cv2.setRNGSeed(a * 100003 + b)      # RANSAC must not depend on which thread ran which pair
    m = MATCHER.match(a, b) if MATCHER is not None else vm.match(A, B)
    if len(m) < 30:
        return None
    XA, XB = A.xyz[m[:, 0]], B.xyz[m[:, 1]]
    R1, t1, i1 = pnp(XA, B.kp[m[:, 1]].astype(np.float64))      # X_b = R1 X_a + t1 (a-units)
    R2, t2, i2 = pnp(XB, A.kp[m[:, 0]].astype(np.float64))      # X_a = R2 X_b + t2 (b-units)
    if R1 is None or R2 is None:
        return None
    # the two directions must describe the same rotation
    if np.degrees(np.linalg.norm(R.from_matrix(R1 @ R2).as_rotvec())) > PAIR["rot_check_deg"]:
        return None
    # relative scale of the two clouds from distances between the same physical points
    both = np.intersect1d(i1, i2)
    logr = np.nan
    if len(both) >= 20:
        rng = np.random.default_rng(a * 100003 + b)
        p = rng.integers(0, len(both), (3000, 2))
        dA = np.linalg.norm(XA[both[p[:, 0]]] - XA[both[p[:, 1]]], axis=1)
        dB = np.linalg.norm(XB[both[p[:, 0]]] - XB[both[p[:, 1]]], axis=1)
        g = (dA > 1.0) & (dB > 1.0)
        if g.sum() >= 100:
            logr = float(np.median(np.log(dB[g] / dA[g])))      # k_a dA = k_b dB
    Rab = R.from_rotvec((R.from_matrix(R1.T).as_rotvec() + R.from_matrix(R2).as_rotvec()) / 2).as_matrix()
    if args.pnp_cov > 0:
        return a, b, Rab, -R1.T @ t1, len(i1), t2, len(i2), logr, pnp_covariance(R1, t1, XA[i1], B.kp[m[i1, 1]].astype(np.float64), a * 100003 + b)
    return a, b, Rab, -R1.T @ t1, len(i1), t2, len(i2), logr


def pnp_covariance(R1, t1, X, uv, seed, nmax=400, eps=1e-6):
    """6x6 covariance of the pose of camera b in camera a (gtsam right-perturbation: T_ab Exp(xi), xi = (w, v), a-units)
    from the reprojection of the inlier points X (camera a) into image b: sigma_px^2 (J^T J)^-1"""
    if len(X) > nmax:
        sel = np.random.default_rng(seed).choice(len(X), nmax, replace=False); X, uv = X[sel], uv[sel]
    Rb, tb = R1.T, -R1.T @ t1                         # camera b in camera a

    def res(xi):
        Rp = Rb @ cv2.Rodrigues(xi[:3])[0]; tp = tb + Rb @ xi[3:]
        Xb = (X - tp) @ Rp                            # = Rp^T (X - tp)
        pr = Xb @ K.T
        return (pr[:, :2] / pr[:, 2:3] - uv).ravel()
    r0 = res(np.zeros(6))
    J = np.empty((len(r0), 6))
    for j in range(6):
        d = np.zeros(6); d[j] = eps
        J[:, j] = (res(d) - res(-d)) / (2 * eps)
    sig = max(1.4826 * np.median(np.abs(r0 - np.median(r0))), 0.3)
    return (sig ** 2 * np.linalg.inv(J.T @ J + 1e-12 * np.eye(6))).astype(np.float32)


PAIR_FILE = [None]
PAIR_CACHE = {}   # (a, b) -> result or None; saved to disk so that re-runs only match new pairs


def run_pairs(todo, min_inl):
    import pickle
    cache_file = f"{res_dir}/pairs_{_key(zmax=args.zmax, feat=_key(**FEAT), **PAIR, **GEOM, **(dict(cov=1) if args.pnp_cov > 0 else {}))}.pkl"
    PAIR_FILE[0] = cache_file
    if not PAIR_CACHE and os.path.exists(cache_file):
        PAIR_CACHE.update(pickle.load(open(cache_file, "rb")))
    new = [p for p in todo if p not in PAIR_CACHE]
    if new:
        # the GPU does the descriptor matching, threads do the PnP (OpenCV releases the GIL)
        with ThreadPoolExecutor(args.procs if MATCHER is not None else 4) as ex:
            for p, r in zip(new, ex.map(pair, new)):
                PAIR_CACHE[p] = r
        tmp = f"{cache_file}.{os.getpid()}.tmp"
        pickle.dump(PAIR_CACHE, open(tmp, "wb")); os.replace(tmp, cache_file)
    out = {}
    for p in todo:
        r = PAIR_CACHE[p]
        if r is not None and min(r[4], r[6]) >= min_inl:
            out[p] = r
    return out


def pair_T(r, k):
    """metric T_a_b of a pair with the cloud scales k"""
    a, b, Rab, ta, n1, tb, n2 = r[:7]
    T = np.eye(4)
    T[:3, :3] = Rab
    T[:3, 3] = (n1 * k[a] * ta + n2 * k[b] * tb) / (n1 + n2)
    return T


# ---------------------------------------------------------------- C. scale graph
def solve_scales(pairs, H, ground_sigma, pair_sigma=0.01, bias=True):
    """log-scale x_i of every cloud (metric = k_i * raw, k_i = exp(x_i)).

    measurements:  ratio of a pair      l_ab = x_a - x_b + beta * forward_ab
                   road height (prior)  x_i  = log(median(H) / H_i)
    beta absorbs a systematic effect of the clouds: the same far points get a different size when the
    camera moves towards them (far depth is compressed), which would otherwise add up along the route.
    """
    prior = np.log(np.median(H) / H)
    DPR = None
    if args.depth_prior:
        ls_ = np.zeros(N)
        for nm in args.depth_prior.split("+"):          # several networks: mean of their log scales
            z_ = np.load(f"{target}/depth_{nm}.npz")
            l1 = np.full(N, np.nan); sel_ = z_["frames"] < N; l1[z_["frames"][sel_]] = np.log(z_["s_all"][sel_])
            ls_ += (l1 - np.nanmedian(l1)) / len(args.depth_prior.split("+"))
        ok_ = np.isfinite(ls_); ls_ = np.interp(np.arange(N), np.where(ok_)[0], ls_[ok_])
        DPR = ls_ - np.median(ls_) + np.median(prior)          # same gauge as the camera-height prior
        if args.depth_prior_mode == "replace":
            prior = DPR; DPR = None
    if pairs is None:
        return np.exp(median_filter(prior, size=9, mode="nearest"))
    E = [(a, b, r[7], r[3][2]) for (a, b), r in pairs.items() if np.isfinite(r[7])]
    ia = np.array([e[0] for e in E]); ib = np.array([e[1] for e in E]); l = np.array([e[2] for e in E])
    fwd = np.array([e[3] for e in E]) * np.exp(prior[ia])       # forward motion of b seen from a [m]
    w = np.ones(len(E)); wp = np.ones(N); wd = np.ones(N)
    # dash periods (--dash-scale): one unknown log period z (column N+1); a period spanning steps j has length
    # sum_j exp(x_j) r_j  ->  sum_j w_j x_j - z = -log sum_j r_j   (w_j = r_j / sum r_j, first order)
    DROWS = []
    if DASH_SPANS and args.dash_scale > 0:
        steps = {i: np.linalg.norm(r[3]) for (i, j), r in pairs.items() if j == i + 1}
        for ta, tb in DASH_SPANS:
            js = list(range(int(round(ta)), int(round(tb))))
            rj = np.array([steps.get(j, np.nan) for j in js])
            if len(js) < 3 or not np.isfinite(rj).all():
                continue
            DROWS.append((np.array(js), rj / rj.sum(), -np.log(rj.sum())))
    x = np.r_[prior, 0.0, 0.0]
    for _ in range(8):   # IRLS (Huber)
        rows = np.r_[np.arange(len(E)), np.arange(len(E)), np.arange(len(E)), len(E) + np.arange(N)]
        cols = np.r_[ia, ib, np.full(len(E), N), np.arange(N)]
        vals = np.r_[w / pair_sigma, -w / pair_sigma, (w * fwd / pair_sigma) if bias else np.zeros(len(E)), wp / ground_sigma]
        rhs = np.r_[w * l / pair_sigma, wp * prior / ground_sigma]
        nrow = len(E) + N
        if DPR is not None:          # independent depth network: a second prior on every cloud scale
            rows = np.r_[rows, nrow + np.arange(N)]; cols = np.r_[cols, np.arange(N)]
            vals = np.r_[vals, wd / args.depth_sigma]; rhs = np.r_[rhs, wd * DPR / args.depth_sigma]
            nrow += N
        for js, wj, rhs_ in DROWS:
            sd = args.dash_scale
            rows = np.r_[rows, np.full(len(js) + 1, nrow)]; cols = np.r_[cols, js, N + 1]
            vals = np.r_[vals, wj / sd, -1.0 / sd]; rhs = np.r_[rhs, rhs_ / sd]; nrow += 1
        A = sp.csr_matrix((vals, (rows, cols)), shape=(nrow, N + 2))
        x = lsqr(A, rhs, x0=x, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
        res = x[ia] - x[ib] + x[N] * fwd - l
        re = np.abs(res) / pair_sigma
        rp = np.abs(x[:N] - prior) / ground_sigma
        w = np.sqrt(np.where(re < 1.5, 1.0, 1.5 / re)); wp = np.sqrt(np.where(rp < 1.5, 1.0, 1.5 / rp))
        if DPR is not None and args.depth_robust:
            rd = np.abs(x[:N] - DPR) / args.depth_sigma
            wd = np.sqrt(np.where(rd < 1.345, 1.0, 1.345 / rd))
    if DROWS:
        per = np.array([np.exp(x[js]) @ (wj * np.exp(-rhs_)) for js, wj, rhs_ in DROWS])
        print(f"  scale graph: {len(DROWS)} dash periods, spread {100*1.4826*np.median(np.abs(per/np.median(per)-1)):.1f} %")
    beta, x = x[N], x[:N]
    print(f"  scale graph: {len(E)} ratios, residual median {np.median(np.abs(res)):.4f}, bias {beta*100:+.2f} %/m, "
          f"|x - ground prior| median {np.median(np.abs(x-prior)):.3f}, k {np.exp(x).min():.2f}..{np.exp(x).max():.2f}")
    return np.exp(x - np.median(x))


# ---------------------------------------------------------------- D. pose graph
GYRO_STEPS = tuple(int(x) for x in args.gyro_steps.split(","))


def speed_factor(i, dt1, dt2, sigma):
    """error = speed of step i -> i+1 minus speed of step i-1 -> i  (keys i-1, i, i+1; positions only)"""
    def err(this, values, H):
        p0, p1, p2 = (values.atPose3(this.keys()[j]) for j in range(3))
        a, b, c = p0.translation(), p1.translation(), p2.translation()
        d1, d2 = b - a, c - b
        n1, n2 = max(np.linalg.norm(d1), 1e-9), max(np.linalg.norm(d2), 1e-9)
        u1, u2 = d1 / n1, d2 / n2
        if H is not None:
            z = np.zeros((1, 3))
            H[0] = np.hstack([z, (u1 / dt1)[None] @ p0.rotation().matrix()])
            H[1] = np.hstack([z, (-u2 / dt2 - u1 / dt1)[None] @ p1.rotation().matrix()])
            H[2] = np.hstack([z, (u2 / dt2)[None] @ p2.rotation().matrix()])
        return np.array([n2 / dt2 - n1 / dt1])
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), gtsam.noiseModel.Isotropic.Sigma(1, sigma))
    return gtsam.CustomFactor(noise, [i - 1, i, i + 1], err)


def planar_factor(i, n, m, sigma):
    """error = signed distance of camera i to the plane (normal n through m)"""
    def err(this, values, H):
        p = values.atPose3(this.keys()[0])
        if H is not None:
            H[0] = np.hstack([np.zeros((1, 3)), n[None] @ p.rotation().matrix()])
        return np.array([n @ (p.translation() - m)])
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), gtsam.noiseModel.Isotropic.Sigma(1, sigma))
    return gtsam.CustomFactor(noise, [i], err)


def true_camera():
    """(K, dist) of the lens the poses are built with: --cam if given, else the given K with --k1.  The BEV / ruler
    stages sample the images with the same camera, so poses and markings live in one geometry."""
    if args.cam:
        c_ = [float(x) for x in args.cam.split(",")]
        return np.array([[c_[0], 0, c_[2]], [0, c_[1], c_[3]], [0, 0, 1.0]]), np.array(c_[4:8] + [0.0])
    return K.copy(), np.array([args.k1, 0.0, 0.0, 0.0, 0.0])


def mount_deg():
    return (float(np.degrees(NHC_CAL[0])), float(np.degrees(NHC_CAL[1]))) if NHC_CAL else (-0.14, -1.88)


NHC_CAL = []
DASH_SPANS = []   # (t_a, t_b) fractional frames of consecutive dash-edge crossings, for --dash-scale
EXTRA = []   # factors added to every pose graph from the stage on that found them (opposite-direction loops)
EXTRA_VALUES = {}   # extra variables of those factors (key -> initial numpy vector), e.g. the dash period


def optimise(pairs, k, init, gyro=None, car=None, points=None, heights=None, verbose=False, vo=None, vrot=None):
    """Every input of the graph is an argument (no hidden stage state):
    pairs/k  cloud (PnP) relative poses and the cloud scales that make them metric
    gyro     camera rotations integrated from the gyro, or None
    car      (mount yaw, mount pitch, lever arm) of the car motion constraint, or None
    vo       {(a, b): (unit direction of b in a, inliers)} image-only translation directions, or None
    points   triangulated tracks -> bundle adjustment;  heights = (pairs, h0, up) camera-height factors"""
    R_GYRO, NHC = gyro, car
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    for i in range(N):
        v.insert(i, gtsam.Pose3(init[i]))
    g.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(init[0]), gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-4)))
    for (a, b), r in pairs.items():
        T = pair_T(r, k)
        q = np.clip(np.sqrt(150.0 / min(r[4], r[6])), 0.7, 2.5)     # fewer PnP inliers -> less trusted
        st = (0.02 + 0.05 * np.linalg.norm(T[:3, 3])) * q * (args.ba_between if points is not None else 1.0)
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.ones(3) * 0.01 * q, np.ones(3) * st])
        if args.pnp_cov > 0 and len(r) > 8 and r[8] is not None:
            C_ = r[8].astype(np.float64).copy()
            S_ = np.diag([1, 1, 1, k[a], k[a], k[a]])       # translation in metres of cloud a
            C_ = S_ @ C_ @ S_
            # the clouds' depth error: a wrong cloud scale moves the PnP translation along itself; its direction comes from the
            # image geometry.  Translation of the right perturbation is in frame b: t_b = R_ab^T t_ab
            ss_, sd_ = (float(x) for x in args.pnp_cov_scale.split(","))
            tb_ = T[:3, :3].T @ T[:3, 3]; nt_ = np.linalg.norm(tb_)
            if nt_ > 1e-6:
                u_ = tb_ / nt_
                C_[3:, 3:] += (ss_ * nt_) ** 2 * np.outer(u_, u_) + (sd_ * nt_) ** 2 * (np.eye(3) - np.outer(u_, u_))
            C_ *= args.pnp_cov ** 2
            C_[:3, :3] = np.eye(3) * (0.01 * q) ** 2      # rotation as before (only the translation model changes)
            C_[:3, 3:] = 0.0; C_[3:, :3] = 0.0
            fr_, ft_ = (float(x) for x in args.pnp_cov_floor.split(","))
            C_ += np.diag([np.radians(fr_) ** 2] * 3 + [ft_ ** 2] * 3)
            noise = gtsam.noiseModel.Gaussian.Covariance(C_)
        noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), noise)
        g.add(gtsam.BetweenFactorPose3(a, b, gtsam.Pose3(T), noise))
    if R_GYRO is not None:   # rotation-only constraints from the gyro
        for s_ in GYRO_STEPS:
            sig = np.radians(args.gyro_sigma) * s_ ** args.gyro_growth
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.ones(3) * sig, np.ones(3) * 1e3])
            for i in range(0, N - s_, max(1, s_ // 5)):
                T = np.eye(4); T[:3, :3] = R_GYRO[i].T @ R_GYRO[i + s_]
                g.add(gtsam.BetweenFactorPose3(i, i + s_, gtsam.Pose3(T), noise))
    if NHC is not None:      # the car cannot move sideways (see utils/nhc.py)
        sa = np.radians(args.nhc_sigma)
        for i in range(N - 1):
            d = np.linalg.norm(init[i + 1, :3, 3] - init[i, :3, 3])
            g.add(nhc.factor(i, i + 1, NHC, [0.001 + sa * d, 0.002 + 2 * sa * d]))
    if vo is not None:       # translation direction from image geometry only (utils/vodir.py)
        s0 = np.radians([args.vo_sigma, args.vo_sigma_up])
        for (a, b), (m, n) in vo.items():
            g.add(vodir.factor(a, b, m, s0 * np.clip(np.sqrt(500.0 / n), 0.7, 2.0)))
    if args.planar_sigma > 0:  # flat route: the cameras stay near one plane (fitted to the current trajectory)
        C_ = init[:, :3, 3]; m_ = C_.mean(0); n_ = np.linalg.svd(C_ - m_)[2][2]
        for i in range(N):
            g.add(planar_factor(i, n_, m_, args.planar_sigma))
    if args.gravity_sigma > 0:  # vertical vanishing direction = gravity in the camera (pitch / roll without an IMU)
        GV = np.load(f"{target}/gravity.npz")["up"][:N]
        Rw = init[:, :3, :3]
        okg = np.isfinite(GV[:, 0])
        nw = np.einsum("nij,nj->ni", Rw[okg], GV[okg]).mean(0); nw /= np.linalg.norm(nw)
        dev = np.degrees(np.arccos(np.clip(np.einsum("nij,nj->ni", Rw[okg], GV[okg]) @ nw, -1, 1)))
        use = np.where(okg)[0][dev < 3.0]                    # gate frames whose vertical disagrees with the rest
        for i in use:
            g.add(gtsam.Pose3AttitudeFactor(int(i), gtsam.Unit3(nw), gtsam.noiseModel.Isotropic.Sigma(2, np.radians(args.gravity_sigma)), gtsam.Unit3(GV[i])))
    if args.speed_sigma > 0:  # the car's speed changes smoothly (timestamps of the images)
        for i in range(1, N - 1):
            dt1, dt2 = TSEC[i] - TSEC[i - 1], TSEC[i + 1] - TSEC[i]
            g.add(speed_factor(i, dt1, dt2, args.speed_sigma * 0.5 * (dt1 + dt2)))
    if vrot is not None:     # relative rotations from image geometry only (utils/vodir.py)
        for (a, b), (Rab, n) in vrot.items():
            sig = np.radians(args.vo_rot_sigma) * ((b - a) / 3.0) ** 0.6 * np.clip(np.sqrt(500.0 / n), 0.7, 2.0)
            g.add(vodir.rotation_factor(a, b, Rab, sig))
    for f_ in EXTRA:
        g.add(f_)
    for k_, x_ in EXTRA_VALUES.items():
        v.insert(k_, np.asarray(x_, dtype=float))
    p = gtsam.LevenbergMarquardtParams(); p.setMaxIterations(100)
    if points is not None:      # bundle adjustment: reprojection factors of the triangulated tracks
        L0 = ba.add_factors(g, v, K, points, sigma_px=args.ba_sigma)
        if heights is not None:      # constant camera height above the triangulated road points
            hp, h0, up_c = heights
            for i, j in hp:
                g.add(ba.height_factor(i, L0 + j, up_c, h0, args.height_sigma))
        p.setMaxIterations(30); p.setRelativeErrorTol(1e-4)
        if verbose:
            p.setVerbosityLM("SUMMARY")
    out = gtsam.LevenbergMarquardtOptimizer(g, v, p).optimize()
    for k_ in EXTRA_VALUES:
        EXTRA_VALUES[k_] = out.atVector(k_)
    new = np.array([out.atPose3(i).matrix() for i in range(N)])
    if points is not None:
        return new, [(out.atPoint3(L0 + j), fr, xy, gr) for j, (_, fr, xy, gr) in enumerate(points)]
    return new


def chain(pairs, k):
    """dead reckoning with the shortest available odometry pair"""
    poses = [np.eye(4)]
    for i in range(1, N):
        for s in ODOM_STEPS:
            if i - s >= 0 and (i - s, i) in pairs:
                poses.append(poses[i - s] @ pair_T(pairs[(i - s, i)], k)); break
        else:
            d = np.linalg.inv(poses[-2]) @ poses[-1] if i >= 2 else np.eye(4)
            print(f"  [warn] no odometry pair for frame {i}: constant velocity")
            poses.append(poses[-1] @ d)
    return np.array(poses)


def loop_candidates(poses, radius, min_gap=150, per_frame=10, sep=4, max_heading=40.0):
    P = poses[:, :3, 3]; Z = poses[:, :3, 2]
    tree = cKDTree(P)
    todo = set()
    for a in range(N):
        nb = np.array([b for b in tree.query_ball_point(P[a], radius) if abs(b - a) > min_gap])
        if len(nb) == 0:
            continue
        nb = nb[Z[nb] @ Z[a] > np.cos(np.radians(max_heading))]
        nb = nb[np.argsort(np.linalg.norm(P[nb] - P[a], axis=1))]
        chosen = []
        for b in nb:
            if all(abs(b - c) >= sep for c in chosen):
                chosen.append(b)
            if len(chosen) >= per_frame:
                break
        todo.update((min(a, b), max(a, b)) for b in chosen)
    return sorted(todo)


def place_mean(poses, x, lap, radius=3.0, max_heading=30.0):
    """mean of a per-frame value over all passes of the same place: the frame itself plus the nearest frame of
    every other lap (within radius, similar heading).  Returns (mean, number of passes)."""
    from scipy.spatial import cKDTree
    C, Z = poses[:, :3, 3], poses[:, :3, 2]
    acc, cnt = x.astype(float).copy(), np.ones(len(x))
    for L in np.unique(lap):
        idx = np.where(lap == L)[0]
        d, j = cKDTree(C[idx]).query(C, distance_upper_bound=radius)
        jj = idx[np.minimum(j, len(idx) - 1)]
        ok = (lap != L) & np.isfinite(d) & (np.einsum("ij,ij->i", Z, Z[jj]) > np.cos(np.radians(max_heading)))
        acc[ok] += x[jj[ok]]; cnt[ok] += 1
    return acc / cnt, cnt


def yaw_bias_removed(vrot, P0, beta):
    """remove a yaw bias of beta [deg/m] (about the vertical of the trajectory plane) from relative rotations"""
    C_ = P0[:, :3, 3]
    up_w = np.linalg.svd(C_ - C_.mean(0))[2][2]
    up_w *= np.sign(up_w @ -P0[:, :3, 1].mean(0))         # camera y points down
    up_c = np.einsum("nji,j->ni", P0[:, :3, :3], up_w).mean(0)
    up_c /= np.linalg.norm(up_c)
    return {(a_, b_): (Rab @ R.from_rotvec(-np.radians(beta * np.linalg.norm(C_[b_] - C_[a_])) * up_c).as_matrix(), n_)
            for (a_, b_), (Rab, n_) in vrot.items()}, up_c


def loop_yaw_drift(P0, pairs_, min_len=20.0):
    """yaw drift of a loop-free trajectory P0 per metre driven, from loop pairs: the yaw of the relative rotation
    through the trajectory minus the measured one (a loop pair has almost no baseline, so no per-distance bias),
    against the path length between the two frames.  Returns (deg/m, robust scatter of the per-pair rates, pairs)."""
    C_ = P0[:, :3, 3]
    up_w = np.linalg.svd(C_ - C_.mean(0))[2][2]
    up_w *= np.sign(up_w @ -P0[:, :3, 1].mean(0))
    s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C_, axis=0), axis=1))]
    L, e = [], []
    for (a_, b_), r in pairs_.items():
        if s_[b_] - s_[a_] < min_len:
            continue
        Ro = P0[a_, :3, :3].T @ P0[b_, :3, :3]
        e.append(np.degrees(R.from_matrix(r[2].T @ Ro).as_rotvec() @ (P0[a_, :3, :3].T @ up_w)))
        L.append(s_[b_] - s_[a_])
    if len(L) < 20:
        return None
    rate = np.array(e) / np.array(L)
    return float(np.median(rate)), float(1.4826 * np.median(np.abs(rate - np.median(rate)))), len(L)


def lap_index(poses):
    """lap number of every image from the accumulated heading (0, 1, 2, ...)"""
    Z = poses[:, :3, 2]
    Vt = np.linalg.svd(Z - Z.mean(0))[2]
    yaw = np.unwrap(np.arctan2(Z @ np.cross(Vt[2], Vt[0]), Z @ Vt[0]))
    lap = np.floor(np.abs(yaw - yaw[0]) / (2 * np.pi)).astype(int)
    lap = np.maximum.accumulate(lap)
    if (lap == lap.max()).sum() < 0.5 * N / max(lap.max(), 1):     # an incomplete last turn belongs to the lap before
        lap[lap == lap.max()] -= 1
    return lap


def thin_loops(loops, keep):
    """keep a loop pair if it is among the `keep` best (most inliers) of one of its two images"""
    if keep <= 0:
        return loops
    per = {}
    for (a, b), r in loops.items():
        n = min(r[4], r[6])
        per.setdefault(a, []).append((n, (a, b))); per.setdefault(b, []).append((n, (a, b)))
    chosen = set()
    for lst in per.values():
        chosen.update(p for _, p in sorted(lst, reverse=True)[:keep])
    return {p: loops[p] for p in chosen}


def residuals(pairs, k, poses):
    """translation / rotation disagreement between pairs and a trajectory"""
    dt, dr = [], []
    for (a, b), r in pairs.items():
        E = np.linalg.inv(pair_T(r, k)) @ np.linalg.inv(poses[a]) @ poses[b]
        dt.append(np.linalg.norm(E[:3, 3])); dr.append(np.degrees(np.linalg.norm(R.from_matrix(E[:3, :3]).as_rotvec())))
    return np.array(dt), np.array(dr)


def report(name, pairs, k, poses):
    if len(pairs):
        dt, dr = residuals(pairs, k, poses)
        print(f"  {name:>17}: {len(pairs):6d} pairs  trans median {np.median(dt):.3f} m  rmse {np.sqrt((dt**2).mean()):.3f} m  "
              f"p95 {np.percentile(dt,95):.3f} m | rot median {np.median(dr):.2f} deg")


if __name__ == "__main__":
    t0 = time.time()
    with Pool(args.procs) as pool:
        feats = pool.map(extract, range(N), chunksize=8)
    assert not (args.k1 and args.cam), "--k1 and --cam are alternatives"
    assert not (args.cam and (args.road_seg or args.seg_objects)), "--cam with the segmentation options is not implemented"
    if args.mask_v_max > 0:
        n_hm = 0
        for i_f in range(len(feats)):
            kp_, dc_, xyz_, h_ = feats[i_f]
            km = kp_[:, 1] <= args.mask_v_max
            n_hm += int((~km).sum())
            feats[i_f] = (kp_[km], dc_[km], xyz_[km], h_)
        print(f"  hood/bottom mask (v > {args.mask_v_max:.1f}): dropped {n_hm} keypoints")
    _v_raw_list = [f[0][:, 1].astype(np.float64).copy() for f in feats]
    if args.k1 or args.cam:      # undistort the keypoints, re-cast their cloud points along the undistorted rays (z kept)
        dist = np.array([args.k1, 0.0, 0.0, 0.0, 0.0]); K_TRUE = K
        if args.cam:
            c_ = [float(x) for x in args.cam.split(",")]
            K_TRUE = np.array([[c_[0], 0, c_[2]], [0, c_[1], c_[3]], [0, 0, 1.0]]); dist = np.array(c_[4:8] + [0.0])
        def _undist(f):
            kp_u = cv2.undistortPoints(f[0].reshape(-1, 1, 2).astype(np.float64), K_TRUE, dist, P=K).reshape(-1, 2)
            x_u = np.c_[(kp_u[:, 0] - K[0, 2]) / K[0, 0] * f[2][:, 2], (kp_u[:, 1] - K[1, 2]) / K[1, 1] * f[2][:, 2], f[2][:, 2]]
            return kp_u.astype(np.float32), f[1], x_u, f[3]
        feats = [_undist(f) for f in feats]
        print(f"  keypoints undistorted with " + (f"camera {args.cam}" if args.cam else f"k1 = {args.k1:+.3f}"))
    if args.rs_readout_ms != 0.0 and os.path.exists(f"{data}/imu.csv"):
        import json as _json
        from utils.io import load_imu as _load_imu
        _imu = _load_imu(f"{data}/imu.csv")
        _Rci = np.array(_json.load(open(f"{data}/camera_info.json"))["R_cam_imu"], dtype=np.float64)
        _ts = np.array([int(t) for t in stamps], dtype=np.float64) * 1e-9
        _ti = _imu["timestamp"].values.astype(np.float64) * 1e-9
        _wi = _imu[["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]].values.astype(np.float64)
        _stat = (_ti - _ti[0] < 3.0) | (_ti[-1] - _ti < 3.0)
        _wb = _wi[_stat].mean(0) if _stat.sum() > 20 else np.zeros(3)
        _v_cam_all = None
        if args.rs_6dof_traj:
            from scipy.interpolate import CubicSpline as _CS
            _rows = open(args.rs_6dof_traj).read().strip().split("\n")[1:]
            _T6 = np.array([[float(x) for x in r.split(",")[1:]] for r in _rows], dtype=np.float64).reshape(-1, 4, 4)
            _ts6 = _ts - _ts[0]
            _Vw = _CS(_ts6, _T6[:len(_ts), :3, 3], axis=0)(_ts6, 1)
            _v_cam_all = np.einsum("nji,nj->ni", _T6[:len(_ts), :3, :3], _Vw)
        _max_sh = 0.0
        _max_tsh = 0.0
        _H_img = float(cv2.imread(f"{data}/images/{stamps[0]}.jpg").shape[0])
        for i_f in range(len(feats)):
            kp_, dc_, xyz_, h_ = feats[i_f]
            if len(kp_) == 0:
                continue
            _tq = _ts[i_f] - args.gyro_offset
            _m = np.abs(_ti - _tq) < 0.025
            _w_c = _Rci @ ((_wi[_m].mean(0) if _m.sum() else _wi[np.clip(np.searchsorted(_ti, _tq), 0, len(_ti) - 1)]) - _wb)
            _dt_v = (args.rs_readout_ms * 1e-3) * (_v_raw_list[i_f] - K[1, 2]) / _H_img
            _rays = np.c_[(kp_[:, 0].astype(np.float64) - K[0, 2]) / K[0, 0], (kp_[:, 1].astype(np.float64) - K[1, 2]) / K[1, 1], np.ones(len(kp_))]
            _r0 = R.from_rotvec(_w_c[None, :] * _dt_v[:, None]).apply(_rays)
            _u0 = _r0[:, :2] / _r0[:, 2:3]
            _x_new = np.c_[_u0[:, 0] * xyz_[:, 2], _u0[:, 1] * xyz_[:, 2], xyz_[:, 2]]
            if _v_cam_all is not None:
                _x_gs = _x_new + _v_cam_all[i_f][None, :] * _dt_v[:, None]
                _okz = (xyz_[:, 2] > 0.5) & (_x_gs[:, 2] > 0.5)
                _u_gs = _u0.copy()
                _u_gs[_okz] = _x_gs[_okz, :2] / _x_gs[_okz, 2:3]
                _kp_rot = np.c_[_u0[:, 0] * K[0, 0] + K[0, 2], _u0[:, 1] * K[1, 1] + K[1, 2]]
                _kp_new = np.c_[_u_gs[:, 0] * K[0, 0] + K[0, 2], _u_gs[:, 1] * K[1, 1] + K[1, 2]]
                _tsh = np.linalg.norm(_kp_new - _kp_rot, axis=1)
                _okz &= _tsh < 25.0
                _kp_new[~_okz] = _kp_rot[~_okz]
                _x_new[_okz] = _x_gs[_okz]
                if _okz.any():
                    _max_tsh = max(_max_tsh, float(_tsh[_okz].max()))
            else:
                _kp_new = np.c_[_u0[:, 0] * K[0, 0] + K[0, 2], _u0[:, 1] * K[1, 1] + K[1, 2]]
            _max_sh = max(_max_sh, float(np.linalg.norm(_kp_new - kp_, axis=1).max()))
            feats[i_f] = (_kp_new.astype(np.float32), dc_, _x_new.astype(xyz_.dtype), h_)
        print(f"  IMU rolling-shutter deskew (T_rs={args.rs_readout_ms:+.1f} ms, dt={args.gyro_offset*1e3:+.1f} ms): max pixel shift {_max_sh:.2f} px" + (f" (6-DOF trans max {_max_tsh:.2f} px)" if _v_cam_all is not None else ""))
    H = np.array([f[3] for f in feats])
    good = np.isfinite(H); H = np.interp(np.arange(N), np.where(good)[0], H[good])
    GROUND_KP = []       # keypoints that lie on the road according to the cloud (used to pick road points in the BA)
    SEG = np.load(f"{target}/seg.npz") if (args.road_seg or args.seg_objects) else None
    def seg_at(name, i, kp):
        """segmentation probability (0..1) at the keypoints (given in undistorted pixels when --k1)"""
        u, v = kp[:, 0].astype(np.float64), kp[:, 1].astype(np.float64)
        if args.k1:
            x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
            d = 1 + args.k1 * (x * x + y * y)
            u, v = x * d * K[0, 0] + K[0, 2], y * d * K[1, 1] + K[1, 2]
        m = SEG[name][i]
        r = np.clip((v / 4).astype(int), 0, m.shape[0] - 1); c = np.clip((u / 4).astype(int), 0, m.shape[1] - 1)
        return m[r, c] / 255.0
    for i, f in enumerate(feats):
        if args.seg_objects:
            keep = seg_at("obj", i, f[0]) < 0.5
            f = feats[i] = (f[0][keep], f[1][keep], f[2][keep], f[3])
        x_ = f[2]
        gk = np.isfinite(x_[:, 2]) & (np.abs(x_[:, 1] - H[i]) < 0.1 * H[i]) & (np.abs(x_[:, 0]) < 4) & (x_[:, 2] < 12)
        if args.road_seg:
            gk &= seg_at("road", i, f[0]) > 0.5
        GROUND_KP.append(gk)
        F_ALL[i] = Frame(f[0], f[1], None)
        ok = np.isfinite(f[2][:, 2])
        F[i] = Frame(f[0][ok], f[1][ok].astype(np.float32), f[2][ok])
    del feats
    print(f"A. features: {np.mean([len(f.kp) for f in F_ALL]):.0f} per image, {np.mean([len(f.kp) for f in F]):.0f} with depth  ({time.time()-t0:.0f}s)")

    if args.zmax < 30:
        for f in F:
            f.xyz = np.where(f.xyz[:, 2:3] < args.zmax, f.xyz, np.nan)
        for i, f in enumerate(F):
            keep = np.isfinite(f.xyz[:, 2])
            F[i] = Frame(f.kp[keep], f.desc[keep], f.xyz[keep])
    DEV = "cuda" if (HAS_CUDA and not args.cpu) else "cpu"      # same matcher code on both; the CPU is only slower
    print(f"  matching device: {DEV}")
    globals()["MATCHER"] = GpuMatcher([f.desc for f in F], ratio=PAIR["ratio"], device=DEV)
    SG = dict(bias=not args.no_scale_bias)
    odom = run_pairs([(i, i + s) for s in ODOM_STEPS for i in range(N - s)], 30)
    print(f"B. odometry pairs: {len(odom)}  ({time.time()-t0:.0f}s)")
    k = solve_scales(None if args.no_scale_graph else odom, H, args.ground_sigma, **SG)
    R_GYRO = None     # stage state, passed explicitly to optimise()
    NHC = None
    used, allp = {}, dict(odom)
    stages = ["odometry"]
    poses = chain(odom, k)
    poses = optimise(odom, k, poses)
    vis_odom_poses = poses.copy()
    if args.no_gyro or not os.path.exists(f"{data}/imu.csv"):
        print("  [stage] gyro factors OFF (" + ("--no-gyro" if args.no_gyro else "no imu.csv") + ")")
    if not args.no_gyro and os.path.exists(f"{data}/imu.csv"):
        import json
        from utils.gyro import Gyro
        from utils.io import load_imu
        ts = np.array([int(t) for t in stamps]) * 1e-9
        gyro = Gyro(load_imu(f"{data}/imu.csv"), json.load(open(f"{data}/camera_info.json"))["R_cam_imu"], time_offset=args.gyro_offset)
        step = np.r_[np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1), 0]
        # "still" relative to the usual step, so that it does not depend on the unit of the clouds (Track02)
        moving = np.median(step[step > 0.2 * np.median(step)])
        still = median_filter(step, size=5, mode="nearest") < 0.017 * moving
        print(f"  still frames: {int(still.sum())} (step below {0.017*moving:.4f}; usual step {moving:.3f})")
        print("  gyro bias from still periods (t, bias, samples):", gyro.set_bias_from_still(ts, still))
        Rg = gyro.camera_rotations(ts)
        # refine the IMU-to-camera rotation: the visual and the gyro rotation vectors of the same interval
        # must be the same vector in the camera frame
        rv = lambda Rs, w: np.array([R.from_matrix(Rs[i].T @ Rs[i + w]).as_rotvec() for i in range(0, N - w, 3)])
        av, ag = rv(poses[:, :3, :3], 10), rv(Rg, 10)
        big = np.linalg.norm(av, axis=1) > np.radians(3)
        U, _, Vt = np.linalg.svd(ag[big].T @ av[big])
        Rc = (U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt).T      # av ~ Rc ag
        if args.gyro_extrinsic == "none":
            Rc = np.eye(3)
        elif args.gyro_extrinsic == "tilt":
            # the car turns almost only about one axis; a correction about that axis itself cannot be
            # observed (it does not change those rotations), so remove it (swing-twist decomposition)
            u = av[big].sum(0); u /= np.linalg.norm(u)
            q = R.from_matrix(Rc).as_quat()
            tw = np.r_[(q[:3] @ u) * u, q[3]]; tw /= np.linalg.norm(tw)
            Rc = (R.from_matrix(Rc) * R.from_quat(tw).inv()).as_matrix()
        print(f"  gyro: extrinsic refinement {np.degrees(np.linalg.norm(R.from_matrix(Rc).as_rotvec())):.2f} deg, "
              f"10-frame rotation difference to vision {np.degrees(np.median(np.linalg.norm(av-ag,axis=1))):.3f} -> "
              f"{np.degrees(np.median(np.linalg.norm(av-ag@Rc.T,axis=1))):.3f} deg")
        Rg = Rg @ Rc.T
        R_GYRO = Rg; stages.append("gyro")
        poses = optimise(odom, k, poses, gyro=R_GYRO)
    if not args.no_nhc:
        # calibrate the camera mounting direction and the lever arm to the rear axle from the PnP translations
        sam = []
        for s_ in (3, 5, 8):
            for i in range(N - s_):
                r = odom.get((i, i + s_))
                if r is not None:
                    t = pair_T(r, k)[:3, 3]
                    if np.linalg.norm(t) > 0.1 * s_:
                        sam.append((poses[i, :3, :3].T @ poses[i + s_, :3, :3], t))
        cal, rob = nhc.calibrate(sam, with_height=args.nhc_height < 0)
        if args.nhc_height >= 0:
            cal[3] = args.nhc_height
        print(f"  car constraint: mount yaw {np.degrees(cal[0]):+.2f} deg, pitch {np.degrees(cal[1]):+.2f} deg, lever arm {cal[2]:.3f} m, height {cal[3]:.3f} m "
              f"({len(sam)} samples; PnP direction scatter {rob[0]:.2f} / {rob[1]:.2f} deg lateral / vertical)")
        NHC = tuple(cal); stages.append("car"); NHC_CAL[:] = list(cal)
        poses = optimise(odom, k, poses, gyro=R_GYRO, car=NHC)
    VO = VOR = None
    if args.vo:
        # ---------------------------------------------------------------- VO translation directions
        import pickle
        VO_STEPS = tuple(int(x) for x in args.vo_steps.split(","))
        vo_cfg = dict(feat=_key(**FEAT), **GEOM, ratio=PAIR["ratio"], thr_px=args.vo_thr, steps=VO_STEPS,
                      half=DEV == "cuda", version=1)
        vo_file = f"{res_dir}/vo_{_key(**vo_cfg)}.pkl"
        def vo_one(gm_vo, pq):
            """essential-matrix inliers of all keypoints of a pair"""
            cv2.setNumThreads(1)
            a, b = pq
            m = gm_vo.match(a, b)
            pa = F_ALL[a].kp[m[:, 0]].astype(np.float64); pb = F_ALL[b].kp[m[:, 1]].astype(np.float64)
            ok = vodir.essential_inliers(K, pa, pb, args.vo_thr, seed=a * 100003 + b)
            return pa[ok].astype(np.float32), pb[ok].astype(np.float32)
        if os.path.exists(vo_file):
            vo_m = pickle.load(open(vo_file, "rb"))
            vo_m = {p_: m_ for p_, m_ in vo_m.items() if p_[1] < N}          # (--frames: only the pairs inside)
        else:
            gm_vo = GpuMatcher([f.desc.astype(np.float32) for f in F_ALL], ratio=PAIR["ratio"], half=True, device=DEV)
            vo_pairs = [(i, i + s_) for s_ in VO_STEPS for i in range(N - s_)]
            with ThreadPoolExecutor(args.procs) as ex:
                vo_m = dict(zip(vo_pairs, ex.map(lambda pq: vo_one(gm_vo, pq), vo_pairs)))
            del gm_vo
            if DEV == "cuda":
                import torch
                torch.cuda.empty_cache()
            if not args.frames:            # a partial run must not leave a partial cache under the full key
                tmp = f"{vo_file}.{os.getpid()}.tmp"
                pickle.dump(vo_m, open(tmp, "wb")); os.replace(tmp, vo_file)
        VO, vst = vodir.debias(vodir.measure(K, poses, vo_m), poses)
        print(f"  VO directions: {len(VO)} pairs (steps {VO_STEPS}), E inliers median {np.median([n for _, n in VO.values()]):.0f}; "
              f"constant offset to the trajectory {vst['offset_lat_deg']:+.2f} / {vst['offset_vert_deg']:+.2f} deg (removed), "
              f"scatter {vst['resid_lat_deg']:.2f} / {vst['resid_vert_deg']:.2f} deg lateral / vertical  ({time.time()-t0:.0f}s)")
        stages.append("vo")
        if args.vo_rot:
            VOR = vodir.rotations(K * np.array([[args.vo_rot_fscale] * 2 + [1.0]] * 2 + [[0, 0, 1.0]]), poses, vo_m, procs=args.procs)
            VOR_RAW, POSES_VOR = VOR, poses.copy()
            if args.vo_rot_bias not in (0.0, "auto"):
                # the image rotations turn by a constant amount per metre driven (about the vertical) more than the
                # gyro and the loops say: remove it
                VOR, up_c = yaw_bias_removed(VOR_RAW, POSES_VOR, args.vo_rot_bias)
                print(f"  VO rotations: yaw bias {args.vo_rot_bias:+.4f} deg/m removed (vertical in the camera {np.round(up_c, 3)})")
            e = [np.degrees(np.linalg.norm(R.from_matrix(Rab.T @ (np.linalg.inv(poses[a_]) @ poses[b_])[:3, :3]).as_rotvec()))
                 for (a_, b_), (Rab, _) in VOR.items()]
            print(f"  VO rotations: {len(VOR)} pairs, differ from the current trajectory by median {np.median(e):.3f} deg  ({time.time()-t0:.0f}s)")
            stages.append("vo-rot")
        poses = optimise(odom, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
    odom_poses = poses.copy()
    if args.dash_scale > 0:
        from utils import dashruler
        Kt_, Dt_ = true_camera()
        dk_ = _key(K=np.round(Kt_, 4).tolist(), D=np.round(Dt_, 6).tolist(), m=np.round(mount_deg(), 3).tolist())
        prof_, lat_ = dashruler.profiles(f"{data}/images", f"{res_dir}/dash2_prof_{dk_}.npz", K=Kt_, D=Dt_, mount=mount_deg())
        pr_, med_ = dashruler.periods(prof_, lat_, poses, nhc.axes(*np.radians(mount_deg()))[0], cruise=args.dash_cruise)
        DASH_SPANS[:] = [(ta, tb) for ta, tb, _ in pr_]
        print(f"  dash spans for the scale graph: {len(DASH_SPANS)} (median period {med_:.3f} on the odometry)")
        k = solve_scales(None if args.no_scale_graph else odom, H, args.ground_sigma, **SG)

    loops, held = {}, {}
    lap = lap_index(poses)
    print("  laps start at frames", [int(np.argmax(lap == l)) for l in range(lap.max() + 1)])
    if args.holdout_lap_starts:
        lap = np.searchsorted([int(x) for x in args.holdout_lap_starts.split(",")], np.arange(N), side="right") - 1
        print("  holdout laps start at frames", [int(np.argmax(lap == l)) for l in range(lap.max() + 1)], "(--holdout-lap-starts)")
    if not args.no_loop:
        rng = np.random.default_rng(args.seed)
        if args.loop_retrieval:
            st_ = args.loop_retrieval
            sub = list(range(0, N, st_))
            cnt = {}
            for a_ in sub:
                for b_ in sub:
                    if b_ - a_ > 150:
                        cnt[(a_, b_)] = len(MATCHER.match(a_, b_))
            hit = [p_ for p_, c_ in cnt.items() if c_ >= args.loop_retrieval_min]
            todo = sorted({(a_ + da, b_ + db) for a_, b_ in hit for da in range(-st_ // 2, st_ // 2 + 1, 2)
                           for db in range(-st_ // 2, st_ // 2 + 1, 2) if 0 <= a_ + da < N and 0 <= b_ + db < N})
            new = run_pairs(todo, args.loop_inliers)
            for key, r in new.items():
                out = (args.holdout_lap in (lap[key[0]], lap[key[1]])) if args.holdout_lap >= 0 else rng.random() < args.holdout
                (held if out else loops)[key] = r
            print(f"loop retrieval: {len(cnt)} image pairs matched, {len(hit)} with >= {args.loop_retrieval_min} matches, "
                  f"{len(todo)} pairs tried, {len(new)} verified  ({time.time()-t0:.0f}s)")
        for it, radius in enumerate(float(x) for x in args.loop_radii.split(",")):
            todo = [p for p in loop_candidates(poses, radius) if p not in loops and p not in held]
            new = run_pairs(todo, args.loop_inliers)
            for key, r in new.items():
                out = (args.holdout_lap in (lap[key[0]], lap[key[1]])) if args.holdout_lap >= 0 else rng.random() < args.holdout
                (held if out else loops)[key] = r
            used = thin_loops(loops, args.loop_keep)
            allp = {**odom, **used}
            print(f"loop round {it}: tried {len(todo)}, verified {len(new)}, total loops {len(loops)} (+{len(held)} held out)  ({time.time()-t0:.0f}s)")
            k = solve_scales(None if args.no_scale_graph else allp, H, args.ground_sigma, **SG)
            poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
        stages.append(f"loops({len(used)})")
        if args.height_scale == 0 and args.no_ba and not args.loop_after_height and not args.vo_loops:
            globals()["MATCHER"] = None          # no more matching in this run: give the GPU back to other runs
            if DEV == "cuda":
                import torch
                torch.cuda.empty_cache()
        yd = loop_yaw_drift(odom_poses, {**loops, **held})
        if yd is not None:
            print(f"  yaw drift of the loop-free trajectory from the loop pairs: {yd[0]*1000:+.1f} mdeg/m (scatter {yd[1]*1000:.1f}, {yd[2]} pairs)")
        if VOR is not None and args.vo_rot_bias == "auto":
            # self-calibration: the yaw drift per metre from the loops that are used (never the held-out ones),
            # removed from the image rotations; odometry and loop solution again (twice: the bias is re-measured)
            beta = 0.0
            for rep in range(2):
                yd = loop_yaw_drift(odom_poses, loops)
                if yd is None:
                    if rep == 0 and args.vo_rot_bias_fallback:
                        beta = args.vo_rot_bias_fallback
                        print(f"  [warn] --vo-rot-bias auto: too few loop pairs -> fallback {beta*1000:+.1f} mdeg/m (assumes Track01's camera)")
                        VOR, up_c = yaw_bias_removed(VOR_RAW, POSES_VOR, beta)
                        odom_poses = optimise(odom, k, POSES_VOR, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
                        poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
                    else:
                        print("  [warn] --vo-rot-bias auto: too few loop pairs, no yaw-bias correction")
                    break
                beta += yd[0]
                VOR, up_c = yaw_bias_removed(VOR_RAW, POSES_VOR, beta)
                odom_poses = optimise(odom, k, POSES_VOR, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
                poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
                print(f"  --vo-rot-bias auto: yaw drift {yd[0]*1000:+.1f} mdeg/m from {yd[2]} loop pairs -> bias {beta*1000:+.1f} mdeg/m removed")
            stages.append(f"vo-rot-bias({beta*1000:+.1f})")
        if args.rev_loops:
            from utils import revloop
            REV, cc_ = {}, None
            for rr in range(args.rev_loops):
                if cc_ is None and args.loop_seg and not args.rev_keep_cars:
                    cc_ = revloop.CloudCache(files, seg=f"{target}/seg.npz", K=K)
                import pickle, hashlib
                rk_ = hashlib.md5(np.round(poses[:, :3, :], 6).tobytes() + np.round(k, 8).tobytes()
                                  + repr((args.loop_seg and not args.rev_keep_cars, len(files))).encode()).hexdigest()[:10]
                rf_ = f"{res_dir}/revloop_{rk_}.pkl"          # same input -> same registrations (deterministic): reuse
                if os.path.exists(rf_):
                    new_ = pickle.load(open(rf_, "rb"))
                    print(f"  opposite-direction loops: {len(new_)} from cache {os.path.basename(rf_)}")
                else:
                    new_, cc_ = revloop.search(files, poses, k, cc_)
                    tmp_ = f"{rf_}.{os.getpid()}.tmp"; pickle.dump(new_, open(tmp_, "wb")); os.replace(tmp_, rf_)
                REV.update(new_)
                EXTRA[:] = revloop.factors(REV, *[float(x) for x in args.rev_sigma.split(",")])
                poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
            stages.append(f"rev-loops({len(REV)})")
            print(f"  opposite-direction loops in the graph: {len(REV)}  ({time.time()-t0:.0f}s)")
        if args.bev_loops:
            from utils import bevloop
            seg_ = f"{target}/seg.npz" if args.loop_seg else None
            mount_ = mount_deg()
            if args.bev_pitch_cal:
                mount_ = bevloop.calibrate_pitch(f"{data}/images", poses, seg=seg_, mount=mount_)
            Kt_, Dt_ = true_camera()
            rk_ = _key(m=np.round(mount_, 3).tolist())
            roll_ = bevloop.road_roll(files, f"{target}/seg.npz", K, f"{res_dir}/road_roll_{rk_}.npy", mount=mount_) if args.bev_roll else None
            mb_ = bevloop.MarkingBEV(f"{data}/images", seg=seg_, mount=mount_, roll=roll_, K=Kt_, D=Dt_)
            h_, nh_ = bevloop.unit_height(mb_, poses)
            if args.bev_unit_scale != 1.0:
                h_ *= args.bev_unit_scale
                print(f"  marking BEV: unit scaled by {args.bev_unit_scale}")
            print(f"  marking BEV: camera height {h_:.3f} trajectory units ({nh_} frame pairs)")
            REVF = list(EXTRA)
            hist_ = []
            for rr in range(args.bev_loops):
                corr_, _ = bevloop.search(f"{data}/images", poses, h_, mb=mb_, ahead=[float(x) for x in args.bev_ahead.split(",")])
                if args.bev_audit:
                    hist_.append((poses.copy(), dict(corr_), dict(bevloop.LAST_RAW)))
                EXTRA[:] = REVF + bevloop.factors(poses, corr_, mb_.F, mb_.L, tuple(float(x) for x in args.bev_sigma.split(",")), center_fix=args.bev_center_fix)
                poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
            if args.bev_audit and hist_:
                bevloop.audit(mb_, h_, hist_, poses, f"{res_dir}/bevaudit_{os.path.splitext(os.path.basename(args.out))[0]}.npz")
            stages.append(f"bev-loops({len(corr_)})")
        if args.vo_loops and VO is not None:
            # ---------------------------------------------------------------- VO directions of the loop pairs
            # the side / height offset between two passes of the same place, from all keypoints (no cloud depth);
            # rotation from the current solution, the constant offset of the odometry directions removed as well
            dist = lambda p: np.linalg.norm(poses[p[1], :3, 3] - poses[p[0], :3, 3])
            lp = [p for p in sorted(used) if dist(p) > args.vo_loop_min]
            vol_file = f"{res_dir}/vol_{_key(**vo_cfg)}.pkl"        # grows: every run adds the pairs it needed
            vol_m = pickle.load(open(vol_file, "rb")) if os.path.exists(vol_file) else {}
            todo = [p for p in lp if p not in vol_m]
            if todo:
                gm_vo = GpuMatcher([f.desc.astype(np.float32) for f in F_ALL], ratio=PAIR["ratio"], half=True, device=DEV)
                with ThreadPoolExecutor(args.procs) as ex:
                    vol_m.update(zip(todo, ex.map(lambda pq: vo_one(gm_vo, pq), todo)))
                del gm_vo
                if DEV == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                tmp = f"{vol_file}.{os.getpid()}.tmp"
                pickle.dump(vol_m, open(tmp, "wb")); os.replace(tmp, vol_file)
            VOL, lst = vodir.debias(vodir.measure(K, poses, {p: vol_m[p] for p in lp}, min_dist=args.vo_loop_min), poses, R=None if args.vo_loop_debias else vst["R"])
            print(f"  VO directions of loop pairs: {len(VOL)} of {len(used)} (apart > {args.vo_loop_min} m; {len(todo)} newly matched), "
                  f"E inliers median {np.median([n for _, n in VOL.values()]):.0f}; offset to the trajectory {lst['bias_after_lat_deg']:+.2f} / "
                  f"{lst['bias_after_vert_deg']:+.2f} deg, scatter {lst['resid_lat_deg']:.2f} / {lst['resid_vert_deg']:.2f} deg  ({time.time()-t0:.0f}s)")
            VO = {**VO, **VOL}
            stages.append("vo-loops")
            poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)

    if args.dash_ruler > 0:
        from utils import dashruler
        Kt_, Dt_ = true_camera()
        dk_ = _key(K=np.round(Kt_, 4).tolist(), D=np.round(Dt_, 6).tolist(), m=np.round(mount_deg(), 3).tolist())
        prof_, lat_ = dashruler.profiles(f"{data}/images", f"{res_dir}/dash2_prof_{dk_}.npz", K=Kt_, D=Dt_, mount=mount_deg())
        Fd_ = nhc.axes(*np.radians(mount_deg()))[0]
        pr_, med_ = dashruler.periods(prof_, lat_, poses, Fd_, cruise=args.dash_cruise)
        EXTRA_VALUES[dashruler.DKEY] = np.array([med_])
        REVD = list(EXTRA)
        EXTRA[:] = REVD + [dashruler.factor(ta, tb, args.dash_ruler * med_, robust=args.dash_robust) for ta, tb, _ in pr_]
        per0 = np.array([p_[2] for p_ in pr_])
        poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
        C_ = poses[:, :3, 3]; s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C_, axis=0), axis=1))]
        per1 = np.array([np.interp(tb, np.arange(N), s_) - np.interp(ta, np.arange(N), s_) for ta, tb, _ in pr_])
        print(f"  dash ruler: {len(pr_)} periods, median {med_:.3f}; robust spread {100*1.4826*np.median(np.abs(per0/np.median(per0)-1)):.1f} % "
              f"-> {100*1.4826*np.median(np.abs(per1/np.median(per1)-1)):.1f} % after; period D = {EXTRA_VALUES[dashruler.DKEY][0]:.3f}  ({time.time()-t0:.0f}s)")
        stages.append(f"dash-ruler({len(pr_)})")
    if args.line_straight > 0:
        from utils import dashruler
        Kt_, Dt_ = true_camera()
        dk_ = _key(K=np.round(Kt_, 4).tolist(), D=np.round(Dt_, 6).tolist(), m=np.round(mount_deg(), 3).tolist())
        prof_, lat_ = dashruler.profiles(f"{data}/images", f"{res_dir}/dash2_prof_{dk_}.npz", K=Kt_, D=Dt_, mount=mount_deg())
        Fm_, Lm_, Um_ = nhc.axes(*np.radians(mount_deg()))
        C_ = poses[:, :3, 3]; up_w_ = np.linalg.svd(C_ - C_.mean(0))[2][2]; up_w_ *= np.sign(up_w_ @ -poses[:, :3, 1].mean(0))
        tri_ = sorted(set(dashruler.line_triplets(lat_, poses, Um_, span=20, max_turn=1.5)
                          + dashruler.line_triplets(lat_, poses, Um_, span=40, max_turn=3.0)
                          + dashruler.line_triplets(lat_, poses, Um_, span=75, max_turn=3.0)))
        h_ = 1.18
        before_ = []
        for i_, j_, k_ in tri_:
            Ps_ = [gtsam.Pose3(poses[q]) for q in (i_, j_, k_)]
            f_ = dashruler.line_points_factor(i_, j_, k_, lat_, Fm_, Lm_, up_w_, h_, 5.0, args.line_straight)
            vv_ = gtsam.Values(); [vv_.insert(q, P_) for q, P_ in zip((i_, j_, k_), Ps_)]
            before_.append(f_.unwhitenedError(vv_)[0])
            EXTRA.append(f_)
        poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
        after_ = []
        for i_, j_, k_ in tri_:
            vv_ = gtsam.Values(); [vv_.insert(q, gtsam.Pose3(poses[q])) for q in (i_, j_, k_)]
            after_.append(dashruler.line_points_factor(i_, j_, k_, lat_, Fm_, Lm_, up_w_, h_, 5.0, 1.0).unwhitenedError(vv_)[0])
        print(f"  straight centre line: {len(tri_)} triplets, bend of the line (middle point off the chord) median |.| "
              f"{np.median(np.abs(before_)):.3f} -> {np.median(np.abs(after_)):.3f} units  ({time.time()-t0:.0f}s)")
        stages.append(f"line-straight({len(tri_)})")
    pg_poses = poses.copy()
    need_tracks = (not args.no_ba) or args.height_scale > 0
    if need_tracks:
        # ---------------------------------------------------------------- E. bundle adjustment
        import torch
        globals()["MATCHER"] = None      # pair matching is finished; free its descriptors (8 GB card)
        if DEV == "cuda":
            torch.cuda.empty_cache()
        gm = GpuMatcher([f.desc.astype(np.float32) for f in F_ALL], ratio=PAIR["ratio"], half=True, device=DEV)
        offsets = np.r_[0, np.cumsum([len(f.kp) for f in F_ALL])]
        kps = [f.kp for f in F_ALL]
        ground_kp = np.concatenate(GROUND_KP)
        up_c = nhc.axes(*(NHC[:2] if NHC is not None else (0.0, 0.0)))[2]

        def hstat(h):
            g_ = np.isfinite(h)
            if g_.sum() < 50:
                return f"only {g_.sum()} frames with enough road points"
            sm = median_filter(np.interp(np.arange(N), np.where(g_)[0], h[g_]), size=41, mode="nearest")
            return (f"{g_.sum()} frames, median {np.median(h[g_]):.3f} m, spread (robust std) {1.4826*np.median(np.abs(h[g_]-np.median(h[g_])))*100:.1f} cm; "
                    f"smoothed, relative to median [%]: " + " ".join(f"{100*(sm[i]/np.median(h[g_])-1):+.1f}" for i in range(0, N, 60)))
        ba_pairs = sorted(set(odom) | set(used) | {(i, i + s_) for s_ in (13, 21) for i in range(N - s_)})
        raw_m = {}
        with ThreadPoolExecutor(8) as ex:
            for pq, m in zip(ba_pairs, ex.map(lambda pq: gm.match(*pq), ba_pairs)):
                raw_m[pq] = m
        del gm                           # all matching is done: give the card back to other runs (BA takes long)
        if DEV == "cuda":
            torch.cuda.empty_cache()
        print(f"E. bundle adjustment: matched {len(ba_pairs)} pairs on all keypoints  ({time.time()-t0:.0f}s)")
        def triangulated(poses):
            edges = []
            for (a, b), m in raw_m.items():
                if len(m) < 20:
                    continue
                ok = ba.epipolar_inliers(K, poses[a], poses[b], kps[a][m[:, 0]].astype(np.float64), kps[b][m[:, 1]].astype(np.float64))
                edges.append(np.c_[offsets[a] + m[ok, 0], offsets[b] + m[ok, 1]])
            edges = np.concatenate(edges)
            tracks = ba.build_tracks(edges, offsets, offsets[-1])
            return (edges,) + ba.make_points(K, poses, tracks, offsets, kps, ground_kp=ground_kp)

        h0 = None      # reference camera height: fixed once, from the pose graph solution
        report("held, before", held, k, poses)
        hs_dump = dict(k0=k.copy(), poses0=poses.copy())
        for it in range(args.height_scale):
            # the local scale of the solution, measured without the cloud depth: camera height above the road
            # points triangulated from the images.  Make it constant by rescaling the clouds, solve again.
            _, points, st = triangulated(poses)
            hh, hn = ba.camera_heights(poses, [p_ for p_ in points if p_[3]], up_c, radius=args.height_radius, lat=args.height_lat)
            print(f"  height-scale {it}: camera height above the triangulated road: {hstat(hh)}")
            g_ = np.isfinite(hh)
            if h0 is None:
                h0 = float(np.median(hh[g_]))
            if not args.height_scale_all:
                # only where the car moves: the height of a standing camera does not react to its cloud scale,
                # so repeated corrections there would grow without bound; such frames take the neighbours' value
                st_ = np.r_[np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1), 0]
                st_ = median_filter(st_, size=5, mode="nearest")
                g_ &= st_ > args.height_min_step * np.median(st_[st_ > 0.2 * np.median(st_)])
            sm = median_filter(np.interp(np.arange(N), np.where(g_)[0], hh[g_]), size=args.height_scale_win, mode="nearest")
            if args.height_lap_gain is None:
                k = k * (h0 / sm) ** args.height_gain
            else:
                # the shared part is what the loops cannot see (all laps wrong in the same way at that place);
                # a lap's own deviation is partly noise of the height measurement
                ls = np.log(sm / h0)
                lc, nc = place_mean(poses, ls, lap_index(poses), radius=args.height_place_radius)
                if args.height_turn_detrend:
                    rr = np.r_[[np.linalg.norm(R.from_matrix(poses[i, :3, :3].T @ poses[i + 1, :3, :3]).as_rotvec()) for i in range(N - 1)], 0.0]
                    rr = median_filter(rr, size=args.height_scale_win, mode="nearest")     # turn per frame [rad], smoothed like the heights
                    A_ = np.c_[rr[g_] - rr[g_].mean(), np.ones(g_.sum())]
                    slope = np.linalg.lstsq(A_, lc[g_], rcond=None)[0][0]
                    lc = lc - slope * (rr - rr[g_].mean())
                    print(f"           turn-rate part of the shared deviation removed: {100*slope*np.radians(1):+.2f} % per deg/frame, "
                          f"std {100*np.std(slope*(rr[g_]-rr[g_].mean())):.2f} %")
                act = np.ones(N)
                if args.height_freeze is not None and it > 0:
                    act = median_filter((np.abs(lc) < args.height_freeze * np.abs(lc_prev)).astype(float), size=args.height_scale_win, mode="nearest")
                    act = np.minimum(act, act_prev)       # once frozen, a place stays frozen
                    print(f"           still corrected: {100*act.mean():.0f} % of the frames")
                lc_prev, act_prev = lc, act
                fac = np.exp(act * (-args.height_gain * lc - args.height_lap_gain * (ls - lc)))
                k = k * fac
                print(f"           shared by the passes of a place: {100*np.std(lc):.2f} %, own deviation of a lap {100*np.std(ls-lc):.2f} % "
                      f"(passes per frame: " + ", ".join(f"{int(v)}: {(nc == v).sum()}" for v in np.unique(nc)) + f"); "
                      f"applied {100*fac.min()-100:+.1f} .. {100*fac.max()-100:+.1f} %")
            poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
            hs_dump.update({f"h{it}": hh, f"k{it+1}": k.copy(), f"poses{it+1}": poses.copy()})
            print(f"           cloud scales changed by {100*(h0/sm).min()-100:+.1f} .. {100*(h0/sm).max()-100:+.1f} %  ({time.time()-t0:.0f}s)")
            report(f"held, it {it}", held, k, poses)
        if args.height_scale:
            stages.append(f"height-scale({args.height_scale})")
            hh, hn = ba.camera_heights(poses, [p_ for p_ in triangulated(poses)[1] if p_[3]], up_c, radius=args.height_radius, lat=args.height_lat)
            print(f"  height-scale final: {hstat(hh)}")
            if args.dump_height:
                np.savez(f"{res_dir}/{args.out}.hs.npz", h_final=hh, **hs_dump)
        if args.loop_after_height > 0 and not args.no_loop:
            # the height correction can move places of the route by metres: revisits that were too far apart in the
            # estimate before (never tried) can be found now
            todo = [p for p in loop_candidates(poses, args.loop_after_height) if p not in loops and p not in held]
            new = run_pairs(todo, args.loop_inliers)
            for key_, r_ in new.items():
                out_ = (args.holdout_lap in (lap[key_[0]], lap[key_[1]])) if args.holdout_lap >= 0 else False
                (held if out_ else loops)[key_] = r_
            used = thin_loops(loops, args.loop_keep)
            allp = {**odom, **used}
            report("new loops, before", {p_: loops[p_] for p_ in new if p_ in loops}, k, poses)
            poses = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, vo=VO, vrot=VOR)
            stages.append(f"loops-after-height(+{len(new)})")
            print(f"  loops after the height correction: tried {len(todo)}, verified {len(new)}, total loops {len(loops)}  ({time.time()-t0:.0f}s)")
        radial = None
        for rnd in range(0 if args.no_ba else args.ba_rounds):
            edges, points, st = triangulated(poses)
            if st["obs"] > args.ba_max_obs:      # keep all road points and the longest other tracks
                points.sort(key=lambda p: (not p[3], -len(p[1])))
                tot = np.cumsum([len(p[1]) for p in points])
                points = points[:int(np.searchsorted(tot, args.ba_max_obs))]
            hh, hn = ba.camera_heights(poses, points, up_c)
            print(f"  round {rnd}: camera height above the triangulated road before optimisation: {hstat(hh)}")
            e0 = ba.reprojection_stats(K, poses, points)[0]
            nobs = sum(len(p[1]) for p in points)
            print(f"  round {rnd}: {len(edges)} verified matches -> {st['tracks']} tracks -> {st['points']} points "
                  f"(dropped: behind {st['behind']}, reprojection {st['reproj']}, parallax {st['parallax']}); "
                  f"using {len(points)} points / {nobs} observations, mean track length {nobs/len(points):.1f}")
            heights = None
            if not args.no_height and np.isfinite(hh).sum() > 50:
                if h0 is None:
                    h0 = float(np.nanmedian(hh))     # fixes only the overall scale, which does not matter
                hp = ba.height_pairs(poses, points, up_c, h0, per_frame=args.height_per_frame)
                heights = (hp, h0, up_c)
                print(f"           camera-height factors: {len(hp)} (h0 = {h0:.3f} m, {len(set(i for i, _ in hp))} frames)")
            poses, points = optimise(allp, k, poses, gyro=R_GYRO, car=NHC, points=points, heights=heights, vo=VO, vrot=VOR)
            e1, rad, radial = ba.reprojection_stats(K, poses, points)
            print(f"           reprojection error median {np.median(e0):.2f} -> {np.median(e1):.2f} px, rms {np.sqrt((e0**2).mean()):.2f} -> "
                  f"{np.sqrt((e1**2).mean()):.2f} px  ({time.time()-t0:.0f}s)")
            hh, hn = ba.camera_heights(poses, points, up_c)
            print(f"           camera height above the triangulated road after optimisation:  {hstat(hh)}")
        np.save(f"{res_dir}/{args.out}.camh.npy", hh)
        if radial is not None:
          stages.append(f"ba({args.ba_rounds}" + ("" if heights is None else "+height") + ")")
          bins = [0, 150, 300, 450, 600, 900]
          print("           radial residual by distance from the image centre [px]: " +
              ", ".join(f"{bins[i]}-{bins[i+1]}: {np.median(radial[(rad>=bins[i])&(rad<bins[i+1])]):+.2f}" for i in range(len(bins) - 1)))

    print("consistency of pairs with the final trajectory:")
    if VO is not None:
        _, vst = vodir.debias(VO, poses)
        print(f"  {'VO directions':>17}: {len(VO):6d} pairs  offset {vst['offset_lat_deg']:+.2f} / {vst['offset_vert_deg']:+.2f} deg, "
              f"scatter {vst['resid_lat_deg']:.2f} / {vst['resid_vert_deg']:.2f} deg lateral / vertical")
    report("odometry", odom, k, poses)
    report("loops (used)", used, k, poses)
    report("loops (held)", held, k, poses)
    if held or loops:
        print("... and with odometry only (drift before loop closure):")
        report("loops (vis odom)", {**loops, **held}, k, vis_odom_poses)
        report("loops (full odom)", {**loops, **held}, k, odom_poses)
    if not np.array_equal(pg_poses, poses):
        print("... and with the pose graph before bundle adjustment:")
        report("loops (held)", held, k, pg_poses)
        from process.compare import ate
        print(f"  bundle adjustment moved the trajectory by {ate(poses[:, :3, 3], pg_poses[:, :3, 3])[0]:.3f} m (Sim(3) ATE)")

    if R_GYRO is not None or (os.path.exists(f"{data}/imu.csv") and args.no_gyro):
        if R_GYRO is None:      # only as an independent check
            import json
            from utils.gyro import Gyro
            from utils.io import load_imu
            ts = np.array([int(t) for t in stamps]) * 1e-9
            gyro = Gyro(load_imu(f"{data}/imu.csv"), json.load(open(f"{data}/camera_info.json"))["R_cam_imu"], time_offset=args.gyro_offset)
            step = np.r_[np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1), 0]
            gyro.set_bias_from_still(ts, median_filter(step, size=5, mode="nearest") < 0.017 * np.median(step[step > 0.2 * np.median(step)]))
            Rg = gyro.camera_rotations(ts)
        else:
            Rg = R_GYRO
        for name, Pz in (("visual odometry", vis_odom_poses), ("final", poses)):
            line = []
            for w in (10, 100, 400, N - 1):
                e = [np.degrees(np.linalg.norm(R.from_matrix((Pz[i, :3, :3].T @ Pz[i + w, :3, :3]).T @ (Rg[i].T @ Rg[i + w])).as_rotvec()))
                     for i in range(0, N - w, 5)]
                line.append(f"{w}: {np.median(e):.2f}")
            print(f"  rotation vs gyro [deg] ({name:>15}), median over windows of frames " + ", ".join(line))

    header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
    df = pd.DataFrame([[t] + T.reshape(-1).tolist() for t, T in zip(stamps, poses)], columns=header.split(","))
    df.to_csv(f"{res_dir}/{args.out}", index=False)
    np.save(f"{res_dir}/{args.out}.scale.npy", k)
    import json
    json.dump({"argv": sys.argv, "args": vars(args), "stages": stages, "device": DEV, "caches": [feat_dir, PAIR_FILE[0]] + ([vo_file] if VO is not None else []) + ([vol_file] if "vol_file" in globals() else [])},
              open(f"{res_dir}/{args.out}.run.json", "w"), indent=1)     # what exactly produced this file
    print("stages:", " -> ".join(stages))
    step = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    print(f"saved {res_dir}/{args.out}  ({N} poses, path length {step.sum():.1f} m, {time.time()-t0:.0f}s)")
