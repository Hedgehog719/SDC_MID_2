"""Dense asphalt photometric homography scale ruler: measures step_k / h_cam per frame pair (k, k+step).

Uses high-pass undistorted ego-lane asphalt texture (no SIFT keypoints required!) with exact rotational +
translational CMOS rolling-shutter correction and batched 1D global ZNCC search over s_obs = ||t|| / h
(zero initial-scale dependence, zero local-minimum trap).

Usage:
  python process/road_homography.py data/Track2 data/Track2/result/t2_sfmx_nodash_rs34_6dof_hm.csv \
      --out road_step_t2 --camera 1036.25,722.20,463.57,-0.0580,0.0866 --rs-readout-ms 34.0
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="Track data directory (contains data/images, seg.npz, result/)")
    ap.add_argument("traj", help="Reference trajectory CSV (only relative rotation R_21 & travel direction t_hat are used)")
    ap.add_argument("--out", default="road_step", help="Output basename saved to <target>/result/<out>.npz")
    ap.add_argument("--camera", default=None, help="RADIAL camera params 'f,cx,cy,k1,k2' OR path to sfm_ba.py .json output (default: auto-read <traj>.json)")
    ap.add_argument("--rs-readout-ms", type=float, default=34.0, help="CMOS top-to-bottom rolling-shutter readout [ms]")
    ap.add_argument("--step", type=int, default=2, help="Frame step k -> k+step (default 2)")
    ap.add_argument("--normal", default="0.01074,0.99944,0.03178", help="Unit road plane normal in camera frame: nx,ny,nz")
    ap.add_argument("--roi-v", default="600,820", help="Vertical pixel ROI [v0,v1] on clean asphalt above car hood")
    ap.add_argument("--roi-u", default="460,980", help="Horizontal pixel ROI [u0,u1] on ego lane")
    ap.add_argument("--h-range", default="0.85,1.45", help="Search range for camera height h_lo,h_hi [traj units] (default 0.85,1.45)")
    ap.add_argument("--min-zncc", type=float, default=0.55)
    ap.add_argument("--min-sharp", type=float, default=0.005)
    ap.add_argument("--max-wy", type=float, default=20.0, help="Exclude frames with |yaw_rate| > max_wy [deg/s]")
    ap.add_argument("--min-sobs", type=float, default=0.0, help="Exclude slow-speed pairs with scale-free s_obs < min_sobs (default 0.0)")
    ap.add_argument("--per-frame-pitch", action="store_true", help="Rotate fixed road normal per frame using scale-free trajectory pitch theta_i = asin(c_y . d_hat_i) to remove braking/slope bias")
    ap.add_argument("--future-road-plane", action="store_true", help="Fit exact scale-invariant 3D road plane (n_patch, d_off) from future trajectory points inside the ROI [2.5, 7.5] u ahead (removes both suspension pitch AND road vertical crest/sag curvature)")
    args = ap.parse_args()

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res_dir = f"{args.target}/result"
    img_dir = f"{args.target}/data/images"
    if not os.path.isdir(img_dir):
        img_dir = f"{args.target}/images"

    rows = open(args.traj).read().strip().split("\n")[1:]
    stamps = [r.split(",")[0] for r in rows]
    ts = (np.array([int(s) for s in stamps], dtype=np.int64) - int(stamps[0])) * 1e-9
    T = np.array([[float(x) for x in r.split(",")[1:]] for r in rows], dtype=np.float64).reshape(-1, 4, 4)
    R_wc, C_w = T[:, :3, :3], T[:, :3, 3]
    N = len(stamps)

    w_cam = np.zeros((N, 3), dtype=np.float64)
    for i in range(1, N - 1):
        dt = ts[i + 1] - ts[i - 1]
        if dt > 1e-6:
            w_cam[i] = Rot.from_matrix(R_wc[i - 1].T @ R_wc[i + 1]).as_rotvec() / dt
    wy_deg = np.degrees(w_cam[:, 1])

    im0 = cv2.imread(f"{img_dir}/{stamps[0]}.jpg", cv2.IMREAD_GRAYSCALE)
    H, W = im0.shape
    cam_src = args.camera
    if cam_src is None:
        auto_json = os.path.splitext(args.traj)[0] + ".json"
        if not os.path.exists(auto_json):
            raise ValueError(f"--camera not provided and auto-json not found: {auto_json}")
        cam_src = auto_json
    if cam_src.endswith(".json") and os.path.exists(cam_src):
        j_cam = json.load(open(cam_src))
        cam_vals = j_cam.get("camera") or j_cam["rounds"][-1]["camera"]
        f_cam, cx, cy, k1, k2 = [float(x) for x in cam_vals[:5]]
        cam_str = ",".join(f"{x:.6f}" for x in (f_cam, cx, cy, k1, k2))
        print(f"Loaded camera from {cam_src}: {cam_str}", flush=True)
    else:
        f_cam, cx, cy, k1, k2 = [float(x) for x in cam_src.split(",")]
        cam_str = cam_src
    K = np.array([[f_cam, 0.0, cx], [0.0, f_cam, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (W, H), cv2.CV_32FC1)
    T_rs = args.rs_readout_ms * 1e-3

    seg = np.load(f"{args.target}/seg.npz")
    seg_road, seg_obj = seg["road"], seg["obj"]

    v0, v1 = [int(x) for x in args.roi_v.split(",")]
    u0, u1 = [int(x) for x in args.roi_u.split(",")]
    vs_grid = np.arange(v0, v1, 2, dtype=np.float32)
    us_grid = np.arange(u0, u1, 2, dtype=np.float32)
    UU, VV = np.meshgrid(us_grid, vs_grid)
    X1_t = torch.tensor(np.stack([(UU - cx) / f_cam, (VV - cy) / f_cam, np.ones_like(UU)], axis=-1), device=dev, dtype=torch.float32)
    V1_t = torch.tensor(VV, device=dev, dtype=torch.float32)
    r_seg = np.clip((VV / 4.0).astype(int), 0, seg_road.shape[1] - 1)
    c_seg = np.clip((UU / 4.0).astype(int), 0, seg_road.shape[2] - 1)

    n_fix = np.array([float(x) for x in args.normal.split(",")], dtype=np.float64)
    n_fix = n_fix / np.linalg.norm(n_fix)
    n_t = torch.tensor(n_fix, device=dev, dtype=torch.float32)

    # Scale-free per-frame road pitch theta_i = asin(c_y . d_hat_i) for braking/slope normal correction
    n_per_frame = None
    d_off_per_frame = np.zeros(N, dtype=np.float64)
    used_fp_frame = np.zeros(N, dtype=bool)
    if args.per_frame_pitch or args.future_road_plane:
        dC = C_w[2:] - C_w[:-2]
        dn = np.linalg.norm(dC, axis=1)
        d_cam = np.einsum("nji,nj->ni", R_wc[1:-1], dC / np.maximum(dn[:, None], 1e-9))
        th = np.full(N, np.nan)
        mov_th = dn > 0.08
        th[1:-1] = np.where(mov_th, np.arcsin(np.clip(d_cam[:, 1], -1.0, 1.0)), np.nan)
        th_bar = float(np.nanmedian(th))
        th = np.where(np.isfinite(th), th, th_bar)
        # Rotate n_fix about camera +X by -(th - th_bar): when nose lifts (th > th_bar), road normal tilts backward (smaller n_z)
        n_per_frame = np.empty((N, 3), dtype=np.float64)
        for i_ in range(N):
            dth_ = float(np.clip(th[i_] - th_bar, -np.radians(4.0), np.radians(4.0)))
            n_per_frame[i_] = Rot.from_euler("x", -dth_).as_matrix() @ n_fix
        print(f"Per-frame scale-free pitch enabled: theta_bar={np.degrees(th_bar):+.3f} deg, std={np.degrees(np.std(th - th_bar)):.3f} deg", flush=True)

    if args.future_road_plane:
        # Fit exact road plane of the asphalt patch ahead using future camera trajectory p_k = R_i^T (C_k - C_i).
        # Future road points lie at Q_k = p_k + h * n_patch, so n_patch . Q_k = d_off + h (100% scale-invariant!).
        n_fp_used = 0
        rx_ratio = n_fix[0] / n_fix[1]
        for i_ in range(N - 3):
            k_end = min(N, i_ + 80)
            p_c = (C_w[i_ + 1:k_end] - C_w[i_]) @ R_wc[i_]
            # Stop before any U-turn / reversal where forward z stops increasing
            z_all = p_c[:, 2]
            dz = np.diff(np.r_[0.0, z_all])
            bad = np.where((dz < -0.05) | (np.abs(p_c[:, 0]) > 2.5))[0]
            if len(bad) > 0:
                p_c = p_c[:bad[0]]
            if len(p_c) < 4:
                continue
            sel = (p_c[:, 2] >= 2.5) & (p_c[:, 2] <= 7.5)
            pts = p_c[sel]
            if len(pts) >= 4 and (pts[:, 2].max() - pts[:, 2].min()) >= 1.5:
                z = pts[:, 2]
                y_corr = pts[:, 1] + rx_ratio * pts[:, 0]
                w = 1.0 / (z ** 2)  # uniform pixel density in ROI v-space (|dv/dz| ~ 1/z^2)
                W_n = w / w.sum()
                zm = float(np.sum(W_n * z))
                ym = float(np.sum(W_n * y_corr))
                vz = float(np.sum(W_n * (z - zm) ** 2))
                if vz > 0.1:
                    b = float(np.sum(W_n * (z - zm) * (y_corr - ym)) / vz)
                    a = ym - b * zm
                    n_raw = np.array([rx_ratio, 1.0, -b], dtype=np.float64)
                    nm = float(np.linalg.norm(n_raw))
                    d_off_c = a / nm
                    if abs(d_off_c) <= 0.25 and abs(-b - n_fix[2]) <= 0.15:
                        n_per_frame[i_] = n_raw / nm
                        d_off_per_frame[i_] = d_off_c
                        used_fp_frame[i_] = True
                        n_fp_used += 1
        print(f"Future-road-plane enabled: {n_fp_used}/{N} frames fitted (d_off med={np.median(d_off_per_frame):+.4f}, std={np.std(d_off_per_frame):.4f} u)", flush=True)

    print(f"Loading & undistorting {N} images ({W}x{H}) on {dev}...", flush=True)
    # Streaming sliding buffer (only need step+1 images in RAM/VRAM at a time -> minimal memory!)
    def load_hp_gpu(fi):
        im = cv2.imread(f"{img_dir}/{stamps[fi]}.jpg", cv2.IMREAD_GRAYSCALE)
        imu = cv2.remap(im, map1, map2, cv2.INTER_LINEAR).astype(np.float32)
        hp = imu - cv2.GaussianBlur(imu, (15, 15), 3.0)
        return torch.tensor(hp, device=dev, dtype=torch.float32)

    step_k = args.step
    # Search scale factor relative to trajectory step over [0.75, 1.35] (61 steps = 1% resolution + parabolic subgrid)
    # Wait: even cleaner -> search h_cand in [0.85, 1.45] -> s_obs = ||t_21|| / h_opt is 100% invariant to trajectory scale!
    # Verify: if trajectory step is scaled by c, t_21 scales by c, h_opt scales by c, so s_obs = ||t_21|| / h_opt is IDENTICAL!
    # To guarantee the search box covers any local scale error up to +/-30%, search mult m = s_obs / (||t_21|| / 1.125) in [0.75, 1.35]
    h_lo, h_hi = [float(x) for x in args.h_range.split(",")]
    B = 61
    h_step = (h_hi - h_lo) / (B - 1)
    h_grid = torch.linspace(h_lo, h_hi, B, device=dev)
    a_grid = (1.0 / h_grid)[:, None, None]

    buf = {fi: load_hp_gpu(fi) for fi in range(min(step_k + 1, N))}
    out_i, out_j, out_sobs, out_h, out_z, out_sh, out_wy, out_fp = [], [], [], [], [], [], [], []

    with torch.no_grad():
        for fi in range(N - step_k):
            fj = fi + step_k
            if fj not in buf:
                buf[fj] = load_hp_gpu(fj)
            if fi - 1 in buf:
                del buf[fi - 1]
            step_u = float(np.linalg.norm(C_w[fj] - C_w[fi]))
            dt_f = float(ts[fj] - ts[fi])
            if step_u < 0.12 or dt_f < 0.02 or dt_f > 0.25 or abs(wy_deg[fi]) > args.max_wy or abs(wy_deg[fj]) > args.max_wy:
                continue
            m_np = (seg_road[fi, r_seg, c_seg] >= 200) & (seg_obj[fi, r_seg, c_seg] < 50)
            if m_np.sum() < 2000:
                continue
            I1 = buf[fi]
            I2 = buf[fj]
            I1_roi = I1[v0:v1:2, u0:u1:2]
            mask = torch.tensor(m_np, device=dev)
            R21 = torch.tensor(R_wc[fj].T @ R_wc[fi], device=dev, dtype=torch.float32)
            t21 = torch.tensor(R_wc[fj].T @ (C_w[fi] - C_w[fj]), device=dev, dtype=torch.float32)
            w1 = torch.tensor(w_cam[fi], device=dev, dtype=torch.float32)
            w2 = torch.tensor(w_cam[fj], device=dev, dtype=torch.float32)
            dt_v1 = (T_rs / H) * (V1_t - cy)
            X1_gs = X1_t + torch.cross(w1[None, None, :] * dt_v1[..., None], X1_t, dim=-1)
            RX1 = torch.einsum("ij,hwj->hwi", R21, X1_gs)
            n_use = torch.tensor(n_per_frame[fi], device=dev, dtype=torch.float32) if n_per_frame is not None else n_t
            nX = torch.einsum("k,hwk->hw", n_use, X1_gs)
            a_cur = (1.0 / (h_grid + float(d_off_per_frame[fi])).clamp(min=0.3))[:, None, None]

            X2_0 = RX1[None] + (a_cur * nX[None])[..., None] * t21[None, None, None, :]
            v2_0 = (X2_0[..., 1] / X2_0[..., 2].clamp(min=0.1)) * f_cam + cy
            rs_mult = 1.0 + (T_rs / (H * dt_f)) * (v2_0 - V1_t[None])
            X2_gs = RX1[None] + (a_cur * rs_mult * nX[None])[..., None] * t21[None, None, None, :]
            dt_v2 = (T_rs / H) * (v2_0 - cy)
            X2 = X2_gs - torch.cross(w2[None, None, None, :] * dt_v2[..., None], X2_gs, dim=-1)
            u2 = (X2[..., 0] / X2[..., 2].clamp(min=0.1)) * f_cam + cx
            v2 = (X2[..., 1] / X2[..., 2].clamp(min=0.1)) * f_cam + cy
            gx = (u2 / (W - 1.0)) * 2.0 - 1.0
            gy = (v2 / (H - 1.0)) * 2.0 - 1.0
            grid = torch.stack([gx, gy], dim=-1)
            I2_w = F.grid_sample(I2[None, None].expand(B, 1, H, W), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
            in_b = mask[None] & (u2 > 2) & (u2 < W - 3) & (v2 > 2) & (v2 < 840)
            m_f = in_b.float()
            cnt = m_f.sum(dim=(1, 2)).clamp(min=1.0)
            mu1 = (I1_roi[None] * m_f).sum(dim=(1, 2)) / cnt
            mu2 = (I2_w * m_f).sum(dim=(1, 2)) / cnt
            d1 = (I1_roi[None] - mu1[:, None, None]) * m_f
            d2 = (I2_w - mu2[:, None, None]) * m_f
            zs = (d1 * d2).sum(dim=(1, 2)) / torch.sqrt((d1 ** 2).sum(dim=(1, 2)) * (d2 ** 2).sum(dim=(1, 2)) + 1e-6)
            zs_np = zs.cpu().numpy()
            k = int(np.argmax(zs_np))
            z_pk = float(zs_np[k])
            if 1 <= k < B - 1:
                y0, y1, y2 = zs_np[k - 1], zs_np[k], zs_np[k + 1]
                denom = y0 - 2.0 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
                h_pk = float(h_grid[k].item()) + float(np.clip(delta, -0.5, 0.5)) * h_step
            else:
                h_pk = float(h_grid[k].item())
            z_side = 0.5 * (zs_np[max(0, k - 5)] + zs_np[min(B - 1, k + 5)])
            sharp = z_pk - float(z_side)
            if z_pk >= args.min_zncc and sharp >= args.min_sharp and (h_lo + 2 * h_step) < h_pk < (h_hi - 2 * h_step) and (step_u / h_pk) >= args.min_sobs:
                out_i.append(fi)
                out_j.append(fj)
                out_sobs.append(step_u / h_pk)
                out_h.append(h_pk)
                out_z.append(z_pk)
                out_sh.append(sharp)
                out_wy.append(0.5 * (wy_deg[fi] + wy_deg[fj]))
                out_fp.append(bool(used_fp_frame[fi]))
            if (fi + 1) % 1000 == 0:
                print(f"  {fi+1}/{N-step_k} pairs processed ({len(out_i)} valid, {time.time()-t0:.1f}s)", flush=True)

    out_path = f"{res_dir}/{args.out}.npz"
    out_h_arr = np.array(out_h, dtype=np.float64)
    h_med = float(np.median(out_h_arr))
    h_mad = float(np.median(np.abs(out_h_arr - h_med)) * 1.4826 / h_med * 100.0)
    np.savez(
        out_path,
        i=np.array(out_i, dtype=np.int32),
        j=np.array(out_j, dtype=np.int32),
        s_obs=np.array(out_sobs, dtype=np.float64),
        h_ref=out_h_arr,
        zncc=np.array(out_z, dtype=np.float32),
        sharp=np.array(out_sh, dtype=np.float32),
        wy_deg=np.array(out_wy, dtype=np.float32),
        used_fp=np.array(out_fp, dtype=bool),
        frp_enabled=bool(args.future_road_plane),
        step=int(step_k),
        camera=cam_str,
        rs_readout_ms=float(args.rs_readout_ms),
    )
    print(f"Saved {out_path}: {len(out_i)}/{N-step_k} valid pairs ({len(out_i)/(N-step_k)*100:.1f}%), "
          f"med ZNCC={np.median(out_z):.3f}, med h={h_med:.4f}, frame MAD={h_mad:.2f}% ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
