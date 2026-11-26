"""Does an independent metric depth network know the per-frame scale errors of the LoGeR clouds?

For every frame: metric depth of a pretrained model (Depth Anything V2 metric-outdoor / Depth Pro, public), sampled at the
pixels of the LoGeR cloud; s_D = median(depth_model / z_loger).  Reference: the per-frame cloud scale k of v10 (gyro,
loops, height correction) = the factor that makes each LoGeR cloud metric-consistent.  If log s_D follows log k, the
model brings scale information that LoGeR does not have.
usage: python process/depth_scale_check.py Track1 <model: da2|depthpro> [--every 1]   -> <track>/depth_<model>.npz
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("model", choices=["da2", "da2s", "da2b", "depthpro", "zoe", "zoenk"])
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--size", type=int, default=0, help="input size for DA2 / Zoe (shorter side, multiple of 14 for DA2); 0 = the model's default")
ap.add_argument("--tag", default="", help="suffix of the output file")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, a.target, "data")
IMG = sorted(glob.glob(f"{D}/images/*.jpg"))
PLY = sorted(glob.glob(f"{D}/raw_ply/*.ply"))
import json
cam = json.load(open(f"{D}/camera_info.json"))
fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]


def read_ply(p):
    with open(p, "rb") as f:
        n = 0
        while True:
            l = f.readline().decode().strip()
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
            if l == "end_header":
                break
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        v = np.frombuffer(f.read(n * dt.itemsize), dt, n)
    return np.c_[v["x"], v["y"], v["z"]].astype(np.float64)


if a.model.startswith("da2") or a.model.startswith("zoe"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = {"da2": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "da2s": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
            "da2b": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf", "zoe": "Intel/zoedepth-kitti", "zoenk": "Intel/zoedepth-nyu-kitti"}[a.model]
    proc = AutoImageProcessor.from_pretrained(name)
    net = AutoModelForDepthEstimation.from_pretrained(name).cuda().eval()

    def depth(img):
        kw = dict(size={"height": a.size, "width": a.size}) if a.size else {}
        x = proc(images=img, return_tensors="pt", **kw).to("cuda")
        with torch.no_grad():
            d = net(**x).predicted_depth
        return torch.nn.functional.interpolate(d[:, None], size=img.shape[:2], mode="bilinear")[0, 0].cpu().numpy()
else:
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
    proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    net = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").cuda().eval()

    def depth(img):
        x = proc(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = net(**x)
        pp = proc.post_process_depth_estimation(o, target_sizes=[img.shape[:2]])[0]
        d = pp["predicted_depth"].float().cpu().numpy()
        f_est = float(pp["focal_length"]) if "focal_length" in pp and pp["focal_length"] is not None else fx
        return d * fx / f_est           # Depth Pro estimates its own focal length: use the known one

idx = list(range(0, len(IMG), a.every))
sD, sG, fov = [], [], []
for j, i in enumerate(idx):
    img = cv2.imread(IMG[i])[:, :, ::-1].copy()
    dm = depth(img)
    X = read_ply(PLY[i])
    z = X[:, 2]
    ok = (z > 1) & (z < 30)
    u = np.clip(np.round(fx * X[ok, 0] / z[ok] + cx).astype(int), 0, dm.shape[1] - 1)
    v = np.clip(np.round(fy * X[ok, 1] / z[ok] + cy).astype(int), 0, dm.shape[0] - 1)
    r = dm[v, u] / z[ok]
    sD.append(np.median(r))
    g = (X[ok, 1] > 0.8) & (z[ok] < 12)                # below the camera, near: mostly road
    sG.append(np.median(r[g]) if g.sum() > 50 else np.nan)
    if j % 200 == 0:
        print(j, len(idx), f"scale {sD[-1]:.3f}", flush=True)
np.savez(os.path.join(ROOT, a.target, f"depth_{a.model}{a.tag}.npz"), frames=np.array(idx), s_all=np.array(sD), s_near_ground=np.array(sG))
print("saved", f"{a.target}/depth_{a.model}{a.tag}.npz")
