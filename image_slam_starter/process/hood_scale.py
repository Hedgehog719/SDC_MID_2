"""Per-frame cloud scale from the car's hood: the hood is fixed to the camera, so its true depth is the same in every frame
and the depth the cloud gives it measures that frame's scale error (LoGeR treats the hood as road, but consistently).
s_i = median(z_hood) / z_hood,i  ->  <track>/depth_hood.npz (same format as process/depth_scale_check.py, --depth-prior hood).
usage: python process/hood_scale.py Track1 [--rows 912] [--half-width 250]"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--rows", type=float, default=912); ap.add_argument("--half-width", type=float, default=250)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, a.target, "data")
cam = json.load(open(f"{D}/camera_info.json")); fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
hz = []
for p in sorted(glob.glob(f"{D}/raw_ply/*.ply")):
    with open(p, "rb") as fh:
        n = 0
        while True:
            l = fh.readline().decode().strip()
            if l.startswith("element vertex"): n = int(l.split()[-1])
            if l == "end_header": break
        v = np.frombuffer(fh.read(n * dt.itemsize), dt, n)
    X = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    u = fx * X[:, 0] / X[:, 2] + cx; vv = fy * X[:, 1] / X[:, 2] + cy
    s = (vv > a.rows) & (np.abs(u - cam["width"] / 2) < a.half_width)
    hz.append(np.median(X[s, 2]) if s.sum() > 20 else np.nan)
hz = np.array(hz)
np.savez(os.path.join(ROOT, a.target, "depth_hood.npz"), frames=np.arange(len(hz)), s_all=np.nanmedian(hz) / hz)
print(f"{np.isfinite(hz).sum()} frames, hood depth median {np.nanmedian(hz):.2f}, spread {100*np.nanstd(np.log(hz)):.1f} %")
