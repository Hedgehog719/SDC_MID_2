"""LoGeR depth maps (process/loger_run.py --save-depth) -> per-frame clouds in the format of the given raw_ply:
binary PLY, camera frame (x right, y down, z forward), points = depth z of a network pixel back-projected with the
GIVEN pinhole K from the matching original-image pixel  u = (c + 0.5) * W / w - 0.5  (the given Track01 clouds are
exactly this with w x h = 630 x 392 and every 2nd pixel), colour from the image.
A frame that is in two chunks is taken from the chunk where it is further from the chunk ends.
usage: python process/loger_to_ply.py Track1 <loger folder name> <out target> [--sub 2]
       -> <out target>/data/raw_ply/*.ply  (images / imu / camera_info are linked from the source track)"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("name")
ap.add_argument("out")
ap.add_argument("--sub", type=int, default=2)
ap.add_argument("--chain", action="store_true",
                help="put all chunks on one scale: the depth ratio of consecutive chunks on their overlapping frames (LoGeR's scale "
                     "jumps by ~10 % (up to 29 %) from chunk to chunk, which the loop-free odometry cannot absorb)")
ap.add_argument("--match-given", action="store_true",
                help="multiply all depths by ONE global factor: the median ratio to the given raw_ply clouds (their 'roughly metre' unit)")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, a.source, "data")
IMG = sorted(glob.glob(f"{SRC}/images/*.jpg"))
cam = json.load(open(f"{SRC}/camera_info.json"))
fx, fy, cx, cy, W, H = cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["width"], cam["height"]
OUT = os.path.join(ROOT, a.out, "data")
os.makedirs(f"{OUT}/raw_ply", exist_ok=True)
for item in ("images", "imu.csv", "camera_info.json", "sample_submission.csv"):
    if os.path.exists(f"{SRC}/{item}") and not os.path.exists(f"{OUT}/{item}"):
        os.symlink(os.path.realpath(f"{SRC}/{item}"), f"{OUT}/{item}")
os.makedirs(os.path.join(ROOT, a.out, "result"), exist_ok=True)

best = {}                                    # frame -> (distance to the chunk ends, chunk file, row)
chunks = sorted(glob.glob(os.path.join(ROOT, a.source, a.name, "chunk_*.npz")))
for f in chunks:
    zf = np.load(f); fr = zf["frames"]
    for r, i in enumerate(fr):
        d = int(zf["dist"][r]) if "dist" in zf else min(r, len(fr) - 1 - r)
        if i not in best or d > best[i][0]:
            best[i] = (d, f, r)
CS = {f: 1.0 for f in chunks}                # scale factor of every chunk
if a.chain:
    prev = None
    for f in chunks:
        z_ = np.load(f); fr_ = z_["frames"]; d_ = z_["depth"]
        if prev is not None:
            pf, pfr, pd = prev
            common = np.intersect1d(pfr, fr_)[2:-2]
            r_ = [np.median(pd[np.where(pfr == i)[0][0]].astype(np.float64) / np.maximum(d_[np.where(fr_ == i)[0][0]].astype(np.float64), 1e-6))
                  for i in common]
            CS[f] = CS[pf] * float(np.median(r_))
        prev = (f, fr_, d_)
    print("chunk scales:", " ".join(f"{v:.3f}" for v in CS.values()))
dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])


def read_ply(p):
    with open(p, "rb") as fh:
        n = 0
        while True:
            l = fh.readline().decode().strip()
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
            if l == "end_header":
                break
        v = np.frombuffer(fh.read(n * dt.itemsize), dt, n)
    return np.c_[v["x"], v["y"], v["z"]].astype(np.float64)


GAIN = 1.0
if a.match_given:
    ratios = []
    for i in sorted(best)[::10]:
        _, f, r = best[i]
        z = np.load(f)["depth"][r].astype(np.float64) * CS[f]
        h, w = z.shape
        G = read_ply(os.path.join(SRC, "raw_ply", os.path.basename(IMG[i])[:-4] + ".ply"))
        u = fx * G[:, 0] / G[:, 2] + cx; v = fy * G[:, 1] / G[:, 2] + cy
        c = np.clip(np.round((u + 0.5) * w / W - 0.5).astype(int), 0, w - 1); rr_ = np.clip(np.round((v + 0.5) * h / H - 0.5).astype(int), 0, h - 1)
        ok = (G[:, 2] < 20) & (z[rr_, c] > 0)
        ratios.append(np.median(G[ok, 2] / z[rr_, c][ok]))
    GAIN = float(np.median(ratios))
    print(f"global depth factor to the given clouds: {GAIN:.4f} (frames: p10 {np.percentile(ratios, 10):.4f}, p90 {np.percentile(ratios, 90):.4f})")
cache = {}
for i in sorted(best):
    _, f, r = best[i]
    if f not in cache:
        cache.clear(); cache[f] = np.load(f)
    z = cache[f]["depth"][r].astype(np.float64) * CS[f] * GAIN
    h, w = z.shape
    cc, rr = np.meshgrid(np.arange(0, w, a.sub), np.arange(0, h, a.sub))
    zz = z[rr, cc]
    u = (cc + 0.5) * W / w - 0.5
    v = (rr + 0.5) * H / h - 0.5
    ok = np.isfinite(zz) & (zz > 0)
    img = cv2.imread(IMG[i])[:, :, ::-1]
    col = img[np.clip(np.round(v[ok]).astype(int), 0, H - 1), np.clip(np.round(u[ok]).astype(int), 0, W - 1)]
    pts = np.zeros(ok.sum(), dt)
    pts["x"] = (u[ok] - cx) / fx * zz[ok]; pts["y"] = (v[ok] - cy) / fy * zz[ok]; pts["z"] = zz[ok]
    pts["r"], pts["g"], pts["b"] = col[:, 0], col[:, 1], col[:, 2]
    name = os.path.basename(IMG[i])[:-4] + ".ply"
    with open(f"{OUT}/raw_ply/{name}", "wb") as fh:
        fh.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        fh.write(pts.tobytes())
print(f"{len(best)} frames -> {OUT}/raw_ply (grid {w // a.sub} x {h // a.sub} of {w} x {h})")
