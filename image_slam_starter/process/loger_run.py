"""Run the LoGeR model (pretrained, public; code + checkpoint from Code_LoGeR) on an image sequence in chunks that fit an
8 GB GPU, and keep only what we use: the camera poses it predicts (camera-to-world, per chunk) and the median depth of
every frame (to relate its scale to the given clouds).

The given raw_ply clouds are LoGeR's per-frame geometry with the poses removed; its poses are consistent over a whole
window / chunk (one scale per chunk), which our per-frame clouds are not.
usage: ../.venv-loger/bin/python process/loger_run.py Track1 [--chunk 160 --chunk-overlap 32 --window 32 --overlap 3]
out:   <track>/loger/chunk_<start>_<end>.npz  (frames, camera_poses (n,4,4), depth_med (n), conf_med (n))
"""
import argparse
import glob
import inspect
import math
import os
import sys
import time

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

LOGER = os.environ.get("LOGER_DIR", os.path.expanduser("~/sdc/mid/Code_LoGeR"))
sys.path.insert(0, os.path.join(LOGER, "vendor", "loger"))
from loger.models.pi3 import Pi3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("--chunk", type=int, default=160)
ap.add_argument("--chunk-overlap", type=int, default=32)
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--overlap", type=int, default=3)
ap.add_argument("--pixels", type=int, default=255000)
ap.add_argument("--bf16-weights", action="store_true", help="cast the weights to bfloat16 (halves the GPU memory)")
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--last", type=int, default=-1)
ap.add_argument("--name", default="loger", help="output folder name under <track>")
ap.add_argument("--save-depth", action="store_true", help="also keep every frame's depth map and confidence (float16 / uint8)")
ap.add_argument("--keep-center", action="store_true",
                help="sliding windows (--chunk-overlap = --chunk - 1): keep only each window's centre frame (the first / last "
                     "window also keep their first / last half), so every frame's depth comes from a window centred on it")
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = sorted(glob.glob(os.path.join(ROOT, args.target, "data", "images", "*.jpg")))
OUT = os.path.join(ROOT, args.target, args.name)
os.makedirs(OUT, exist_ok=True)

cfg = yaml.safe_load(open(os.path.join(LOGER, "vendor", "loger", "original_config.yaml")))
mc = cfg.get("model", {})
valid = {n for n, p in inspect.signature(Pi3.__init__).parameters.items() if n not in {"self", "args", "kwargs"}}
model = Pi3(**{k: v for k, v in mc.items() if k in valid})
sd = torch.load(os.path.join(LOGER, "latest.pt"), map_location="cpu", weights_only=False)
sd = sd.get("model_state_dict", sd)
model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}, strict=True)
del sd
if args.bf16_weights:
    model = model.to(torch.bfloat16)
model = model.cuda().eval()
print(f"model on GPU: {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
fw = dict(window_size=args.window, overlap_size=args.overlap, reset_every=cfg.get("training_settings", {}).get("reset_every", 0),
          num_iterations=cfg.get("num_iterations", 1), sim3=cfg.get("sim3", False), se3=bool(mc.get("se3", cfg.get("se3", False))),
          sim3_scale_mode="median", turn_off_ttt=False, turn_off_swa=False, keep_overlap=False, return_attn=False)
print("forward:", fw, flush=True)

W0, H0 = Image.open(IMG[0]).size
s = math.sqrt(args.pixels / (W0 * H0))
k_, m_ = round(W0 * s / 14), round(H0 * s / 14)
while (k_ * 14) * (m_ * 14) > args.pixels:
    if k_ / m_ > W0 / H0:
        k_ -= 1
    else:
        m_ -= 1
TW, TH = k_ * 14, m_ * 14
print(f"{len(IMG)} images -> {TW} x {TH}", flush=True)
tt = transforms.ToTensor()
last = len(IMG) if args.last < 0 else args.last
starts = list(range(args.first, max(last - args.chunk_overlap, args.first + 1), args.chunk - args.chunk_overlap))
for a in starts:
    b = min(a + args.chunk, last)
    f = os.path.join(OUT, f"chunk_{a:05d}_{b:05d}.npz")
    if os.path.exists(f):
        continue
    t0 = time.time()
    imgs = torch.stack([tt(Image.open(p).convert("RGB").resize((TW, TH), Image.Resampling.LANCZOS)) for p in IMG[a:b]])
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(imgs[None].cuda(), **fw)
    poses = pred["camera_poses"][0].float().cpu().numpy()
    z = pred["local_points"][0][..., 2].float()
    conf = torch.sigmoid(pred["conf"][0].float())[..., 0] if pred["conf"].dim() == 5 else torch.sigmoid(pred["conf"][0].float())
    dmed = torch.stack([zz[cc > 0.3].median() if (cc > 0.3).any() else zz.median() for zz, cc in zip(z, conf)]).cpu().numpy()
    rows = np.arange(b - a)
    if args.keep_center:
        c = (b - a) // 2
        rows = np.arange(0, c + 1) if a == starts[0] else (np.arange(c, b - a) if b == last else np.array([c]))
    extra = dict(depth=z[rows].cpu().numpy().astype(np.float16), conf=(conf[rows].clamp(0, 1) * 255).round().byte().cpu().numpy(),
                 dist=np.minimum(rows, b - a - 1 - rows)) if args.save_depth else {}
    np.savez(f, frames=np.arange(a, b)[rows], camera_poses=poses[rows], depth_med=dmed[rows], conf_med=conf.flatten(1).median(1).values.cpu().numpy()[rows], **extra)
    print(f"chunk {a}-{b}: {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    del pred, imgs, z, conf
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
