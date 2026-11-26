"""Thin vertical landmarks (pole, traffic light, traffic sign: Cityscapes classes 5-7) of every image with SegFormer-B2,
for bearing-only triangulation (process/pole_landmarks.py).  Per image, at half resolution: connected components of
P(pole-like) > 0.5 that are taller than wide; for each the base (lowest pixels) and top, in full-resolution pixels.
Saved in <track>/result/poles.npz: frame, xb, yb (base, median x of the lowest rows), xt, yt (top), height, width, prob.
usage: python process/segment_poles.py Track2 [--batch 8]   (GPU; transformers)"""
import argparse, glob, os
import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--min-h", type=int, default=20, help="minimum component height [half-res px]")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
names = sorted(glob.glob(os.path.join(ROOT, a.target, "data", "images", "*.jpg")))
proc = SegformerImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL).cuda().eval().half()
rows = []
trunks = []
for b in range(0, len(names), a.batch):
    imgs = [np.ascontiguousarray(cv2.imread(n)[:, :, ::-1]) for n in names[b:b + a.batch]]
    h, w = imgs[0].shape[:2]
    with torch.no_grad():
        x = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
        p = torch.softmax(model(pixel_values=x).logits.float(), 1)
        pv = torch.nn.functional.interpolate(p[:, 8:9], size=(h // 2, w // 2), mode="bilinear")[:, 0].cpu().numpy()   # vegetation
        p = torch.nn.functional.interpolate(p[:, 5:8].sum(1, keepdim=True), size=(h // 2, w // 2), mode="bilinear")[:, 0].cpu().numpy()
    for k, vk in enumerate(pv):
        # tree trunks: in the band between the horizon and the road (full-res rows 400-560) a trunk is a run of columns that
        # are mostly vegetation, separated from the next by background
        band = vk[200:280] > 0.5
        col = band.mean(0) > 0.7
        edges = np.flatnonzero(np.diff(np.r_[0, col.astype(np.int8), 0]))
        for c0, c1 in zip(edges[0::2], edges[1::2]):
            if 2 <= c1 - c0 <= 20:
                trunks.append((b + k, 2 * (c0 + c1) / 2.0, 2 * (c1 - c0), float(vk[200:280, c0:c1].mean())))
    for k, pk in enumerate(p):
        m = (pk > 0.5).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for c in range(1, n):
            x0, y0, ww, hh, area = st[c]
            if hh < a.min_h or hh < 2.5 * min(ww, 40):
                continue
            ys, xs = np.nonzero(lab[y0:y0 + hh, x0:x0 + ww] == c)
            ys = ys + y0; xs = xs + x0
            low = ys >= ys.max() - max(3, hh // 10); top = ys <= ys.min() + max(3, hh // 10)
            rows.append((b + k, 2 * np.median(xs[low]), 2 * ys.max(), 2 * np.median(xs[top]), 2 * ys.min(), 2 * hh, 2 * ww,
                         float(pk[ys, xs].mean())))
    if b % 400 == 0:
        print(b, "/", len(names), len(rows), flush=True)
r = np.array(rows, dtype=np.float64)
np.savez(os.path.join(ROOT, a.target, "result", "poles.npz"), frame=r[:, 0].astype(int), xb=r[:, 1], yb=r[:, 2], xt=r[:, 3], yt=r[:, 4],
         height=r[:, 5], width=r[:, 6], prob=r[:, 7])
t = np.array(trunks, dtype=np.float64)
np.savez(os.path.join(ROOT, a.target, "result", "trunks.npz"), frame=t[:, 0].astype(int), x=t[:, 1], width=t[:, 2], prob=t[:, 3])
print("saved", len(t), "trunk candidates")
print("saved", len(r), "components")
