"""Road / sidewalk / road-user probabilities of every image with SegFormer-B2 (Cityscapes; public pretrained model,
as in Midterm01's segment_road.py).  Saved at quarter resolution, uint8 0..255, in <track>/seg.npz:
  road = P(road), side = P(sidewalk), obj = P(person, rider, car, truck, bus, train, motorcycle, bicycle)
usage: python process/segment_road.py Track1 [--batch 8]   (GPU; transformers)"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
names = sorted(glob.glob(os.path.join(ROOT, a.target, "data", "images", "*.jpg")))
proc = SegformerImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL).cuda().eval().half()
road, side, obj = [], [], []
for b in range(0, len(names), a.batch):
    imgs = [np.ascontiguousarray(cv2.imread(n)[:, :, ::-1]) for n in names[b:b + a.batch]]
    h, w = imgs[0].shape[:2]
    with torch.no_grad():
        x = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
        p = torch.softmax(model(pixel_values=x).logits.float(), 1)
        p = torch.nn.functional.interpolate(p, size=(h // 4, w // 4), mode="bilinear")
    q = lambda t: (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    road.append(q(p[:, 0])); side.append(q(p[:, 1])); obj.append(q(p[:, 11:19].sum(1)))
    if b % 400 == 0:
        print(b, "/", len(names), flush=True)
np.savez_compressed(os.path.join(ROOT, a.target, "seg.npz"), names=np.array([os.path.basename(n) for n in names]),
                    road=np.concatenate(road), side=np.concatenate(side), obj=np.concatenate(obj))
print("saved", os.path.join(a.target, "seg.npz"))
