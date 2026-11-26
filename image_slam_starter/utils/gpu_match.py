"""Mutual nearest-neighbour + ratio-test matching of RootSIFT descriptors with torch: on the GPU when there is
one (~100x faster than OpenCV brute force), otherwise the same code on the CPU, so that every stage of the
pipeline runs on both kinds of machine."""
import threading

import numpy as np

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:   # torch not installed
    torch, HAS_CUDA = None, False


class GpuMatcher:
    def __init__(self, descs, ratio=0.75, half=False, device=None):
        """descs: list of (Ni, 128) unit-norm float32 descriptors, one per image.
        half: store them as float16 (GPU memory); ignored on the CPU"""
        self.ratio = ratio
        self.dev = device or ("cuda" if HAS_CUDA else "cpu")
        half = half and self.dev == "cuda"
        self.lock = threading.Lock()    # one similarity matrix on the GPU at a time (8 GB card)
        # convert on the CPU before the copy: the GPU never holds the float32 version (peak 4 GB less on Track01,
        # so that several runs fit on the card); same rounding, same matches
        self.d = [(lambda t: t.half() if half else t)(torch.from_numpy(np.ascontiguousarray(x))).to(self.dev) for x in descs]

    @torch.no_grad() if torch is not None else (lambda f: f)
    def match(self, a, b):
        da, db = self.d[a], self.d[b]
        if len(da) < 8 or len(db) < 8:
            return np.zeros((0, 2), int)
        with self.lock:
            s = (da @ db.T).float()
            v, i = s.topk(2, dim=1)
            dist = (2 - 2 * v).clamp_min(0).sqrt()          # unit-norm descriptors
            ia = torch.arange(len(da), device=self.dev)
            ok = (dist[:, 0] < self.ratio * dist[:, 1]) & (s.argmax(0)[i[:, 0]] == ia)
            return torch.stack([ia[ok], i[ok, 0]], 1).cpu().numpy()
