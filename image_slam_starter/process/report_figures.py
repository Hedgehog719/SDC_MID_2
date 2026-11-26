"""Figures for the report (v9 / v10): lap-wise holdout results and the camera-height deviation along the route.
The holdout numbers are the ones in NOTES.md; the height profile is recomputed from a --dump-height run.
Usage: python process/report_figures.py [out_dir]   (default ../figures)"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "figures")
RES = os.path.join(ROOT, "Track1", "result")
os.makedirs(OUT, exist_ok=True)

# reference palette (light): first three categorical slots, neutral ink for text / the mean
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})
LAPS = ("lap 3 held out", "lap 2 held out", "lap 1 held out")

# ------------------------------------------------------------------ 1. iterations of the shared height correction
it = [1, 2, 3, 4, 5, 6, 8]
r = np.array([[0.367, 0.274, 0.544], [0.279, 0.271, 0.480], [0.245, 0.266, 0.403], [0.247, 0.265, 0.326],
              [0.271, 0.267, 0.245], [0.317, 0.273, 0.191], [0.416, 0.292, 0.199]])
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for j, (c, nm) in enumerate(zip((S1, S2, S3), LAPS)):
    ax.plot(it, r[:, j], color=c, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=nm)
ax.plot(it, r.mean(1), color=INK2, lw=2, ls="--", marker="o", ms=6, mec=SURF, mew=1.5, label="mean of the 3 laps")
ax.annotate(f"5 iterations: mean {r[4].mean():.3f} m,\nworst lap {r[4].max():.3f} m", xy=(5, r[4].mean()), xytext=(1.0, 0.175),
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.axhline(0.475, color=INK2, lw=1, ls=":")
ax.text(8, 0.482, "v7 default (mean 0.475)", ha="right", va="bottom", color=INK2, fontsize=9)
ax.set_xlabel("iterations of the shared height correction (--height-scale, with --vo --height-lap-gain 0)")
ax.set_ylabel("held-out loop pairs, rmse [m]")
ax.set_title("Lap-wise holdout vs. number of height iterations")
ax.set_xticks(it); ax.set_ylim(0.15, 0.6)
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/holdout_iterations.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 2. contribution of the ideas (mean of 3 laps)
cfg = [("v7 default: no VO, height-scale 1 (public 0.069)", 0.475),
       ("+ VO directions, height-scale 1", 0.450),
       ("+ VO directions, height-scale 0", 0.444),
       ("no VO, shared height correction x5", 0.372),
       ("+ VO, shared height correction x1", 0.395),
       ("v10: + VO, shared height correction x5 (public 0.067)", 0.261)]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
y = np.arange(len(cfg))[::-1]
ax.barh(y, [v for _, v in cfg], height=0.6, color=[S1 if i == len(cfg) - 1 else "#9fc1ea" for i in range(len(cfg))])
for yy, (nm, v) in zip(y, cfg):
    ax.text(v + 0.006, yy, f"{v:.3f}", va="center", color=INK, fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([nm for nm, _ in cfg], fontsize=9)
ax.set_xlim(0, 0.53); ax.grid(axis="y", visible=False)
ax.set_xlabel("held-out rmse [m], mean of 3 laps (noise ±0.005)")
ax.set_title("What each idea contributes")
fig.tight_layout(); fig.savefig(f"{OUT}/holdout_contributions.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 3. camera-height deviation along the route
def split(poses, h, win=41, radius=3.0):
    N = len(h); lap = np.searchsorted([474, 834], np.arange(N), side="right")
    g = np.isfinite(h)
    sm = median_filter(np.interp(np.arange(N), np.where(g)[0], h[g]), size=win, mode="nearest")
    ls = np.log(sm / np.median(sm)); C, Z = poses[:, :3, 3], poses[:, :3, 2]
    acc, cnt = ls.copy(), np.ones(N)
    for L in range(3):
        idx = np.where(lap == L)[0]
        d, j = cKDTree(C[idx]).query(C, distance_upper_bound=radius)
        jj = idx[np.minimum(j, len(idx) - 1)]
        ok = (lap != L) & np.isfinite(d) & (np.einsum("ij,ij->i", Z, Z[jj]) > np.cos(np.radians(30)))
        acc[ok] += ls[jj[ok]]; cnt[ok] += 1
    lc = acc / cnt
    return 100 * lc, 100 * (ls - lc)

dump = os.path.join(RES, "_dbg_hs.csv.hs.npz")            # any --dump-height run: iteration 0 does not depend on the mode
final = os.path.join(RES, "v10_vo_shared_it5.csv")
if os.path.exists(dump) and os.path.exists(final + ".camh.npy"):
    import pandas as pd
    d = np.load(dump)
    sh0, own0 = split(d["poses0"], d["h0"])
    P5 = pd.read_csv(final).iloc[:, 1:].to_numpy().reshape(-1, 4, 4)
    sh5, _ = split(P5, np.load(final + ".camh.npy"))
    fr = np.arange(len(sh0))
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.plot(fr, sh0, color=S1, lw=2, label=f"shared by the 3 passes, before ({sh0.std():.1f} %)")
    ax.plot(fr, own0, color=S2, lw=2, label=f"lap's own, before ({own0.std():.1f} %)")
    ax.plot(fr, sh5, color=S3, lw=2, label=f"shared, after 5 iterations ({sh5.std():.1f} %)")
    for b in (474, 834):
        ax.axvline(b, color=INK2, lw=1, ls=":")
    for x0, nm in ((0, "lap 1"), (474, "lap 2"), (834, "lap 3")):
        ax.text(x0 + 8, ax.get_ylim()[1] if False else 7.6, nm, color=INK2, fontsize=9, va="top")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-6.5, 8)
    ax.set_xlabel("frame        (std of each curve in brackets)"); ax.set_ylabel("camera height above the\ntriangulated road, vs. median [%]")
    ax.set_title("The height deviation belongs to the place, not to the lap")
    ax.legend(frameon=False, loc="lower center", fontsize=8.5, ncol=3)
    fig.tight_layout(); fig.savefig(f"{OUT}/height_profile.png", dpi=150); plt.close(fig)
print("figures in", os.path.abspath(OUT))
