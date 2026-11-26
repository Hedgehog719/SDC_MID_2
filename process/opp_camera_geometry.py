"""For every public frame: the best opposite-pass camera and how far it is from what the BEV pair search needs.

Candidates b: any frame with |b - a| > 150 whose optical axis points into the opposite half-space (Za.Zb < 0); the
search's own heading gate (Za.Zb < -0.95, i.e. < 18 deg off antiparallel) is NOT applied.
  nearest : b closest in position to a                       (same piece of road, any facing)
  ahead14 : b minimising |forward - 14 h| + |lateral| in a    (the geometry a 14 h BEV pair needs)
Plots, against the road coordinate (arc length along 1141-1344): the heading deviation from antiparallel and the lateral
offset of b in a's camera frame [camera heights].

usage: python process/opp_camera_geometry.py data/Track2/result/t2_cam_z16_it3.csv out.png
"""
import sys
import numpy as np
from bev_public_profile import load_traj, road_coord, PASSES

H = 1.178


def main(csv, png):
    m = np.loadtxt(csv, delimiter=",", skiprows=1)[:, 1:].reshape(-1, 4, 4)
    R, C = m[:, :3, :3], m[:, :3, 3]
    X, Z = R[:, :, 0], R[:, :, 2]
    s_of = road_coord(C, 1141, 1345)
    idx = np.arange(len(C))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ("#2a78d6", "#eb6834")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    for (nm, f0, f1), c in zip(PASSES, col):
        fr = np.arange(f0, f1)
        s, _ = s_of(C[fr])
        out = {"nearest": [], "ahead14": []}
        for a in fr:
            cand = np.where((np.abs(idx - a) > 150) & (Z @ Z[a] < 0))[0]
            d = C[cand] - C[a]
            fwd, lat = d @ Z[a], d @ X[a]
            dev = 180 - np.degrees(np.arccos(np.clip(Z[cand] @ Z[a], -1, 1)))
            k1 = np.argmin(np.linalg.norm(d, axis=1))
            k2 = np.argmin(np.abs(fwd - 14 * H) + np.abs(lat))
            out["nearest"].append((cand[k1], dev[k1], lat[k1] / H, fwd[k1] / H))
            out["ahead14"].append((cand[k2], dev[k2], lat[k2] / H, fwd[k2] / H))
        for key, ls in (("nearest", "-"), ("ahead14", ":")):
            o = np.array(out[key])
            ax[0].plot(s, o[:, 1], ls, color=c, lw=1.6, label=f"{nm}, {key}")
            ax[1].plot(s, o[:, 2], ls, color=c, lw=1.6)
            print(f"{nm} {key}:")
            for a, (b, dv, lt, fw), ss in list(zip(fr, out[key], s))[::10] + [(fr[-1], out[key][-1], s[-1])]:
                print(f"   a {a} s {ss:5.1f}  b {int(b)}  heading off antiparallel {dv:5.1f} deg  lateral {lt:+5.1f} h  forward {fw:+5.1f} h")
            gate = o[:, 1] < 18
            print(f"   frames within the 18 deg gate: {gate.sum()} / {len(fr)}; > 30 deg: {(o[:, 1] > 30).sum()}; > 60 deg: {(o[:, 1] > 60).sum()}")
    for y in (18, 30, 60):
        ax[0].axhline(y, color="#999", lw=0.7, ls="--")
        ax[0].text(0.5, y + 1, f"{y} deg" + ("  (search gate)" if y == 18 else ""), fontsize=7, color="#666")
    ax[0].axvline(42, color="#ccc", lw=0.8); ax[1].axvline(42, color="#ccc", lw=0.8)
    ax[0].set_ylabel("heading off antiparallel [deg]")
    ax[0].set_title("best opposite-pass camera for each public frame (solid: nearest in position; dotted: best 14 h-ahead geometry, no heading gate)\n"
                    "grey line at s 42 = last public frame with a BEV pair (1240)", fontsize=8)
    ax[0].legend(fontsize=7, ncol=2)
    for y in (0,):
        ax[1].axhline(y, color="#999", lw=0.7)
    ax[1].set_ylabel("lateral of b in a [camera heights]")
    ax[1].set_xlabel("road coordinate: arc length along 1141-1344 [traj units]")
    for a_ in ax:
        a_.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print("saved", png)


if __name__ == "__main__":
    main(*sys.argv[1:3])
