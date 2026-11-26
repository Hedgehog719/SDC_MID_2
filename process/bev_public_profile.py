"""Along-road BEV correction of the public opposite-pass pairs against road position, per pass (diagnostic).

Reads the --bev-audit dump (result/bevaudit_<run>.npz) and the run's trajectory, takes the fixed set of public pairs
accepted in round 0, and plots their forward correction re-registered under the final poses against a common road
coordinate (arc length along the westbound/first pass 1141-1344, every frame projected onto it).  Sign: for both pass
directions +x means "the opposite pass must move forward along the first pass's direction" (see NOTES).
Prints, per pass: median, |median|, robust sd, lag-1 autocorrelation, sd(diff)/(sqrt2 sd) (1 = white, << 1 = smooth),
running-median amplitude, adjacent same-ahead differences over the neighbour gate 0.4 h, and which raw pairs of the
last round the neighbour gate dropped.

usage: python process/bev_public_profile.py data/Track2/result/bevaudit_t2_audit_z16it3.npz data/Track2/result/t2_cam_z16_it3.csv out.png
"""
import sys
import numpy as np

H = 1.178                    # camera height [trajectory units] from the run's log (G*h = 0.047)
G = 0.04 * H
PASSES = (("first pass 1141-1344", 1141, 1345), ("second pass 2731-2947", 2731, 2948))


def load_traj(csv):
    m = np.loadtxt(csv, delimiter=",", skiprows=1)[:, 1:]
    return m.reshape(-1, 4, 4)[:, :3, 3]


def road_coord(C, f0, f1):
    """arc length along the polyline C[f0:f1]; returns s(p) for any points p (projection onto nearest segment)"""
    Q = C[f0:f1]
    seg = np.diff(Q, axis=0)
    L = np.linalg.norm(seg, axis=1)
    S0 = np.r_[0, np.cumsum(L)]

    def s(p):
        p = np.atleast_2d(p)
        t = np.clip(np.einsum("nkj,kj->nk", p[:, None] - Q[None, :-1], seg) / L ** 2, 0, 1)
        d = np.linalg.norm(p[:, None] - (Q[None, :-1] + t[..., None] * seg), axis=2)
        k = d.argmin(1)
        return S0[k] + t[np.arange(len(p)), k] * L[k], d[np.arange(len(p)), k]
    return s


def stats(x):
    med = np.median(x)
    rsd = 1.4826 * np.median(np.abs(x - med))
    sd = x.std()
    ac1 = np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 3 and sd > 0 else np.nan
    rough = np.diff(x).std() / (np.sqrt(2) * sd) if sd > 0 else np.nan
    return med, np.median(np.abs(x)), rsd, sd, ac1, rough


def main(npz, csv, png):
    d = np.load(npz)
    P, M, ok, ah = d["pairs"], d["meas"], d["ok"], d["ahead"]
    A = P[:, 0]
    C = load_traj(csv)
    s_of = road_coord(C, 1141, 1345)
    sa, da = s_of(C[A]); sb, db = s_of(C[P[:, 1]])
    smid = 0.5 * (sa + sb)
    fixed = d["in_round"][0]
    J = len(d["names"]) - 1                        # after last round

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ("#2a78d6", "#eb6834")
    mk = {10.0: "s", 14.0: "o", 18.0: "^"}
    CLIP = 2.5 * H                                 # register() searches +-3 h: |x| ~ 3.5 is the search boundary (alias)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw=dict(height_ratios=(3, 1.2)))
    R = len(d["in_round"]) - 1
    for (nm, f0, f1), c in zip(PASSES, col):
        m = fixed & (A >= f0) & (A < f1)
        g = m & ok[J]
        o = np.argsort(smid[g])
        x, y, a_, ahd = smid[g][o], M[J, g, 0][o], A[g][o], ah[g][o]
        edge = np.abs(y) > CLIP
        med, amed, rsd, sd, ac1, rough = stats(y[~edge])
        print(f"{nm}: {m.sum()} fixed pairs, {g.sum()} pass after last, {edge.sum()} at the search boundary; without those: forward median "
              f"{med:+.3f} (|.| {amed:.3f}), robust sd {rsd:.3f}, sd {sd:.3f}, lag-1 ac {ac1:+.2f}, sd(diff)/(sqrt2 sd) {rough:.2f}, "
              f"|x| > 0.15: {(np.abs(y[~edge]) > 0.15).sum()}")
        big = []
        for k in np.unique(ahd):
            mm = (ahd == k) & ~edge
            aa, yy = a_[mm], y[mm]
            oo = np.argsort(aa); aa, yy = aa[oo], yy[oo]
            big += [(int(k), int(aa[i]), int(aa[i + 1]), round(yy[i], 3), round(yy[i + 1], 3)) for i in range(len(aa) - 1)
                    if aa[i + 1] - aa[i] <= 5 and abs(yy[i + 1] - yy[i]) >= 0.4 * H]
        print(f"   adjacent (same ahead, da <= 5) differences >= 0.4 h = {0.4 * H:.3f}: {big}")
        drop = (A >= f0) & (A < f1) & d["raw_round"][R] & ~d["in_round"][R]
        print(f"   last round: raw {((A >= f0) & (A < f1) & d['raw_round'][R]).sum()}, in graph {((A >= f0) & (A < f1) & d['in_round'][R]).sum()}, "
              f"dropped by neighbour gate {drop.sum()}: forward {np.round(M[R, drop, 0], 3).tolist()} at a {A[drop].tolist()}")
        for k in mk:
            mm = (ahd == k) & ~edge
            if mm.any():
                ax[0].plot(x[mm], y[mm], mk[k], ms=6.5, mfc=c, mec="white", mew=1.0, ls="none", label=f"{nm}, ahead {k:.0f} h")
            mm = (ahd == k) & edge
            if mm.any():
                ax[0].plot(x[mm], np.sign(y[mm]) * 1.15, mk[k], ms=6.5, mfc="none", mec=c, mew=1.2, ls="none")
        k14 = (ahd == 14) & ~edge
        ax[0].plot(x[k14], y[k14], "-", color=c, lw=1.2, alpha=0.6)
        if drop.any():
            ax[0].plot(smid[drop], np.clip(M[R, drop, 0], -1.15, 1.15), "x", color=c, ms=7, mew=1.5)
        # which public frames of this pass appear in any pair (as a or b), any round
        anyp = d["raw_round"].any(0)
        fr = np.unique(np.r_[A[anyp], P[anyp, 1]])
        fr = fr[(fr >= f0) & (fr < f1)]
        sf, _ = s_of(C[f0:f1])
        inp = np.isin(np.arange(f0, f1), fr)
        lvl = 1 if f0 < 2000 else 0
        ax[1].plot(sf[~inp], np.full((~inp).sum(), lvl), "|", color="#bbb", ms=9)
        ax[1].plot(sf[inp], np.full(inp.sum(), lvl), "|", color=c, ms=9)
        print(f"   public frames of this pass in some pair (any round, raw): {inp.sum()} / {f1 - f0}; without: s {sf[~inp].min():.1f} .. {sf[~inp].max():.1f}")
    for k in range(-25, 26):
        ax[0].axhline(k * G, color="#e6e6e6", lw=0.4, zorder=0)
    ax[0].axhspan(-0.15, 0.15, color="#f0f0f0", zorder=0)
    ax[0].axhline(0, color="#999", lw=0.8)
    ax[0].set_ylim(-1.25, 1.25)
    ax[0].set_ylabel("forward correction, final poses [traj units]")
    ax[0].set_title("BEV public fixed pairs (round 0, 47) re-registered under the final poses\nline = ahead 14 h; hollow at +-1.15 = "
                    "search boundary +-3 h (alias); x = dropped by neighbour gate in the last round; faint lines = 1 cell 0.047", fontsize=8)
    ax[0].legend(fontsize=7, ncol=2, loc="lower left")
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["2731-2947", "1141-1344"]); ax[1].set_ylim(-0.6, 1.6)
    ax[1].set_ylabel("frames in a pair")
    ax[1].set_xlabel("road coordinate: arc length along 1141-1344 [traj units] (pairs at their midpoint, frames at their projection)")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print("saved", png)


if __name__ == "__main__":
    main(*sys.argv[1:4])
