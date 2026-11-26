"""Close-range yellow centre-line dash endpoint registration across opposite passes (180-deg invariant).

On opposite-direction passes over a single dashed yellow centre line, the yellow paint is flat on the road
(zero 3D visible-side thickness bias) and visible from both passes at close range x in [2.3, 4.2] h.
Once the trajectory is pre-aligned within < 0.4 * dash_period (e.g. BA output), cross-correlating the
close-range yellow dash intensity profiles I_W(s) vs I_E(s) directly measures the along-track offset
lag = s_E - s_W along W's travel axis ax_s with +-3-6 cm accuracy and zero period aliasing.

Safety guardrails:
  1. Per-window auto-correlation estimates local dash_period; search lag strictly within min(max_lag, 0.40 * period).
  2. Requires peak correlation >= --min-corr (default 0.90) AND secondary-peak uniqueness margin >= 0.10.
  3. Contiguous-corridor continuity filter rejects isolated lag jumps > 0.30 u from neighboring windows.
  4. Emits cross-pass along-track constraints ((P_j - P_i) . dir - target) / sigma for pass_consensus.py / sfm_ba.py.

Usage:
  python process/dash_along.py Track2 TRAJ.csv --centerline CL.npz --out Track2/result/dash_along.npz
"""
import argparse
import os
import sys
import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import nhc

ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("traj")
ap.add_argument("--centerline", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cam", default="1035.94,1035.94,722.04,456.56,-0.059663,0.091900,0,0")
ap.add_argument("--mount", default="-0.21,-1.49")
ap.add_argument("--h", type=float, default=1.1341)
ap.add_argument("--max-dbl-dist", type=float, default=0.30)
ap.add_argument("--max-occ", type=float, default=0.80)
ap.add_argument("--radius", type=float, default=4.0)
ap.add_argument("--min-gap", type=int, default=150)
ap.add_argument("--win", type=int, default=45, help="window length in frames along first pass")
ap.add_argument("--stride", type=int, default=15, help="window stride in frames")
ap.add_argument("--min-frames", type=int, default=12, help="min valid single-dash frames per pass in window")
ap.add_argument("--min-corr", type=float, default=0.90)
ap.add_argument("--min-margin", type=float, default=0.08, help="min gap between best peak and 2nd distinct peak")
ap.add_argument("--max-lag", type=float, default=1.5, help="hard max lag [traj units]")
ap.add_argument("--sigma", type=float, default=0.05, help="per-window along-track uncertainty [traj units]")
ap.add_argument("--rs-readout-ms", type=float, default=26.0, help="top-to-bottom CMOS rolling-shutter readout time [ms] to compensate bottom-row road delay")
ap.add_argument("--max-pitch-nod", type=float, default=0.35, help="drop frames where |camera pitch vs road tangent| exceeds this [deg]")
ap.add_argument("--road-chord", action="store_true", help="project onto exact per-frame 3D road chord P(s+3.25h)-P(s) instead of fixed camera pitch")
ap.add_argument("--dbl-dil", type=int, default=0, help="dilate double-yellow mask by +-K frames to exclude single/double transitions")
a = ap.parse_args()

src = a.traj if os.path.exists(a.traj) else os.path.join(a.target, "result", a.traj)
lines = open(src).read().splitlines()
ts_raw = np.array([int(l.split(",", 1)[0]) for l in lines[1:]])
ts = (ts_raw - ts_raw[0]) * 1e-9
T = np.array([[float(x) for x in l.split(",")[1:]] for l in lines[1:]]).reshape(-1, 4, 4)
N = len(T)
P = T[:, :3, 3]
spd_u = np.linalg.norm(np.gradient(P, ts, axis=0), axis=1)  # speed [traj units / s]

c = [float(x) for x in a.cam.split(",")]
K = np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1.0]])
D = np.array(c[4:8] + [0.0])

z = np.load(a.centerline)
y0_all = z["y0"]
k_all = z["k"]
dbl = z["double"].copy()
if a.max_dbl_dist > 0:
    dbl[np.abs(dbl) > a.max_dbl_dist] = 0.0
is_dbl = dbl != 0
if a.dbl_dil > 0:
    is_dbl = np.convolve(is_dbl.astype(np.int32), np.ones(2 * a.dbl_dil + 1, dtype=np.int32), mode="same") > 0
single = (~np.isnan(y0_all)) & (~is_dbl) & (z["occ"] < a.max_occ)

tree = cKDTree(P)
up = np.zeros_like(P)
for i in range(0, N, 5):
    nb = tree.query_ball_point(P[i], 40.0)
    Q = P[nb] - P[nb].mean(0)
    n = np.linalg.svd(Q, full_matrices=False)[2][2]
    up[i:i + 5] = n if n @ (-T[i, :3, 1]) > 0 else -n

fw = np.gradient(P, axis=0)
fw = np.array([fw[max(0, i - 3):i + 4].mean(0) for i in range(N)])
slow = np.linalg.norm(fw, axis=1) < 1e-3
fw[slow] = T[slow, :3, 2]
fw -= (fw * up).sum(1)[:, None] * up
fw /= np.linalg.norm(fw, axis=1, keepdims=True) + 1e-12

# Self-calibrated camera-to-road basis
U_c = np.median(np.einsum("nji,nj->ni", T[:, :3, :3], up), axis=0)
U_c /= np.linalg.norm(U_c)
F0, L0, U0 = nhc.axes(*np.radians([float(x) for x in a.mount.split(",")]))
F_c = F0 - (F0 @ U_c) * U_c
F_c /= np.linalg.norm(F_c)
L_c = np.cross(U_c, F_c)

if a.road_chord:
    s_cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    F_w_all = np.zeros_like(P); L_w_all = np.zeros_like(P); U_w_all = np.zeros_like(P)
    for f in range(N):
        f2 = min(N - 1, int(np.searchsorted(s_cum, s_cum[f] + 3.25 * a.h)))
        if f2 <= f + 1 or s_cum[f2] - s_cum[f] < 1.0:
            fw_f = T[f, :3, :3] @ F_c
        else:
            fw_f = P[f2] - P[f]
            fw_f /= np.linalg.norm(fw_f)
        lw_f = np.cross(up[f], fw_f); lw_f /= np.linalg.norm(lw_f) + 1e-12
        uw_f = np.cross(fw_f, lw_f)
        F_w_all[f], L_w_all[f], U_w_all[f] = fw_f, lw_f, uw_f

if a.max_pitch_nod > 0:
    dP_g = np.gradient(P, axis=0)
    t_road = dP_g / np.maximum(np.linalg.norm(dP_g, axis=1, keepdims=True), 1e-9)
    cam_fwd = np.einsum("nij,j->ni", T[:, :3, :3], F_c)
    pitch_nod = np.degrees(np.arcsin(np.clip(((cam_fwd - t_road) * up).sum(1), -1, 1)))
    single &= np.abs(pitch_nod) <= a.max_pitch_nod

img_dir = os.path.join(a.target, "data", "images")
files = sorted(os.listdir(img_dir))

# Precompute per-frame 1D close-range yellow centre-line samples ONLY for frames participating in opposite-pass single-dash pairs
opp_partner = {}
for i in range(N):
    if not single[i]:
        continue
    nb = [j for j in tree.query_ball_point(P[i], a.radius)
          if j - i > a.min_gap and single[j] and (fw[i] @ fw[j]) < -0.95]
    if nb:
        opp_partner[i] = nb

needed_frames = set(opp_partner.keys())
for nb in opp_partner.values():
    needed_frames.update(nb)
needed_frames = sorted(needed_frames)
print(f"Opposite-pass single-dash frames: {len(opp_partner)} W-frames, {len(needed_frames)} total frames to sample", flush=True)

xs = np.arange(2.3, 4.2, 0.03)
dys = np.arange(-0.15, 0.151, 0.03)
gx, gdy = np.meshgrid(xs, dys, indexing="ij")

frame_pts_w = {}
frame_sig = {}
for idx_f, f in enumerate(needed_frames):
    im = cv2.imread(os.path.join(img_dir, files[f]))
    if im is None:
        continue
    im = im.astype(np.float32)
    y0f, kf = float(y0_all[f]), float(k_all[f])
    gy = y0f + kf * gx + gdy
    if a.road_chord:
        Fw, Lw, Uw = F_w_all[f], L_w_all[f], U_w_all[f]
        Fc_f, Lc_f, Uc_f = T[f, :3, :3].T @ Fw, T[f, :3, :3].T @ Lw, T[f, :3, :3].T @ Uw
    else:
        Fc_f, Lc_f, Uc_f = F_c, L_c, U_c
        Fw, Lw, Uw = T[f, :3, :3] @ Fc_f, T[f, :3, :3] @ Lc_f, T[f, :3, :3] @ Uc_f
    Pg = gx[..., None] * Fc_f + gy[..., None] * Lc_f - Uc_f
    uv = cv2.projectPoints(Pg.reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(gx.shape + (2,)).astype(np.float32)
    b = cv2.remap(im, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
    valid = b[..., 0] >= 0
    yel = np.minimum(b[..., 1], b[..., 2]) - b[..., 0]
    yel[~valid] = np.nan
    with np.errstate(all="ignore"):
        sig = np.nanmax(yel, axis=1) - np.nanpercentile(yel, 20, axis=1)
    pts_w = P[f] + a.h * (xs[:, None] * Fw + (y0f + kf * xs)[:, None] * Lw - Uw)
    ok = ~np.isnan(sig)
    frame_pts_w[f] = pts_w[ok]
    frame_sig[f] = sig[ok]

print(f"Sampled {len(frame_sig)} frames successfully.", flush=True)

# Helper: build 1D profile on s_grid and estimate auto-correlation period + cross-correlation lag
ds = 0.02

def make_profile(frs, origin, ax_s, s_grid):
    ps = np.zeros(len(s_grid), dtype=np.float64)
    pc = np.zeros(len(s_grid), dtype=np.float64)
    for f in frs:
        if f not in frame_sig:
            continue
        s_w = (frame_pts_w[f] - origin) @ ax_s
        idx = np.round((s_w - s_grid[0]) / ds).astype(int)
        ok = (idx >= 0) & (idx < len(s_grid))
        np.add.at(ps, idx[ok], frame_sig[f][ok])
        np.add.at(pc, idx[ok], 1.0)
    return np.where(pc > 0, ps / np.maximum(pc, 1.0), np.nan)

def estimate_period(prof):
    """Estimate local dash period [units] from first positive auto-correlation peak in [2.0, 9.0] u."""
    min_l, max_l = int(2.0 / ds), int(9.0 / ds)
    vals = []
    for lag in range(min_l, min(max_l, len(prof) // 2)):
        aa, bb = prof[:-lag], prof[lag:]
        m = ~np.isnan(aa) & ~np.isnan(bb)
        if m.sum() < 80:
            vals.append(-1.0)
            continue
        vals.append(float(np.corrcoef(aa[m], bb[m])[0, 1]))
    if not vals:
        return 5.0
    vals = np.array(vals)
    # Find local maxima above 0.5
    pk = int(np.argmax(vals))
    if vals[pk] > 0.4:
        return (min_l + pk) * ds
    return 5.0

# Sweep sliding windows over W-frames that have opposite partners
windows = []
w_starts = range(0, N - a.win + 1, a.stride)
for w0 in w_starts:
    w1 = w0 + a.win
    w_fr = [f for f in range(w0, w1) if f in opp_partner]
    if len(w_fr) < a.min_frames:
        continue
    # Check straightness of window (heading change < 12 deg)
    if (fw[w_fr[0]] @ fw[w_fr[-1]]) < np.cos(np.radians(12.0)):
        continue
    # Group opposite partners into contiguous E-pass blocks (avoid mixing 2 different E passes!)
    e_all = sorted(set(j for f in w_fr for j in opp_partner[f]))
    if len(e_all) < a.min_frames:
        continue
    # Split e_all into contiguous runs separated by > 40 frames
    runs = []
    cur = [e_all[0]]
    for j in e_all[1:]:
        if j - cur[-1] > 40:
            runs.append(cur); cur = [j]
        else:
            cur.append(j)
    runs.append(cur)

    for e_fr in runs:
        if len(e_fr) < a.min_frames:
            continue
        # Keep only w_fr that actually have a partner in this e_fr run
        e_set = set(e_fr)
        w_sub = [f for f in w_fr if any(j in e_set for j in opp_partner[f])]
        if len(w_sub) < a.min_frames:
            continue
        origin = P[w_sub].mean(0)
        ax_s = P[w_sub[-1]] - P[w_sub[0]]
        ax_s -= (ax_s @ up[w_sub[0]]) * up[w_sub[0]]
        ax_s /= np.linalg.norm(ax_s) + 1e-12
        sw = (P[w_sub] - origin) @ ax_s
        s_grid = np.arange(sw.min() - 2.0, sw.max() + 6.0, ds)
        pw = make_profile(w_sub, origin, ax_s, s_grid)
        pe = make_profile(e_fr, origin, ax_s, s_grid)
        period = estimate_period(pw)
        max_lag_u = min(a.max_lag, 0.40 * period)
        ml = int(max_lag_u / ds)
        lags = np.arange(-ml, ml + 1)
        corrs = np.full(len(lags), -2.0)
        for q, lag in enumerate(lags):
            aa, bb = (pw[:len(pw) - lag], pe[lag:]) if lag >= 0 else (pw[-lag:], pe[:len(pe) + lag])
            m = ~np.isnan(aa) & ~np.isnan(bb)
            if m.sum() < 100:
                continue
            sa, sb = aa[m].std(), bb[m].std()
            if sa < 1e-3 or sb < 1e-3:
                continue
            corrs[q] = float(np.corrcoef(aa[m], bb[m])[0, 1])
        pk = int(np.argmax(corrs))
        c1 = float(corrs[pk])
        lag_u = float(lags[pk] * ds)
        # Check uniqueness against secondary peaks at least 0.6 u away
        far = np.abs(lags * ds - lag_u) >= 0.6
        c2 = float(corrs[far].max()) if far.any() else -1.0
        margin = c1 - c2
        if c1 < a.min_corr or margin < a.min_margin:
            continue
        # Parabolic sub-bin refinement around peak
        if 0 < pk < len(corrs) - 1 and corrs[pk - 1] > 0 and corrs[pk + 1] > 0:
            denom = (corrs[pk - 1] - 2 * corrs[pk] + corrs[pk + 1])
            if abs(denom) > 1e-9:
                delta = 0.5 * (corrs[pk - 1] - corrs[pk + 1]) / denom
                lag_u += float(np.clip(delta, -0.5, 0.5)) * ds
        # Anchor pair: median W frame and its nearest E frame in e_fr
        wi = w_sub[len(w_sub) // 2]
        ej = min(e_fr, key=lambda j: np.linalg.norm(P[j] - P[wi]))
        windows.append(dict(w0=w_sub[0], w1=w_sub[-1], e0=e_fr[0], e1=e_fr[-1],
                            wi=wi, ej=ej, ax_s=ax_s, lag=lag_u, corr=c1, margin=margin,
                            period=period, w_sub=w_sub, e_fr=e_fr))

print(f"\nRaw accepted windows (corr >= {a.min_corr}, margin >= {a.min_margin}): {len(windows)}")
# Continuity filter across neighboring windows on the same corridor (|wi_a - wi_b| <= 2*stride and |ej_a - ej_b| <= 2*stride)
kept = []
for idx_w, w in enumerate(windows):
    nbs = [u["lag"] for k, u in enumerate(windows)
           if k != idx_w and abs(u["wi"] - w["wi"]) <= 2.5 * a.stride and abs(u["ej"] - w["ej"]) <= 2.5 * a.stride]
    if len(nbs) >= 1 and abs(w["lag"] - np.median(nbs)) > 0.25:
        print(f"  [DROP jump] W {w['w0']:4d}-{w['w1']:4d} E {w['e0']:4d}-{w['e1']:4d}: lag={w['lag']:+.3f} vs nb_med={np.median(nbs):+.3f}")
        continue
    kept.append(w)

print(f"Kept after continuity filter: {len(kept)} windows:")
for w in kept:
    print(f"  W {w['w0']:4d}-{w['w1']:4d} <-> E {w['e0']:4d}-{w['e1']:4d} (anchor {w['wi']:4d},{w['ej']:4d}): "
          f"lag(E-W)={w['lag']:+.3f} u ({w['lag']*1.5:+.3f} m), corr={w['corr']:.3f}, margin={w['margin']:.3f}, P={w['period']:.2f} u")

# Mean vertical row fraction (v - cy) / H for BEV x in [2.3, 4.2] h (~x_mid = 3.25 h)
pt_mid = 3.25 * F_c + 0.5 * L_c - U_c
v_mid = float(cv2.projectPoints(pt_mid.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)[0].ravel()[1])
rs_dt_bot = (a.rs_readout_ms * 1e-3) * (v_mid - c[3]) / 928.0  # seconds later than image center cy
print(f"Rolling-shutter bottom-row delay at x=3.25 h (v={v_mid:.1f} px, T_rs={a.rs_readout_ms:.1f} ms): {rs_dt_bot*1e3:+.2f} ms", flush=True)

# Emit per-window constraints: for each kept window, emit constraints for up to 5 evenly spaced W-E pairs inside the window
# scaled by sqrt(n_pairs_in_win) so each window carries total weight 1/sigma^2!
ii_l, jj_l, dir_l, tgt_l, sig_l, lag_l = [], [], [], [], [], []
for w in kept:
    # Pick evenly spaced W frames from w_sub (stride ~6 frames)
    ws = w["w_sub"][::max(1, len(w["w_sub"]) // 5)]
    pairs_w = []
    for wi in ws:
        ej = min(w["e_fr"], key=lambda j: np.linalg.norm(P[j] - P[wi]))
        pairs_w.append((wi, ej))
    sig_pair = a.sigma * np.sqrt(len(pairs_w))
    for wi, ej in pairs_w:
        # Subtract bottom-row rolling-shutter apparent lag: +(spd_W + spd_E) * rs_dt_bot
        rs_bias = (spd_u[wi] + spd_u[ej]) * rs_dt_bot
        lag_phys = w["lag"] - rs_bias
        cur = float((P[ej] - P[wi]) @ w["ax_s"])
        ii_l.append(wi); jj_l.append(ej); dir_l.append(w["ax_s"])
        tgt_l.append(cur - lag_phys); sig_l.append(sig_pair); lag_l.append(lag_phys)

np.savez(
    a.out,
    i=np.array(ii_l, dtype=np.int32),
    j=np.array(jj_l, dtype=np.int32),
    dir=np.array(dir_l, dtype=np.float64),
    target=np.array(tgt_l, dtype=np.float64),
    sigma=np.array(sig_l, dtype=np.float64),
    lag=np.array(lag_l, dtype=np.float64),
    kind=np.full(len(ii_l), 2, dtype=np.int8),
)
print(f"\nWrote {len(ii_l)} along-track dash constraints ({len(kept)} windows) to {a.out}")
