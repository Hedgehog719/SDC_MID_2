"""Cross-pass consensus after the bundle adjustment: what the scene says about two passes over the same road.

Track02 has no IMU and, in several places (the public L corner among them), no image track or marking loop that links
one pass to another one over the same road; there the relative position of the passes is carried only by the odometry
chain between them.  Two things in the images hold for every pass and need no cross-pass matching:
  * height: the same car on the same road keeps its camera at the same height above the road
      -> frames of different passes at the same place have the same camera height (sigma --sigma-h);
  * lateral: the yellow centre line is one physical line; each frame measures its own distance y0 to it
      (process/centerline_sep.py: bird's-eye view, line fit, value at the camera, independent of pitch to first order)
      -> the centre-line points P + y0*h*left of two frames at the same place coincide across the road (sigma --sigma-l).
      h (camera height in trajectory units) is one number for the route, least squares over all pairs.
On an L-shaped junction the lateral constraint of one leg is the along-road constraint of the other, so the two legs
together fix the horizontal offset between the passes.
Unknowns: a 3-D correction d_k per frame (world frame), with
  along-track first difference ((d_k+1 - d_k) . f_k, sigma --sigma-step: keeps the step lengths the images fixed),
  lateral and vertical second differences (--curv: only slow bends), a weak prior d ~ 0 (--sigma0),
solved as one sparse linear least-squares problem with Huber re-weighting of the pair residuals.  Rotations unchanged.
usage: python process/pass_consensus.py Track2 IN.csv --out OUT [--centerline Track2/result/centerline_0-4206_-0.21_-1.49.npz]
"""
import argparse, json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("traj")
ap.add_argument("--out", required=True)
ap.add_argument("--centerline", default="", help="npz with y0, k per frame (centerline_sep.py); empty = height only")
ap.add_argument("--no-height", action="store_true")
ap.add_argument("--radius", type=float, default=3.5, help="same-place distance in the road plane [units]")
ap.add_argument("--min-gap", type=int, default=150, help="frames between two passes")
ap.add_argument("--par-deg", type=float, default=15.0, help="lateral pairs: headings parallel / antiparallel within this")
ap.add_argument("--sigma-h", type=float, default=0.02, help="height pair residual [units]")
ap.add_argument("--sigma-l", type=float, default=0.04, help="centre-line pair residual [units]")
ap.add_argument("--gate-l", type=float, default=0.6, help="drop centre-line pairs whose residual exceeds this (another line) [units]")
ap.add_argument("--h", type=float, default=0.0, help="camera height in trajectory units (0 = least squares over all pairs)")
ap.add_argument("--max-occ", type=float, default=0.8, help="opposite pairs: line occupancy below this (dashed)")
ap.add_argument("--gate-ref", default="", help="trajectory whose cross-pass loops are trusted (the SLAM output): a centre-line pair "
                "is used only if its residual there is below --gate-ref-thr (a wrongly detected line gives a large residual "
                "on every trajectory; a pass that drifted in the bundle adjustment does not)")
ap.add_argument("--gate-ref-thr", type=float, default=0.35)
ap.add_argument("--report", default="", help="frame ranges a:b,c:d,... for a per-segment residual report")
ap.add_argument("--joint", action="store_true",
                help="DIAGNOSTIC ONLY: estimate h and a centre-line offset beta = h*b together with the corrections and the noise of "
                     "each type by variance-component estimation.  Opposite pairs only see h*D, which a proportional lateral correction "
                     "imitates, so (h, beta, corrections) are not identifiable together: on Track02 it runs to h 0.06")
ap.add_argument("--h-auto", action="store_true",
                help="h from the measurements that do not involve the separation of opposite passes: same-direction centre-line "
                     "pairs (lat = h * (y0_i - y0_j), block bootstrap) combined with --h-prior (triangulated road plane) by inverse variance")
ap.add_argument("--cross-slope", type=float, default=0.0,
                help="height pairs: model error = this * horizontal distance of the pair (unknown cross slope / crown, e.g. 0.02)")
ap.add_argument("--line-sys", type=float, default=0.0, help="centre-line pairs: systematic line-detection error [camera heights], e.g. 0.02")
ap.add_argument("--dump-pairs", default="", help="write the cross-pass constraints (i, j, dir, target, sigma, kind) for sfm_ba.py --cross-pass")
ap.add_argument("--dump-sigma-scale", type=float, default=1.0, help="multiply dumped constraint sigmas by this factor (default 1.0)")
ap.add_argument("--dump-drop-kind", default="", help="comma-separated constraint kind IDs to omit from --dump-pairs (0=height, 1=centre line, 2=dash along)")
ap.add_argument("--cv-folds", type=int, default=0, help="choose --sigma-step / --curv by K-fold cross-validation over the pairs")
ap.add_argument("--cv-step", default="0.001,0.003,0.01")
ap.add_argument("--cv-curv", default="5e-5,2e-4,1e-3", help="lateral (and vertical unless --cv-curv-up) curvature grid")
ap.add_argument("--cv-curv-up", default="", help="separate vertical curvature grid")
ap.add_argument("--curv-up", type=float, default=0.0, help="vertical second difference (0 = same as --curv)")
ap.add_argument("--sigma-auto", action="store_true",
                help="noise of each pair type from the measurement's own frame-to-frame jitter (true lateral offsets and height gaps "
                     "vary slowly), independent of the corrections")
ap.add_argument("--h-prior", default="", help="joint: independent measurement of h, 'mean,sigma' (e.g. the triangulated road plane)")
ap.add_argument("--beta-sigma", type=float, default=0.035, help="joint: prior sigma of beta (physics: the intercept at the camera is unbiased)")
ap.add_argument("--vce", type=int, default=6, help="joint: iterations of Huber re-weighting + variance-component estimation")
ap.add_argument("--sigma-step", type=float, default=0.003, help="along-track change of the correction per frame [units]")
ap.add_argument("--curv", type=float, default=2e-4, help="lateral / vertical second difference per frame^2 [units]")
ap.add_argument("--sigma0", type=float, default=1.0)
ap.add_argument("--huber", type=float, default=2.0, help="Huber threshold in sigmas")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--line-normal", action="store_true", help="use true BEV line normal in world frame instead of trajectory tangent normal")
ap.add_argument("--mount", default="-0.21,-1.49", help="camera mount yaw,pitch [deg] for --line-normal")
ap.add_argument("--max-dbl-dist", type=float, default=0.0, help="treat |double| > this as 0.0 (false twin detection filter)")
ap.add_argument("--turn-stiff", type=float, default=0.0, help="lateral first-difference stiffness in sharp turns")
ap.add_argument("--road-normal", action="store_true", help="height pairs use per-frame road normal R_k U_mount (captures local cross-slope & grade)")
ap.add_argument("--vertical-only", action="store_true", help="restrict correction d_k to vertical (along up[k]) only, keeping horizontal path unchanged")
ap.add_argument("--dash-along", default="", help="npz of process/dash_along.py: opposite-pass close-range yellow dash along-track constraints")
ap.add_argument("--opp-dbl-dil", type=int, default=0, help="dilate double-yellow mask by +-K frames for opposite-pass single-line filter")
ap.add_argument("--fix-huber", action="store_true", help="evaluate Huber residuals in true per-pair sigmas (dividing out frame-count weight w)")
a = ap.parse_args()

src = a.traj if os.path.exists(a.traj) else os.path.join(a.target, "result", a.traj)
lines = open(src).read().splitlines()
stamps = [l.split(",", 1)[0] for l in lines[1:]]
T = np.array([[float(x) for x in l.split(",")[1:]] for l in lines[1:]]).reshape(-1, 4, 4)
N = len(T); P = T[:, :3, 3].copy()

# local frames: up = normal of the positions within 40 units (oriented like camera -y); f = direction of travel; l = up x f
tree = cKDTree(P)
up = np.zeros_like(P)
for i in range(0, N, 5):
    nb = tree.query_ball_point(P[i], 40.0); Q = P[nb] - P[nb].mean(0)
    n = np.linalg.svd(Q, full_matrices=False)[2][2]
    up[i:i + 5] = n if n @ (-T[i, :3, 1]) > 0 else -n
fw = np.gradient(P, axis=0)
fw = np.array([fw[max(0, i - 3):i + 4].mean(0) for i in range(N)])
slow = np.linalg.norm(fw, axis=1) < 1e-3
fw[slow] = T[slow, :3, 2]                                    # standing still: camera axis
fw -= (fw * up).sum(1)[:, None] * up; fw /= np.linalg.norm(fw, axis=1, keepdims=True)
lf = np.cross(up, fw)

# candidate pairs between passes
PI, PJ = [], []
for i in range(0, N, a.stride):
    for j in tree.query_ball_point(P[i], a.radius):
        if j - i > a.min_gap:
            PI.append(i); PJ.append(j)
PI, PJ = np.array(PI, int), np.array(PJ, int)
uij = up[PI] + up[PJ]; uij /= np.linalg.norm(uij, axis=1, keepdims=True)
dP = P[PJ] - P[PI]
hgap = (dP * uij).sum(1)
horiz = np.linalg.norm(dP - hgap[:, None] * uij, axis=1)
inr = horiz < a.radius
PI, PJ, uij, dP, hgap, horiz = PI[inr], PJ[inr], uij[inr], dP[inr], hgap[inr], horiz[inr]

rows = []            # (coefficient triples) built as sparse blocks below
blocks_A, blocks_b, blocks_w0, kinds = [], [], [], []


def pair_block(ii, jj, dirs_i, dirs_j, r0, sig):
    """residual r0 + d_j . dirs_j - d_i . dirs_i, scaled by 1/sig"""
    m = len(ii); rr = np.repeat(np.arange(m), 3)
    sig = np.broadcast_to(np.asarray(sig, float), (m,)); s3 = np.repeat(sig, 3)        # one sigma per pair
    cols_j = (3 * jj[:, None] + np.arange(3)).ravel(); cols_i = (3 * ii[:, None] + np.arange(3)).ravel()
    A = sp.csr_matrix((np.r_[dirs_j.ravel() / s3, -dirs_i.ravel() / s3], (np.r_[rr, rr], np.r_[cols_j, cols_i])), shape=(m, 3 * N))
    return A, -r0 / sig


info = dict(argv=sys.argv)
H_SD = 0.0                                 # uncertainty of h (set by --h-auto)
cnt = np.bincount(PI, minlength=N)


def jitter_sigma(series_idx, series_val, max_jump):
    """noise of a slowly varying series from its second difference: e = v_k - (v_k-1 + v_k+1)/2, var(e) = 1.5 sigma^2"""
    o = np.argsort(series_idx); i_, v_ = series_idx[o], series_val[o]
    ok_ = (np.diff(i_)[:-1] <= max_jump) & (np.diff(i_)[1:] <= max_jump)
    e = v_[1:-1] - (v_[:-2] + v_[2:]) / 2
    e = e[ok_]
    return float(1.4826 * np.median(np.abs(e - np.median(e))) * np.sqrt(2 / 3)), int(len(e))


if a.road_normal and not a.no_height:
    U_cam_all = np.einsum("nji,nj->ni", T[:, :3, :3], up)
    U_mount = np.median(U_cam_all, axis=0); U_mount /= np.linalg.norm(U_mount)
    n_road = np.einsum("nij,j->ni", T[:, :3, :3], U_mount)
    n_road /= np.linalg.norm(n_road, axis=1, keepdims=True)
    h_dir = n_road[PI] + n_road[PJ]; h_dir /= np.linalg.norm(h_dir, axis=1, keepdims=True)
    h_gap = (dP * h_dir).sum(1)
    print(f"road-normal height: U_mount={np.round(U_mount, 5).tolist()}, rms {np.sqrt((hgap**2).mean()):.4f} -> {np.sqrt((h_gap**2).mean()):.4f} units")
    info["U_mount"] = U_mount.tolist()
else:
    h_dir, h_gap = uij, hgap

if a.sigma_auto and not a.no_height:
    # one value per first frame: the median gap over its partners (the pair weights make one frame one measurement)
    ui = np.unique(PI); gmed = np.array([np.median(h_gap[PI == q]) for q in ui])
    s_, n_ = jitter_sigma(ui, gmed, 2 * a.stride)
    print(f"height pairs: noise from the gap's frame-to-frame jitter {s_:.4f} units ({n_} triples); --sigma-h {a.sigma_h} -> {s_:.4f}")
    info["sigma_h_auto"] = s_; a.sigma_h = s_
if not a.no_height:
    w = 1.0 / np.sqrt(np.maximum(cnt[PI], 1))
    sig_h = np.sqrt(a.sigma_h ** 2 + (a.cross_slope * horiz) ** 2)        # white noise + unknown cross slope over the gap
    A, b = pair_block(PI, PJ, h_dir * w[:, None], h_dir * w[:, None], h_gap * w, sig_h)
    blocks_A.append(A); blocks_b.append(b); kinds.append(("height", len(PI), h_gap, PI, PJ, h_dir, w, sig_h))
    info["height_pairs"] = int(len(PI)); info["height_gap_rms_before"] = float(np.sqrt((h_gap ** 2).mean()))

if a.centerline:
    z = np.load(a.centerline); y0 = z["y0"] / np.sqrt(1 + z["k"] ** 2)
    ok = ~np.isnan(y0)
    cosang = (fw[PI] * fw[PJ]).sum(1)
    sel = ok[PI] & ok[PJ] & (np.abs(cosang) > np.cos(np.radians(a.par_deg)))
    if "double" in z.files:
        # opposite passes see the same line only if it is one line: single (no parallel twin) and dashed;
        # passes in the same direction see the nearer line of a double line too, as long as both frames agree on the type
        dbl_val = z["double"].copy()
        if a.max_dbl_dist > 0:
            dbl_val[np.abs(dbl_val) > a.max_dbl_dist] = 0.0
        dbl = dbl_val != 0
        dbl_opp = dbl.copy()
        for sh in range(1, a.opp_dbl_dil + 1):
            dbl_opp[sh:] |= dbl[:-sh]
            dbl_opp[:-sh] |= dbl[sh:]
        single = (~dbl_opp) & (z["occ"] < a.max_occ)
        opp = cosang < 0
        sel &= np.where(opp, single[PI] & single[PJ], dbl[PI] == dbl[PJ])
        print(f"centre-line pairs: {int((sel & opp).sum())} opposite (single dashed line), {int((sel & ~opp).sum())} same direction")
    li, lj = PI[sel], PJ[sel]
    if a.line_normal:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import nhc
        F, L, U = nhc.axes(*np.radians([float(x) for x in a.mount.split(",")]))
        Fw = np.einsum("nij,j->ni", T[:, :3, :3], F)
        Fw -= (Fw * up).sum(1)[:, None] * up; Fw /= np.linalg.norm(Fw, axis=1, keepdims=True) + 1e-12
        Lw = np.cross(up, Fw)
        kk = np.nan_to_num(z["k"], nan=0.0)
        Nw = (Lw - kk[:, None] * Fw) / np.sqrt(1 + kk[:, None] ** 2)
        sgn = np.where((Nw[lj] * Nw[li]).sum(1) >= 0, 1.0, -1.0)
        Nij = Nw[li] + sgn[:, None] * Nw[lj]; Nij /= np.linalg.norm(Nij, axis=1, keepdims=True) + 1e-12
        lat = (dP[sel] * Nij).sum(1)
        Dm = y0[li] - sgn * y0[lj]
    else:
        lat = (dP[sel] * lf[li]).sum(1)                                       # (P_j - P_i) . l_i
        Dm = y0[li] - y0[lj] * (lf[lj] * lf[li]).sum(1)                       # centre line: lat = h * Dm
    h = a.h if a.h > 0 else float((lat * Dm).sum() / (Dm * Dm).sum())
    res = lat - h * Dm
    g = np.abs(res) < a.gate_l
    if a.gate_ref:
        Tr = np.array([[float(x) for x in l.split(",")[1:]] for l in open(a.gate_ref).read().splitlines()[1:]]).reshape(-1, 4, 4)
        Pr = Tr[:, :3, 3]
        fr = np.gradient(Pr, axis=0); fr = np.array([fr[max(0, i - 3):i + 4].mean(0) for i in range(N)])
        fr -= (fr * up).sum(1)[:, None] * up; fr /= np.linalg.norm(fr, axis=1, keepdims=True) + 1e-12; lr = np.cross(up, fr)
        lat_ref = ((Pr[lj] - Pr[li]) * lr[li]).sum(1); D_ref = y0[li] - y0[lj] * (lr[lj] * lr[li]).sum(1)
        h_ref = float((lat_ref * D_ref).sum() / (D_ref * D_ref).sum())            # the reference's own unit: the gate only
        res_ref = lat_ref - h_ref * D_ref                                          # tests whether a pair fits the reference
        g &= np.abs(res_ref) < a.gate_ref_thr
        print(f"gate on the reference trajectory (its unit {h_ref:.3f}): {int(g.sum())} of {len(g)} pairs kept (|residual there| < {a.gate_ref_thr})")
    if a.h <= 0 and not a.h_auto:                                          # refit h on the gated pairs
        h = float((lat[g] * Dm[g]).sum() / (Dm[g] * Dm[g]).sum()); res = lat - h * Dm; g = np.abs(res) < a.gate_l
    if a.h_auto:
        # same-direction pairs: lat = h * (y0_i - y0_j); pairs along one revisit are correlated -> bootstrap over 20-frame blocks
        same = g & ((lf[lj] * lf[li]).sum(1) > 0) & (np.abs(Dm) > 0.15)
        idx = np.where(same)[0]; blk = li[idx] // 20; ub = np.unique(blk)
        h_sd_dir = float((lat[idx] * Dm[idx]).sum() / (Dm[idx] ** 2).sum())
        rng_b = np.random.default_rng(0); boots = []
        for _ in range(400):
            pick = rng_b.choice(ub, len(ub)); sel_ = np.concatenate([idx[blk == q] for q in pick])
            boots.append((lat[sel_] * Dm[sel_]).sum() / (Dm[sel_] ** 2).sum())
        s_sd_dir = float(np.std(boots))
        parts = [(h_sd_dir, s_sd_dir, f"same-direction pairs ({len(idx)} in {len(ub)} blocks)")]
        if a.h_prior:
            hm, hs = (float(x) for x in a.h_prior.split(",")); parts.append((hm, hs, "--h-prior"))
        wsum = sum(1 / s ** 2 for _, s, _ in parts)
        h = float(sum(v / s ** 2 for v, s, _ in parts) / wsum)
        print("camera height from independent measurements: " + "; ".join(f"{nm} {v:.4f} +- {s:.4f}" for v, s, nm in parts)
              + f" -> h {h:.4f} +- {np.sqrt(1 / wsum):.4f}")
        info.update(h_auto=h, h_auto_sd=float(np.sqrt(1 / wsum)), h_same_direction=h_sd_dir, h_same_direction_sd=s_sd_dir)
        H_SD = float(np.sqrt(1 / wsum))
        res = lat - h * Dm
        g &= np.abs(res) < a.gate_l
    if a.report:
        oppg = (lf[lj] * lf[li]).sum(1) < 0
        for r_ in a.report.split(","):
            s0, s1 = (int(x) for x in r_.split(":"))
            for nm, mm in (("opposite", oppg), ("same", ~oppg)):
                m = g & mm & (li >= s0) & (li < s1)
                if m.sum() >= 10:
                    print(f"   report {s0}-{s1} {nm:8s}: n {m.sum():5d}, residual median {np.median(res[m]):+.3f} u"
                          + (f", lat/D {np.median(lat[m] / Dm[m]):.3f}" if nm == "opposite" else ""))
    if a.sigma_auto:
        fr_ok = np.where(ok)[0]
        s_y0, n_ = jitter_sigma(fr_ok, y0[fr_ok], 1)
        s_l = h * np.sqrt(2) * s_y0
        print(f"centre line: y0 noise from frame-to-frame jitter {s_y0:.4f} h ({n_} triples) -> pair sigma {s_l:.4f} units (was {a.sigma_l})")
        info.update(sigma_y0=s_y0, sigma_l_auto=float(s_l)); a.sigma_l = float(s_l)
    oppm = ((lf[lj] * lf[li]).sum(1) < 0)[g]
    Dg = Dm[g]
    li, lj, res = li[g], lj[g], res[g]
    dir_l = Nij[g] if a.line_normal else lf[li]
    cl = np.bincount(li, minlength=N)
    w = 1.0 / np.sqrt(np.maximum(cl[li], 1))
    # residual new = res + (d_j - d_i) . l_i  [- dh * D + 2 beta * opposite  with --joint]
    sig_l = np.sqrt(a.sigma_l ** 2 + (H_SD * Dg) ** 2 + (a.line_sys * h) ** 2)   # white + unit uncertainty * D + line detection
    A, b = pair_block(li, lj, dir_l * w[:, None], dir_l * w[:, None], res * w, sig_l)
    blocks_A.append(A); blocks_b.append(b); kinds.append(("centre line", len(li), res, li, lj, dir_l, w, sig_l))
    CL_GLOBAL = (-Dg * w / sig_l, 2.0 * oppm * w / sig_l)                  # columns of dh and beta in the centre-line block
    info.update(h=h, centre_pairs=int(len(li)), centre_res_rms_before=float(np.sqrt((res ** 2).mean())))
    print(f"centre line: camera height {h:.4f} units; {len(li)} pairs (gate {a.gate_l}); residual rms {np.sqrt((res**2).mean()):.3f} units")

if a.dash_along:
    zd = np.load(a.dash_along)
    di, dj, ddir, dtgt, dsig = zd["i"].astype(int), zd["j"].astype(int), zd["dir"].astype(float), zd["target"].astype(float), zd["sigma"].astype(float)
    dres = ((P[dj] - P[di]) * ddir).sum(1) - dtgt
    dw = np.ones(len(di), dtype=float)
    Ad, bd = pair_block(di, dj, ddir, ddir, dres, dsig)
    blocks_A.append(Ad); blocks_b.append(bd); kinds.append(("dash along", len(di), dres, di, dj, ddir, dw, dsig))
    info.update(dash_pairs=int(len(di)), dash_res_rms_before=float(np.sqrt((dres ** 2).mean())))
    print(f"dash along: {len(di)} pairs; residual rms {np.sqrt((dres**2).mean()):.3f} units (|.| median {np.median(np.abs(dres)):.3f})")

npair_rows = sum(B.shape[0] for B in blocks_A)
if a.dump_pairs:
    # each constraint: ((c_j - c_i) . dir - target) / sigma, camera centres in this trajectory's frame and units;
    # sigma includes the pair weight (one frame with many partners counts as one measurement)
    parts = []
    kind_id = {"height": 0, "centre line": 1, "dash along": 2}
    drop_kinds = {int(x) for x in a.dump_drop_kind.split(",") if x.strip()}
    for name, m, r0, ii, jj, dirs, w, sig in kinds:
        kid = kind_id.get(name, 1)
        if kid in drop_kinds:
            continue
        if name == "height":
            tgt = np.zeros(m)
        else:                                            # centre line / dash along
            tgt = ((P[jj] - P[ii]) * dirs).sum(1) - r0
        parts.append((ii, jj, dirs, tgt, (np.asarray(sig) / np.asarray(w)) * float(a.dump_sigma_scale), np.full(m, kid)))
    cat = [np.concatenate(x) for x in zip(*parts)]
    np.savez(a.dump_pairs, i=cat[0].astype(np.int32), j=cat[1].astype(np.int32), dir=cat[2], target=cat[3], sigma=cat[4],
             kind=cat[5].astype(np.int8), h=h if a.centerline else 0.0)
    print(f"wrote {len(cat[0])} cross-pass constraints to {a.dump_pairs} (sigma_scale={a.dump_sigma_scale}, drop_kind={sorted(drop_kinds)})")


def make_R(sig_step, curv_lat, curv_up):
    """regularisers: along-track first difference, lateral / vertical second differences, weak prior"""
    k = np.arange(N - 1)
    rr = np.repeat(k, 3)
    cols_step = np.r_[(3 * (k + 1)[:, None] + np.arange(3)).ravel(), (3 * k[:, None] + np.arange(3)).ravel()]
    step = sp.csr_matrix((np.r_[fw[k].ravel(), -fw[k].ravel()] / sig_step,
                          (np.r_[rr, rr], cols_step)), shape=(N - 1, 3 * N))
    extra_R = []
    if a.turn_stiff > 0:
        dfw = np.linalg.norm(np.diff(fw, axis=0), axis=1)
        wt = np.clip((dfw - 0.005) * a.turn_stiff, 0.0, 1.0) / sig_step
        lfw = lf[k] * wt[:, None]
        extra_R.append(sp.csr_matrix((np.r_[lfw.ravel(), -lfw.ravel()], (np.r_[rr, rr], cols_step)), shape=(N - 1, 3 * N)))
    if a.vertical_only:
        # Lock horizontal (along-track and lateral) components of d_k to zero
        k_all = np.arange(N); rr_all = np.repeat(k_all, 3); cols_all = (3 * k_all[:, None] + np.arange(3)).ravel()
        for ax_h in (fw, lf):
            extra_R.append(sp.csr_matrix((ax_h.ravel() / 1e-6, (rr_all, cols_all)), shape=(N, 3 * N)))
    k2 = np.arange(1, N - 1)
    curv = []
    for axis, curv_ in ((lf, curv_lat), (up, curv_up)):
        rr2 = np.repeat(np.arange(N - 2), 3)
        dvals = np.r_[axis[k2].ravel(), -2 * axis[k2].ravel(), axis[k2].ravel()] / curv_
        cols = np.r_[(3 * (k2 - 1)[:, None] + np.arange(3)).ravel(), (3 * k2[:, None] + np.arange(3)).ravel(), (3 * (k2 + 1)[:, None] + np.arange(3)).ravel()]
        curv.append(sp.csr_matrix((dvals, (np.r_[rr2, rr2, rr2], cols)), shape=(N - 2, 3 * N)))
    prior = sp.identity(3 * N, format="csr") / a.sigma0
    return sp.vstack([step] + extra_R + curv + [prior]).tocsr()


if a.cv_folds and not a.joint:
    # choose the smoothing by K-fold cross-validation over the cross-pass measurements themselves (no public score):
    # folds = 100-frame chunks of the first frame, so a held-out chunk must be predicted from the corrections around it
    from scipy.sparse.linalg import factorized as _fz
    folds = [(np.asarray(k_[3]) // 100) % a.cv_folds for k_ in kinds]
    best = None
    print(f"cross-validation, {a.cv_folds} folds: held-out residual rms in sigmas (" + ", ".join(k_[0] for k_ in kinds) + ")")
    import itertools
    cu_list = a.cv_curv_up.split(",") if a.cv_curv_up else [None]
    for s_step, c_, cu_ in itertools.product((float(x) for x in a.cv_step.split(",")), [float(x) for x in a.cv_curv.split(",")],
                                             [float(x) if x else None for x in cu_list]):
        if True:
            cu_ = c_ if cu_ is None else cu_
            Rc = make_R(s_step, c_, cu_); RtRc = (Rc.T @ Rc).tocsc()
            sq = [[] for _ in kinds]
            for f_ in range(a.cv_folds):
                AtA = RtRc.copy(); Atb = np.zeros(3 * N)
                for B, bb, fo in zip(blocks_A, blocks_b, folds):
                    tr_ = (fo != f_).astype(float)
                    AtA = AtA + (B.T @ sp.diags(tr_) @ B); Atb += B.T @ (tr_ * bb)
                dd = _fz(AtA.tocsc())(Atb)
                for q, (B, bb, fo, k_) in enumerate(zip(blocks_A, blocks_b, folds, kinds)):
                    ho = fo == f_
                    r_sig = (B[ho] @ dd - bb[ho]) / np.asarray(k_[6])[ho]          # per-pair residual in sigmas
                    sq[q].append(np.minimum(r_sig ** 2, 9.0))                        # robust: clip at 3 sigma
            rms = [float(np.sqrt(np.concatenate(s).mean())) for s in sq]
            score = float(np.mean(rms))
            print(f"   step {s_step:g}, curvature lateral {c_:g} vertical {cu_:g}: " + ", ".join(f"{v:.3f}" for v in rms) + f"  -> {score:.3f}", flush=True)
            if best is None or score < best[0]:
                best = (score, s_step, c_, cu_)
    a.sigma_step, a.curv, a.curv_up = best[1], best[2], best[3]
    print(f"cross-validation choice: --sigma-step {a.sigma_step:g} --curv {a.curv:g} --curv-up {a.curv_up:g}")
    info.update(cv_sigma_step=a.sigma_step, cv_curv=a.curv, cv_curv_up=a.curv_up, cv_score=best[0])
R = make_R(a.sigma_step, a.curv, a.curv_up if a.curv_up > 0 else a.curv)
NG = 0
if a.joint:
    # global unknowns appended after the 3N corrections: dh (h = h0 + dh) and beta; height pairs do not see them
    NG = 2
    R = sp.hstack([R, sp.csr_matrix((R.shape[0], NG))]).tocsr()
    for k_, (nm_, *_r) in enumerate(kinds):
        extra = sp.csr_matrix((blocks_A[k_].shape[0], NG))
        if nm_ == "centre line":
            extra = sp.csr_matrix(np.c_[CL_GLOBAL[0], CL_GLOBAL[1]])
        blocks_A[k_] = sp.hstack([blocks_A[k_], extra]).tocsr()
    pri_rows, pri_b = [], []
    if a.h_prior:
        hm, hs = (float(x) for x in a.h_prior.split(","))
        pri_rows.append(np.r_[np.zeros(3 * N), 1.0 / hs, 0.0]); pri_b.append((hm - h) / hs)
    pri_rows.append(np.r_[np.zeros(3 * N), 0.0, 1.0 / a.beta_sigma]); pri_b.append(0.0)
    R = sp.vstack([R, sp.csr_matrix(np.array(pri_rows))]).tocsr()
    bR = np.r_[np.zeros(R.shape[0] - len(pri_b)), pri_b]
else:
    bR = np.zeros(R.shape[0])
RtR = (R.T @ R).tocsc(); Rtb = R.T @ bR
NX = 3 * N + NG

from scipy.sparse.linalg import factorized
wts = [np.ones(B.shape[0]) for B in blocks_A]
scale = [1.0 for _ in blocks_A]                          # variance components: sigma_k -> sigma_k * scale_k
rng_ = np.random.default_rng(0)
for it in range(a.vce if a.joint else 4):
    AtA = RtR.copy(); Atb = Rtb.copy()
    for B, bb, ww, s_ in zip(blocks_A, blocks_b, wts, scale):
        AtA = AtA + (B.T @ sp.diags(ww / s_ ** 2) @ B); Atb += B.T @ (ww * bb / s_ ** 2)
    solve_ = factorized(AtA.tocsc())
    d = solve_(Atb)
    new_w, new_s = [], []
    for B, bb, ww, s_, k_ in zip(blocks_A, blocks_b, wts, scale, kinds):
        r = (B @ d - bb) / (s_ * (np.asarray(k_[6]) if a.fix_huber else 1.0))  # residual in (current) sigmas
        new_w.append(np.where(np.abs(r) <= a.huber, 1.0, a.huber / np.maximum(np.abs(r), 1e-9)))
        if a.joint:
            # Foerstner VCE: sigma^2 <- r'Pr / redundancy, redundancy = n - tr(W A N^-1 A'), trace by Hutchinson probes
            Bs = sp.diags(np.sqrt(ww) / s_) @ B
            tr = 0.0
            for _ in range(8):
                v = rng_.choice([-1.0, 1.0], Bs.shape[0]); u = Bs.T @ v; tr += u @ solve_(u) / 8
            red = max(Bs.shape[0] - tr, 1.0)
            new_s.append(s_ * np.sqrt((ww * r ** 2).sum() / red))
    wts = new_w
    if a.joint:
        scale = new_s
        print(f"   iteration {it}: h {h + d[3 * N]:.4f}, beta {d[3 * N + 1]:+.4f}; noise "
              + ", ".join(f"{k_[0]} {np.median(k_[7]) * s_:.4f}" for k_, s_ in zip(kinds, scale)), flush=True)
if a.joint:
    eh = np.zeros(NX); eh[3 * N] = 1.0; eb = np.zeros(NX); eb[3 * N + 1] = 1.0
    sd_h, sd_b = np.sqrt(solve_(eh)[3 * N]), np.sqrt(solve_(eb)[3 * N + 1])
    h_est, beta = h + d[3 * N], d[3 * N + 1]
    print(f"joint estimate: camera height h {h_est:.4f} +- {sd_h:.4f} units, centre-line offset b = beta/h {beta / h_est:+.4f} h "
          f"(beta {beta:+.4f} +- {sd_b:.4f}); noise: " + ", ".join(f"{k_[0]} {np.median(k_[7]) * s_:.4f} units" for k_, s_ in zip(kinds, scale)))
    info.update(h_joint=float(h_est), h_sd=float(sd_h), beta=float(beta), beta_sd=float(sd_b),
                noise={k_[0]: float(np.median(k_[7]) * s_) for k_, s_ in zip(kinds, scale)})
D3 = d[:3 * N].reshape(N, 3)
for name, m, r0, ii, jj, dirs, w, sig in kinds:
    after = r0 + ((D3[jj] - D3[ii]) * dirs).sum(1)
    if a.joint and name == "centre line":
        after = after + (CL_GLOBAL[0] * d[3 * N] + CL_GLOBAL[1] * d[3 * N + 1]) * sig / w
    print(f"{name}: {m} pairs, residual rms {np.sqrt((r0**2).mean()):.3f} -> {np.sqrt((after**2).mean()):.3f} units "
          f"(|.| median {np.median(np.abs(r0)):.3f} -> {np.median(np.abs(after)):.3f})")
    info[f"{name.replace(' ', '_')}_rms_after"] = float(np.sqrt((after ** 2).mean()))
mv = np.linalg.norm(D3, axis=1)
al = (D3 * fw).sum(1); la = (D3 * lf).sum(1); vu = (D3 * up).sum(1)
print(f"correction rms {np.sqrt((mv**2).mean()):.3f} units (max {mv.max():.3f}); along {np.sqrt((al**2).mean()):.3f}, lateral {np.sqrt((la**2).mean()):.3f}, vertical {np.sqrt((vu**2).mean()):.3f}")
info.update(move_rms=float(np.sqrt((mv ** 2).mean())), move_max=float(mv.max()))

Tn = T.copy(); Tn[:, :3, 3] = P + D3
res_dir = os.path.join(a.target, "result")
out = os.path.join(res_dir, a.out + ".csv")
with open(out + ".tmp", "w") as fh:
    fh.write(lines[0] + "\n")
    for s_, M in zip(stamps, Tn.reshape(N, 16)):
        fh.write(s_ + "," + ",".join(repr(float(x)) for x in M) + "\n")
os.replace(out + ".tmp", out)
json.dump(info, open(os.path.join(res_dir, a.out + ".json"), "w"), indent=1)
print(f"wrote {out}")
