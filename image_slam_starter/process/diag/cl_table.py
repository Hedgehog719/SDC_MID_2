"""Centre-line lateral check per route segment for several trajectories: ratio = trajectory separation of two opposite passes
/ (y0_a + y0_b) (camera heights).  A right trajectory gives the same ratio (= camera height in its units) everywhere."""
import sys
import numpy as np
from scipy.spatial import cKDTree

z = np.load("Track2/result/centerline2_0-4206_-0.21_-1.49.npz"); y0 = z["y0"]; ok = ~np.isnan(y0)
S = "../submissions/track2/"
TR = {"SLAM cz16_bu1035cf": "Track2/result/t2_cz16_bu1035cf.csv",
      "BA bu1035cf (0.203)": S + "t2_sfm_bu1035cf.csv", "BA bu1035_sf (0.210)": S + "t2_sfm_bu1035_sf.csv",
      "BA pp06zr (0.218)": S + "t2_sfm_pp06zr.csv", "BA pp06z rotfix (0.330)": S + "t2_sfm_pp06z.csv",
      "SLAM dash1r (0.421)": S + "t2_dash1r.csv", "SLAM cam_z16_it3 (0.506)": S + "t2_cam_z16_it3.csv"}
for k, v in [x.split("=") for x in sys.argv[1:]]:
    TR[k] = v
SEG = [(0, 200), (200, 800), (800, 1004), (1004, 1076), (1136, 1206), (1206, 1400), (1400, 1700), (1700, 2050), (2050, 2400)]
print("segment of the earlier pass:        " + " ".join(f"{a}-{b}".rjust(10) for a, b in SEG))
for name, fn in TR.items():
    T = np.genfromtxt(fn, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4); P = T[:, :3, 3]; N = len(P)
    tree = cKDTree(P)
    up = np.zeros_like(P)
    for i in range(0, N, 5):
        nb = tree.query_ball_point(P[i], 40.0); Q = P[nb] - P[nb].mean(0); n = np.linalg.svd(Q, full_matrices=False)[2][2]
        up[i:i + 5] = n if n @ (-T[i, :3, 1]) > 0 else -n
    fw = np.gradient(P, axis=0); fw = np.array([fw[max(0, i - 3):i + 4].mean(0) for i in range(N)])
    fw -= (fw * up).sum(1)[:, None] * up; fw /= np.linalg.norm(fw, axis=1, keepdims=True) + 1e-12; lf = np.cross(up, fw)
    I, J = [], []
    for i in range(0, N, 2):
        for j in tree.query_ball_point(P[i], 3.5):
            if j - i > 150 and ok[i] and ok[j]:
                I.append(i); J.append(j)
    I, J = np.array(I), np.array(J)
    cs = (fw[I] * fw[J]).sum(1); m0 = cs < -np.cos(np.radians(15))
    I, J = I[m0], J[m0]
    lat = ((P[J] - P[I]) * lf[I]).sum(1); D = y0[I] + y0[J]
    cells = []
    for a, b in SEG:
        m = (I >= a) & (I < b)
        cells.append(f"{np.median(lat[m] / D[m]):.3f}({m.sum():4d})".rjust(10) if m.sum() >= 15 else "-".rjust(10))
    print(f"{name:34s} " + " ".join(cells))
