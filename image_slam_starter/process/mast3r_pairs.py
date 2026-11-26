"""Two-view MASt3R (naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric, public weights) on Track02 image pairs:
relative pose from the two pointmaps (dust3r PairViewer), compared with a trajectory.

Why: for two cameras facing each other along a road the essential matrix puts all matches near the epipole and the
translation direction was off by 11-17 deg (NOTES 11-20); MASt3R regresses both pointmaps in one frame, so the
relative position does not come from the epipolar geometry alone.
Per pair: rotation difference, translation direction difference, |t| ratio (MASt3R metres per trajectory unit).
Images are undistorted with the self-calibrated lens first.
usage (env with torch): python process/mast3r_pairs.py Track2 TRAJ.csv a:b a:b ... [--mast3r ext/mast3r]"""
import argparse, os, sys
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("traj"); ap.add_argument("pairs", nargs="+")
ap.add_argument("--mast3r", default=os.path.expanduser("ext/mast3r"))
ap.add_argument("--model", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
ap.add_argument("--cam", default="1035.5,1033.88,722.17,456.84,-0.05833,0.088413,0,0")
ap.add_argument("--out", default="")
a = ap.parse_args()
sys.path.insert(0, a.mast3r); sys.path.insert(0, os.path.join(a.mast3r, "dust3r"))
import torch
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

c = [float(x) for x in a.cam.split(",")]
K = np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1.0]]); D = np.array(c[4:8] + [0.0])
files = sorted(os.listdir(f"{a.target}/data/images"))
T = np.genfromtxt(a.traj, delimiter=",", skip_header=1)[:, 1:].reshape(-1, 4, 4)
tmp = "/tmp/mast3r_pairs_" + str(os.getpid()); os.makedirs(tmp, exist_ok=True)
dev = "cuda"
model = AsymmetricMASt3R.from_pretrained(a.model).to(dev).eval()


def und(f):
    p = f"{tmp}/{f}.png"
    if not os.path.exists(p):
        im = cv2.imread(f"{a.target}/data/images/{files[f]}")
        cv2.imwrite(p, cv2.undistort(im, K, D))
    return p


def ang(u, v):
    return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


rows = []
for pr in a.pairs:
    i, j = (int(x) for x in pr.split(":"))
    imgs = load_images([und(i), und(j)], size=512, verbose=False)
    out = inference(make_pairs(imgs, scene_graph="complete", symmetrize=True), model, dev, batch_size=1, verbose=False)
    sc = global_aligner(out, device=dev, mode=GlobalAlignerMode.PairViewer, verbose=False)
    P = sc.get_im_poses().detach().cpu().numpy()            # cam-to-world, world = camera i
    rel = np.linalg.inv(P[0]) @ P[1]                          # pose of j in i's camera frame (MASt3R)
    tr = np.linalg.inv(T[i]) @ T[j]                           # the same from the trajectory
    dR = np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3].T @ tr[:3, :3]) - 1) / 2, -1, 1)))
    dt = ang(rel[:3, 3], tr[:3, 3]); ratio = np.linalg.norm(rel[:3, 3]) / np.linalg.norm(tr[:3, 3])
    yaw = np.degrees(np.arccos(np.clip(T[i][:3, 2] @ T[j][:3, 2], -1, 1)))
    rows.append((i, j, yaw, np.linalg.norm(tr[:3, 3]), dR, dt, ratio, *rel[:3, 3], *tr[:3, 3]))
    print(f"{i}:{j}  axes {yaw:5.1f} deg apart, trajectory |t| {np.linalg.norm(tr[:3, 3]):6.2f} u | rotation diff {dR:5.2f} deg, "
          f"translation direction diff {dt:5.2f} deg, |t| ratio {ratio:.3f} m/u", flush=True)
if a.out:
    np.save(a.out, np.array(rows))
