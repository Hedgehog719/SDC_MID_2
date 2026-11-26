"""Track02 public-road checks of trajectories against a reference (run from image_slam_starter):
separation of frames 1185 / 2858 (the two passes of the public road), lateral / vertical offset of the eastbound pass
from the westbound polyline, and RMSE after Sim(3) on the public frames / on all frames.
usage: python process/t2_gate.py REF traj1 traj2 ...   (names without .csv, in Track2/result)"""
import sys, numpy as np
def L(t): return np.genfromtxt(f'Track2/result/{t}.csv',delimiter=',',skip_header=1)[:,1:].reshape(-1,4,4)
def ume(s,d):
    ms,md=s.mean(0),d.mean(0); X,Y=s-ms,d-md; U,D,Vt=np.linalg.svd(Y.T@X/len(s)); S=np.diag([1,1,np.sign(np.linalg.det(U)*np.linalg.det(Vt))]); R=U@S@Vt; c=np.trace(np.diag(D)@S)/X.var(0).sum(); return c,R,md-c*R@ms
pub=np.r_[1141:1345,2731:2948]
ref=L(sys.argv[1])[:,:3,3]
for t in sys.argv[2:]:
    T=L(t); C=T[:,:3,3]; W=C[1141:1345]
    lat=[];up=[]
    for p in C[2731:2948]:
        j=np.argmin(np.linalg.norm(W-p,axis=1)); j=min(max(j,1),len(W)-2); f=W[j+1]-W[j-1]; f/=np.linalg.norm(f); u=-T[1141+j,:3,1]; l=np.cross(u,f); l/=np.linalg.norm(l); d=p-W[j]; lat.append(d@l); up.append(d@u)
    c,R,tt=ume(C[pub],ref[pub]); pr=np.sqrt(((c*C[pub]@R.T+tt-ref[pub])**2).sum(1).mean())
    c,R,tt=ume(C,ref); al=np.sqrt(((c*C@R.T+tt-ref)**2).sum(1).mean())
    print(f'{t:16s} sep {np.linalg.norm(C[1185]-C[2858]):6.2f} lat {np.median(lat):+.3f} vert {np.median(up):+.3f} | vs {sys.argv[1]}: public {pr:.3f} all {al:.3f}')
