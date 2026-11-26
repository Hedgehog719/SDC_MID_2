"""BEV unit (camera height in trajectory units) from same-direction marking registration at frame gaps 4/8/16/24 on the
public road: a range-dependent BEV scale (pitch error) would make it depend on the gap.  usage: python process/bev_unit_gap.py"""
import sys, numpy as np
sys.path.insert(0,'.')
from utils import bevloop
from multiprocessing import Pool
T=np.genfromtxt('Track2/result/t2_cam_z16_it3.csv',delimiter=',',skip_header=1)[:,1:].reshape(-1,4,4)
K=np.array([[1035.5,0,722.17],[0,1033.88,456.84],[0,0,1.]]); D=np.array([-0.05833,0.088413,0,0,0.])
mb=None
def init():
    global mb
    mb=bevloop.MarkingBEV('Track2/data/images',seg='Track2/seg.npz',mount=(-0.22,-1.48),roll=np.load('Track2/result/road_roll_bc38a89a.npy'),K=K,D=D)
def one(args):
    i,step=args
    Tij=np.linalg.inv(T[i])@T[i+step]; xb,_,_=bevloop.road_pose(Tij,mb.F,mb.L)
    if xb<=0.05*step: return None
    r=bevloop.register(mb,i,i+step,Tij,1.0,yaw_range=0.5,shift=2.5)
    if r is None or r[3]<0.5 or r[4]<0.05: return None
    return (step, i, xb/(xb+r[0]))
jobs=[(i,s) for s in (4,8,16,24) for i in list(range(1100,1400,3))+list(range(2700,2990,3))]
with Pool(24,initializer=init) as p: out=[o for o in p.map(one,jobs) if o]
o=np.array(out)
for s in (4,8,16,24):
    for nm,(a,b) in (('W',(1100,1400)),('E',(2700,2990))):
        m=(o[:,0]==s)&(o[:,1]>=a)&(o[:,1]<b)
        print(s,nm,m.sum(),'h(units) median %.4f'%np.median(o[m,2]) if m.sum() else '')
