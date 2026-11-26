import gtsam
import numpy as np

class PoseGraph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.isam = gtsam.ISAM2()

    
    def add_node(self, idx, T):
        if not self.initial.exists(idx):
            self.initial.insert(idx, gtsam.Pose3(T))
        
    def add_prior(self):
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-3)
        self.initial.insert(0, gtsam.Pose3())
        self.graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), noise))
        
        
    def add_odom(self, i, j, T, rot_sigma=0.1, trans_sigma=0.1):
        # gtsam Pose3 tangent order: rotation (3) then translation (3)
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.ones(3) * rot_sigma, np.ones(3) * trans_sigma])
        # print(f"odom edge added {i}-{j}")
        self.graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(T), noise))
        
    def add_loop(self, i, j, T, rot_sigma=0.08, trans_sigma=0.08, robust=True):
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.ones(3) * rot_sigma, np.ones(3) * trans_sigma])
        if robust:   # a wrong loop must not bend the whole trajectory
            noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), noise)
        # print(f"Loop edge added {i}-{j}")
        self.graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(T), noise))

    def optimize(self):
        self.isam.update(self.graph, self.initial)

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        result = self.isam.calculateEstimate()

        opt_poses = {}
        for k in result.keys():
            opt_poses[k] = result.atPose3(k).matrix()

        return opt_poses