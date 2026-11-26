import numpy as np

'''
Loop closure: candidates by position / heading / keyframe gap, verified with GICP.
'''
class LoopClosure:
    def __init__(
        self,
        icp,
        dist_thresh=15.0,       # [m] search radius around the current keyframe (must cover the drift)
        fitness_thresh=0.7,     # min inlier ratio of the verified match (GICP mode)
        max_candidates=8,
        min_kf_gap=40,          # keyframes (~metres) between the two ends of a loop
        max_heading_deg=45.0,   # forward camera: only same-direction revisits overlap
        min_interval=1,         # keyframes between two accepted loops
        max_correction=6.0,     # [m] reject matches that move the keyframe further than this (GICP mode)
        matcher=None,           # VisualMatcher: verify loops with image features + PnP instead of GICP
        min_inliers=60,         # PnP inliers needed to accept a visual loop
        cand_stride=3,          # keyframes between two candidates
    ):
        """
        icp: ICP instance (small_gicp wrapper)
        """

        self.icp = icp

        # ===== parameter =====
        self.dist_thresh = dist_thresh

        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates
        self.min_kf_gap = min_kf_gap
        self.max_heading = np.deg2rad(max_heading_deg)
        self.min_interval = min_interval
        self.max_correction = max_correction
        self.matcher = matcher
        self.min_inliers = min_inliers
        self.cand_stride = cand_stride
        self.frames = []        # image features of each keyframe (visual mode)

        # ===== state =====
        self.keyframes = []
        self.poses = []

        self.last_loop_accum = 0.0
        self.last_loop_idx = -10**9
        self.loops = []   # (loop_idx, curr_idx, fitness, correction[m]) log

    # =========================
    #       Add Keyframe
    # =========================
    def add_keyframe(self, pts, pose, frame=None):
        # local frame points
        self.keyframes.append(pts)
        self.frames.append(frame)

        # world frame poses
        self.poses.append(pose)


    # ==========================
    #    Candidate selection
    # ==========================
    def find_candidates(self, curr_idx):
        candidates = []
        if curr_idx - self.last_loop_idx < self.min_interval:
            return candidates

        last = curr_idx - self.min_kf_gap
        if last < 0:
            return candidates
        cur = self.poses[curr_idx]
        P = np.array([p[:3, 3] for p in self.poses[:last + 1]])
        d = np.linalg.norm(P - cur[:3, 3], axis=1)
        # viewing direction (camera z axis) must agree
        Z = np.array([p[:3, 2] for p in self.poses[:last + 1]])
        ang = np.arccos(np.clip(Z @ cur[:3, 2], -1, 1))
        ok = np.where((d < self.dist_thresh) & (ang < self.max_heading))[0]
        if len(ok) == 0:
            return candidates

        # nearest first, but skip direct neighbours of an already chosen one (their submaps overlap)
        for i in ok[np.argsort(d[ok])]:
            if all(abs(i - c) >= self.cand_stride for c in candidates):
                candidates.append(int(i))
            if len(candidates) >= self.max_candidates:
                break
        return candidates

    # =========================
    #       ICP matching
    # =========================
    def match(self, curr_idx, candidates, submap_mgr):
        '''
        curr_idx: current
        candidates: candidates that you choose to do loop closure detection with current frame
        submap_mgr: use as submap_mgr.submaps[idx] to get idx-th submap points(those are in world frame)
        '''

        if self.matcher is not None:
            return self.match_visual(curr_idx, candidates)

        best_score = np.inf
        best_idx = None
        best_T = None
        best_fit = 0.0
        best_corr = 0.0

        pts = self.keyframes[curr_idx]
        init = self.poses[curr_idx]

        for idx in candidates:
            tgt = submap_mgr.submaps[idx]
            # coarse (large correspondence distance, absorbs the drift) then fine
            T, _ = self.icp.align(pts, tgt, init, max_corr=3.0)
            T, score = self.icp.align(pts, tgt, T, max_corr=1.0)
            fit = self.icp.fitness
            corr = np.linalg.norm(T[:3, 3] - init[:3, 3])
            if fit < self.fitness_thresh or corr > self.max_correction:
                continue
            if score < best_score:
                best_score, best_idx, best_fit, best_corr = score, idx, fit, corr
                # relative pose loop keyframe -> current keyframe
                best_T = np.linalg.inv(self.poses[idx]) @ T

        if best_idx is None:
            return None
        self.last_loop_idx = curr_idx
        self.loops.append((best_idx, curr_idx, best_fit, best_corr))
        return best_idx, best_T, best_score

    def match_visual(self, curr_idx, candidates):
        """Verify with image features: the candidate with most PnP inliers wins.
        Then its neighbours are tried too (hill climbing), because the nearest keyframe by the
        drifted pose is usually not the keyframe that really sees the same place."""
        cur = self.frames[curr_idx]
        tried = {}

        def run(idx):
            if idx not in tried:
                tried[idx] = self.matcher.relative_pose_sym(self.frames[idx], cur)
            return tried[idx][1] if tried[idx][0] is not None else 0

        best_idx, best_n = None, 0
        for idx in candidates:
            n = run(idx)
            if n > best_n:
                best_idx, best_n = idx, n
        while best_idx is not None:
            moved = False
            for nb in (best_idx - 1, best_idx + 1):
                if 0 <= nb <= curr_idx - self.min_kf_gap and run(nb) > best_n:
                    best_idx, best_n, moved = nb, run(nb), True
            if not moved:
                break
        if best_idx is None or best_n < self.min_inliers:
            return None
        T = tried[best_idx][0]   # T_loop_curr
        self.last_loop_idx = curr_idx
        self.loops.append((best_idx, curr_idx, best_n, float(np.linalg.norm(T[:3, 3]))))
        return best_idx, T, best_n

    # =========================
    #        Pose Update
    # =========================
    def update_poses(self, new_kf_poses):
        """ Sync poses and recalculate accumulated distances after PGO """
        # Update poses list
        for i in range(len(new_kf_poses)):
            if i < len(self.poses):
                self.poses[i] = new_kf_poses[i].copy()

    # =========================
    #            API
    # =========================
    def detect(self, submap_mgr):
        curr_idx = len(self.keyframes) - 1
        # Could block some too-early detections
        if curr_idx < 30:
            return None

        candidates = self.find_candidates(curr_idx)
        if not candidates:
            return None
        result = self.match(curr_idx, candidates, submap_mgr)
        return result
