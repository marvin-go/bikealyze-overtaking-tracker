# associate.py
# ----------------
# Interfaces and implementations for association methods

from abc import ABC, abstractmethod
import numpy as np

class Association(ABC):
    @abstractmethod
    def associate(self, preds: list, dets: list) -> tuple:
        """
        Match track predictions to new detections. Returns:
          - matched: list of (track, detection) pairs
          - unmatched_tracks: tracks with no match
          - unmatched_dets: detections not assigned to any track
        """
        pass

class GreedyDistanceAssociation(Association):
    """
    Simple greedy matcher using either L2 or Mahalanobis distance.
    Gating applied via max_distance.
    """
    def __init__(self, max_distance: float = 50.0, use_mahalanobis: bool = False):
        self.max_distance = max_distance           # Threshold to reject poor matches
        self.use_mahalanobis = use_mahalanobis  # Flag to select distance metric

    def associate(self, preds: list, dets: list):
        # Extract track objects
        tracks = [trk for trk, _ in preds]
        # Extract predicted XY and corresponding 2×2 covariance submatrix
        positions = np.array([mean[:2] for _, mean in preds])  # (N, 2)
        covs_xy = [trk.covariance[:2, :2] for trk, _ in preds]
        # Extract detection XY positions
        detections = np.array([d['global_position'][:2] for d in dets])  # (M,2)
        N, M = len(tracks), len(dets)
        if N == 0 or M == 0:
            return [], tracks, dets.copy()
        # Compute cost matrix
        cost = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                diff = positions[i] - detections[j]
                if self.use_mahalanobis:
                    # Mahalanobis distance using 2×2 covariance
                    inv_cov = np.linalg.inv(covs_xy[i])
                    cost[i, j] = float(np.sqrt(diff.T @ inv_cov @ diff))
                else:
                    cost[i, j] = np.linalg.norm(diff)
        # Greedy assignment with gate
        matched = []
        unmatched_t = list(range(N))
        unmatched_d = list(range(M))
        while True:
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            if cost[i, j] > self.max_distance:
                break
            matched.append((tracks[i], dets[j]))
            unmatched_t.remove(i)
            unmatched_d.remove(j)
            cost[i, :] = np.inf
            cost[:, j] = np.inf
        return (
            matched,
            [tracks[i] for i in unmatched_t],
            [dets[j]   for j in unmatched_d]
        )
