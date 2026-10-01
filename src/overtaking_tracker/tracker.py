# tracker.py
# -----------------
# Modular multi-object tracker with separate lifecycle management

from .preprocess import Preprocessor
from .motion import MotionModel
from .associate import Association
from .lifecycle import LifecycleManager


class MultiObjectTracker:
    """
    Orchestrates all components: Preprocessor, MotionModel, Association, LifecycleManager.
    Manages active vs. finished tracks and provides simple frame-by-frame API.
    """
    def __init__(self,
                 preprocessor: Preprocessor,
                 motion_model: MotionModel,
                 association: Association,
                 lifecycle: LifecycleManager):
        self.pre = preprocessor
        self.mm = motion_model
        self.ass = association
        self.lc = lifecycle
        # Active tracks currently being updated
        self.active_tracks = []
        # Finished tracks for post-analysis
        self.finished_tracks = []
        # Counter to assign unique track IDs
        self._next_id = 1

    def forward_step(self, detections: list) -> dict:
        """
        Process one frame of detections:
          1. Preprocess
          2. Predict active tracks
          3. Associate to detections
          4. Update matches
          5. Handle unmatched (kill)
          6. Spawn new
          7. Return mapping of active track_id->detection
        """
        # Filter raw detections
        dets = self.pre.filter(detections)
        # Convert timestamp (ns->s) if we have any
        timestamp = dets[0]['timestamp_ns']*1e-9 if dets else None

        # If no detections, just age and possibly kill all active tracks
        if not dets:
            to_kill = self.lc.handle_unmatched(self.active_tracks)
            for trk in to_kill:
                self._terminate_track(trk)
            # Return current active mapping (might be empty)
            return {t.track_id: t.history[-1] for t in self.active_tracks}

        # Predict step for each active track
        preds = []
        for trk in list(self.active_tracks):
            dt = timestamp - trk.last_timestamp  # elapsed time
            preds.append((trk, trk.predict(dt, self.mm)))

        # Associate predictions to current detections
        matched, unmatched_tracks, unmatched_dets = self.ass.associate(preds, dets)

        # Update matched tracks
        for trk, det in matched:
            trk.update(det, self.mm)

        # Handle unmatched tracks (increment miss, potentially kill)
        to_kill = self.lc.handle_unmatched(unmatched_tracks)
        for trk in to_kill:
            #print(f"{dets[0]['frame_token']}: killed track {trk.track_id}")
            self._terminate_track(trk)

        # Spawn new tracks for unassigned detections
        new_tracks = self.lc.spawn_new(unmatched_dets, timestamp)
        for nt in new_tracks:
            nt.track_id = self._next_id
            self._next_id += 1
            self.active_tracks.append(nt)

        # Return mapping of active tracks to their latest detection
        return {t.track_id: t.history[-1] for t in self.active_tracks}

    def get_all_tracks(self) -> dict:
        """
        After all frames are processed, retrieve a mapping of every track (active + finished)
        to its final detection.
        """
        all_t = self.active_tracks + self.finished_tracks
        return {t.track_id: t.history for t in all_t}

    def _terminate_track(self, track):
        """
        Move a track from active to finished list when it's terminated.
        """
        if track in self.active_tracks:
            self.active_tracks.remove(track)
            self.finished_tracks.append(track)

    
def create_default_tracker():
    """
    Factory to create a default multi-object tracker with common components.
    """

    from .preprocess import ScorePreprocessor
    from .motion import make_CV_KalmanModel
    from .associate import GreedyDistanceAssociation
    from .lifecycle import SimpleLifecycleManager

    preprocessor = ScorePreprocessor(min_score=0.1)
    motion_model = make_CV_KalmanModel()
    association = GreedyDistanceAssociation(max_distance=3.0, use_mahalanobis=False)
    lifecycle_manager = SimpleLifecycleManager(max_misses=10)

    return MultiObjectTracker(
        preprocessor=preprocessor,
        motion_model=motion_model,
        association=association,
        lifecycle=lifecycle_manager)

def create_maha_tracker():
    """
    Factory to create a mahalanobis multi-object tracker with common components.
    """

    from .preprocess import ScorePreprocessor
    from .motion import make_CV_KalmanModel
    from .associate import GreedyDistanceAssociation
    from .lifecycle import SimpleLifecycleManager

    preprocessor = ScorePreprocessor(min_score=0.1)
    motion_model = make_CV_KalmanModel()
    association = GreedyDistanceAssociation(max_distance=3.0, use_mahalanobis=True)
    lifecycle_manager = SimpleLifecycleManager(max_misses=10)

    return MultiObjectTracker(
        preprocessor=preprocessor,
        motion_model=motion_model,
        association=association,
        lifecycle=lifecycle_manager)