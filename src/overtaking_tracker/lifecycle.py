# lifecycle.py
# ---------------------
# Interfaces and default implementation for track lifecycle management

from abc import ABC, abstractmethod

from .track import Track

class LifecycleManager(ABC):
    @abstractmethod
    def handle_unmatched(self, unmatched_tracks: list) -> list:
        """Increment miss counters and return tracks to terminate."""
        pass

    @abstractmethod
    def spawn_new(self, unmatched_dets: list, timestamp: float) -> list:
        """Create new Track objects for unmatched detections."""
        pass

class SimpleLifecycleManager(LifecycleManager):
    """
    Basic lifecycle:
      - Kill tracks with misses > max_misses
      - Immediately birth new tracks for all unmatched detections
    """
    def __init__(self, max_misses: int = 2):
        self.max_misses = max_misses

    def handle_unmatched(self, unmatched_tracks: list) -> list:
        to_kill = []
        for trk in unmatched_tracks:
            trk.misses += 1             # Increment miss count
            if trk.misses > self.max_misses:
                to_kill.append(trk)      # Mark for termination
        return to_kill

    def spawn_new(self, unmatched_dets: list, timestamp: float) -> list:
        new_tracks = []
        for det in unmatched_dets:
            # Initialize track_id=-1; real ID assigned in tracker loop
            nt = Track(track_id=-1, initial_det=det, timestamp=timestamp)
            new_tracks.append(nt)
        return new_tracks
