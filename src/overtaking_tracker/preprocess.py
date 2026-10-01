# preprocess.py
# -------------
# Interfaces and implementations for detection preprocessing

from abc import ABC, abstractmethod

class Preprocessor(ABC):
    @abstractmethod
    def filter(self, dets: list) -> list:
        """
        Select a subset of raw detections for tracking (e.g. by score or ROI).
        """
        pass


class ScorePreprocessor(Preprocessor):
    """
    Filter detections based on a minimum confidence score.
    """
    def __init__(self, min_score: float = 0.5):
        self.min_score = min_score

    def filter(self, dets: list) -> list:
        return [d for d in dets if d.get('score', 1) >= self.min_score]


class NMSPreprocessor(Preprocessor):
    """
    Filter detections using Non-Maximum Suppression (NMS) to remove overlapping boxes.
    Will be implemented to keep the highest-scoring detections first.
    """
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def filter(self, dets: list) -> list:
        # TODO: implement NMS logic here
        # Steps:
        # 1. Sort detections by score descending
        # 2. Iteratively pick highest score det and remove all dets with IoU > threshold
        return dets  # placeholder
