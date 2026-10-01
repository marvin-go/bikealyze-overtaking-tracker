# evaluation.py
# -------------
# CLEAR MOT and identity metrics using py-motmetrics (https://github.com/cheind/py-motmetrics)

from pathlib import Path
import json

import numpy as np

# metrics reported with their short display names
METRICS = {
    'num_frames': 'Frames',
    'idf1': 'IDF1',
    'idp': 'IDP',
    'idr': 'IDR',
    'recall': 'Rcll',
    'precision': 'Prcn',
    'num_objects': 'GT',
    'mostly_tracked': 'MT',
    'partially_tracked': 'PT',
    'mostly_lost': 'ML',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_switches': 'IDsw',
    'num_fragmentations': 'FM',
    'mota': 'MOTA',
    'motp': 'MOTP',
}


def load_sustech_labels(label_dir: Path) -> dict[str, list[dict]]:
    """
    Load per-frame SUSTechPOINTS label files (<frame>.json) from a directory.
    Returns a mapping from file stem to the list of objects in that frame.
    """
    frames = {}
    for label_file in sorted(Path(label_dir).glob("*.json")):
        with open(label_file) as f:
            frames[label_file.stem] = json.load(f)
    return frames


def _bev_boxes(objs: list[dict]) -> tuple[list[int], np.ndarray]:
    """
    Convert SUSTechPOINTS objects to axis-aligned bird's-eye-view boxes in the ego frame,
    in the (left, top, width, height) format expected by motmetrics.
    The box extent is the object width along x and its length along y; yaw is ignored.
    """
    ids, boxes = [], []
    for obj in objs:
        pos, sca = obj['psr']['position'], obj['psr']['scale']
        bb_width, bb_height = sca['y'], sca['x']
        ids.append(int(obj['obj_id']))
        boxes.append([pos['x'] - bb_width / 2, pos['y'] - bb_height / 2, bb_width, bb_height])
    return ids, np.array(boxes, dtype=float).reshape(-1, 4)


def _iou_distance(gt_boxes: np.ndarray, trk_boxes: np.ndarray, min_iou: float) -> np.ndarray:
    """
    Distance matrix 1 - IoU between axis-aligned boxes (left, top, width, height),
    with NaN where IoU < min_iou. Same result as motmetrics.distances.iou_matrix
    with max_iou = 1 - min_iou, which is not compatible with NumPy 2 in motmetrics 1.4.
    """
    if len(gt_boxes) == 0 or len(trk_boxes) == 0:
        return np.empty((len(gt_boxes), len(trk_boxes)))
    a_min, a_max = gt_boxes[:, None, :2], gt_boxes[:, None, :2] + gt_boxes[:, None, 2:]
    b_min, b_max = trk_boxes[None, :, :2], trk_boxes[None, :, :2] + trk_boxes[None, :, 2:]
    inter = np.prod(np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0.0, None), axis=2)
    union = np.prod(gt_boxes[:, 2:], axis=1)[:, None] + np.prod(trk_boxes[:, 2:], axis=1)[None, :] - inter
    dist = 1.0 - inter / union
    dist[dist > 1.0 - min_iou] = np.nan
    return dist


def evaluate_mot(
    gt_frames: dict[str, list[dict]],
    track_frames: dict[str, list[dict]],
    min_iou: float = 0.5,
    name: str = 'tracker',
):
    """
    Compute CLEAR MOT and identity metrics.

    Both inputs map frame_token -> list of objects in SUSTechPOINTS format
    (see export_tracks_by_frame). All frames in gt_frames are evaluated, in their
    order in the dict; include frames without ground-truth objects as empty lists.
    Tracks in frames that are not in gt_frames are ignored.
    A ground-truth object and a track can only be matched if their BEV IoU >= min_iou.

    Returns a pandas DataFrame with one row (named `name`) and one column per metric.
    Requires the optional dependency `motmetrics`.
    """
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for frame_token, gt_objs in gt_frames.items():
        gt_ids, gt_boxes = _bev_boxes(gt_objs)
        trk_ids, trk_boxes = _bev_boxes(track_frames.get(frame_token, []))
        acc.update(gt_ids, trk_ids, _iou_distance(gt_boxes, trk_boxes, min_iou))

    mh = mm.metrics.create()
    return mh.compute(acc, metrics=list(METRICS), name=name)


def render_summary(summary) -> str:
    """Format the output of evaluate_mot as a text table."""
    import motmetrics as mm

    return mm.io.render_summary(summary, namemap=METRICS, formatters=mm.metrics.create().formatters)
