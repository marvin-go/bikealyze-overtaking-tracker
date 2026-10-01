"""
Run the tracker with the configuration used in the paper and evaluate it against ground truth.

Usage:
    python run_example.py [detections.json] [ground_truth.json] [output.csv]
"""
import json
import sys
from pathlib import Path

from overtaking_tracker import create_default_tracker, track_detections, filter_tracks, save_tracks_to_csv
from overtaking_tracker import export_tracks_by_frame, evaluate_mot, render_summary

# post-processing parameters used in the paper
VALID_LABELS = {"car", "bus", "truck"}
MIN_SCORE = 0.6
MIN_LENGTH = 10


def main(det_path="example_detections.json", gt_path="example_ground_truth.json", out_path="example_tracks.csv"):
    with open(det_path) as f:
        detections = json.load(f)

    # score threshold 0.1, constant-velocity Kalman filter, greedy Euclidean matching (3.0 m),
    # tracks terminated after more than 10 consecutive missed frames
    tracker = create_default_tracker()
    tracks = track_detections(tracker=tracker, detections=detections)

    valid_tracks = filter_tracks(tracks, by_label=VALID_LABELS, by_score=MIN_SCORE, by_length=MIN_LENGTH)

    print(f"{len(tracks)} raw tracks, {len(valid_tracks)} after post-processing")
    for tid, dets in sorted(valid_tracks.items()):
        labels = sorted({d["label"] for d in dets})
        print(f"  track {tid}: {len(dets)} detections, labels {labels}")

    save_tracks_to_csv(valid_tracks, Path(out_path))

    # evaluation (requires: pip install -e ".[eval]")
    with open(gt_path) as f:
        ground_truth = json.load(f)
    try:
        summary = evaluate_mot(ground_truth, export_tracks_by_frame(valid_tracks))
    except ImportError:
        print("Install motmetrics to evaluate the tracks: pip install -e \".[eval]\"")
        return
    print()
    print(render_summary(summary))


if __name__ == "__main__":
    main(*sys.argv[1:])
