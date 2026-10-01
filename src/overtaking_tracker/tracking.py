
# standard library
from pathlib import Path
import json
from collections import defaultdict
from itertools import groupby

# third-party
import pandas as pd
import numpy as np

# local modules
from .tracker import MultiObjectTracker

def track_detections(tracker: MultiObjectTracker, detections: list[dict]):

    detections_sorted = sorted(
        detections,
        key=lambda d: (d['timestamp_ns'], d['frame_token'])
    )

    for frame_token, frame_group in groupby(detections_sorted, key=lambda d: d['frame_token']):
        frame_dets = list(frame_group)      # all detections for this frame
        active_map = tracker.forward_step(frame_dets)
        # inspect active_map here
        #print(f"Frame {frame_token}: {len(active_map)} active tracks")

    return tracker.get_all_tracks()

def filter_tracks_by_labels(
    tracks: dict[int, list[dict]],
    valid_labels: set[str],
    score_thresh: float
) -> set[int]:
    """
    Return the set of track_ids where any detection in the track
    has a label in `valid_labels` and score >= score_thresh.

    Args:
      tracks: dictionary from track_id to list of detection dicts
      valid_labels: set of class names to include (e.g. {'car','bus','truck'})
      score_thresh: minimum detection score

    Returns:
      A set of track_ids satisfying the criteria.
    """
    selected = {
        tid : dets
        for tid, dets in tracks.items()
        if any(
            (det.get('label') in valid_labels) and
            (det.get('score', 0.0) >= score_thresh)
            for det in dets
        )
    }
    return selected

def filter_tracks(
    tracks: dict[int, list[dict]],
    by_label: set[str] = None,
    by_score: float = None,
    by_length: int = None,
) -> dict[int, list[dict]]:
    """
    Flexible filter for tracks. Returns dict of track_id -> detections that satisfy:
      - If by_label is set: at least one detection's label is in by_label
      - If by_score is set: at least one detection's score >= by_score
      - If by_length is set: track history is at least by_length long

    Args:
      tracks: dict from track_id to list of detection dicts
      by_label: set of valid labels to filter by (optional)
      by_score: minimum detection score to filter by (optional)
      by_length: minimum track length to filter by (optional)

    Returns:
      Dict of filtered tracks (track_id -> list of detections)
    """
    def satisfies(tid, dets):
        if by_label is not None:
            if not any(det.get('label') in by_label for det in dets):
                return False
        if by_score is not None:
            if not any(det.get('score', 0.0) >= by_score for det in dets):
                return False
        if by_length is not None:
            if len(dets) < by_length:
                return False
        return True

    return {tid: dets for tid, dets in tracks.items() if satisfies(tid, dets)}

def export_tracks_by_frame(tracks: dict[int, list[dict]]) -> dict[str, list[dict]]:
    """
    Given a mapping from track_id to its list of detections,
    create a mapping from frame_token to a list of objects.

    Format for SUSTech:
    [{'obj_id': str,
      'obj_type': str,
      'psr': {'position': {...},
              'rotation': {...},
              'scale': {...} }
     },
    ...]
    
    """
    # Group detections by frame_token
    frames = {}
    for tid, dets in tracks.items():
        for det in dets:
            ft = det['frame_token']
            frames.setdefault(ft, []).append((tid, det))
    # Create output format per frame
    tracks_frame_map = {}
    for ft, items in frames.items():
        objs = []
        for tid, det in items:
            pos = det['translation']
            rot = det['rotation']
            sca = det['scale']
            objs.append({
                'obj_id': str(tid),
                'obj_type': det.get('label', '').capitalize(),
                'psr': {
                    'position': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
                    'rotation': {'x': rot[0], 'y': rot[1], 'z': rot[2]},
                    'scale':    {'x': sca[0], 'y': sca[1], 'z': sca[2]}
                },
                # for debugging, you could include more fields does not interfere with SUS
                #'score': det.get('score', 1.0),
                # for reconstruction directly include global_position and global_velocity
                #'ego2global': det.get('ego2global', None),
                #'velocity': {
                #    'vx': det['global_velocity'][0],
                #    'vy': det['global_velocity'][1],
                #},
                #'timestamp_ns': det['timestamp_ns'],
            })
        
        tracks_frame_map[ft] = objs
    
    return tracks_frame_map

def save_tracks_by_frame(
    tracks_frame_map: dict[str, list[dict]],
    frames: list[dict],
    track_dir: str,
) -> None:
    """
    Save per-frame detection outputs to JSON files.
    """

    # Build lookups for fast access
    # Index frames by token for prev/next lookups
    frames_by_token = {f["token"]: f for f in frames}

    for frame_token, result in tracks_frame_map.items():
        # Lookup the original frame info
        frame_info = frames_by_token.get(frame_token)
        if frame_info is None:
            continue
        # Path to the .pcd file for this frame
        pcd_file = Path(frame_info['pcd_filename'])
        # Assume the directory structure: mcap_dir/track_dir/model_name/
        mcap_dir = pcd_file.parent.parent
        out_dir = mcap_dir / track_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # JSON filename matches the .pcd stem
        output_file = out_dir / f"{pcd_file.stem}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)


def save_tracks_to_csv(tracks: dict[int, list[dict]], output: Path) -> None:
    """
    Save tracks to CSV with fixed columns.
    Each row is a detection, with columns:
    track_id, timestamp_ns, x, y, vx, vy, score
    """
    rows = []
    for track_id, detections in tracks.items():
        for det in detections:
            
            ego2global = det.get('ego2global')

            theta_global = np.arctan2(ego2global[1][0], ego2global[0][0])
            ref_pos_global = [ego2global[0][3], ego2global[1][3]]  # coordiante reference frame [x, y] in global frame

            row = {
                'object_id': track_id,
                'timestamp': det.get('timestamp_ns') * 1e-6,  # convert to milliseconds
                'frame_token': det.get('frame_token'),
                'label': det.get('label'),
                'x': det['global_position'][0],
                'y': det['global_position'][1],
                'vx': det['global_velocity'][0],
                'vy': det['global_velocity'][1],
                'theta_ref': theta_global,
                'x_ref': ref_pos_global[0],
                'y_ref': ref_pos_global[1],
                'position_x': det['translation'][0],
                'position_y': det['translation'][1],
                'position_z': det['translation'][2],
                'rotation_x': det['rotation'][0],
                'rotation_y': det['rotation'][1],
                'rotation_z': det['rotation'][2],
                'scale_x': det['scale'][0],
                'scale_y': det['scale'][1],
                'scale_z': det['scale'][2],
                'score': det.get('score'),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False, sep=';')
    print(f"Tracks saved to {output}")

def read_tracks_from_csv(input_file: Path) -> dict[int, list[dict]]:
    """
    Read tracks from a CSV file and return a mapping from track_id to list of detections.
    Each detection is represented as a dictionary.
    """
    df = pd.read_csv(input_file, sep=',')
    tracks = defaultdict(list)
    
    for _, row in df.iterrows():
        track_id = int(row['object_id'])

        # Calculate the new relative position and rotation from the reconstructed global position
        theta_ref = row['theta_ref']

        if not pd.isna(row['theta_recon']):
            last_theta_recon = row['theta_recon']
        else:
            row['theta_recon'] = last_theta_recon

        # The rotation_z_recon is the difference between the reconstructed theta and ref frame theta
        rotation_z_recon = row['theta_recon'] - theta_ref
        # Calculate the relative position in the reference frame
        position_x_recon =  np.cos(theta_ref) * (row['x_recon'] - row['x_ref']) + np.sin(theta_ref) * (row['y_recon'] - row['y_ref'])
        position_y_recon = -np.sin(theta_ref) * (row['x_recon'] - row['x_ref']) + np.cos(theta_ref) * (row['y_recon'] - row['y_ref'])

        detection = {
            'timestamp_ns': int(row['timestamp'] * 1e6),  # convert back to nanoseconds
            'frame_token': row['frame_token'],
            'label': row['label'],
            'translation': [position_x_recon, position_y_recon, row['position_z']], # position_x, position_y is new
            'rotation': [row['rotation_x'], row['rotation_y'], rotation_z_recon], # rotation_z is new
            'scale': [row['scale_x'], row['scale_y'], row['scale_z']],
            #'global_position': [row['x_recon'], row['y_recon'], 0.0],  # z is not in the CSV
            #'global_velocity': [row['vx'], row['vy']],
            'score': row.get('score', 1.0),
        }
        tracks[track_id].append(detection)
    
    return dict(tracks)
