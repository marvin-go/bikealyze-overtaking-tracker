"""
Generate a small synthetic detection file in the input format expected by the tracker,
and the matching ground truth in SUSTechPOINTS label format.

The scene is 6 s at 10 Hz. A sensor-carrying bicycle moves along the x-axis while
two cars and a truck overtake it. It includes noisy detections, a short occlusion,
a low-confidence false positive, and a single spurious detection. Global coordinates
are UTM zone 33N (EPSG:32633) around Residenzplatz, Salzburg. All headings are zero,
so ego and global frames differ by a translation only.

Usage:
    python make_synthetic_detections.py [detections.json] [ground_truth.json]
"""
import json
import sys

import numpy as np

RATE_HZ = 10
N_FRAMES = 60
T0_NS = 1_790_000_000_000_000_000
ORIGIN = np.array([353750.0, 5295750.0])  # UTM 33N, Residenzplatz, Salzburg

rng = np.random.default_rng(42)

# ego (bicycle) moving at 5 m/s along x
EGO_START, EGO_VEL = np.array([0.0, 0.0]), np.array([5.0, 0.0])

# objects: label, start position relative to ORIGIN, velocity, box size (l, w, h), occluded frames
OBJECTS = [
    ("car",   np.array([-20.0, 1.5]), np.array([12.0, 0.0]), [4.5, 1.8, 1.5], set(range(25, 30))),
    ("car",   np.array([-45.0, 1.8]), np.array([14.0, 0.0]), [4.2, 1.8, 1.5], set()),
    ("truck", np.array([-20.0, 5.0]), np.array([9.0, 0.0]),  [9.0, 2.5, 3.2], set()),
]


def make_det(frame_idx, label, pos_rel, vel, size, score, ego_pos):
    pos_global = ORIGIN + pos_rel
    ego_global = ORIGIN + ego_pos
    ego2global = np.eye(4)
    ego2global[:2, 3] = ego_global
    translation = pos_global - ego_global  # heading is zero, so no rotation
    return {
        "frame_token": f"frame_{frame_idx:04d}",
        "timestamp_ns": T0_NS + frame_idx * 1_000_000_000 // RATE_HZ,
        "label": label,
        "score": round(float(score), 3),
        "global_position": [float(pos_global[0]), float(pos_global[1]), 0.0],
        "global_velocity": [float(vel[0]), float(vel[1])],
        "translation": [float(translation[0]), float(translation[1]), size[2] / 2],
        "rotation": [0.0, 0.0, 0.0],
        "scale": list(size),
        "ego2global": ego2global.tolist(),
    }


def make_gt_obj(obj_id, label, pos_rel, size, ego_pos):
    x, y = pos_rel - ego_pos
    return {
        "obj_id": str(obj_id),
        "obj_type": label.capitalize(),
        "psr": {
            "position": {"x": float(x), "y": float(y), "z": size[2] / 2},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale": {"x": size[0], "y": size[1], "z": size[2]},
        },
    }


def main(det_path="example_detections.json", gt_path="example_ground_truth.json"):
    detections, ground_truth = [], {}
    for k in range(N_FRAMES):
        t = k / RATE_HZ
        ego_pos = EGO_START + EGO_VEL * t
        # ground truth contains every object in every frame, also while it is occluded
        ground_truth[f"frame_{k:04d}"] = [
            make_gt_obj(i + 1, label, p0 + v * t, size, ego_pos)
            for i, (label, p0, v, size, _) in enumerate(OBJECTS)
        ]
        for label, p0, v, size, occluded in OBJECTS:
            if k in occluded:
                continue
            pos = p0 + v * t + rng.normal(0.0, 0.15, 2)
            vel = v + rng.normal(0.0, 0.5, 2)
            detections.append(make_det(k, label, pos, vel, size, rng.uniform(0.4, 0.95), ego_pos))
        # low-confidence false positive of a standing pedestrian on the sidewalk (discarded by post-processing)
        if 10 <= k < 30:
            pos = np.array([12.0, -4.0]) + rng.normal(0.0, 0.2, 2)
            detections.append(make_det(k, "pedestrian", pos, np.zeros(2), [0.6, 0.6, 1.7], 0.2, ego_pos))
    # a single spurious car detection at the roadside (discarded by the minimum-length filter)
    detections.append(make_det(40, "car", np.array([45.0, -6.0]), np.zeros(2), [4.5, 1.8, 1.5], 0.7, EGO_START + EGO_VEL * 4.0))

    with open(det_path, "w") as f:
        json.dump(detections, f, indent=1)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=1)
    print(f"Wrote {len(detections)} detections in {N_FRAMES} frames to {det_path}")
    print(f"Wrote ground truth for {len(OBJECTS)} objects to {gt_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
