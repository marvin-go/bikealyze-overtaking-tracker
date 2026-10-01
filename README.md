# Overtaking Tracker

A lightweight, modular multi-object tracker for 3D LiDAR detections. It follows the
tracking-by-detection design of [SimpleTrack](https://github.com/tusen-ai/SimpleTrack) and
the track persistence idea of [Immortal Tracker](https://github.com/ImmortalTracker/ImmortalTracker).

This repository contains the tracker used in the paper:

> **BikeAlyze - Understanding car-to-bicycle overtaking**
> Moritz Beeking, Marvin Götze, Hannah Wies, Markus Steinmaßl, Karl Rehrl, Julian Kooij, Holger Caesar. 
> 2026.
> DOI: TODO

It is published so the tracking step of that paper can be reconstructed.
It is not actively developed.

## Method

The pipeline is split into four components that can be swapped independently:

| Component | Implementation | Setting used in the paper |
|---|---|---|
| Detection preprocessing | `ScorePreprocessor` | keep detections with confidence ≥ 0.1 |
| Motion model | Constant-velocity Kalman filter, state `(x, y, vx, vy)` in global UTM coordinates | initialised from the detector's position and velocity |
| Data association | `GreedyDistanceAssociation` | greedy matching on the planar Euclidean distance, gate 3.0 m |
| Track management | `SimpleLifecycleManager` | unmatched detections start new tracks immediately; tracks end after more than 10 consecutive missed frames (1 s at 10 Hz) |

Post-processing (`filter_tracks`) keeps a track only if it has

- at least 10 detections,
- at least one detection with confidence ≥ 0.6,
- at least one detection labelled `car`, `bus` or `truck`.

Kalman filter details: both position and velocity are observed (`H = I`). The measurement noise
covariance is `R = diag(0.1 m², 0.1 m², 1 m²/s², 1 m²/s²)`, which gives standard deviations of
about 0.32 m and 1.0 m/s. The process noise is continuous white-noise acceleration with spectral
density `q = 1`. The initial state covariance is `10³ · I`.

When using the output:

- The Kalman filter is only used to predict positions for association. A track's output is the
  list of raw detections assigned to it, not the filtered state. Frames in which a track was
  not matched do not appear in its output.
- Association does not take the class into account, so one track can contain detections with
  different labels.
- The miss counter only advances on frames that contain at least one detection. A frame that has
  no detections at all is skipped entirely.

## Installation

Requires Python ≥ 3.9. The tracker depends only on `numpy` and `pandas`. The optional
evaluation also needs [py-motmetrics](https://github.com/cheind/py-motmetrics), and the example
plot needs `matplotlib`.

```bash
git clone <repository-url>
cd bikealyze-overtaking-tracker
pip install -e ".[eval,plot]"   # or: pip install -e .   (tracker only)
```

## Quick start

### Example

A synthetic scene can be generated with the following code.

```bash
cd examples
python make_synthetic_detections.py   # writes example_detections.json and example_ground_truth.json
python run_example.py                 # tracks with the paper configuration, evaluates, writes example_tracks.csv
python plot_example.py                # plots ground truth and tracks, writes example_tracks.png
```

The synthetic scene lasts 6 s at 10 Hz and is placed in UTM zone 33N. A bicycle is overtaken by two cars and a truck. One car is occluded
for 5 frames, and the scene also contains a low-confidence pedestrian false positive and a single
spurious detection. Expected output:

```
5 raw tracks, 3 after post-processing
  track 1: 55 detections, labels ['car']
  track 2: 60 detections, labels ['car']
  track 3: 60 detections, labels ['truck']
Tracks saved to example_tracks.csv

        Frames  IDF1    IDP   IDR  Rcll   Prcn  GT MT PT ML FP FN IDsw  FM  MOTA  MOTP
tracker     60 98.6% 100.0% 97.2% 97.2% 100.0% 180  3  0  0  0  5    0   1 97.2% 0.143
```

The occluded car stays one track. Its 5 missing frames show up as false negatives (FN) and one
fragmentation (FM), because no output is produced while a track coasts.

`plot_example.py` draws the ground truth (grey) and the tracks (coloured dots) in the ego frame of
the bicycle: the longitudinal distance over time, where each overtaking manoeuvre crosses zero, and
a bird's-eye view. The occlusion shows up as a gap in track 1 at about 2.5 s.

### To use the tracker in your own code

```python
from overtaking_tracker import create_default_tracker, track_detections, filter_tracks

tracker = create_default_tracker()
tracks = track_detections(tracker, detections)          # {track_id: [detection, ...]}
tracks = filter_tracks(tracks, by_label={"car", "bus", "truck"}, by_score=0.6, by_length=10)
```

## Input format

`track_detections` takes a flat list of detections for one recording. Each detection is a
`dict`. Detections are grouped into frames by `frame_token` and processed in `timestamp_ns` order.

| Key | Type | Required for | Description |
|---|---|---|---|
| `frame_token` | `str` | tracking | Identifier of the LiDAR frame |
| `timestamp_ns` | `int` | tracking | Frame timestamp in nanoseconds |
| `global_position` | `[x, y, ...]` | tracking | Object centre in a global metric frame (e.g. UTM), in m. Only x and y are used. |
| `global_velocity` | `[vx, vy, ...]` | tracking | Object velocity in the same frame, in m/s. Only vx and vy are used. |
| `score` | `float` | preprocessing, filtering | Detection confidence in [0, 1] |
| `label` | `str` | filtering | Class name, e.g. `car`, `truck` |
| `translation` | `[x, y, z]` | export | Box centre in the ego (sensor) frame, in m |
| `rotation` | `[rx, ry, rz]` | export | Box rotation in the ego frame, in rad |
| `scale` | `[l, w, h]` | export | Box dimensions, in m |
| `ego2global` | 4×4 nested list | CSV export | Transform from the ego frame to the global frame |

Any additional keys are kept unchanged in the output. See
[examples/make_synthetic_detections.py](examples/make_synthetic_detections.py) for a complete example.

## Output

`track_detections` returns `{track_id: [detection, ...]}`, where each list holds the input
detections assigned to that track, in time order. Track IDs start at 1.

Export helpers:

- `save_tracks_to_csv(tracks, path)` writes one row per detection (`;`-separated), with the track
  ID, the timestamp in ms, global position and velocity, the ego reference pose
  (`x_ref`, `y_ref`, `theta_ref`), the box in the ego frame, and the score.
- `export_tracks_by_frame(tracks)` / `save_tracks_by_frame(...)` convert the tracks into per-frame
  object lists in the [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS) label format
  (`obj_id`, `obj_type`, `psr`). `save_tracks_by_frame` expects frame records with `token` and
  `pcd_filename` keys, which come from the original data pipeline.
- `read_tracks_from_csv(path)` reads back a `,`-separated CSV produced by a later trajectory
  reconstruction step (columns `x_recon`, `y_recon`, `theta_recon`). That step is not part of this
  repository.

## Evaluation

`evaluate_mot(gt_frames, track_frames)` computes CLEAR MOT and identity metrics (MOTA, MOTP, IDF1,
ID switches, …) with [py-motmetrics](https://github.com/cheind/py-motmetrics):

- Both inputs map `frame_token` to a list of objects in SUSTechPOINTS format. The track side comes
  from `export_tracks_by_frame(tracks)`. Ground-truth label files in that format can be loaded
  with `load_sustech_labels(label_dir)`.
- Objects are compared as axis-aligned bird's-eye-view boxes in the ego frame, centred at
  `position`, with the object's width (`scale.y`) as the x extent and its length (`scale.x`) as the
  y extent. Yaw is not taken into account.
- A ground-truth object and a track can only be matched if their IoU is at least 0.5 (`min_iou`).
  MOTP is therefore reported as the mean 1 − IoU of matched pairs (lower is better).
- Every frame in `gt_frames` is evaluated. Frames without ground-truth objects must be included
  as empty lists, so that tracks in them are counted as false positives.

```python
from overtaking_tracker import export_tracks_by_frame, evaluate_mot, render_summary, load_sustech_labels

summary = evaluate_mot(load_sustech_labels("path/to/label"), export_tracks_by_frame(tracks))
print(render_summary(summary))
```

The IoU is computed within this package instead of with `motmetrics.distances.iou_matrix`,
because motmetrics 1.4 does not work with NumPy 2. The results are the same.

## Customisation

`MultiObjectTracker` accepts any implementation of the four interfaces `Preprocessor`,
`MotionModel`, `Association` and `LifecycleManager`:

```python
from overtaking_tracker import MultiObjectTracker
from overtaking_tracker.preprocess import ScorePreprocessor
from overtaking_tracker.motion import make_CV_KalmanModel
from overtaking_tracker.associate import GreedyDistanceAssociation
from overtaking_tracker.lifecycle import SimpleLifecycleManager

tracker = MultiObjectTracker(
    preprocessor=ScorePreprocessor(min_score=0.1),
    motion_model=make_CV_KalmanModel(process_noise=1.0, measurement_noise=1.0),
    association=GreedyDistanceAssociation(max_distance=3.0, use_mahalanobis=False),
    lifecycle=SimpleLifecycleManager(max_misses=10),
)
```

`create_maha_tracker()` is a variant that uses the Mahalanobis distance on the predicted position
covariance. It was **not** used in the paper. `NMSPreprocessor` is a placeholder that returns its
input unchanged.

## Repository structure

```
├── src/overtaking_tracker/
│   ├── tracker.py       # MultiObjectTracker and factory functions
│   ├── track.py         # Track state, counters and detection history
│   ├── preprocess.py    # Detection preprocessing
│   ├── motion.py        # Kalman filter motion model
│   ├── associate.py     # Track-to-detection association
│   ├── lifecycle.py     # Track creation and termination
│   ├── tracking.py      # Frame loop, post-processing filters, export helpers
│   └── evaluation.py    # MOT metrics with py-motmetrics
└── examples/
    ├── make_synthetic_detections.py   # Synthetic detections and ground truth
    ├── example_detections.json
    ├── example_ground_truth.json
    ├── run_example.py                 # Tracking and evaluation with the paper configuration
    └── plot_example.py                # Plot of ground truth and tracks
```

## Acknowledgements

The modular design is based on SimpleTrack:

> Z. Pang, Z. Li, N. Wang. *SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking.* ECCV Workshops, 2022.

The track persistence strategy is inspired by Immortal Tracker:

> Q. Wang, Y. Chen, Z. Pang, N. Wang, Z. Zhang. *Immortal Tracker: Tracklet Never Dies.* arXiv:2111.13672, 2021.

No code from either project is included in this repository.

The evaluation uses [py-motmetrics](https://github.com/cheind/py-motmetrics) (MIT license):

> C. Heindl et al. *py-motmetrics: Benchmark multiple object trackers (MOT) in Python.* https://github.com/cheind/py-motmetrics

## Citation

If you use this code, please cite the paper:

```bibtex
@article{TODO,
  title   = {BikeAlyze - Understanding car-to-bicycle overtaking},
  author  = {Moritz Beeking, Marvin Götze, Hannah Wies, Markus Steinmaßl, Karl Rehrl, Julian Kooij, Holger Caesar},
  journal = {TODO},
  year    = {2026},
  doi     = {TODO}
}
```

## License

TODO: add a `LICENSE` file.
