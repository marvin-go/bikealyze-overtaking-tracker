"""
Plot the ground truth and the tracks written by run_example.py, both in the ego frame of the bicycle.

Left: longitudinal distance to the bicycle over time. An overtaking manoeuvre is a crossing of zero.
Right: bird's-eye view, with the bicycle at the origin.

Usage (requires: pip install -e ".[plot]"):
    python plot_example.py [ground_truth.json] [tracks.csv] [output.png]
"""
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

RATE_HZ = 10
TRACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical palette, fixed order
GT_COLOR = "#c3c2b7"


def main(gt_path="example_ground_truth.json", tracks_path="example_tracks.csv", out_path="example_tracks.png"):
    with open(gt_path) as f:
        ground_truth = json.load(f)
    frame_index = {token: k for k, token in enumerate(sorted(ground_truth))}

    gt = pd.DataFrame([
        {"obj_id": obj["obj_id"], "obj_type": obj["obj_type"], "t": frame_index[token] / RATE_HZ,
         "x": obj["psr"]["position"]["x"], "y": obj["psr"]["position"]["y"]}
        for token, objs in ground_truth.items() for obj in objs
    ])
    tracks = pd.read_csv(tracks_path, sep=";")
    tracks["t"] = tracks["frame_token"].map(frame_index) / RATE_HZ

    fig, (ax_t, ax_bev) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for obj_id, obj in gt.groupby("obj_id"):
        label = "ground truth" if obj_id == gt["obj_id"].iloc[0] else None
        ax_t.plot(obj["t"], obj["x"], color=GT_COLOR, lw=6, solid_capstyle="round", label=label, zorder=1)
        ax_bev.plot(obj["x"], obj["y"], color=GT_COLOR, lw=6, solid_capstyle="round", zorder=1)
        last = obj.iloc[-1]
        ax_t.annotate(f"GT {obj_id} ({last['obj_type'].lower()})", (last["t"], last["x"]),
                      xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color="#52514e")

    for i, (track_id, trk) in enumerate(tracks.groupby("object_id")):
        style = dict(color=TRACK_COLORS[i % len(TRACK_COLORS)], s=10, zorder=2)
        ax_t.scatter(trk["t"], trk["position_x"], label=f"track {track_id} ({trk['label'].iloc[0]})", **style)
        ax_bev.scatter(trk["position_x"], trk["position_y"], **style)

    ax_bev.scatter([0], [0], marker=">", s=80, color="#0b0b0b", zorder=3)
    ax_bev.annotate("bicycle (sensor)", (0, 0), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)

    ax_t.axhline(0, color="#0b0b0b", lw=0.8, ls="--")
    ax_t.set(xlabel="time [s]", ylabel="longitudinal distance to bicycle [m]",
             title="Longitudinal distance over time")
    ax_t.set_xlim(right=gt["t"].max() * 1.25)  # room for the direct labels
    ax_t.legend(loc="upper left", fontsize=8, frameon=False)

    ax_bev.set(xlabel="longitudinal x [m]", ylabel="lateral y [m]", title="Bird's-eye view (ego frame)")
    ax_bev.set_ylim(-4, 8)  # axes are not to scale

    for ax in (ax_t, ax_bev):
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:])
