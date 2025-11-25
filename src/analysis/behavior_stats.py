"""
Behavior statistics over tracked lamb trajectories.

We start from a tracks CSV produced by src.tracking.track_from_detections:

    frame_index,image_id,object_id,class_name,score,
    xc,yc,xmin,ymin,xmax,ymax

Given an assumed frame rate (frames per second), we estimate:

- Time spent (seconds) per activity per lamb (object_id)
- Overall time spent per activity across all lambs

This is a straightforward way to move from frame-level detections
to interpretable behavior statistics, in line with what we report
in the paper.
"""

import os
import yaml
import pandas as pd


def _load_config():
    """
    Load the central paths/config file.

    We resolve the path relative to the repository root so that the script
    works both when called from the project root or as a module.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(repo_root, "config", "paths.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file at: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg, repo_root


def main(frame_rate=25.0):
    """
    Entry point for behavior statistics.

    Parameters
    ----------
    frame_rate : float, optional
        Number of frames per second in the analyzed video segment.
        Adjust this to match the real frame rate used when extracting frames.

    Usage:
        python -m src.analysis.behavior_stats

    Optionally pass frame_rate by editing the default or by wrapping this
    function in a small CLI if needed.
    """
    cfg, _ = _load_config()
    tracks_csv = cfg.get("tracks_csv", "./data/detections/tracks.csv")

    if not os.path.exists(tracks_csv):
        print(f"Tracks CSV not found: {tracks_csv}")
        return

    df = pd.read_csv(tracks_csv)

    if df.empty:
        print("Tracks CSV is empty, nothing to analyze.")
        return

    # Ensure correct dtypes.
    df["frame_index"] = df["frame_index"].astype(int)
    df["object_id"] = df["object_id"].astype(int)

    # Each row corresponds to a single object in a single frame.
    # So duration per row is 1 / frame_rate seconds.
    df["duration_sec"] = 1.0 / float(frame_rate)

    # Time spent per activity per lamb (object_id).
    # We pivot so that each row is an object_id and each column is an activity.
    per_lamb = df.pivot_table(
        index="object_id",
        columns="class_name",
        values="duration_sec",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Overall time per activity across all lambs.
    overall = (
        df.groupby("class_name")["duration_sec"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTime spent (seconds) per activity per lamb (object_id):")
    print(per_lamb.round(2))

    print("\nOverall time (seconds) per activity (all lambs combined):")
    print(overall.round(2))

    # If we want, we can also compute simple proportions.
    total_time = overall.sum()
    if total_time > 0:
        proportions = (overall / total_time) * 100.0
        print("\nProportion of time per activity (% of total):")
        print(proportions.round(2))
    else:
        print("\nTotal time is zero; cannot compute proportions.")


if __name__ == "__main__":
    main()
