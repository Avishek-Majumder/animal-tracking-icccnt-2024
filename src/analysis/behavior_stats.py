# src/analysis/behavior_stats.py

import yaml
import pandas as pd

CONFIG_PATH = "./config/paths.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def main(frame_rate=25.0):
    """
    frame_rate: frames per second of the analyzed video segment (e.g., ~25 fps)
    """
    cfg = load_config()
    tracks_csv = cfg["tracks_csv"]

    df = pd.read_csv(tracks_csv)

    # ensure ordering
    df.sort_values(["object_id", "frame_index"], inplace=True)

    # basic stats: total time per behavior per object
    df["duration_sec"] = 1.0 / frame_rate

    pivot = df.pivot_table(
        index="object_id",
        columns="class_name",
        values="duration_sec",
        aggfunc="sum",
        fill_value=0.0
    )

    print("Time spent (seconds) per behavior per lamb (object_id):")
    print(pivot)

    overall = df.groupby("class_name")["duration_sec"].sum()
    print("\nOverall time (seconds) per behavior (all lambs combined):")
    print(overall)

if __name__ == "__main__":
    main()
