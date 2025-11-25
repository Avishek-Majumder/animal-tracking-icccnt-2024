# src/tracking/track_from_detections.py

import os
import yaml
import pandas as pd
from tqdm import tqdm
from .centroid_tracker import CentroidTracker

CONFIG_PATH = "./config/paths.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    detections_csv = cfg["detections_csv"]
    tracks_csv = cfg["tracks_csv"]
    os.makedirs(os.path.dirname(tracks_csv), exist_ok=True)

    df = pd.read_csv(detections_csv)

    # filter low-confidence detections if needed
    df = df[df["score"] >= 0.3].copy()

    frames = sorted(df["frame_index"].unique())

    tracker = CentroidTracker(max_distance=50, max_disappeared=10)

    track_records = []

    for frame_idx in tqdm(frames, desc="Tracking"):
        frame_df = df[df["frame_index"] == frame_idx]

        centroids = []
        boxes = []

        for _, row in frame_df.iterrows():
            x_min, y_min, x_max, y_max = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            xc = (x_min + x_max) / 2.0
            yc = (y_min + y_max) / 2.0
            centroids.append((xc, yc))
            boxes.append((x_min, y_min, x_max, y_max, row["class_name"], row["score"]))

        objects = tracker.update(centroids)

        # objects: dict object_id -> (xc, yc) in same order as centroids
        # need to map centroids to boxes again
        # simple nearest centroid per object for logging
        for object_id, (xc, yc) in objects.items():
            # find nearest detection in this frame
            if len(centroids) == 0:
                continue

            dists = [( (xc - c[0])**2 + (yc - c[1])**2, idx ) for idx, c in enumerate(centroids)]
            _, nearest_idx = min(dists, key=lambda x: x[0])

            x_min, y_min, x_max, y_max, cls_name, score = boxes[nearest_idx]

            track_records.append({
                "frame_index": frame_idx,
                "object_id": object_id,
                "class_name": cls_name,
                "score": score,
                "xc": xc,
                "yc": yc,
                "xmin": x_min,
                "ymin": y_min,
                "xmax": x_max,
                "ymax": y_max,
            })

    tracks_df = pd.DataFrame(track_records)
    tracks_df.to_csv(tracks_csv, index=False)
    print(f"Tracks saved to {tracks_csv}")

if __name__ == "__main__":
    main()
