"""
Tracking script: from per-frame detections to consistent object trajectories.

We start from a detection CSV produced by one of our models (e.g. YOLOv5),
with one row per bounding box:

    frame_index,image_id,class_name,score,xmin,ymin,xmax,ymax

Where:
    - frame_index : integer index of the frame in the video segment
    - image_id    : identifier for the frame (usually filename without extension)
    - class_name  : predicted activity ("standing", "eating", "laying", ...)
    - score       : confidence score of the detection
    - xmin, ymin, xmax, ymax : bounding box coordinates in pixels

We then apply a centroid-based tracker to assign a stable object_id
to the same lamb across frames.

The output is a tracks CSV with columns:

    frame_index, image_id, object_id, class_name, score,
    xc, yc, xmin, ymin, xmax, ymax

and is saved to the path configured as tracks_csv in config/paths.yaml.
"""

import os
import yaml
import pandas as pd
from tqdm import tqdm

from .centroid_tracker import CentroidTracker


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


def _validate_detection_columns(df):
    """
    Ensure the detections CSV has all required columns.
    """
    required_cols = {
        "frame_index",
        "image_id",
        "class_name",
        "score",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Detections CSV is missing columns: {missing}")


def main():
    """
    Entry point for tracking.

    Usage:
        python -m src.tracking.track_from_detections
    """
    cfg, _ = _load_config()

    detections_csv = cfg.get("detections_csv", "./data/detections/sample_detections.csv")
    tracks_csv = cfg.get("tracks_csv", "./data/detections/tracks.csv")

    if not os.path.exists(detections_csv):
        print(f"Detections CSV not found: {detections_csv}")
        return

    # Load detections into a DataFrame.
    detections = pd.read_csv(detections_csv)
    _validate_detection_columns(detections)

    if detections.empty:
        print("Detections CSV is empty, nothing to track.")
        return

    # Optionally, we could filter on score here, e.g.:
    # detections = detections[detections["score"] >= 0.3].copy()
    # For now we keep all and let the user decide the threshold upstream.

    # Ensure consistent dtypes.
    detections["frame_index"] = detections["frame_index"].astype(int)

    # Sort by frame_index so we process frames in temporal order.
    detections = detections.sort_values(["frame_index", "image_id"]).reset_index(drop=True)

    unique_frames = sorted(detections["frame_index"].unique())

    tracker = CentroidTracker(max_distance=50.0, max_disappeared=10)

    track_records = []

    print(f"Starting tracking over {len(unique_frames)} frames...")

    for frame_idx in tqdm(unique_frames, desc="Tracking"):
        frame_detections = detections[detections["frame_index"] == frame_idx]

        # Prepare centroids and keep index mapping to the original detections.
        centroids = []
        det_rows = []

        for _, row in frame_detections.iterrows():
            x_min = float(row["xmin"])
            y_min = float(row["ymin"])
            x_max = float(row["xmax"])
            y_max = float(row["ymax"])

            xc = (x_min + x_max) / 2.0
            yc = (y_min + y_max) / 2.0

            centroids.append((xc, yc))
            det_rows.append(row)

        # Update tracker with current frame centroids.
        objects = tracker.update(centroids)

        # objects: dict[object_id] -> numpy array([xc, yc])
        # For each tracked object, we want to attach the nearest detection
        # in the current frame (to get its box, class, score, etc.).
        for object_id, centroid in objects.items():
            if not centroids:
                # No detections in this frame, nothing to log.
                continue

            xc, yc = float(centroid[0]), float(centroid[1])

            # Find the nearest detection in this frame.
            best_sq_dist = None
            best_idx = None

            for idx, (cx, cy) in enumerate(centroids):
                dx = xc - cx
                dy = yc - cy
                sq_dist = dx * dx + dy * dy

                if best_sq_dist is None or sq_dist < best_sq_dist:
                    best_sq_dist = sq_dist
                    best_idx = idx

            if best_idx is None:
                # Should not really happen, but we guard just in case.
                continue

            det_row = det_rows[best_idx]

            track_records.append(
                {
                    "frame_index": int(frame_idx),
                    "image_id": det_row["image_id"],
                    "object_id": int(object_id),
                    "class_name": det_row["class_name"],
                    "score": float(det_row["score"]),
                    "xc": xc,
                    "yc": yc,
                    "xmin": float(det_row["xmin"]),
                    "ymin": float(det_row["ymin"]),
                    "xmax": float(det_row["xmax"]),
                    "ymax": float(det_row["ymax"]),
                }
            )

    if not track_records:
        print("No track records were created. Check your detections CSV.")
        return

    tracks_df = pd.DataFrame(track_records)
    os.makedirs(os.path.dirname(tracks_csv), exist_ok=True)

    tracks_df.to_csv(tracks_csv, index=False)
    print(f"Tracking complete. Tracks saved to: {tracks_csv}")
    print(f"Total tracked records: {len(tracks_df)}")


if __name__ == "__main__":
    main()
