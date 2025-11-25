"""
Utility script to convert YOLO prediction text files into a unified CSV format.

YOLO's detect.py typically writes one text file per image in a "labels" folder, e.g.:

    runs/.../labels/frame_0001.txt

Each line in those files usually looks like:

    <class_id> <x_center> <y_center> <width> <height> <confidence?>

Where:
    - coordinates are normalized to [0, 1]
    - confidence may or may not be present depending on the version/options

This script:
    1. Reads all such prediction files from a given labels_dir.
    2. Finds the matching image in images_dir to get width/height.
    3. Converts YOLO-normalized boxes back to pixel coordinates.
    4. Maps class_id → class_name using config/paths.yaml.
    5. Assigns a deterministic frame_index based on sorted filenames.
    6. Writes a CSV with columns:

        frame_index,image_id,class_name,score,xmin,ymin,xmax,ymax

The resulting CSV is compatible with:
    - src.detection.evaluate_detection
    - src.tracking.track_from_detections
"""

import os
import glob
import argparse

import yaml
import cv2
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


def _find_image_for_label(images_dir, base_name):
    """
    Given the base filename (without extension), try to locate the corresponding image
    in images_dir with common image extensions.
    """
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _parse_yolo_prediction_line(line):
    """
    Parse a single line from a YOLO prediction text file.

    Expected formats:
        class x_center y_center width height
        class x_center y_center width height confidence

    Returns:
        class_id (int),
        xc (float), yc (float), w (float), h (float),
        score (float or None)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    cls_id = int(parts[0])
    xc = float(parts[1])
    yc = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    score = None
    if len(parts) >= 6:
        try:
            score = float(parts[5])
        except ValueError:
            score = None

    return cls_id, xc, yc, w, h, score


def _yolo_to_pixel(xc, yc, w, h, img_width, img_height):
    """
    Convert YOLO-normalized bounding box to absolute pixel coordinates.

    YOLO (normalized): xc, yc, w, h in [0, 1]
    Pixel coords: xmin, ymin, xmax, ymax
    """
    xc_abs = xc * img_width
    yc_abs = yc * img_height
    w_abs = w * img_width
    h_abs = h * img_height

    xmin = int(round(xc_abs - w_abs / 2.0))
    ymin = int(round(yc_abs - h_abs / 2.0))
    xmax = int(round(xc_abs + w_abs / 2.0))
    ymax = int(round(yc_abs + h_abs / 2.0))

    xmin = max(0, min(img_width - 1, xmin))
    ymin = max(0, min(img_height - 1, ymin))
    xmax = max(0, min(img_width - 1, xmax))
    ymax = max(0, min(img_height - 1, ymax))

    return xmin, ymin, xmax, ymax


def export_detections(labels_dir, images_dir, output_csv, default_score=1.0):
    """
    Convert YOLO prediction text files into a CSV.

    Parameters
    ----------
    labels_dir : str
        Directory containing YOLO prediction *.txt files.
    images_dir : str
        Directory containing the corresponding images.
    output_csv : str
        Path to the CSV file to write.
    default_score : float, optional
        Fallback confidence score if none is present in the YOLO files.
    """
    cfg, _ = _load_config()
    class_names = cfg.get("classes", [])

    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"labels_dir does not exist: {labels_dir}")

    txt_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if not txt_files:
        print(f"No prediction text files found in: {labels_dir}")
        return

    print(f"Found {len(txt_files)} prediction files in: {labels_dir}")

    # Determine a deterministic frame_index per image by sorting the base names.
    base_names = [os.path.splitext(os.path.basename(p))[0] for p in txt_files]
    sorted_unique = sorted(set(base_names))
    frame_index_map = {name: idx for idx, name in enumerate(sorted_unique)}

    records = []

    for txt_path in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        image_path = _find_image_for_label(images_dir, base_name)

        if image_path is None:
            # If the matching image cannot be found, we skip this file.
            print(f"Warning: image not found for prediction file: {txt_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: could not read image: {image_path}")
            continue

        img_height, img_width = img.shape[:2]

        frame_index = frame_index_map[base_name]
        image_id = base_name

        with open(txt_path, "r") as f:
            for line in f:
                parsed = _parse_yolo_prediction_line(line)
                if parsed is None:
                    continue

                cls_id, xc, yc, w, h, score = parsed

                if cls_id < 0 or cls_id >= len(class_names):
                    # Unknown class id, skip.
                    continue

                class_name = class_names[cls_id]
                if score is None:
                    score = float(default_score)

                xmin, ymin, xmax, ymax = _yolo_to_pixel(
                    xc, yc, w, h, img_width, img_height
                )

                records.append(
                    {
                        "frame_index": int(frame_index),
                        "image_id": image_id,
                        "class_name": class_name,
                        "score": float(score),
                        "xmin": float(xmin),
                        "ymin": float(ymin),
                        "xmax": float(xmax),
                        "ymax": float(ymax),
                    }
                )

    if not records:
        print("No valid detections parsed from YOLO prediction files.")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"Export complete. Detection CSV written to: {output_csv}")
    print(f"Total detections: {len(df)}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert YOLO prediction text files to a unified CSV format."
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Directory containing YOLO prediction *.txt files.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=False,
        help="Directory containing the corresponding images (default: config images_dir).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=False,
        help="Path to the output CSV file (default: config detections_csv).",
    )
    parser.add_argument(
        "--default_score",
        type=float,
        default=1.0,
        help="Fallback confidence score if none is present in the YOLO files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg, _ = _load_config()

    images_dir = args.images_dir or cfg["images_dir"]
    output_csv = args.output_csv or cfg.get("detections_csv", "./data/detections/sample_detections.csv")

    export_detections(
        labels_dir=args.labels_dir,
        images_dir=images_dir,
        output_csv=output_csv,
        default_score=args.default_score,
    )


if __name__ == "__main__":
    main()
