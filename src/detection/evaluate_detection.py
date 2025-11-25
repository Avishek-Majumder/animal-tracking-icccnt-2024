"""
Detection evaluation script (AP and mAP @ IoU=0.5).

We evaluate object detection performance by comparing:
    - Ground truth annotations from Pascal VOC XML files
    - Predicted detections from a CSV file

Expected detection CSV format (one row per bounding box):
    image_id,class_name,score,xmin,ymin,xmax,ymax

Where:
    - image_id   : filename without extension (e.g. "frame_0001")
    - class_name : one of the activity labels (standing / eating / laying)
    - score      : confidence score (float)
    - xmin, ymin, xmax, ymax : pixel coordinates

We then compute:
    - AP per class (using an 11-point interpolated precision–recall curve)
    - Overall mAP across all classes
"""

import os
import glob
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
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


def _load_ground_truth(voc_dir):
    """
    Load ground truth annotations from Pascal VOC XML files.

    Returns a nested dict:
        gt[image_id][class_name] = list of boxes [xmin, ymin, xmax, ymax]
    """
    gt = defaultdict(lambda: defaultdict(list))

    xml_files = sorted(glob.glob(os.path.join(voc_dir, "*.xml")))
    if not xml_files:
        print(f"No VOC XML files found in: {voc_dir}")
        return gt

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename")
        if not filename:
            continue

        image_id = os.path.splitext(filename)[0]

        for obj in root.findall("object"):
            cls_name = obj.findtext("name")
            bndbox = obj.find("bndbox")
            if cls_name is None or bndbox is None:
                continue

            xmin = int(bndbox.findtext("xmin", default="0"))
            ymin = int(bndbox.findtext("ymin", default="0"))
            xmax = int(bndbox.findtext("xmax", default="0"))
            ymax = int(bndbox.findtext("ymax", default="0"))

            gt[image_id][cls_name].append([xmin, ymin, xmax, ymax])

    return gt


def _iou(box_a, box_b):
    """
    Compute Intersection-over-Union (IoU) between two boxes.

    Boxes are [xmin, ymin, xmax, ymax] in pixel coordinates.
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a + 1)
    inter_h = max(0, y_b - y_a + 1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def _evaluate_class(detections_cls, gt, cls_name, iou_thresh=0.5):
    """
    Evaluate AP for a single class.

    detections_cls: DataFrame filtered to a single class, columns:
        image_id, class_name, score, xmin, ymin, xmax, ymax
    gt: ground truth dict from _load_ground_truth
    cls_name: name of the class (e.g. "standing")
    """
    # Number of positive ground truth boxes for this class
    n_positives = 0
    for image_id in gt:
        n_positives += len(gt[image_id].get(cls_name, []))

    if n_positives == 0:
        return 0.0

    # Sort detections by descending confidence score
    detections_cls = detections_cls.sort_values("score", ascending=False)

    tp = []
    fp = []

    # Keep track of which GT boxes have already been matched
    matched = {
        image_id: np.zeros(len(gt[image_id].get(cls_name, [])), dtype=bool)
        for image_id in gt
    }

    for _, row in detections_cls.iterrows():
        image_id = row["image_id"]
        det_box = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]

        gt_boxes = gt.get(image_id, {}).get(cls_name, [])

        if not gt_boxes:
            # No ground truth boxes of this class for this image → false positive
            fp.append(1)
            tp.append(0)
            continue

        # Compute IoU against all GT boxes of this class
        ious = np.array([_iou(det_box, g) for g in gt_boxes])
        best_index = int(np.argmax(ious))
        best_iou = ious[best_index]

        if best_iou >= iou_thresh and not matched[image_id][best_index]:
            # Correct match and this GT box was not used yet
            tp.append(1)
            fp.append(0)
            matched[image_id][best_index] = True
        else:
            # Either IoU too low or GT box already matched → false positive
            tp.append(0)
            fp.append(1)

    tp = np.array(tp)
    fp = np.array(fp)

    # Cumulative sums
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    # Precision and recall
    recall = cum_tp / float(n_positives)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

    # 11-point interpolated AP (0.0, 0.1, ..., 1.0)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recall >= t) == 0:
            p = 0.0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0

    return ap


def evaluate_map(detections_csv, gt, iou_thresh=0.5):
    """
    Compute AP per class and overall mAP.

    detections_csv: path to CSV with predicted detections
    gt: ground truth dict from _load_ground_truth
    """
    if not os.path.exists(detections_csv):
        raise FileNotFoundError(f"Detections CSV not found: {detections_csv}")

    df = pd.read_csv(detections_csv)

    required_cols = {"image_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Detections CSV is missing columns: {missing}")

    # It is common to filter out very low-confidence detections
    # before evaluation. Here we keep everything and leave
    # confidence handling to the precision–recall curve.
    classes = sorted(df["class_name"].unique())

    aps = {}
    for cls_name in classes:
        detections_cls = df[df["class_name"] == cls_name].copy()
        ap = _evaluate_class(detections_cls, gt, cls_name, iou_thresh=iou_thresh)
        aps[cls_name] = ap

    if aps:
        mAP = float(np.mean(list(aps.values())))
    else:
        mAP = 0.0

    return aps, mAP


def main():
    """
    Entry point for detection evaluation.

    Usage:
        python -m src.detection.evaluate_detection
    """
    cfg, _ = _load_config()

    voc_dir = cfg["voc_xml_dir"]
    detections_csv = cfg.get("detections_csv", "./data/detections/sample_detections.csv")

    print(f"Loading ground truth from: {voc_dir}")
    gt = _load_ground_truth(voc_dir)

    if not gt:
        print("No ground truth annotations found. Check VOC XML files.")
        return

    print(f"Evaluating detections from: {detections_csv}")
    aps, mAP = evaluate_map(detections_csv, gt, iou_thresh=0.5)

    print("\nAP per class (IoU = 0.5):")
    for cls_name, ap in aps.items():
        print(f"  {cls_name:10s} : {ap:.4f}")

    print(f"\nOverall mAP@0.5: {mAP:.4f}")


if __name__ == "__main__":
    main()
