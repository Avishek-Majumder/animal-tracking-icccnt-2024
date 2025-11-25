# src/detection/evaluate_detection.py

import os
import glob
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import pandas as pd

CONFIG_PATH = "./config/paths.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_ground_truth(voc_dir):
    """
    Returns dict:
        gt[image_id][class_name] = list of boxes [xmin,ymin,xmax,ymax]
    """
    gt = defaultdict(lambda: defaultdict(list))
    for xml_file in glob.glob(os.path.join(voc_dir, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        image_id = os.path.splitext(filename)[0]

        for obj in root.findall("object"):
            cls = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            gt[image_id][cls].append([xmin, ymin, xmax, ymax])
    return gt

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH

    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = areaA + areaB - inter

    if union <= 0:
        return 0.0
    return inter / union

def evaluate_map(detections_csv, gt, iou_thresh=0.5):
    """
    detections_csv columns:
    image_id, class_name, score, xmin, ymin, xmax, ymax
    """
    df = pd.read_csv(detections_csv)

    aps = {}
    classes = df["class_name"].unique()

    for cls in classes:
        cls_df = df[df["class_name"] == cls].copy()
        cls_df.sort_values("score", ascending=False, inplace=True)

        tp = []
        fp = []
        npos = sum(len(gt[img_id][cls]) for img_id in gt if cls in gt[img_id])

        # keep track of which gt boxes are already matched
        matched = {img_id: np.zeros(len(gt[img_id][cls])) if cls in gt[img_id] else []
                   for img_id in gt}

        for _, row in cls_df.iterrows():
            img_id = row["image_id"]
            box_det = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]

            if cls not in gt[img_id] or len(gt[img_id][cls]) == 0:
                fp.append(1)
                tp.append(0)
                continue

            gt_boxes = gt[img_id][cls]
            ious = [iou(box_det, g) for g in gt_boxes]
            best_iou = max(ious)
            best_idx = np.argmax(ious)

            if best_iou >= iou_thresh and matched[img_id][best_idx] == 0:
                tp.append(1)
                fp.append(0)
                matched[img_id][best_idx] = 1
            else:
                fp.append(1)
                tp.append(0)

        tp = np.array(tp)
        fp = np.array(fp)

        if tp.size == 0:
            aps[cls] = 0.0
            continue

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = tp / float(npos) if npos > 0 else np.zeros_like(tp)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        # 11-point interpolated AP
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        aps[cls] = ap

    mAP = np.mean(list(aps.values())) if aps else 0.0
    return aps, mAP

def main():
    cfg = load_config()
    voc_dir = cfg["voc_xml_dir"]
    detections_csv = "./data/detections/sample_detections.csv"  # adjust if needed

    gt = load_ground_truth(voc_dir)
    aps, mAP = evaluate_map(detections_csv, gt, iou_thresh=0.5)

    print("AP per class:")
    for cls, ap in aps.items():
        print(f"  {cls}: {ap:.4f}")
    print(f"\nOverall mAP@0.5: {mAP:.4f}")

if __name__ == "__main__":
    main()
