# src/data_prep/augment_dataset.py

import os
import glob
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

CONFIG_PATH = "./config/paths.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            boxes.append({
                "class": int(cls),
                "xc": float(xc),
                "yc": float(yc),
                "w": float(w),
                "h": float(h),
            })
    return boxes

def yolo_to_albumentations(boxes):
    # Albumentations expects [x_min, y_min, x_max, y_max] normalized in [0,1]
    converted = []
    for b in boxes:
        x_min = b["xc"] - b["w"] / 2
        y_min = b["yc"] - b["h"] / 2
        x_max = b["xc"] + b["w"] / 2
        y_max = b["yc"] + b["h"] / 2
        converted.append([x_min, y_min, x_max, y_max, b["class"]])
    return converted

def albumentations_to_yolo(boxes):
    # boxes: [x_min, y_min, x_max, y_max, class_id]
    converted = []
    for b in boxes:
        x_min, y_min, x_max, y_max, cls = b
        w = x_max - x_min
        h = y_max - y_min
        xc = x_min + w / 2
        yc = y_min + h / 2
        converted.append({
            "class": int(cls),
            "xc": float(xc),
            "yc": float(yc),
            "w": float(w),
            "h": float(h),
        })
    return converted

def save_yolo_labels(label_path, boxes):
    with open(label_path, "w") as f:
        for b in boxes:
            f.write(f"{b['class']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

def main():
    cfg = load_config()
    img_dir = cfg["images_dir"]
    label_dir = cfg["yolo_labels_dir"]

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Define augmentations similar to the paper: flip, rotation, blur, brightness, etc.
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.RandomBrightnessContrast(p=0.4),
        ],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["class_labels"])
    )

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                         glob.glob(os.path.join(img_dir, "*.png")))

    augmentations_per_image = 3  # adjust to reach ~9652 images

    for img_path in tqdm(image_paths, desc="Augmenting dataset"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")
        boxes = load_yolo_labels(label_path)

        if not boxes:
            continue

        alb_boxes = yolo_to_albumentations(boxes)
        bboxes = [b[:4] for b in alb_boxes]
        class_labels = [b[4] for b in alb_boxes]

        for i in range(augmentations_per_image):
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]

            if not aug_bboxes:
                continue

            # convert back to yolo
            combined = [list(bb) + [cls] for bb, cls in zip(aug_bboxes, aug_classes)]
            yolo_boxes = albumentations_to_yolo(combined)

            aug_name = f"{base_name}_aug{i}"
            aug_img_path = os.path.join(img_dir, aug_name + ".jpg")
            aug_label_path = os.path.join(label_dir, aug_name + ".txt")

            cv2.imwrite(aug_img_path, aug_img)
            save_yolo_labels(aug_label_path, yolo_boxes)

    print("Augmentation completed.")

if __name__ == "__main__":
    main()
