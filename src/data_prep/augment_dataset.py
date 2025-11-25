"""
Dataset augmentation script for lamb activity detection.

We take the original labeled images (YOLO format),
apply a series of geometric and photometric augmentations,
and write the augmented images + updated YOLO labels back to disk.

All paths and class names are read from config/paths.yaml.
"""

import os
import glob
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A


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

    return cfg


def _load_yolo_labels(label_path):
    """
    Read a YOLO label file.

    Format (one object per line):
        <class_id> <x_center> <y_center> <width> <height>
    where all coordinates are normalized to [0, 1].
    """
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                # silently skip malformed lines
                continue

            cls_id, xc, yc, w, h = parts
            boxes.append(
                {
                    "class": int(cls_id),
                    "xc": float(xc),
                    "yc": float(yc),
                    "w": float(w),
                    "h": float(h),
                }
            )

    return boxes


def _yolo_to_albu(boxes):
    """
    Convert YOLO-normalized boxes to Albumentations format.

    YOLO: cx, cy, w, h (normalized)
    Albumentations: x_min, y_min, x_max, y_max (normalized), label

    Returns a list: [x_min, y_min, x_max, y_max, class_id]
    """
    converted = []

    for b in boxes:
        x_min = b["xc"] - b["w"] / 2.0
        y_min = b["yc"] - b["h"] / 2.0
        x_max = b["xc"] + b["w"] / 2.0
        y_max = b["yc"] + b["h"] / 2.0

        # Clamp to [0, 1] just in case
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        converted.append([x_min, y_min, x_max, y_max, b["class"]])

    return converted


def _albu_to_yolo(boxes):
    """
    Convert Albumentations boxes back to YOLO-normalized format.

    Input: [x_min, y_min, x_max, y_max, class_id]
    Output: dict with keys: class, xc, yc, w, h
    """
    converted = []

    for b in boxes:
        x_min, y_min, x_max, y_max, cls_id = b

        w = x_max - x_min
        h = y_max - y_min
        xc = x_min + w / 2.0
        yc = y_min + h / 2.0

        converted.append(
            {
                "class": int(cls_id),
                "xc": float(xc),
                "yc": float(yc),
                "w": float(w),
                "h": float(h),
            }
        )

    return converted


def _save_yolo_labels(label_path, boxes):
    """
    Write YOLO labels back to disk.
    """
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:
        for b in boxes:
            f.write(
                f"{b['class']} "
                f"{b['xc']:.6f} {b['yc']:.6f} "
                f"{b['w']:.6f} {b['h']:.6f}\n"
            )


def main():
    """
    Entry point for dataset augmentation.

    Usage:
        python -m src.data_prep.augment_dataset
    """
    cfg = _load_config()

    images_dir = cfg["images_dir"]
    labels_dir = cfg["yolo_labels_dir"]

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Augmentation pipeline.
    # We keep it reasonably close to what we described in the paper:
    # flips, rotations, blur, noise, and brightness/contrast changes.
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.RandomBrightnessContrast(p=0.4),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["class_labels"],
        ),
    )

    # Collect all images in the dataset folder.
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )

    # Number of augmented versions per original image.
    # This can be adjusted to reach the desired final dataset size.
    augmentations_per_image = 3

    if not image_paths:
        print(f"No images found in: {images_dir}")
        return

    print(f"Found {len(image_paths)} base images.")
    print(f"Will create up to {augmentations_per_image} augmented copies per image.")

    for img_path in tqdm(image_paths, desc="Augmenting dataset"):
        img = cv2.imread(img_path)
        if img is None:
            # Just skip if OpenCV cannot read the file.
            continue

        height, width = img.shape[:2]

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")

        boxes = _load_yolo_labels(label_path)
        if not boxes:
            # If there is no annotation, we do not generate augmented versions.
            continue

        # Convert YOLO → Albumentations format
        albu_boxes = _yolo_to_albu(boxes)
        bboxes = [b[:4] for b in albu_boxes]
        class_labels = [b[4] for b in albu_boxes]

        for i in range(augmentations_per_image):
            augmented = transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels,
            )

            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]

            if not aug_bboxes:
                # Sometimes heavy augmentation can drop boxes; we just skip those.
                continue

            # Combine boxes with their labels for conversion back to YOLO.
            combined = [
                list(bb) + [cls_id]
                for bb, cls_id in zip(aug_bboxes, aug_classes)
            ]

            yolo_boxes = _albu_to_yolo(combined)

            # Build filenames for the augmented sample.
            aug_name = f"{base_name}_aug{i}"
            aug_img_path = os.path.join(images_dir, aug_name + ".jpg")
            aug_label_path = os.path.join(labels_dir, aug_name + ".txt")

            cv2.imwrite(aug_img_path, aug_img)
            _save_yolo_labels(aug_label_path, yolo_boxes)

    print("Augmentation finished successfully.")


if __name__ == "__main__":
    main()
