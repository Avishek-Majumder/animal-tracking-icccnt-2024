"""
Convert YOLO annotations to Pascal VOC XML format.

This script is used to prepare data for the TensorFlow Object Detection API.
We read YOLO-normalized bounding boxes, convert them to absolute pixel
coordinates, and write a VOC-style XML file per image.

Each XML file contains:
    - image size
    - one <object> tag per lamb
    - bounding box (xmin, ymin, xmax, ymax)
    - class name (standing / eating / laying)

All paths and classes are read from config/paths.yaml.
"""

import os
import glob
import yaml
import cv2
from lxml import etree


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


def _load_yolo_labels(label_path):
    """
    Read a YOLO label file.

    Format:
        <class_id> <x_center> <y_center> <width> <height>
    with all coordinates normalized to [0, 1].
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
                # Skip malformed lines
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


def _yolo_to_pixel(box, img_width, img_height):
    """
    Convert a single YOLO-normalized box to absolute pixel coordinates.

    YOLO: cx, cy, w, h in [0, 1]
    VOC: xmin, ymin, xmax, ymax in pixel coordinates
    """
    xc = box["xc"] * img_width
    yc = box["yc"] * img_height
    w = box["w"] * img_width
    h = box["h"] * img_height

    x_min = int(round(xc - w / 2.0))
    y_min = int(round(yc - h / 2.0))
    x_max = int(round(xc + w / 2.0))
    y_max = int(round(yc + h / 2.0))

    # Clamp to valid image region
    x_min = max(0, min(img_width - 1, x_min))
    y_min = max(0, min(img_height - 1, y_min))
    x_max = max(0, min(img_width - 1, x_max))
    y_max = max(0, min(img_height - 1, y_max))

    return x_min, y_min, x_max, y_max


def _create_voc_xml(
    img_path,
    img_width,
    img_height,
    boxes,
    class_names,
    xml_output_path,
):
    """
    Build a Pascal VOC XML tree and write it to disk.
    """
    filename = os.path.basename(img_path)

    annotation = etree.Element("annotation")

    folder_el = etree.SubElement(annotation, "folder")
    folder_el.text = "images"

    filename_el = etree.SubElement(annotation, "filename")
    filename_el.text = filename

    size_el = etree.SubElement(annotation, "size")
    width_el = etree.SubElement(size_el, "width")
    width_el.text = str(img_width)
    height_el = etree.SubElement(size_el, "height")
    height_el.text = str(img_height)
    depth_el = etree.SubElement(size_el, "depth")
    depth_el.text = "3"  # we assume RGB images

    segmented_el = etree.SubElement(annotation, "segmented")
    segmented_el.text = "0"

    for b in boxes:
        cls_id = b["class"]
        x_min, y_min, x_max, y_max = _yolo_to_pixel(b, img_width, img_height)

        obj_el = etree.SubElement(annotation, "object")

        name_el = etree.SubElement(obj_el, "name")
        name_el.text = class_names[cls_id]

        pose_el = etree.SubElement(obj_el, "pose")
        pose_el.text = "Unspecified"

        truncated_el = etree.SubElement(obj_el, "truncated")
        truncated_el.text = "0"

        difficult_el = etree.SubElement(obj_el, "difficult")
        difficult_el.text = "0"

        bndbox_el = etree.SubElement(obj_el, "bndbox")
        xmin_el = etree.SubElement(bndbox_el, "xmin")
        xmin_el.text = str(x_min)
        ymin_el = etree.SubElement(bndbox_el, "ymin")
        ymin_el.text = str(y_min)
        xmax_el = etree.SubElement(bndbox_el, "xmax")
        xmax_el.text = str(x_max)
        ymax_el = etree.SubElement(bndbox_el, "ymax")
        ymax_el.text = str(y_max)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(xml_output_path), exist_ok=True)

    tree = etree.ElementTree(annotation)
    tree.write(xml_output_path, pretty_print=True, xml_declaration=False, encoding="utf-8")


def main():
    """
    Entry point for YOLO → VOC XML conversion.

    Usage:
        python -m src.data_prep.convert_yolo_to_voc
    """
    cfg, _ = _load_config()

    images_dir = cfg["images_dir"]
    labels_dir = cfg["yolo_labels_dir"]
    voc_dir = cfg["voc_xml_dir"]
    class_names = cfg["classes"]  # e.g. ["standing", "eating", "laying"]

    os.makedirs(voc_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )

    if not image_paths:
        print(f"No images found in: {images_dir}")
        return

    print(f"Found {len(image_paths)} images. Converting YOLO → VOC XML...")

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            # If we cannot read the image, skip it.
            continue

        img_height, img_width = img.shape[:2]

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")

        boxes = _load_yolo_labels(label_path)
        if not boxes:
            # If there is no annotation, we skip VOC generation for this image.
            continue

        xml_output_path = os.path.join(voc_dir, base_name + ".xml")
        _create_voc_xml(
            img_path=img_path,
            img_width=img_width,
            img_height=img_height,
            boxes=boxes,
            class_names=class_names,
            xml_output_path=xml_output_path,
        )

    print(f"Conversion completed. VOC XML files are in: {voc_dir}")


if __name__ == "__main__":
    main()
