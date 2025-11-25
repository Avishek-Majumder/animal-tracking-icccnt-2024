# src/data_prep/convert_yolo_to_voc.py

import os
import glob
import yaml
import cv2
from lxml import etree

CONFIG_PATH = "./config/paths.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def create_voc_xml(img_path, boxes, classes, img_width, img_height, voc_path):
    img_filename = os.path.basename(img_path)

    annotation = etree.Element("annotation")

    folder = etree.SubElement(annotation, "folder")
    folder.text = "images"

    filename = etree.SubElement(annotation, "filename")
    filename.text = img_filename

    size = etree.SubElement(annotation, "size")
    width = etree.SubElement(size, "width")
    width.text = str(img_width)
    height = etree.SubElement(size, "height")
    height.text = str(img_height)
    depth = etree.SubElement(size, "depth")
    depth.text = "3"

    segmented = etree.SubElement(annotation, "segmented")
    segmented.text = "0"

    for b in boxes:
        cls_id = b["class"]
        xc = b["xc"] * img_width
        yc = b["yc"] * img_height
        w = b["w"] * img_width
        h = b["h"] * img_height

        x_min = max(int(xc - w / 2), 0)
        y_min = max(int(yc - h / 2), 0)
        x_max = min(int(xc + w / 2), img_width - 1)
        y_max = min(int(yc + h / 2), img_height - 1)

        obj = etree.SubElement(annotation, "object")

        name = etree.SubElement(obj, "name")
        name.text = classes[cls_id]

        pose = etree.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = etree.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = etree.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = etree.SubElement(obj, "bndbox")
        xmin_el = etree.SubElement(bndbox, "xmin")
        xmin_el.text = str(x_min)
        ymin_el = etree.SubElement(bndbox, "ymin")
        ymin_el.text = str(y_min)
        xmax_el = etree.SubElement(bndbox, "xmax")
        xmax_el.text = str(x_max)
        ymax_el = etree.SubElement(bndbox, "ymax")
        ymax_el.text = str(y_max)

    tree = etree.ElementTree(annotation)
    os.makedirs(os.path.dirname(voc_path), exist_ok=True)
    tree.write(voc_path, pretty_print=True, xml_declaration=False, encoding="utf-8")

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

def main():
    cfg = load_config()
    img_dir = cfg["images_dir"]
    label_dir = cfg["yolo_labels_dir"]
    voc_dir = cfg["voc_xml_dir"]
    classes = cfg["classes"]

    os.makedirs(voc_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                       glob.glob(os.path.join(img_dir, "*.png")))

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        boxes = load_yolo_labels(label_path)
        if not boxes:
            continue

        voc_path = os.path.join(voc_dir, base + ".xml")
        create_voc_xml(img_path, boxes, classes, w, h, voc_path)

    print("Conversion YOLO → VOC XML completed.")

if __name__ == "__main__":
    main()
