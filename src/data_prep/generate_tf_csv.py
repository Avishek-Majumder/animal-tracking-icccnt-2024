"""
Generate a CSV annotation file from Pascal VOC XML annotations.

This script reads all XML files in data/labels_voc_xml (or the folder
configured in config/paths.yaml) and produces a single CSV file:

    data/tf_records/annotations.csv

Each row corresponds to one bounding box and contains:
    filename,width,height,class,xmin,ymin,xmax,ymax

This CSV can then be used to generate TFRecord files for the
TensorFlow Object Detection API.
"""

import os
import glob
import yaml
import pandas as pd
import xml.etree.ElementTree as ET


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


def _xml_to_records(voc_dir):
    """
    Parse all VOC XML files under voc_dir and return a list of dicts.

    Each dict has:
        filename, width, height, class, xmin, ymin, xmax, ymax
    """
    records = []

    xml_files = sorted(glob.glob(os.path.join(voc_dir, "*.xml")))
    if not xml_files:
        print(f"No XML files found in: {voc_dir}")
        return records

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename")
        size_el = root.find("size")

        if size_el is None:
            # If size is missing, we cannot reliably use this annotation.
            continue

        width = int(size_el.findtext("width", default="0"))
        height = int(size_el.findtext("height", default="0"))

        # One row per <object>
        for obj in root.findall("object"):
            cls_name = obj.findtext("name")
            bndbox = obj.find("bndbox")

            if bndbox is None:
                continue

            xmin = int(bndbox.findtext("xmin", default="0"))
            ymin = int(bndbox.findtext("ymin", default="0"))
            xmax = int(bndbox.findtext("xmax", default="0"))
            ymax = int(bndbox.findtext("ymax", default="0"))

            records.append(
                {
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "class": cls_name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )

    return records


def main():
    """
    Entry point for VOC XML → CSV conversion.

    Usage:
        python -m src.data_prep.generate_tf_csv
    """
    cfg, _ = _load_config()

    voc_dir = cfg["voc_xml_dir"]
    tf_records_dir = cfg["tf_records_dir"]

    os.makedirs(tf_records_dir, exist_ok=True)

    print(f"Reading VOC XML annotations from: {voc_dir}")
    records = _xml_to_records(voc_dir)

    if not records:
        print("No records extracted. Check that XML files exist and are valid.")
        return

    df = pd.DataFrame(records)
    out_csv = os.path.join(tf_records_dir, "annotations.csv")

    df.to_csv(out_csv, index=False)
    print(f"CSV annotations saved to: {out_csv}")
    print(f"Total annotated boxes: {len(df)}")


if __name__ == "__main__":
    main()
