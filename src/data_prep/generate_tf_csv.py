# src/data_prep/generate_tf_csv.py

import os
import glob
import yaml
import pandas as pd
import xml.etree.ElementTree as ET

CONFIG_PATH = "./config/paths.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def xml_to_csv(voc_dir):
    xml_list = []
    for xml_file in glob.glob(os.path.join(voc_dir, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        for member in root.findall("object"):
            label = member.find("name").text
            bndbox = member.find("bndbox")
            x_min = int(bndbox.find("xmin").text)
            y_min = int(bndbox.find("ymin").text)
            x_max = int(bndbox.find("xmax").text)
            y_max = int(bndbox.find("ymax").text)

            xml_list.append({
                "filename": filename,
                "width": width,
                "height": height,
                "class": label,
                "xmin": x_min,
                "ymin": y_min,
                "xmax": x_max,
                "ymax": y_max,
            })
    return pd.DataFrame(xml_list)

def main():
    cfg = load_config()
    voc_dir = cfg["voc_xml_dir"]
    tf_records_dir = cfg["tf_records_dir"]
    os.makedirs(tf_records_dir, exist_ok=True)

    df = xml_to_csv(voc_dir)
    out_csv = os.path.join(tf_records_dir, "annotations.csv")
    df.to_csv(out_csv, index=False)
    print(f"TF CSV saved to {out_csv}")

if __name__ == "__main__":
    main()
