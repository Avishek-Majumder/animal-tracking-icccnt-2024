# src/data_prep/split_dataset.py

import os
import glob
import random
import yaml

CONFIG_PATH = "./config/paths.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def main(train_ratio=0.7, val_ratio=0.2, seed=42):
    random.seed(seed)
    cfg = load_config()

    images_dir = cfg["images_dir"]
    splits_dir = cfg["splits_dir"]

    os.makedirs(splits_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                         glob.glob(os.path.join(images_dir, "*.png")))

    random.shuffle(image_paths)

    n = len(image_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:n_train + n_val]
    test_paths = image_paths[n_train + n_val:]

    def write_split(file_path, paths):
        with open(file_path, "w") as f:
            for p in paths:
                f.write(os.path.abspath(p) + "\n")

    write_split(os.path.join(splits_dir, "train.txt"), train_paths)
    write_split(os.path.join(splits_dir, "val.txt"), val_paths)
    write_split(os.path.join(splits_dir, "test.txt"), test_paths)

    print(f"Total images: {n}")
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

if __name__ == "__main__":
    main()
