"""
Train / validation / test split script.

We read all images from the dataset folder, shuffle them,
and write three text files that list the image paths:

    data/splits/train.txt
    data/splits/val.txt
    data/splits/test.txt

These files are then used by YOLOv5 / YOLOv7 via config/yolo_dataset.yaml.
"""

import os
import glob
import random
import yaml


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


def _collect_image_paths(images_dir):
    """
    Collect all image paths (jpg, jpeg, png) from the given directory.
    """
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths = []

    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(images_dir, pattern)))

    # Sort for reproducibility before shuffling
    paths = sorted(paths)
    return paths


def _write_list(file_path, paths):
    """
    Write one path per line to the split file.

    We use absolute paths here to avoid ambiguity when training
    from different working directories.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        for p in paths:
            abs_path = os.path.abspath(p)
            f.write(abs_path + "\n")


def main(train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Create train/val/test splits.

    The remaining portion (1 - train_ratio - val_ratio) is used for test.
    """
    cfg, _ = _load_config()

    images_dir = cfg["images_dir"]
    splits_dir = cfg["splits_dir"]

    image_paths = _collect_image_paths(images_dir)

    if not image_paths:
        print(f"No images found in: {images_dir}")
        return

    random.seed(seed)
    random.shuffle(image_paths)

    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train : n_train + n_val]
    test_paths = image_paths[n_train + n_val :]

    train_file = os.path.join(splits_dir, "train.txt")
    val_file = os.path.join(splits_dir, "val.txt")
    test_file = os.path.join(splits_dir, "test.txt")

    _write_list(train_file, train_paths)
    _write_list(val_file, val_paths)
    _write_list(test_file, test_paths)

    print(f"Total images: {n_total}")
    print(f"Train: {len(train_paths)}")
    print(f"Val:   {len(val_paths)}")
    print(f"Test:  {len(test_paths)}")
    print(f"Splits saved to: {splits_dir}")


if __name__ == "__main__":
    main()
