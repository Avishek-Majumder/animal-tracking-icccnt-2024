# Lamb Activity Tracking and Behavior Analysis (ICCCNT 2024 – Official Codebase)

This repository contains the official code implementation of our ICCCNT 2024 paper:

> **Advancements in Animal Tracking: Assessing Deep Learning Algorithms**  
> IEEE ICCCNT 2024  
> Paper link: https://ieeexplore.ieee.org/abstract/document/10724124

Our goal with this codebase is to provide a clean, faithful, and readable implementation of the full experimental pipeline described in the paper:

- Detect lambs in three activity states: **standing**, **eating**, and **laying**
- Compare **YOLOv5** and **YOLOv7** with TensorFlow Object Detection models  
  (**Faster R-CNN ResNet152** and **SSD ResNet101**)
- Track individual lambs over time using a **centroid-based tracking** method
- Derive **behavior statistics** from trajectories (e.g., time spent per activity)

The repository is organized so that anyone reading our paper can easily follow and reproduce the key steps.

---

## Repository layout

The directory structure reflects the pipeline of our study:

- `config/` – Central configuration of paths, classes, and dataset definitions  
- `data/` – Images, annotations, dataset splits, detections, and tracking outputs  
- `src/data_prep/` – Dataset augmentation and format conversions (YOLO → VOC → TF CSV)  
- `src/detection/` – Evaluation code for AP / mAP at IoU 0.5  
- `src/tracking/` – Centroid-based multi-object tracker to maintain lamb IDs over time  
- `src/analysis/` – Scripts for behavior statistics (e.g., time per activity per lamb)  
- `scripts/` – Markdown guides for running YOLOv5 / YOLOv7 training with this dataset

We use the official YOLOv5/YOLOv7 and TensorFlow Object Detection repositories for model training; this repo provides everything around them:
data preparation, evaluation, tracking, and analysis.

---

## Dataset and labels

In our experiments, we worked with lamb videos from a commercial farm, extracted frames, and manually annotated three activity classes:

- `standing`
- `eating`
- `laying`

In this repository we assume:

- Images are stored in: `data/images/`
- YOLO-format labels are stored in: `data/labels_yolo/`

YOLO label format (per line):

```text
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

Class IDs:

- `0` → `standing`  
- `1` → `eating`  
- `2` → `laying`  

These mappings are configurable in `config/paths.yaml`, so we can modify them without touching the core code.

---

## Pipeline overview

The full workflow implemented in this codebase mirrors the pipeline described in our paper:

1. **Data preparation and augmentation**
   - Load original labeled images
   - Apply geometric and photometric augmentations  
     (flips, rotations, blur, brightness/contrast changes, noise)
   - Keep annotations in YOLO format after augmentation

2. **Dataset splitting**
   - Create `train`, `val`, and `test` splits
   - Save file lists that are directly compatible with YOLOv5 / YOLOv7

3. **Format conversion for TensorFlow models**
   - Convert YOLO annotations → Pascal VOC XML
   - Convert VOC XML → CSV (and then to TFRecord in a standard TF Object Detection workflow)

4. **Model training (external repositories)**
   - Train YOLOv5 / YOLOv7 using `config/yolo_dataset.yaml`
   - Train Faster R-CNN ResNet152 and SSD ResNet101 using the TensorFlow Object Detection API

5. **Detection evaluation**
   - Collect model detections in a simple CSV format
   - Compute AP per class and overall mAP at IoU 0.5, as reported in the paper

6. **Tracking and behavior analysis**
   - Run centroid-based tracking over frame-by-frame detections
   - Export per-lamb trajectories to CSV
   - Compute time spent in each activity per lamb and overall behavior statistics

---

## Design principles

While building this repository, we focused on:

- **Clarity** – Descriptive variable names, straightforward logic, and minimal “magic”  
- **Faithfulness to the paper** – The structure follows the methodology we described in the ICCCNT 2024 publication  
- **Extensibility** – The code can be adapted to other farms, animal species, or additional behaviors with minimal changes

We want this repo to be something we, as authors, are proud to show alongside the paper.

---

## Getting started

The intended order of use is:

1. Set paths and classes in `config/paths.yaml`.
2. Prepare and augment the dataset using the scripts in `src/data_prep/`.
3. Split the dataset into train/val/test (`src/data_prep/split_dataset.py`).
4. Convert labels for TensorFlow models  
   (`src/data_prep/convert_yolo_to_voc.py` and `src/data_prep/generate_tf_csv.py`).
5. Train YOLOv5 / YOLOv7 using the guides in `scripts/`.
6. Export detections and compute metrics (`src/detection/evaluate_detection.py`).
7. Run tracking and behavior analysis  
   (`src/tracking/track_from_detections.py` and `src/analysis/behavior_stats.py`).

This README serves as a high-level map for the entire codebase so that our implementation and our paper stay tightly aligned.

