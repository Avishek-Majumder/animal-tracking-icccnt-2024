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
