# YOLOv7 Training Guide – Lamb Activity Detection

This document describes how we trained YOLOv7 models for our lamb activity detection experiments, in parallel with YOLOv5, following the same dataset and evaluation setup.

## 1. Clone YOLOv7 and install dependencies

From the repository root:

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt
```
We assume Python, PyTorch, and GPU drivers are already configured on the machine.

2. Dataset configuration

We reuse the same dataset and splits as for YOLOv5:
```bash
Images: data/images/

YOLO labels: data/labels_yolo/

Split lists:

data/splits/train.txt

data/splits/val.txt

data/splits/test.txt
```
The dataset definition is still in:
```bash
config/yolo_dataset.yaml
```
YOLOv7 supports this style of dataset config in the same way: it expects train, val, and test paths, as well as names and nc.

3. Train YOLOv7

From inside the yolov7 folder:
```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data ../config/yolo_dataset.yaml \
  --cfg cfg/training/yolov7-tiny.yaml \
  --weights yolov7-tiny.pt \
  --project ../runs_yolov7 \
  --name lamb_activity_yolov7_tiny
```
Notes:

--img 640 keeps the same spatial resolution as in our YOLOv5 experiments.

--cfg and --weights can be switched to other YOLOv7 variants depending on the model size we want to explore.

Outputs (weights, logs, metrics) will be stored in ../runs_yolov7/lamb_activity_yolov7_tiny.

4. Export detections for evaluation and tracking

After training, we generate detections on the test images and convert them to our standard CSV format.

From inside yolov7:
```bash
python detect.py \
  --img 640 \
  --weights ../runs_yolov7/lamb_activity_yolov7_tiny/weights/best.pt \
  --source ../data/images \
  --save-txt \
  --save-conf \
  --project ../runs_yolov7 \
  --name lamb_activity_test
```
This will write YOLO-format prediction text files under:
```bash
../runs_yolov7/lamb_activity_test/labels/
```
We then transform these predictions into a CSV with the columns:
```bash
frame_index,image_id,class_name,score,xmin,ymin,xmax,ymax
```
and save it as:
```bash
data/detections/sample_detections.csv
```
This conversion step is handled by a small utility script in our codebase (e.g. src/detection/export_detections_template.py), which reads YOLO text outputs and produces the unified detection CSV.

5. Evaluate YOLOv7 and compare with other models

Once the detection CSV is ready, we run from the repository root:
```bash
python -m src.detection.evaluate_detection
python -m src.tracking.track_from_detections
python -m src.analysis.behavior_stats
```
These commands:

1. Compute AP and mAP at IoU 0.5 against our VOC ground truth.

2. Generate data/detections/tracks.csv with tracked lamb IDs over time.

3. Summarize behavior statistics (time spent per activity per lamb and overall).

This allows us to directly compare YOLOv7 performance against YOLOv5 and the TensorFlow Object Detection models in a consistent evaluation framework.
