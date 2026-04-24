# Urban Road Object Detection and Localization
 
> Computer Vision project — Detection and localization of objects in urban road environments using YOLOv8s, with class imbalance analysis, Data Augmentation (Albumentations) and Oversampling strategies applied to the TESI Computer Vision Dataset.
 
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/Dataset-Roboflow%20Universe-orange)](https://universe.roboflow.com/cv-2022-kyjj6/tesi)
[![Colab](https://img.shields.io/badge/Notebook-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-Academic-green)](./LICENSE)
 
---
 
## Table of Contents
 
- [Project Overview](#project-overview)
- [Target Classes](#target-classes)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Output Paths Explained](#output-paths-explained)
- [Installation](#installation)
- [Usage](#usage)
- [Training Configuration](#training-configuration)
- [Inference](#inference)
- [References](#references)
---
 
## Project Overview
 
This project addresses the problem of **object detection and localization in urban road environments** using deep convolutional neural networks. The model used is **YOLOv8s** (*You Only Look Once*, version 8 small), trained on the ["TESI Computer Vision Dataset"](https://universe.roboflow.com/cv-2022-kyjj6/tesi) from Roboflow Universe.
 
The project was developed and executed on **Google Colab** with Google Drive for data persistence, covering the full ML pipeline: exploratory data analysis (EDA), class imbalance visualization, data balancing strategies (Data Augmentation and Oversampling), and model training.
 
---
 
## Target Classes
 
| ID | Class          | Description                              |
|----|----------------|------------------------------------------|
| 0  | `bus`          | Buses and public transport               |
| 1  | `construction` | Construction elements / road works       |
| 2  | `cyclist`      | Cyclists                                 |
| 3  | `pedestrian`   | Pedestrians                              |
| 4  | `vehicle`      | Vehicles (cars, trucks, vans, etc.)      |
 
---
 
## Dataset
 
**Source:** [TESI Computer Vision Dataset — Roboflow Universe](https://universe.roboflow.com/cv-2022-kyjj6/tesi)
 
| Split     | bus   | construction | cyclist | pedestrian | vehicle | Total      |
|-----------|-------|--------------|---------|------------|---------|------------|
| Train     | 797   | 1,781        | 532     | 2,588      | 6,515   | **12,213** |
| Valid     | 217   | 534          | 137     | 760        | 1,868   | **3,516**  |
| Test      | 114   | 254          | 82      | 368        | 922     | **1,740**  |
| **Global**| 1,128 | 2,569        | 751     | 3,716      | 9,305   | **17,469** |
 
> ⚠️ **Class imbalance:** `vehicle` outnumbers `cyclist` by a ratio of ~**12.4×**, which motivates the use of balancing techniques.
 
---
 
## Repository Structure
 
```
deteccion-localizacion-objetos-urbanos/
│
├── runs/                                   <- YOLOv8 training output (auto-generated)
│   └── detect/
│       └── train/                          <- Results of the training run
│           ├── weights/
│           │   ├── best.pt                 <- Best model weights (lowest val loss)
│           │   └── last.pt                 <- Weights from the last epoch
│           ├── results.csv                 <- Per-epoch metrics (loss, mAP, precision, recall)
│           ├── results.png                 <- Training curves plot
│           ├── confusion_matrix.png        <- Confusion matrix on validation set
│           ├── confusion_matrix_normalized.png
│           ├── PR_curve.png                <- Precision-Recall curve
│           ├── F1_curve.png                <- F1 score curve
│           ├── P_curve.png                 <- Precision curve
│           ├── R_curve.png                 <- Recall curve
│           ├── val_batch0_labels.jpg       <- Ground truth labels on validation batch
│           ├── val_batch0_pred.jpg         <- Model predictions on validation batch
│           └── args.yaml                   <- Hyperparameters used for this run
│
├── tesiComputerVisionDatasetV1-1/          ← Dataset root (not committed, generated locally)
│   ├── train/
│   │   ├── images/                         ← Original training images
│   │   ├── labels/                         ← Original YOLO labels (.txt)
│   │   ├── images_OS/                      ← Oversampled images (minority classes ×3)
│   │   └── labels_OS/                      ← Labels for oversampled images
│   ├── train_DA/
│   │   ├── images/                         ← Copy of train images for DA pipeline
│   │   ├── labels/                         ← Copy of train labels for DA pipeline
│   │   ├── images_DA/                      ← Augmented images (albumentations output)
│   │   └── labels_DA/                      ← Labels for augmented images
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── dataModel.yaml                          ← YOLO dataset config (train/val paths + classes)
├── deteccion_localizacion_objetos_via_urbana.ipynb  ← Main notebook
├── yolov8s.pt                              ← YOLOv8s base weights (pretrained on COCO)
├── yolo26n.pt                              ← Alternative nano weights
├── requirements.txt
├── .gitignore
└── README.md
```
 
---
 
## Output Paths Explained
 
This section details every folder under `runs/detect/train/` that YOLOv8 generates automatically after training, so you know exactly where to find each output.
 
### `runs/detect/train/weights/`
 
| File       | What it is                                                                 |
|------------|----------------------------------------------------------------------------|
| `best.pt`  | Model checkpoint with the **best validation mAP** across all epochs. Use this for inference. |
| `last.pt`  | Checkpoint saved at the **last epoch**. Useful to resume training.         |
 
```python
# Load the best model for inference
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
```
 
### `runs/detect/train/results.csv`
 
Per-epoch log with all training and validation metrics:
 
| Column                  | Description                                      |
|-------------------------|--------------------------------------------------|
| `epoch`                 | Epoch number                                     |
| `train/box_loss`        | Bounding box regression loss (train)             |
| `train/cls_loss`        | Classification loss (train)                      |
| `train/dfl_loss`        | Distribution focal loss (train)                  |
| `val/box_loss`          | Bounding box regression loss (validation)        |
| `val/cls_loss`          | Classification loss (validation)                 |
| `val/dfl_loss`          | Distribution focal loss (validation)             |
| `metrics/precision(B)`  | Precision on validation set                      |
| `metrics/recall(B)`     | Recall on validation set                         |
| `metrics/mAP50(B)`      | Mean Average Precision @ IoU=0.50                |
| `metrics/mAP50-95(B)`   | Mean Average Precision @ IoU=0.50:0.95           |
 
### `runs/detect/train/` — Plots and visual outputs
 
| File                              | What it shows                                                   |
|-----------------------------------|-----------------------------------------------------------------|
| `results.png`                     | All training curves (losses + mAP) across epochs               |
| `confusion_matrix.png`            | Raw confusion matrix on the validation set                      |
| `confusion_matrix_normalized.png` | Normalized confusion matrix (row-wise, useful for imbalanced data) |
| `PR_curve.png`                    | Precision-Recall curve per class                                |
| `F1_curve.png`                    | F1 score vs. confidence threshold per class                     |
| `P_curve.png`                     | Precision vs. confidence threshold                              |
| `R_curve.png`                     | Recall vs. confidence threshold                                 |
| `val_batch0_labels.jpg`           | Ground truth bounding boxes on the first validation batch       |
| `val_batch0_pred.jpg`             | Model predictions on the first validation batch                 |
 
### `runs/detect/train/args.yaml`
 
Full record of every hyperparameter and argument used in the training run. Useful for reproducibility.
 
```yaml
# Example entries from args.yaml
model: yolov8s.pt
data: dataModel.yaml
epochs: 70
imgsz: 640
batch: 16
cls: 2.0         # classification loss weight (increased to help minority classes)
project: runs
```
 
---
 
## Installation
 
### Prerequisites
 
- Python 3.10+
- CUDA-compatible GPU recommended (tested on Google Colab T4)
### Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### `requirements.txt`
 
```
ultralytics
roboflow
albumentations
opencv-python
torch
torchvision
pandas
matplotlib
numpy
jupyterlab
ipykernel
```
 
---
   
## Author
 
Developed as part of the **Neural Networks 2026-1** course.
 
- **Repository:** [github.com/SebastianIsaacRV/deteccion-localizacion-objetos-urbanos](https://github.com/SebastianIsaacRV/deteccion-localizacion-objetos-urbanos)
- **Dataset:** [TESI Computer Vision Dataset — Roboflow Universe](https://universe.roboflow.com/cv-2022-kyjj6/tesi)
- **Base model:** [YOLOv8s — Ultralytics](https://github.com/ultralytics/ultralytics)
---
 
## License
 
This project is for academic use. The original dataset is subject to the terms of [Roboflow Universe](https://roboflow.com/licensing).

---

## Results

## Training Results — YOLOv8s

## Experiment Configuration

| Parameter | Value |
|---|---|
| Base model | `yolov8s.pt` (pretrained) |
| Epochs | 70 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | AdamW (lr=0.001111, momentum=0.9) |
| Classes | 5 |
| Training images | 7,679 (6,709 annotated + 970 backgrounds) |
| Validation images | 1,920 |
| Hardware | NVIDIA A100-SXM4-80GB |
| Total training time | 1.41 hours |

---

## Overall Metrics (Best Model)

| Metric | Value |
|---|---|
| Precision (P) | **81.9%** |
| Recall (R) | **72.6%** |
| mAP@50 | **79.9%** |
| mAP@50-95 | **50.8%** |
| Fitness | 0.5080 |

> Best checkpoint saved at `runs/detect/train/weights/best.pt` (22.5 MB).

---

## Per-Class Metrics (Best Model)

| Class | Instances (val) | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| bus | 281 | 82.0% | 81.5% | 85.8% | 65.6% |
| construction | 2,637 | 76.9% | 64.6% | 72.6% | 37.4% |
| cyclist | 221 | 75.9% | 65.7% | 71.1% | 46.4% |
| pedestrian | 3,304 | 84.2% | 67.8% | 78.3% | 42.0% |
| vehicle | 18,390 | 90.6% | 83.4% | 91.6% | 62.5% |
| **all** | **24,833** | **81.9%** | **72.6%** | **79.9%** | **50.8%** |

---

## mAP Training Progression

| Epoch | mAP@50 | mAP@50-95 |
|---:|---:|---:|
| 1 | 0.495 | 0.248 |
| 10 | 0.620 | 0.352 |
| 20 | 0.706 | 0.414 |
| 30 | 0.752 | 0.452 |
| 40 | 0.779 | 0.481 |
| 50 | 0.792 | 0.497 |
| 60 | 0.798 | 0.508 |
| 70 | 0.800 | 0.508 |

---

## Observations

- **`vehicle`** is the best-performing class (mAP@50 = 91.6%), benefiting from its dominant representation in the dataset (18,390 instances).
- **`bus`** achieves the highest mAP@50-95 (65.6%), indicating strong localization precision in addition to accurate detection.
- **`construction`** has the lowest recall (64.6%), suggesting a higher false negative rate, likely due to the visual heterogeneity of construction-related objects.
- **`cyclist`** and **`pedestrian`** show expected cross-confusion due to morphological similarity, reflected in their moderate recall scores.
- The model converges steadily across all 70 epochs with no clear signs of overfitting.

## Confusion Matrix — YOLOv8s

> Source: `runs/detect/train/confusion_matrix.png` and `confusion_matrix_normalized.png`

---

## Raw Counts

| Predicted \ True | bus | construction | cyclist | pedestrian | vehicle | background |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **bus** | **228** | 0 | 0 | 0 | 26 | 63 |
| **construction** | 0 | **2026** | 0 | 3 | 8 | 1081 |
| **cyclist** | 0 | 0 | **151** | 32 | 6 | 45 |
| **pedestrian** | 4 | 41 | 2497 | **2497** | 9 | 793 |
| **vehicle** | 35 | 1 | 2 | 7 | **16207** | 2807 |
| **background (FN)** | 18 | 606 | 27 | 765 | 2134 | — |

---

## Normalized Confusion Matrix

> Values represent the proportion of ground truth instances per class.
> Cells with value 0.00 are omitted for readability.

| Predicted \ True | bus | construction | cyclist | pedestrian | vehicle | background |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **bus** | **0.81** | — | — | — | — | 0.01 |
| **construction** | — | **0.77** | — | — | — | 0.23 |
| **cyclist** | — | — | **0.68** | 0.01 | — | 0.01 |
| **pedestrian** | — | — | 0.19 | **0.76** | — | 0.17 |
| **vehicle** | 0.12 | — | 0.01 | — | **0.88** | 0.59 |
| **background (FN)** | 0.06 | 0.23 | 0.12 | 0.23 | 0.12 | — |

---

## Per-Class Summary

| Class | TP Rate (diagonal) | Main confusion | FN Rate (→ background) |
|---|:---:|---|:---:|
| bus | 0.81 | Confused with `vehicle` (0.12) | 0.06 |
| construction | 0.77 | Lost to background (0.23) | 0.23 |
| cyclist | 0.68 | Confused with `pedestrian` (0.19) | 0.12 |
| pedestrian | 0.76 | Confused with `cyclist` (0.19) | 0.23 |
| vehicle | 0.88 | Confused with `background` (0.12) | 0.12 |

---

## Key Takeaways

- **`vehicle`** is the best-detected class (TP = 0.88), but a significant share of background instances are incorrectly predicted as vehicle (0.59), indicating the model is somewhat trigger-happy on this dominant class.
- **`bus`** has a solid TP rate (0.81) but is frequently confused with `vehicle` (0.12), which is expected given their visual overlap. Its FN rate is low (0.06).
- **`construction`** scores 0.77 on the diagonal but loses 23% of instances to background (FN = 0.23), reflecting the visual heterogeneity of this class.
- **`cyclist`** and **`pedestrian`** show strong mutual confusion: 19% of cyclists are predicted as pedestrians and vice versa. This is the most prominent inter-class error in the model.
- **`pedestrian`** also suffers a high false negative rate (0.23), tied with `construction` as the most-missed class.