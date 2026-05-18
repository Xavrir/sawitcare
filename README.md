# SawitCare

SawitCare is the machine learning pipeline for early screening of oil palm trees in drone images and videos. It detects individual oil palm tree crowns, crops each detected tree, then classifies each crop as `healthy` or `suspicious`.

This repository contains only the ML part. It does not include the full application UI.

## Model architecture

- Main detector: YOLO11n
- Baseline detector: YOLOv8n
- Classifier: EfficientNet B0
- Detection class: `oil_palm_tree`
- Health classes: `healthy`, `suspicious`

The word `suspicious` is used intentionally. SawitCare is for early screening and field-prioritization, not final disease diagnosis.

## Repository layout

```text
sawitcare-ml/
  configs/                  # YOLO and classifier configs
  data/raw/                 # Original datasets
  data/processed/           # Split/clean training data
  models/                   # Saved trained weights
  outputs/                  # Annotated media, crops, CSVs, metrics
  src/data/                 # Dataset cleaning and splitting scripts
  src/training/             # Detector and classifier training
  src/inference/            # Image/video inference pipeline
  src/evaluation/           # Evaluation scripts
  src/utils/                # Shared utilities
```

## Installation

```bash
cd sawitcare-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset setup

One setup helper is included:

```bash
python src/data/setup_datasets.py
```

This downloads and extracts the public Mendeley classification dataset automatically. The Roboflow detection dataset requires an API key for export downloads; set `ROBOFLOW_API_KEY` before rerunning the helper if you want it to download the detection dataset too.

### Detection dataset

Use **Oil Palm Tree Crown Detection from Aerial Image** for palm crown detection.

Put raw data under:

```text
data/raw/detection/
  images/
  labels/
```

All palm-related labels are merged into one YOLO class:

```text
0 oil_palm_tree
```

Clean labels:

```bash
python src/data/clean_detection_labels.py \
  --labels-dir data/raw/detection/labels \
  --output-labels-dir data/processed/detection/clean_labels
```

Split detection data into YOLO format:

```bash
python src/data/split_detection_data.py \
  --raw-dir data/raw/detection \
  --output-dir data/processed/detection
```

If downloading from Roboflow, set your API key first:

```bash
export ROBOFLOW_API_KEY=your_key_here
python src/data/setup_datasets.py --skip-classification
python src/data/split_detection_data.py --raw-dir data/raw/detection --output-dir data/processed/detection
```

Expected processed detection layout:

```text
data/processed/detection/
  images/train images/val images/test
  labels/train labels/val labels/test
```

### Classification dataset

Use **Oil Palm Tree Detection for Anomaly Identification**.

Raw labels are mapped as:

- `PalmSan` -> `healthy`
- `PalmAnom` -> `suspicious`

Put raw images under folders or filenames containing `PalmSan` and `PalmAnom`, then run:

```bash
python src/data/split_classification_data.py \
  --raw-dir data/raw/classification \
  --output-dir data/processed/classification
```

The splitter also supports the Mendeley/Roboflow TensorFlow export format with `_annotations.csv`; it crops annotated palms and maps `PalmSan` to `healthy` and `PalmAnom` to `suspicious`.

Expected processed classification layout:

```text
data/processed/classification/
  train/healthy train/suspicious
  val/healthy val/suspicious
  test/healthy test/suspicious
```

## Train detectors

Train the main YOLO11n detector:

```bash
python src/training/train_detector_yolo.py --model yolo11n.pt --data configs/detection_data.yaml --imgsz 640 --epochs 100 --batch 8 --device 0
```

Train the YOLOv8n baseline:

```bash
python src/training/train_detector_yolo.py --model yolov8n.pt --data configs/detection_data.yaml --imgsz 640 --epochs 100 --batch 8 --device 0
```

Best weights are copied to `models/detector/`.

## Train classifier

Train EfficientNet B0:

```bash
python src/training/train_classifier.py --data data/processed/classification --epochs 50 --batch 32 --device cuda
```

The script uses ImageNet pretrained weights, trains the classifier head first, then fine-tunes the last layers. The best checkpoint by validation F1 is saved to:

```text
models/classifier/efficientnet_b0_best.pt
```

## Image inference

```bash
python src/inference/predict_image.py \
  --image samples/drone_001.jpg \
  --detector models/detector/yolo11n_best.pt \
  --classifier models/classifier/efficientnet_b0_best.pt
```

Outputs:

- Annotated image: `outputs/images/`
- Prediction CSV: `outputs/predictions/`

CSV columns:

```text
image_name, tree_id, x1, y1, x2, y2, detector_confidence, health_label, classifier_confidence
```

## Video inference

```bash
python src/inference/predict_video.py \
  --video samples/drone_video.mp4 \
  --detector models/detector/yolo11n_best.pt \
  --classifier models/classifier/efficientnet_b0_best.pt \
  --frame_step 15
```

Outputs:

- Annotated video: `outputs/videos/`
- Frame-level CSV summary: `outputs/predictions/`

CSV columns:

```text
frame_id, total_trees, healthy, suspicious, suspicious_ratio
```

## Evaluation

Detector test metrics:

```bash
python src/evaluation/eval_detector.py --model models/detector/yolo11n_best.pt --data configs/detection_data.yaml
```

Classifier test metrics:

```bash
python src/evaluation/eval_classifier.py --model models/classifier/efficientnet_b0_best.pt --data data/processed/classification
```

Pipeline sample evaluation:

```bash
python src/evaluation/eval_pipeline_samples.py \
  --samples samples \
  --detector models/detector/yolo11n_best.pt \
  --classifier models/classifier/efficientnet_b0_best.pt
```

Metrics are saved under `outputs/metrics/`.

## Expected outputs

- `outputs/images/`: annotated images with tree boxes and health labels
- `outputs/videos/`: annotated videos
- `outputs/crops/`: cropped tree images from sample pipeline runs
- `outputs/predictions/`: CSV prediction results
- `outputs/metrics/`: detector, classifier, and pipeline metrics

## Limitations

- The model does not diagnose exact disease.
- The `suspicious` label means the tree needs field inspection.
- Drone altitude, lighting, camera quality, season, and dataset quality can affect performance.
- The detector and classifier datasets may come from different sources, so final pipeline results need manual review.
- This first version is intended as a simple working baseline before broader app integration.
