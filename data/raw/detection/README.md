# Detection dataset

Dataset: Oil Palm Tree Crown Detection from Aerial Image
Source: https://universe.roboflow.com/mahir-sehmi-fgblg/oil-palm-tree-crown-detection-from-aerial-image

Roboflow requires an API key for YOLO export downloads. Set it and rerun:

```bash
export ROBOFLOW_API_KEY=your_key_here
python src/data/setup_datasets.py --skip-classification
python src/data/split_detection_data.py --raw-dir data/raw/detection --output-dir data/processed/detection
```
