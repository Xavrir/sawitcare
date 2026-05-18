from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

CLASSIFICATION_URL = "https://data.mendeley.com/public-api/zip/nh7d23dgnw/download/1"
DETECTION_ROBOFLOW_URL = "https://universe.roboflow.com/mahir-sehmi-fgblg/oil-palm-tree-crown-detection-from-aerial-image/dataset/1/download/yolov8?key={api_key}"
DETECTION_PROJECT_API = "https://api.roboflow.com/mahir-sehmi-fgblg/oil-palm-tree-crown-detection-from-aerial-image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/extract SawitCare dataset sources where public access is available.")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--roboflow-api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    parser.add_argument("--roboflow-version", type=int, default=None, help="Roboflow dataset version. Defaults to latest.")
    return parser.parse_args()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"Already downloaded: {output}")
        return
    safe_url = url.split("?key=")[0] + ("?key=***" if "?key=" in url else "")
    safe_url = safe_url.split("?api_key=")[0] + ("?api_key=***" if "?api_key=" in url else "")
    print(f"Downloading {safe_url} -> {output}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response, output.open("wb") as file:
        shutil.copyfileobj(response, file)


def extract(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    print(f"Extracted {zip_path} -> {output_dir}")


def setup_classification() -> None:
    zip_path = Path("data/raw/classification_mendeley.zip")
    raw_dir = Path("data/raw/classification")
    download(CLASSIFICATION_URL, zip_path)
    extract(zip_path, raw_dir)
    for nested in raw_dir.rglob("*.zip"):
        extract_dir = nested.parent / nested.stem
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            extract(nested, extract_dir)
    print("Now run: python src/data/split_classification_data.py --raw-dir data/raw/classification --output-dir data/processed/classification")


def setup_detection(api_key: str | None) -> None:
    if not api_key:
        readme = Path("data/raw/detection/README.md")
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(
            "# Detection dataset\n\n"
            "Dataset: Oil Palm Tree Crown Detection from Aerial Image\n"
            "Source: https://universe.roboflow.com/mahir-sehmi-fgblg/oil-palm-tree-crown-detection-from-aerial-image\n\n"
            "Roboflow requires an API key for YOLO export downloads. Set it and rerun:\n\n"
            "```bash\n"
            "export ROBOFLOW_API_KEY=your_key_here\n"
            "python src/data/setup_datasets.py --skip-classification\n"
            "python src/data/split_detection_data.py --raw-dir data/raw/detection --output-dir data/processed/detection\n"
            "```\n",
            encoding="utf-8",
        )
        print(f"Detection dataset needs ROBOFLOW_API_KEY. Wrote instructions: {readme}")
        return
    zip_path = Path("data/raw/detection_roboflow_yolov8.zip")
    raw_dir = Path("data/raw/detection")
    download(DETECTION_ROBOFLOW_URL.format(api_key=api_key), zip_path)
    extract(zip_path, raw_dir)


def setup_detection_from_api(api_key: str | None, version: int | None = None) -> None:
    if not api_key:
        setup_detection(api_key)
        return
    project_url = f"{DETECTION_PROJECT_API}?api_key={api_key}"
    request = urllib.request.Request(project_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        project = json.load(response)
    version = version or int(project["project"]["versions"])
    export_url = f"{DETECTION_PROJECT_API}/{version}/yolov8?api_key={api_key}"
    request = urllib.request.Request(export_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        export = json.load(response)
    link = export.get("export", {}).get("link")
    if not link:
        raise RuntimeError("Roboflow did not return a YOLOv8 export link.")
    zip_path = Path("data/raw/detection_roboflow_yolov8.zip")
    raw_dir = Path("data/raw/detection")
    download(link, zip_path)
    extract(zip_path, raw_dir)


def main() -> None:
    args = parse_args()
    if not args.skip_classification:
        setup_classification()
    if not args.skip_detection:
        setup_detection_from_api(args.roboflow_api_key, args.roboflow_version)


if __name__ == "__main__":
    main()
