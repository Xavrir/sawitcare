from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


WORK_DIR = Path("/kaggle/working/sawitcare")
KAGGLE_INPUT_DETECTION_DIR = Path("/kaggle/input/sawitcare-detection-roboflow-v19")
KAGGLE_INPUT_DETECTION_ZIP = Path("/kaggle/input/sawitcare-detection-roboflow-v19/detection_roboflow_yolov8.zip")
RAW_DIR = WORK_DIR / "data/raw/detection"
PROCESSED_DIR = WORK_DIR / "data/processed/detection"
MODELS_DIR = WORK_DIR / "models/detector"
METRICS_DIR = WORK_DIR / "outputs/metrics"
DATA_YAML = WORK_DIR / "configs/detection_data.yaml"
ROBOFLOW_PROJECT_API = "https://api.roboflow.com/mahir-sehmi-fgblg/oil-palm-tree-crown-detection-from-aerial-image"
ROBOFLOW_VERSION = int(os.environ.get("ROBOFLOW_VERSION", "19"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH = int(os.environ.get("BATCH", "8"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
SEED = int(os.environ.get("SEED", "42"))
CANDIDATES = (
    {"name": "yolov8n", "source": "yolov8n.pt", "train": True},
    {"name": "yolo11n", "source": "yolo11n_best.pt", "train": False},
    {"name": "yolo26", "source": "yolo26n.pt", "train": True},
)


def get_roboflow_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
    except Exception as exc:
        raise RuntimeError(
            "Add a Kaggle secret named ROBOFLOW_API_KEY, or set the ROBOFLOW_API_KEY environment variable."
        ) from exc


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"Already downloaded: {output}")
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_detection_dataset(api_key: str) -> None:
    export_url = f"{ROBOFLOW_PROJECT_API}/{ROBOFLOW_VERSION}/yolov8?api_key={api_key}"
    request = urllib.request.Request(export_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        export = json.load(response)
    link = export.get("export", {}).get("link")
    if not link:
        raise RuntimeError("Roboflow did not return a YOLOv8 export link.")

    zip_path = WORK_DIR / "data/raw/detection_roboflow_yolov8.zip"
    download(link, zip_path)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW_DIR)
    print(f"Extracted detection dataset to {RAW_DIR}")


def find_attached_detection_source() -> Path | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    mounted = sorted(path for path in input_root.iterdir() if path.is_dir())
    print("Mounted Kaggle input directories:", [str(path) for path in mounted], flush=True)
    data_yamls = sorted(input_root.rglob("data.yaml"))
    if data_yamls:
        print("Found data.yaml candidates:", [str(path) for path in data_yamls[:10]], flush=True)
        return data_yamls[0].parent
    zip_files = sorted(input_root.rglob("detection_roboflow_yolov8.zip"))
    if zip_files:
        print("Found detection zip candidates:", [str(path) for path in zip_files[:10]], flush=True)
        return zip_files[0]
    for path in mounted:
        if (path / "data.yaml").exists():
            return path
        zip_path = path / "detection_roboflow_yolov8.zip"
        if zip_path.exists():
            return zip_path
    return None


def find_offline_assets_dir() -> Path | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    for wheel in sorted(input_root.rglob("ultralytics-*.whl")):
        return wheel.parent
    return None


def install_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401

        print("Ultralytics already available.", flush=True)
        return
    except ImportError:
        pass
    assets_dir = find_offline_assets_dir()
    if assets_dir is None:
        print("No offline Ultralytics wheel found; trying PyPI.", flush=True)
        subprocess.run(["pip", "install", "-q", "-U", "ultralytics"], check=True)
        return
    wheels = [str(path) for path in sorted(assets_dir.glob("*.whl"))]
    print(f"Installing offline wheels from {assets_dir}: {[Path(w).name for w in wheels]}", flush=True)
    subprocess.run(["pip", "install", "--no-index", "--no-deps", *wheels], check=True)


def resolve_model_source(source: str) -> str:
    local_source = Path(source)
    if local_source.exists():
        return str(local_source)
    input_root = Path("/kaggle/input")
    for candidate in sorted(input_root.rglob(source)):
        return str(candidate)
    return source


def require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not provide a CUDA GPU. Stop this run and enable a GPU accelerator.")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}", flush=True)


def setup_detection_dataset() -> Path:
    attached = find_attached_detection_source()
    if attached and attached.is_dir():
        print(f"Using attached Kaggle dataset directory: {attached}", flush=True)
        return attached
    if attached and attached.suffix == ".zip":
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(attached) as archive:
            archive.extractall(RAW_DIR)
        print(f"Extracted attached Kaggle dataset zip to {RAW_DIR}", flush=True)
        return RAW_DIR
    if (KAGGLE_INPUT_DETECTION_DIR / "data.yaml").exists():
        print(f"Using attached Kaggle dataset directory: {KAGGLE_INPUT_DETECTION_DIR}")
        return KAGGLE_INPUT_DETECTION_DIR
    if KAGGLE_INPUT_DETECTION_ZIP.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(KAGGLE_INPUT_DETECTION_ZIP) as archive:
            archive.extractall(RAW_DIR)
        print(f"Extracted attached Kaggle dataset to {RAW_DIR}")
        return RAW_DIR
    download_detection_dataset(get_roboflow_key())
    return RAW_DIR


def find_images(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def label_for(image: Path, raw_dir: Path) -> Path | None:
    candidates = [raw_dir / "labels" / f"{image.stem}.txt", image.with_suffix(".txt")]
    if image.parent.name == "images":
        candidates.append(image.parent.parent / "labels" / f"{image.stem}.txt")
    if "images" in image.parts:
        idx = len(image.parts) - 1 - list(reversed(image.parts)).index("images")
        rel_after_images = Path(*image.parts[idx + 1 :]).with_suffix(".txt")
        candidates.append(raw_dir / "labels" / rel_after_images)
    return next((path for path in candidates if path.exists()), None)


def collect_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    roots = [raw_dir / "images"] if (raw_dir / "images").exists() else [raw_dir]
    pairs: list[tuple[Path, Path]] = []
    for root in roots:
        for image in find_images(root):
            if "labels" in image.parts:
                continue
            label = label_for(image, raw_dir)
            if label:
                pairs.append((image, label))
    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {raw_dir}")
    return pairs


def clean_label_text(label: Path) -> str:
    rows: list[str] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            [float(value) for value in parts[1:]]
        except ValueError:
            continue
        rows.append(" ".join(["0", *parts[1:]]))
    return "\n".join(rows) + ("\n" if rows else "")


def prepare_detection_split(raw_dir: Path) -> None:
    pairs = collect_pairs(raw_dir)
    random.Random(SEED).shuffle(pairs)
    n = len(pairs)
    splits = {
        "train": pairs[: int(n * 0.70)],
        "val": pairs[int(n * 0.70) : int(n * 0.85)],
        "test": pairs[int(n * 0.85) :],
    }
    for split, split_pairs in splits.items():
        image_dir = PROCESSED_DIR / "images" / split
        label_dir = PROCESSED_DIR / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, label in split_pairs:
            shutil.copy2(image, image_dir / image.name)
            (label_dir / f"{image.stem}.txt").write_text(clean_label_text(label), encoding="utf-8")
        print(f"{split}: {len(split_pairs)} pairs")


def write_data_yaml() -> None:
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(
        "\n".join(
            [
                f"path: {PROCESSED_DIR}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: oil_palm_tree",
                "",
            ]
        ),
        encoding="utf-8",
    )


def train_and_eval() -> list[dict[str, object]]:
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        name = str(candidate["name"])
        source = str(candidate["source"])
        should_train = bool(candidate["train"])
        resolved_source = resolve_model_source(source)
        source_stem = Path(source).stem
        final_name = f"{source_stem}.pt" if source_stem.endswith("_best") else f"{source_stem}_best.pt"
        final_path = MODELS_DIR / final_name
        if should_train:
            print(f"\n=== Training {name} from {resolved_source} ===", flush=True)
            model = YOLO(resolved_source)
            model.train(
                data=str(DATA_YAML),
                imgsz=IMGSZ,
                epochs=EPOCHS,
                batch=BATCH,
                device=0,
                project=str(METRICS_DIR / "detector_runs"),
                name=name,
                exist_ok=True,
            )
            best_path = Path(model.trainer.best)
            shutil.copy2(best_path, final_path)
            mode = "trained_on_kaggle"
        else:
            print(f"\n=== Evaluating existing {name} checkpoint: {resolved_source} ===", flush=True)
            resolved_path = Path(resolved_source)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Existing checkpoint not found for {name}: {source}")
            shutil.copy2(resolved_path, final_path)
            mode = "existing_checkpoint"

        eval_model = YOLO(str(final_path))
        results = eval_model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=IMGSZ,
            device=0,
            project=str(METRICS_DIR / "detector_comparison_runs"),
            name=name,
            exist_ok=True,
            plots=False,
        )
        box = results.box
        speed = getattr(results, "speed", {}) or {}
        rows.append(
            {
                "name": name,
                "mode": mode,
                "checkpoint": str(final_path),
                "precision": float(box.mp),
                "recall": float(box.mr),
                "mAP50": float(box.map50),
                "mAP50_95": float(box.map),
                "speed_inference_ms": speed.get("inference"),
                "weight_mb": round(final_path.stat().st_size / (1024 * 1024), 2),
            }
        )
    return rows


def save_outputs(rows: list[dict[str, object]]) -> None:
    json_path = METRICS_DIR / "detector_comparison.json"
    csv_path = METRICS_DIR / "detector_comparison.csv"
    md_path = METRICS_DIR / "detector_comparison.md"
    json_path.write_text(json.dumps({"models": rows}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SawitCare YOLO Nano Detector Comparison",
        "",
        "| Model | Mode | Precision | Recall | mAP50 | mAP50-95 | Infer ms/img | Weight MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['mode']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['mAP50']:.4f} | {row['mAP50_95']:.4f} | "
            f"{row['speed_inference_ms']:.3f} | {row['weight_mb']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = setup_detection_dataset()
    prepare_detection_split(raw_dir)
    write_data_yaml()

    print("\nGPU check:")
    os.system("nvidia-smi")
    print("\nInstalling/checking Ultralytics:")
    install_ultralytics()
    require_gpu()

    rows = train_and_eval()
    save_outputs(rows)
    print(f"Saved weights and metrics under: {WORK_DIR}")


if __name__ == "__main__":
    main()
