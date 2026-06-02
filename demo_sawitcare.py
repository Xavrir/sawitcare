from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Default video yang dipakai untuk demo presentasi.
DEFAULT_VIDEO = Path("/home/xavrir/Downloads/Flying over an Indonesian oil palm plantation.mp4")

# Model yang sudah dilatih.
DETECTOR_MODEL = PROJECT_ROOT / "models/detector/yolo11n_best.pt"
CLASSIFIER_MODEL = PROJECT_ROOT / "models/classifier/efficientnet_b0_best.pt"

# Pengaturan demo video.
# Tujuannya: output terlihat bersih, stabil, dan mudah dijelaskan saat presentasi.
DETECTION_CONFIDENCE = 0.8
CLASSIFIER_CONFIDENCE = 0.65
TILE_SIZE = 640
TILE_OVERLAP = 0.2
NMS_IOU = 0.5
FRAME_STEP = 1
MAX_FRAMES = 600  # sekitar 10 detik pada video 60 FPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SawitCare presentation demo.")
    parser.add_argument("--video", type=Path, default=None, help="Path ke video drone kelapa sawit.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame awal untuk demo.")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Jumlah frame demo yang diproses.")
    parser.add_argument("--run", action="store_true", help="Langsung jalankan demo tanpa menu interaktif.")
    return parser.parse_args()


def print_header() -> None:
    print("\n" + "=" * 62)
    print("SawitCare Demo Menu")
    print("Deteksi pohon kelapa sawit + screening kesehatan visual")
    print("=" * 62)


def print_settings(video_path: Path, start_frame: int, max_frames: int) -> None:
    print("\nPengaturan demo saat ini:")
    print(f"Video input          : {video_path}")
    print(f"Detector             : {DETECTOR_MODEL.name}")
    print(f"Classifier           : {CLASSIFIER_MODEL.name}")
    print(f"Detection confidence : {DETECTION_CONFIDENCE}")
    print(f"Classifier threshold : {CLASSIFIER_CONFIDENCE}")
    print(f"Tile size            : {TILE_SIZE}")
    print(f"Tile overlap         : {TILE_OVERLAP}")
    print(f"NMS IoU              : {NMS_IOU}")
    print(f"Frame step           : {FRAME_STEP}")
    print(f"Start frame          : {start_frame}")
    print(f"Max frames           : {max_frames}")


def print_output_paths(video_path: Path) -> None:
    print("\nOutput yang dihasilkan:")
    print(PROJECT_ROOT / "outputs/videos" / f"{video_path.stem}_annotated.mp4")
    print(PROJECT_ROOT / "outputs/predictions" / f"{video_path.stem}_frame_summary.csv")
    print(PROJECT_ROOT / "outputs/predictions" / f"{video_path.stem}_tree_predictions.csv")


def ask_video_path(current_video: Path) -> Path:
    print(f"\nVideo saat ini: {current_video}")
    user_input = input("Masukkan path video baru, atau tekan Enter untuk tetap memakai video ini: ").strip()
    if not user_input:
        return current_video
    return Path(user_input).expanduser()


def ask_int(label: str, current_value: int) -> int:
    user_input = input(f"{label} [{current_value}]: ").strip()
    if not user_input:
        return current_value
    return int(user_input)


def run_demo(video_path: Path, start_frame: int, max_frames: int) -> None:
    """Menjalankan pipeline SawitCare untuk demo presentasi."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")
    if not DETECTOR_MODEL.exists():
        raise FileNotFoundError(f"Model detector tidak ditemukan: {DETECTOR_MODEL}")
    if not CLASSIFIER_MODEL.exists():
        raise FileNotFoundError(f"Model classifier tidak ditemukan: {CLASSIFIER_MODEL}")

    command = [
        sys.executable,
        "src/inference/predict_video.py",
        "--video",
        str(video_path),
        "--detector",
        str(DETECTOR_MODEL),
        "--classifier",
        str(CLASSIFIER_MODEL),
        "--conf",
        str(DETECTION_CONFIDENCE),
        "--frame_step",
        str(FRAME_STEP),
        "--start-frame",
        str(start_frame),
        "--max-frames",
        str(max_frames),
        "--tile-size",
        str(TILE_SIZE),
        "--tile-overlap",
        str(TILE_OVERLAP),
        "--nms-iou",
        str(NMS_IOU),
        "--classifier-conf",
        str(CLASSIFIER_CONFIDENCE),
        "--short-labels",
        "--hide-classifier-conf",
        "--hide-detector-conf",
        "--summary-box",
        "--display-mode",
        "all",
    ]

    print("\nMenjalankan demo SawitCare...")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("\nDemo selesai.")
    print_output_paths(video_path)


def run_menu(video_path: Path, start_frame: int, max_frames: int) -> None:
    while True:
        print_header()
        print("1. Lihat pengaturan demo")
        print("2. Ganti video / durasi demo")
        print("3. Jalankan demo")
        print("4. Lihat lokasi output")
        print("5. Keluar")

        choice = input("\nPilih menu: ").strip()

        if choice == "1":
            print_settings(video_path, start_frame, max_frames)
        elif choice == "2":
            video_path = ask_video_path(video_path)
            start_frame = ask_int("Start frame", start_frame)
            max_frames = ask_int("Max frames", max_frames)
        elif choice == "3":
            run_demo(video_path, start_frame, max_frames)
        elif choice == "4":
            print_output_paths(video_path)
        elif choice == "5":
            print("Keluar dari demo SawitCare.")
            break
        else:
            print("Pilihan tidak valid.")

        input("\nTekan Enter untuk kembali ke menu...")


def main() -> None:
    args = parse_args()
    video_path = args.video or DEFAULT_VIDEO
    if args.run:
        run_demo(video_path, args.start_frame, args.max_frames)
    else:
        run_menu(video_path, args.start_frame, args.max_frames)


if __name__ == "__main__":
    main()
