from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("outputs/metrics/detector_comparison.json")
DEFAULT_OUTPUT_PNG = Path("outputs/metrics/detector_comparison_charts.png")
DEFAULT_OUTPUT_SVG = Path("outputs/metrics/detector_comparison_charts.svg")

MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolo11n": "YOLO11n",
    "yolo26": "YOLO26n",
}


def load_models(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    models = payload.get("models", [])
    if not models:
        raise ValueError(f"No models found in {path}")
    return models


def annotate_bars(ax: plt.Axes, bars: Any, *, precision: int = 3) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{precision}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f2937",
        )


def plot_detector_comparison(
    models: list[dict[str, Any]], outputs: list[Path], *, chosen: str = "yolo11n"
) -> None:
    labels = [MODEL_LABELS.get(model["name"], model["name"]) for model in models]
    chosen_idx = next(
        (i for i, model in enumerate(models) if model["name"] == chosen), None
    )

    accuracy_metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("mAP50_95", "mAP50-95"),
    ]
    metric_colors = ["#2563eb", "#16a34a", "#ea580c"]
    x = np.arange(len(labels))
    width = 0.26

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), dpi=180)
    fig.patch.set_facecolor("#f8fafc")

    # --- Left: accuracy metrics ---
    ax = axes[0]
    ax.set_facecolor("#ffffff")
    for offset, (key, metric_label), color in zip(
        [-width, 0, width], accuracy_metrics, metric_colors, strict=True
    ):
        values = [float(model[key]) for model in models]
        bars = ax.bar(
            x + offset, values, width, label=metric_label, color=color, alpha=0.92
        )
        annotate_bars(ax, bars)

    ax.set_title("Detection Accuracy (higher is better)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.78, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
    # Legend lifted above the axes so it never overlaps the bars.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Right: runtime and size ---
    ax = axes[1]
    ax.set_facecolor("#ffffff")
    speed_values = [float(model["speed_inference_ms"]) for model in models]
    weight_values = [float(model["weight_mb"]) for model in models]

    bars_speed = ax.bar(
        x - width / 2, speed_values, width, color="#2563eb", label="Infer ms/img"
    )
    bars_weight = ax.bar(
        x + width / 2, weight_values, width, color="#ea580c", label="Weight MB"
    )
    annotate_bars(ax, bars_speed)
    annotate_bars(ax, bars_weight, precision=2)

    ax.set_title("Runtime and Model Size (lower is better)")
    ax.set_ylabel("Value")
    ax.set_ylim(0, max(max(speed_values), max(weight_values)) + 1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Highlight the chosen model on both panels ---
    if chosen_idx is not None:
        chosen_label = labels[chosen_idx]
        for axis in axes:
            axis.axvspan(
                chosen_idx - 0.5,
                chosen_idx + 0.5,
                color="#fde68a",
                alpha=0.35,
                zorder=0,
            )
            for tick in axis.get_xticklabels():
                if tick.get_text() == chosen_label:
                    tick.set_fontweight("bold")
                    tick.set_color("#b45309")

    best_map = max(models, key=lambda model: float(model["mAP50_95"]))
    chosen_name = MODEL_LABELS.get(chosen, chosen)
    subtitle = (
        f"Selected model: {chosen_name}  -  best mAP50-95 "
        f"({best_map['mAP50_95']:.3f}) and highest recall, at nano size and speed"
    )
    fig.suptitle(
        "YOLO Nano Comparison for SawitCare",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
        y=0.99,
    )
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=11, color="#334155")
    fig.text(
        0.5,
        0.02,
        "Source: Kaggle run rizkymirzaviandy/sawitcare-yolo-nano-comparison-dataset, completed 2026-06-02.",
        ha="center",
        fontsize=8.5,
        color="#64748b",
    )

    fig.tight_layout(rect=(0.025, 0.05, 0.975, 0.88))

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create detector comparison charts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Detector comparison JSON.")
    parser.add_argument("--png", type=Path, default=DEFAULT_OUTPUT_PNG, help="Output PNG path.")
    parser.add_argument("--svg", type=Path, default=DEFAULT_OUTPUT_SVG, help="Output SVG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_models(args.input)
    plot_detector_comparison(models, [args.png, args.svg])
    print(f"Wrote {args.png}")
    print(f"Wrote {args.svg}")


if __name__ == "__main__":
    main()
