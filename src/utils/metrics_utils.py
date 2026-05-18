from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "per_class": {
            name: {"precision": float(per_precision[i]), "recall": float(per_recall[i]), "f1": float(per_f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(data, indent=2, default=default), encoding="utf-8")
