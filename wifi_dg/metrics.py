from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    envs: Optional[np.ndarray] = None,
    num_classes: int = 8,
) -> Tuple[Dict[str, Any], np.ndarray]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(num_classes))
    class_support = matrix.sum(axis=1)
    class_recall = np.divide(
        np.diag(matrix),
        class_support,
        out=np.zeros_like(class_support, dtype=np.float64),
        where=class_support > 0,
    )
    metrics: Dict[str, Any] = {
        "overall_acc": float((labels == predictions).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_acc": float(class_recall[class_support > 0].mean()) if np.any(class_support > 0) else 0.0,
        "per_class_f1": {
            str(class_idx): float(score)
            for class_idx, score in enumerate(
                f1_score(labels, predictions, average=None, labels=np.arange(num_classes), zero_division=0)
            )
        },
    }
    if envs is not None:
        envs = np.asarray(envs)
        per_env = {}
        for env_id in sorted(int(env) for env in np.unique(envs)):
            env_mask = envs == env_id
            per_env[str(env_id)] = float((labels[env_mask] == predictions[env_mask]).mean())
        metrics["per_env_acc"] = per_env
        metrics["worst_env_acc"] = float(min(per_env.values())) if per_env else None
        metrics["worst_source_env_acc"] = float(min(per_env.values())) if per_env else None
    return metrics, matrix


def summarize_numeric_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if not records:
        return summary
    numeric_keys = [
        key
        for key, value in records[0].items()
        if isinstance(value, (int, float))
    ]
    for key in numeric_keys:
        values = np.array([float(record[key]) for record in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary
