from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sparse_ops.net_utils import ensure_dir


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Visualization requires matplotlib. Please install it with: pip install matplotlib") from exc
    return plt


def save_training_curves(history_rows: List[Dict[str, Any]], plot_dir: Path) -> Dict[str, str]:
    plt = _load_matplotlib()
    ensure_dir(plot_dir)

    epochs = [int(row["epoch"]) + 1 for row in history_rows]
    train_loss = [float(row["train_loss"]) for row in history_rows]
    val_loss = [float(row["val_loss"]) for row in history_rows]
    train_acc = [float(row["train_acc"]) for row in history_rows]
    val_acc = [float(row["val_acc"]) for row in history_rows]

    loss_png = plot_dir / "curves_loss.png"
    acc_png = plot_dir / "curves_acc.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", label="Train Loss")
    ax.plot(epochs, val_loss, marker="o", label="Val Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, marker="o", label="Train Acc")
    ax.plot(epochs, val_acc, marker="o", label="Val Acc")
    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(acc_png, dpi=180)
    plt.close(fig)

    return {
        "curves_loss_png": str(loss_png),
        "curves_acc_png": str(acc_png),
    }


def save_confusion_matrix_artifacts(
    matrix: np.ndarray,
    plot_dir: Path,
    prefix: str = "confusion_matrix_test",
) -> Dict[str, str]:
    plt = _load_matplotlib()
    ensure_dir(plot_dir)

    matrix = np.asarray(matrix, dtype=np.float64)
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )

    output_path = plot_dir / f"{prefix}_normalized.png"
    labels = [str(i) for i in range(matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Normalized Test Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for row_idx in range(normalized.shape[0]):
        for col_idx in range(normalized.shape[1]):
            value = normalized[row_idx, col_idx]
            color = "white" if value >= 0.5 else "black"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"confusion_matrix_normalized_png": str(output_path)}
