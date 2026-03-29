from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm.auto import tqdm

from sparse_ops.logging import CsvLogger
from sparse_ops.net_utils import ensure_dir, resolve_device, save_checkpoint, seed_everything, write_json
from sparse_ops.schedulers import get_lr_scheduler
from wifi_dg.config import load_config, save_config
from wifi_dg.data import build_wifi_dataloaders
from wifi_dg.models import build_model
from wifi_dg.selection import is_better
from wifi_dg.sparsity import collect_sparsity_report
from wifi_dg.trainer import run_epoch
from wifi_dg.visualization import (
    save_confusion_matrix_artifacts,
    save_training_curves,
)


def _build_optimizer(model: torch.nn.Module, config: Dict[str, Any]):
    optimizer_name = str(config["optimizer"]).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=0.9,
            weight_decay=config["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def _seed_output_dir(config: Dict[str, Any], seed: int) -> Path:
    return Path(config["output_root"]) / config["dataset_name"] / config["method_name"] / f"seed{seed}"


def _flatten_log_row(
    epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], lr: float
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "lr": lr,
        "train_loss": float(train_metrics["loss"]),
        "train_acc": float(train_metrics["overall_acc"]),
        "train_penalty": float(train_metrics["penalty"]),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "train_balanced_acc": float(train_metrics["balanced_acc"]),
        "val_loss": float(val_metrics["loss"]),
        "val_acc": float(val_metrics["overall_acc"]),
        "val_balanced_acc": float(val_metrics["balanced_acc"]),
        "val_worst_source_env_acc": float(val_metrics.get("worst_source_env_acc", 0.0) or 0.0),
    }


def run_experiment(config_path: Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = load_config(config_path, overrides=overrides)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(config.get("device"))

    output_dir = _seed_output_dir(config, seed)
    ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    logs_dir = ensure_dir(output_dir / "logs")
    data_bundle = build_wifi_dataloaders(config)
    config["input_shape"] = [int(v) for v in data_bundle["input_shape"]]
    config["input_adapter"] = str(data_bundle["input_adapter"])
    if bool(config.get("auto_infer_num_classes", True)):
        config["num_classes"] = int(data_bundle["num_classes"])
    if bool(config.get("auto_infer_envs", True)):
        source_envs = [int(v) for v in data_bundle["source_envs"]]
        target_envs = [int(v) for v in data_bundle["target_envs"]]
        config["num_source_envs"] = int(len(source_envs))
        if target_envs:
            config["target_env"] = int(target_envs[0])
        config["source_envs"] = source_envs
        config["target_envs"] = target_envs
    save_config(config, output_dir / "config.yaml")
    dataset_info = {
        "split_sizes": data_bundle["split_sizes"],
        "input_shape": data_bundle["input_shape"],
        "model_family": str(config.get("model_family", "wifi_csi_default")),
        "normalization": data_bundle["normalization"],
        "train_subset": data_bundle.get("train_subset", {}),
        "num_classes": data_bundle["num_classes"],
        "source_envs": data_bundle["source_envs"],
        "target_envs": data_bundle["target_envs"],
        "nonfinite_report": data_bundle["nonfinite_report"],
    }

    model = build_model(config).to(device)
    optimizer = _build_optimizer(model, config)
    lr_scheduler = get_lr_scheduler(optimizer, config)
    amp_enabled = bool(config.get("amp_enabled", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    csv_logger = CsvLogger(logs_dir / "train_log.csv")

    best_val = None
    best_epoch = -1
    best_val_acc_value = float("-inf")
    best_val_acc_epoch = -1
    best_val_acc_train_metrics: Optional[Dict[str, float]] = None
    best_val_acc_val_metrics: Optional[Dict[str, float]] = None
    checkpoint_path = checkpoints_dir / "checkpoint_best.pth"
    early_stopping_enabled = bool(config.get("early_stopping_enabled", True))
    early_stopping_patience = int(config.get("early_stopping_patience", 15))
    early_stopping_min_epochs = int(config.get("early_stopping_min_epochs", 10))
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = -1
    show_progress = bool(config.get("show_progress", True))
    leave_progress_bars = bool(config.get("leave_progress_bars", False))
    progress_refresh_rate = int(config.get("progress_refresh_rate", 1))
    history_rows: List[Dict[str, Any]] = []

    epoch_bar = tqdm(
        range(int(config["epochs"])),
        desc="epochs",
        leave=leave_progress_bars,
        disable=not show_progress,
        miniters=progress_refresh_rate,
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        lr_scale = lr_scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = config["lr"] * lr_scale

        train_metrics = run_epoch(
            model,
            data_bundle["loaders"]["train"],
            device=device,
            config=config,
            optimizer=optimizer,
            epoch=epoch,
            split_name="train",
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model,
            data_bundle["loaders"]["val"],
            device=device,
            config=config,
            optimizer=None,
            epoch=epoch,
            split_name="val",
            scaler=None,
        )
        is_best = is_better(val_metrics, best_val)
        log_row = _flatten_log_row(epoch, train_metrics, val_metrics, optimizer.param_groups[0]["lr"])
        log_row["is_best"] = bool(is_best)
        history_rows.append(log_row)
        csv_logger.write_row(log_row)
        current_val_acc = float(val_metrics["overall_acc"])
        if current_val_acc > best_val_acc_value:
            best_val_acc_value = current_val_acc
            best_val_acc_epoch = epoch
            best_val_acc_train_metrics = {
                "train_acc": float(train_metrics["overall_acc"]),
                "train_loss": float(train_metrics["loss"]),
                "train_balanced_acc": float(train_metrics["balanced_acc"]),
            }
            best_val_acc_val_metrics = {
                "val_acc": current_val_acc,
                "val_loss": float(val_metrics["loss"]),
                "val_balanced_acc": float(val_metrics["balanced_acc"]),
            }

        if is_best:
            best_val = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        epoch_bar.set_postfix(
            train_loss=f"{float(train_metrics['loss']):.4f}",
            train_acc=f"{float(train_metrics['overall_acc']):.4f}",
            val_loss=f"{float(val_metrics['loss']):.4f}",
            val_acc=f"{float(val_metrics['overall_acc']):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.3e}",
            best_epoch=best_epoch,
        )
        tqdm.write(
            "Epoch {}/{} | train_loss={:.4f} train_acc={:.4f} | val_loss={:.4f} val_acc={:.4f}{}".format(
                epoch + 1,
                int(config["epochs"]),
                float(train_metrics["loss"]),
                float(train_metrics["overall_acc"]),
                float(val_metrics["loss"]),
                float(val_metrics["overall_acc"]),
                " | new_best" if is_best else "",
            )
        )
        if (
            early_stopping_enabled
            and (epoch + 1) >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            stopped_epoch = epoch
            tqdm.write(
                "Early stopping triggered at epoch {}/{} (patience={}, min_epochs={}).".format(
                    epoch + 1,
                    int(config["epochs"]),
                    early_stopping_patience,
                    early_stopping_min_epochs,
                )
            )
            break

    epoch_bar.close()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    best_val = run_epoch(
        model,
        data_bundle["loaders"]["val"],
        device=device,
        config=config,
        optimizer=None,
        epoch=best_epoch,
        split_name="val",
        scaler=None,
    )
    test_metrics = run_epoch(
        model,
        data_bundle["loaders"]["test"],
        device=device,
        config=config,
        optimizer=None,
        epoch=best_epoch,
        split_name="test",
        scaler=None,
    )
    sparsity = collect_sparsity_report(model, tuple(data_bundle["input_shape"]), device)
    if best_val_acc_train_metrics is not None and best_val_acc_val_metrics is not None:
        best_val_record = {
            "epoch": int(best_val_acc_epoch + 1),
            **best_val_acc_val_metrics,
            **best_val_acc_train_metrics,
        }
    else:
        best_val_record = {}

    save_training_curves(history_rows, plots_dir)
    save_confusion_matrix_artifacts(
        test_metrics["confusion_matrix"],
        plots_dir,
        prefix="confusion_matrix_test",
    )

    final_summary = {
        "best_epoch": best_epoch + 1,
        "best_val_acc": float(best_val["overall_acc"]),
        "best_val_loss": float(best_val["loss"]),
        "best_val_acc_epoch": int(best_val_acc_epoch + 1) if best_val_acc_epoch >= 0 else None,
        "train_acc_at_best_val_acc": (
            float(best_val_acc_train_metrics["train_acc"]) if best_val_acc_train_metrics is not None else None
        ),
        "test_acc": float(test_metrics["overall_acc"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_balanced_acc": float(test_metrics["balanced_acc"]),
        "worst_env_acc": float(test_metrics.get("worst_env_acc", 0.0) or 0.0),
        "early_stopping_enabled": early_stopping_enabled,
        "early_stopped": bool(stopped_early),
        "stopped_epoch": int(stopped_epoch + 1) if stopped_epoch >= 0 else None,
        "run_dir": str(output_dir),
    }
    run_metrics: Dict[str, Any] = {
        "dataset_info": dataset_info,
        "best_val_record": best_val_record,
        "val_best_metrics": best_val,
        "test_metrics": test_metrics,
        "final_summary": final_summary,
    }
    if str(config.get("backbone_type", "")).lower() == "sparse":
        run_metrics["sparsity"] = sparsity
    write_json(run_metrics, metrics_dir / "run_metrics.json")
    final_payload = {
        "dataset": str(config["dataset_name"]),
        "model": str(config["model_name"]),
        "method": str(config["method_name"]),
        "best_epoch": best_epoch + 1,
        "best_val_acc": float(best_val["overall_acc"]),
        "final_test_acc": float(test_metrics["overall_acc"]),
        "run_dir": str(output_dir),
    }
    tqdm.write(
        "Final Test (best-val checkpoint) | epoch={:03d}/{} | loss={:.4f}, acc={:.4f}, worst_env={:.4f} {}".format(
            best_epoch + 1,
            int(config["epochs"]),
            float(test_metrics["loss"]),
            float(test_metrics["overall_acc"]),
            float(test_metrics.get("worst_env_acc", 0.0) or 0.0),
            json.dumps(final_payload, ensure_ascii=False),
        )
    )

    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "method_name": config["method_name"],
        "dataset_name": config["dataset_name"],
        "val_worst_source_env_acc": float(best_val.get("worst_source_env_acc", 0.0) or 0.0),
        "val_overall_acc": float(best_val["overall_acc"]),
        "best_val_acc_epoch": int(best_val_acc_epoch + 1) if best_val_acc_epoch >= 0 else 0,
        "train_acc_at_best_val_acc": (
            float(best_val_acc_train_metrics["train_acc"]) if best_val_acc_train_metrics is not None else 0.0
        ),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_balanced_acc": float(test_metrics["balanced_acc"]),
        "test_overall_acc": float(test_metrics["overall_acc"]),
        "global_sparsity_rate": float(sparsity["global_sparsity_rate"]),
        "effective_params": float(sparsity["effective_params"]),
        "approx_effective_flops": float(sparsity["approx_effective_flops"]),
        "output_dir": str(output_dir),
    }
    return summary


def write_method_summary(method_dir: Path, records: List[Dict[str, Any]]) -> None:
    ensure_dir(method_dir)
    json_path = method_dir / "summary.json"
    csv_path = method_dir / "summary.csv"
    write_json({"runs": records}, json_path)
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
