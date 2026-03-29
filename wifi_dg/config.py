from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset_name": "wifi5300_har",
    "dataset_root": "wifi_data/5300",
    "output_root": "outputs",
    "model_name": "wifi_dg_backbone",
    "model_family": "wifi_csi_default",
    "method_name": None,
    "input_adapter": "auto",
    "normalization_mode": "train_channelwise",
    "auto_infer_num_classes": True,
    "auto_infer_envs": True,
    "nonfinite_policy": "none",
    "backbone_type": "sparse",
    "penalty_type": "irm",
    "num_classes": 7,
    "num_source_envs": 4,
    "target_env": 4,
    "batch_size": 64,
    "train_subset_size": None,
    "train_subset_strategy": "random",
    "train_subset_seed": None,
    "num_workers": 0,
    "epochs": 40,
    "early_stopping_enabled": True,
    "early_stopping_patience": 15,
    "early_stopping_min_epochs": 10,
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler": "cosine",
    "warmup_epochs": 0,
    "penalty_weight": 1000.0,
    "penalty_anneal_epochs": 5,
    "prune_rate": 0.7,
    "dropout": 0.2,
    "stem_channels": 32,
    "stage_channels": [64, 128, 128],
    "stage_pool_kernels": [[2, 1], [2, 2], [2, 2]],
    "amp_enabled": True,
    "grad_clip_norm": 1.0,
    "seed": 0,
    "seeds": [0, 1, 2],
    "device": "auto",
    "log_every_n_steps": 10,
    "show_progress": True,
    "leave_progress_bars": False,
    "progress_refresh_rate": 1,
}


def infer_method_name(backbone_type: str, penalty_type: str) -> str:
    mapping = {
        ("dense", "none"): "erm_dense",
        ("dense", "irm"): "irm_dense",
        ("dense", "rex"): "rex_dense",
        ("sparse", "none"): "erm_sparse",
        ("sparse", "irm"): "sparseirm",
        ("sparse", "rex"): "sparserex",
    }
    try:
        return mapping[(backbone_type, penalty_type)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported combination backbone_type={backbone_type}, penalty_type={penalty_type}"
        ) from exc


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "none":
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["dataset_root"] = str(Path(normalized["dataset_root"]))
    normalized["output_root"] = str(Path(normalized["output_root"]))
    normalized["dataset_name"] = str(normalized["dataset_name"])
    normalized["model_family"] = str(normalized.get("model_family", "wifi_csi_default")).lower()
    normalized["input_adapter"] = str(normalized.get("input_adapter", "auto")).lower()
    normalized["normalization_mode"] = str(
        normalized.get("normalization_mode", "train_channelwise")
    ).lower()
    normalized["nonfinite_policy"] = str(normalized.get("nonfinite_policy", "none")).lower()
    normalized["auto_infer_num_classes"] = bool(normalized.get("auto_infer_num_classes", True))
    normalized["auto_infer_envs"] = bool(normalized.get("auto_infer_envs", True))
    normalized["backbone_type"] = str(normalized["backbone_type"]).lower()
    normalized["penalty_type"] = str(normalized["penalty_type"]).lower()
    normalized["epochs"] = int(normalized["epochs"])
    normalized["early_stopping_enabled"] = bool(normalized.get("early_stopping_enabled", True))
    normalized["early_stopping_patience"] = max(1, int(normalized.get("early_stopping_patience", 15)))
    normalized["early_stopping_min_epochs"] = max(1, int(normalized.get("early_stopping_min_epochs", 10)))
    normalized["batch_size"] = int(normalized["batch_size"])
    train_subset_size = normalized.get("train_subset_size", None)
    normalized["train_subset_size"] = None if train_subset_size is None else max(1, int(train_subset_size))
    normalized["train_subset_strategy"] = str(normalized.get("train_subset_strategy", "random")).lower()
    if normalized["train_subset_strategy"] not in {"random", "stratified"}:
        raise ValueError("train_subset_strategy must be one of: random, stratified")
    train_subset_seed = normalized.get("train_subset_seed", None)
    normalized["train_subset_seed"] = None if train_subset_seed is None else int(train_subset_seed)
    normalized["num_workers"] = int(normalized["num_workers"])
    normalized["num_classes"] = int(normalized["num_classes"])
    normalized["num_source_envs"] = int(normalized["num_source_envs"])
    normalized["target_env"] = int(normalized["target_env"])
    normalized["seed"] = int(normalized["seed"])
    normalized["prune_rate"] = float(normalized["prune_rate"])
    normalized["lr"] = float(normalized["lr"])
    normalized["weight_decay"] = float(normalized["weight_decay"])
    normalized["warmup_epochs"] = max(0, int(normalized.get("warmup_epochs", 0)))
    normalized["penalty_weight"] = float(normalized["penalty_weight"])
    normalized["penalty_anneal_epochs"] = int(normalized["penalty_anneal_epochs"])
    normalized["dropout"] = float(normalized["dropout"])
    normalized["stem_channels"] = int(normalized.get("stem_channels", 32))
    normalized["stage_channels"] = [int(v) for v in normalized.get("stage_channels", [64, 128, 128])]
    pool_kernels = []
    for value in normalized.get("stage_pool_kernels", [[2, 1], [2, 2], [2, 2]]):
        if value is None:
            pool_kernels.append(None)
        else:
            pool_kernels.append([int(value[0]), int(value[1])])
    normalized["stage_pool_kernels"] = pool_kernels
    if len(normalized["stage_channels"]) != 3 or len(normalized["stage_pool_kernels"]) != 3:
        raise ValueError("stage_channels and stage_pool_kernels must both contain exactly 3 items.")
    normalized["amp_enabled"] = bool(normalized.get("amp_enabled", True))
    clip_value = normalized.get("grad_clip_norm", None)
    normalized["grad_clip_norm"] = None if clip_value is None else float(clip_value)
    normalized["show_progress"] = bool(normalized["show_progress"])
    normalized["leave_progress_bars"] = bool(normalized["leave_progress_bars"])
    normalized["progress_refresh_rate"] = max(1, int(normalized["progress_refresh_rate"]))
    normalized["seeds"] = [int(seed) for seed in normalized.get("seeds", [normalized["seed"]])]
    if normalized["method_name"] is None:
        normalized["method_name"] = infer_method_name(
            normalized["backbone_type"], normalized["penalty_type"]
        )
    return normalized


def load_config(config_path: Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    with Path(config_path).open("r", encoding="utf-8") as handle:
        file_config = yaml.safe_load(handle) or {}
    config.update(file_config)
    applied_overrides = overrides or {}
    for key, value in applied_overrides.items():
        if value is not None:
            config[key] = _coerce_scalar(value)
    if "method_name" not in applied_overrides and (
        "backbone_type" in applied_overrides or "penalty_type" in applied_overrides
    ):
        config["method_name"] = None
    return normalize_config(config)


def save_config(config: Dict[str, Any], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
