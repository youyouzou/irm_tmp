from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


LEGACY_TIMESTEPS = 500
LEGACY_FEATURES = 232
LEGACY_CHANNELS = 4
LEGACY_SUBCARRIERS = 58


def _stats_cache_path(dataset_root: Path) -> Path:
    return dataset_root / "normalization_stats.npz"


def _is_mmfi_like(shape: Tuple[int, ...]) -> bool:
    return len(shape) == 4


def _is_legacy_like(shape: Tuple[int, ...]) -> bool:
    return len(shape) == 2 and shape[0] == LEGACY_TIMESTEPS and shape[1] == LEGACY_FEATURES


def _resolve_input_adapter(config: Dict, sample_shape: Tuple[int, ...]) -> str:
    adapter = str(config.get("input_adapter", "auto")).lower()
    if adapter != "auto":
        return adapter
    if _is_mmfi_like(sample_shape):
        return "mmfi_tcsx"
    if _is_legacy_like(sample_shape):
        return "legacy_4x58"
    if len(sample_shape) == 2:
        return "two_d_single_channel"
    raise ValueError(f"Cannot infer input adapter for sample shape {sample_shape}")


def _reshape_sample(sample: np.ndarray, adapter: str) -> np.ndarray:
    sample = np.asarray(sample, dtype=np.float32)
    if adapter == "two_d_single_channel":
        if sample.ndim != 2:
            raise ValueError(f"two_d_single_channel expects 2D sample, got shape {sample.shape}")
        return sample[None, :, :]

    if adapter == "legacy_4x58":
        if sample.shape != (LEGACY_TIMESTEPS, LEGACY_FEATURES):
            raise ValueError(
                f"legacy_4x58 expects {(LEGACY_TIMESTEPS, LEGACY_FEATURES)}, got {sample.shape}"
            )
        return sample.reshape(LEGACY_TIMESTEPS, LEGACY_CHANNELS, LEGACY_SUBCARRIERS).transpose(1, 0, 2)

    if adapter == "mmfi_tcsx":
        if sample.ndim != 4:
            raise ValueError(f"mmfi_tcsx expects 4D sample, got shape {sample.shape}")
        # (T, C1, S, C2) -> (C1*C2, T, S)
        t, c1, s, c2 = sample.shape
        return sample.transpose(1, 3, 0, 2).reshape(c1 * c2, t, s)

    raise ValueError(f"Unsupported input_adapter: {adapter}")


def _interpolate_invalid_time_series(series: np.ndarray) -> Tuple[np.ndarray, bool]:
    finite_mask = np.isfinite(series)
    if finite_mask.all():
        return series, False

    fixed = series.copy()
    valid_idx = np.flatnonzero(finite_mask)
    if valid_idx.size == 0:
        fixed.fill(0.0)
        return fixed, True

    invalid_idx = np.flatnonzero(~finite_mask)
    fixed[invalid_idx] = np.interp(invalid_idx.astype(np.float32), valid_idx.astype(np.float32), fixed[finite_mask])
    return fixed, False


def _handle_nonfinite(sample: np.ndarray, nonfinite_policy: str) -> Tuple[np.ndarray, Dict[str, int]]:
    policy = str(nonfinite_policy).lower()
    nonfinite_count = int((~np.isfinite(sample)).sum())
    report = {"nonfinite_count": nonfinite_count, "all_invalid_series_count": 0}
    if nonfinite_count == 0:
        return sample, report
    if policy not in {"linear_interp"}:
        raise ValueError(
            f"Non-finite values detected but nonfinite_policy={nonfinite_policy} is unsupported. "
            "Use nonfinite_policy=linear_interp."
        )

    fixed = sample.copy()
    channels, timesteps, subcarriers = fixed.shape
    all_invalid_series = 0
    for channel in range(channels):
        for subcarrier in range(subcarriers):
            series = fixed[channel, :, subcarrier]
            fixed_series, was_all_invalid = _interpolate_invalid_time_series(series)
            fixed[channel, :, subcarrier] = fixed_series
            if was_all_invalid:
                all_invalid_series += 1

    if not np.isfinite(fixed).all():
        raise ValueError("Non-finite values remain after linear interpolation.")
    report["all_invalid_series_count"] = int(all_invalid_series)
    return fixed, report


def _adapt_batch(raw_batch: np.ndarray, adapter: str, nonfinite_policy: str) -> Tuple[np.ndarray, Dict[str, int]]:
    raw_batch = np.asarray(raw_batch, dtype=np.float32)
    adapted_samples = []
    total_nonfinite = 0
    total_all_invalid_series = 0
    for sample in raw_batch:
        adapted = _reshape_sample(sample, adapter)
        adapted, report = _handle_nonfinite(adapted, nonfinite_policy)
        adapted_samples.append(adapted)
        total_nonfinite += int(report["nonfinite_count"])
        total_all_invalid_series += int(report["all_invalid_series_count"])
    stacked = np.stack(adapted_samples, axis=0).astype(np.float32)
    return stacked, {
        "nonfinite_count": int(total_nonfinite),
        "all_invalid_series_count": int(total_all_invalid_series),
    }


def _scan_nonfinite(split_x: np.ndarray, adapter: str, nonfinite_policy: str, chunk_size: int = 128) -> Dict[str, int]:
    nonfinite_count = 0
    all_invalid_series_count = 0
    for start in range(0, int(split_x.shape[0]), int(chunk_size)):
        chunk = split_x[start : start + int(chunk_size)]
        _, report = _adapt_batch(chunk, adapter=adapter, nonfinite_policy=nonfinite_policy)
        nonfinite_count += int(report["nonfinite_count"])
        all_invalid_series_count += int(report["all_invalid_series_count"])
    return {
        "nonfinite_count": int(nonfinite_count),
        "all_invalid_series_count": int(all_invalid_series_count),
    }


def _resolve_normalization_mode(config: Dict) -> str:
    mode = str(config.get("normalization_mode", "train_channelwise")).lower()
    if mode not in {"train_channelwise", "train_global_scalar"}:
        raise ValueError(f"Unsupported normalization_mode: {mode}")
    return mode


def _build_stratified_indices(labels: np.ndarray, subset_size: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = int(labels.shape[0])
    if subset_size >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes = sorted(int(v) for v in np.unique(labels))
    class_to_indices = {cls: np.flatnonzero(labels == cls).astype(np.int64) for cls in classes}
    for cls in classes:
        rng.shuffle(class_to_indices[cls])

    quota = subset_size // max(1, len(classes))
    picked = []
    for cls in classes:
        cls_indices = class_to_indices[cls]
        take = min(len(cls_indices), quota)
        if take > 0:
            picked.extend(cls_indices[:take].tolist())
            class_to_indices[cls] = cls_indices[take:]

    remaining = subset_size - len(picked)
    if remaining > 0:
        pool = np.concatenate([class_to_indices[cls] for cls in classes if len(class_to_indices[cls]) > 0], axis=0)
        if pool.size > 0:
            rng.shuffle(pool)
            picked.extend(pool[:remaining].tolist())

    if len(picked) < subset_size:
        all_indices = np.arange(n, dtype=np.int64)
        rng.shuffle(all_indices)
        chosen = set(picked)
        for idx in all_indices:
            if int(idx) not in chosen:
                picked.append(int(idx))
                chosen.add(int(idx))
                if len(picked) >= subset_size:
                    break

    result = np.asarray(picked[:subset_size], dtype=np.int64)
    rng.shuffle(result)
    return result


def _resolve_train_subset(labels: np.ndarray, config: Dict) -> Dict[str, object]:
    subset_size = config.get("train_subset_size", None)
    if subset_size is None:
        return {
            "enabled": False,
            "strategy": None,
            "seed": None,
            "requested_size": None,
            "actual_size": int(labels.shape[0]),
        }

    total = int(labels.shape[0])
    requested = max(1, int(subset_size))
    actual_size = min(requested, total)
    strategy = str(config.get("train_subset_strategy", "random")).lower()
    seed = int(config["seed"]) if config.get("train_subset_seed") is None else int(config["train_subset_seed"])
    rng = np.random.default_rng(seed)

    if actual_size >= total:
        indices = np.arange(total, dtype=np.int64)
    elif strategy == "stratified":
        indices = _build_stratified_indices(labels, subset_size=actual_size, seed=seed)
    else:
        indices = rng.choice(total, size=actual_size, replace=False).astype(np.int64)

    return {
        "enabled": True,
        "strategy": strategy,
        "seed": int(seed),
        "requested_size": int(requested),
        "actual_size": int(actual_size),
        "indices": indices,
    }


def _compute_train_stats(
    train_x: np.ndarray,
    adapter: str,
    nonfinite_policy: str,
    normalization_mode: str,
    cache_path: Path,
    chunk_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    expected_shape = _reshape_sample(np.asarray(train_x[0], dtype=np.float32), adapter).shape
    channels, _, subcarriers = expected_shape
    if cache_path.exists():
        cached = np.load(cache_path)
        mean = cached["mean"]
        std = cached["std"]
        cache_adapter = str(cached["adapter"]) if "adapter" in cached else ""
        cache_policy = str(cached["nonfinite_policy"]) if "nonfinite_policy" in cached else ""
        cache_norm_mode = str(cached["normalization_mode"]) if "normalization_mode" in cached else ""
        cache_match = (not cache_adapter or cache_adapter == adapter) and (
            not cache_policy or cache_policy == nonfinite_policy
        ) and (
            not cache_norm_mode or cache_norm_mode == normalization_mode
        )
        if normalization_mode == "train_channelwise":
            shape_match = mean.shape == (channels, subcarriers) and std.shape == (channels, subcarriers)
        else:
            shape_match = mean.ndim == 0 and std.ndim == 0
            if not shape_match:
                shape_match = mean.size == 1 and std.size == 1
        if cache_match and shape_match:
            train_nonfinite_count = int(cached["train_nonfinite_count"]) if "train_nonfinite_count" in cached else 0
            train_all_invalid_series_count = (
                int(cached["train_all_invalid_series_count"])
                if "train_all_invalid_series_count" in cached
                else 0
            )
            if normalization_mode == "train_global_scalar":
                mean_out = np.float32(float(np.asarray(mean).reshape(-1)[0]))
                std_out = np.float32(float(np.asarray(std).reshape(-1)[0]))
            else:
                mean_out = mean.astype(np.float32)
                std_out = std.astype(np.float32)
            return mean_out, std_out, {
                "nonfinite_count": train_nonfinite_count,
                "all_invalid_series_count": train_all_invalid_series_count,
            }

    if normalization_mode == "train_channelwise":
        running_sum = np.zeros((channels, subcarriers), dtype=np.float64)
        running_sq_sum = np.zeros((channels, subcarriers), dtype=np.float64)
        running_count = np.zeros((channels, subcarriers), dtype=np.float64)
    else:
        running_sum = 0.0
        running_sq_sum = 0.0
        running_count = 0.0
    total_nonfinite = 0
    total_all_invalid_series = 0

    for start in range(0, int(train_x.shape[0]), int(chunk_size)):
        chunk = train_x[start : start + int(chunk_size)]
        adapted_chunk, report = _adapt_batch(chunk, adapter=adapter, nonfinite_policy=nonfinite_policy)
        total_nonfinite += int(report["nonfinite_count"])
        total_all_invalid_series += int(report["all_invalid_series_count"])
        if normalization_mode == "train_channelwise":
            # adapted_chunk: (B, C, T, S)
            running_sum += adapted_chunk.sum(axis=(0, 2), dtype=np.float64)
            running_sq_sum += np.square(adapted_chunk, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
            running_count += float(adapted_chunk.shape[0] * adapted_chunk.shape[2])
        else:
            running_sum += float(adapted_chunk.sum(dtype=np.float64))
            running_sq_sum += float(np.square(adapted_chunk, dtype=np.float64).sum(dtype=np.float64))
            running_count += float(adapted_chunk.size)

    if normalization_mode == "train_channelwise":
        mean = np.divide(
            running_sum,
            running_count,
            out=np.zeros_like(running_sum, dtype=np.float64),
            where=running_count > 0,
        )
        var = np.divide(
            running_sq_sum,
            running_count,
            out=np.zeros_like(running_sq_sum, dtype=np.float64),
            where=running_count > 0,
        ) - np.square(mean)
        var = np.clip(var, 1e-12, None)
        std = np.sqrt(var)
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
    else:
        mean_val = running_sum / max(1.0, running_count)
        var_val = running_sq_sum / max(1.0, running_count) - (mean_val * mean_val)
        var_val = max(var_val, 1e-12)
        mean = np.float32(mean_val)
        std = np.float32(np.sqrt(var_val))

    np.savez(
        cache_path,
        mean=mean,
        std=std,
        adapter=np.asarray(adapter),
        nonfinite_policy=np.asarray(nonfinite_policy),
        normalization_mode=np.asarray(normalization_mode),
        train_nonfinite_count=np.int64(total_nonfinite),
        train_all_invalid_series_count=np.int64(total_all_invalid_series),
    )
    return mean, std, {
        "nonfinite_count": int(total_nonfinite),
        "all_invalid_series_count": int(total_all_invalid_series),
    }


def _load_npz_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    data = np.load(path, mmap_mode="r")
    x = data["x"]
    y = data["y"]
    env = data["env"]
    subject = data["subject"] if "subject" in data else None
    if x.shape[0] != y.shape[0] or y.shape[0] != env.shape[0]:
        raise ValueError(f"{path.name} has inconsistent sample count: x={x.shape}, y={y.shape}, env={env.shape}")
    return x, y, env, subject


@dataclass
class SplitInfo:
    x: np.ndarray
    y: np.ndarray
    env: np.ndarray
    subject: Optional[np.ndarray] = None


class WiFiCSIDataset(Dataset):
    def __init__(self, split: SplitInfo, mean: np.ndarray, std: np.ndarray, adapter: str, nonfinite_policy: str):
        self.x = split.x
        self.y = split.y.astype(np.int64)
        self.env = split.env.astype(np.int64)
        self.subject = split.subject.astype(np.int64) if split.subject is not None else None
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.adapter = adapter
        self.nonfinite_policy = nonfinite_policy

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        sample = np.asarray(self.x[index], dtype=np.float32)
        sample = _reshape_sample(sample, self.adapter)
        sample, _ = _handle_nonfinite(sample, nonfinite_policy=self.nonfinite_policy)
        if np.ndim(self.mean) == 0:
            sample = (sample - float(self.mean)) / float(self.std)
        else:
            sample = (sample - self.mean[:, None, :]) / self.std[:, None, :]
        if not np.isfinite(sample).all():
            raise ValueError(f"Non-finite value encountered after normalization at index {index}")
        batch = (
            torch.from_numpy(sample.copy()),
            torch.tensor(self.y[index], dtype=torch.long),
            torch.tensor(self.env[index], dtype=torch.long),
        )
        if self.subject is None:
            return batch
        return (
            batch[0],
            batch[1],
            batch[2],
            torch.tensor(self.subject[index], dtype=torch.long),
        )


def build_wifi_dataloaders(config: Dict) -> Dict[str, object]:
    dataset_root = Path(config["dataset_root"])
    train_x, train_y, train_env, train_subject = _load_npz_arrays(dataset_root / "train.npz")
    val_x, val_y, val_env, val_subject = _load_npz_arrays(dataset_root / "val.npz")
    test_x, test_y, test_env, test_subject = _load_npz_arrays(dataset_root / "test.npz")

    adapter = _resolve_input_adapter(config, sample_shape=tuple(train_x.shape[1:]))
    nonfinite_policy = str(config.get("nonfinite_policy", "none")).lower()
    normalization_mode = _resolve_normalization_mode(config)
    mean, std, train_nonfinite = _compute_train_stats(
        train_x,
        adapter=adapter,
        nonfinite_policy=nonfinite_policy,
        normalization_mode=normalization_mode,
        cache_path=_stats_cache_path(dataset_root),
    )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics contain NaN/Inf")

    split_map = {
        "train": SplitInfo(train_x, train_y, train_env, train_subject),
        "val": SplitInfo(val_x, val_y, val_env, val_subject),
        "test": SplitInfo(test_x, test_y, test_env, test_subject),
    }
    datasets = {
        split_name: WiFiCSIDataset(
            split_info,
            mean,
            std,
            adapter=adapter,
            nonfinite_policy=nonfinite_policy,
        )
        for split_name, split_info in split_map.items()
    }
    train_subset = _resolve_train_subset(split_map["train"].y, config)
    if bool(train_subset["enabled"]):
        datasets["train"] = Subset(datasets["train"], train_subset["indices"].tolist())

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        ),
    }

    inferred_num_classes = int(np.unique(train_y).shape[0])
    source_envs = sorted(int(v) for v in np.unique(np.concatenate([train_env, val_env], axis=0)))
    target_envs = sorted(int(v) for v in np.unique(test_env))

    input_shape = list(_reshape_sample(np.asarray(train_x[0], dtype=np.float32), adapter).shape)
    nonfinite_report = {
        "train": train_nonfinite,
        "val": _scan_nonfinite(val_x, adapter=adapter, nonfinite_policy=nonfinite_policy),
        "test": _scan_nonfinite(test_x, adapter=adapter, nonfinite_policy=nonfinite_policy),
    }

    if np.ndim(mean) == 0:
        mean_payload = float(mean)
        std_payload = float(std)
    else:
        mean_payload = mean.tolist()
        std_payload = std.tolist()

    return {
        "loaders": loaders,
        "normalization": {"mode": normalization_mode, "mean": mean_payload, "std": std_payload},
        "input_shape": [int(v) for v in input_shape],
        "input_adapter": adapter,
        "num_classes": inferred_num_classes,
        "source_envs": source_envs,
        "target_envs": target_envs,
        "nonfinite_report": nonfinite_report,
        "train_subset": {
            "enabled": bool(train_subset["enabled"]),
            "strategy": train_subset["strategy"],
            "seed": train_subset["seed"],
            "requested_size": train_subset["requested_size"],
            "actual_size": int(len(datasets["train"])),
            "original_train_size": int(train_y.shape[0]),
        },
        "split_sizes": {
            "train": int(len(datasets["train"])),
            "val": int(val_y.shape[0]),
            "test": int(test_y.shape[0]),
        },
    }
