from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from wifi_dg.metrics import compute_classification_metrics


def _env_losses(logits: torch.Tensor, targets: torch.Tensor, envs: torch.Tensor) -> torch.Tensor:
    losses = []
    for env_id in torch.unique(envs):
        mask = envs == env_id
        if int(mask.sum()) == 0:
            continue
        losses.append(F.cross_entropy(logits[mask], targets[mask]))
    if not losses:
        return torch.zeros(1, device=logits.device)
    return torch.stack(losses)


def compute_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    envs: torch.Tensor,
    penalty_type: str,
) -> torch.Tensor:
    penalty_type = penalty_type.lower()
    if penalty_type == "none":
        return torch.tensor(0.0, device=logits.device)

    env_losses = _env_losses(logits, targets, envs)
    if env_losses.numel() <= 1:
        return torch.tensor(0.0, device=logits.device)

    if penalty_type == "rex":
        return env_losses.var(unbiased=False)

    if penalty_type == "irm":
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        penalties = []
        for env_id in torch.unique(envs):
            mask = envs == env_id
            if int(mask.sum()) == 0:
                continue
            env_loss = F.cross_entropy(logits[mask] * scale, targets[mask])
            grad = autograd.grad(env_loss, [scale], create_graph=True)[0]
            penalties.append(grad.pow(2))
        return torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=logits.device)

    raise ValueError(f"Unsupported penalty_type: {penalty_type}")


def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    config: Dict,
    optimizer=None,
    epoch: int = 0,
    split_name: str = "train",
    scaler=None,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_penalty = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []
    all_envs = []

    show_progress = bool(config.get("show_progress", True))
    leave_progress_bars = bool(config.get("leave_progress_bars", False))
    progress_refresh_rate = int(config.get("progress_refresh_rate", 1))
    amp_enabled = bool(config.get("amp_enabled", True)) and device.type == "cuda"
    grad_clip_norm = config.get("grad_clip_norm", None)
    penalty_weight = (
        float(config["penalty_weight"])
        if is_train and epoch >= int(config["penalty_anneal_epochs"])
        else 0.0
    )

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=f"{split_name} epoch {epoch + 1}",
        leave=leave_progress_bars,
        disable=not show_progress,
        miniters=progress_refresh_rate,
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(progress_bar):
        if not isinstance(batch, (tuple, list)):
            raise ValueError(f"Unexpected batch type: {type(batch)}")
        if len(batch) == 3:
            inputs, targets, envs = batch
        elif len(batch) == 4:
            # subject is optionally returned by MMFI datasets but not consumed in current training objective.
            inputs, targets, envs, _ = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        envs = envs.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs)
                erm_loss = F.cross_entropy(logits, targets)
                penalty = (
                    compute_penalty(logits, targets, envs, config["penalty_type"])
                    if is_train
                    else torch.tensor(0.0, device=logits.device)
                )
                loss = erm_loss + penalty_weight * penalty

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                    optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_acc = float((predictions == targets).float().mean().item())
        batch_size = int(targets.size(0))
        total_samples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_penalty += float(penalty.detach().item()) * batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())
        all_envs.append(envs.detach().cpu().numpy())
        progress_bar.set_postfix(
            loss=f"{float(loss.detach().item()):.4f}",
            penalty=f"{float(penalty.detach().item()):.4f}",
            batch_acc=f"{batch_acc:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.3e}" if optimizer is not None else "-",
        )

    progress_bar.close()

    labels = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    envs = np.concatenate(all_envs)
    metrics, matrix = compute_classification_metrics(
        labels,
        predictions,
        envs=envs if split_name in {"val", "test"} else None,
        num_classes=int(config["num_classes"]),
    )
    metrics["loss"] = total_loss / max(1, total_samples)
    metrics["acc"] = float(metrics["overall_acc"])
    metrics["penalty"] = total_penalty / max(1, total_samples)
    metrics["split"] = split_name
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics
