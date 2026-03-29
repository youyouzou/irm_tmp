from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _layer_density(module: nn.Module) -> float:
    if hasattr(module, "mask_density"):
        return float(module.mask_density())
    if hasattr(module, "weight") and module.weight is not None:
        return 1.0
    return 1.0


def _count_params(module: nn.Module) -> int:
    return int(module.weight.numel()) if hasattr(module, "weight") and module.weight is not None else 0


def collect_sparsity_report(model: nn.Module, input_shape: Tuple[int, int, int], device: torch.device) -> Dict[str, object]:
    layers: List[Dict[str, object]] = []
    total_params = 0
    effective_params = 0.0
    dense_flops = 0.0
    effective_flops = 0.0
    hooks = []

    def register(name: str, module: nn.Module):
        nonlocal total_params, effective_params
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            param_count = _count_params(module)
            density = _layer_density(module)
            total_params += param_count
            effective_params += param_count * density

            def hook(_module, inputs, outputs):
                nonlocal dense_flops, effective_flops
                if isinstance(_module, nn.Conv2d):
                    output = outputs
                    kernel_ops = _module.kernel_size[0] * _module.kernel_size[1] * (_module.in_channels / _module.groups)
                    layer_dense_flops = float(output.numel() * kernel_ops)
                else:
                    input_tensor = inputs[0]
                    layer_dense_flops = float(input_tensor.shape[-1] * _module.out_features * input_tensor.shape[0])
                dense_flops += layer_dense_flops
                effective_flops += layer_dense_flops * _layer_density(_module)

            hooks.append(module.register_forward_hook(hook))
            layers.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "param_count": param_count,
                    "density": density,
                    "sparsity": 1.0 - density,
                    "effective_params": float(param_count * density),
                }
            )

    for name, module in model.named_modules():
        register(name, module)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros((1, *input_shape), device=device)
        model(dummy)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    global_density = 0.0 if total_params == 0 else effective_params / total_params
    return {
        "global_sparsity_rate": float(1.0 - global_density),
        "total_params": int(total_params),
        "effective_params": float(effective_params),
        "approx_dense_flops": float(dense_flops),
        "approx_effective_flops": float(effective_flops),
        "layers": layers,
    }
