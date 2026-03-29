import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: Optional[str] = None) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(state, path)


def write_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
