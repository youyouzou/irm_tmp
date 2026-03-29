from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def selection_score(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(metrics.get("overall_acc", 0.0)),
        float(metrics.get("worst_source_env_acc", 0.0) or 0.0),
        -float(metrics.get("loss", float("inf"))),
    )


def is_better(candidate: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
    if best is None:
        return True
    return selection_score(candidate) > selection_score(best)
