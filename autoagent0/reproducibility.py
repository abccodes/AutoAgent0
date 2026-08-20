"""Opt-in process seeding for reproducible HUGSIM benchmark controls."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def apply_benchmark_seed() -> Optional[int]:
    """Seed Python, NumPy, and Torch when HUGSIM_BENCHMARK_SEED is set."""

    raw_seed = os.environ.get("HUGSIM_BENCHMARK_SEED", "").strip()
    if not raw_seed:
        return None
    seed = int(raw_seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        deterministic = getattr(torch, "use_deterministic_algorithms", None)
        if deterministic is not None and _enabled("HUGSIM_DETERMINISTIC_ALGORITHMS"):
            try:
                deterministic(True, warn_only=True)
            except TypeError:
                deterministic(True)
    except ImportError:
        pass
    return seed
