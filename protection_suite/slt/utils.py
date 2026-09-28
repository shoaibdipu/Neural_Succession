from __future__ import annotations
import json, os, random, time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_arg(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | os.PathLike) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(x: Any):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)


class WallTimer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
