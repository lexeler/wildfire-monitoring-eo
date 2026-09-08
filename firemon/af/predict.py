"""Обёртка инференса модуля активного горения для сервиса (один чип)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..io import AFChip
from .model import AFModel, predict_af

_MODEL: AFModel | None = None


def predict_af_chip(chip: AFChip, weights_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Один AF-чип -> (маска 0/1 uint8, вероятность горения float32)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = AFModel.load(weights_dir or Path(__file__).resolve().parents[2] / "weights")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, p = predict_af(_MODEL, chip.viirs[None], chip.aux[None], device=dev)
    return m[0], p[0]
