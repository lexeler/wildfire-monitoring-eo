"""Общие утилиты: фиксация случайности, таймер."""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timer(name: str, log=print):
    t = time.perf_counter()
    yield
    log(f"[time] {name}: {time.perf_counter() - t:.2f}s")
