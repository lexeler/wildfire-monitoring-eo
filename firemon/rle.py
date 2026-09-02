"""RLE-кодирование масок в формате соревнования.

Пиксели нумеруются построчно (слева направо, сверху вниз) начиная с 1.
Строка — пары «начало длина» через пробел, по возрастанию начала; серии
не пересекаются и не соприкасаются. Пустая строка — пикселей класса нет.
"""
from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    """Бинарная маска (H, W) -> строка RLE."""
    flat = np.asarray(mask, dtype=bool).ravel(order="C")
    if not flat.any():
        return ""
    padded = np.concatenate([[False], flat, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts = edges[0::2] + 1  # нумерация с единицы
    lengths = edges[1::2] - edges[0::2]
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Строка RLE -> бинарная маска (H, W). Проверяет корректность серий."""
    h, w = shape
    out = np.zeros(h * w, dtype=bool)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        raise ValueError("RLE is NaN")
    rle = str(rle).strip()
    if not rle:
        return out.reshape(h, w)
    nums = np.array(rle.split(), dtype=np.int64)
    if nums.size % 2:
        raise ValueError("odd number of RLE tokens")
    starts, lengths = nums[0::2], nums[1::2]
    if (lengths <= 0).any():
        raise ValueError("non-positive run length")
    if (starts < 1).any() or (starts + lengths - 1 > h * w).any():
        raise ValueError("run out of chip bounds")
    ends = starts + lengths  # exclusive, 1-based
    if (starts[1:] <= ends[:-1]).any():  # пересечение или касание серий
        raise ValueError("runs overlap/touch or are unsorted")
    for s, l in zip(starts, lengths):
        out[s - 1 : s - 1 + l] = True
    return out.reshape(h, w)
