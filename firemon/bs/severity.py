"""Модель степени поражения внутри гари (GBDT по dNBR и составу покрова).

EDA: для травы и пашни эталонный класс — ровно интервалы медианного dNBR
между порогами постановки (согласие 99%). Для леса и болот согласие ниже
(92-94%): на гистограммах видно, что у части «лесных» пикселей границы
классов совпадают с порогами травы/пашни — эталон, по-видимому, брал тип
покрова в другом разрешении, и у краёв массивов действуют пороги соседей.
Модель получает dNBR (медиана/сырой/среднее), класс по правилу каждой группы
порогов и доли типов покрова в окнах 3-17 px и предсказывает класс 1..3.

ОТВЕРГНУТЫЙ ЭКСПЕРИМЕНТ: прирост bs_part +0.0014 не окупает ~10 с CPU в
инференсе; в финале степень задаётся правилом (rules.py). inference.py и
train.py модуль не вызывают; оставлен для воспроизводимости отчёта.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .rules import LANDCOVER_GROUP, THRESHOLDS, dnbr, nbr

LC_CODES = (10, 20, 30, 40, 50, 60, 80, 90)
WINDOWS = (3, 5, 9, 17)


def severity_features(s2_pre: np.ndarray, s2_post: np.ndarray, landcover: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """-> (признаки (F, H, W) float32, имена)."""
    d_med = dnbr(s2_pre, s2_post, median=3)
    d_raw = dnbr(s2_pre, s2_post, median=0)
    f = {"d_med": d_med, "d_raw": d_raw, "d_mean3": ndi.uniform_filter(d_raw, 3), "d_med5": ndi.median_filter(d_raw, 5),
         "nbr_pre": nbr(s2_pre), "nbr_post": nbr(s2_post), "lc": landcover.astype(np.float32)}
    for g, th in THRESHOLDS.items():
        f[f"rule_{g}"] = np.digitize(d_med, th).astype(np.float32)
    lc = landcover.astype(np.int32)
    for code in LC_CODES:
        m = (lc == code).astype(np.float32)
        for w in WINDOWS:
            f[f"lc{code}_w{w}"] = ndi.uniform_filter(m, w)
    names = list(f)
    return np.stack([f[k] for k in names]).astype(np.float32), names


def group_of(landcover: np.ndarray) -> np.ndarray:
    out = np.full(landcover.shape, "grass", dtype=object)
    for code, g in LANDCOVER_GROUP.items():
        out[landcover == code] = g
    return out
