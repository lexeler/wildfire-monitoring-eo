"""Предметные правила степени поражения.

Пороги dNBR по типу покрова взяты из постановки кейса (рис. 6): для леса и
кустарника — шкала USGS, для травы, пашни и болот/поймы — региональная
калибровка организаторов. EDA на train показала, что эталон строится по dNBR,
сглаженному медианным фильтром 3x3 (совпадение степени внутри гарей 98.4%
против 93.2% без сглаживания).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# Границы классов 1|2|3 (dNBR ниже первой границы — «не горело»).
# Трава и пашня — ровно пороги постановки (на train они оптимальны).
# Для леса и болот/поймы граница 2|3 откалибрована по train (перебор с шагом
# 0.0025 по пикселям эталонных гарей): 0.66 -> 0.575 (согласие 0.907 -> 0.919)
# и 0.677 -> 0.61 (0.928 -> 0.943). Границы 0|1 и 1|2 совпадают с постановкой.
THRESHOLDS_STATEMENT = {
    "forest": (0.100, 0.270, 0.660),  # USGS
    "grass": (0.062, 0.204, 0.386),
    "crop": (0.070, 0.177, 0.380),
    "wetland": (0.075, 0.331, 0.677),
}
THRESHOLDS = {
    "forest": (0.100, 0.270, 0.575),
    "grass": (0.062, 0.204, 0.386),
    "crop": (0.070, 0.177, 0.380),
    "wetland": (0.075, 0.331, 0.610),
}

# ESA WorldCover v200 -> группа порогов
LANDCOVER_GROUP = {
    10: "forest",  # деревья
    20: "forest",  # кустарник
    30: "grass",  # травянистая растительность
    40: "crop",  # пашня
    50: "grass",  # застройка (редко горит; пороги травы)
    60: "grass",  # голый грунт / редкая растительность
    70: "grass",  # снег
    80: "wetland",  # вода (поймы)
    90: "wetland",  # травянистые болота
    95: "wetland",  # мангры
    100: "grass",  # мхи и лишайники
}

# SCL-классы, которые в эталоне всегда «не горело» (EDA: доля гари среди
# пикселей с dNBR выше порога = 0.000-0.005): нет данных, тень облака, вода,
# облако высокой вероятности, снег — на любой из двух дат
SCL_EXCLUDE = (0, 3, 6, 9, 11)


def excluded(scl_pre: np.ndarray, scl_post: np.ndarray, extra_post: tuple = ()) -> np.ndarray:
    return np.isin(scl_pre, SCL_EXCLUDE) | np.isin(scl_post, SCL_EXCLUDE + tuple(extra_post))


def nbr(s2: np.ndarray) -> np.ndarray:
    """NBR = (B8A - B12) / (B8A + B12) по массиву (10, H, W) uint16."""
    b8a = s2[6].astype(np.float32)
    b12 = s2[8].astype(np.float32)
    return (b8a - b12) / np.maximum(b8a + b12, 1.0)


def dnbr(s2_pre: np.ndarray, s2_post: np.ndarray, median: int = 3) -> np.ndarray:
    d = nbr(s2_pre) - nbr(s2_post)
    return ndi.median_filter(d, size=median) if median and median > 1 else d


def severity_from_dnbr(d: np.ndarray, landcover: np.ndarray, thresholds: dict | None = None) -> np.ndarray:
    """Класс 0..3 по порогам своей группы покрова."""
    th = thresholds or THRESHOLDS
    out = np.zeros(d.shape, np.uint8)
    lc = landcover.astype(np.int32)
    for code, group in LANDCOVER_GROUP.items():
        sel = lc == code
        if sel.any():
            out[sel] = np.digitize(d[sel], th[group]).astype(np.uint8)
    # неизвестные коды — пороги травы (доминирующий покров региона)
    unknown = ~np.isin(lc, list(LANDCOVER_GROUP))
    if unknown.any():
        out[unknown] = np.digitize(d[unknown], th["grass"]).astype(np.uint8)
    return out


def output_rule(s2_pre: np.ndarray, s2_post: np.ndarray, landcover: np.ndarray,
                mode: str = "legacy") -> np.ndarray:
    """Output severity; kept separate from the frozen model input features.

    class_median: classify raw dNBR, zero SCL-excluded neighbours, then take
    the ordinal class median over 3x3. On OOF this improves all five folds.
    The old rule remains available for reproducing historical experiments.
    """
    if mode == "legacy":
        return severity_from_dnbr(dnbr(s2_pre, s2_post), landcover)
    if mode != "class_median":
        raise ValueError(f"Unknown severity rule: {mode}")
    r = severity_from_dnbr(dnbr(s2_pre, s2_post, median=0), landcover)
    ex = excluded(s2_pre[9], s2_post[9])
    r[ex] = 0
    r = ndi.median_filter(r, size=3)
    r[ex] = 0
    return r


# Minimum mapping unit of the reference masks: every burned component in train
# (8-connected, binary burn mask) has at least 25 pixels = 1 ha at 20 m; of 4997
# components only 4 are smaller (1-2 px leftovers). Smaller predicted objects
# therefore cannot be present in the reference and are removed.
MIN_AREA_PX = 25


def sieve_small(burned: np.ndarray, min_px: int = MIN_AREA_PX) -> np.ndarray:
    """Remove 8-connected components of a boolean mask smaller than min_px pixels."""
    if min_px <= 1 or not burned.any():
        return burned
    lab, n = ndi.label(burned, structure=np.ones((3, 3), bool))
    if n == 0:
        return burned
    sizes = np.bincount(lab.ravel())
    return burned & (sizes[lab] >= min_px)
