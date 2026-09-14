"""Общие константы и компактное хранение геометрии.

Модуль используется и сборщиком каталога (``service/build_catalog.py``), и
работающим сервисом, поэтому зависит только от numpy/shapely.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely import GeometryType

CATALOG_VERSION = 1

SOURCES = ("model", "reference")
SOURCE_LABELS = {"model": "Модель", "reference": "Эталонная разметка"}

# Классы степени поражения (маска BS: 0 — не горело, 1..3 — степень)
SEVERITY = {
    1: {"key": "low", "label": "слабая", "color": "#f2c14e"},
    2: {"key": "moderate", "label": "средняя", "color": "#ec7c30"},
    3: {"key": "high", "label": "сильная", "color": "#b3261e"},
}
SEVERITY_CODES = (1, 2, 3)

# ESA WorldCover v200 (канал landcover во вспомогательных растрах)
LANDCOVER = {
    10: "Древесная растительность",
    20: "Кустарники",
    30: "Травянистая растительность",
    40: "Пахотные земли",
    50: "Застройка",
    60: "Открытый грунт, редкая растительность",
    70: "Снег и лёд",
    80: "Водные объекты",
    90: "Травянистые водно-болотные угодья",
    95: "Мангровые заросли",
    100: "Мхи и лишайники",
}
LANDCOVER_UNKNOWN = "Нет данных"

SATELLITES = {"SNPP": "Suomi NPP", "NOAA20": "NOAA-20", "NOAA21": "NOAA-21"}

# Пороговое значение зенитного угла Солнца для деления на день/ночь (как в VIIRS AF)
DAY_SZA_MAX = 85.0

AF_GSD = 375.0  # м, пиксель VIIRS I-диапазона
BS_GSD = 20.0  # м, пиксель чипов Sentinel-2


def landcover_name(code) -> str:
    try:
        return LANDCOVER.get(int(code), LANDCOVER_UNKNOWN)
    except (TypeError, ValueError):
        return LANDCOVER_UNKNOWN


# ---------------------------------------------------------------------------
# Компактное хранение полигонов, лежащих на пиксельной сетке чипа.
#
# Контуры получены векторизацией растров, поэтому все вершины — целые номера
# узлов пиксельной сетки (col, row). Храним их как int16 с дельта-кодированием
# в «рваных» массивах shapely (to_ragged_array) — в разы компактнее WKB и
# восстанавливается без потерь. Перевод в UTM: x = x0 + col*gsd, y = y0 - row*gsd.
# ---------------------------------------------------------------------------


def to_multipolygon(g):
    if g is None or g.is_empty:
        return shapely.MultiPolygon()
    t = shapely.get_type_id(g)
    if t == GeometryType.MULTIPOLYGON:
        return g
    if t == GeometryType.POLYGON:
        return shapely.MultiPolygon([g])
    parts = [p for p in shapely.get_parts(g) if shapely.get_type_id(p) in (3, 6)]
    polys = []
    for p in parts:
        polys.extend(shapely.get_parts(p) if shapely.get_type_id(p) == 6 else [p])
    return shapely.MultiPolygon(polys) if polys else shapely.MultiPolygon()


def encode_pixel_geoms(geoms) -> dict[str, np.ndarray]:
    """Список (Multi)Polygon в пиксельных координатах -> словарь массивов."""
    geoms = np.asarray([to_multipolygon(g) for g in geoms], dtype=object)
    if len(geoms) == 0:
        return {
            "d": np.zeros((0, 2), np.int16),
            "o0": np.zeros(1, np.int32),
            "o1": np.zeros(1, np.int32),
            "o2": np.zeros(1, np.int32),
        }
    typ, coords, offsets = shapely.to_ragged_array(geoms)
    assert typ == GeometryType.MULTIPOLYGON
    ci = np.rint(coords).astype(np.int64)
    if np.abs(coords - ci).max(initial=0) > 1e-6:
        raise ValueError("координаты не на пиксельной сетке")
    d = np.diff(ci, axis=0, prepend=np.zeros((1, 2), np.int64))
    if np.abs(d).max(initial=0) > 32767:
        raise ValueError("слишком большие приращения координат")
    return {
        "d": d.astype(np.int16),
        "o0": offsets[0].astype(np.int32),
        "o1": offsets[1].astype(np.int32),
        "o2": offsets[2].astype(np.int32),
    }


def concat_encoded(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Склейка нескольких закодированных блоков в один (со сдвигом смещений)."""
    ds, o0s, o1s, o2s = [], [np.zeros(1, np.int64)], [np.zeros(1, np.int64)], [np.zeros(1, np.int64)]
    nc = nr = npoly = 0
    for p in parts:
        c = np.cumsum(p["d"].astype(np.int64), axis=0)  # абсолютные координаты блока
        ds.append(c)
        o0s.append(p["o0"][1:].astype(np.int64) + nc)
        o1s.append(p["o1"][1:].astype(np.int64) + nr)
        o2s.append(p["o2"][1:].astype(np.int64) + npoly)
        nc += len(c)
        nr += len(p["o0"]) - 1
        npoly += len(p["o1"]) - 1
    coords = np.concatenate(ds) if ds else np.zeros((0, 2), np.int64)
    d = np.diff(coords, axis=0, prepend=np.zeros((1, 2), np.int64))
    return {
        "d": d.astype(np.int16),
        "o0": np.concatenate(o0s).astype(np.int32),
        "o1": np.concatenate(o1s).astype(np.int32),
        "o2": np.concatenate(o2s).astype(np.int32),
    }


def decode_geoms(enc: dict[str, np.ndarray], x0: np.ndarray, y0: np.ndarray, gsd: np.ndarray) -> np.ndarray:
    """Восстановление массива MultiPolygon в координатах UTM.

    x0, y0, gsd — массивы длины n_geoms (параметры сетки чипа для каждой геометрии).
    """
    o0, o1, o2 = enc["o0"].astype(np.int64), enc["o1"].astype(np.int64), enc["o2"].astype(np.int64)
    n = len(o2) - 1
    if n <= 0:
        return np.empty(0, dtype=object)
    px = np.cumsum(enc["d"].astype(np.int64), axis=0).astype(np.float64)
    # число вершин каждой геометрии
    start = o0[o1[o2]]
    counts = np.diff(start)
    gx0 = np.repeat(np.asarray(x0, np.float64), counts)
    gy0 = np.repeat(np.asarray(y0, np.float64), counts)
    gg = np.repeat(np.asarray(gsd, np.float64), counts)
    xy = np.column_stack([gx0 + px[:, 0] * gg, gy0 - px[:, 1] * gg])
    return shapely.from_ragged_array(GeometryType.MULTIPOLYGON, xy, (o0, o1, o2))
