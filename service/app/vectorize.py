"""Векторизация растровых результатов (общая для сборщика каталога и /analyze).

* Гари: связные компоненты маски (степень >= 1) по 8-связности — это «контуры
  гарей» (burn_scars). Внутри каждого контура пиксели одной степени образуют
  «зону степени поражения» (severity_zones).
* Полигоны строятся по 4-связности (rasterio.features.shapes) и затем
  группируются по метке 8-связной компоненты в MultiPolygon: так геометрия
  всегда валидна (части касаются только в вершинах), а площадь полигона ровно
  равна числу пикселей * площадь пикселя.
* Термоточки: центр каждого пикселя маски активного горения.

Все координаты возвращаются в пикселях (col, row) сетки чипа; перевод в UTM
делает вызывающий код (x = x0 + col*gsd, y = y0 - row*gsd).

rasterio и scipy импортируются лениво: работающему сервису они нужны только
для необязательных эндпоинтов /api/v1/analyze/*.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import shape

from .common import DAY_SZA_MAX, to_multipolygon

EIGHT = np.ones((3, 3), dtype=np.int32)


def _polygonize(values: np.ndarray, valid: np.ndarray) -> dict[int, shapely.MultiPolygon]:
    from rasterio import features
    from rasterio.transform import Affine

    groups: dict[int, list] = {}
    for g, v in features.shapes(values.astype(np.int32), mask=valid, transform=Affine.identity(), connectivity=4):
        groups.setdefault(int(v), []).append(shape(g))
    return {k: to_multipolygon(shapely.MultiPolygon(v) if len(v) > 1 else v[0]) for k, v in groups.items()}


def vectorize_severity(
    mask: np.ndarray,
    prob: np.ndarray | None = None,
    landcover: np.ndarray | None = None,
    simplify_px: float = 1.0,
) -> dict:
    """Маска степени (H, W) uint8 0..3 -> контуры гарей и зоны степени.

    Возвращает словарь:
      scars: список {k, n_px, sev_px[3], prob_mean, lc_px{code: [n1, n2, n3]}}
      scar_geoms / scar_disp: точные / упрощённые MultiPolygon (пиксельные коорд.)
      zones: список {scar_k, severity, n_px}
      zone_geoms: точные MultiPolygon зон (пиксельные коорд.)
    Номера контуров k = 1..n упорядочены по убыванию площади.
    """
    from scipy import ndimage

    m = np.asarray(mask).astype(np.uint8)
    m[m > 3] = 0
    burned = m > 0
    lab, n = ndimage.label(burned, structure=EIGHT)
    out = {"scars": [], "scar_geoms": [], "scar_disp": [], "zones": [], "zone_geoms": []}
    if n == 0:
        return out
    cnt = np.bincount(lab.ravel(), minlength=n + 1)
    order = np.argsort(-cnt[1:], kind="stable") + 1  # старые метки по убыванию площади
    remap = np.zeros(n + 1, np.int32)
    remap[order] = np.arange(1, n + 1, dtype=np.int32)
    lab = remap[lab]
    cnt = np.bincount(lab.ravel(), minlength=n + 1)

    sev_px = np.bincount((lab * 4 + m).ravel(), minlength=(n + 1) * 4).reshape(n + 1, 4)
    prob_mean = None
    if prob is not None:
        p = np.nan_to_num(np.asarray(prob, np.float64), nan=0.0)
        prob_mean = np.bincount(lab.ravel(), weights=p.ravel(), minlength=n + 1) / np.maximum(cnt, 1)
    lc_px: list[dict] = [dict() for _ in range(n + 1)]
    if landcover is not None:
        lc = np.asarray(landcover).astype(np.int64)
        sel = burned
        keys = (lab[sel].astype(np.int64) * 1000 + lc[sel]) * 4 + m[sel]
        uk, uc = np.unique(keys, return_counts=True)
        for key, c in zip(uk.tolist(), uc.tolist()):
            sev = key % 4
            rest = key // 4
            code = rest % 1000
            k = rest // 1000
            lc_px[k].setdefault(int(code), [0, 0, 0])[sev - 1] += int(c)

    scar_geoms = _polygonize(lab, burned)
    zone_geoms = _polygonize(lab * 4 + m, burned)

    tol = float(simplify_px)
    for k in range(1, n + 1):
        g = scar_geoms[k]
        disp = shapely.simplify(g, tol, preserve_topology=True) if tol > 0 else g
        out["scars"].append(
            {
                "k": k,
                "n_px": int(cnt[k]),
                "sev_px": [int(sev_px[k, 1]), int(sev_px[k, 2]), int(sev_px[k, 3])],
                "prob_mean": None if prob_mean is None else float(prob_mean[k]),
                "lc_px": {str(c): v for c, v in sorted(lc_px[k].items())},
            }
        )
        out["scar_geoms"].append(g)
        out["scar_disp"].append(to_multipolygon(disp))
        for s in (1, 2, 3):
            if sev_px[k, s] > 0:
                out["zones"].append({"scar_k": k, "severity": s, "n_px": int(sev_px[k, s])})
                out["zone_geoms"].append(zone_geoms[k * 4 + s])
    return out


def extract_hotspots(mask: np.ndarray, viirs: np.ndarray, aux: np.ndarray | None, prob: np.ndarray | None = None) -> dict:
    """Пиксели маски активного горения -> атрибуты термоточек (пиксельные координаты центров)."""
    m = np.asarray(mask) == 1
    rows, cols = np.nonzero(m)

    def band(i):
        if viirs is None or viirs.shape[0] <= i:
            return np.full(len(rows), np.nan)
        v = viirs[i][rows, cols].astype(np.float64)
        return np.where(np.isfinite(v), v, np.nan)

    i4, i5, sza, vza = band(3), band(4), band(5), band(6)
    lc = aux[0][rows, cols].astype(np.int64) if aux is not None and aux.shape[0] > 0 else np.zeros(len(rows), np.int64)
    conf = np.ones(len(rows)) if prob is None else np.asarray(prob, np.float64)[rows, cols]
    daynight = np.where(np.isfinite(sza) & (sza < DAY_SZA_MAX), "D", "N")
    return {
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "i4": i4,
        "i5": i5,
        "sza": sza,
        "vza": vza,
        "lc": lc,
        "conf": np.clip(np.nan_to_num(conf, nan=0.0), 0, 1),
        "daynight": daynight,
    }
