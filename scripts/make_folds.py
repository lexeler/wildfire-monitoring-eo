"""Схема валидации: фолды по пространственным кластерам.

Тестовые выборки отличаются от обучающей территорией (и временем), поэтому
случайное разбиение по чипам завышает качество: соседние/повторные снимки
одной территории (особенно одни и те же тайлы AF в разные даты) попадают и в
обучение, и в валидацию. Центры чипов переводятся в WGS84, кластеризуются
KMeans на N_CLUSTERS пространственных блоков, блоки раскладываются по 5 фолдам
(GroupKFold). Результат фиксируется в configs/folds_{af,bs}.csv.

Запуск: python scripts/make_folds.py --data-dir /path/to/train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

N_FOLDS = 5
N_CLUSTERS = 15
SEED = 42


def centers_lonlat(meta: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(meta), 2))
    for epsg, idx in meta.groupby("epsg").groups.items():
        tr = Transformer.from_crs(int(epsg), 4326, always_xy=True)
        sub = meta.loc[idx]
        cx = (sub.x_min + sub.x_max) / 2
        cy = (sub.y_min + sub.y_max) / 2
        lon, lat = tr.transform(cx.values, cy.values)
        out[meta.index.get_indexer(idx)] = np.c_[lon, lat]
    return out


def make(meta: pd.DataFrame, kind: str) -> pd.DataFrame:
    ll = centers_lonlat(meta)
    xy = np.c_[ll[:, 0] * np.cos(np.deg2rad(ll[:, 1].mean())), ll[:, 1]]
    cl = KMeans(N_CLUSTERS, random_state=SEED, n_init=10).fit_predict(xy)
    folds = np.zeros(len(meta), int)
    for f, (_, va) in enumerate(GroupKFold(N_FOLDS).split(xy, groups=cl)):
        folds[va] = f
    res = pd.DataFrame({"chip_id": meta.chip_id, "cluster": cl, "fold": folds, "lon": ll[:, 0], "lat": ll[:, 1]})
    date_col = "acq_datetime" if kind == "af" else "date_post"
    res["year"] = pd.to_datetime(meta[date_col], utc=True).dt.year.values
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "configs"))
    a = ap.parse_args()
    root = Path(a.data_dir)
    for kind in ("af", "bs"):
        meta = pd.read_csv(root / kind / "meta.csv")
        res = make(meta, kind)
        res.to_csv(Path(a.out_dir) / f"folds_{kind}.csv", index=False)
        print(kind, res.groupby("fold").size().to_dict())


if __name__ == "__main__":
    main()
