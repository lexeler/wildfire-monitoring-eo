"""Чтение чипов в структуре выданного набора.

<root>/af/viirs/<id>_VIIRS_I1-I5.tif     8 каналов: I1..I5, solar_zenith, sensor_zenith, valid
<root>/af/aux/<id>_AUX.tif               5 каналов: landcover, dem, t2m, rh2m, wind_speed
<root>/af/masks/<id>_MASK.tif            эталон (только train)
<root>/bs/sentinel2_{pre,post}/<id>_Sentinel-2_{pre,post}.tif   B2 B3 B4 B5 B6 B7 B8A B11 B12 SCL
<root>/bs/sentinel1_{pre,post}/<id>_Sentinel-1_{pre,post}.tif   VV VH (дБ x100)
<root>/bs/aux/<id>_AUX.tif               dem, slope, landcover
<root>/bs/masks/<id>_MASK.tif            эталон 0..3 (только train)
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import numpy as np
import rasterio

T = TypeVar("T")

# Порядок каналов Sentinel-2 внутри файлов pre/post
S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")
S2_IDX = {b: i for i, b in enumerate(S2_BANDS)}
SCL_IDX = 9


def read_tif(path: str | os.PathLike) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


@dataclass
class AFChip:
    chip_id: str
    viirs: np.ndarray  # (8, H, W) float32
    aux: np.ndarray  # (5, H, W) float32
    mask: np.ndarray | None = None  # (H, W) uint8


@dataclass
class BSChip:
    chip_id: str
    s2_pre: np.ndarray  # (10, H, W) uint16
    s2_post: np.ndarray  # (10, H, W) uint16
    s1_pre: np.ndarray  # (2, H, W) int16
    s1_post: np.ndarray  # (2, H, W) int16
    aux: np.ndarray  # (3, H, W) int16
    mask: np.ndarray | None = None  # (H, W) uint8


def af_paths(root: Path, cid: str) -> dict:
    return {
        "viirs": root / "af" / "viirs" / f"{cid}_VIIRS_I1-I5.tif",
        "aux": root / "af" / "aux" / f"{cid}_AUX.tif",
        "mask": root / "af" / "masks" / f"{cid}_MASK.tif",
    }


def bs_paths(root: Path, cid: str) -> dict:
    return {
        "s2_pre": root / "bs" / "sentinel2_pre" / f"{cid}_Sentinel-2_pre.tif",
        "s2_post": root / "bs" / "sentinel2_post" / f"{cid}_Sentinel-2_post.tif",
        "s1_pre": root / "bs" / "sentinel1_pre" / f"{cid}_Sentinel-1_pre.tif",
        "s1_post": root / "bs" / "sentinel1_post" / f"{cid}_Sentinel-1_post.tif",
        "aux": root / "bs" / "aux" / f"{cid}_AUX.tif",
        "mask": root / "bs" / "masks" / f"{cid}_MASK.tif",
    }


def load_af(root: str | os.PathLike, cid: str, with_mask: bool = False) -> AFChip:
    p = af_paths(Path(root), cid)
    mask = read_tif(p["mask"])[0] if with_mask and p["mask"].exists() else None
    return AFChip(cid, read_tif(p["viirs"]).astype(np.float32), read_tif(p["aux"]).astype(np.float32), mask)


def load_bs(root: str | os.PathLike, cid: str, with_mask: bool = False) -> BSChip:
    p = bs_paths(Path(root), cid)
    mask = read_tif(p["mask"])[0] if with_mask and p["mask"].exists() else None
    return BSChip(
        cid,
        read_tif(p["s2_pre"]),
        read_tif(p["s2_post"]),
        read_tif(p["s1_pre"]),
        read_tif(p["s1_post"]),
        read_tif(p["aux"]),
        mask,
    )


def parallel_map(fn: Callable[..., T], items: Iterable, workers: int | None = None) -> list[T]:
    """Параллельное чтение: GDAL отпускает GIL, поэтому потоков достаточно."""
    items = list(items)
    workers = workers or min(32, (os.cpu_count() or 4) * 2)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
