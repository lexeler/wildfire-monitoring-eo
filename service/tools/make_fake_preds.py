#!/usr/bin/env python
"""Синтетические «предсказания модели» для отладки сервиса (НЕ результаты модели).

Берёт эталонные маски и портит их: пропуски/ложные срабатывания для термоточек,
сглаживание, сдвиг степени и выпадение части гарей для BS. Формат совпадает с
реальными предсказаниями: <out>/{af,bs}/<chip_id>.npz с массивами mask, prob.

    python service/tools/make_fake_preds.py --data-dir /path/to/train --out /tmp/fake_preds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.data_dir), Path(a.out)
    rng = np.random.default_rng(a.seed)
    (out / "af").mkdir(parents=True, exist_ok=True)
    (out / "bs").mkdir(parents=True, exist_ok=True)

    for cid in pd.read_csv(root / "af" / "meta.csv").chip_id:
        with rasterio.open(root / "af" / "masks" / f"{cid}_MASK.tif") as s:
            ref = s.read(1) == 1
        keep = ref & (rng.random(ref.shape) > 0.12)
        ring = ndimage.binary_dilation(ref, iterations=1) & ~ref
        fp = ring & (rng.random(ref.shape) < 0.06)
        m = keep | fp
        prob = np.where(m, rng.uniform(0.5, 0.99, ref.shape), rng.uniform(0, 0.3, ref.shape) * ring)
        np.savez_compressed(out / "af" / f"{cid}.npz", mask=m.astype(np.uint8), prob=prob.astype(np.float16))

    for cid in pd.read_csv(root / "bs" / "meta.csv").chip_id:
        with rasterio.open(root / "bs" / "masks" / f"{cid}_MASK.tif") as s:
            ref = s.read(1).astype(np.uint8)
        sev = ndimage.median_filter(ref, size=5)
        burned = ndimage.binary_opening(sev > 0, iterations=1)
        if rng.random() < 0.5:
            burned = ndimage.binary_dilation(burned, iterations=1)
        lab, n = ndimage.label(burned)
        if n > 3:  # часть мелких гарей «пропущена»
            drop = rng.choice(np.arange(1, n + 1), size=max(1, n // 8), replace=False)
            burned &= ~np.isin(lab, drop)
        shift = ndimage.zoom(rng.integers(-1, 2, (8, 8)), ref.shape[0] / 8, order=0)[: ref.shape[0], : ref.shape[1]]
        sev = np.where(burned, np.clip(np.maximum(sev, 1) + (shift * (rng.random() < 0.5)), 1, 3), 0).astype(np.uint8)
        prob = ndimage.gaussian_filter(burned.astype(np.float32), 2) * 0.9 + rng.uniform(0, 0.1, ref.shape)
        np.savez_compressed(out / "bs" / f"{cid}.npz", mask=sev, prob=np.clip(prob, 0, 1).astype(np.float16))
    print("готово:", out)


if __name__ == "__main__":
    main()
