"""Ансамбль AF: OOF LightGBM (npz из firemon.af.train --oof-dir) + OOF сети
(npy из firemon.af.cnn). Перебор веса сети и порога по F1.

ОТВЕРГНУТЫЙ ЭКСПЕРИМЕНТ: прироста над одним LightGBM нет (отчёт, 6.1, A9).
Оставлен для воспроизведения таблицы; inference.py и train.py его не вызывают.

python scripts/af_blend_eval.py --data-dir data/train --lgb-oof runs/af_oof --cnn-runs runs/af_cnn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.metric import Counts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lgb-oof", required=True)
    ap.add_argument("--cnn-runs", required=True)
    a = ap.parse_args()
    meta = pd.read_csv(Path(a.data_dir) / "af" / "meta.csv")
    L, C, M = [], [], []
    for cid in meta.chip_id:
        z = np.load(Path(a.lgb_oof) / f"{cid}.npz")
        L.append(z["prob"].astype(np.float32))
        C.append(np.load(next(Path(a.cnn_runs).glob(f"f*/oof/{cid}.npy"))).astype(np.float32))
        with rasterio.open(Path(a.data_dir) / "af" / "masks" / f"{cid}_MASK.tif") as s:
            M.append(s.read(1) == 1)
    L, C, M = np.stack(L), np.stack(C), np.stack(M)
    cand = L > 0  # кандидаты бустинга (вне их вероятность 0)
    rows = []
    for w in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        for gate in (True, False):
            p = (1 - w) * L + w * C
            if gate:
                p = np.where(cand, p, 0)
            for th in np.arange(0.3, 0.81, 0.025):
                c = Counts()
                c.add(p > th, M)
                rows.append(dict(w_cnn=w, gate=gate, th=round(th, 3), f1=c.f1, p=c.precision, r=c.recall))
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print(df.head(10).round(4).to_string(index=False))
    print("lgb only:", df[(df.w_cnn == 0)].f1.max().round(4), " cnn only:", df[(df.w_cnn == 1) & (~df.gate)].f1.max().round(4))


if __name__ == "__main__":
    main()
