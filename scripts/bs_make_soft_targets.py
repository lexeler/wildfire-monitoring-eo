"""Мягкие цели для дистилляции: OOF-вероятности области пожара ансамбля-учителя.

ЭКСПЕРИМЕНТ (в финал не вошёл: вложенная CV показала, что прирост — утечка через
учителя, см. reports/experiments_log.md). Для каждого чипа train берётся прогноз моделей, которые этот чип не видели
(5-fold CV), и усредняется с весами бленда. Сети-ученики обучаются на цели
alpha·эталон + (1 - alpha)·P_учителя (firemon/bs/train.py, --soft-dir).

Зачем: граница области события в эталоне во многом случайна (дуги буфера
вокруг термоточек, которых на снимке не видно). Жёсткая разметка заставляет
сеть заучивать этот шум; усреднённый прогноз ансамбля на чипах, которых он не
видел, — оценка «ожидаемой» разметки, на ней ученик обобщает лучше.

python scripts/bs_make_soft_targets.py --data-dir data/train --out runs/soft \
    --runs runs/cv/gbdt_s1 runs/cv/ctx2_s2 ... --weights 1 0.5 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bs_oof_analysis import burn_prob  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="каталоги CV-прогонов учителей (f*/oof/<chip>.npy)")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="веса усреднения (по числу --runs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true", help="пропускать чипы без OOF (вложенная проверка: учителя без одного фолда)")
    a = ap.parse_args()
    w = np.asarray(a.weights if a.weights else [1.0] * len(a.runs), np.float64)
    if len(w) != len(a.runs):
        raise SystemExit("--weights: нужно по одному весу на каждый прогон")
    w = w / w.sum()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")
    for cid in meta.chip_id:
        probs = []
        hits = [next(Path(r).glob(f"f*/oof/{cid}.npy"), None) for r in a.runs]
        if any(h is None for h in hits):
            if a.allow_missing:
                continue
            raise SystemExit(f"нет OOF для {cid}")
        for hit in hits:
            probs.append(burn_prob(np.load(hit).astype(np.float32) / 255.0))
        p = np.tensordot(w, np.stack(probs), 1)
        np.save(out / f"{cid}.npy", (p * 255).round().astype(np.uint8))
    print(f"soft targets: {len(meta)} chips -> {out}")


if __name__ == "__main__":
    main()
