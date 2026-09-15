"""Экспорт out-of-fold предсказаний по train в формат каталога сервиса.

Для сервиса нужны прогнозы модели по georeferenced-сценам (train). Чтобы карта
и справка честно показывали качество, используются OOF-прогнозы: каждый чип
предсказан моделью, не видевшей его при обучении.

python scripts/export_oof_preds.py --data-dir data/train --bs-runs runs/A runs/B \
    --bs-threshold 0.3 --af-oof runs/af_oof --out runs/service_preds
-> <out>/af/<chip>.npz (mask, prob), <out>/bs/<chip>.npz (mask 0..3, prob)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.rules import excluded, output_rule, sieve_small  # noqa: E402
from firemon.io import load_bs, parallel_map  # noqa: E402
from scripts.bs_oof_analysis import load_oof  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bs-runs", nargs="+", required=True)
    ap.add_argument("--bs-weights", nargs="+", type=float, default=None)
    ap.add_argument("--bs-threshold", type=float, required=True)
    ap.add_argument("--bs-severity-mode", choices=["legacy", "class_median"], default="legacy")
    ap.add_argument("--bs-min-area", type=int, default=0, help="минимальная площадь объекта, px (как min_area_px в bs_config.json)")
    ap.add_argument("--af-oof", required=True, help="каталог npz из firemon.af.train --oof-dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "bs").mkdir(parents=True, exist_ok=True)
    (out / "af").mkdir(parents=True, exist_ok=True)
    for f in Path(a.af_oof).glob("*.npz"):
        shutil.copy(f, out / "af" / f.name)
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")

    def one(cid):
        p = load_oof(a.bs_runs, cid, a.bs_weights)[0]
        c = load_bs(a.data_dir, cid)
        rule = output_rule(c.s2_pre, c.s2_post, c.aux[2], a.bs_severity_mode)
        ok = (rule > 0) & ~excluded(c.s2_pre[9], c.s2_post[9])
        mask = np.where(sieve_small((p > a.bs_threshold) & ok, a.bs_min_area), np.clip(rule, 1, 3), 0).astype(np.uint8)
        np.savez_compressed(out / "bs" / f"{cid}.npz", mask=mask, prob=p.astype(np.float16))
        return int((mask > 0).sum())

    n = parallel_map(one, meta.chip_id, workers=8)
    print(f"bs: {len(n)} chips, burned px {sum(n)}; af: {len(list((out / 'af').glob('*.npz')))} chips")


if __name__ == "__main__":
    main()
