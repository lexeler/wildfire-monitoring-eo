"""Постобработка «области пожара» на OOF: закрытие диском и заливка дыр.
Эталонная область — объединение кругов радиусом 2-4 км, поэтому мелкие дыры
и разрывы в прогнозе области, вероятно, ложные.

ОТВЕРГНУТЫЙ ЭКСПЕРИМЕНТ: морфология хуже базового варианта (отчёт, 6.3).
inference.py и train.py его не вызывают.

python scripts/bs_morph_analysis.py --data-dir data/train --runs runs/X [runs/Y]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.rules import dnbr, excluded, severity_from_dnbr  # noqa: E402
from firemon.io import load_bs, parallel_map  # noqa: E402
from firemon.metric import ScoreAccumulator  # noqa: E402
from scripts.bs_oof_analysis import burn_prob, load_oof  # noqa: E402


def disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")

    def prep(cid):
        p = load_oof(a.runs, cid)
        if p is None:
            return None
        c = load_bs(a.data_dir, cid, with_mask=True)
        rule = severity_from_dnbr(dnbr(c.s2_pre, c.s2_post), c.aux[2])
        ok = (rule > 0) & ~excluded(c.s2_pre[9], c.s2_post[9])
        return burn_prob(p), rule, ok, c.mask

    items = [x for x in parallel_map(prep, meta.chip_id) if x is not None]
    rows = []
    for th in (0.25, 0.3, 0.35, 0.4, 0.5):
        for r in (0, 4, 8, 16):
            for fill in (False, True):
                acc = ScoreAccumulator()
                for pb, rule, ok, m in items:
                    reg = pb > th
                    if r:
                        # закрытие на сетке 1/4 для скорости
                        small = reg[::4, ::4]
                        small = ndi.binary_closing(small, structure=disk(max(1, r // 4)), border_value=0) | small
                        reg = reg | np.kron(small, np.ones((4, 4), bool))[: reg.shape[0], : reg.shape[1]]
                    if fill:
                        reg = ndi.binary_fill_holes(reg)
                    pred = np.where(reg & ok, np.clip(rule, 1, 3), 0)
                    acc.add_bs(pred, m)
                res = acc.result()
                rows.append(dict(th=th, close_r=r, fill=fill, iou_burn=res["iou_burn"], miou=res["miou_sev"],
                                 bs_part=0.35 * res["iou_burn"] + 0.30 * res["miou_sev"]))
    print(pd.DataFrame(rows).sort_values("bs_part", ascending=False).head(15).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
