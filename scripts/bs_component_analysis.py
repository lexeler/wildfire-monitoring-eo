"""Гипотеза «один пожар на чип»: эталонная область — события, к которому
относится чип; другие гари в кадре не размечены. Проверяем отбор компонент
прогнозной области по «массе» вероятности.

ОТВЕРГНУТЫЙ ЭКСПЕРИМЕНТ: отбор компонент хуже базового варианта (отчёт, 6.3).
inference.py и train.py его не вызывают.

python scripts/bs_component_analysis.py --data-dir data/train --runs runs/A [runs/B]
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
from scripts.bs_oof_analysis import load_oof  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")

    def prep(cid):
        p = load_oof(a.runs, cid)[0]
        c = load_bs(a.data_dir, cid, with_mask=True)
        rule = severity_from_dnbr(dnbr(c.s2_pre, c.s2_post), c.aux[2])
        ok = (rule > 0) & ~excluded(c.s2_pre[9], c.s2_post[9])
        g = c.mask > 0
        # компоненты эталона (на сетке 1/4 с закрытием, как «области»)
        gl, gn = ndi.label(ndi.binary_closing(g[::4, ::4], np.ones((5, 5))), np.ones((3, 3)))
        return p, rule, ok, c.mask, gn

    items = parallel_map(prep, meta.chip_id)
    gns = np.array([x[4] for x in items])
    print("GT region components per chip (1/4 grid, closed): pct", np.percentile(gns, [10, 25, 50, 75, 90]))
    rows = []
    for th in (0.3, 0.35, 0.4):
        for close in (0, 3, 7):
            for alpha in (0.0, 0.05, 0.15, 0.3, 0.5):
                acc = ScoreAccumulator()
                for p, rule, ok, m, _ in items:
                    reg = p > th
                    if alpha > 0:
                        small = reg[::4, ::4]
                        if close:
                            small = ndi.binary_closing(small, np.ones((close, close))) | small
                        lab, n = ndi.label(small, np.ones((3, 3)))
                        if n > 1:
                            mass = ndi.sum((p * ok)[::4, ::4], lab, np.arange(1, n + 1))
                            keep = np.flatnonzero(mass >= alpha * mass.max()) + 1
                            km = np.isin(lab, keep)
                            reg &= np.kron(km, np.ones((4, 4), bool))[: reg.shape[0], : reg.shape[1]]
                    acc.add_bs(np.where(reg & ok, np.clip(rule, 1, 3), 0), m)
                r = acc.result()
                rows.append(dict(th=th, close=close, alpha=alpha, iou_burn=r["iou_burn"], miou=r["miou_sev"],
                                 bs_part=0.35 * r["iou_burn"] + 0.30 * r["miou_sev"]))
    print(pd.DataFrame(rows).sort_values("bs_part", ascending=False).head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
