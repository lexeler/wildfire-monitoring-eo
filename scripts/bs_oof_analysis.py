"""Анализ OOF-предсказаний модуля гарей и подбор постобработки.

python scripts/bs_oof_analysis.py --data-dir data/train --runs runs/r34c384 [runs/other ...]
Несколько каталогов — усреднение вероятностей (ансамбль).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.rules import dnbr, excluded, severity_from_dnbr
from firemon.io import load_bs, parallel_map
from firemon.metric import ScoreAccumulator


def burn_prob(prob: np.ndarray) -> np.ndarray:
    """OOF хранит либо softmax 4 класса, либо 1 канал вероятности области."""
    return prob[0] if prob.shape[0] == 1 else 1 - prob[0]


def load_oof(runs: list[str], cid: str, weights: list[float] | None = None) -> np.ndarray | None:
    """Средняя (взвешенная) P(гарь) по нескольким прогонам -> (1, H, W)."""
    ps = []
    for r in runs:
        hits = list(Path(r).glob(f"f*/oof/{cid}.npy"))
        if not hits:
            return None
        ps.append(burn_prob(np.load(hits[0]).astype(np.float32) / 255.0))
    w = np.asarray(weights if weights else [1.0] * len(ps), np.float32)
    return (np.tensordot(w / w.sum(), np.stack(ps), 1))[None]


def assemble(prob: np.ndarray, rule: np.ndarray, scl: tuple, th: float, gate: bool, cloud: str, min_px: int) -> np.ndarray:
    burned = burn_prob(prob) > th
    if gate:
        burned &= rule > 0
    if cloud != "none":
        burned &= ~excluded(scl[0], scl[1], (8,) if cloud == "excl+8" else ())
    if min_px:
        lab, n = ndi.label(burned, structure=np.ones((3, 3)))
        if n:
            sizes = ndi.sum(burned, lab, index=np.arange(1, n + 1))
            burned &= np.isin(lab, np.flatnonzero(sizes >= min_px) + 1)
    sev = np.clip(rule, 1, 3)
    return np.where(burned, sev, 0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    a = ap.parse_args()
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")

    def prep(cid):
        p = load_oof(a.runs, cid, a.weights)
        if p is None:
            return None
        c = load_bs(a.data_dir, cid, with_mask=True)
        rule = severity_from_dnbr(dnbr(c.s2_pre, c.s2_post), c.aux[2])
        return cid, p, rule, (c.s2_pre[9], c.s2_post[9]), c.mask

    items = [x for x in parallel_map(prep, meta.chip_id) if x is not None]
    print(f"chips with OOF: {len(items)}")
    grid = []
    for th in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5):
        for gate in (True,):
            for cloud in ("excl", "excl+8"):
                for min_px in (0,):
                    acc = ScoreAccumulator()
                    for cid, p, rule, scl, m in items:
                        acc.add_bs(assemble(p, rule, scl, th, gate, cloud, min_px), m)
                    r = acc.result()
                    grid.append(dict(th=th, gate=gate, cloud=cloud, min_px=min_px, iou_burn=r["iou_burn"], miou_sev=r["miou_sev"],
                                     s1=r["iou_sev1"], s2=r["iou_sev2"], s3=r["iou_sev3"], bs_part=0.35 * r["iou_burn"] + 0.30 * r["miou_sev"]))
    g = pd.DataFrame(grid).sort_values("bs_part", ascending=False)
    print(g.head(12).round(4).to_string(index=False))
    best = g.iloc[0]
    # поштучный разбор при лучшей настройке
    rows = []
    for cid, p, rule, scl, m in items:
        pred = assemble(p, rule, scl, best.th, bool(best.gate), str(best.cloud), int(best.min_px))
        pb, tb = pred > 0, m > 0
        rows.append(dict(chip_id=cid, true_px=int(tb.sum()), pred_px=int(pb.sum()), tp=int((pb & tb).sum()),
                         fp=int((pb & ~tb).sum()), fn=int((~pb & tb).sum())))
    pc = pd.DataFrame(rows)
    pc["iou"] = pc.tp / (pc.tp + pc.fp + pc.fn).clip(lower=1)
    print("total fp", pc.fp.sum(), "fn", pc.fn.sum(), "tp", pc.tp.sum())
    print("worst by fp+fn:")
    print(pc.assign(err=pc.fp + pc.fn).sort_values("err", ascending=False).head(12).to_string(index=False))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        g.to_csv(a.out, index=False)
        pc.to_csv(str(a.out).replace(".csv", "_perchip.csv"), index=False)
        print(json.dumps(best.to_dict(), default=float))


if __name__ == "__main__":
    main()
