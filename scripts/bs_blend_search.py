"""Поиск весов усреднения OOF-вероятностей области и порога (грубая сетка).

Все прогоны загружаются один раз; метрика считается только по допустимым
пикселям (правило ∩ не исключено) — остальные пиксели прогноза всегда 0,
их вклад (FN эталонной гари вне допустимой зоны) — константа.

python scripts/bs_blend_search.py --data-dir data/train --runs runs/A runs/B runs/C [--grid 0 0.5 1 2]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.rules import excluded, output_rule  # noqa: E402
from firemon.io import load_bs, parallel_map  # noqa: E402
from scripts.bs_oof_analysis import burn_prob  # noqa: E402

THRESHOLDS = np.round(np.arange(0.25, 0.56, 0.05), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--grid", nargs="+", type=float, default=[0, 0.5, 1, 2])
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--severity-mode", choices=["legacy", "class_median"], default="legacy")
    ap.add_argument("--fixed-weights", nargs="+", type=float, default=None,
                    help="только перебор порогов по классу правила (t1 t2 t3) при заданных весах")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="сравнить заданные наборы весов (через запятую, по числу --runs), единый порог")
    a = ap.parse_args()
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")

    def prep(cid):
        c = load_bs(a.data_dir, cid, with_mask=True)
        rule = output_rule(c.s2_pre, c.s2_post, c.aux[2], a.severity_mode)
        ok = (rule > 0) & ~excluded(c.s2_pre[9], c.s2_post[9])
        P = []
        for r in a.runs:
            hit = next(Path(r).glob(f"f*/oof/{cid}.npy"))
            P.append(burn_prob(np.load(hit).astype(np.float32) / 255.0)[ok])
        sev = np.clip(rule[ok], 1, 3).astype(np.int64)
        t = c.mask[ok].astype(np.int64)
        # эталон вне допустимой зоны — неизбежные FN (по классам)
        out = c.mask[~ok]
        fn_out = np.array([(out > 0).sum()] + [(out == k).sum() for k in (1, 2, 3)])
        return np.stack(P), sev, t, fn_out

    items = parallel_map(prep, meta.chip_id, workers=8)
    P = np.concatenate([it[0] for it in items], 1)
    sev = np.concatenate([it[1] for it in items])
    t = np.concatenate([it[2] for it in items])
    fn_out = np.sum([it[3] for it in items], 0)
    print("ok pixels", P.shape, flush=True)
    def score(pred):
        cm = np.bincount(t * 4 + pred, minlength=16).reshape(4, 4)
        tp, fp, fn = cm[1:, 1:].sum(), cm[0, 1:].sum(), cm[1:, 0].sum() + fn_out[0]
        iou_b = tp / (tp + fp + fn)
        miou = float(np.mean([cm[k, k] / (cm[:, k].sum() + cm[k, :].sum() - cm[k, k] + fn_out[k]) for k in (1, 2, 3)]))
        return iou_b, miou, 0.35 * iou_b + 0.3 * miou

    if a.configs:
        for cfg in a.configs:
            w = np.asarray([float(x) for x in cfg.split(",")])
            p = (w / w.sum()) @ P
            best = max((score(np.where(p > th, sev, 0)) + (th,) for th in np.round(np.arange(0.3, 0.61, 0.025), 3)), key=lambda r: r[2])
            print(f"{cfg:40s} th {best[3]:.3f} iou {best[0]:.4f} miou {best[1]:.4f} bs_part {best[2]:.4f}", flush=True)
        return

    if a.fixed_weights:
        w = np.asarray(a.fixed_weights)
        p = (w / w.sum()) @ P
        grid_t = np.round(np.arange(0.25, 0.61, 0.05), 3)
        rows = []
        for t1, t2, t3 in itertools.product(grid_t, repeat=3):
            th = np.where(sev == 1, t1, np.where(sev == 2, t2, t3))
            iou_b, miou, bp = score(np.where(p > th, sev, 0))
            rows.append(dict(t1=t1, t2=t2, t3=t3, iou_burn=iou_b, miou=miou, bs_part=bp))
        df = pd.DataFrame(rows).sort_values("bs_part", ascending=False)
        print(df.head(a.top).round(4).to_string(index=False))
        same = df[(df.t1 == df.t2) & (df.t2 == df.t3)].head(3)
        print("single threshold best:\n", same.round(4).to_string(index=False))
        return

    rows = []
    for w in itertools.product(a.grid, repeat=len(a.runs)):
        w = np.asarray(w)
        if w.sum() == 0 or np.gcd.reduce((w * 2).astype(int)) != 1:  # пропуск кратных наборов
            continue
        p = (w / w.sum()) @ P
        for th in THRESHOLDS:
            pred = np.where(p > th, sev, 0)
            cm = np.bincount(t * 4 + pred, minlength=16).reshape(4, 4)  # строки — эталон, столбцы — прогноз
            tp = cm[1:, 1:].sum()
            fp = cm[0, 1:].sum()
            fn = cm[1:, 0].sum() + fn_out[0]
            iou_b = tp / (tp + fp + fn)
            ious = [cm[k, k] / (cm[:, k].sum() + cm[k, :].sum() - cm[k, k] + fn_out[k]) for k in (1, 2, 3)]
            miou = float(np.mean(ious))
            rows.append(dict(w=tuple(w.tolist()), th=th, iou_burn=iou_b, miou=miou, bs_part=0.35 * iou_b + 0.3 * miou))
    df = pd.DataFrame(rows).sort_values("bs_part", ascending=False)
    print(df.head(a.top).round(4).to_string(index=False))
    for i, r in enumerate(a.runs):  # одиночные модели
        sub = df[df.w.apply(lambda w: sum(x > 0 for x in w) == 1 and w[i] > 0)]
        if len(sub):
            print(f"single {r}: {sub.bs_part.max():.4f}")


if __name__ == "__main__":
    main()
