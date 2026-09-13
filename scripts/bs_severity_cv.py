"""CV модели степени (firemon.bs.severity) против правила порогов.

ОТВЕРГНУТЫЙ ЭКСПЕРИМЕНТ: +0.0014 к Score ценой ~10 с CPU на прогон — в финале
осталось правило порогов (отчёт, 6.3). inference.py и train.py его не вызывают.

Обучение и оценка — на пикселях эталонной гари (класс 1..3); сравнивается
согласие с эталоном и mIoU классов при идеальном контуре, а также эффект на
итоговой OOF-маске (контур из --region-runs).

python scripts/bs_severity_cv.py --data-dir data/train [--region-runs runs/X ...] [--save-final weights]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.rules import dnbr, excluded, severity_from_dnbr  # noqa: E402
from firemon.bs.severity import severity_features  # noqa: E402
from firemon.io import load_bs, parallel_map  # noqa: E402
from firemon.metric import ScoreAccumulator  # noqa: E402
from scripts.bs_oof_analysis import load_oof  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.1, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=10, deterministic=True, force_row_wise=True)
ROUNDS = 300
MAX_ROWS_PER_CHIP = 20000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--region-runs", nargs="*", default=None)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--save-final", default=None)
    a = ap.parse_args()
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")
    folds = pd.read_csv(REPO / "configs" / "folds_bs.csv").set_index("chip_id").loc[meta.chip_id, "fold"].values
    rng = np.random.default_rng(0)

    def prep(cid):
        c = load_bs(a.data_dir, cid, with_mask=True)
        F, names = severity_features(c.s2_pre, c.s2_post, c.aux[2])
        rule = severity_from_dnbr(dnbr(c.s2_pre, c.s2_post), c.aux[2])
        ok = (rule > 0) & ~excluded(c.s2_pre[9], c.s2_post[9])
        g = c.mask > 0
        ys, xs = np.nonzero(g)
        if len(ys) > MAX_ROWS_PER_CHIP:
            sel = rng.choice(len(ys), MAX_ROWS_PER_CHIP, replace=False)
            ys, xs = ys[sel], xs[sel]
        return F, names, rule, ok, c.mask, F[:, ys, xs].T, c.mask[ys, xs] - 1

    items = parallel_map(prep, meta.chip_id, workers=6)
    names = items[0][1]
    R = np.concatenate([it[5] for it in items])
    Y = np.concatenate([it[6] for it in items])
    C = np.concatenate([np.full(len(it[6]), i) for i, it in enumerate(items)])
    print("rows", R.shape, flush=True)
    boosters = {}
    for f in range(5):
        tr = folds[C] != f
        boosters[f] = lgb.train(dict(PARAMS, seed=f), lgb.Dataset(R[tr], Y[tr], feature_name=names), ROUNDS)
    # согласие на всех пикселях эталонной гари (OOF по чипам)
    acc_rule, acc_model, n = 0, 0, 0
    sev_pred = {}
    for i, (F, _, rule, ok, m, *_r) in enumerate(items):
        g = m > 0
        if not g.any():
            sev_pred[i] = np.clip(rule, 1, 3)
            continue
        b = boosters[folds[i]]
        # предсказываем на всех допустимых пикселях и эталонной гари
        cand = g | ok
        p = np.zeros(m.shape, np.uint8)
        p[cand] = b.predict(F[:, cand].T).argmax(1) + 1
        sev_pred[i] = np.where(cand, p, np.clip(rule, 1, 3))
        acc_rule += int((np.clip(rule[g], 1, 3) == m[g]).sum())
        acc_model += int((p[g] == m[g]).sum())
        n += int(g.sum())
    print(f"agreement inside GT burn: rule {acc_rule / n:.4f} -> model {acc_model / n:.4f}", flush=True)
    res = {"agree_rule": acc_rule / n, "agree_model": acc_model / n}
    for mode in ("rule", "model"):
        accm = ScoreAccumulator()
        for i, (F, _, rule, ok, m, *_r) in enumerate(items):
            sev = np.clip(rule, 1, 3) if mode == "rule" else sev_pred[i]
            accm.add_bs(np.where(m > 0, sev, 0), m)
        r = accm.result()
        res[f"perfect_region_{mode}"] = {"miou": r["miou_sev"], "s1": r["iou_sev1"], "s2": r["iou_sev2"], "s3": r["iou_sev3"]}
        print(mode, "perfect region mIoU", round(r["miou_sev"], 4), flush=True)
    if a.region_runs:
        for mode in ("rule", "model"):
            accm = ScoreAccumulator()
            for i, (F, _, rule, ok, m, *_r) in enumerate(items):
                pb = load_oof(a.region_runs, meta.chip_id[i])[0]
                sev = np.clip(rule, 1, 3) if mode == "rule" else sev_pred[i]
                accm.add_bs(np.where((pb > a.threshold) & ok, sev, 0), m)
            r = accm.result()
            bs_part = 0.35 * r["iou_burn"] + 0.30 * r["miou_sev"]
            res[f"oof_{mode}"] = {"iou_burn": r["iou_burn"], "miou": r["miou_sev"], "bs_part": bs_part}
            print(mode, "OOF region: iou_burn", round(r["iou_burn"], 4), "miou", round(r["miou_sev"], 4), "bs_part", round(bs_part, 4), flush=True)
    if a.save_final:
        b = lgb.train(dict(PARAMS, seed=0), lgb.Dataset(R, Y, feature_name=names), ROUNDS)
        b.save_model(str(Path(a.save_final) / "bs_severity.txt"))
        (Path(a.save_final) / "bs_severity.json").write_text(json.dumps({"features": names, "cv": res}, indent=1))


if __name__ == "__main__":
    main()
