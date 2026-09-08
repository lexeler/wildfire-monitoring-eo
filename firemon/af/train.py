"""Обучение модуля детекции активного горения.

python -m firemon.af.train --data-dir data/train --out weights [--oof-dir runs/af_oof]

1. Признаки и кандидаты для всех чипов train.
2. OOF по пространственным фолдам (configs/folds_af.csv) -> подбор порога по F1.
3. Финальные модели: по одной на фолд-разбиение не нужны — обучаем N_SEEDS
   моделей на всём train с разными seed и усредняем (стабильнее одной).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from ..io import load_af, parallel_map
from ..metric import Counts
from ..utils import seed_everything
from .features import CAND_DAY, CAND_NIGHT
from .model import candidate_rows

REPO = Path(__file__).resolve().parents[2]

PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=0,
    deterministic=True,
    force_row_wise=True,
)
N_ROUNDS = 600
N_SEEDS = 3


def train_booster(X, y, names, seed: int, rounds: int = N_ROUNDS) -> lgb.Booster:
    """y — 0/1 либо мягкая цель из [0, 1] (дистилляция, objective cross_entropy)."""
    params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    if y.dtype != bool:
        params["objective"] = "cross_entropy"
    return lgb.train(params, lgb.Dataset(X, y, feature_name=names, free_raw_data=False), num_boost_round=rounds)


def f1_at(prob_rows: np.ndarray, y_rows: np.ndarray, total_pos: int, th: float) -> tuple[float, float, float]:
    pr = prob_rows > th
    tp = int((pr & y_rows).sum())
    fp = int((pr & ~y_rows).sum())
    fn = total_pos - tp
    c = Counts(tp, fp, fn)
    return c.f1, c.precision, c.recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=str(REPO / "weights"))
    ap.add_argument("--folds-file", default=str(REPO / "configs" / "folds_af.csv"))
    ap.add_argument("--oof-dir", default=None, help="куда сохранить OOF-вероятности по чипам (npz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--exact", action="store_true", help="признаки точного контекстного теста VNP14IMG (features.exact_context_rows)")
    ap.add_argument("--soft-oof", default=None, help="каталог OOF-вероятностей учителя (<chip>.npz, prob): мягкая цель")
    ap.add_argument("--soft-alpha", type=float, default=0.5, help="доля эталона в мягкой цели")
    a = ap.parse_args()
    seed_everything(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    meta = pd.read_csv(Path(a.data_dir) / "af" / "meta.csv")
    chips = parallel_map(lambda c: load_af(a.data_dir, c, with_mask=True), meta.chip_id)
    V = np.stack([c.viirs for c in chips])
    A = np.stack([c.aux for c in chips])
    M = np.stack([c.mask for c in chips]) == 1
    X, idx, names = candidate_rows(V, A, dev, exact=a.exact)
    y = M[idx[:, 0], idx[:, 1], idx[:, 2]]
    total_pos = int(M.sum())
    print(f"chips {len(meta)}, fire px {total_pos}, candidates {len(X)} (recall {y.sum() / total_pos:.4f}), features {len(names)}, {time.time() - t0:.0f}s", flush=True)

    folds = pd.read_csv(a.folds_file).set_index("chip_id").loc[meta.chip_id, "fold"].values
    row_fold = folds[idx[:, 0]]
    y_fit = y
    if a.soft_oof:  # цель = alpha·эталон + (1 - alpha)·OOF-вероятность учителя (шум эталона на краях кластеров)
        P = np.stack([np.load(Path(a.soft_oof) / f"{c}.npz")["prob"].astype(np.float32) for c in meta.chip_id])
        y_fit = (a.soft_alpha * y + (1.0 - a.soft_alpha) * P[idx[:, 0], idx[:, 1], idx[:, 2]]).astype(np.float32)
    oof = np.zeros(len(X), np.float32)
    for f in np.unique(folds):
        tr, va = row_fold != f, row_fold == f
        b = train_booster(X[tr], y_fit[tr], names, a.seed + int(f))
        oof[va] = b.predict(X[va])
        pos_f = int(M[folds == f].sum())
        print(f"fold {f}: F1@0.5={f1_at(oof[va], y[va], pos_f, 0.5)[0]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    ths = np.round(np.arange(0.2, 0.81, 0.025), 3)
    scores = [f1_at(oof, y, total_pos, t) for t in ths]
    best_i = int(np.argmax([s[0] for s in scores]))
    # берём середину плато (в пределах 0.001 от максимума) — устойчивее к сдвигу
    plateau = [t for t, s in zip(ths, scores) if s[0] >= scores[best_i][0] - 0.001]
    th = float(np.median(plateau))
    f1, p, r = f1_at(oof, y, total_pos, th)
    report = {"oof_f1": f1, "oof_precision": p, "oof_recall": r, "threshold": th, "candidate_recall": float(y.sum() / total_pos),
              "per_fold_f1": {}, "curve": {str(t): s[0] for t, s in zip(ths, scores)}}
    for f in np.unique(folds):
        va = row_fold == f
        report["per_fold_f1"][int(f)] = f1_at(oof[va], y[va], int(M[folds == f].sum()), th)[0]
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=1), flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "af_cv_report.json").write_text(json.dumps(report, indent=1))
    if a.oof_dir:
        od = Path(a.oof_dir)
        od.mkdir(parents=True, exist_ok=True)
        prob = np.zeros(M.shape, np.float32)
        prob[idx[:, 0], idx[:, 1], idx[:, 2]] = oof
        for i, cid in enumerate(meta.chip_id):
            np.savez_compressed(od / f"{cid}.npz", mask=(prob[i] > th).astype(np.uint8), prob=prob[i].astype(np.float16))

    if not a.skip_final:
        files = []
        for s in range(N_SEEDS):
            b = train_booster(X, y_fit, names, a.seed + 100 + s)
            fn = f"af_lgbm_s{s}.txt"
            b.save_model(str(out / fn))
            files.append(fn)
        cfg = {"model_files": files, "threshold": th, "feature_names": names, "n_rounds": N_ROUNDS, "params": PARAMS,
               "candidates": {"day": CAND_DAY, "night": CAND_NIGHT}}
        (out / "af_config.json").write_text(json.dumps(cfg, indent=1))
        print(f"saved final models to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
