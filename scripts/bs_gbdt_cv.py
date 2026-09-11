"""5-fold CV GBDT-модели области пожара (firemon.bs.gbdt), OOF в формате сетей.

python scripts/bs_gbdt_cv.py --data-dir data/train --out runs/gbdt_v1 [--save-final weights]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.gbdt import block_target, chip_features, predict_blocks, stack_features, upsample  # noqa: E402
from scripts.bs_oof_analysis import load_oof  # noqa: E402
from firemon.io import load_bs, parallel_map  # noqa: E402
from firemon.metric import ScoreAccumulator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PARAMS = dict(objective="cross_entropy", learning_rate=0.03, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.6,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, deterministic=True, force_row_wise=True)
ROUNDS = 800


def _sparse_from_full(full: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Прореженная сетка из полной карты OOF: predict_blocks с «предсказанием» из таблицы."""
    flat_idx = np.arange(full.size).reshape(full.shape)
    Xi = np.concatenate([X[:1], flat_idx[None].astype(np.float32)])  # канал 0 — ok, канал 1 — индекс блока
    return predict_blocks(lambda rows: full.ravel()[rows[:, 1].astype(np.int64)], Xi, sparse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-final", default=None, help="каталог для итоговой модели на всём train")
    ap.add_argument("--name", default="bs_gbdt_s1.txt", help="имя файла итоговой модели")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--pos", action="store_true", help="признаки положения блока в чипе")
    ap.add_argument("--param", nargs="*", default=[], help="переопределение параметров LightGBM: ключ=значение (перебор гиперпараметров)")
    ap.add_argument("--sparse", action="store_true", help="OOF как в инференсе с gbdt.sparse: деревья для половины блоков (шахматка)")
    ap.add_argument("--hot", action="store_true", help="признаки «видимый огонь» на снимках Sentinel-2")
    ap.add_argument("--soft-dir", default=None, help="мягкие цели учителя (scripts/bs_make_soft_targets.py)")
    ap.add_argument("--soft-alpha", type=float, default=0.5, help="доля эталона в мягкой цели")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--stack-runs", nargs="*", default=None, help="OOF первой ступени для автоконтекста")
    ap.add_argument("--lr", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--threads", type=int, default=PARAMS["num_threads"])
    a = ap.parse_args()
    PARAMS.update(learning_rate=a.lr, num_leaves=a.leaves, num_threads=a.threads)
    for kv in a.param:
        k, v = kv.split("=", 1)
        PARAMS[k] = json.loads(v) if v[0] in "0123456789-.tf[\"" else v
    t0 = time.time()
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")
    folds = pd.read_csv(REPO / "configs" / "folds_bs.csv").set_index("chip_id").loc[meta.chip_id, "fold"].values

    def prep(cid):
        c = load_bs(a.data_dir, cid, with_mask=True)
        X, names, full = chip_features(c, a.block, pos=a.pos, hot=a.hot)
        if a.stack_runs:
            S, snames = stack_features(load_oof(a.stack_runs, cid)[0], a.block)
            X, names = np.concatenate([X, S]), names + snames
        y, w = block_target(c.mask, full["ok"], a.block)
        if a.soft_dir:  # дистилляция: цель = alpha·эталон + (1 - alpha)·P учителя (среднее по допустимым пикселям блока)
            soft = np.load(Path(a.soft_dir) / f"{cid}.npy").astype(np.float32) / 255.0
            okf = full["ok"].astype(np.float32)
            bm = lambda x: x.reshape(x.shape[0] // a.block, a.block, x.shape[1] // a.block, a.block).mean((1, 3))
            ys = bm(soft * okf) / np.maximum(bm(okf), 1e-6)
            y = a.soft_alpha * y + (1.0 - a.soft_alpha) * ys
        return X, names, y, w, full["rule"], full["ok"], c.mask

    items = parallel_map(prep, meta.chip_id, workers=8)
    names = items[0][1]
    print(f"features {len(names)} in {time.time() - t0:.0f}s", flush=True)
    rows, ys, ws, chip_of = [], [], [], []
    for i, (X, _, y, w, *_rest) in enumerate(items):
        sel = w > 0
        rows.append(X[:, sel].T)
        ys.append(y[sel])
        ws.append(w[sel])
        chip_of.append(np.full(sel.sum(), i))
    R, Y, W, C = np.concatenate(rows), np.concatenate(ys), np.concatenate(ws), np.concatenate(chip_of)
    print("rows", R.shape, flush=True)
    oof_blocks = np.zeros(len(R), np.float32)
    for f in range(5):
        tr, va = folds[C] != f, folds[C] == f
        b = lgb.train(dict(PARAMS, seed=f), lgb.Dataset(R[tr], Y[tr], weight=W[tr], feature_name=names), a.rounds)
        oof_blocks[va] = b.predict(R[va])
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    results = {}
    maps = {}
    for i, (X, _, y, w, rule, ok, m) in enumerate(items):
        sel = w > 0
        grid = np.zeros(w.shape, np.float32)
        grid[sel] = oof_blocks[C == i]
        if a.sparse:  # та же прореженная сетка, что в инференсе (значения берутся из уже посчитанных OOF)
            lut = grid.copy()
            grid = _sparse_from_full(lut, X)
        p = upsample(grid, m.shape)
        maps[i] = p
        d = out / f"f{folds[i]}" / "oof"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{meta.chip_id[i]}.npy", (p[None] * 255).round().astype(np.uint8))
    for th in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        acc = ScoreAccumulator()
        for i, (X, _, y, w, rule, ok, m) in enumerate(items):
            acc.add_bs(np.where((maps[i] > th) & ok, np.clip(rule, 1, 3), 0), m)
        r = acc.result()
        results[th] = {"iou_burn": r["iou_burn"], "miou_sev": r["miou_sev"], "bs_part": 0.35 * r["iou_burn"] + 0.30 * r["miou_sev"]}
        print(th, {k: round(v, 4) for k, v in results[th].items()}, flush=True)
    (out / "cv.json").write_text(json.dumps(results, indent=1))
    if a.save_final:
        b = lgb.train(dict(PARAMS, seed=0), lgb.Dataset(R, Y, weight=W, feature_name=names), a.rounds)
        Path(a.save_final).mkdir(parents=True, exist_ok=True)
        b.save_model(str(Path(a.save_final) / a.name))
        meta_out = {"block": a.block, "rounds": a.rounds, "params": PARAMS, "features": names, "stack_runs": a.stack_runs,
                    "cv": results}
        (Path(a.save_final) / (Path(a.name).stem + ".json")).write_text(json.dumps(meta_out, indent=1, default=str))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
