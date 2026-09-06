"""Отправная точка (baseline): классические индексные/пороговые методы без
обучения. Организаторы baseline не выдали, поэтому за точку отсчёта взяты
стандартные подходы, которые описаны в постановке:

AF  — фиксированные пороги VNP14IMG (день I4>325 K и I4-I5>25 K, ночь I4>295 K
      и I4-I5>10 K) и их вариант с контекстными тестами (окно 11x11).
BS  — B0: единая шкала USGS по сырому dNBR на весь чип;
      B1: пороги по типу покрова (рис. 6 постановки);
      B2: B1 + медиана 3x3 по dNBR;
      B3: B2 + исключение облаков/теней/воды по SCL.
Метрика считается на всём train (модели здесь не обучаются).

    python scripts/baselines.py --data-dir data/train --out reports/baselines.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.af.features import compute_features  # noqa: E402
from firemon.bs.rules import THRESHOLDS_STATEMENT, dnbr, excluded, severity_from_dnbr  # noqa: E402
from firemon.io import load_af, load_bs, parallel_map  # noqa: E402
from firemon.metric import Counts, ScoreAccumulator  # noqa: E402


def af_baselines(data_dir: str) -> dict:
    meta = pd.read_csv(Path(data_dir) / "af" / "meta.csv")
    chips = parallel_map(lambda c: load_af(data_dir, c, with_mask=True), meta.chip_id)
    res = {"fixed": Counts(), "fixed+context": Counts()}
    for s in range(0, len(chips), 64):
        part = chips[s : s + 64]
        V = np.stack([c.viirs for c in part])
        A = np.stack([c.aux for c in part])
        M = np.stack([c.mask for c in part]) == 1
        with torch.no_grad():
            X, _, names = compute_features(V, A, "cpu")
        f = {n: X[:, i].numpy() for i, n in enumerate(names)}
        day = f["day"] > 0
        fixed = np.where(day, (f["I4"] > 325) & (f["dT"] > 25), (f["I4"] > 295) & (f["dT"] > 10)) & (f["valid"] > 0)
        ctx = (f["dT_exc_w11"] > 3.5 * f["dT_sd_w11"]) & (f["dT_exc_w11"] > 6) & (f["I4_exc_w11"] > 3 * f["I4_sd_w11"])
        res["fixed"].add(fixed, M)
        res["fixed+context"].add(fixed & ctx & (f["cloud"] < 0.5), M)
    return {k: {"f1": c.f1, "precision": c.precision, "recall": c.recall} for k, c in res.items()}


def bs_baselines(data_dir: str) -> dict:
    meta = pd.read_csv(Path(data_dir) / "bs" / "meta.csv")
    variants = {k: ScoreAccumulator() for k in ("B0_usgs_raw", "B1_landcover_raw", "B2_landcover_median3", "B3_+scl_exclusion")}

    def one(cid):
        c = load_bs(data_dir, cid, with_mask=True)
        d_raw = dnbr(c.s2_pre, c.s2_post, median=0)
        d_med = dnbr(c.s2_pre, c.s2_post, median=3)
        usgs = np.digitize(d_raw, THRESHOLDS_STATEMENT["forest"]).astype(np.uint8)
        b1 = severity_from_dnbr(d_raw, c.aux[2], THRESHOLDS_STATEMENT)
        b2 = severity_from_dnbr(d_med, c.aux[2], THRESHOLDS_STATEMENT)
        b3 = np.where(excluded(c.s2_pre[9], c.s2_post[9]), 0, b2)
        return c.mask, (usgs, b1, b2, b3)

    for m, preds in parallel_map(one, meta.chip_id, workers=8):
        for acc, p in zip(variants.values(), preds):
            acc.add_bs(p, m)
    out = {}
    for k, acc in variants.items():
        r = acc.result()
        out[k] = {x: r[x] for x in ("iou_burn", "miou_sev", "iou_sev1", "iou_sev2", "iou_sev3")}
        out[k]["bs_part"] = 0.35 * r["iou_burn"] + 0.30 * r["miou_sev"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="reports/baselines.json")
    a = ap.parse_args()
    res = {"af": af_baselines(a.data_dir), "bs": bs_baselines(a.data_dir)}
    best_af = max(v["f1"] for v in res["af"].values())
    res["score_fixed_af_plus_B0"] = 0.35 * res["af"]["fixed"]["f1"] + res["bs"]["B0_usgs_raw"]["bs_part"]
    res["score_best_classic"] = 0.35 * best_af + max(v["bs_part"] for v in res["bs"].values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
