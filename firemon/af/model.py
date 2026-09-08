"""Модель детекции активного горения: LightGBM по пикселям-кандидатам.

Пиксели, не прошедшие мягкий предварительный фильтр (features.CAND_*),
считаются фоном. Для кандидатов бустинг оценивает вероятность «природное
горение»; финальная маска — порог, подобранный по F1 на OOF.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from .features import EXACT_NAMES, compute_features, exact_context_rows

# Чипов за один проход признаков: ограничивает пиковую память GPU/CPU
FEATURE_BATCH = 64


@dataclass
class AFModel:
    boosters: list  # список lgb.Booster (усредняются)
    threshold: float
    feature_names: list[str]

    @classmethod
    def load(cls, weights_dir: str | Path) -> "AFModel":
        weights_dir = Path(weights_dir)
        cfg = json.loads((weights_dir / "af_config.json").read_text())
        boosters = [lgb.Booster(model_file=str(weights_dir / f)) for f in cfg["model_files"]]
        return cls(boosters, float(cfg["threshold"]), cfg["feature_names"])

    def predict_proba_rows(self, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, np.float32)
        p = np.mean([b.predict(X, num_threads=num_threads) for b in self.boosters], 0)
        return p.astype(np.float32)


def _rows_for(viirs, aux, device, exact: bool = False):
    with torch.no_grad():
        X, cand, names = compute_features(viirs, aux, device=device)
        ci, yy, xx = torch.nonzero(cand, as_tuple=True)
        r = X[ci, :, yy, xx]
        if exact:  # точный контекст VNP14IMG (динамическое окно, MAD) — только для кандидатов
            r = torch.cat([r, exact_context_rows(viirs, aux, ci, yy, xx, device=device)], 1)
            names = names + EXACT_NAMES
    return r.float().cpu().numpy(), np.stack([ci.cpu().numpy(), yy.cpu().numpy(), xx.cpu().numpy()], 1), names


def candidate_rows(viirs: np.ndarray, aux: np.ndarray, device: str, exact: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """-> (строки признаков кандидатов (M, F) float32, индексы (M, 3): чип, y, x, имена признаков).
    При нехватке памяти GPU батч делится пополам, в пределе — расчёт на CPU."""
    rows, idx, names = [], [], None
    s, bs = 0, FEATURE_BATCH
    while s < len(viirs):
        try:
            r, ix, names = _rows_for(viirs[s : s + bs], aux[s : s + bs], device, exact)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                bs //= 2
                continue
            r, ix, names = _rows_for(viirs[s : s + 1], aux[s : s + 1], "cpu", exact)
        ix[:, 0] += s
        rows.append(r)
        idx.append(ix)
        s += len(viirs[s : s + bs])
    return np.concatenate(rows), np.concatenate(idx), names


def predict_af(model: AFModel, viirs: np.ndarray, aux: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """viirs (N, 8, H, W), aux (N, 5, H, W) -> (маски uint8 (N, H, W), вероятности float32 (N, H, W))."""
    n, _, h, w = viirs.shape
    X, idx, names = candidate_rows(viirs, aux, device, exact=EXACT_NAMES[0] in model.feature_names)
    assert names == model.feature_names, "feature set mismatch with trained model"
    p = model.predict_proba_rows(X)
    prob = np.zeros((n, h, w), np.float32)
    prob[idx[:, 0], idx[:, 1], idx[:, 2]] = p
    return (prob > model.threshold).astype(np.uint8), prob
