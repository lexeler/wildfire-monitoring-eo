"""Композитная метрика соревнования (микро-усреднение по пулу пикселей).

Score = 0.35 * F1_af + 0.35 * IoU_burn + 0.30 * mIoU_sev
Если знаменатель равен нулю (класса нет ни в эталоне, ни в прогнозе) —
величина равна 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _safe(num: float, den: float) -> float:
    return 1.0 if den == 0 else num / den


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred = np.asarray(pred, bool)
        true = np.asarray(true, bool)
        self.tp += int(np.count_nonzero(pred & true))
        self.fp += int(np.count_nonzero(pred & ~true))
        self.fn += int(np.count_nonzero(~pred & true))

    @property
    def iou(self) -> float:
        return _safe(self.tp, self.tp + self.fp + self.fn)

    @property
    def f1(self) -> float:
        return _safe(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def precision(self) -> float:
        return _safe(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _safe(self.tp, self.tp + self.fn)


@dataclass
class ScoreAccumulator:
    af: Counts = field(default_factory=Counts)
    burn: Counts = field(default_factory=Counts)
    sev: dict = field(default_factory=lambda: {k: Counts() for k in (1, 2, 3)})

    def add_af(self, pred: np.ndarray, true: np.ndarray) -> None:
        self.af.add(pred == 1, true == 1)

    def add_bs(self, pred: np.ndarray, true: np.ndarray) -> None:
        self.burn.add(pred >= 1, true >= 1)
        for k in (1, 2, 3):
            self.sev[k].add(pred == k, true == k)

    def result(self) -> dict:
        miou = float(np.mean([self.sev[k].iou for k in (1, 2, 3)]))
        f1 = self.af.f1
        iou_b = self.burn.iou
        return {
            "score": 0.35 * f1 + 0.35 * iou_b + 0.30 * miou,
            "f1_af": f1,
            "precision_af": self.af.precision,
            "recall_af": self.af.recall,
            "iou_burn": iou_b,
            "miou_sev": miou,
            "iou_sev1": self.sev[1].iou,
            "iou_sev2": self.sev[2].iou,
            "iou_sev3": self.sev[3].iou,
        }


def partial_score(res: dict, part: str) -> float:
    """Вклад одного модуля в Score (для раздельной валидации модулей)."""
    if part == "af":
        return 0.35 * res["f1_af"]
    return 0.35 * res["iou_burn"] + 0.30 * res["miou_sev"]
