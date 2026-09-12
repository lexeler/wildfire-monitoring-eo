"""Инференс модуля гарей: сети + GBDT-ступени + предметная сборка маски.

Схема (параметры — weights/bs_config.json, подобраны на OOF):
  1. Сети (U-Net, цель «область пожара»): P_cnn по группам моделей
     (группа = одна конфигурация сети), TTA.
  2. GBDT на сетке блоков 40 м (явный многомасштабный контекст): P1.
  3. P = взвешенное среднее источников [P1, P_cnn(группа 1), ...]
     (веса и порог подобраны на OOF, scripts/bs_blend_search.py).
  4. (опционально, final="s2") GBDT второй ступени по контексту P —
     на OOF с сильными сетями не дал прироста, в финале не используется.
  5. гарь = P_final > threshold ∩ (dNBR выше порога «не горело» своего
     покрова) ∖ (SCL «до»/«после»: нет данных, тень, вода, облако, снег);
  6. степень = класс по порогам dNBR (рис. 6 постановки, лес/болота
     откалиброваны), не ниже 1.
Без GBDT (нет в конфиге) итог = среднее сетей.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..io import BSChip
from .features import channel_index, compute_features, output_class_median
from .gbdt import chip_features, predict_blocks, stack_features, upsample
from .model import build_model
from .rules import sieve_small
from .train import exclusion, predict_probs


_INIT_FUNCS = ("kaiming_normal_", "kaiming_uniform_", "xavier_normal_", "xavier_uniform_", "normal_", "uniform_",
               "trunc_normal_", "constant_", "zeros_", "ones_")


@contextlib.contextmanager
def skip_weight_init():
    """Инициализация весов при создании модели — пустая трата времени, если сразу
    за ней идёт load_state_dict(strict=True): на время сборки функции
    torch.nn.init заменяются на «ничего не делать». На результат не влияет."""
    saved = {n: getattr(torch.nn.init, n) for n in _INIT_FUNCS}
    try:
        for n in _INIT_FUNCS:
            setattr(torch.nn.init, n, lambda tensor, *a, **k: tensor)
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)


@dataclass
class BSConfig:
    cnn: list[dict]  # [{"file": ..., "group": ...}]
    threshold: float = 0.3
    gate: bool = True
    cloud: bool = True
    tta: int = 4
    gbdt: dict | None = None  # {"block": 2, "s1": [...], "s2": [...]}
    blend_sources: list[str] = field(default_factory=list)  # источники: "gbdt_s1", имена групп сетей
    blend_weights: list[float] = field(default_factory=list)
    final: str = "blend"  # "blend" | "s2" | "cnn"
    severity_mode: str = "legacy"  # output rule; model input features stay unchanged
    min_area_px: int = 0  # minimum mapping unit of the reference (25 px = 1 ha); 0 disables the sieve


class BSEnsemble:
    def __init__(self, models: list[tuple[str, torch.nn.Module]], cfg: BSConfig, device: torch.device, weights_dir: Path):
        self.models, self.cfg, self.device, self.weights_dir = models, cfg, device, weights_dir
        self.groups = list(dict.fromkeys(g for g, _ in models))

    @classmethod
    def load(cls, weights_dir: str | Path, device: str | torch.device = "cpu") -> "BSEnsemble":
        weights_dir = Path(weights_dir)
        raw = json.loads((weights_dir / "bs_config.json").read_text())
        cfg = BSConfig(**{k: v for k, v in raw.items() if k in BSConfig.__dataclass_fields__})
        if cfg.severity_mode not in ("legacy", "class_median"):
            raise ValueError(f"Unknown severity mode: {cfg.severity_mode}")
        device = torch.device(device)
        models = []
        for spec in cfg.cnn:
            ck = torch.load(weights_dir / spec["file"], map_location="cpu", weights_only=False)
            with skip_weight_init():
                m = build_model(ck["arch"], ck["encoder"], pretrained=False, classes=ck.get("classes", 4), ctx=int(ck.get("ctx", 0)),
                                coord=bool(ck.get("coord", False)), dec_attn=ck.get("dec_attn"), hot=bool(ck.get("hot", False)))
            m.hot = bool(ck.get("hot", False))
            m.load_state_dict(ck["state_dict"])  # strict=True: все веса и буферы берутся из чекпойнта
            m.ctx = int(ck.get("ctx", 0))
            m.scale = ck.get("scale", 1)
            m.eval().to(device)
            models.append((spec.get("group", "cnn"), m))
        return cls(models, cfg, device, weights_dir)

    @property
    def uses_gbdt(self) -> bool:
        return bool(self.cfg.gbdt) and self.cfg.final != "cnn"

    def gbdt_files(self) -> tuple[list[Path], list[Path]]:
        g = self.cfg.gbdt or {}
        return [self.weights_dir / f for f in g.get("s1", [])], [self.weights_dir / f for f in g.get("s2", [])]

    @torch.inference_mode()
    def cnn_arrays(self, s2_pre, s2_post, s1_pre, s1_post, aux) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """-> ({группа: P (N,H,W) float32}, правило 0..3 (N,H,W) uint8, допустимость ok (N,H,W) bool)."""
        d = self.device
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(d)
        need_ctx = max(m.ctx for _, m in self.models)
        need_hot = any(m.hot for _, m in self.models)
        X, rule = compute_features(t(s2_pre), t(s2_post), t(s1_pre.astype(np.int32)), t(s1_post.astype(np.int32)),
                                   t(aux.astype(np.int32)), ctx=need_ctx, hot=need_hot)
        acc: dict[str, list] = {}
        for g, m in self.models:
            p = predict_probs(m, X[:, channel_index(m.ctx, m.hot, need_ctx, need_hot)], self.cfg.tta)
            pb = p[:, 0] if p.shape[1] == 1 else 1 - p[:, 0]
            acc.setdefault(g, []).append(pb)
        probs = {g: torch.stack(v).mean(0).float().cpu().numpy() for g, v in acc.items()}
        if self.cfg.severity_mode == "class_median":
            rule = output_class_median(t(s2_pre), t(s2_post), t(aux.astype(np.int32)))
        ok = torch.ones_like(rule, dtype=torch.bool)
        if self.cfg.gate:
            ok &= rule > 0
        if self.cfg.cloud:
            ok &= ~exclusion(t(s2_pre[:, 9].astype(np.int64)), t(s2_post[:, 9].astype(np.int64)))
        return probs, rule.to(torch.uint8).cpu().numpy(), ok.cpu().numpy()

    def blend_prob(self, p1_full: np.ndarray | None, cnn: dict[str, np.ndarray]) -> np.ndarray:
        srcs = self.cfg.blend_sources or (["gbdt_s1"] if p1_full is not None else []) + self.groups
        w = self.cfg.blend_weights or [1.0] * len(srcs)
        pairs = [(p1_full if s == "gbdt_s1" else cnn.get(s), wi) for s, wi in zip(srcs, w)]
        pairs = [(m, wi) for m, wi in pairs if m is not None and wi > 0]  # пропавший источник не роняет прогноз
        tot = sum(wi for _, wi in pairs)
        return (sum(m * wi for m, wi in pairs) / tot).astype(np.float32)

    def assemble(self, prob: np.ndarray, rule: np.ndarray, ok: np.ndarray) -> np.ndarray:
        burned = (prob > self.cfg.threshold) & ok
        if self.cfg.min_area_px > 1:
            burned = sieve_small(burned, int(self.cfg.min_area_px))
        return np.where(burned, np.clip(rule, 1, 3), 0).astype(np.uint8)

    # --- все шаги в текущем процессе (сервис, отладка, малые батчи) ---
    def predict_chips(self, chips: list[BSChip]) -> tuple[np.ndarray, np.ndarray]:
        st = lambda k: np.stack([getattr(c, k) for c in chips])
        cnn, rule, ok = self.cnn_arrays(st("s2_pre"), st("s2_post"), st("s1_pre"), st("s1_post"), st("aux"))
        if not self.uses_gbdt:
            pm = np.mean(list(cnn.values()), 0)
            return np.stack([self.assemble(pm[i], rule[i], ok[i]) for i in range(len(chips))]), pm
        import lightgbm as lgb

        s1_files, s2_files = self.gbdt_files()
        s1 = [lgb.Booster(model_file=str(f)) for f in s1_files]
        s2 = [lgb.Booster(model_file=str(f)) for f in s2_files]
        block = int(self.cfg.gbdt["block"])
        masks, probs = [], []
        for i, c in enumerate(chips):
            X, _, _ = chip_features(c, block)
            p1 = predict_blocks(lambda rows: np.mean([b.predict(rows) for b in s1], 0), X, bool(self.cfg.gbdt.get("sparse", False)))
            pf = self.finish(p1, X, {g: v[i] for g, v in cnn.items()}, s2, block)
            masks.append(self.assemble(pf, rule[i], ok[i]))
            probs.append(pf)
        return np.stack(masks), np.stack(probs)

    def finish(self, p1_blocks: np.ndarray, X: np.ndarray | None, cnn: dict[str, np.ndarray], s2_boosters, block: int) -> np.ndarray:
        """P1 (сетка блоков) + сети -> итоговая вероятность полного разрешения."""
        shape = next(iter(cnn.values())).shape
        p1_full = upsample(p1_blocks, shape)
        ps = self.blend_prob(p1_full, cnn)
        if self.cfg.final != "s2" or not s2_boosters or X is None:
            return ps
        S, _ = stack_features(ps, block)
        sel = X[0] > 0
        p2 = np.zeros(sel.shape, np.float32)
        if sel.any():
            p2[sel] = np.mean([b.predict(np.concatenate([X[:, sel], S[:, sel]]).T) for b in s2_boosters], 0)
        return upsample(p2, shape)


_ENSEMBLE: BSEnsemble | None = None


def predict_bs_chip(chip: BSChip, weights_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Обёртка для сервиса: один чип -> (маска 0..3, P(гарь))."""
    global _ENSEMBLE
    if _ENSEMBLE is None:
        wd = weights_dir or Path(__file__).resolve().parents[2] / "weights"
        _ENSEMBLE = BSEnsemble.load(wd, "cuda" if torch.cuda.is_available() else "cpu")
    m, p = _ENSEMBLE.predict_chips([chip])
    return m[0], p[0]
