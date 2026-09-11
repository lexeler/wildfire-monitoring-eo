"""GBDT-модель «области пожара» на грубой сетке (блоки 4x4 пикселя = 80 м).

Эталонная область пожара — объединение кругов радиусом 2-4 км вокруг
термоточек, т.е. гладкая крупная форма. Её удобно описывать явными
многомасштабными признаками: плотностью гари в окнах 8-256 px, расстояниями до
ядер сильной гари, размерами связных компонент. Модель предсказывает долю
«разрешённых правилом» пикселей блока, попавших в эталон; итоговая маска —
билинейно поднятая вероятность ∩ правило ∩ не облако (как у сети).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..io import BSChip
from .rules import dnbr, excluded, nbr, severity_from_dnbr

BLOCK = 4
RADII_PX = (8, 16, 32, 64, 128, 256)  # 160 м .. 5 км


def _block_mean(x: np.ndarray, b: int = BLOCK) -> np.ndarray:
    h, w = x.shape
    return x[: h // b * b, : w // b * b].reshape(h // b, b, w // b, b).mean((1, 3))


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return ndi.uniform_filter(x, size=2 * r + 1, mode="constant")


def _active_fire(s2: np.ndarray) -> np.ndarray:
    """Пиксели активного горения на снимке Sentinel-2 (см. features.active_fire_s2)."""
    b = s2[:9].astype(np.float32) / 1e4
    h = (b[8] > 0.5) & (b[8] > 1.17 * b[7]) & (b[8] > 2.0 * np.maximum(b[6], 0.01)) & ~((s2[9] >= 8) & (s2[9] <= 10))
    return h & (ndi.uniform_filter(h.astype(np.float32), 3, mode="constant") * 9 >= 1.5)


def chip_features(c: BSChip, block: int = BLOCK, pos: bool = False, hot: bool = False) -> tuple[np.ndarray, list[str], dict]:
    """-> (признаки (F, h, w) на сетке блоков, имена, вспомогательные полноразмерные слои).
    pos=True добавляет положение блока в чипе (чипы вырезаны вокруг события)."""
    def bm(x: np.ndarray) -> np.ndarray:
        return _block_mean(x, block)

    d = dnbr(c.s2_pre, c.s2_post)
    rule = severity_from_dnbr(d, c.aux[2])
    ex = excluded(c.s2_pre[9], c.s2_post[9])
    ok = (rule > 0) & ~ex
    pre = c.s2_pre[:9].astype(np.float32) / 1e4
    post = c.s2_post[:9].astype(np.float32) / 1e4
    f: dict[str, np.ndarray] = {}
    f["ok"] = bm(ok.astype(np.float32))
    for k in (1, 2, 3):
        f[f"rule{k}"] = bm((rule == k).astype(np.float32))
    f["excl"] = bm(ex.astype(np.float32))
    f["dnbr"] = bm(np.clip(d, -1, 1))
    f["nbr_pre"] = bm(nbr(c.s2_pre))
    f["nbr_post"] = bm(nbr(c.s2_post))
    ndvi = lambda s: (s[6] - s[2]) / np.maximum(s[6] + s[2], 1e-3)
    f["ndvi_pre"] = bm(ndvi(pre))
    f["ndvi_post"] = bm(ndvi(post))
    for i, b in enumerate(("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")):
        f[f"post_{b}"] = bm(post[i])
        f[f"diff_{b}"] = bm(post[i] - pre[i])
    for k in (2, 4, 5, 7, 8, 10):
        f[f"scl_post{k}"] = bm((c.s2_post[9] == k).astype(np.float32))
        f[f"scl_pre{k}"] = bm((c.s2_pre[9] == k).astype(np.float32))
    for code in (10, 30, 40, 50, 80, 90):
        f[f"lc{code}"] = bm((c.aux[2] == code).astype(np.float32))
    dem = c.aux[0].astype(np.float32)
    f["dem_rel"] = bm(dem - dem.mean())
    f["slope"] = bm(c.aux[1].astype(np.float32))
    sar_ok = (c.s1_pre[0] != 0) & (c.s1_post[0] != 0)
    f["sar_ok"] = bm(sar_ok.astype(np.float32))
    for i, nm in enumerate(("vv", "vh")):
        dd = (c.s1_post[i].astype(np.float32) - c.s1_pre[i]) / 100 * sar_ok
        f[f"d{nm}"] = bm(dd)

    # текстура: разброс dNBR внутри блока
    f["dnbr_std"] = np.sqrt(np.maximum(bm(np.clip(d, -1, 1) ** 2) - f["dnbr"] ** 2, 0))
    # многомасштабный контекст
    strong = f["rule2"] + f["rule3"]
    for nm in ("diff_B5", "diff_B8A", "nbr_post"):
        for rp in (32, 128):
            f[f"{nm}_r{rp}"] = _box(f[nm], max(1, rp // block))
    for rp in RADII_PX:
        r = max(1, rp // block)
        f[f"ok_r{rp}"] = _box(f["ok"], r)
        f[f"strong_r{rp}"] = _box(strong, r)
        f[f"high_r{rp}"] = _box(f["rule3"], r)
        f[f"dnbr_r{rp}"] = _box(f["dnbr"], r)
    # расстояния (в пикселях 20 м) до «ядер» разной силы; ядро = блоки с
    # высокой долей класса и достаточно крупные компоненты
    dmax = 600.0
    cores = [("high", f["rule3"] > 0.5), ("strong", strong > 0.6), ("ok", f["ok"] > 0.7),
             ("dnbr04", f["dnbr"] > 0.4), ("dnbr06", f["dnbr"] > 0.6)]
    for nm, core in cores:
        for min_px in (0, 2000):
            cc = core
            if min_px and core.any():
                lab, n = ndi.label(core, structure=np.ones((3, 3)))
                sizes = ndi.sum(core, lab, np.arange(1, n + 1)) * block * block
                cc = np.isin(lab, np.flatnonzero(sizes >= min_px) + 1)
            key = f"dist_{nm}" + ("_big" if min_px else "")
            if cc.any():
                f[key] = np.minimum(ndi.distance_transform_edt(~cc) * block, dmax).astype(np.float32)
            else:
                f[key] = np.full(core.shape, dmax, np.float32)
    # связные компоненты допустимой зоны: размер и «сила» компоненты
    lab, n = ndi.label(f["ok"] > 0.3, structure=np.ones((3, 3)))
    if n:
        idx = np.arange(1, n + 1)
        size = ndi.sum(np.ones_like(f["ok"]), lab, idx)
        hi = ndi.sum(f["rule3"], lab, idx)
        dm = ndi.mean(f["dnbr"], lab, idx)
        f["comp_size"] = np.where(lab > 0, size[np.maximum(lab - 1, 0)], 0).astype(np.float32)
        f["comp_high"] = np.where(lab > 0, hi[np.maximum(lab - 1, 0)], 0).astype(np.float32)
        f["comp_dnbr"] = np.where(lab > 0, dm[np.maximum(lab - 1, 0)], 0).astype(np.float32)
    else:
        for k in ("comp_size", "comp_high", "comp_dnbr"):
            f[k] = np.zeros_like(f["ok"])
    # доля чипа и распределение dNBR по чипу
    for k in ("ok", "rule3", "excl"):
        f[f"chip_{k}"] = np.full_like(f["ok"], f[k].mean())
    valid_d = np.clip(d[~ex], -1, 1) if (~ex).any() else np.zeros(1, np.float32)
    for q in (50, 90, 99):
        f[f"chip_dnbr_q{q}"] = np.full_like(f["ok"], np.percentile(valid_d, q))
    if hot:  # видимый на снимках фронт огня: расстояние (px 20 м) на каждой дате и число пикселей огня
        for nm, s2 in (("post", c.s2_post), ("pre", c.s2_pre)):
            h = _block_mean(_active_fire(s2).astype(np.float32), block) > 0
            f[f"dist_hot_{nm}"] = (np.minimum(ndi.distance_transform_edt(~h) * block, 600.0) if h.any() else np.full(h.shape, 600.0)).astype(np.float32)
            f[f"chip_hot_{nm}"] = np.full(h.shape, float(h.sum()), np.float32)
    if pos:
        hb, wb = f["ok"].shape
        yy, xx = np.meshgrid(np.linspace(-1, 1, hb), np.linspace(-1, 1, wb), indexing="ij")
        f["pos_r"] = np.hypot(yy, xx).astype(np.float32)
        f["pos_edge"] = np.maximum(np.abs(yy), np.abs(xx)).astype(np.float32)
    names = list(f)
    X = np.stack([f[k] for k in names]).astype(np.float32)
    return X, names, {"rule": rule, "ok": ok}


_CROSS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)


def predict_blocks(predict_fn, X: np.ndarray, sparse: bool = False) -> np.ndarray:
    """Вероятность области на сетке блоков. predict_fn: строки признаков (M, F) -> (M,).

    sparse=True — деревья считаются только для блоков «шахматной» половины сетки,
    остальные получают среднее предсказанных соседей по кресту (блоки без таких
    соседей считаются честно). Область события — гладкий объект масштаба
    километров, а блок — 40 м, поэтому потеря качества бленда ≤ 0.0005 (OOF),
    а работа CPU на обход деревьев — 80 % времени GBDT — сокращается вдвое."""
    sel = X[0] > 0  # доля допустимых пикселей в блоке («ok») > 0
    p = np.zeros(sel.shape, np.float32)
    if not sparse:
        if sel.any():
            p[sel] = predict_fn(X[:, sel].T)
        return p
    ii, jj = np.indices(sel.shape)
    keep = sel & ((ii + jj) % 2 == 0)
    if keep.any():
        p[keep] = predict_fn(X[:, keep].T)
    den = ndi.convolve(keep.astype(np.float32), _CROSS, mode="constant")
    num = ndi.convolve(p, _CROSS, mode="constant")
    rest = sel & ~keep
    fill = rest & (den > 0)
    p[fill] = num[fill] / den[fill]
    lone = rest & (den == 0)
    if lone.any():
        p[lone] = predict_fn(X[:, lone].T)
    return p


def block_target(mask: np.ndarray, ok: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Доля эталонной гари среди допустимых пикселей блока и вес (их число)."""
    okf = ok.astype(np.float32)
    w = _block_mean(okf, block) * block * block
    y = _block_mean(((mask > 0) & ok).astype(np.float32), block) * block * block / np.maximum(w, 1)
    return y, w


def upsample(p: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Билинейный подъём сетки блоков к полному разрешению (центры блоков)."""
    zoom = (shape[0] / p.shape[0], shape[1] / p.shape[1])
    return np.clip(ndi.zoom(p, zoom, order=1, mode="nearest", grid_mode=True), 0, 1)


def stack_features(prob: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, list[str]]:
    """Автоконтекст: многомасштабные признаки по карте вероятности области
    первой ступени (полное разрешение) на сетке блоков."""
    p = _block_mean(prob.astype(np.float32), block)
    f = {"s1_p": p}
    for rp in (16, 32, 64, 128, 256):
        r = max(1, rp // block)
        f[f"s1_mean_r{rp}"] = _box(p, r)
        f[f"s1_max_r{rp}"] = ndi.maximum_filter(p, size=2 * r + 1, mode="constant")
    for t in (0.5, 0.7):
        core = p > t
        f[f"s1_dist{int(t * 10)}"] = (np.minimum(ndi.distance_transform_edt(~core) * block, 600.0) if core.any()
                                      else np.full(p.shape, 600.0)).astype(np.float32)
    f["s1_chip_mean"] = np.full_like(p, p.mean())
    names = list(f)
    return np.stack([f[k] for k in names]).astype(np.float32), names
