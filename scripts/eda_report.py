#!/usr/bin/env python
"""EDA для технического отчёта: рисунки reports/figures/eda_*.png и факты reports/eda_facts.json.

    python scripts/eda_report.py --data-dir data/train --out reports [--test-dir data/test] [--workers 6]

Все числа, на которые ссылается раздел «EDA и наблюдения по данным», считаются
здесь и сохраняются в eda_facts.json (value + короткое описание по-русски).
Скрипт только читает данные и модули firemon (пороги, правила), ничего не обучает.
Считается на CPU; тяжёлая часть (224 чипа BS 512x512) — пул процессов.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from firemon.af.features import DAY_POT, NIGHT_POT, NIGHT_SZA  # noqa: E402
from firemon.bs.rules import LANDCOVER_GROUP, SCL_EXCLUDE, THRESHOLDS, dnbr, excluded, nbr, severity_from_dnbr  # noqa: E402
from firemon.io import load_af, load_bs, parallel_map  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ----------------------------------------------------------------------------- стиль
C_FIRE = "#d7301f"
C_SEV = {1: "#f6c667", 2: "#f08a3c", 3: "#b3261e"}
C_INK, C_INK2, C_MUTED = "#0b0b0b", "#52514e", "#898781"
C_GRID, C_AXIS = "#e1e0d9", "#c3c2b7"
C_GREY, C_LGREY = "#9a9893", "#d9d8d2"
C_TRAIN, C_TEST = "#52514e", "#2a78d6"
C_HOT = "#00b7d4"
C_GROUP = {"forest": "#2f7d4f", "grass": "#8fbf5a", "crop": "#d9a931", "wetland": "#3b7fc4"}
GREY_CMAP = LinearSegmentedColormap.from_list("greys_eda", ["#ecebe6", "#9a9893", "#2b2a28"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "figure.titlesize": 14,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_AXIS,
    "axes.labelcolor": C_INK, "text.color": C_INK, "xtick.color": C_INK2, "ytick.color": C_INK2,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "legend.frameon": False, "axes.titlelocation": "left",
})

GROUPS = ("forest", "grass", "crop", "wetland")
GROUP_RU = {"forest": "лес / кустарник", "grass": "травы", "crop": "пашня", "wetland": "болота / поймы"}
LC_RU = {10: "деревья", 20: "кустарник", 30: "травы", 40: "пашня", 50: "застройка", 60: "голый грунт",
         70: "снег", 80: "вода", 90: "болота", 95: "мангры", 100: "мхи"}
SCL_RU = {0: "нет данных", 1: "дефект", 2: "тёмные обл.", 3: "тень облака", 4: "растительность", 5: "грунт",
          6: "вода", 7: "неклассиф.", 8: "облако (ср.)", 9: "облако (выс.)", 10: "перистые", 11: "снег"}
SEV_RU = {1: "слабая", 2: "средняя", 3: "сильная"}
PX_HA = 0.04  # 20 м x 20 м
PX_KM = 0.02

# гистограммы
DN_LO, DN_STEP, DN_NB = -0.3, 0.005, 320  # dNBR медиана 3x3, [-0.3, 1.3)
SAR_FEATS = ("dnbr", "dvv", "dvh", "dvv7", "dvh7")
SAR_FEAT_RU = {"dnbr": "dNBR", "dvv": "ΔVV", "dvh": "ΔVH", "dvv7": "ΔVV 7×7", "dvh7": "ΔVH 7×7"}
SAR_BINS = {"dnbr": (-1.0, 0.0025, 1000), "dvv": (-20.0, 0.05, 800), "dvh": (-20.0, 0.05, 800),
            "dvv7": (-20.0, 0.05, 800), "dvh7": (-20.0, 0.05, 800)}
G3_RU = ("лес / кустарник", "травы + пашня", "прочее")
DIST_STEP, DIST_NB = 0.1, 200  # км, до 20 км
RADII_KM = np.round(np.arange(0.5, 10.01, 0.25), 2)
# примеры для eda_bs_region.png (выбраны по таблице region.hotspot_distance.per_chip: дуга буфера видна целиком)
DEFAULT_REGION_EXAMPLES = ("BS_tr_000004", "BS_tr_000109", "BS_tr_000206")
SMOOTH_VARIANTS = ("raw", "mean3", "gauss1", "mode3", "median3")
SMOOTH_RU = {"raw": "без сглаживания", "mean3": "среднее 3×3", "gauss1": "Гаусс σ=1",
             "mode3": "мода классов 3×3", "median3": "медиана 3×3"}


def to_py(x):
    if isinstance(x, dict):
        return {str(k): to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_py(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not np.isfinite(x):
            return None
        return float(f"{x:.6g}")
    return x


class Facts:
    def __init__(self):
        self.d: dict = {}

    def put(self, sec: str, key: str, value, desc: str) -> None:
        self.d.setdefault(sec, {})[key] = {"value": to_py(value), "desc": desc}

    def get(self, sec: str, key: str):
        return self.d[sec][key]["value"]


def save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}", flush=True)


def fi(n) -> str:
    """Целое с пробелом-разделителем тысяч."""
    return f"{int(n):,}".replace(",", " ")


def note(ax, text, loc=(0.02, 0.97), ha="left", va="top", fs=9.5):
    ax.text(*loc, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, color=C_INK2,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_GRID, alpha=0.92))


def ecdf(ax, x, **kw):
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    ax.step(np.r_[x[0], x], np.r_[0, np.arange(1, len(x) + 1) / len(x)], where="post", **kw)


def auc_from_hist(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(score_pos > score_neg) по гистограммам с общими бинами (ничьи — 1/2)."""
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (P * N))


def hist_quantile(h: np.ndarray, lo: float, step: float, q: float) -> float:
    c = np.cumsum(h)
    if c[-1] == 0:
        return float("nan")
    i = int(np.searchsorted(c, q * c[-1]))
    return lo + (i + 0.5) * step


# ============================================================================ AF
def af_section(data_dir: Path, meta: pd.DataFrame, F: Facts, figdir: Path, workers: int) -> dict:
    t0 = time.time()
    ids = meta.chip_id.tolist()
    chips = parallel_map(lambda c: load_af(data_dir, c, with_mask=True), ids, workers)
    V = np.stack([c.viirs for c in chips])
    LC = np.clip(np.nan_to_num(np.stack([c.aux[0] for c in chips]), nan=0), 0, 255).astype(np.int32)
    M = np.stack([c.mask for c in chips])
    del chips
    N, _, H, W = V.shape
    I4, I5, sza = V[:, 3], V[:, 4], V[:, 5]
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(I4) & np.isfinite(I5) & (V[:, 7] > 0.5)
        dT = I4 - I5
        day = ~(np.nan_to_num(sza, nan=0.0) > NIGHT_SZA)
        rule = valid & np.where(day, (I4 > DAY_POT[0]) & (dT > DAY_POT[1]), (I4 > NIGHT_POT[0]) & (dT > NIGHT_POT[1]))
    fire = M == 1
    print(f"AF loaded {N} chips in {time.time() - t0:.1f}s", flush=True)
    S = "af"
    n_px, n_fire = N * H * W, int(fire.sum())
    cnt = fire.sum((1, 2))
    pos = cnt > 0
    F.put(S, "n_chips", N, "Число чипов AF (VIIRS 375 м, 256x256) в train")
    F.put(S, "mask_values", np.unique(M), "Значения в масках AF")
    F.put(S, "n_pixels", n_px, "Всего пикселей во всех чипах AF")
    F.put(S, "n_valid_pixels", int(valid.sum()), "Валидных пикселей (I4, I5 конечны, valid=1)")
    F.put(S, "n_fire_px", n_fire, "Пикселей природного пожара (класс 1)")
    F.put(S, "fire_share_pct", 100 * n_fire / n_px, "Доля пикселей пожара среди всех пикселей, %")
    F.put(S, "fire_share_valid_pct", 100 * n_fire / valid.sum(), "Доля пикселей пожара среди валидных, %")
    F.put(S, "imbalance_bg_per_fire", (n_px - n_fire) / n_fire, "Фоновых пикселей на один пиксель пожара")
    F.put(S, "fire_px_invalid", int((fire & ~valid).sum()), "Пикселей пожара с NaN в I4/I5")
    F.put(S, "n_pos_chips", int(pos.sum()), "Чипов с хотя бы одним пикселем пожара")
    F.put(S, "n_neg_chips", int((~pos).sum()), "Чипов без пожара")
    q = np.percentile(cnt[pos], [10, 25, 50, 75, 90])
    F.put(S, "fire_px_per_pos_chip", {"mean": cnt[pos].mean(), "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3],
                                      "p90": q[4], "max": cnt.max()}, "Пикселей пожара на позитивный чип")
    top10 = np.sort(cnt)[::-1][: max(1, int(round(0.1 * N)))].sum() / n_fire
    F.put(S, "top10pct_chips_fire_share_pct", 100 * top10, "Доля всех пикселей пожара в 10% самых «горящих» чипов, %")
    if "n_fire_px" in meta:
        F.put(S, "meta_n_fire_px_max_abs_diff", np.nanmax(np.abs(meta.n_fire_px.values - cnt)),
              "Макс. расхождение meta.n_fire_px и подсчёта по маске")

    # день / ночь, спутники
    night_fire = fire & ~day
    chip_sza = np.nanmedian(sza.reshape(N, -1), 1)
    chip_night = chip_sza > NIGHT_SZA
    F.put(S, "night_fire_share_pct", 100 * night_fire.sum() / n_fire, "Доля пикселей пожара ночью (SZA > 85°), %")
    F.put(S, "n_night_chips", int(chip_night.sum()), "Ночных чипов (медиана SZA > 85°)")
    F.put(S, "n_day_chips", int((~chip_night).sum()), "Дневных чипов")
    F.put(S, "night_chip_pos_share_pct", 100 * pos[chip_night].mean(), "Доля ночных чипов с пожаром, %")
    F.put(S, "day_chip_pos_share_pct", 100 * pos[~chip_night].mean(), "Доля дневных чипов с пожаром, %")
    sats = {}
    for s in sorted(meta.satellite.dropna().unique()):
        sel = (meta.satellite == s).values
        sats[s] = {"chips": int(sel.sum()), "fire_px": int(cnt[sel].sum()), "pos_chips": int(pos[sel].sum())}
    F.put(S, "by_satellite", sats, "Чипы и пиксели пожара по спутникам")

    # пороги VNP14IMG
    fd, fn_ = fire & day & valid, fire & ~day & valid
    thr = {}
    for name, sel in (("day", fd), ("night", fn_)):
        a4, ad = I4[sel], dT[sel]
        thr[name] = {"n": int(sel.sum()), "I4_min": a4.min(), "I4_p1": np.percentile(a4, 1), "I4_p5": np.percentile(a4, 5),
                     "I4_median": np.median(a4), "dT_min": ad.min(), "dT_p1": np.percentile(ad, 1),
                     "dT_p5": np.percentile(ad, 5), "dT_median": np.median(ad)}
    thr["day"]["pass_fixed_pct"] = 100 * (rule & fd).sum() / fd.sum()
    thr["night"]["pass_fixed_pct"] = 100 * (rule & fn_).sum() / fn_.sum()
    F.put(S, "fire_thresholds", thr, "Статистика I4 и dT=I4-I5 пикселей пожара (день/ночь) и доля, проходящая пороги VNP14IMG")
    F.put(S, "vnp14img_thresholds", {"day": DAY_POT, "night": NIGHT_POT, "night_sza": NIGHT_SZA},
          "Фиксированные пороги VNP14IMG: день I4>325 K и dT>25 K; ночь I4>295 K и dT>10 K")

    def prf(pred, true):
        tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * tp / max(2 * tp + fp + fn, 1)}

    F.put(S, "fixed_rule_all", prf(rule, fire), "Только фиксированные пороги VNP14IMG как классификатор (микро по пикселям)")
    F.put(S, "fixed_rule_day", prf(rule & day, fire & day), "То же, только дневные пиксели")
    F.put(S, "fixed_rule_night", prf(rule & ~day, fire & ~day), "То же, только ночные пиксели")
    nb_fp = (rule & ~fire).sum()
    F.put(S, "background_pass_fixed", int(nb_fp), "Фоновых пикселей, проходящих фиксированные пороги (ложные срабатывания)")

    # локальные максимумы и кластеры
    I4f = np.where(valid, I4, -np.inf)
    mx = ndi.maximum_filter(I4f, size=(1, 3, 3), mode="nearest")
    lm = fire & valid & (I4f >= mx)
    F.put(S, "fire_local_max_share_pct", 100 * lm.sum() / (fire & valid).sum(),
          "Доля пикселей пожара, являющихся локальным максимумом I4 в окне 3x3, %")
    sizes = []
    for i in np.flatnonzero(pos):
        lab, n = ndi.label(fire[i], structure=np.ones((3, 3)))
        sizes.extend(np.bincount(lab.ravel())[1:].tolist())
    sizes = np.array(sizes)
    F.put(S, "clusters", {"n": len(sizes), "median_size": np.median(sizes), "mean_size": sizes.mean(), "max_size": sizes.max(),
                          "single_px_cluster_share_pct": 100 * (sizes == 1).mean(),
                          "fire_px_in_clusters_ge2_pct": 100 * sizes[sizes >= 2].sum() / sizes.sum()},
          "Связные (8-соседство) кластеры пикселей пожара")

    # покров
    codes = [c for c in sorted(np.unique(LC[valid]).tolist()) if c > 0]
    fire_lc = np.bincount(LC[fire].ravel(), minlength=256)
    terr_lc = np.bincount(LC[valid].ravel(), minlength=256)
    fp_lc = np.bincount(LC[rule & ~fire].ravel(), minlength=256)
    lc_tab = {}
    for c in codes:
        lc_tab[str(c)] = {"name": LC_RU.get(c, str(c)), "fire_px": int(fire_lc[c]), "fire_share_pct": 100 * fire_lc[c] / n_fire,
                          "territory_share_pct": 100 * terr_lc[c] / terr_lc.sum(),
                          "enrichment": (fire_lc[c] / n_fire) / max(terr_lc[c] / terr_lc.sum(), 1e-12),
                          "rule_fp_share_pct": 100 * fp_lc[c] / max(fp_lc.sum(), 1)}
    F.put(S, "landcover", lc_tab, "Покров (ESA WorldCover) пикселей пожара vs доля территории и ложных срабатываний порогов")

    # невалидные пиксели: вне полосы обзора (meta.valid_frac) и полосы NaN внутри неё
    nanm = ~np.isfinite(I4)
    nanfrac = nanm.reshape(N, -1).mean(1)
    vf = meta.valid_frac.values if "valid_frac" in meta else np.ones(N)
    stripe = np.clip(nanfrac - (1 - vf), 0, 1)  # NaN внутри полосы обзора
    vza = np.nanmedian(V[:, 6].reshape(N, -1), 1)
    num = meta.chip_id.str.extract(r"(\d+)$")[0].astype(int).values
    corrupt = (num >= 409) & (num <= 420)
    has = stripe > 0.01
    F.put(S, "corrupted_chips", {"ids": meta.chip_id[corrupt].tolist(), "satellite": meta.satellite[corrupt].tolist(),
                                 "date": meta.acq_datetime[corrupt].str[:10].tolist(), "I4_nan_frac": nanfrac[corrupt],
                                 "stripe_nan_frac": stripe[corrupt], "meta_valid_frac": vf[corrupt], "median_vza": vza[corrupt],
                                 "fire_px": cnt[corrupt]},
          "Чипы 000409-000420 (организаторы подтвердили пропуски строк): доля NaN в I4, NaN внутри полосы обзора, meta.valid_frac, VZA")
    F.put(S, "nan_frac_corrupted_max_pct", 100 * nanfrac[corrupt].max(), "Макс. доля NaN в I4 среди чипов 409-420, %")
    F.put(S, "nan_frac_all", {"median_pct": 100 * np.median(nanfrac), "p95_pct": 100 * np.percentile(nanfrac, 95), "max_pct": 100 * nanfrac.max(),
                              "chips_with_any_nan": int((nanfrac > 0).sum()), "chips_nan_gt10pct": int((nanfrac > 0.1).sum()),
                              "valid_px_share_pct": 100 * valid.mean()},
          "Доля NaN в I4 по всем чипам train")
    F.put(S, "meta_valid_frac", {"chips_lt1": int((vf < 1).sum()), "min": vf.min(),
                                 "corr_with_1_minus_nan": np.corrcoef(vf, 1 - nanfrac)[0, 1]},
          "meta.valid_frac описывает только долю чипа внутри полосы обзора (не учитывает полосы NaN)")
    F.put(S, "stripe_nan", {"chips_gt1pct": int(has.sum()), "chips_gt1pct_share_pct": 100 * has.mean(),
                            "median_pct_among_affected": 100 * np.median(stripe[has]) if has.any() else None,
                            "max_pct": 100 * stripe.max(), "vza_min_affected": vza[has].min() if has.any() else None,
                            "vza_max_unaffected": vza[~has].max() if (~has).any() else None,
                            "affected_share_vza_gt_threshold_pct": 100 * has[vza > vza[has].min()].mean() if has.any() else None,
                            "spearman_vza": stats.spearmanr(vza, stripe).statistic},
          "Полосы NaN внутри полосы обзора (удаление «bow-tie» на краях скана VIIRS): число чипов, связь с зенитным углом сенсора")
    fbd = fire & valid
    nb_nan = ndi.maximum_filter(nanm.astype(np.uint8), size=(1, 3, 3)) > 0
    F.put(S, "fire_px_next_to_nan_pct", 100 * (fbd & nb_nan).sum() / fbd.sum(), "Доля пикселей пожара, соседствующих (3x3) с NaN, %")
    # пропуски фиксированных порогов днём
    fn_day = fire & valid & day & ~rule
    F.put(S, "fixed_rule_missed_day", {"n": int(fn_day.sum()), "dT_lt0": int((dT[fn_day] < 0).sum()),
                                       "I4_median": np.median(I4[fn_day]), "I5_median": np.median(I5[fn_day]),
                                       "I4_lt325": int((I4[fn_day] <= DAY_POT[0]).sum()), "dT_le25": int((dT[fn_day] <= DAY_POT[1]).sum())},
          "Дневные пиксели пожара, не проходящие фиксированные пороги (контекстные тесты VNP14IMG)")

    # повторяемость ложных срабатываний (техногенные источники): абсолютная сетка 375 м
    dates = pd.to_datetime(meta.acq_datetime, utc=True).dt.date.values
    rec_rows = []
    for i in range(N):
        rr, cc = np.nonzero(rule[i])
        if len(rr) == 0:
            continue
        e = int(meta.epsg.iloc[i])
        col = int(round(meta.x_min.iloc[i] / 375)) + cc
        row = int(round(meta.y_max.iloc[i] / 375)) - rr
        rec_rows.append(pd.DataFrame({"key": (e % 100) * 10**12 + col.astype(np.int64) * 10**6 + row, "date": dates[i],
                                      "fire": fire[i, rr, cc], "lc": LC[i, rr, cc]}))
    rec = pd.concat(rec_rows, ignore_index=True)
    rec["n_dates"] = rec.key.map(rec.groupby("key").date.nunique())
    bins = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 10**6)]
    rtab = []
    for lo, hi in bins:
        s = rec[(rec.n_dates >= lo) & (rec.n_dates <= hi)]
        rtab.append({"n_dates": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f">={lo}"), "rule_px": len(s),
                     "fire_share_pct": 100 * s.fire.mean() if len(s) else None, "built_up_share_pct": 100 * (s.lc == 50).mean() if len(s) else None})
    F.put(S, "rule_hot_recurrence", rtab,
          "Пиксели, проходящие пороги, по числу разных дат, когда та же точка (сетка 375 м) была «горячей»: доля эталонного пожара")
    loc = rec.groupby("key").agg(n_dates=("n_dates", "first"), fire=("fire", "mean"), lc=("lc", "first"))
    top = loc.sort_values("n_dates", ascending=False).index[0]
    e_top, col_top, row_top = int(top // 10**12), int(top % 10**12 // 10**6), int(top % 10**6)
    try:
        from pyproj import Transformer
        lon_t, lat_t = Transformer.from_crs(f"EPSG:326{e_top:02d}", "EPSG:4326", always_xy=True).transform((col_top + 0.5) * 375, (row_top - 0.5) * 375)
    except Exception:
        lon_t = lat_t = None
    F.put(S, "recurrent_locations", {"n_locations_ge3": int((loc.n_dates >= 3).sum()), "n_locations_ge10": int((loc.n_dates >= 10).sum()),
                                     "ge3_built_up_share_pct": 100 * (loc[loc.n_dates >= 3].lc == 50).mean(),
                                     "ge3_mean_fire_label_share_pct": 100 * loc[loc.n_dates >= 3].fire.mean(),
                                     "top_location_lonlat": [lon_t, lat_t], "top_location_n_dates": int(loc.n_dates.max()),
                                     "rule_hot_built_up_fire_share_pct": 100 * rec[rec.lc == 50].fire.mean(),
                                     "rule_hot_other_fire_share_pct": 100 * rec[rec.lc != 50].fire.mean()},
          "Точки (сетка 375 м), «горячие» по порогам в >=3 разные даты: почти все — застройка (промышленные объекты), "
          "но эталон помечает их пожаром в ~половине случаев => статической маски техногенных источников в эталоне нет")
    fp_rec = rec[~rec.fire]
    tp_rec = rec[rec.fire]
    F.put(S, "fp_recurrent_ge3_pct", 100 * (fp_rec.n_dates >= 3).mean(), "Доля ложных срабатываний порогов в точках, горячих в >=3 разные даты, %")
    F.put(S, "tp_recurrent_ge3_pct", 100 * (tp_rec.n_dates >= 3).mean(), "То же для истинных пикселей пожара, %")
    F.put(S, "fp_built_up_share_pct", 100 * (fp_rec.lc == 50).mean(), "Доля ложных срабатываний порогов на застройке (50), %")
    F.put(S, "tp_built_up_share_pct", 100 * (tp_rec.lc == 50).mean(), "Доля пикселей пожара на застройке (50), %")

    # горячие точки для связки с BS (центры пикселей пожара в CRS чипа)
    hot = []
    for i in np.flatnonzero(pos):
        rr, cc = np.nonzero(fire[i])
        hot.append({"epsg": int(meta.epsg.iloc[i]), "time": pd.Timestamp(meta.acq_datetime.iloc[i]),
                    "x": meta.x_min.iloc[i] + (cc + 0.5) * 375.0, "y": meta.y_max.iloc[i] - (rr + 0.5) * 375.0})
    footprints = [{"epsg": int(meta.epsg.iloc[i]), "time": pd.Timestamp(meta.acq_datetime.iloc[i]),
                   "bounds": (meta.x_min.iloc[i], meta.y_min.iloc[i], meta.x_max.iloc[i], meta.y_max.iloc[i])} for i in range(N)]

    fig_af_thresholds(I4, dT, day, valid, fire, rule, thr, figdir)
    fig_af_imbalance(cnt, pos, chip_night, meta, fire_lc, terr_lc, codes, n_fire, nanfrac, stripe, vza, corrupt, sizes, F, figdir)
    train_chip = {"nanfrac": nanfrac, "stripe": stripe, "night": chip_night, "rule_cnt": rule.sum((1, 2))}
    print(f"AF section done in {time.time() - t0:.1f}s", flush=True)
    return {"hot": hot, "footprints": footprints, "chip": train_chip}


def fig_af_thresholds(I4, dT, day, valid, fire, rule, thr, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    setups = ((axes[0], True, DAY_POT, (285, 372), (-5, 85), "а) День (SZA ≤ 85°)"),
              (axes[1], False, NIGHT_POT, (265, 345), (-5, 60), "б) Ночь (SZA > 85°)"))
    for ax, isday, (t4, tdt), r4, rd, title in setups:
        dsel = day if isday else ~day
        sel = valid & ~fire & dsel
        xe = np.arange(r4[0], r4[1] + 0.01, 0.5)
        ye = np.arange(rd[0], rd[1] + 0.01, 0.5)
        h, _, _ = np.histogram2d(I4[sel], dT[sel], bins=[xe, ye])
        h = np.ma.masked_less(h, 1)
        pc = ax.pcolormesh(xe, ye, h.T, norm=LogNorm(vmin=1, vmax=max(h.max(), 10)), cmap=GREY_CMAP, rasterized=True)
        fs = fire & valid & dsel
        ax.fill_between([t4, r4[1]], tdt, rd[1], color=C_FIRE, alpha=0.06, lw=0)
        ax.scatter(I4[fs], dT[fs], s=5, c=C_FIRE, alpha=0.55, lw=0, rasterized=True)
        ax.axvline(t4, ls="--", lw=1.2, c=C_INK)
        ax.axhline(tdt, ls="--", lw=1.2, c=C_INK)
        ax.set_xlim(*r4)
        ax.set_ylim(*rd)
        ax.set_xlabel("I4, яркостная температура 3,74 мкм, K")
        ax.set_ylabel("I4 − I5, K")
        ax.set_title(title)
        t = thr["day" if isday else "night"]
        bg_hot = int((rule & ~fire & dsel).sum())
        note(ax, (f"пороги VNP14IMG: I4 > {t4:.0f} K и I4−I5 > {tdt:.0f} K\n"
                  f"пожар: n = {fi(t['n'])}, выше порогов {t['pass_fixed_pct']:.1f}%\n"
                  f"мин. I4 = {t['I4_min']:.1f} K; 1-й перц. = {t['I4_p1']:.1f} K\n"
                  f"мин. I4−I5 = {t['dT_min']:.1f} K; 1-й перц. = {t['dT_p1']:.1f} K\n"
                  f"фон выше порогов: {fi(bg_hot)} пикс."), loc=(0.02, 0.97))
        cb = fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.045)
        cb.set_label("фоновых пикселей в ячейке 0,5×0,5 K")
        cb.outline.set_visible(False)
    handles = [Patch(color="#9a9893", label="фоновые пиксели (плотность)"),
               Line2D([], [], marker="o", ls="", ms=5, color=C_FIRE, label="пиксели пожара (эталон)"),
               Line2D([], [], ls="--", c=C_INK, label="фиксированные пороги VNP14IMG")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.suptitle("AF: эталонные пиксели пожара ограничены снизу ровно порогами VNP14IMG (Schroeder et al., 2014)")
    save(fig, figdir / "eda_af_thresholds.png")


def fig_af_imbalance(cnt, pos, chip_night, meta, fire_lc, terr_lc, codes, n_fire, nanfrac, stripe, vza, corrupt, sizes, F, figdir):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    # а) гистограмма пикселей пожара на чип
    ax = axes[0, 0]
    c = cnt[pos]
    bins = np.unique(np.round(np.logspace(0, np.log10(c.max() + 1), 22))) - 0.5  # границы между целыми
    bins = np.r_[bins, c.max() + 0.5] if bins[-1] < c.max() else bins
    ax.hist(c, bins=bins, color=C_FIRE, alpha=0.85, edgecolor="white", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.8, c.max() * 1.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.axvline(np.median(c), c=C_INK, ls="--", lw=1.2)
    ax.set_xlabel("пикселей пожара в чипе (лог. шкала)")
    ax.set_ylabel("число чипов")
    ax.set_title("а) Пиксели пожара на чип (позитивные чипы)")
    note(ax, (f"пожар: {fi(n_fire)} пикс. = {100 * n_fire / (len(cnt) * 65536):.3f}% всех пикселей\n"
              f"позитивных чипов: {pos.sum()}, пустых: {(~pos).sum()} ({100 * (~pos).mean():.1f}%)\n"
              f"медиана = {np.median(c):.0f}, макс. = {fi(c.max())} пикс./чип\n"
              f"кластеров (8-связн.): {fi(len(sizes))}, медиана размера {np.median(sizes):.0f}"),
         loc=(0.98, 0.97), ha="right")
    # б) доли чипов и пикселей пожара: спутники, день/ночь
    ax = axes[0, 1]
    cats, sh_chip, sh_fire = [], [], []
    sat_ru = {"SNPP": "S-NPP", "NOAA20": "NOAA-20", "NOAA21": "NOAA-21"}
    for s in ("SNPP", "NOAA20", "NOAA21"):
        sel = (meta.satellite == s).values
        if sel.any():
            cats.append(sat_ru[s])
            sh_chip.append(100 * sel.mean())
            sh_fire.append(100 * cnt[sel].sum() / n_fire)
    for nm, sel in (("день", ~chip_night), ("ночь", chip_night)):
        cats.append(nm)
        sh_chip.append(100 * sel.mean())
        sh_fire.append(100 * cnt[sel].sum() / n_fire)
    x = np.arange(len(cats))
    ax.bar(x - 0.2, sh_chip, 0.38, color=C_GREY, label="доля чипов")
    ax.bar(x + 0.2, sh_fire, 0.38, color=C_FIRE, label="доля пикселей пожара")
    for xi, a, b in zip(x, sh_chip, sh_fire):
        ax.text(xi - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=9.5, color=C_INK2)
        ax.text(xi + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=9.5, color=C_INK2)
    ax.axvline(2.5, c=C_AXIS, lw=1)
    ax.set_xticks(x, cats)
    ax.set_ylabel("%")
    ax.set_ylim(0, 112)
    ax.set_title("б) Спутники и время суток (по чипам, SZA > 85° = ночь)")
    ax.legend(loc="upper left", ncol=2)
    # в) покров
    ax = axes[1, 0]
    order = sorted([c for c in codes if fire_lc[c] > 0 or terr_lc[c] / terr_lc.sum() > 0.005], key=lambda c: -fire_lc[c])
    y = np.arange(len(order))[::-1]
    fs = np.array([100 * fire_lc[c] / n_fire for c in order])
    ts = np.array([100 * terr_lc[c] / terr_lc.sum() for c in order])
    ax.barh(y + 0.2, ts, 0.38, color=C_GREY, label="доля территории")
    ax.barh(y - 0.2, fs, 0.38, color=C_FIRE, label="доля пикселей пожара")
    for yi, c, f_, t_ in zip(y, order, fs, ts):
        ax.text(max(f_, t_) + 0.8, yi, f"{fi(fire_lc[c])} пикс., ×{f_ / max(t_, 1e-9):.2f}",
                va="center", fontsize=9.5, color=C_INK2)
    ax.set_yticks(y, [f"{LC_RU.get(c, c)} ({c})" for c in order])
    ax.set_xlabel("%")
    ax.set_xlim(0, max(fs.max(), ts.max()) * 1.45)
    ax.set_title("в) Покров WorldCover: пожары vs территория (×: обогащение)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    # г) полосы NaN внутри полосы обзора vs зенитный угол сенсора
    ax = axes[1, 1]
    ax.scatter(vza[~corrupt], 100 * stripe[~corrupt], s=16, c=C_GREY, alpha=0.8, lw=0, label="чипы train")
    ax.scatter(vza[corrupt], 100 * stripe[corrupt], s=34, c=C_FIRE, marker="D", lw=0, label="чипы 000409–000420 (пропуски строк)")
    ax.set_xlabel("медианный зенитный угол сенсора в чипе, °")
    ax.set_ylabel("NaN в I4 внутри полосы обзора, %")
    ax.set_title("г) Полосы NaN: удаление «bow-tie» на краях скана")
    ax.legend(loc="upper left")
    has = stripe > 0.01
    note(ax, (f"NaN-полосы (> 1%): {has.sum()} из {len(has)} чипов ({100 * has.mean():.0f}%), до {100 * stripe.max():.0f}%\n"
              f"все при VZA ≥ {vza[has].min():.0f}°; всего NaN в чипе до {100 * nanfrac.max():.0f}%\n"
              f"вне полосы обзора (meta.valid_frac < 1): {(meta.valid_frac < 1).sum()} чипов"),
         loc=(0.02, 0.80))
    fig.suptitle(f"AF: сильный дисбаланс классов (пожар = {100 * n_fire / (len(cnt) * 65536):.3f}% пикселей) и неоднородность выборки")
    save(fig, figdir / "eda_af_imbalance.png")


# ============================================================================ BS: вспомогательное
def mode3(cls: np.ndarray) -> np.ndarray:
    """Мода классов 0..3 в окне 3x3 (при ничьей с центром — центр)."""
    cnt = np.stack([np.rint(ndi.uniform_filter((cls == k).astype(np.float32), 3) * 9) for k in range(4)])
    best = cnt.argmax(0)
    center = np.take_along_axis(cnt, cls[None].astype(np.int64), 0)[0]
    return np.where(center >= cnt.max(0), cls, best).astype(np.uint8)


def lc_mode(lc: np.ndarray, size: int) -> np.ndarray:
    """Мажоритарный код покрова в окне size x size."""
    codes = np.unique(lc)
    cnt = np.stack([ndi.uniform_filter((lc == c).astype(np.float32), size) for c in codes])
    return codes[cnt.argmax(0)].astype(lc.dtype)


def group_index(lc: np.ndarray) -> np.ndarray:
    g = np.full(lc.shape, GROUPS.index("grass"), np.int8)
    for code, grp in LANDCOVER_GROUP.items():
        g[lc == code] = GROUPS.index(grp)
    return g


def group3_index(lc: np.ndarray) -> np.ndarray:
    g = np.full(lc.shape, 2, np.int8)
    g[np.isin(lc, (10, 20))] = 0
    g[np.isin(lc, (30, 40))] = 1
    return g


def binned(x: np.ndarray, lo: float, step: float, nb: int) -> np.ndarray:
    b = np.floor((x - lo) / step).astype(np.int64)
    return np.clip(b, 0, nb - 1)


def fit_circle(p: np.ndarray):
    """Алгебраическая подгонка окружности (Kasa) по точкам (row, col)."""
    x, y = p[:, 1].astype(float), p[:, 0].astype(float)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 0))
    return cy, cx, r


def ransac_circles(pts: np.ndarray, seed: int, max_circles=8, iters=1500, tol=1.5, min_inl=60,
                   rmin=10.0, rmax=400.0, nbr_rad=150.0, min_arc_deg=60.0):
    """Последовательный RANSAC: окружности на точках «срезанной» границы гари (row, col, пиксели)."""
    rng = np.random.default_rng(seed)
    P = pts.astype(float)
    out = []
    for _ in range(max_circles):
        if len(P) < min_inl:
            break
        tree = cKDTree(P)
        i0 = rng.integers(0, len(P), iters)
        nbl = tree.query_ball_point(P[i0], nbr_rad)
        trip = []
        for a, lst in zip(i0, nbl):
            if len(lst) >= 3:
                j, k = rng.choice(lst, 2, replace=False)
                trip.append((a, j, k))
        if not trip:
            break
        trip = np.array(trip)
        a, b, c = P[trip[:, 0]], P[trip[:, 1]], P[trip[:, 2]]
        ay, ax_, by, bx, cy, cx = a[:, 0], a[:, 1], b[:, 0], b[:, 1], c[:, 0], c[:, 1]
        d = 2 * (ax_ * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ok = np.abs(d) > 1e-6
        d = np.where(ok, d, 1.0)
        s1, s2, s3 = ax_ ** 2 + ay ** 2, bx ** 2 + by ** 2, cx ** 2 + cy ** 2
        ux = (s1 * (by - cy) + s2 * (cy - ay) + s3 * (ay - by)) / d
        uy = (s1 * (cx - bx) + s2 * (ax_ - cx) + s3 * (bx - ax_)) / d
        r = np.hypot(ax_ - ux, ay - uy)
        ok &= (r > rmin) & (r < rmax)
        if not ok.any():
            break
        ux, uy, r = ux[ok], uy[ok], r[ok]
        Ps = P if len(P) <= 6000 else P[rng.choice(len(P), 6000, replace=False)]
        best, best_n = -1, 0
        for s in range(0, len(r), 200):
            dd = np.abs(np.hypot(Ps[None, :, 1] - ux[s:s + 200, None], Ps[None, :, 0] - uy[s:s + 200, None]) - r[s:s + 200, None])
            n = (dd < tol).sum(1)
            j = int(n.argmax())
            if n[j] > best_n:
                best, best_n = s + j, int(n[j])
        cy0, cx0, r0 = uy[best], ux[best], r[best]
        inl = np.abs(np.hypot(P[:, 1] - cx0, P[:, 0] - cy0) - r0) < tol
        if inl.sum() >= 5:
            cy1, cx1, r1 = fit_circle(P[inl])
            if rmin < r1 < rmax:
                inl2 = np.abs(np.hypot(P[:, 1] - cx1, P[:, 0] - cy1) - r1) < tol
                if inl2.sum() >= inl.sum():
                    cy0, cx0, r0, inl = cy1, cx1, r1, inl2
        n_inl = int(inl.sum())
        if n_inl < min_inl:
            break
        ang = np.degrees(np.arctan2(P[inl, 0] - cy0, P[inl, 1] - cx0))
        arc = 10.0 * np.count_nonzero(np.bincount(((ang + 180) // 10).astype(int) % 36, minlength=36))
        if arc >= min_arc_deg:
            out.append({"cy": cy0, "cx": cx0, "r": r0, "n_inl": n_inl, "arc_deg": arc})
        # убираем полосу вокруг найденной дуги (граница толщиной в несколько пикселей)
        P = P[np.abs(np.hypot(P[:, 1] - cx0, P[:, 0] - cy0) - r0) >= 2 * tol]
    return out


def bs_maps(c, lc):
    d_med = dnbr(c.s2_pre, c.s2_post, median=3)
    sev = severity_from_dnbr(d_med, lc)
    ex = excluded(c.s2_pre[9], c.s2_post[9])
    return d_med, sev, ex


def hotspot_distance_km(hot_rc: np.ndarray, shape=(512, 512)) -> np.ndarray:
    tree = cKDTree(hot_rc)
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    d, _ = tree.query(np.c_[rr.ravel(), cc.ravel()], workers=1)
    return (d * PX_KM).reshape(shape)


# ============================================================================ BS: воркер на чип
def bs_worker(args):
    root, cid, hot_rc, seed = args
    c = load_bs(root, cid, with_mask=True)
    m = c.mask.astype(np.uint8)
    n255 = int((m == 255).sum())
    m[m == 255] = 0
    lc = c.aux[2].astype(np.int32)
    g4 = group_index(lc)
    g3 = group3_index(lc)
    burned = m > 0
    nb = int(burned.sum())
    scl_pre, scl_post = c.s2_pre[9].astype(np.int64), c.s2_post[9].astype(np.int64)
    ex = excluded(scl_pre, scl_post)
    out = {"cid": cid, "n255": n255, "n_burned": nb, "sev_px": np.bincount(m.ravel(), minlength=4)[:4]}

    # --- варианты сглаживания dNBR
    d_raw = dnbr(c.s2_pre, c.s2_post, median=0)
    d_med = ndi.median_filter(d_raw, size=3)
    dv = {"raw": d_raw, "mean3": ndi.uniform_filter(d_raw, 3), "gauss1": ndi.gaussian_filter(d_raw, 1.0), "median3": d_med}
    sev = {k: severity_from_dnbr(v, lc) for k, v in dv.items()}
    sev["mode3"] = mode3(sev["raw"])
    out["agree_exact"] = {k: int(((s == m) & burned).sum()) for k, s in sev.items()}
    out["agree_clip"] = {k: int(((np.clip(s, 1, 3) == m) & burned).sum()) for k, s in sev.items()}
    edge = burned & (ndi.maximum_filter(m, 3) != ndi.minimum_filter(m, 3))
    inter = burned & ~edge
    out["edge"] = {"n_edge": int(edge.sum()), "n_inter": int(inter.sum())}
    for k in ("raw", "median3"):
        err = np.clip(sev[k], 1, 3) != m
        out["edge"][f"err_edge_{k}"] = int((err & edge).sum())
        out["edge"][f"err_inter_{k}"] = int((err & inter).sum())

    # --- какой покров использован для выбора порогов: попиксельный, сглаженный или доминирующий по чипу
    rep_code = {"forest": 10, "grass": 30, "crop": 40, "wetland": 90}
    lcv = {"pixel": lc, "mode3": lc_mode(lc, 3), "mode5": lc_mode(lc, 5), "mode9": lc_mode(lc, 9),
           "forest_wetland_as_grass": np.where(np.isin(lc, (10, 20, 80, 90, 95)), 30, lc),
           "chip_dominant": np.full_like(lc, rep_code[GROUPS[int(np.bincount(g4.ravel(), minlength=4).argmax())]])}
    out["lc_group_burned"] = np.bincount(g4[burned], minlength=4)
    out["lc_variant_agree"] = {nm: np.bincount(g4[burned], weights=(np.clip(severity_from_dnbr(d_med, v), 1, 3) == m)[burned].astype(float), minlength=4)
                               for nm, v in lcv.items()}

    # --- правило vs эталон
    r1 = sev["median3"] >= 1
    cand = r1 & ~ex
    out["conf"] = {"r1_b": int((r1 & burned).sum()), "r1_nb": int((r1 & ~burned).sum()), "r0_b": int((~r1 & burned).sum()),
                   "cand_b": int((cand & burned).sum()), "cand_nb": int((cand & ~burned).sum()),
                   "ex_b": int((ex & burned).sum()), "ex_px": int(ex.sum())}
    out["n_r1"] = int(r1.sum())
    out["n_cand"] = int(cand.sum())
    # правило как прогноз степени (IoU/mIoU без ограничения области)
    pr = np.where(cand, np.clip(sev["median3"], 1, 3), 0)
    out["rule_counts"] = {k: (int(((pr == k) & (m == k)).sum()), int(((pr == k) & (m != k)).sum()), int(((pr != k) & (m == k)).sum()))
                          for k in (1, 2, 3)}
    out["rule_burn_counts"] = (int((cand & burned).sum()), int((cand & ~burned).sum()), int((~cand & burned).sum()))

    # --- SCL: доля «горело» среди r1 по классу SCL
    out["scl"] = {
        "pre_r1": np.bincount(scl_pre[r1], minlength=12)[:12], "pre_r1_b": np.bincount(scl_pre[r1 & burned], minlength=12)[:12],
        "post_r1": np.bincount(scl_post[r1], minlength=12)[:12], "post_r1_b": np.bincount(scl_post[r1 & burned], minlength=12)[:12],
        "pre_all": np.bincount(scl_pre.ravel(), minlength=12)[:12], "post_all": np.bincount(scl_post.ravel(), minlength=12)[:12],
    }
    cl = lambda s, codes: np.isin(s, codes)
    out["cloud_defs"] = {
        "post_8_9_10": cl(scl_post, (8, 9, 10)).mean(), "prepost_8_9_10": (cl(scl_pre, (8, 9, 10)) | cl(scl_post, (8, 9, 10))).mean(),
        "post_3_8_9_10": cl(scl_post, (3, 8, 9, 10)).mean(),
        "prepost_3_8_9_10": (cl(scl_pre, (3, 8, 9, 10)) | cl(scl_post, (3, 8, 9, 10))).mean(),
        "prepost_0_3_8_9_10": (cl(scl_pre, (0, 3, 8, 9, 10)) | cl(scl_post, (0, 3, 8, 9, 10))).mean(),
        "pre_8_9_10": cl(scl_pre, (8, 9, 10)).mean(), "pre_3_8_9_10": cl(scl_pre, (3, 8, 9, 10)).mean(),
        "post_8_9": cl(scl_post, (8, 9)).mean(), "prepost_8_9": (cl(scl_pre, (8, 9)) | cl(scl_post, (8, 9))).mean(),
        "post_3_8_9": cl(scl_post, (3, 8, 9)).mean(), "prepost_3_8_9": (cl(scl_pre, (3, 8, 9)) | cl(scl_post, (3, 8, 9))).mean(),
        "post_0_1_3_8_9_10_11": cl(scl_post, (0, 1, 3, 8, 9, 10, 11)).mean(),
        "excluded": ex.mean()}

    # --- гистограммы медианного dNBR по группе покрова и классу эталона
    sel = ~ex | burned
    key = (g4.astype(np.int64) * 4 + m) * DN_NB + binned(d_med, DN_LO, DN_STEP, DN_NB)
    inr = (d_med >= DN_LO) & (d_med < DN_LO + DN_STEP * DN_NB) & sel
    out["dn_hist"] = np.bincount(key[inr], minlength=16 * DN_NB).reshape(4, 4, DN_NB)
    out["grp_cls"] = np.bincount((g4.astype(np.int64) * 4 + m).ravel(), minlength=16).reshape(4, 4)
    out["lc_cls"] = {int(v): np.bincount(m[lc == v], minlength=4)[:4] for v in np.unique(lc)}
    out["mean_dnbr_burned"] = float(d_med[burned].mean()) if nb else float("nan")
    out["dom_group"] = GROUPS[int(np.bincount(g4[burned], minlength=4).argmax())] if nb else GROUPS[int(np.bincount(g4.ravel(), minlength=4).argmax())]

    # --- SAR
    vv0, vh0 = c.s1_pre.astype(np.float32)
    vv1, vh1 = c.s1_post.astype(np.float32)
    out["s1_zero"] = {"pre": float(((vv0 == 0) | (vh0 == 0)).mean()), "post": float(((vv1 == 0) | (vh1 == 0)).mean())}
    s1ok = (vv0 != 0) & (vh0 != 0) & (vv1 != 0) & (vh1 != 0)
    vv0, vh0, vv1, vh1 = vv0 / 100, vh0 / 100, vv1 / 100, vh1 / 100
    w = s1ok.astype(np.float32)
    den = np.maximum(ndi.uniform_filter(w, 7), 1e-6)
    feats = {"dnbr": d_med, "dvv": vv1 - vv0, "dvh": vh1 - vh0}
    feats["dvv7"] = ndi.uniform_filter(feats["dvv"] * w, 7) / den
    feats["dvh7"] = ndi.uniform_filter(feats["dvh"] * w, 7) / den
    ssel = s1ok & ~ex
    sar_h = {}
    for f, x in feats.items():
        lo, st, nbin = SAR_BINS[f]
        bi = binned(x, lo, st, nbin)
        k = (g3.astype(np.int64) * 2 + burned) * nbin + bi
        h_all = np.bincount(k[ssel], minlength=6 * nbin).reshape(3, 2, nbin)
        h_r1 = np.bincount(k[ssel & r1], minlength=6 * nbin).reshape(3, 2, nbin)
        sar_h[f] = np.stack([h_all, h_r1])  # (subset, g3, label, bins)
    out["sar_hist"] = sar_h

    # спекл: однородные блоки 7x7 (без гари, облаков, одного покрова, с ровным NBR)
    B = 7
    n = 512 // B

    def blocks(x):
        return x[: n * B, : n * B].reshape(n, B, n, B).swapaxes(1, 2).reshape(n * n, B * B)

    okb = blocks(s1ok & ~ex & ~burned).all(1)
    lcb = blocks(lc)
    okb &= (lcb == lcb[:, :1]).all(1)
    okb &= blocks(nbr(c.s2_pre)).std(1) < 0.02
    okb &= blocks(nbr(c.s2_post)).std(1) < 0.02
    spk = None
    if okb.sum() >= 5:
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(okb)
        if len(idx) > 400:
            idx = rng.choice(idx, 400, replace=False)
        spk = {"g3": lcb[idx, 0]}
        for nm, x in (("vv", vv0), ("vh", vh0)):
            xb = blocks(x)[idx]
            lin = 10 ** (xb / 10)
            spk[f"std_{nm}"] = xb.std(1)
            spk[f"enl_{nm}"] = lin.mean(1) ** 2 / np.maximum(lin.var(1), 1e-12)
            spk[f"dev_{nm}"] = (xb - xb.mean(1, keepdims=True))[:, ::5].ravel()
        for nm in ("dvv", "dvh", "dnbr"):
            src = d_raw if nm == "dnbr" else feats[nm]
            xb = blocks(src)[idx]
            spk[f"pxstd_{nm}"] = xb.std(1)
            spk[f"bmean_{nm}"] = xb.mean(1) - np.median(xb.mean(1))
        spk["g3"] = group3_index(spk["g3"])
    out["speckle"] = spk

    # --- граница гари: «срезанная» (рядом r1 & не исключено, но не гарь) vs естественная
    st4 = ndi.generate_binary_structure(2, 1)
    cut = burned & ndi.binary_dilation(cand & ~burned, structure=st4)
    bnd = burned & ndi.binary_dilation(~burned, structure=st4)
    out["boundary"] = {"n_boundary": int(bnd.sum()), "n_cut": int(cut.sum())}
    pts = np.argwhere(cut)
    out["circles"] = ransac_circles(pts, seed) if len(pts) >= 60 else []

    # --- связь с горячими точками AF
    out["hot"] = None
    if hot_rc is not None and len(hot_rc):
        dist = hotspot_distance_km(hot_rc)
        inside = (hot_rc[:, 0] >= 0) & (hot_rc[:, 0] < 512) & (hot_rc[:, 1] >= 0) & (hot_rc[:, 1] < 512)
        hb = np.bincount(binned(dist[burned], 0, DIST_STEP, DIST_NB), minlength=DIST_NB)
        hn = np.bincount(binned(dist[cand & ~burned], 0, DIST_STEP, DIST_NB), minlength=DIST_NB)
        iou = []
        for R in RADII_KM:
            p = cand & (dist <= R)
            iou.append((int((p & burned).sum()), int((p & ~burned).sum()), int((~p & burned).sum())))
        cc = []
        for ci in out["circles"]:
            cc.append(float(np.min(np.hypot(hot_rc[:, 0] - ci["cy"], hot_rc[:, 1] - ci["cx"])) * PX_KM))
        out["hot"] = {"n_hot": len(hot_rc), "n_hot_inside": int(inside.sum()), "h_burned": hb, "h_cand_nb": hn,
                      "iou_counts": np.array(iou), "burned_dist_q": np.percentile(dist[burned], [50, 90]) if nb else None,
                      "circle_center_to_hot_km": cc}
    return out


def bs_test_worker(args):
    root, cid = args
    c = load_bs(root, cid, with_mask=False)
    ex = excluded(c.s2_pre[9], c.s2_post[9])
    vv0, vh0 = c.s1_pre
    vv1, vh1 = c.s1_post
    return {"cid": cid, "excluded": float(ex.mean()),
            "s1_zero_pre": float(((vv0 == 0) | (vh0 == 0)).mean()), "s1_zero_post": float(((vv1 == 0) | (vh1 == 0)).mean())}


def af_test_worker(args):
    root, cid = args
    c = load_af(root, cid)
    v = c.viirs
    I4, I5 = v[3], v[4]
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(I4) & np.isfinite(I5) & (v[7] > 0.5)
        dT = I4 - I5
        day = ~(np.nan_to_num(v[5], nan=0) > NIGHT_SZA)
        rule = valid & np.where(day, (I4 > DAY_POT[0]) & (dT > DAY_POT[1]), (I4 > NIGHT_POT[0]) & (dT > NIGHT_POT[1]))
    return {"cid": cid, "nanfrac": float((~np.isfinite(I4)).mean()), "night": bool(np.nanmedian(v[5]) > NIGHT_SZA),
            "rule_cnt": int(rule.sum())}


# ============================================================================ AF -> BS
def map_hotspots(af: dict, bs_meta: pd.DataFrame, days: int = 3, margin_px: int = 500):
    from pyproj import Transformer

    tcache = {}

    def tr(src, dst):
        if (src, dst) not in tcache:
            tcache[(src, dst)] = Transformer.from_crs(f"EPSG:{src}", f"EPSG:{dst}", always_xy=True)
        return tcache[(src, dst)]

    res, covered = {}, {}
    for _, row in bs_meta.iterrows():
        e = int(row.epsg)
        t0 = pd.Timestamp(row.date_pre, tz="UTC") - pd.Timedelta(days=days)
        t1 = pd.Timestamp(row.date_post, tz="UTC") + pd.Timedelta(days=days + 1)
        cx, cy = (row.x_min + row.x_max) / 2, (row.y_min + row.y_max) / 2
        n_cov = 0
        for fp in af["footprints"]:
            if not (t0 <= fp["time"] < t1):
                continue
            x, y = (cx, cy) if fp["epsg"] == e else tr(e, fp["epsg"]).transform(cx, cy)
            b = fp["bounds"]
            n_cov += int(b[0] <= x <= b[2] and b[1] <= y <= b[3])
        covered[row.chip_id] = n_cov
        rc = []
        for h in af["hot"]:
            if not (t0 <= h["time"] < t1):
                continue
            x, y = (h["x"], h["y"]) if h["epsg"] == e else tr(h["epsg"], e).transform(h["x"], h["y"])
            col = (np.asarray(x) - row.x_min) / 20.0 - 0.5
            r = (row.y_max - np.asarray(y)) / 20.0 - 0.5
            k = (col > -margin_px) & (col < 512 + margin_px) & (r > -margin_px) & (r < 512 + margin_px)
            if k.any():
                rc.append(np.c_[r[k], col[k]])
        res[row.chip_id] = np.concatenate(rc) if rc else None
    return res, covered


def lonlat(meta: pd.DataFrame):
    from pyproj import Transformer

    lon, lat = np.zeros(len(meta)), np.zeros(len(meta))
    for e in meta.epsg.unique():
        sel = (meta.epsg == e).values
        t = Transformer.from_crs(f"EPSG:{int(e)}", "EPSG:4326", always_xy=True)
        lon[sel], lat[sel] = t.transform(((meta.x_min + meta.x_max) / 2)[sel].values, ((meta.y_min + meta.y_max) / 2)[sel].values)
    return lon, lat


# ============================================================================ BS: агрегаты и рисунки
def bs_section(data_dir: Path, meta: pd.DataFrame, hot_map: dict, covered: dict, F: Facts, figdir: Path, workers: int,
               examples: list | None):
    t0 = time.time()
    tasks = [(str(data_dir), cid, hot_map.get(cid), i) for i, cid in enumerate(meta.chip_id)]
    ctx = mp.get_context("fork") if sys.platform.startswith("linux") else None
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        R = list(ex.map(bs_worker, tasks, chunksize=1))
    print(f"BS workers done in {time.time() - t0:.1f}s", flush=True)
    S = "bs"
    meta = meta.copy()
    meta["n_burned"] = [r["n_burned"] for r in R]
    n_chips = len(R)
    tot_px = n_chips * 512 * 512
    nb = meta.n_burned.sum()
    sev_px = np.sum([r["sev_px"] for r in R], 0)
    F.put(S, "n_chips", n_chips, "Число чипов BS (Sentinel-2/1, 20 м, 512x512) в train; один чип = один пожар")
    F.put(S, "mask_nodata_px", sum(r["n255"] for r in R), "Пикселей со значением 255 (nodata) в масках")
    F.put(S, "burned_share_pct", 100 * nb / tot_px, "Доля пикселей гари (класс >= 1) среди всех пикселей, %")
    F.put(S, "severity_px", sev_px, "Пикселей по классам эталона 0/1/2/3")
    F.put(S, "severity_share_of_burned_pct", 100 * sev_px[1:] / sev_px[1:].sum(), "Доли степеней 1/2/3 среди гари, % (постановка: 40,0/37,1/22,9)")
    ch_share = meta.n_burned / (512 * 512)
    F.put(S, "burned_share_per_chip", {"median_pct": 100 * ch_share.median(), "p10_pct": 100 * ch_share.quantile(0.1),
                                       "p90_pct": 100 * ch_share.quantile(0.9), "min_pct": 100 * ch_share.min(), "max_pct": 100 * ch_share.max(),
                                       "chips_without_burn": int((meta.n_burned == 0).sum())}, "Доля гари в чипе")
    area = meta.n_burned * PX_HA
    F.put(S, "burned_area_ha_per_chip", {"median": area.median(), "p10": area.quantile(0.1), "p90": area.quantile(0.9), "max": area.max()},
          "Площадь гари на чип, га")
    if "burn_area_ha" in meta:
        F.put(S, "meta_burn_area_max_abs_diff_ha", np.abs(meta.burn_area_ha - area).max(), "Расхождение meta.burn_area_ha и маски, га")
    dpre, dpost = pd.to_datetime(meta.date_pre), pd.to_datetime(meta.date_post)
    gap = (dpost - dpre).dt.days
    F.put(S, "date_gap_days", {"min": gap.min(), "median": gap.median(), "max": gap.max()}, "Интервал между снимками до/после, дней")

    # --- сглаживание
    ag_e = {k: sum(r["agree_exact"][k] for r in R) / nb for k in SMOOTH_VARIANTS}
    ag_c = {k: sum(r["agree_clip"][k] for r in R) / nb for k in SMOOTH_VARIANTS}
    F.put(S, "rule_agreement_exact", ag_e, "Совпадение класса правила порогов с эталоном внутри гари (0 правила = ошибка)")
    F.put(S, "rule_agreement_clip", ag_c, "То же, класс правила обрезан до [1,3] (так собирается итоговая маска)")
    ed = {k: sum(r["edge"][k] for r in R) for k in R[0]["edge"]}
    er = {"edge_raw": ed["err_edge_raw"] / ed["n_edge"], "interior_raw": ed["err_inter_raw"] / ed["n_inter"],
          "edge_median3": ed["err_edge_median3"] / ed["n_edge"], "interior_median3": ed["err_inter_median3"] / ed["n_inter"],
          "edge_px_share": ed["n_edge"] / (ed["n_edge"] + ed["n_inter"])}
    F.put(S, "rule_error_edge_vs_interior", er, "Доля ошибок класса правила (обрезан до 1..3) на границах классов эталона vs внутри (raw и медиана 3x3)")
    lgb = np.sum([r["lc_group_burned"] for r in R], 0)
    lva = {nm: np.sum([r["lc_variant_agree"][nm] for r in R], 0) for nm in R[0]["lc_variant_agree"]}
    F.put(S, "landcover_variant_agreement", {nm: {"all": v.sum() / lgb.sum(), **{g: v[i] / max(lgb[i], 1) for i, g in enumerate(GROUPS)}}
                                             for nm, v in lva.items()},
          "Совпадение класса правила (медиана 3x3, обрезан до 1..3) внутри гари при разных картах покрова для выбора порогов: "
          "попиксельная, мажоритарная 3/5/9, лес и болота с порогами трав, доминирующая группа чипа")

    # --- правило ∩ область
    cf = {k: sum(r["conf"][k] for r in R) for k in R[0]["conf"]}
    F.put(S, "rule_confusion", cf, "r1: dNBR выше порога «не горело» своей группы; cand = r1 и SCL не исключён; _b/_nb: эталон гарь/не гарь")
    F.put(S, "burned_satisfy_rule_pct", 100 * cf["r1_b"] / nb, "Доля пикселей гари, где правило даёт класс >= 1, %")
    F.put(S, "rule1_labeled_burned_pct", 100 * cf["r1_b"] / (cf["r1_b"] + cf["r1_nb"]), "Доля размеченных гарью среди всех r1-пикселей, %")
    F.put(S, "cand_labeled_burned_pct", 100 * cf["cand_b"] / (cf["cand_b"] + cf["cand_nb"]), "То же после исключения SCL, %")
    F.put(S, "burned_in_excluded_scl_px", cf["ex_b"], "Пикселей гари в исключаемых SCL (0,3,6,9,11 на любой дате)")
    rb = np.sum([r["rule_burn_counts"] for r in R], 0)
    rc = {k: np.sum([r["rule_counts"][k] for r in R], 0) for k in (1, 2, 3)}
    iou_b = rb[0] / rb.sum()
    miou = np.mean([rc[k][0] / rc[k].sum() for k in (1, 2, 3)])
    F.put(S, "rule_only_scores", {"iou_burn": iou_b, "miou_sev": miou, **{f"iou_sev{k}": rc[k][0] / rc[k].sum() for k in (1, 2, 3)}},
          "Правило без ограничения области (r1 и не исключено, класс = правило): IoU гари и mIoU степеней")

    # --- SCL
    sc = {k: np.sum([r["scl"][k] for r in R], 0) for k in R[0]["scl"]}
    scl_tab = {}
    for when in ("pre", "post"):
        scl_tab[when] = {str(s): {"name": SCL_RU[s], "r1_px": int(sc[f"{when}_r1"][s]), "burned_share": sc[f"{when}_r1_b"][s] / sc[f"{when}_r1"][s] if sc[f"{when}_r1"][s] else None,
                                  "all_px_share_pct": 100 * sc[f"{when}_all"][s] / sc[f"{when}_all"].sum()} for s in range(12)}
    F.put(S, "scl_burned_share_among_r1", scl_tab, "Доля размеченных гарью среди r1-пикселей по классу SCL (до/после)")
    exs = np.array([r["cloud_defs"]["excluded"] for r in R])
    F.put(S, "excluded_scl_share", {"mean_pct": 100 * exs.mean(), "median_pct": 100 * np.median(exs), "max_pct": 100 * exs.max()},
          "Доля пикселей чипа с исключаемым SCL (0,3,6,9,11 до или после), %")

    # --- облачность и meta.cloud_frac
    cd = pd.DataFrame([r["cloud_defs"] for r in R])
    mae = {k: float(np.abs(cd[k].values - meta.cloud_frac.values).mean()) for k in cd}
    best = min(mae, key=mae.get)
    F.put(S, "cloud_frac_definition_mae", mae, "MAE между meta.cloud_frac и долями SCL при разных определениях облачности (коды SCL; pre/post/prepost = «до»/«после»/любая)")
    F.put(S, "cloud_frac_best_definition", {"name": best, "mae": mae[best], "corr": np.corrcoef(cd[best], meta.cloud_frac)[0, 1]},
          "Ближайшее к meta.cloud_frac определение облачности по SCL")
    F.put(S, "valid_plus_cloud", {"min": (meta.valid_frac + meta.cloud_frac).min(), "max": (meta.valid_frac + meta.cloud_frac).max()},
          "valid_frac + cloud_frac (ровно 1 => valid_frac = 1 − cloud_frac)")
    F.put(S, "cloud_frac", {"median_pct": 100 * meta.cloud_frac.median(), "mean_pct": 100 * meta.cloud_frac.mean(), "max_pct": 100 * meta.cloud_frac.max(),
                            "share_gt10_pct": 100 * (meta.cloud_frac > 0.1).mean(), "share_zero_pct": 100 * (meta.cloud_frac == 0).mean()},
          "Облачность чипов BS (meta.cloud_frac)")
    rho, p = stats.spearmanr(meta.cloud_frac, ch_share)
    F.put(S, "cloud_vs_burned_share_spearman", {"rho": rho, "p": p}, "Ранговая корреляция облачности и доли гари в чипе")
    hi = meta.cloud_frac > 0.1
    F.put(S, "burned_share_by_cloud", {"cloud_le10_median_pct": 100 * ch_share[~hi].median(), "cloud_gt10_median_pct": 100 * ch_share[hi].median(),
                                       "n_gt10": int(hi.sum()),
                                       "mannwhitney_p": stats.mannwhitneyu(ch_share[~hi], ch_share[hi]).pvalue if 0 < hi.sum() < len(hi) else None},
          "Медианная доля гари в чипах с облачностью <=10% и >10%")
    cand_ex_share = np.array([(r["n_r1"] - r["n_cand"]) / max(r["n_r1"], 1) for r in R])
    F.put(S, "r1_under_excluded_share", {"median_pct": 100 * np.median(cand_ex_share), "p90_pct": 100 * np.percentile(cand_ex_share, 90)},
          "Доля r1-пикселей чипа, попавших под исключаемые SCL, %")

    # --- покров, годы, долгота
    gc = np.sum([r["grp_cls"] for r in R], 0)  # (group, cls)
    grp_tab = {}
    for gi, g in enumerate(GROUPS):
        b = gc[gi, 1:].sum()
        grp_tab[g] = {"territory_share_pct": 100 * gc[gi].sum() / gc.sum(), "burned_share_of_group_pct": 100 * b / max(gc[gi].sum(), 1),
                      "share_of_all_burned_pct": 100 * b / nb, "sev_share_pct": 100 * gc[gi, 1:] / max(b, 1), "burned_px": int(b)}
    F.put(S, "by_landcover_group", grp_tab, "Гарь и степень по группам покрова (пороги dNBR свои для каждой группы)")
    lcc = {}
    for r in R:
        for k, v in r["lc_cls"].items():
            lcc[k] = lcc.get(k, 0) + v
    F.put(S, "by_landcover_code", {str(k): {"name": LC_RU.get(k, str(k)), "px": int(v.sum()), "burned_share_pct": 100 * v[1:].sum() / max(v.sum(), 1),
                                            "sev_share_pct": 100 * v[1:] / max(v[1:].sum(), 1)} for k, v in sorted(lcc.items())},
          "Гарь и степень по кодам WorldCover")
    meta["year"] = dpost.dt.year
    lon, lat = lonlat(meta)
    meta["lon"], meta["lat"] = lon, lat
    lob = np.floor(lon / 2) * 2
    meta["lon_band"] = [f"{int(a)}–{int(a) + 2}° в.д." for a in lob]
    meta["lon_band_lo"] = lob
    sevm = np.array([r["sev_px"] for r in R])
    yr_tab, lb_tab = {}, {}
    for key, tab, col in (("year", yr_tab, "year"), ("lon_band", lb_tab, "lon_band")):
        for v in sorted(meta[col].unique(), key=lambda z: (meta.loc[meta[col] == z, "lon_band_lo"].iloc[0] if col == "lon_band" else z)):
            sel = (meta[col] == v).values
            sp = sevm[sel].sum(0)
            tab[str(v)] = {"chips": int(sel.sum()), "burned_share_pct": 100 * sp[1:].sum() / (sel.sum() * 512 * 512),
                           "sev_share_pct": 100 * sp[1:] / max(sp[1:].sum(), 1), "median_area_ha": float(np.median(sevm[sel, 1:].sum(1) * PX_HA)),
                           "median_cloud_pct": 100 * meta.cloud_frac[sel].median()}
    F.put(S, "by_year", yr_tab, "Гарь и степень по годам (дата «после»)")
    F.put(S, "by_lon_band", lb_tab, "Гарь и степень по полосам долготы центра чипа")
    F.put(S, "extent", {"lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
                        "epsg": meta.epsg.value_counts().to_dict()}, "Охват территории train BS (центры чипов)")
    kw = stats.kruskal(*[sevm[(meta.year == y).values, 3] / np.maximum(sevm[(meta.year == y).values, 1:].sum(1), 1) for y in sorted(meta.year.unique())])
    F.put(S, "sev3_share_by_year_kruskal_p", kw.pvalue, "Критерий Краскела-Уоллиса: доля сильной степени в чипе различается по годам?")

    # --- dNBR гистограммы
    dh = np.sum([r["dn_hist"] for r in R], 0)
    within = {}
    for gi, g in enumerate(GROUPS):
        th = THRESHOLDS[g]
        edges = [-np.inf, *th, np.inf]
        centers = DN_LO + (np.arange(DN_NB) + 0.5) * DN_STEP
        within[g] = {}
        for k in (1, 2, 3):
            h = dh[gi, k]
            inb = h[(centers >= edges[k]) & (centers < edges[k + 1])].sum()
            within[g][str(k)] = inb / max(h.sum(), 1)
    emp = {}
    for gi, g in enumerate(GROUPS):
        e = {"class1_p0.5": hist_quantile(dh[gi, 1], DN_LO, DN_STEP, 0.005)}
        for k in (1, 2):
            a, b = dh[gi, k].astype(float), dh[gi, k + 1].astype(float)
            # порог t на границе бина j: ошибки = класс k справа от t + класс k+1 слева от t
            err = (a.sum() - np.cumsum(np.r_[0, a])) + np.cumsum(np.r_[0, b])
            j = int(err.argmin())
            e[f"{k}|{k + 1}"] = DN_LO + j * DN_STEP
            e[f"{k}|{k + 1}_err_share"] = err[j] / max(a.sum() + b.sum(), 1)
        e["statement"] = THRESHOLDS[g]
        emp[g] = e
    F.put(S, "dnbr_empirical_thresholds", emp, "Эмпирические границы классов по медианному dNBR (минимум ошибок между соседними классами) vs пороги постановки")
    F.put(S, "dnbr_within_class_interval", within, "Доля пикселей класса k, у которых медианный dNBR в интервале порогов класса k (по гистограмме, шаг 0,005)")

    # --- SAR
    sh = {f: np.sum([r["sar_hist"][f] for r in R], 0) for f in SAR_FEATS}  # (subset, g3, label, bins)
    auc = {}
    for si, sub in enumerate(("all", "rule1")):
        auc[sub] = {}
        for gname, gsel in (("forest", [0]), ("grass_crop", [1]), ("all", [0, 1, 2])):
            auc[sub][gname] = {}
            for f in SAR_FEATS:
                pos = sh[f][si, gsel, 1].sum(0)
                neg = sh[f][si, gsel, 0].sum(0)
                a = auc_from_hist(pos, neg)
                auc[sub][gname][f] = {"auc": a, "sep": max(a, 1 - a), "direction": "burned higher" if a >= 0.5 else "burned lower"}
    F.put("sar", "auc", auc, "AUC одного признака (гарь vs не гарь), sep = max(AUC, 1−AUC); all — все пиксели с покрытием S1 и без исключ. SCL; rule1 — только r1")
    med = {}
    for gname, gsel in (("forest", [0]), ("grass_crop", [1]), ("all", [0, 1, 2])):
        med[gname] = {}
        for f in SAR_FEATS:
            lo, st, _ = SAR_BINS[f]
            mb = hist_quantile(sh[f][0, gsel, 1].sum(0), lo, st, 0.5)
            mn = hist_quantile(sh[f][0, gsel, 0].sum(0), lo, st, 0.5)
            med[gname][f] = {"median_burned": mb, "median_unburned": mn, "diff": mb - mn}
    F.put("sar", "median_change", med, "Медианы признаков для гари и не гари (дБ для SAR)")
    z = pd.DataFrame([{"pre": r["s1_zero"]["pre"], "post": r["s1_zero"]["post"]} for r in R])
    zmax = z.max(axis=1)
    F.put("sar", "coverage", {"chips_partial_pct": 100 * ((zmax > 0.001) & (zmax < 0.999)).mean(), "chips_no_s1_pct": 100 * (zmax >= 0.999).mean(),
                              "chips_partial_n": int(((zmax > 0.001) & (zmax < 0.999)).sum()), "chips_no_s1_n": int((zmax >= 0.999).sum()),
                              "mean_nocov_px_pct": 100 * zmax.mean(), "pre_partial_n": int(((z.pre > 0.001)).sum()), "post_partial_n": int((z.post > 0.001).sum())},
          "Покрытие Sentinel-1: чипы с частичным (0 = нет данных) или полным отсутствием радара")
    spk = [r["speckle"] for r in R if r["speckle"] is not None]
    cat = lambda k: np.concatenate([s[k] for s in spk])
    sp_f = {"n_blocks": len(cat("std_vv")), "n_chips": len(spk)}
    for nm in ("vv", "vh"):
        sp_f[f"std_{nm}_db_median"] = np.median(cat(f"std_{nm}"))
        sp_f[f"enl_{nm}_median"] = np.median(cat(f"enl_{nm}"))
    rob = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    for nm in ("dvv", "dvh", "dnbr"):
        sp_f[f"pixel_noise_{nm}"] = np.median(cat(f"pxstd_{nm}"))
        sp_f[f"block7_noise_{nm}"] = rob(cat(f"bmean_{nm}"))
    for nm in ("dvv", "dvh", "dnbr"):
        diff = abs(med["all"][nm]["diff"])
        sp_f[f"snr_pixel_{nm}"] = diff / sp_f[f"pixel_noise_{nm}"]
        sp_f[f"snr_block7_{nm}"] = diff / sp_f[f"block7_noise_{nm}"]
    F.put("sar", "speckle", sp_f, "Шум в однородных блоках 7x7 без гари: СКО дБ, ENL (линейная мощность), шум разности на пиксель и после усреднения 7x7, SNR = |Δмедиан гарь−фон| / шум")

    # --- область пожара: горячие точки и окружности
    circ = [ci for r in R for ci in r["circles"]]
    radii = np.array([ci["r"] * PX_KM for ci in circ])
    bd = {k: sum(r["boundary"][k] for r in R) for k in ("n_boundary", "n_cut")}
    chips_c = sum(1 for r in R if r["circles"])
    inl = sum(ci["n_inl"] for ci in circ)
    F.put("region", "cut_boundary_share_pct", 100 * bd["n_cut"] / bd["n_boundary"],
          "Доля граничных пикселей гари, за которыми правило разрешает гарь, но эталон «не горело» (срезанная граница), %")
    F.put("region", "ransac_circles", {"n_circles": len(circ), "chips_with_circles": chips_c, "radius_km_median": np.median(radii) if len(radii) else None,
                                       "radius_km_p25": np.percentile(radii, 25) if len(radii) else None, "radius_km_p75": np.percentile(radii, 75) if len(radii) else None,
                                       "cut_px_on_circles_pct": 100 * inl / max(bd["n_cut"], 1)},
          "RANSAC-окружности по срезанной границе (допуск 1,5 пикс., дуга >= 60°, R 0,2-8 км)")
    H = [r for r in R if r["hot"] is not None]
    Hin = [r for r in H if r["hot"]["n_hot_inside"] > 0]
    F.put("region", "af_bs_link", {"bs_chips_covered_by_af_in_window": int(sum(v > 0 for v in covered.values())),
                                    "bs_chips_with_hotspots_within_10km": len(H), "bs_chips_with_hotspots_inside": len(Hin)},
          "Связка BS–AF: чипы BS, покрытые чипом AF с датой в [date_pre−3 дн, date_post+3 дн], и с эталонными очагами AF рядом/внутри")
    region_facts = {}
    if Hin:
        hb = np.sum([r["hot"]["h_burned"] for r in Hin], 0)
        hn = np.sum([r["hot"]["h_cand_nb"] for r in Hin], 0)
        q = {f"burned_dist_km_p{int(qq * 100)}": hist_quantile(hb, 0, DIST_STEP, qq) for qq in (0.5, 0.8, 0.9, 0.95)}
        q.update({f"cand_unburned_dist_km_p{int(qq * 100)}": hist_quantile(hn, 0, DIST_STEP, qq) for qq in (0.1, 0.5)})
        pb = hb / np.maximum(hb + hn, 1)
        q["p_burned_given_cand_by_km"] = {f"{a:.0f}-{a + 1:.0f}": (hb[int(a / DIST_STEP):int((a + 1) / DIST_STEP)].sum()
                                                                   / max((hb + hn)[int(a / DIST_STEP):int((a + 1) / DIST_STEP)].sum(), 1)) for a in range(0, 8)}
        ic = np.sum([r["hot"]["iou_counts"] for r in Hin], 0)  # (R, 3)
        iou = ic[:, 0] / ic.sum(1)
        rb_h = np.sum([r["rule_burn_counts"] for r in Hin], 0)
        q["iou_by_radius"] = {f"{Rk:.2f}": v for Rk, v in zip(RADII_KM, iou)}
        q["best_radius_km"] = RADII_KM[int(iou.argmax())]
        q["best_iou"] = iou.max()
        q["rule_only_iou_same_chips"] = rb_h[0] / rb_h.sum()
        per_best, tab = [], []
        i3 = int(np.argmin(np.abs(RADII_KM - 3.0)))
        for r in Hin:
            c3 = r["hot"]["iou_counts"]
            iou_c = c3[:, 0] / np.maximum(c3.sum(1), 1)
            per_best.append(RADII_KM[int(iou_c.argmax())])
            rbc = r["rule_burn_counts"]
            tab.append({"cid": r["cid"], "n_hot_inside": r["hot"]["n_hot_inside"], "n_cut": r["boundary"]["n_cut"],
                        "burned_px": r["n_burned"], "iou_rule_only": rbc[0] / max(sum(rbc), 1), "iou_buffer_3km": iou_c[i3],
                        "best_R_km": RADII_KM[int(iou_c.argmax())], "best_iou": iou_c.max()})
            r["_iou3"] = iou_c[i3]
        q["per_chip"] = tab
        q["per_chip_best_radius_km_median"] = np.median(per_best)
        q["per_chip_best_radius_km_iqr"] = np.percentile(per_best, [25, 75])
        cc = np.array([d for r in Hin for d in r["hot"]["circle_center_to_hot_km"]])
        rr = np.array([ci["r"] * PX_KM for r in Hin for ci in r["circles"]])
        if len(cc):
            q["circle_center_to_nearest_hotspot_km_median"] = np.median(cc)
            q["circle_center_within_1km_pct"] = 100 * (cc <= 1.0).mean()
            q["circle_center_within_radius_pct"] = 100 * (cc < rr).mean()
        region_facts = q
        F.put("region", "hotspot_distance", q, "Расстояния до ближайшего эталонного очага AF (чипы BS с очагами внутри), IoU(r1 ∩ буфер R) и центры окружностей")
    print(f"BS aggregates done in {time.time() - t0:.1f}s", flush=True)

    fig_bs_dnbr(dh, emp, figdir)
    fig_bs_smoothing(ag_c, er, figdir)
    fig_bs_scl(sc, figdir)
    fig_bs_region(data_dir, R, Hin, hot_map, region_facts, radii, figdir, examples, F)
    fig_bs_severity(sev_px, gc, meta, sevm, yr_tab, lb_tab, grp_tab, figdir)
    fig_sar(sh, auc, spk, z, sp_f, figdir)
    return {"meta": meta, "cd": cd, "ch_share": ch_share, "zmax": zmax, "exs": exs}


def fig_bs_dnbr(dh, emp, figdir):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.9), constrained_layout=True, sharey=True)
    edges = DN_LO + np.arange(DN_NB + 1) * DN_STEP
    for ax, (gi, g) in zip(axes, enumerate(GROUPS)):
        h0 = dh[gi, 0]
        ax.stairs(np.maximum(h0, 0.5), edges, color=C_GREY, lw=1.0, label="0 — не горело")
        for k in (1, 2, 3):
            ax.stairs(np.maximum(dh[gi, k], 0.5), edges, color=C_SEV[k], fill=True, alpha=0.85, label=f"{k} — {SEV_RU[k]}")
        for j, t in enumerate(THRESHOLDS[g]):
            ax.axvline(t, c=C_INK, ls="--", lw=1)
            ax.text(t, 1.015 + 0.06 * (j % 2), f"{t:g}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color=C_INK)
        for k in (1, 2):
            ax.plot(emp[g][f"{k}|{k + 1}"], 0.0, marker="^", ms=8, color=C_INK2, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_yscale("log")
        ax.set_ylim(0.8, dh.max() * 3)
        ax.set_xlim(-0.2, 1.1)
        n = dh[gi, 1:].sum()
        ax.set_title(f"{GROUP_RU[g]}\n(гарь: {fi(n)} пикс.)", pad=26)
        ax.set_xlabel("dNBR (медиана 3×3)")
    axes[0].set_ylabel("пикселей (лог. шкала)")
    h, l = axes[-1].get_legend_handles_labels()
    h += [Line2D([], [], ls="--", c=C_INK), Line2D([], [], marker="^", ls="", color=C_INK2, ms=8)]
    l += ["пороги постановки", "эмпирич. граница 1|2, 2|3"]
    fig.legend(h, l, loc="outside lower center", ncol=6, title="класс эталона и пороги", fontsize=10, title_fontsize=10)
    fig.suptitle("BS: классы эталона — это интервалы медианного dNBR между порогами постановки (рис. 6), свои для каждого покрова")
    save(fig, figdir / "eda_bs_dnbr_thresholds.png")


def fig_bs_smoothing(ag, er, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ks = sorted(SMOOTH_VARIANTS, key=lambda k: ag[k])
    y = np.arange(len(ks))
    cols = [C_FIRE if k == "median3" else C_GREY for k in ks]
    ax.barh(y, [100 * ag[k] for k in ks], color=cols, height=0.6)
    for yi, k in zip(y, ks):
        ax.text(100 * ag[k] + 0.15, yi, f"{100 * ag[k]:.1f}%", va="center", fontsize=10, color=C_INK)
    ax.set_yticks(y, [SMOOTH_RU[k] for k in ks])
    lo = np.floor(100 * min(ag.values()) - 2)
    ax.set_xlim(lo, 100.5)
    ax.set_xlabel("совпадение класса правила (обрезан до 1..3)\nс эталоном внутри гари, %")
    ax.set_title("а) Какое сглаживание dNBR использовал разметчик")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    x = np.arange(2)
    ax.bar(x - 0.2, [100 * er["interior_raw"], 100 * er["edge_raw"]], 0.38, color=C_GREY, label="без сглаживания")
    ax.bar(x + 0.2, [100 * er["interior_median3"], 100 * er["edge_median3"]], 0.38, color=C_FIRE, label="медиана 3×3")
    for xi, v in zip(np.r_[x - 0.2, x + 0.2], [er["interior_raw"], er["edge_raw"], er["interior_median3"], er["edge_median3"]]):
        ax.text(xi, 100 * v + 0.2, f"{100 * v:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x, ["внутри классов", f"на границах классов\n({100 * er['edge_px_share']:.0f}% пикселей гари)"])
    ax.set_ylabel("доля ошибок класса, %")
    ax.set_title("б) Ошибки правила сосредоточены на границах")
    ax.legend(loc="upper left")
    fig.suptitle("BS: эталон степени = пороги по dNBR, сглаженному медианой 3×3")
    save(fig, figdir / "eda_bs_smoothing.png")


def fig_bs_scl(sc, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True, sharey=True)
    codes = [c for c in range(12) if sc["pre_r1"][c] + sc["post_r1"][c] > 0]
    y = np.arange(len(codes))[::-1]
    for ax, when, title in ((axes[0], "pre", "а) SCL снимка «до»"), (axes[1], "post", "б) SCL снимка «после»")):
        tot, b = sc[f"{when}_r1"], sc[f"{when}_r1_b"]
        share = np.array([b[c] / tot[c] if tot[c] else np.nan for c in codes])
        for yi, c in zip(y, codes):
            if c in SCL_EXCLUDE:
                ax.axhspan(yi - 0.5, yi + 0.5, color=C_LGREY, alpha=0.6, lw=0, zorder=0)
        ax.barh(y, 100 * np.nan_to_num(share), 0.62, color=[C_INK2 if c in SCL_EXCLUDE else C_FIRE for c in codes])
        for yi, c, v in zip(y, codes, share):
            n = tot[c]
            ns = f"{n / 1e6:.2f} млн" if n >= 1e5 else f"{n / 1e3:.0f} тыс."
            ax.text((0 if np.isnan(v) else 100 * v) + 1, yi, ("—" if np.isnan(v) else f"{100 * v:.1f}%") + f"  (n = {ns})",
                    va="center", fontsize=9.5, color=C_INK)
        ax.set_xlim(0, 100)
        ax.set_xlabel("размечено гарью среди пикселей с dNBR ≥ порога, %")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{SCL_RU[c]} ({c})" for c in codes])
    handles = [Patch(color=C_INK2, label=f"исключаем жёстко: SCL ∈ {{{', '.join(map(str, SCL_EXCLUDE))}}} на любой из дат (доля гари ≤ 0.5%)"),
               Patch(color=C_FIRE, label="прочие классы")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    fig.suptitle("BS: пиксели с SCL «нет данных», «тень облака», «вода», «облако (выс.)», «снег» в эталоне — «не горело»")
    save(fig, figdir / "eda_bs_scl.png")


def fig_bs_region(data_dir, R, Hin, hot_map, q, radii, figdir, examples, F):
    byid = {r["cid"]: r for r in R}
    if not examples and all(c in {r["cid"] for r in Hin} for c in DEFAULT_REGION_EXAMPLES):
        examples = list(DEFAULT_REGION_EXAMPLES)
    if not examples:
        # наглядные: много очагов, длинная срезанная граница, буфер 3 км хорошо объясняет эталон
        # наглядные: много очагов, заметная срезанная граница, буфер 3 км хорошо объясняет эталон и даёт большой прирост к правилу
        pool = [r for r in Hin if r["hot"]["n_hot_inside"] >= 5 and r["boundary"]["n_cut"] >= 800 and r.get("_iou3", 0) >= 0.75] or Hin
        gain = lambda r: r.get("_iou3", 0) - r["rule_burn_counts"][0] / max(sum(r["rule_burn_counts"]), 1)
        examples = [r["cid"] for r in sorted(pool, key=lambda r: -gain(r))[:3]]
    F.put("region", "example_chips", examples, "Чипы на рис. eda_bs_region.png")
    fig = plt.figure(figsize=(16, 11.5), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.2, 1], hspace=0.02)
    cmap = ListedColormap(["#f4f3ef", "#b8b6ae", "#f4a99c", "#8c1d18"])
    axs = top.subplots(1, 3)
    for j, (ax, cid) in enumerate(zip(axs, examples)):
        c = load_bs(data_dir, cid, with_mask=True)
        m = c.mask.copy()
        m[m == 255] = 0
        _, sev, ex = bs_maps(c, c.aux[2].astype(np.int32))
        img = np.zeros(m.shape, np.uint8)
        img[ex] = 1
        img[(sev >= 1) & ~ex & (m == 0)] = 2
        img[m > 0] = 3
        ax.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest", extent=(0, 512 * PX_KM, 512 * PX_KM, 0))
        h = hot_map.get(cid)
        if h is not None:
            ax.scatter(h[:, 1] * PX_KM + PX_KM / 2, h[:, 0] * PX_KM + PX_KM / 2, s=24, c=C_HOT, edgecolors="#004b5a", lw=0.6, zorder=5)
        if h is not None and q:
            dist = hotspot_distance_km(h)
            xy = (np.arange(512) + 0.5) * PX_KM
            ax.contour(xy, xy, dist, levels=[q["best_radius_km"]], colors=[C_INK], linestyles="--", linewidths=1.3)
        ax.set_xlim(0, 512 * PX_KM)
        ax.set_ylim(512 * PX_KM, 0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("км")
        if j == 0:
            ax.set_ylabel("км")
        nh = byid[cid]["hot"]["n_hot_inside"] if byid[cid]["hot"] else 0
        ax.set_title(f"{'абв'[j]}) {cid} (очагов в чипе: {nh})")
    handles = [Patch(color="#8c1d18", label="гарь (эталон)"), Patch(color="#f4a99c", label="dNBR ≥ порога, но эталон «не горело»"),
               Patch(color="#b8b6ae", label="исключено по SCL"),
               Line2D([], [], marker="o", ls="", color=C_HOT, mec="#004b5a", label="очаги AF (эталон VIIRS) в окне дат"),
               Line2D([], [], ls="--", c=C_INK, label=f"буфер {q.get('best_radius_km', 3.0):.1f} км вокруг очагов (оптимум IoU)")]
    top.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=10)
    fig.suptitle("BS: эталон = правило dNBR ∩ «область события» (круги вокруг очагов пожара)", fontsize=14)
    axb = bot.subplots(1, 3)
    # г) расстояния
    ax = axb[0]
    if Hin:
        hb = np.sum([r["hot"]["h_burned"] for r in Hin], 0)
        hn = np.sum([r["hot"]["h_cand_nb"] for r in Hin], 0)
        edges = np.arange(DIST_NB + 1) * DIST_STEP
        ax.stairs(hb / hb.sum() / DIST_STEP, edges, color="#8c1d18", fill=True, alpha=0.75, label="гарь")
        ax.stairs(hn / hn.sum() / DIST_STEP, edges, color="#e0705f", lw=1.6, label="dNBR ≥ порога, не гарь")
        p90 = q["burned_dist_km_p90"]
        ax.axvline(p90, c=C_INK, ls="--", lw=1)
        ax.text(p90 + 0.2, 0.95, f"90% гари\nв пределах {p90:.1f} км", transform=ax.get_xaxis_transform(), va="top", fontsize=9.5)
        ax.set_xlim(0, 12)
        ax.set_xlabel("расстояние до ближайшего очага AF, км")
        ax.set_ylabel("плотность")
        ax.legend(loc="center right")
        ax.set_title(f"г) Расстояние до очага ({len(Hin)} чипов BS с очагами)")
    # д) IoU vs радиус буфера
    ax = axb[1]
    if Hin:
        rs = np.array([float(k) for k in q["iou_by_radius"]])
        iou = np.array(list(q["iou_by_radius"].values()))
        ax.plot(rs, iou, c=C_FIRE, lw=2, marker="o", ms=4)
        ax.axhline(q["rule_only_iou_same_chips"], c=C_GREY, ls="--", lw=1.2)
        ax.text(rs[-1], q["rule_only_iou_same_chips"] - 0.015, "только правило (без области)", ha="right", va="top", fontsize=9.5, color=C_INK2)
        ax.axvline(q["best_radius_km"], c=C_INK, ls=":", lw=1)
        ax.text(q["best_radius_km"] + 0.15, 0.05, f"макс. IoU = {q['best_iou']:.2f}\nпри R = {q['best_radius_km']:.2f} км",
                transform=ax.get_xaxis_transform(), fontsize=9.5)
        ax.set_xlabel("радиус буфера вокруг очагов R, км")
        ax.set_ylabel("IoU гари")
        ax.set_ylim(0, 1)
        ax.set_title("д) IoU(правило ∩ буфер R вокруг очагов)")
    # е) радиусы RANSAC
    ax = axb[2]
    if len(radii):
        ax.hist(radii, bins=np.arange(0, 8.01, 0.25), color=C_GREY, edgecolor="white")
        md, p25, p75 = np.median(radii), *np.percentile(radii, [25, 75])
        ax.axvspan(p25, p75, color=C_FIRE, alpha=0.12, lw=0)
        ax.axvline(md, c=C_FIRE, lw=1.5)
        note(ax, f"медиана {md:.1f} км\nIQR {p25:.1f}–{p75:.1f} км\nn = {len(radii)} окружн.", loc=(0.97, 0.97), ha="right")
        ax.set_xlabel("радиус окружности, км")
        ax.set_ylabel("число окружностей")
        ax.set_title("е) Радиусы дуг срезанной границы (все чипы)")
    save(fig, figdir / "eda_bs_region.png")


def fig_bs_severity(sev_px, gc, meta, sevm, yr_tab, lb_tab, grp_tab, figdir):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    # а) общая
    ax = axes[0, 0]
    sh = 100 * sev_px[1:] / sev_px[1:].sum()
    ref = (40.0, 37.1, 22.9)
    x = np.arange(3)
    ax.bar(x, sh, 0.6, color=[C_SEV[k] for k in (1, 2, 3)])
    for xi, v in zip(x, sh):
        ax.text(xi, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x, [f"{k} — {SEV_RU[k]}" for k in (1, 2, 3)])
    ax.set_ylim(0, max(sh) * 1.3)
    note(ax, f"постановка: {ref[0]} / {ref[1]} / {ref[2]}%", loc=(0.98, 0.97), ha="right")
    ax.set_ylabel("% пикселей гари")
    ax.set_title(f"а) Степень поражения: гарь = {100 * sev_px[1:].sum() / sev_px.sum():.1f}% пикселей")

    def stacked(ax, labels, shares, notes, title):
        y = np.arange(len(labels))[::-1]
        left = np.zeros(len(labels))
        for k in (1, 2, 3):
            v = np.array([s[k - 1] for s in shares])
            ax.barh(y, v, left=left, color=C_SEV[k], height=0.62, edgecolor="white", lw=1)
            for yi, l_, vv in zip(y, left, v):
                if vv >= 8:
                    ax.text(l_ + vv / 2, yi, f"{vv:.0f}", ha="center", va="center", fontsize=9, color=C_INK if k < 3 else "white")
            left += v
        for yi, nt in zip(y, notes):
            ax.text(101.5, yi, nt, va="center", fontsize=9, color=C_INK2)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% пикселей гари")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)

    stacked(axes[0, 1], [GROUP_RU[g] for g in GROUPS], [grp_tab[g]["sev_share_pct"] for g in GROUPS],
            [f"гарь {grp_tab[g]['burned_share_of_group_pct']:.0f}% покрова" for g in GROUPS], "б) По группам покрова")
    ys = list(yr_tab)
    stacked(axes[0, 2], ys, [yr_tab[y]["sev_share_pct"] for y in ys], [f"n={yr_tab[y]['chips']}" for y in ys], "в) По годам")
    lbs = list(lb_tab)
    stacked(axes[1, 0], lbs, [lb_tab[b]["sev_share_pct"] for b in lbs], [f"n={lb_tab[b]['chips']}" for b in lbs], "г) По долготе центра чипа")
    # д) площадь гари на чип
    ax = axes[1, 1]
    area = meta.n_burned.values * PX_HA
    ax.hist(area[area > 0], bins=np.logspace(np.log10(max(area[area > 0].min(), 1)), np.log10(area.max()) + 0.02, 25),
            color=C_SEV[2], edgecolor="white")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 10, 100, 1000, 5000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fi(v)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvline(np.median(area), c=C_INK, ls="--", lw=1.2)
    ax.set_xlabel("площадь гари в чипе, га (лог. шкала; чип = 10 485 га)")
    ax.set_ylabel("число чипов")
    ax.set_title("д) Площадь гари на чип")
    note(ax, f"медиана {np.median(area):.0f} га ({100 * np.median(area) / 10485.76:.1f}% чипа)\nмакс. {area.max():.0f} га", loc=(0.02, 0.97))
    # е) покров: территория vs гарь
    ax = axes[1, 2]
    x = np.arange(len(GROUPS))
    ax.bar(x - 0.2, [grp_tab[g]["territory_share_pct"] for g in GROUPS], 0.38, color=C_GREY, label="доля территории")
    ax.bar(x + 0.2, [grp_tab[g]["share_of_all_burned_pct"] for g in GROUPS], 0.38, color=C_FIRE, label="доля гари")
    ax.set_xticks(x, [GROUP_RU[g] for g in GROUPS])
    ax.set_ylabel("%")
    ax.legend()
    ax.set_title("е) Покров: территория чипов vs гарь")
    handles = [Patch(color=C_SEV[k], label=f"{k} — {SEV_RU[k]}") for k in (1, 2, 3)]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, title="степень поражения")
    fig.suptitle("BS: распределение степени поражения и различия по покрову, годам и территории")
    save(fig, figdir / "eda_bs_severity.png")


def fig_sar(sh, auc, spk, z, sp_f, figdir):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    # а) спекл
    ax = axes[0, 0]
    for nm, col in (("vv", C_INK2), ("vh", C_TEST)):
        dev = np.concatenate([s[f"dev_{nm}"] for s in spk])
        ax.hist(dev, bins=np.arange(-8, 8.01, 0.2), density=True, histtype="step", lw=1.8, color=col,
                label=f"{nm.upper()}: СКО {sp_f[f'std_{nm}_db_median']:.2f} дБ, ENL {sp_f[f'enl_{nm}_median']:.1f}")
    ax.set_xlabel("отклонение от среднего блока 7×7, дБ")
    ax.set_ylabel("плотность")
    ax.set_title(f"а) Спекл в однородных блоках ({fi(sp_f['n_blocks'])} блоков)")
    ax.legend(loc="upper left", fontsize=9)
    # б, в) ΔVH по группам
    for ax, gi, title in ((axes[0, 1], 0, "б) ΔVH, лес / кустарник"), (axes[0, 2], 1, "в) ΔVH, травы + пашня")):
        lo, st, nb = SAR_BINS["dvh"]
        edges = lo + np.arange(nb + 1) * st
        for f, ls, lw in (("dvh", "-", 1.0), ("dvh7", "-", 2.2)):
            for lab, col in ((0, C_GREY), (1, C_FIRE)):
                h = sh[f][0, gi, lab].astype(float)
                h4 = h.reshape(-1, 4).sum(1)
                e4 = edges[::4]
                ax.stairs(h4 / h4.sum() / (st * 4), e4, color=col, lw=lw, ls=ls, alpha=0.9 if f == "dvh7" else 0.6)
        a_raw, a_7 = auc["all"]["forest" if gi == 0 else "grass_crop"]["dvh"]["sep"], auc["all"]["forest" if gi == 0 else "grass_crop"]["dvh7"]["sep"]
        note(ax, f"разделимость (AUC):\nпиксель {a_raw:.2f}, 7×7 {a_7:.2f}", loc=(0.02, 0.97))
        ax.set_xlim(-8, 8)
        ax.set_xlabel("ΔVH = VH«после» − VH«до», дБ")
        ax.set_ylabel("плотность")
        ax.set_title(title)
    hl = [Line2D([], [], c=C_FIRE, lw=2, label="гарь"), Line2D([], [], c=C_GREY, lw=2, label="не гарь"),
          Line2D([], [], c=C_INK2, lw=1, label="пиксель 20 м"), Line2D([], [], c=C_INK2, lw=2.2, label="среднее 7×7")]
    axes[0, 2].legend(handles=hl, loc="upper right", fontsize=9)
    # г) AUC по признакам и покрову
    ax = axes[1, 0]
    x = np.arange(len(SAR_FEATS))
    for off, (g, col, lab) in zip((-0.2, 0.2), (("forest", C_GROUP["forest"], "лес / кустарник"), ("grass_crop", C_GROUP["crop"], "травы + пашня"))):
        v = [auc["all"][g][f]["sep"] for f in SAR_FEATS]
        ax.bar(x + off, v, 0.38, color=col, label=lab)
        for xi, vv in zip(x + off, v):
            ax.text(xi, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=8.5)
    ax.axhline(0.5, c=C_INK2, lw=0.8, ls=":")
    ax.set_xticks(x, [SAR_FEAT_RU[f] for f in SAR_FEATS])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUC (max(AUC, 1−AUC))")
    ax.set_title("г) Разделимость гарь / не гарь одним признаком")
    ax.legend(loc="upper right", fontsize=9)
    # д) AUC внутри r1
    ax = axes[1, 1]
    v = [auc["rule1"]["all"][f]["sep"] for f in SAR_FEATS]
    ax.bar(x, v, 0.6, color=[C_FIRE if f == "dnbr" else C_TEST for f in SAR_FEATS])
    for xi, vv in zip(x, v):
        ax.text(xi, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, c=C_INK2, lw=0.8, ls=":")
    ax.set_xticks(x, [SAR_FEAT_RU[f] for f in SAR_FEATS])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUC (max(AUC, 1−AUC))")
    ax.set_title("д) То же среди пикселей с dNBR ≥ порога")
    # е) покрытие S1
    ax = axes[1, 2]
    zm = z.max(axis=1).values
    part = zm[(zm > 0.001)]
    ax.hist(100 * part, bins=np.arange(0, 100.1, 5), color=C_TEST, edgecolor="white")
    ax.set_xlabel("доля пикселей чипа без данных S1 (значение 0), %")
    ax.set_ylabel("число чипов")
    ax.set_title("е) Неполное покрытие Sentinel-1")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    note(ax, (f"с пропусками S1: {len(part)} из {len(zm)} чипов ({100 * len(part) / len(zm):.0f}%)\n"
              f"полностью без S1: {int((zm >= 0.999).sum())}"), loc=(0.98, 0.97), ha="right")
    fig.suptitle("SAR (Sentinel-1): сильный спекл, слабый сигнал изменения на 20 м — только как вспомогательные сглаженные каналы")
    save(fig, figdir / "eda_sar.png")


# ============================================================================ облака и сдвиг train/test
def shift_section(test_dir: Path, af_train: dict, af_meta: pd.DataFrame, bs_train: dict, F: Facts, figdir: Path, workers: int):
    tm = pd.read_csv(test_dir / "meta.csv")
    S = "shift"
    af_t = tm[tm.kind == "af"]
    bs_t = tm[tm.kind == "bs"]
    bm = bs_train["meta"]
    ks = stats.ks_2samp(bm.cloud_frac, bs_t.cloud_frac)
    F.put(S, "n_test", {"af": len(af_t), "bs": len(bs_t)}, "Число тестовых чипов AF и BS")
    F.put(S, "bs_cloud_frac", {"train_median_pct": 100 * bm.cloud_frac.median(), "test_median_pct": 100 * bs_t.cloud_frac.median(),
                               "train_mean_pct": 100 * bm.cloud_frac.mean(), "test_mean_pct": 100 * bs_t.cloud_frac.mean(),
                               "train_max_pct": 100 * bm.cloud_frac.max(), "test_max_pct": 100 * bs_t.cloud_frac.max(),
                               "train_gt10_pct": 100 * (bm.cloud_frac > 0.1).mean(), "test_gt10_pct": 100 * (bs_t.cloud_frac > 0.1).mean(),
                               "ks_stat": ks.statistic, "ks_p": ks.pvalue}, "Облачность BS: train vs test (meta.cloud_frac)")
    ks2 = stats.ks_2samp(af_meta.valid_frac, af_t.valid_frac)
    F.put(S, "af_valid_frac_meta", {"train_median": af_meta.valid_frac.median(), "test_median": af_t.valid_frac.median(),
                                    "train_lt1_pct": 100 * (af_meta.valid_frac < 1).mean(), "test_lt1_pct": 100 * (af_t.valid_frac < 1).mean(),
                                    "train_lt09_n": int((af_meta.valid_frac < 0.9).sum()), "test_lt09_n": int((af_t.valid_frac < 0.9).sum()),
                                    "ks_p": ks2.pvalue}, "Доля валидных пикселей AF (meta.valid_frac): train vs test")
    ta = pd.DataFrame(parallel_map(af_test_worker, [(str(test_dir), c) for c in af_t.chip_id], workers))
    ctx = mp.get_context("fork") if sys.platform.startswith("linux") else None
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        tb = pd.DataFrame(list(ex.map(bs_test_worker, [(str(test_dir), c) for c in bs_t.chip_id])))
    tr_ch = af_train["chip"]
    tzm = tb[["s1_zero_pre", "s1_zero_post"]].max(axis=1)
    comp = {
        "af_night_chips_pct": (100 * tr_ch["night"].mean(), 100 * ta.night.mean()),
        "af_chips_nan_gt10_pct": (100 * (tr_ch["nanfrac"] > 0.1).mean(), 100 * (ta.nanfrac > 0.1).mean()),
        "af_chips_stripe_nan_gt1_pct": (100 * (tr_ch["stripe"] > 0.01).mean(),
                                        100 * (np.clip(ta.nanfrac.values - (1 - af_t.valid_frac.values), 0, 1) > 0.01).mean()),
        "af_chips_outside_swath_pct": (100 * (af_meta.valid_frac < 1).mean(), 100 * (af_t.valid_frac < 1).mean()),
        "af_rule_hot_px_per_chip_median": (float(np.median(tr_ch["rule_cnt"])), float(ta.rule_cnt.median())),
        "af_chips_with_rule_hot_pct": (100 * (tr_ch["rule_cnt"] > 0).mean(), 100 * (ta.rule_cnt > 0).mean()),
        "bs_cloud_gt10_pct": (100 * (bm.cloud_frac > 0.1).mean(), 100 * (bs_t.cloud_frac > 0.1).mean()),
        "bs_s1_partial_pct": (100 * (bs_train["zmax"] > 0.001).mean(), 100 * (tzm > 0.001).mean()),
        "bs_excluded_scl_median_pct": (100 * np.median(bs_train["exs"]), 100 * tb.excluded.median()),
    }
    F.put(S, "train_vs_test", {k: {"train": a, "test": b} for k, (a, b) in comp.items()}, "Сравнение train/test по характеристикам снимков (без геопривязки теста)")
    ks3 = stats.ks_2samp(tr_ch["nanfrac"], ta.nanfrac)
    F.put(S, "af_nanfrac_ks_p", ks3.pvalue, "KS-тест доли NaN(I4) AF train vs test")
    ks4 = stats.ks_2samp(tr_ch["rule_cnt"], ta.rule_cnt)
    F.put(S, "af_rule_hot_ks_p", ks4.pvalue, "KS-тест числа пикселей выше порогов VNP14IMG на чип, train vs test")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    ax = axes[0, 0]
    ecdf(ax, 100 * bm.cloud_frac, color=C_TRAIN, lw=2, label=f"train (n={len(bm)})")
    ecdf(ax, 100 * bs_t.cloud_frac, color=C_TEST, lw=2, label=f"test (n={len(bs_t)})")
    ax.set_xlabel("облачность чипа BS (meta.cloud_frac = 1 − valid_frac), %")
    ax.set_ylabel("доля чипов (ЭФР)")
    ax.set_title("а) Облачность BS: train vs test")
    ax.legend(loc="lower right")
    note(ax, f"медиана: train {100 * bm.cloud_frac.median():.1f}%, test {100 * bs_t.cloud_frac.median():.1f}%\nKS p = {ks.pvalue:.2f}", loc=(0.35, 0.35))
    ax = axes[0, 1]
    ecdf(ax, 100 * tr_ch["nanfrac"], color=C_TRAIN, lw=2, label=f"train (n={len(tr_ch['nanfrac'])})")
    ecdf(ax, 100 * ta.nanfrac, color=C_TEST, lw=2, label=f"test (n={len(ta)})")
    ax.set_xlabel("доля NaN в I4 чипа AF, %")
    ax.set_ylabel("доля чипов (ЭФР)")
    ax.set_title("б) Невалидные пиксели AF: train vs test")
    ax.legend(loc="lower right")
    note(ax, f"чипов с NaN > 10%: train {comp['af_chips_nan_gt10_pct'][0]:.1f}%, test {comp['af_chips_nan_gt10_pct'][1]:.1f}%\nKS p = {ks3.pvalue:.2f}",
         loc=(0.40, 0.40))
    ax = axes[1, 0]
    share = 100 * bs_train["ch_share"]
    ax.scatter(100 * bm.cloud_frac, share, s=18, c=C_FIRE, alpha=0.6, lw=0)
    ax.set_xlabel("облачность чипа, %")
    ax.set_ylabel("доля гари в чипе, %")
    rho = F.get("bs", "cloud_vs_burned_share_spearman")["rho"]
    pv = F.get("bs", "cloud_vs_burned_share_spearman")["p"]
    bc = F.get("bs", "burned_share_by_cloud")
    ax.set_title("в) Облачность и размеченная гарь (train)")
    note(ax, (f"Спирмен ρ = {rho:.2f} (p = {pv:.2g})\nмедиана доли гари: облачн. ≤ 10% — {bc['cloud_le10_median_pct']:.1f}%,\n"
              f"> 10% — {bc['cloud_gt10_median_pct']:.1f}%"), loc=(0.98, 0.97), ha="right")
    ax = axes[1, 1]
    labs = {"af_night_chips_pct": "AF: ночные чипы", "af_chips_stripe_nan_gt1_pct": "AF: полосы NaN\n(> 1% пикселей)",
            "af_chips_outside_swath_pct": "AF: край полосы\nобзора", "af_chips_with_rule_hot_pct": "AF: есть пиксели\nвыше порогов",
            "bs_cloud_gt10_pct": "BS: облачн. > 10%", "bs_s1_partial_pct": "BS: пропуски S1"}
    ks_ = list(labs)
    y = np.arange(len(ks_))[::-1]
    ax.barh(y + 0.2, [comp[k][0] for k in ks_], 0.38, color=C_TRAIN, label="train")
    ax.barh(y - 0.2, [comp[k][1] for k in ks_], 0.38, color=C_TEST, label="test")
    for yi, k in zip(y, ks_):
        ax.text(comp[k][0] + 1, yi + 0.2, f"{comp[k][0]:.0f}%", va="center", fontsize=9)
        ax.text(comp[k][1] + 1, yi - 0.2, f"{comp[k][1]:.0f}%", va="center", fontsize=9)
    ax.set_yticks(y, [labs[k] for k in ks_])
    ax.set_xlim(0, 105)
    ax.set_xlabel("% чипов")
    ax.set_title("г) Характеристики снимков: train vs test")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Облачность, валидность пикселей и сдвиг распределений train → test")
    save(fig, figdir / "eda_clouds_shift.png")


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="каталог train (af/, bs/)")
    ap.add_argument("--test-dir", default=None, help="каталог test с meta.csv (по умолчанию <data-dir>/../test)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-bs", type=int, default=0, help="отладка: только первые N чипов BS")
    ap.add_argument("--examples", nargs="*", default=None, help="чипы BS для примеров на eda_bs_region.png")
    a = ap.parse_args()
    t0 = time.time()
    data_dir = Path(a.data_dir)
    test_dir = Path(a.test_dir) if a.test_dir else data_dir.parent / "test"
    out = Path(a.out)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    F = Facts()
    af_meta = pd.read_csv(data_dir / "af" / "meta.csv")
    bs_meta = pd.read_csv(data_dir / "bs" / "meta.csv")
    if a.max_bs:
        bs_meta = bs_meta.head(a.max_bs)

    af = af_section(data_dir, af_meta, F, figdir, a.workers)
    hot_map, covered = map_hotspots(af, bs_meta)
    bs = bs_section(data_dir, bs_meta, hot_map, covered, F, figdir, a.workers, a.examples)
    if (test_dir / "meta.csv").exists():
        shift_section(test_dir, af, af_meta, bs, F, figdir, a.workers)
    F.d["_meta"] = {"script": "scripts/eda_report.py", "data_dir": str(data_dir), "n_bs_chips_used": len(bs_meta),
                    "runtime_s": round(time.time() - t0, 1),
                    "notes": "Все доли в %, если в ключе есть _pct; dNBR — по firemon.bs.rules.dnbr (медиана 3x3), пороги THRESHOLDS; "
                             "AF-пороги — firemon.af.features DAY_POT/NIGHT_POT/NIGHT_SZA."}
    (out / "eda_facts.json").write_text(json.dumps(F.d, ensure_ascii=False, indent=1))
    print(f"done in {time.time() - t0:.1f}s -> {out / 'eda_facts.json'}", flush=True)


if __name__ == "__main__":
    main()
