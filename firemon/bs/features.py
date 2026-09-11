"""Признаки модуля картирования гарей (Sentinel-2 / Sentinel-1, 20 м).

Один и тот же код (torch, CPU или GPU) строит входной тензор сети в обучении
и в инференсе. Каналы:
  * отражение B2..B12 «до», «после» и их разность;
  * спектральные индексы: NBR, dNBR (сырой и медиана 3x3), RdNBR, NDVI, NBR2,
    NDWI, MIRBI;
  * класс степени по правилу порогов постановки (one-hot) — сильная
    предметная подсказка, эталон строится именно так;
  * SCL «до»/«после» (one-hot по группам: облака, тени, вода, грунт…);
  * Sentinel-1 VV/VH на обе даты, разности и сглаженные разности, маска
    покрытия радаром;
  * рельеф (высота относительно среднего по чипу, уклон) и тип покрова.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .rules import LANDCOVER_GROUP, SCL_EXCLUDE, THRESHOLDS

SCL_GROUPS = ((0, 1), (2,), (3,), (4,), (5,), (6,), (7,), (8, 9), (10,), (11,))
LANDCOVER_CODES = (10, 20, 30, 40, 50, 60, 80, 90)


def median3(x: torch.Tensor) -> torch.Tensor:
    """Медианный фильтр 3x3 (отражающая граница, как scipy.ndimage по умолчанию)."""
    n, c, h, w = x.shape
    p = F.pad(x, (1, 1, 1, 1), mode="reflect")
    u = F.unfold(p, 3)  # (N, C*9, H*W)
    return u.view(n, c, 9, h * w).median(2).values.view(n, c, h, w)


def _nd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b) / (a + b).clamp_min(1e-4)


def severity_rule(dnbr_med: torch.Tensor, landcover: torch.Tensor) -> torch.Tensor:
    """Класс 0..3 по порогам dNBR своей группы покрова (N, 1, H, W) -> long."""
    out = torch.zeros_like(dnbr_med, dtype=torch.long)
    lc = landcover.long()
    known = torch.zeros_like(lc, dtype=torch.bool)
    for code, group in LANDCOVER_GROUP.items():
        sel = lc == code
        known |= sel
        th = torch.tensor(THRESHOLDS[group], device=dnbr_med.device, dtype=dnbr_med.dtype)
        cls = torch.bucketize(dnbr_med, th, right=True)
        out = torch.where(sel, cls, out)
    th = torch.tensor(THRESHOLDS["grass"], device=dnbr_med.device, dtype=dnbr_med.dtype)
    out = torch.where(known, out, torch.bucketize(dnbr_med, th, right=True))
    return out


def output_class_median(s2_pre: torch.Tensor, s2_post: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    """Final class rule, independent of the legacy rule used by trained models.

    Median of nine ordinal classes equals three majority votes for class >= k.
    A one-pixel replicated border matches scipy.ndimage's default reflect mode
    for a 3x3 window (unlike torch's reflect padding, which omits the edge).
    """
    def raw_nbr(s2):
        a, b = s2[:, 6:7].float(), s2[:, 8:9].float()
        return (a - b) / (a + b).clamp_min(1.0)

    r = severity_rule(raw_nbr(s2_pre) - raw_nbr(s2_post), aux[:, 2:3])
    codes = torch.tensor(SCL_EXCLUDE, device=r.device)
    ex = torch.isin(s2_pre[:, 9:10].long(), codes) | torch.isin(s2_post[:, 9:10].long(), codes)
    r = torch.where(ex, 0, r)
    ranks = torch.cat([(r >= k).float() for k in (1, 2, 3)], dim=1)
    votes = F.avg_pool2d(F.pad(ranks, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    result = (votes > .5).sum(dim=1)
    return result.masked_fill(ex[:, 0], 0)


def _blur_ds(x: torch.Tensor, factor: int, k: int) -> torch.Tensor:
    """Дешёвое крупномасштабное сглаживание: усреднение блоками factor x factor,
    бокс-фильтр k x k на грубой сетке, билинейный возврат к исходному размеру."""
    h, w = x.shape[-2:]
    y = F.avg_pool2d(x, factor)
    y = F.avg_pool2d(F.pad(y, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


def _distance(core: torch.Tensor, factor: int = 4, steps: int = 64) -> torch.Tensor:
    """Приближённое расстояние (в долях 256 px) до ближайшего пикселя ядра:
    итеративная дилатация на сетке 1/factor."""
    h, w = core.shape[-2:]
    c = F.max_pool2d(core, factor)
    dist = torch.full_like(c, float(steps))
    reached = c > 0
    dist[reached] = 0
    for i in range(1, steps + 1):
        reached_new = F.max_pool2d(reached.float(), 3, stride=1, padding=1) > 0
        dist[reached_new & ~reached] = i
        reached = reached_new
    dist = dist * factor / 256.0
    return F.interpolate(dist, size=(h, w), mode="bilinear", align_corners=False)


def context_features(rule: torch.Tensor, dnbr_med: torch.Tensor) -> list[torch.Tensor]:
    """Дальний контекст (область пожара в эталоне — круги радиусом 2-4 км,
    т.е. 100-200 px): плотность гари и dNBR в окнах ~40/80/150 px и расстояние
    до ядра сильной гари."""
    burn = (rule > 0).float()
    strong = (rule >= 2).float()
    d = dnbr_med.clamp(0, 1)
    out = []
    for factor, k in ((8, 5), (16, 5), (16, 9)):
        out += [_blur_ds(burn, factor, k), _blur_ds(strong, factor, k), _blur_ds(d, factor, k) * 2.0]
    out.append(_distance(strong).clamp(max=1.0))
    out.append(_distance((rule == 3).float()).clamp(max=1.0))
    return out


N_CONTEXT = 3 * 3 + 2


def global_features(rule: torch.Tensor, dnbr_med: torch.Tensor, pre: torch.Tensor, post: torch.Tensor,
                    excl: torch.Tensor) -> list[torch.Tensor]:
    """Контекст уровня всего чипа (уровень ctx=2): у GBDT статистики чипа —
    одни из самых важных признаков, а у сети рецептивное поле ограничено.
    Постоянные каналы: доли классов правила и исключённых пикселей, квантили
    dNBR; плюс сглаженное изменение red-edge B5 и расстояние до ядер dNBR>0.4."""
    n = rule.shape[0]
    ok = (~excl).float()
    out = []
    for m in ((rule >= 1).float(), (rule >= 2).float(), (rule == 3).float(), excl.float()):
        frac = m.mean((1, 2, 3), keepdim=True)
        out.append(frac.expand_as(dnbr_med) * 4.0)
    flat = torch.where(ok > 0, dnbr_med, torch.full_like(dnbr_med, float("nan"))).reshape(n, -1)
    for q in (0.5, 0.9, 0.99):
        v = torch.nan_to_num(torch.nanquantile(flat, q, dim=1), nan=0.0).view(n, 1, 1, 1)
        out.append(v.expand_as(dnbr_med) * 2.0)
    db5 = (post[:, 3:4] - pre[:, 3:4]) * 3.0
    out += [_blur_ds(db5, 8, 5), _blur_ds(db5, 16, 9)]
    out.append(_distance((dnbr_med > 0.4).float()).clamp(max=1.0))
    return out


N_GLOBAL = 4 + 3 + 2 + 1


def active_fire_s2(refl: torch.Tensor, scl: torch.Tensor) -> torch.Tensor:
    """Пиксели активного горения на снимке Sentinel-2: тепловое излучение фронта
    огня даёт аномально яркий SWIR2 (B12) при тёмном NIR — B12 > 0.5, B12 > 1.17·B11,
    B12 / B8A > 2 (облака по SCL исключены). refl (N, 9, H, W) в долях единицы."""
    b8a, b11, b12 = refl[:, 6:7], refl[:, 7:8], refl[:, 8:9]
    cloud = (scl >= 8) & (scl <= 10)
    return ((b12 > 0.5) & (b12 > 1.17 * b11) & (b12 > 2.0 * b8a.clamp_min(0.01)) & ~cloud).float()


def hot_features(pre: torch.Tensor, post: torch.Tensor, scl_pre: torch.Tensor, scl_post: torch.Tensor) -> list[torch.Tensor]:
    """Каналы «видимый огонь»: на ~10 % пар снимков фронт пожара виден прямо на
    снимке «до» или «после». Это единственное прямое указание на положение
    термоточек, вокруг которых построена область события эталона: расстояние
    до пикселей огня на каждой дате и их сглаженная плотность."""
    hp, hq = active_fire_s2(pre, scl_pre), active_fire_s2(post, scl_post)
    # одиночные яркие пиксели (крыши, блики) отсекаем: нужен хотя бы один сосед
    hp = hp * (F.avg_pool2d(hp, 3, stride=1, padding=1) * 9 >= 2).float()
    hq = hq * (F.avg_pool2d(hq, 3, stride=1, padding=1) * 9 >= 2).float()
    return [_distance(hq).clamp(max=1.0), _distance(hp).clamp(max=1.0), _blur_ds(torch.maximum(hp, hq), 8, 5) * 50.0]


N_HOT = 3


def compute_features(s2_pre, s2_post, s1_pre, s1_post, aux, ctx: int = 0, hot: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Входы — тензоры на нужном устройстве (N, C, H, W) в исходных типах
    (приводятся к float). Возвращает (X (N, F, H, W) float32, rule (N, H, W) long).
    ctx>=1 добавляет каналы дальнего контекста (context_features), ctx=2 —
    ещё и контекст уровня чипа (global_features). Каналы уровней дописываются
    в конец, поэтому модель уровня k берёт первые n_features(k) каналов;
    hot=True дописывает последними каналы «видимый огонь» (hot_features),
    выбор каналов конкретной модели — channel_index."""
    pre = s2_pre[:, :9].float() / 10000.0
    post = s2_post[:, :9].float() / 10000.0
    scl_pre = s2_pre[:, 9:10].long()
    scl_post = s2_post[:, 9:10].long()
    B = lambda x, i: x[:, i : i + 1]
    # индексы каналов: 0 B2, 1 B3, 2 B4, 3 B5, 4 B6, 5 B7, 6 B8A, 7 B11, 8 B12
    nbr_pre, nbr_post = _nd(B(pre, 6), B(pre, 8)), _nd(B(post, 6), B(post, 8))
    dnbr = nbr_pre - nbr_post
    dnbr_med = median3(dnbr)
    ndvi_pre, ndvi_post = _nd(B(pre, 6), B(pre, 2)), _nd(B(post, 6), B(post, 2))
    nbr2_pre, nbr2_post = _nd(B(pre, 7), B(pre, 8)), _nd(B(post, 7), B(post, 8))
    ndwi_post = _nd(B(post, 1), B(post, 6))
    mirbi_post = 10 * B(post, 8) - 9.8 * B(post, 7) + 2
    rdnbr = (dnbr_med / nbr_pre.abs().clamp_min(0.05).sqrt()).clamp(-3, 3)

    lc = aux[:, 2:3]
    rule = severity_rule(dnbr_med, lc)
    rule_oh = [(rule == k).float() for k in range(4)]

    scl = [torch.isin(scl_pre, torch.tensor(g, device=scl_pre.device)).float() for g in SCL_GROUPS]
    scl += [torch.isin(scl_post, torch.tensor(g, device=scl_post.device)).float() for g in SCL_GROUPS]

    s1a = s1_pre.float() / 100.0
    s1b = s1_post.float() / 100.0
    sar_valid = ((s1_pre[:, :1] != 0) & (s1_post[:, :1] != 0)).float()
    s1a_n = ((s1a + 15.0) / 7.0) * sar_valid
    s1b_n = ((s1b + 15.0) / 7.0) * sar_valid
    ds1 = (s1b - s1a) * sar_valid / 3.0
    ds1_s = F.avg_pool2d(ds1, 7, stride=1, padding=3, count_include_pad=False)

    dem = aux[:, 0:1].float()
    dem_rel = (dem - dem.mean((2, 3), keepdim=True)) / 50.0
    slope = aux[:, 1:2].float() / 20.0
    lc_oh = [(lc == c).float() for c in LANDCOVER_CODES]

    chans = [
        pre,
        post,
        (post - pre) * 3.0,
        nbr_pre,
        nbr_post,
        dnbr * 2.0,
        dnbr_med * 2.0,
        rdnbr,
        ndvi_pre,
        ndvi_post,
        ndvi_post - ndvi_pre,
        nbr2_pre,
        nbr2_post,
        (nbr2_pre - nbr2_post) * 3.0,
        ndwi_post,
        mirbi_post,
        *rule_oh,
        *scl,
        s1a_n,
        s1b_n,
        ds1,
        ds1_s,
        sar_valid,
        dem_rel,
        slope,
        *lc_oh,
    ]
    if ctx >= 1:
        chans += context_features(rule, dnbr_med)
    if ctx >= 2:
        excl_sc = torch.isin(scl_pre, torch.tensor((0, 3, 6, 9, 11), device=scl_pre.device)) | torch.isin(
            scl_post, torch.tensor((0, 3, 6, 9, 11), device=scl_post.device))
        chans += global_features(rule, dnbr_med, pre, post, excl_sc)
    if hot:
        chans += hot_features(pre, post, scl_pre, scl_post)
    X = torch.cat(chans, 1)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, rule[:, 0]


N_FEATURES = 9 * 3 + 13 + 4 + 2 * len(SCL_GROUPS) + 2 + 2 + 2 + 2 + 1 + 2 + len(LANDCOVER_CODES)


def n_features(ctx: int = 0, hot: bool = False) -> int:
    ctx = int(ctx)
    return N_FEATURES + (N_CONTEXT if ctx >= 1 else 0) + (N_GLOBAL if ctx >= 2 else 0) + (N_HOT if hot else 0)


def channel_index(ctx: int, hot: bool, full_ctx: int, full_hot: bool) -> list[int]:
    """Индексы каналов модели (ctx, hot) в тензоре, посчитанном с (full_ctx, full_hot)."""
    idx = list(range(n_features(ctx)))
    if hot:
        if not full_hot:
            raise ValueError("признаки посчитаны без каналов hot")
        idx += list(range(n_features(full_ctx), n_features(full_ctx) + N_HOT))
    return idx
