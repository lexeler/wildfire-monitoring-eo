"""Признаки модуля детекции активного горения (VIIRS I-band, 375 м).

Логика повторяет устройство контекстного алгоритма VNP14IMG (Schroeder et al.,
2014): пиксель сравнивается со статистикой фоновых пикселей в окне, из фона
исключаются облака, невалидные пиксели и потенциальные очаги. Вместо жёстких
порогов статистики подаются в градиентный бустинг, который дополнительно
учится отсеивать техногенные термоаномалии и отражающие поверхности.

Все вычисления выполняются в torch батчем по чипам (CPU или GPU) — один и тот
же код используется в обучении и инференсе.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Порог «ночь» по зенитному углу Солнца, как в VNP14IMG
NIGHT_SZA = 85.0
# Потенциальные очаги (исключаются из фона): пороги VNP14IMG
DAY_POT = (325.0, 25.0)  # I4 > 325 K и I4-I5 > 25 K
NIGHT_POT = (295.0, 10.0)  # I4 > 295 K и I4-I5 > 10 K
# Кандидаты для классификатора — мягче порогов VNP14IMG, чтобы не терять
# полноту (на train сохраняют >99.5% эталонных пикселей горения)
CAND_DAY = (310.0, 12.0, 4.0)  # I4 > 310, dT > 12, превышение I4 над фоном > 4 K
CAND_NIGHT = (290.0, 6.0, 4.0)

WINDOWS = (7, 11, 21, 41)
WORLDCOVER = (10, 20, 30, 40, 50, 60, 80, 90)


def _box_mean(x: torch.Tensor, w: int) -> torch.Tensor:
    return F.avg_pool2d(x, w, stride=1, padding=w // 2, count_include_pad=False)


def _box_max(x: torch.Tensor, w: int) -> torch.Tensor:
    return F.max_pool2d(x, w, stride=1, padding=w // 2)


def compute_features(viirs: np.ndarray, aux: np.ndarray, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """viirs: (N, 8, H, W), aux: (N, 5, H, W) -> (features (N, F, H, W), candidate (N, H, W), names)."""
    v = torch.as_tensor(viirs, device=device, dtype=torch.float32)
    a = torch.as_tensor(aux, device=device, dtype=torch.float32)
    I1, I2, I3, I4, I5 = (v[:, i : i + 1] for i in range(5))
    sza, vza, valid_band = v[:, 5:6], v[:, 6:7], v[:, 7:8]
    lc, dem, t2m, rh, ws = (a[:, i : i + 1] for i in range(5))

    valid = torch.isfinite(I4) & torch.isfinite(I5) & (valid_band > 0.5)
    fin = lambda x: torch.nan_to_num(x, nan=0.0)
    I1, I2, I3 = fin(I1), fin(I2), fin(I3)
    # Невалидные пиксели заполняем медианой чипа, чтобы не портить окна
    med4 = torch.stack([torch.nanmedian(x) for x in v[:, 3]]).view(-1, 1, 1, 1)
    med5 = torch.stack([torch.nanmedian(x) for x in v[:, 4]]).view(-1, 1, 1, 1)
    med4 = torch.nan_to_num(med4, nan=290.0)
    med5 = torch.nan_to_num(med5, nan=285.0)
    I4 = torch.where(valid, fin(I4), med4)
    I5 = torch.where(valid, fin(I5), med5)
    dT = I4 - I5
    sza_m = torch.nan_to_num(sza, nan=float(np.nanmean(viirs[:, 5])) if np.isfinite(viirs[:, 5]).any() else 45.0)
    day = (sza_m < NIGHT_SZA).float()

    # Облака (тесты VNP14IMG) и вода
    refl = I1 + I2
    cloud_day = (I5 < 265) | ((refl > 0.9) & (I5 < 295)) | ((refl > 0.7) & (I5 < 285))
    cloud = torch.where(day > 0, cloud_day, I5 < 265) & valid
    water = lc == 80

    pot = torch.where(
        day > 0,
        (I4 > DAY_POT[0]) & (dT > DAY_POT[1]),
        (I4 > NIGHT_POT[0]) & (dT > NIGHT_POT[1]),
    )
    bg = valid & ~cloud & ~pot & ~water
    bgf = bg.float()

    feats: dict[str, torch.Tensor] = {}
    feats.update(I1=I1, I2=I2, I3=I3, I4=I4, I5=I5, dT=dT, sza=sza_m, vza=fin(vza), day=day)
    feats["refl12"] = refl
    feats["ndvi"] = (I2 - I1) / (I2 + I1 + 1e-3)
    feats["I3_I2"] = I3 / (I2 + 1e-3)
    feats["t2m"], feats["rh"], feats["ws"], feats["dem"] = t2m, rh, ws, dem
    feats["I4_minus_t2m"] = I4 - t2m
    feats["I5_minus_t2m"] = I5 - t2m
    feats["valid"] = valid.float()
    feats["cloud"] = cloud.float()
    feats["pot"] = pot.float()
    for code in WORLDCOVER:
        feats[f"lc_{code}"] = (lc == code).float()

    # Контекст по окнам: средние/СКО фона
    for w in WINDOWS:
        cnt = _box_mean(bgf, w)
        den = cnt.clamp_min(1e-6)
        for nm, x in (("I4", I4), ("I5", I5), ("dT", dT)):
            m = _box_mean(x * bgf, w) / den
            m2 = _box_mean(x * x * bgf, w) / den
            sd = (m2 - m * m).clamp_min(0).sqrt()
            fallback = {"I4": med4, "I5": med5, "dT": med4 - med5}[nm]
            m = torch.where(cnt > 0, m, fallback.expand_as(m))
            feats[f"{nm}_exc_w{w}"] = x - m
            feats[f"{nm}_z_w{w}"] = (x - m) / sd.clamp_min(0.5)
            feats[f"{nm}_sd_w{w}"] = sd
        feats[f"bgfrac_w{w}"] = cnt
        feats[f"potfrac_w{w}"] = _box_mean(pot.float(), w)
        feats[f"cloudfrac_w{w}"] = _box_mean(cloud.float(), w)
        feats[f"waterfrac_w{w}"] = _box_mean(water.float(), w)
        feats[f"urbanfrac_w{w}"] = _box_mean((lc == 50).float(), w)
        feats[f"cropfrac_w{w}"] = _box_mean((lc == 40).float(), w)
        feats[f"treefrac_w{w}"] = _box_mean((lc == 10).float(), w)
        feats[f"I3_exc_w{w}"] = I3 - _box_mean(I3, w)
        feats[f"dem_exc_w{w}"] = dem - _box_mean(dem, w)

    # Локальная структура: максимум соседей, число горячих соседей
    for k in (3, 5):
        mx4 = _box_max(I4, k)
        mxd = _box_max(dT, k)
        feats[f"I4_minus_max{k}"] = I4 - mx4
        feats[f"dT_minus_max{k}"] = dT - mxd
        feats[f"potcnt{k}"] = _box_mean(pot.float(), k) * k * k
    # максимум I4 среди соседей без центра
    pad = F.pad(I4, (1, 1, 1, 1), mode="replicate")
    shifts = [pad[..., 1 + dy : 1 + dy + I4.shape[-2], 1 + dx : 1 + dx + I4.shape[-1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    nmax = torch.stack(shifts, 0).max(0).values
    feats["I4_minus_nbrmax"] = I4 - nmax
    feats["nbr_hot"] = nmax - _box_mean(I4 * bgf, 11) / _box_mean(bgf, 11).clamp_min(1e-6)

    # Статистика сцены (чипа)
    def chip_stat(x, fn):
        return torch.stack([fn(xx[vv]) if vv.any() else torch.tensor(0.0, device=x.device) for xx, vv in zip(x[:, 0], valid[:, 0])]).view(-1, 1, 1, 1).expand_as(x)

    feats["chip_I4_med"] = chip_stat(I4, torch.median)
    feats["chip_I5_med"] = chip_stat(I5, torch.median)
    feats["chip_dT_med"] = chip_stat(dT, torch.median)
    feats["chip_pot_cnt"] = pot.float().sum((2, 3), keepdim=True).expand_as(I4)
    feats["chip_cloud_frac"] = cloud.float().mean((2, 3), keepdim=True).expand_as(I4)
    feats["chip_valid_frac"] = valid.float().mean((2, 3), keepdim=True).expand_as(I4)
    feats["I4_minus_chipmed"] = I4 - feats["chip_I4_med"]
    feats["dT_minus_chipmed"] = dT - feats["chip_dT_med"]

    names = list(feats)
    X = torch.cat([feats[n] for n in names], 1)

    exc4 = feats["I4_exc_w11"]
    cand = torch.where(
        day > 0,
        (I4 > CAND_DAY[0]) & (dT > CAND_DAY[1]) & (exc4 > CAND_DAY[2]),
        (I4 > CAND_NIGHT[0]) & (dT > CAND_NIGHT[1]) & (exc4 > CAND_NIGHT[2]),
    ) & valid
    return X, cand[:, 0], names


# --- «Точный» контекст VNP14IMG для кандидатов -------------------------------------
# Schroeder et al. (2014): фон — валидные пиксели без облаков, воды и «фоновых
# очагов» (день: I4 > 335 K и dT > 30 K; ночь: I4 > 300 K и dT > 10 K); окно растёт
# от 11x11 до 31x31, пока валидных пикселей не станет >= 25 % окна и >= 10 штук;
# статистики — среднее и среднее абсолютное отклонение (MAD). Тесты (день / ночь):
#   dT > dT_B + 2·MAD_dT (3·)   dT > dT_B + 10 K (9 K)   I4 > I4_B + 3.5·MAD_I4 (3·)
#   днём ещё: I5 > I5_B + MAD_I5 - 4 K  или  MAD I4 «фоновых очагов» > 5 K
# Эталон кейса порождён этим алгоритмом, поэтому запасы по каждому тесту и факт их
# прохождения (у пикселя и у соседей) — самые прямые признаки. Фиксированные окна
# compute_features их не заменяют: там другой фон (исключены «потенциальные» очаги
# 325/25), СКО вместо MAD и нет динамического окна.
BGFIRE_DAY = (335.0, 30.0)
BGFIRE_NIGHT = (300.0, 10.0)
EXACT_RMAX = 15  # окно до 31x31
EXACT_NAMES = ["ex_win", "ex_n", "ex_nfire", "ex_m4", "ex_mad4", "ex_m5", "ex_mad5", "ex_mdt", "ex_maddt", "ex_mad4fire",
               "ex_t1", "ex_t2", "ex_t3", "ex_t4", "ex_tmin", "ex_pass", "ex_short",
               "ex_nb_pass", "ex_nb_tmin_max", "ex_nb5_pass"]


@torch.no_grad()
def exact_context_rows(viirs: np.ndarray, aux: np.ndarray, ci: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor,
                       device: str = "cpu", chunk: int = 8192) -> torch.Tensor:
    """Признаки точного контекстного теста для кандидатов (ci, yy, xx) -> (M, len(EXACT_NAMES))."""
    v = torch.as_tensor(viirs, device=device, dtype=torch.float32)
    a = torch.as_tensor(aux, device=device, dtype=torch.float32)
    n, _, h, w = v.shape
    I1, I2 = torch.nan_to_num(v[:, 0]), torch.nan_to_num(v[:, 1])
    valid = torch.isfinite(v[:, 3]) & torch.isfinite(v[:, 4]) & (v[:, 7] > 0.5)
    I4 = torch.where(valid, torch.nan_to_num(v[:, 3]), torch.zeros_like(v[:, 3]))
    I5 = torch.where(valid, torch.nan_to_num(v[:, 4]), torch.zeros_like(v[:, 4]))
    dT = I4 - I5
    sza = torch.nan_to_num(v[:, 5], nan=float(np.nanmean(viirs[:, 5])) if np.isfinite(viirs[:, 5]).any() else 45.0)
    day = sza < NIGHT_SZA
    refl = I1 + I2
    cloud = torch.where(day, (I5 < 265) | ((refl > 0.9) & (I5 < 295)) | ((refl > 0.7) & (I5 < 285)), I5 < 265) & valid
    water = a[:, 0] == 80
    bgfire = torch.where(day, (I4 > BGFIRE_DAY[0]) & (dT > BGFIRE_DAY[1]), (I4 > BGFIRE_NIGHT[0]) & (dT > BGFIRE_NIGHT[1])) & valid
    bg = (valid & ~cloud & ~water & ~bgfire).float()
    R = EXACT_RMAX
    pad = lambda x: F.pad(x, (R, R, R, R))
    I4p, I5p, dTp, bgp, bfp = pad(I4), pad(I5), pad(dT), pad(bg), pad(bgfire.float())
    off = torch.arange(-R, R + 1, device=device)
    dy, dx = torch.meshgrid(off, off, indexing="ij")
    dy, dx = dy.reshape(-1), dx.reshape(-1)
    ring = torch.maximum(dy.abs(), dx.abs())  # 0..15
    ks = torch.arange(5, R + 1, device=device)  # окна 11..31
    ring_le = (ring[None, :] <= ks[:, None]).float()  # (11, 961)
    need = (0.25 * (2 * ks + 1) ** 2).clamp_min(10.0)
    m = len(ci)
    out = torch.zeros(m, 17, device=device)
    for s in range(0, m, chunk):
        c, y, x = ci[s : s + chunk], yy[s : s + chunk], xx[s : s + chunk]
        iy, ix = y[:, None] + R + dy[None, :], x[:, None] + R + dx[None, :]
        g = lambda t: t[c[:, None], iy, ix]
        B = g(bgp) * (ring > 0).float()[None, :]
        counts = B @ ring_le.T  # (M, 11)
        okk = counts >= need[None, :]
        first = torch.where(okk.any(1), okk.float().argmax(1), torch.full_like(c, len(ks) - 1))
        k = ks[first]
        W = B * (ring[None, :] <= k[:, None]).float()
        nn_ = W.sum(1).clamp_min(1.0)
        stats = []
        for t in (I4p, I5p, dTp):
            vals = g(t)
            mu = (vals * W).sum(1) / nn_
            mad = ((vals - mu[:, None]).abs() * W).sum(1) / nn_
            stats += [mu, mad]
        m4, mad4, m5, mad5, mdt, maddt = stats
        Fm = g(bfp) * (ring[None, :] <= k[:, None]).float() * (ring > 0).float()[None, :]
        nf = Fm.sum(1)
        v4 = g(I4p)
        mu4f = (v4 * Fm).sum(1) / nf.clamp_min(1.0)
        mad4f = ((v4 - mu4f[:, None]).abs() * Fm).sum(1) / nf.clamp_min(1.0)
        i4c, i5c, dtc, d = I4[c, y, x], I5[c, y, x], dT[c, y, x], day[c, y, x]
        t1 = dtc - (mdt + torch.where(d, 2.0, 3.0) * maddt)
        t2 = dtc - (mdt + torch.where(d, 10.0, 9.0))
        t3 = i4c - (m4 + torch.where(d, 3.5, 3.0) * mad4)
        t4 = i5c - (m5 + mad5 - 4.0)
        t4e = torch.where(d, torch.maximum(t4, mad4f - 5.0), torch.full_like(t4, 99.0))
        tmin = torch.minimum(torch.minimum(t1, t2), torch.minimum(t3, t4e))
        short = (~okk.any(1)).float()
        out[s : s + chunk] = torch.stack([2.0 * k + 1, W.sum(1), nf, m4, mad4, m5, mad5, mdt, maddt, mad4f, t1, t2, t3, t4, tmin,
                                          (tmin > 0).float(), short], 1)
    # соседи: сколько кандидатов рядом проходят тест и какой у них лучший запас
    dense_pass = torch.zeros(n, 1, h, w, device=device)
    dense_tmin = torch.full((n, 1, h, w), -50.0, device=device)
    dense_pass[ci, 0, yy, xx] = out[:, 15]
    dense_tmin[ci, 0, yy, xx] = out[:, 14].clamp(-50, 50)
    nb3 = F.avg_pool2d(dense_pass, 3, stride=1, padding=1) * 9 - dense_pass
    nb5 = F.avg_pool2d(dense_pass, 5, stride=1, padding=2) * 25 - dense_pass
    # максимум запаса среди 8 соседей (центр исключён: берём максимум по сдвигам)
    pp = F.pad(dense_tmin, (1, 1, 1, 1), value=-50.0)
    sh = [pp[..., 1 + a_ : 1 + a_ + h, 1 + b_ : 1 + b_ + w] for a_ in (-1, 0, 1) for b_ in (-1, 0, 1) if a_ or b_]
    nbmax = torch.stack(sh, 0).max(0).values
    extra = torch.stack([nb3[ci, 0, yy, xx], nbmax[ci, 0, yy, xx], nb5[ci, 0, yy, xx]], 1)
    return torch.cat([out, extra], 1)
