"""Аналитическая справка по запросу: сводка, распределение по степени, таблицы, сравнение."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from . import common as C
from .catalog import Catalog, Query, ScarHit, SourceStore, apportion, ha

SERVICE_NAME = "Мониторинг природных пожаров — информационно-аналитический сервис (LitWin)"
SERVICE_VERSION = "1.0.0"


def _pct(part, total, nd=2):
    return round(100.0 * part / total, nd) if total else 0.0


def shares_pct(units: list[int]) -> list[float]:
    """Доли в % с округлением до 0.01, сумма ровно 100 (метод наибольших остатков)."""
    tot = sum(units)
    if tot <= 0:
        return [0.0] * len(units)
    return [v / 100.0 for v in apportion(10000, units)]


def dominant(units: list[int]) -> int:
    """Преобладающий класс степени (при равенстве — более сильный)."""
    best = 0
    for s in (1, 2, 3):
        if units[s - 1] > 0 and units[s - 1] >= (units[best - 1] if best else -1):
            best = s
    return best


def scar_row(cat: Catalog, st: SourceStore, h: ScarHit) -> dict:
    s = st.scars[h.idx]
    chip = cat.bs_chips[st.scar_chip[h.idx]]
    u = h.sev_units
    sh = shares_pct(u)
    dom = dominant(u)
    full_units = st.scar_sev_units[h.idx].tolist()
    return {
        "scar_id": s["scar_id"],
        "chip_id": chip["chip_id"],
        "fire_event_id": chip["fire_event_id"],
        "date_pre": chip["date_pre"],
        "date_post": chip["date_post"],
        "area_ha": ha(sum(u)),
        "area_low_ha": ha(u[0]),
        "area_moderate_ha": ha(u[1]),
        "area_high_ha": ha(u[2]),
        "share_low_pct": sh[0],
        "share_moderate_pct": sh[1],
        "share_high_pct": sh[2],
        "dominant_severity": dom,
        "dominant_label": C.SEVERITY[dom]["label"] if dom else None,
        "clipped": bool(h.clipped),
        "area_total_ha": ha(sum(full_units)),
        "n_pixels": int(s["n_px"]),
        "pixel_size_m": float(st.scar_gsd[h.idx]),
        "utm_epsg": int(st.scar_epsg[h.idx]),
        "center_lon": s["center_ll"][0],
        "center_lat": s["center_ll"][1],
        "bbox": s["bbox_ll"],
        "confidence": s["prob_mean"],
        "source": st.name,
    }


def severity_table(hits: list[ScarHit]) -> list[dict]:
    units = [0, 0, 0]
    n_zones = [0, 0, 0]
    n_dom = [0, 0, 0]
    for h in hits:
        for i in range(3):
            units[i] += h.sev_units[i]
            if h.sev_units[i] > 0:
                n_zones[i] += 1
        d = dominant(h.sev_units)
        if d:
            n_dom[d - 1] += 1
    sh = shares_pct(units)
    return [
        {
            "severity": s,
            "key": C.SEVERITY[s]["key"],
            "label": C.SEVERITY[s]["label"],
            "color": C.SEVERITY[s]["color"],
            "area_ha": ha(units[s - 1]),
            "share_pct": sh[s - 1],
            "n_zones": n_zones[s - 1],
            "n_scars_dominant": n_dom[s - 1],
        }
        for s in (1, 2, 3)
    ]


def landcover_table(st: SourceStore, hits: list[ScarHit]) -> tuple[list[dict], bool]:
    """Площадь гарей по типам покрова (ESA WorldCover).

    Для обрезанных контуров площадь зоны каждой степени распределяется по типам
    покрова пропорционально числу пикселей (оценка); для необрезанных — точно.
    """
    acc: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    estimated = False
    for h in hits:
        lc = st.scars[h.idx]["lc_px"]
        if not lc:
            continue
        codes = sorted(lc, key=int)
        full = st.scar_sev_units[h.idx]
        for si in range(3):
            if h.sev_units[si] <= 0:
                continue
            w = [lc[c][si] for c in codes]
            if h.clipped:
                estimated = True
                parts = apportion(h.sev_units[si], w)
            else:
                # точные пиксели -> сотые гектара
                k = full[si] / max(sum(w), 1)
                parts = [int(round(v * k)) for v in w]
            for c, p in zip(codes, parts):
                acc[int(c)][si] += p
    total = sum(sum(v) for v in acc.values())
    rows = []
    for code, v in sorted(acc.items(), key=lambda kv: -sum(kv[1])):
        rows.append(
            {
                "code": code,
                "name": C.landcover_name(code),
                "area_ha": ha(sum(v)),
                "share_pct": _pct(sum(v), total),
                "area_low_ha": ha(v[0]),
                "area_moderate_ha": ha(v[1]),
                "area_high_ha": ha(v[2]),
            }
        )
    return rows, estimated


def hotspot_summary(st: SourceStore, idx: np.ndarray) -> dict:
    n = int(len(idx))
    dn = st.hs_dn[idx]
    sat = st.hs_sat[idx]
    dates = st.hs_date[idx].astype(str)
    by_date: dict[str, list[int]] = {}
    for d, x in zip(dates.tolist(), dn.tolist()):
        v = by_date.setdefault(d, [0, 0])
        v[0 if x == "D" else 1] += 1
    by_month: Counter = Counter(d[:7] for d in dates.tolist())
    i4 = np.asarray([st.hs["i4"][i] for i in idx], dtype=float) if n else np.zeros(0)
    i4 = i4[np.isfinite(i4)]
    lc = Counter(st.hs_lc[idx].tolist())
    satc = Counter(sat.tolist())
    scenes = sorted(set(st.hs["chip"][i] for i in idx.tolist()))
    return {
        "total": n,
        "day": int((dn == "D").sum()),
        "night": int((dn == "N").sum()),
        "first_datetime": st.hs["t"][idx[0]] if n else None,
        "last_datetime": st.hs["t"][idx[-1]] if n else None,
        "n_scenes_with_hotspots": len(scenes),
        "by_satellite": [{"satellite": k, "label": C.SATELLITES.get(k, k), "count": v} for k, v in sorted(satc.items())],
        "by_date": [{"date": d, "count": v[0] + v[1], "day": v[0], "night": v[1]} for d, v in sorted(by_date.items())],
        "by_month": [{"month": m, "count": c} for m, c in sorted(by_month.items())],
        "by_landcover": [
            {"code": int(k), "name": C.landcover_name(k), "count": v} for k, v in sorted(lc.items(), key=lambda kv: -kv[1])
        ],
        "bright_ti4_k": None
        if len(i4) == 0
        else {"min": round(float(i4.min()), 2), "mean": round(float(i4.mean()), 2), "max": round(float(i4.max()), 2)},
    }


def burned_by_month(cat: Catalog, st: SourceStore, hits: list[ScarHit]) -> list[dict]:
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for h in hits:
        m = cat.bs_chips[st.scar_chip[h.idx]]["date_post"][:7]
        for i in range(3):
            acc[m][i] += h.sev_units[i]
    return [
        {
            "month": m,
            "area_ha": ha(sum(v)),
            "area_low_ha": ha(v[0]),
            "area_moderate_ha": ha(v[1]),
            "area_high_ha": ha(v[2]),
        }
        for m, v in sorted(acc.items())
    ]


def _agreement(cm: np.ndarray, px_units: float) -> dict:
    """Метрики согласия по матрице ошибок 4x4 (строки — эталон, столбцы — модель)."""
    tp = int(cm[1:, 1:].sum())
    fp = int(cm[0, 1:].sum())
    fn = int(cm[1:, 0].sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    per_class = []
    for s in (1, 2, 3):
        inter = int(cm[s, s])
        union = int(cm[s, :].sum() + cm[:, s].sum() - cm[s, s])
        per_class.append(
            {
                "severity": s,
                "label": C.SEVERITY[s]["label"],
                "iou": round(inter / union, 4) if union else None,
                "reference_px": int(cm[s, :].sum()),
                "model_px": int(cm[:, s].sum()),
            }
        )
    common = cm[1:, 1:]
    acc = float(np.trace(common) / common.sum()) if common.sum() else None
    r = lambda v: None if v is None else round(v, 4)  # noqa: E731
    return {
        "iou_burned": r(iou),
        "precision": r(prec),
        "recall": r(rec),
        "f1": r(f1),
        "tp_ha": ha(round(tp * px_units)),
        "fp_ha": ha(round(fp * px_units)),
        "fn_ha": ha(round(fn * px_units)),
        "severity_accuracy_on_common": r(acc),
        "per_class": per_class,
        "confusion_matrix_px": cm.astype(int).tolist(),
        "confusion_labels": ["не горело", "слабая", "средняя", "сильная"],
    }


def comparison_block(cat: Catalog, q: Query) -> dict | None:
    if "model" not in cat.sources or "reference" not in cat.sources:
        return None
    out = {}
    hs_keys = {}
    for s in ("model", "reference"):
        st = cat.sources[s]
        # тот же запрос, но для другого источника; кэш перепроецирования в UTM переиспользуется
        qq = Query(
            geom=q.geom, spatial=q.spatial, date_from=q.date_from, date_to=q.date_to, source=s, clip=q.clip,
            bbox=q.bbox, data_bounds=q.data_bounds, _utm=q._utm, _prepared=q._prepared,
        )
        hits = st.scar_hits(qq)
        units = [sum(h.sev_units[i] for h in hits) for i in range(3)]
        hidx = st.hotspot_index(qq)
        hs_keys[s] = set(st.hs_key[hidx].tolist())
        out[s] = {
            "label": C.SOURCE_LABELS[s],
            "burned_area_ha": ha(sum(units)),
            "area_low_ha": ha(units[0]),
            "area_moderate_ha": ha(units[1]),
            "area_high_ha": ha(units[2]),
            "n_scars": len(hits),
            "n_hotspots": int(len(hidx)),
        }
    ref_a, mod_a = out["reference"]["burned_area_ha"], out["model"]["burned_area_ha"]
    cm, n_chips = cat.confusion(q)
    px_units = C.BS_GSD**2 / 100.0
    tp = len(hs_keys["model"] & hs_keys["reference"])
    fp = len(hs_keys["model"] - hs_keys["reference"])
    fn = len(hs_keys["reference"] - hs_keys["model"])
    r = lambda v: None if v is None else round(v, 4)  # noqa: E731
    out.update(
        {
            "burned_area_diff_ha": round(mod_a - ref_a, 2),
            "burned_area_diff_pct": _pct(mod_a - ref_a, ref_a) if ref_a else None,
            "pixel_agreement": {**_agreement(cm, px_units), "n_chips": n_chips, "n_pixels": int(cm.sum())},
            "hotspot_agreement": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": r(tp / (tp + fp) if tp + fp else None),
                "recall": r(tp / (tp + fn) if tp + fn else None),
                "f1": r(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None),
            },
            "method": (
                "Площади — как в основной справке (для каждого источника). Согласие гарей — попиксельно "
                "по растрам 20 м: учитываются пиксели, центр которых внутри полигона запроса; IoU по классу "
                "«горело» (степень >= 1), точность степени — на пикселях, горевших по обоим источникам. "
                "Согласие термоточек — по совпадению пикселя VIIRS (сцена, строка, столбец)."
            ),
        }
    )
    return out


def method_notes(cat: Catalog, q: Query) -> dict:
    return {
        "hotspots": (
            "Термоточка — центр пикселя VIIRS I-диапазона (375 м), отнесённого к активному горению "
            f"({'маска модели' if q.source == 'model' else 'эталонная маска'}); координаты пересчитаны из UTM в WGS84 "
            "(pyproj). День/ночь — по зенитному углу Солнца (< 85° — день). Фильтр по времени: дата съёмки (UTC) "
            "в интервале [date_from, date_to] включительно; по пространству — точка внутри полигона запроса."
        ),
        "burned_area": (
            "Контур гари — связная (8-связность) область пикселей Sentinel-2 20 м со степенью поражения >= 1; "
            "зоны степени — пиксели одного класса внутри контура. Площадь без обрезки = число пикселей x 400 м² "
            "в родной проекции UTM чипа (EPSG:32637/32638), 1 пиксель = 0.04 га. При пересечении контура с "
            "границей запроса точная попиксельная геометрия обрезается полигоном запроса, перепроецированным в "
            "ту же зону UTM, площадь считается планиметрически в UTM. Итоги — точные суммы по контурам "
            "(округление до 0.01 га один раз на уровне зоны)."
        ),
        "time_filter": (
            "Гарь попадает в выборку, если окно наблюдения [date_pre, date_post] пересекается с периодом "
            "запроса: date_pre <= date_to и date_post >= date_from."
        ),
        "landcover": "Тип покрова — ESA WorldCover (канал landcover вспомогательных растров).",
        "crs_output": "Геометрия в выгрузках — WGS84 (EPSG:4326); площади — в UTM соответствующей зоны.",
        "pixel_size_m": {"hotspots": C.AF_GSD, "burn_scars": C.BS_GSD},
        "area_projection": "UTM (WGS84), зоны 37N (EPSG:32637) и 38N (EPSG:32638)",
    }


def build_report(cat: Catalog, q: Query, st: SourceStore, hits: list[ScarHit], hs_idx: np.ndarray, compare: bool = True) -> dict:
    sev = severity_table(hits)
    total_units = sum(h.units for h in hits)
    rows = [scar_row(cat, st, h) for h in hits]
    rows.sort(key=lambda r: -r["area_ha"])
    lc_rows, lc_est = landcover_table(st, hits)
    n_clipped = sum(1 for h in hits if h.clipped)
    full_units = sum(int(st.scar_sev_units[h.idx].sum()) for h in hits)
    hs = hotspot_summary(st, hs_idx)
    q_area = q.area_km2()
    geom = None
    if q.geom is not None:
        import shapely

        geom = shapely.geometry.mapping(q.geom)
    bs_chips = cat.chips_in_query(q, "bs")
    af_chips = cat.chips_in_query(q, "af")
    report = {
        "title": "Аналитическая справка: природные пожары",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "service": {
            "name": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "catalog_generated_at": cat.meta.get("generated_at"),
        },
        "query": {
            "spatial_filter": q.spatial,
            "bbox": q.bbox,
            "geometry": geom,
            "area_km2": q_area,
            "date_from": q.date_from.isoformat(),
            "date_to": q.date_to.isoformat(),
            "source": q.source,
            "source_label": C.SOURCE_LABELS[q.source],
            "clip": q.clip,
        },
        "summary": {
            "burned_area_ha": ha(total_units),
            "n_scars": len(hits),
            "n_scars_clipped": n_clipped,
            "burned_area_unclipped_ha": ha(full_units),
            "burned_share_of_query_pct": round(total_units / 100.0 / (q_area * 100.0) * 100.0, 4) if q_area else None,
            "n_hotspots": hs["total"],
            "n_hotspots_day": hs["day"],
            "n_hotspots_night": hs["night"],
            "n_bs_chips": int(len(bs_chips)),
            "n_af_scenes": int(len(af_chips)),
        },
        "severity": sev,
        "burned_by_month": burned_by_month(cat, st, hits),
        "landcover": lc_rows,
        "landcover_estimated": lc_est,
        "hotspots": hs,
        "scars": rows,
        "comparison": comparison_block(cat, q) if compare else None,
        "method": method_notes(cat, q),
    }
    return report
