"""Необязательная обработка нового чипа: загрузка GeoTIFF -> инференс -> векторизация.

Модели подключаются лениво:
    firemon.bs.predict.predict_bs_chip(BSChip) -> (mask uint8 0..3, prob float32)
    firemon.af.predict.predict_af_chip(AFChip) -> (mask uint8 0/1,  prob float32)
Если модуль или его зависимости (torch, веса) недоступны — HTTP 503.
Для BS можно вместо снимков передать готовую маску степени (mask) — тогда
выполняется только векторизация и расчёт площадей.
"""
from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np
import orjson
import shapely
import shapely.affinity
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from . import common as C

router = APIRouter(prefix="/api/v1/analyze", tags=["Анализ"])


_NO_MODEL = (
    "Инференс {what} в этой установке недоступен: не загружается модуль {mod} ({err}). "
    "Нужны зависимости модели (torch и др. из requirements.txt в корне репозитория) и веса weights/. "
    "Остальной сервис (запросы по каталогу, выгрузки, справка) работает без них."
)
_MASK_HINT = " Без модели можно передать готовую маску степени (поле mask) — будут построены контуры и посчитаны площади."


def _j(obj, code=200):
    return Response(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY), status_code=code, media_type="application/json")


async def _read_tif(f: UploadFile | None, name: str, required: bool = True):
    if f is None:
        if required:
            raise HTTPException(422, f"не передан файл {name}")
        return None
    try:
        from rasterio.io import MemoryFile
    except ImportError as e:  # pragma: no cover
        raise HTTPException(503, f"для анализа нужен rasterio: {e}")
    data = await f.read()
    if not data:
        raise HTTPException(422, f"{name}: пустой файл")
    try:
        with MemoryFile(data) as mf, mf.open() as src:
            if src.crs is None:
                raise HTTPException(422, f"{name}: у GeoTIFF нет системы координат")
            return src.read(), src.transform, src.crs
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        # в тексте ошибки GDAL — внутренний путь /vsimem/<uuid>; пользователю он ни о чём не говорит
        why = re.sub(r"'?/vsimem/\S+?'?(?=\s|$)", "файл", str(e))
        raise HTTPException(422, f"{name}: не удалось прочитать GeoTIFF ({why})")


def _check_grid(transform, name: str):
    if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9 or abs(abs(transform.a) - abs(transform.e)) > 1e-6:
        raise HTTPException(422, f"{name}: поддерживаются только растры «север вверх» с квадратным пикселем")


def _to_ll(geoms, crs):
    from pyproj import Transformer

    tr = Transformer.from_crs(crs, 4326, always_xy=True)

    def f(c):
        if len(c) == 0:
            return c
        x, y = tr.transform(c[:, 0], c[:, 1])
        return np.round(np.column_stack([x, y]), 7)

    return shapely.transform(geoms, f)


def _px_to_crs(g, transform):
    return shapely.affinity.affine_transform(g, [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f])


def _is_projected_m(crs) -> bool:
    try:
        return bool(crs.is_projected) and (crs.linear_units or "").lower() in ("metre", "meter", "m")
    except Exception:  # noqa: BLE001
        return False


def severity_result(mask, prob, landcover, transform, crs, chip_id: str, source: str) -> dict:
    from .report import shares_pct
    from .vectorize import vectorize_severity

    if not _is_projected_m(crs):
        raise HTTPException(422, "растр должен быть в метрической проекции (например, UTM) для расчёта площадей")
    import shapely.affinity  # noqa: F401

    px_m2 = abs(transform.a * transform.e)
    v = vectorize_severity(mask, prob, landcover, simplify_px=0)
    scars, zones = [], []
    tot = [0, 0, 0]
    geoms_s = [_px_to_crs(g, transform) for g in v["scar_geoms"]]
    geoms_z = [_px_to_crs(g, transform) for g in v["zone_geoms"]]
    ll_s = _to_ll(np.asarray(geoms_s, dtype=object), crs) if geoms_s else []
    ll_z = _to_ll(np.asarray(geoms_z, dtype=object), crs) if geoms_z else []
    for s, g in zip(v["scars"], ll_s):
        units = [int(round(n * px_m2 / 100)) for n in s["sev_px"]]
        sh = shares_pct(units)
        for i in range(3):
            tot[i] += units[i]
        sid = f"{chip_id}-U{s['k']:03d}"
        scars.append(
            {
                "type": "Feature",
                "id": sid,
                "geometry": shapely.geometry.mapping(g),
                "properties": {
                    "scar_id": sid,
                    "area_ha": round(sum(units) / 100, 2),
                    "area_low_ha": round(units[0] / 100, 2),
                    "area_moderate_ha": round(units[1] / 100, 2),
                    "area_high_ha": round(units[2] / 100, 2),
                    "share_low_pct": sh[0],
                    "share_moderate_pct": sh[1],
                    "share_high_pct": sh[2],
                    "n_pixels": s["n_px"],
                    "confidence": None if s["prob_mean"] is None else round(s["prob_mean"], 4),
                    "source": source,
                },
            }
        )
    for z, g in zip(v["zones"], ll_z):
        sid = f"{chip_id}-U{z['scar_k']:03d}"
        zones.append(
            {
                "type": "Feature",
                "id": f"{sid}-Z{z['severity']}",
                "geometry": shapely.geometry.mapping(g),
                "properties": {
                    "zone_id": f"{sid}-Z{z['severity']}",
                    "scar_id": sid,
                    "severity": z["severity"],
                    "severity_label": C.SEVERITY[z["severity"]]["label"],
                    "area_ha": round(int(round(z["n_px"] * px_m2 / 100)) / 100, 2),
                    "n_pixels": z["n_px"],
                    "source": source,
                },
            }
        )
    sh = shares_pct(tot)
    crs84 = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    return {
        "chip_id": chip_id,
        "source": source,
        "crs": str(crs),
        "pixel_area_m2": px_m2,
        "summary": {
            "burned_area_ha": round(sum(tot) / 100, 2),
            "n_scars": len(scars),
            "severity": [
                {"severity": s, "label": C.SEVERITY[s]["label"], "area_ha": round(tot[s - 1] / 100, 2), "share_pct": sh[s - 1]}
                for s in (1, 2, 3)
            ],
        },
        "burn_scars": {"type": "FeatureCollection", "crs": crs84, "features": scars},
        "severity_zones": {"type": "FeatureCollection", "crs": crs84, "features": zones},
    }


@router.post("/bs", summary="Гари и степень поражения для нового чипа Sentinel-2 (GeoTIFF)")
async def analyze_bs(
    s2_pre: Optional[UploadFile] = File(None, description="Sentinel-2 до пожара: B2 B3 B4 B5 B6 B7 B8A B11 B12 SCL"),
    s2_post: Optional[UploadFile] = File(None, description="Sentinel-2 после пожара (те же каналы)"),
    s1_pre: Optional[UploadFile] = File(None, description="Sentinel-1 до (VV VH), необязательно"),
    s1_post: Optional[UploadFile] = File(None, description="Sentinel-1 после (VV VH), необязательно"),
    aux: Optional[UploadFile] = File(None, description="dem, slope, landcover"),
    mask: Optional[UploadFile] = File(None, description="Готовая маска степени 0..3 — тогда модель не запускается"),
    chip_id: str = "uploaded",
):
    t0 = time.time()
    if mask is not None:
        m, tr, crs = await _read_tif(mask, "mask")
        _check_grid(tr, "mask")
        auxd = await _read_tif(aux, "aux", required=False)
        lc = auxd[0][2] if auxd is not None and auxd[0].shape[0] >= 3 else None
        res = severity_result(m[0].astype(np.uint8), None, lc, tr, crs, chip_id, "uploaded_mask")
        res["timing_ms"] = round((time.time() - t0) * 1000, 1)
        return _j(res)
    if s2_pre is None or s2_post is None:
        raise HTTPException(422, "передайте снимки s2_pre и s2_post (GeoTIFF) либо готовую маску степени mask")
    try:
        from firemon.bs.predict import predict_bs_chip  # type: ignore
        from firemon.io import BSChip
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, _NO_MODEL.format(what="гарей", mod="firemon.bs.predict", err=e) + _MASK_HINT)
    pre, tr, crs = await _read_tif(s2_pre, "s2_pre")
    post, tr2, _ = await _read_tif(s2_post, "s2_post")
    _check_grid(tr, "s2_pre")
    if pre.shape != post.shape:
        raise HTTPException(422, "s2_pre и s2_post должны иметь одинаковый размер и число каналов")
    h, w = pre.shape[1:]
    warn = []
    s1p = await _read_tif(s1_pre, "s1_pre", required=False)
    s1q = await _read_tif(s1_post, "s1_post", required=False)
    auxd = await _read_tif(aux, "aux", required=False)
    if s1p is None or s1q is None:
        warn.append("Sentinel-1 не передан — используются нулевые каналы")
    if auxd is None:
        warn.append("вспомогательный растр (aux) не передан — используются нулевые каналы")
    chip = BSChip(
        chip_id,
        pre,
        post,
        s1p[0] if s1p is not None else np.zeros((2, h, w), np.int16),
        s1q[0] if s1q is not None else np.zeros((2, h, w), np.int16),
        auxd[0] if auxd is not None else np.zeros((3, h, w), np.int16),
    )
    try:
        m, prob = predict_bs_chip(chip)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"ошибка инференса модели: {e}")
    lc = auxd[0][2] if auxd is not None and auxd[0].shape[0] >= 3 else None
    res = severity_result(np.asarray(m).astype(np.uint8), prob, lc, tr, crs, chip_id, "model")
    res["warnings"] = warn
    res["timing_ms"] = round((time.time() - t0) * 1000, 1)
    return _j(res)


@router.post("/af", summary="Термоточки для новой сцены VIIRS (GeoTIFF)")
async def analyze_af(
    viirs: UploadFile = File(..., description="VIIRS: I1..I5, solar_zenith, sensor_zenith, valid"),
    aux: Optional[UploadFile] = File(None, description="landcover, dem, t2m, rh2m, wind_speed"),
    chip_id: str = "uploaded",
):
    t0 = time.time()
    try:
        from firemon.af.predict import predict_af_chip  # type: ignore
        from firemon.io import AFChip
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, _NO_MODEL.format(what="термоточек", mod="firemon.af.predict", err=e))
    v, tr, crs = await _read_tif(viirs, "viirs")
    _check_grid(tr, "viirs")
    auxd = await _read_tif(aux, "aux", required=False)
    h, w = v.shape[1:]
    a = auxd[0].astype(np.float32) if auxd is not None else np.zeros((5, h, w), np.float32)
    try:
        m, prob = predict_af_chip(AFChip(chip_id, v.astype(np.float32), a))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"ошибка инференса модели: {e}")
    from pyproj import Transformer

    from .vectorize import extract_hotspots

    hs = extract_hotspots(np.asarray(m), v, a, prob)
    x = tr.c + (hs["col"] + 0.5) * tr.a
    y = tr.f + (hs["row"] + 0.5) * tr.e
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    feats = []
    for k in range(len(hs["row"])):
        i4, i5 = hs["i4"][k], hs["i5"][k]
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(float(lon[k]), 6), round(float(lat[k]), 6)]},
                "properties": {
                    "hotspot_id": f"{chip_id}-r{hs['row'][k]:03d}c{hs['col'][k]:03d}",
                    "daynight": str(hs["daynight"][k]),
                    "bright_ti4": None if not np.isfinite(i4) else round(float(i4), 2),
                    "bright_ti5": None if not np.isfinite(i5) else round(float(i5), 2),
                    "confidence": round(float(hs["conf"][k]), 4),
                    "landcover": C.landcover_name(hs["lc"][k]),
                },
            }
        )
    return _j(
        {
            "chip_id": chip_id,
            "n_hotspots": len(feats),
            "hotspots": {"type": "FeatureCollection", "features": feats},
            "timing_ms": round((time.time() - t0) * 1000, 1),
        }
    )
